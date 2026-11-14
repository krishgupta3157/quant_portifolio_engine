"""Train and evaluate standalone multi-asset volatility risk models.

This script is intentionally independent of Streamlit and the application
state. It reads local company CSVs, builds leakage-safe chronological
features, compares three tree-based regressors, tunes the best family with
time-series cross-validation, and exports model artifacts and diagnostics.

Run from the repository root:
    python scripts/train_risk_model_standalone.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("train_risk_model")
RANDOM_STATE = 42
ANNUALIZATION = np.sqrt(252)
TARGET_HORIZON = 20
MIN_ASSETS = 3


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap predictor values using quantile bounds learned from fit data."""

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: Any = None) -> Winsorizer:
        del y
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("Winsorizer quantiles must satisfy 0 <= lower < upper <= 1.")
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if frame.empty or frame.shape[1] == 0:
            raise ValueError("Winsorizer requires a non-empty feature matrix.")

        self.feature_columns_ = frame.columns.copy()
        self.lower_bounds_ = frame.quantile(self.lower_quantile)
        self.upper_bounds_ = frame.quantile(self.upper_quantile)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, ["feature_columns_", "lower_bounds_", "upper_bounds_"])
        if isinstance(X, pd.DataFrame):
            if not X.columns.equals(self.feature_columns_):
                raise ValueError("Winsorizer input columns differ from the fitted features.")
            frame = X
        else:
            frame = pd.DataFrame(X, columns=self.feature_columns_)
        return frame.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis="columns",
        )


def parse_args() -> argparse.Namespace:
    """Parse optional paths while keeping the requested root-level command simple."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing asset CSV files. Defaults to repository/companies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for models and charts. Defaults to repository/models.",
    )
    return parser.parse_args()


def repository_root() -> Path:
    """Return the repository root based on this script's location."""
    return Path(__file__).resolve().parents[1]


def discover_asset_files(data_dir: Path) -> list[Path]:
    """Find usable company CSV files and exclude aggregate/metadata exports."""
    excluded = {"NIFTY50_all.csv", "stock_metadata.csv"}
    files = sorted(
        path
        for path in data_dir.glob("*.csv")
        if path.name not in excluded
    )
    if len(files) < MIN_ASSETS:
        raise FileNotFoundError(
            f"Expected at least {MIN_ASSETS} asset CSV files in {data_dir}, "
            f"found {len(files)}."
        )
    return files


def load_price_matrix(data_dir: Path) -> pd.DataFrame:
    """Load each local asset close series and merge them by normalized date."""
    series: list[pd.Series] = []
    for path in discover_asset_files(data_dir):
        frame = pd.read_csv(path, usecols=["Date", "Close"])
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"]).drop_duplicates("Date")
        if frame.empty:
            LOGGER.warning("Skipping empty asset file: %s", path.name)
            continue
        asset_name = path.stem.upper()
        series.append(frame.set_index("Date")["Close"].rename(asset_name))

    if len(series) < MIN_ASSETS:
        raise ValueError("Not enough valid asset series remained after CSV cleaning.")

    prices = pd.concat(series, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, thresh=max(20, int(len(prices) * 0.70)))
    prices = prices.ffill().dropna(how="all").dropna(axis=0, how="any")
    if len(prices) <= TARGET_HORIZON + 100:
        raise ValueError("The merged price matrix does not contain enough observations.")
    LOGGER.info(
        "Loaded %d assets and %d aligned observations (%s to %s).",
        prices.shape[1],
        prices.shape[0],
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return prices


def build_feature_frame(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe rolling features and a future realized-volatility target."""
    returns = prices.pct_change()
    portfolio_returns = returns.mean(axis=1)

    features = pd.DataFrame(index=prices.index)
    features["portfolio_return"] = portfolio_returns
    features["portfolio_vol_20"] = portfolio_returns.rolling(20).std() * ANNUALIZATION
    features["portfolio_vol_50"] = portfolio_returns.rolling(50).std() * ANNUALIZATION

    asset_vol_20 = returns.rolling(20).std() * ANNUALIZATION
    asset_vol_50 = returns.rolling(50).std() * ANNUALIZATION
    features["cross_asset_vol_20_mean"] = asset_vol_20.mean(axis=1)
    features["cross_asset_vol_20_std"] = asset_vol_20.std(axis=1)
    features["cross_asset_vol_50_mean"] = asset_vol_50.mean(axis=1)
    features["cross_asset_vol_50_std"] = asset_vol_50.std(axis=1)

    rolling_peak = prices.rolling(50, min_periods=20).max()
    drawdowns = prices / rolling_peak - 1.0
    features["drawdown_mean_50"] = drawdowns.mean(axis=1)
    features["drawdown_min_50"] = drawdowns.min(axis=1)

    rolling_correlation = returns.rolling(20, min_periods=15).corr()
    upper_triangle = np.triu(
        np.ones((prices.shape[1], prices.shape[1]), dtype=bool),
        k=1,
    )
    asset_positions = prices.columns.get_indexer(
        rolling_correlation.index.get_level_values(1)
    )
    correlation_values = rolling_correlation.to_numpy(dtype=float)
    valid_positions = asset_positions >= 0
    correlation_mask = np.zeros_like(correlation_values, dtype=bool)
    correlation_mask[valid_positions] = upper_triangle[asset_positions[valid_positions]]
    masked_correlations = np.where(correlation_mask, correlation_values, np.nan)
    pairwise_mean = pd.DataFrame(
        masked_correlations,
        index=rolling_correlation.index,
        columns=rolling_correlation.columns,
    ).groupby(level=0).mean().mean(axis=1)
    features["average_pairwise_correlation_20"] = pairwise_mean.reindex(features.index)

    future_volatility = (
        portfolio_returns.rolling(TARGET_HORIZON).std().shift(-TARGET_HORIZON)
        * ANNUALIZATION
    )
    target = future_volatility.rename("future_20d_realized_volatility")

    dataset = features.replace([np.inf, -np.inf], np.nan).join(target)
    dataset = dataset.dropna()
    return dataset.drop(columns=[target.name]), dataset[target.name]


def candidate_models() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return baseline models and compact tuning grids for supported estimators."""
    return {
        "lightgbm": (
            lgb.LGBMRegressor(
                objective="regression",
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=-1,
            ),
            {
                "n_estimators": [150, 300],
                "learning_rate": [0.03, 0.08],
                "num_leaves": [15, 31],
                "max_depth": [-1, 6],
            },
        ),
        "random_forest": (
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [200],
                "max_depth": [None, 12],
                "min_samples_leaf": [1, 3],
                "max_features": [1.0, "sqrt"],
            },
        ),
        "xgboost": (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
            ),
            {
                "n_estimators": [150, 300],
                "learning_rate": [0.03, 0.08],
                "max_depth": [3, 6],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
        ),
    }


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate one fitted regressor on chronological observations."""
    predictions = np.asarray(model.predict(x_test), dtype=float)
    return evaluate_predictions(y_test, predictions)


def evaluate_predictions(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for predictions aligned to observed targets."""
    return {
        "mse": float(mean_squared_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Save a horizontal feature-importance chart for the selected model."""
    if isinstance(model, Pipeline):
        model = model.named_steps["model"]
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        raise AttributeError("The selected model does not expose feature_importances_.")
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=True)
        .tail(15)
    )
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(importance_frame["feature"], importance_frame["importance"], color="#38BDF8")
    axis.set_title("Feature Importance — Future 20-Day Realized Volatility")
    axis.set_xlabel("Importance")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_predictions(
    dates: pd.Index,
    actual: pd.Series,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    """Save actual-versus-predicted volatility over the chronological test period."""
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(dates, actual.to_numpy(), label="Actual volatility", color="#34D399", linewidth=1.8)
    axis.plot(dates, predicted, label="Predicted volatility", color="#38BDF8", linewidth=1.8)
    axis.set_title("Predicted vs Actual Future 20-Day Realized Volatility")
    axis.set_ylabel("Annualized volatility")
    axis.set_xlabel("Test date")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_pipeline(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the full chronological training, evaluation, tuning, and export flow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    charts_dir = output_dir / "charts"
    models_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)

    prices = load_price_matrix(data_dir)
    features, target = build_feature_frame(prices)
    usable_rows = len(features) - 2 * TARGET_HORIZON
    if usable_rows < 10:
        raise ValueError(
            "Not enough observations for a chronological 60/20/20 split "
            f"with {TARGET_HORIZON}-row gaps."
        )

    train_count = int(usable_rows * 0.60)
    validation_count = int(usable_rows * 0.20)
    validation_start = train_count + TARGET_HORIZON
    validation_end = validation_start + validation_count
    test_start = validation_end + TARGET_HORIZON

    x_train = features.iloc[:train_count]
    y_train = target.iloc[:train_count]
    x_validation = features.iloc[validation_start:validation_end]
    y_validation = target.iloc[validation_start:validation_end]
    x_test = features.iloc[test_start:]
    y_test = target.iloc[test_start:]
    if x_train.empty or x_validation.empty or x_test.empty:
        raise ValueError("The gapped chronological split produced an empty partition.")

    LOGGER.info(
        "Chronological split with %d-row gaps: %d train / %d validation / %d test rows.",
        TARGET_HORIZON,
        len(x_train),
        len(x_validation),
        len(x_test),
    )
    cv_splits = min(5, max(2, len(x_train) // 100))
    fold_size = len(x_train) // (cv_splits + 1)
    initial_train_size = len(x_train) - cv_splits * fold_size
    if initial_train_size <= TARGET_HORIZON:
        raise ValueError(
            "The training partition is too small for time-series CV with "
            f"a {TARGET_HORIZON}-row gap."
        )
    time_cv = TimeSeriesSplit(n_splits=cv_splits, gap=TARGET_HORIZON)
    validation_results: dict[str, dict[str, float]] = {}
    searches: dict[str, GridSearchCV] = {}
    naive_validation = evaluate_predictions(
        y_validation,
        x_validation["portfolio_vol_20"].to_numpy(dtype=float),
    )
    validation_results[f"naive_{TARGET_HORIZON}d_volatility"] = naive_validation

    for name, (model, parameter_grid) in candidate_models().items():
        pipeline = Pipeline(
            [
                ("winsorizer", Winsorizer()),
                ("model", model),
            ]
        )
        pipeline_grid = {
            f"model__{parameter}": values
            for parameter, values in parameter_grid.items()
        }
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=pipeline_grid,
            scoring="neg_root_mean_squared_error",
            cv=time_cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        search.fit(x_train, y_train)
        searches[name] = search
        validation_results[name] = evaluate_model(
            search.best_estimator_,
            x_validation,
            y_validation,
        )
        LOGGER.info(
            "Validation %-12s RMSE %.6f",
            name,
            validation_results[name]["rmse"],
        )

    winning_model_name = min(
        candidate_models(),
        key=lambda name: validation_results[name]["rmse"],
    )
    winning_search = searches[winning_model_name]
    LOGGER.info("Selected on validation data: %s", winning_model_name)
    LOGGER.info("Best parameters: %s", winning_search.best_params_)

    x_train_validation = pd.concat([x_train, x_validation])
    y_train_validation = pd.concat([y_train, y_validation])
    final_model = clone(winning_search.best_estimator_)
    final_model.fit(x_train_validation, y_train_validation)

    # The final model and naïve benchmark are each evaluated once on test data.
    naive_test_metrics = evaluate_predictions(
        y_test,
        x_test["portfolio_vol_20"].to_numpy(dtype=float),
    )
    predictions = np.asarray(final_model.predict(x_test), dtype=float)
    final_metrics = evaluate_predictions(y_test, predictions)
    test_predictions = pd.Series(predictions, index=y_test.index)
    crisis_threshold = float(y_train_validation.quantile(0.75))
    test_regimes = pd.Series(
        np.where(y_test >= crisis_threshold, "crisis", "normal"),
        index=y_test.index,
    )
    regime_test_metrics: dict[str, dict[str, float]] = {}
    for regime in ("normal", "crisis"):
        mask = test_regimes.eq(regime)
        if not mask.any():
            LOGGER.warning(
                "No %s observations in the test set; regime metrics are unavailable.",
                regime,
            )
            continue
        metrics = evaluate_predictions(
            y_test.loc[mask],
            test_predictions.loc[mask].to_numpy(dtype=float),
        )
        regime_test_metrics[regime] = {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }
        LOGGER.info(
            "Test %s regime (n=%d): RMSE %.6f, MAE %.6f",
            regime,
            int(mask.sum()),
            metrics["rmse"],
            metrics["mae"],
        )

    LOGGER.info("Final test metrics for %s:", winning_model_name)
    for metric, value in final_metrics.items():
        LOGGER.info("%s: %.6f", metric.upper(), value)
    LOGGER.info("Naïve baseline test RMSE: %.6f", naive_test_metrics["rmse"])

    model_bundle = {
        "model": final_model,
        "model_name": winning_model_name,
        "feature_names": list(features.columns),
        "target_name": target.name,
        "target_horizon": TARGET_HORIZON,
        "annualization_factor": float(ANNUALIZATION),
        "metrics": final_metrics,
        "regime_test_metrics": regime_test_metrics,
        "validation_metrics": validation_results,
        "naive_test_metrics": naive_test_metrics,
        "best_params": winning_search.best_params_,
        "asset_names": list(prices.columns),
    }
    joblib.dump(model_bundle, models_dir / "best_risk_model.pkl")
    joblib.dump(
        {"feature_names": list(features.columns), "asset_names": list(prices.columns)},
        models_dir / "risk_model_metadata.pkl",
    )
    plot_feature_importance(final_model, list(features.columns), charts_dir / "feature_importance.png")
    plot_predictions(
        x_test.index,
        y_test,
        predictions,
        charts_dir / "predicted_vs_actual_volatility.png",
    )

    print("\nValidation comparison (model selection)")
    for name, metrics in validation_results.items():
        print(f"{name:>22} RMSE: {metrics['rmse']:.6f}")
    print("\nFinal model evaluation on untouched chronological test set")
    print(f"Algorithm: {winning_model_name}")
    for metric, value in final_metrics.items():
        print(f"{metric.upper():>5}: {value:.6f}")
    print(f"Naïve baseline RMSE: {naive_test_metrics['rmse']:.6f}")
    print(f"Model: {models_dir / 'best_risk_model.pkl'}")
    print(f"Feature chart: {charts_dir / 'feature_importance.png'}")
    print(f"Prediction chart: {charts_dir / 'predicted_vs_actual_volatility.png'}")
    return {
        "model_name": winning_model_name,
        "metrics": final_metrics,
        "regime_test_metrics": regime_test_metrics,
        "validation_metrics": validation_results,
        "naive_test_metrics": naive_test_metrics,
        "best_params": winning_search.best_params_,
    }


def main() -> None:
    """Configure paths and execute the standalone training pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    root = repository_root()
    data_dir = args.data_dir or root / "companies"
    output_dir = args.output_dir or root
    train_pipeline(data_dir.resolve(), output_dir.resolve())


if __name__ == "__main__":
    main()
