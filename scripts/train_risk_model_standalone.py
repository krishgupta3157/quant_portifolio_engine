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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("train_risk_model")
RANDOM_STATE = 42
ANNUALIZATION = np.sqrt(252)
TARGET_HORIZON = 20
MIN_ASSETS = 3


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
    """Evaluate one fitted regressor on the untouched chronological holdout."""
    predictions = np.asarray(model.predict(x_test), dtype=float)
    return {
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Save a horizontal feature-importance chart for the selected model."""
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
    split_index = int(len(features) * 0.70)
    x_train, x_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]
    if x_train.empty or x_test.empty:
        raise ValueError("The 70/30 chronological split produced an empty partition.")

    LOGGER.info("Chronological split: %d train rows / %d test rows.", len(x_train), len(x_test))
    cv_splits = min(5, max(2, len(x_train) // 100))
    time_cv = TimeSeriesSplit(n_splits=cv_splits)
    baseline_results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}

    for name, (model, _) in candidate_models().items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        baseline_results[name] = evaluate_model(model, x_test, y_test)
        LOGGER.info("Baseline %-12s RMSE %.6f", name, baseline_results[name]["rmse"])

    baseline_winner = min(baseline_results, key=lambda name: baseline_results[name]["rmse"])
    LOGGER.info("Best baseline algorithm: %s", baseline_winner)
    winning_model, winning_grid = candidate_models()[baseline_winner]
    search = GridSearchCV(
        estimator=winning_model,
        param_grid=winning_grid,
        scoring="neg_root_mean_squared_error",
        cv=time_cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    final_model = search.best_estimator_
    final_metrics = evaluate_model(final_model, x_test, y_test)
    predictions = np.asarray(final_model.predict(x_test), dtype=float)

    LOGGER.info("Tuned winner: %s", baseline_winner)
    LOGGER.info("Best parameters: %s", search.best_params_)
    for metric, value in final_metrics.items():
        LOGGER.info("%s: %.6f", metric.upper(), value)

    model_bundle = {
        "model": final_model,
        "model_name": baseline_winner,
        "feature_names": list(features.columns),
        "target_name": target.name,
        "target_horizon": TARGET_HORIZON,
        "annualization_factor": float(ANNUALIZATION),
        "metrics": final_metrics,
        "best_params": search.best_params_,
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

    print("\nFinal model evaluation on chronological 30% holdout")
    print(f"Algorithm: {baseline_winner}")
    for metric, value in final_metrics.items():
        print(f"{metric.upper():>5}: {value:.6f}")
    print(f"Model: {models_dir / 'best_risk_model.pkl'}")
    print(f"Feature chart: {charts_dir / 'feature_importance.png'}")
    print(f"Prediction chart: {charts_dir / 'predicted_vs_actual_volatility.png'}")
    return {
        "model_name": baseline_winner,
        "metrics": final_metrics,
        "best_params": search.best_params_,
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
