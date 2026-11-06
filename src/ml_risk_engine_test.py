"""Standalone LightGBM test for the dynamic risk engine.

The script deliberately runs outside Streamlit.  It loads the local 42-asset
price matrix, creates market-level risk features, predicts volatility 30
trading days ahead, and evaluates the model on the latest chronological 30%
of observations.

Run from the repository root with:

    python src/ml_risk_engine_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)

# Make the repository root importable when this file is executed directly.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_loader import LocalDataLoader


DATA_DIRECTORY = REPOSITORY_ROOT / "companies"
LOOKAHEAD_DAYS = 30
TRAIN_FRACTION = 0.70
ANNUALIZATION_FACTOR = np.sqrt(252.0)


def load_market_prices(data_directory: Path = DATA_DIRECTORY) -> pd.DataFrame:
    """Load the established aligned multi-asset close-price matrix."""
    loader = LocalDataLoader()
    prices = loader.load_all_portfolio_assets(
        str(data_directory),
        missing_threshold=0.30,
    )
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    prices = prices.ffill().dropna(how="any")
    if prices.shape[1] < 3:
        raise ValueError("At least three usable assets are required.")
    if prices.empty:
        raise ValueError("The unified price matrix is empty.")
    return prices


def engineer_features(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create present-time features and a strictly future volatility target.

    ``shift(-LOOKAHEAD_DAYS)`` aligns the volatility observed at date ``t+30``
    with features observed at date ``t``.  The final 30 rows are removed
    because no future target exists for them.
    """
    equal_weight_market = prices.pct_change().mean(axis=1)
    market_level = prices.div(prices.iloc[0]).mean(axis=1)

    realized_volatility = (
        equal_weight_market.rolling(LOOKAHEAD_DAYS).std()
        * ANNUALIZATION_FACTOR
    )
    moving_average_63 = market_level.rolling(63).mean()

    features = pd.DataFrame(index=prices.index)
    features["market_return_1d"] = equal_weight_market
    features["market_momentum_21d"] = market_level.pct_change(21)
    features["market_price_to_ma_63d"] = market_level / moving_average_63 - 1.0
    features["market_volatility_30d"] = realized_volatility
    features["market_return_volatility_21d"] = (
        equal_weight_market.rolling(21).std() * ANNUALIZATION_FACTOR
    )
    features["market_return_volatility_63d"] = (
        equal_weight_market.rolling(63).std() * ANNUALIZATION_FACTOR
    )

    asset_returns = prices.pct_change()
    asset_volatility_30d = (
        asset_returns.rolling(LOOKAHEAD_DAYS).std()
        * ANNUALIZATION_FACTOR
    )
    features["cross_asset_volatility_mean_30d"] = asset_volatility_30d.mean(axis=1)
    features["cross_asset_volatility_std_30d"] = asset_volatility_30d.std(axis=1)

    # Average rolling pairwise correlation is a compact contagion feature.
    rolling_correlation = asset_returns.rolling(30, min_periods=20).corr()
    features["average_pairwise_correlation_30d"] = (
        rolling_correlation.groupby(level=0).mean().mean(axis=1)
    )

    # The target is future realized volatility, never a contemporaneous value.
    target = realized_volatility.shift(-LOOKAHEAD_DAYS)
    target.name = "future_30d_realized_volatility"

    dataset = features.join(target).replace([np.inf, -np.inf], np.nan).dropna()
    if dataset.empty:
        raise ValueError("Feature engineering produced no complete observations.")
    return dataset.drop(columns=[target.name]), dataset[target.name]


def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the ordered observations into the first 70% and final 30%."""
    split_at = int(len(features) * TRAIN_FRACTION)
    if split_at <= 0 or split_at >= len(features):
        raise ValueError("The dataset is too small for a 70/30 chronological split.")
    return (
        features.iloc[:split_at],
        features.iloc[split_at:],
        target.iloc[:split_at],
        target.iloc[split_at:],
    )


def train_and_evaluate(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> lgb.LGBMRegressor:
    """Train LightGBM and print regression plus crisis-regime diagnostics."""
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.10,
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, predictions)

    crisis_threshold = float(y_train.quantile(0.75))
    actual_regime = (y_test > crisis_threshold).astype(int)
    predicted_regime = (predictions > crisis_threshold).astype(int)

    print("\n" + "=" * 72)
    print("LIGHTGBM DYNAMIC RISK ENGINE — CHRONOLOGICAL HOLDOUT RESULTS")
    print("=" * 72)
    print(f"Train observations: {len(x_train):,}")
    print(f"Test observations:  {len(x_test):,}")
    print(f"Test period:        {x_test.index.min().date()} to {x_test.index.max().date()}")
    print(f"Crisis threshold:   {crisis_threshold:.6f} (75th percentile of training target)")
    print("\nRegression metrics")
    print(f"MAE:                 {mae:.6f}")
    print(f"MSE:                 {mse:.6f}")
    print(f"RMSE:                {rmse:.6f}")
    print("\nRegression metrics in percentage terms")
    print(f"MAE:                 {mae * 100:.2f} percentage points")
    print(f"RMSE:                {rmse * 100:.2f} percentage points")
    print("\nCrisis-regime classification report")
    print(
        classification_report(
            actual_regime,
            predicted_regime,
            labels=[0, 1],
            target_names=["Normal", "Crisis"],
            zero_division=0,
            digits=4,
        )
    )

    report = classification_report(
        actual_regime,
        predicted_regime,
        labels=[0, 1],
        target_names=["Normal", "Crisis"],
        output_dict=True,
        zero_division=0,
    )
    print(f"Crisis recall:      {report['Crisis']['recall']:.4f}")
    print("\nRegime performance percentages")
    print(f"Accuracy:            {report['accuracy'] * 100:.2f}%")
    print(f"Normal precision:    {report['Normal']['precision'] * 100:.2f}%")
    print(f"Normal recall:       {report['Normal']['recall'] * 100:.2f}%")
    print(f"Normal F1-score:     {report['Normal']['f1-score'] * 100:.2f}%")
    print(f"Crisis precision:    {report['Crisis']['precision'] * 100:.2f}%")
    print(f"Crisis recall:       {report['Crisis']['recall'] * 100:.2f}%")
    print(f"Crisis F1-score:     {report['Crisis']['f1-score'] * 100:.2f}%")
    print("\nTop LightGBM features")
    importance = (
        pd.Series(model.feature_importances_, index=x_train.columns)
        .sort_values(ascending=False)
    )
    for feature_name, score in importance.head(10).items():
        print(f"{feature_name:<40}{int(score):>8}")
    return model


def main() -> None:
    """Execute ingestion, feature engineering, chronological split, and testing."""
    print(f"Loading asset prices from: {DATA_DIRECTORY}")
    prices = load_market_prices()
    print(f"Loaded price matrix: {prices.shape[0]:,} dates × {prices.shape[1]} assets")

    features, target = engineer_features(prices)
    x_train, x_test, y_train, y_test = chronological_split(features, target)
    train_and_evaluate(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
