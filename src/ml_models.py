"""
Machine-learning dynamic risk allocation.

The model in this module forecasts the realised volatility observed over the
next ``target_window`` trading observations.  Feature construction is kept
strictly causal: every input is the value available at the prediction date,
whereas the target is shifted backwards so that it belongs to a future date.
Model evaluation uses ``TimeSeriesSplit`` rather than a random split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class DynamicRiskModel:
    """Forecast future market volatility and select a portfolio regime."""

    FEATURE_COLUMNS: Final[tuple[str, ...]] = (
        "VIX_Close",
        "NIFTY_Mom_21d",
        "NIFTY_MA_63d",
        "Risk_Score",
    )

    def __init__(self, macro_data: pd.DataFrame) -> None:
        """Store a defensive copy of the engineered macro data."""
        if not isinstance(macro_data, pd.DataFrame):
            raise TypeError("macro_data must be a pandas DataFrame.")
        if macro_data.empty:
            raise ValueError("macro_data must contain at least one row.")

        self.macro_data = macro_data.copy()
        self.training_data: pd.DataFrame | None = None
        self.feature_columns: list[str] = list(self.FEATURE_COLUMNS)
        self.model: LGBMRegressor | None = None
        self.oof_predictions: pd.Series | None = None
        self.validation_metrics: dict[str, float] | None = None
        self.test_metrics: dict[str, float] | None = None

    def prepare_training_data(self, target_window: int = 30) -> pd.DataFrame:
        """Create causal features and a future realised-volatility target.

        ``shift(-target_window)`` places the volatility at ``t + target_window``
        on row ``t``.  The final ``target_window`` rows consequently have no
        target and are removed.  No feature is shifted forward or calculated
        using observations after the prediction date.
        """
        if target_window <= 0:
            raise ValueError("target_window must be a positive integer.")

        target_source = f"NIFTY_RVol_{target_window}d"
        if target_source not in self.macro_data.columns:
            # Accept the generic name used in the project requirement.
            generic_source = f"RVol_{target_window}d"
            if generic_source not in self.macro_data.columns:
                raise KeyError(
                    f"Missing realised-volatility column: '{target_source}' "
                    f"(or fallback '{generic_source}')."
                )
            target_source = generic_source

        missing_features = [
            column
            for column in self.feature_columns
            if column not in self.macro_data.columns
        ]
        if missing_features:
            raise KeyError(
                "Missing engineered feature columns: "
                + ", ".join(missing_features)
            )

        prepared = self.macro_data.loc[:, self.feature_columns].copy()
        prepared = prepared.apply(pd.to_numeric, errors="coerce")
        prepared[f"Target_Vol_{target_window}d"] = pd.to_numeric(
            self.macro_data[target_source], errors="coerce"
        ).shift(-target_window)
        prepared = prepared.dropna()

        if prepared.empty:
            raise ValueError("No complete rows remain after target preparation.")

        self.training_data = prepared
        return prepared

    def train_walk_forward(
        self,
    ) -> tuple[LGBMRegressor, pd.Series]:
        """Train chronologically with horizon-sized gaps and return OOF predictions.

        Non-gap observations are split 60/20/20 into train, validation, and
        test partitions. Cross-validation and validation are confined to the
        training history; the final test partition is evaluated once after
        fitting on train plus validation data.
        """
        if self.training_data is None:
            self.prepare_training_data()

        assert self.training_data is not None  # Narrowed for static checkers.
        target_column = next(
            column
            for column in self.training_data.columns
            if column.startswith("Target_Vol_")
        )
        features = self.training_data[self.feature_columns]
        target = self.training_data[target_column]

        target_window = int(target_column.rsplit("_", 1)[1][:-1])
        usable_rows = len(self.training_data) - 2 * target_window
        if usable_rows < 10:
            raise ValueError(
                "Not enough observations for a chronological 60/20/20 split "
                f"with {target_window}-row gaps."
            )

        train_count = int(usable_rows * 0.60)
        validation_count = int(usable_rows * 0.20)
        validation_start = train_count + target_window
        validation_end = validation_start + validation_count
        test_start = validation_end + target_window

        x_train = features.iloc[:train_count]
        y_train = target.iloc[:train_count]
        x_validation = features.iloc[validation_start:validation_end]
        y_validation = target.iloc[validation_start:validation_end]
        x_test = features.iloc[test_start:]
        y_test = target.iloc[test_start:]
        if x_train.empty or x_validation.empty or x_test.empty:
            raise ValueError("The gapped chronological split produced an empty partition.")

        n_splits = min(5, max(2, len(x_train) // 100))
        fold_size = len(x_train) // (n_splits + 1)
        initial_train_size = len(x_train) - n_splits * fold_size
        if initial_train_size <= target_window:
            raise ValueError(
                "The training partition is too small for time-series CV "
                f"with a {target_window}-row gap."
            )
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=target_window)
        oof_predictions = pd.Series(float("nan"), index=self.training_data.index)

        for train_indices, validation_indices in splitter.split(x_train):
            fold_model = LGBMRegressor(
                objective="regression",
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1,
            )
            fold_model.fit(
                x_train.iloc[train_indices],
                y_train.iloc[train_indices],
            )
            oof_predictions.loc[x_train.index[validation_indices]] = fold_model.predict(
                x_train.iloc[validation_indices]
            )

        validation_model = LGBMRegressor(
            objective="regression",
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
        validation_model.fit(x_train, y_train)
        validation_predictions = validation_model.predict(x_validation)
        self.validation_metrics = {
            "mae": float(mean_absolute_error(y_validation, validation_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_validation, validation_predictions))),
            "r2": float(r2_score(y_validation, validation_predictions)),
        }

        final_features = pd.concat([x_train, x_validation])
        final_target = pd.concat([y_train, y_validation])
        self.model = LGBMRegressor(
            objective="regression",
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
        self.model.fit(final_features, final_target)
        test_predictions = self.model.predict(x_test)
        self.test_metrics = {
            "mae": float(mean_absolute_error(y_test, test_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
            "r2": float(r2_score(y_test, test_predictions)),
        }
        self.oof_predictions = oof_predictions
        return self.model, oof_predictions

    def get_dynamic_allocation(
        self,
        current_features: pd.DataFrame | pd.Series,
        max_sharpe_weights: Mapping[str, float],
        min_vol_weights: Mapping[str, float],
        risk_threshold: float = 0.20,
    ) -> dict[str, float | str | dict[str, float]]:
        """Select offensive or defensive weights from the predicted regime."""
        if self.model is None:
            raise RuntimeError(
                "The model is not trained. Call train_walk_forward() first."
            )
        if risk_threshold < 0:
            raise ValueError("risk_threshold must be non-negative.")

        if isinstance(current_features, pd.Series):
            feature_row = current_features.to_frame().T
        elif isinstance(current_features, pd.DataFrame):
            feature_row = current_features
        else:
            raise TypeError("current_features must be a pandas Series or DataFrame.")

        if len(feature_row) != 1:
            raise ValueError("current_features must contain exactly one row.")
        missing_features = [
            column for column in self.feature_columns
            if column not in feature_row.columns
        ]
        if missing_features:
            raise KeyError(
                "Missing current feature columns: " + ", ".join(missing_features)
            )

        prediction_features = feature_row[self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if prediction_features.isna().any().any():
            raise ValueError(
                "current_features contains missing or non-numeric feature values."
            )
        predicted_volatility = float(self.model.predict(prediction_features)[0])
        in_defensive_regime = predicted_volatility > risk_threshold
        return {
            "predicted_volatility": predicted_volatility,
            "regime": "Defense" if in_defensive_regime else "Offense",
            "weights": dict(
                min_vol_weights if in_defensive_regime else max_sharpe_weights
            ),
        }


if __name__ == "__main__":
    # Allow ``python src/ml_models.py`` to resolve the sibling modules.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import LocalDataLoader
    from features import FeatureEngineer

    base_dir = Path(__file__).resolve().parent.parent
    nifty_path = base_dir / "NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv"
    vix_path = base_dir / "INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv"

    loader = LocalDataLoader()
    master_data = loader.build_master_dataframe(nifty_path, vix_path)
    macro_data = FeatureEngineer().build_features(master_data)

    risk_model = DynamicRiskModel(macro_data)
    _, oof_predictions = risk_model.train_walk_forward()
    print(f"Validation metrics: {risk_model.validation_metrics}")
    print(f"Final test metrics: {risk_model.test_metrics}")

    max_sharpe = {"TCS": 0.6, "RELIANCE": 0.4}
    min_vol = {"TCS": 0.2, "RELIANCE": 0.8}
    latest_features = macro_data.iloc[-1][risk_model.feature_columns]

    # FeatureEngineer expresses annualised volatility in percentage points
    # (for example, 20.0 means 20%), hence the equivalent threshold is 20.0.
    allocation = risk_model.get_dynamic_allocation(
        latest_features,
        max_sharpe,
        min_vol,
        risk_threshold=20.0,
    )
    print(f"Walk-forward OOF predictions: {oof_predictions.notna().sum()}")
    print(
        "ML allocation for the upcoming month: "
        f"{allocation['regime']} "
        f"(predicted volatility={allocation['predicted_volatility']:.2f}%)"
    )
    print(f"Weights: {allocation['weights']}")
