"""
src/features.py
===============
FeatureEngineer — transforms the raw master DataFrame produced by
``LocalDataLoader.build_master_dataframe()`` into a richly annotated,
ML-ready feature matrix.

Feature families produced
-------------------------
Returns
    NIFTY_Log_Return          — daily log return (ln P_t / P_{t-1})
    NIFTY_Simple_Return       — daily simple % return (for reference)

Rolling risk
    NIFTY_RVol_{w}d           — annualised realised volatility (windows 10, 21, 63 days)
    NIFTY_RVol_30d            — canonical 30-day annualised vol (primary risk signal)

Momentum / trend
    NIFTY_Mom_{w}d            — cumulative log return over past w days (5, 21, 63, 252)
    NIFTY_MA_{w}d             — simple moving average of Close
    NIFTY_Price_to_MA252      — Close / 252-day MA (trend deviation ratio)

Volatility regime
    VIX_30d_MA                — 30-day moving average of VIX_Close
    VIX_Zscore                — z-score of VIX_Close over trailing 252 days
    VIX_Percentile_252d       — rolling 252-day percentile rank of VIX_Close

Drawdown
    NIFTY_Drawdown_Pct        — current drawdown from rolling all-time-high (%)

Educational risk score
    Risk_Score                — integer 1-10 mapping of annualised volatility
    Risk_Label                — human-readable label (e.g. "Low", "Elevated")
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("features")

# --------------------------------------------------------------------------- #
# Risk score thresholds (annualised volatility %)
# Calibrated empirically on NIFTY 500 (2010-2026)
# --------------------------------------------------------------------------- #
_VOL_BINS: list[float] = [
    0.0,   # score 1 boundary (floor)
    8.0,   # score 2
    11.0,  # score 3
    14.0,  # score 4
    17.0,  # score 5
    21.0,  # score 6
    26.0,  # score 7
    32.0,  # score 8
    40.0,  # score 9
    55.0,  # score 10
    np.inf,
]

_RISK_LABELS: dict[int, str] = {
    1:  "Ultra Safe",
    2:  "Very Low",
    3:  "Low",
    4:  "Below Average",
    5:  "Moderate",
    6:  "Elevated",
    7:  "High",
    8:  "Very High",
    9:  "Extreme",
    10: "Panic / Crisis",
}


# =========================================================================== #
# FeatureEngineer
# =========================================================================== #
class FeatureEngineer:
    """Transform the raw master DataFrame into an ML-ready feature matrix.

    All computations are **fully vectorised** using Pandas / NumPy rolling
    and shift operations.  No Python-level row iteration is performed.

    Parameters
    ----------
    vol_windows : sequence of int, optional
        Rolling windows (in calendar days) for realised volatility.
        Defaults to ``(10, 21, 30, 63)``.
    mom_windows : sequence of int, optional
        Look-back windows for momentum features.
        Defaults to ``(5, 21, 63, 252)``.
    ma_windows : sequence of int, optional
        Windows for simple moving average features.
        Defaults to ``(21, 63, 252)``.

    Examples
    --------
    >>> from data_loader import LocalDataLoader
    >>> from features import FeatureEngineer
    >>> master = LocalDataLoader().build_master_dataframe(nifty, vix)
    >>> df = FeatureEngineer().build_features(master)
    >>> df[["NIFTY_Log_Return", "NIFTY_RVol_30d", "Risk_Score", "Risk_Label"]].tail()
    """

    def __init__(
        self,
        vol_windows:  Sequence[int] = (10, 21, 30, 63),
        mom_windows:  Sequence[int] = (5, 21, 63, 252),
        ma_windows:   Sequence[int] = (21, 63, 252),
    ) -> None:
        self.vol_windows  = list(vol_windows)
        self.mom_windows  = list(mom_windows)
        self.ma_windows   = list(ma_windows)

    # ------------------------------------------------------------------ #
    # 1. Log returns
    # ------------------------------------------------------------------ #
    def compute_log_returns(
        self,
        df: pd.DataFrame,
        column_name: str = "NIFTY_Close",
    ) -> pd.DataFrame:
        """Compute daily log returns and simple % returns.

        Why quants use **log returns** instead of simple % returns
        ----------------------------------------------------------
        A simple return on day t is defined as:

            r_t = (P_t - P_{t-1}) / P_{t-1}

        While intuitive, simple returns have an awkward property: they are
        *not* additive over time.  The 5-day simple return is NOT the sum of
        the five daily simple returns; it requires multiplication.

        Log returns solve this elegantly:

            log_r_t = ln(P_t / P_{t-1})

        Key advantages
        ~~~~~~~~~~~~~~
        1. **Time-additivity** — the log return over N days is simply the sum
           of N daily log returns: ``Σ log_r_t``.  This makes multi-period
           aggregation trivially fast with vectorised ``cumsum()``.

        2. **Symmetry** — a +100% simple gain followed by a −50% loss leaves
           you at zero.  Log returns treat gains and losses symmetrically
           (ln(2) + ln(0.5) = 0), which is more mathematically honest.

        3. **Approximate normality** — log returns are closer to normally
           distributed than simple returns, satisfying assumptions of most
           ML models, Markowitz MPT, and Black-Scholes pricing.

        4. **No negative prices** — since ``ln(P_t/P_{t-1})`` requires P > 0,
           the log transform naturally encodes this economic constraint.

        The first row will be NaN (no prior day to compare to); this is
        expected and handled downstream by ``build_features``.

        Parameters
        ----------
        df : pd.DataFrame
            Master DataFrame containing a price column named *column_name*.
        column_name : str
            Column to compute returns on.  Defaults to ``"NIFTY_Close"``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with two new columns appended:
            ``{column_name}_Log_Return`` and ``{column_name}_Simple_Return``.
        """
        out = df.copy()
        price = out[column_name]

        # Log return: ln(P_t / P_{t-1})
        out["NIFTY_Log_Return"] = np.log(price / price.shift(1))

        # Simple return: (P_t - P_{t-1}) / P_{t-1}  — kept for reference
        out["NIFTY_Simple_Return"] = price.pct_change()

        log.info("Computed log & simple returns for column '%s'.", column_name)
        return out

    # ------------------------------------------------------------------ #
    # 2. Rolling volatility
    # ------------------------------------------------------------------ #
    def compute_rolling_volatility(
        self,
        df: pd.DataFrame,
        return_col: str = "NIFTY_Log_Return",
        window: int = 30,
    ) -> pd.DataFrame:
        """Compute annualised rolling realised volatility.

        Realised volatility is the standard deviation of log returns over a
        rolling window, scaled to an annual figure by multiplying by √252
        (the approximate number of trading days in a year).

            σ_ann = rolling_std(log_r, window) × √252

        This is the standard practitioner definition used by risk desks,
        clearinghouses, and regulatory capital models (e.g. Basel III).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing *return_col*.
        return_col : str
            Column name of log returns.  Defaults to ``"NIFTY_Log_Return"``.
        window : int
            Look-back window in *trading days*.  Defaults to 30.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with new column ``NIFTY_RVol_{window}d``
            expressing annualised volatility as a **percentage** (×100).
        """
        out = df.copy()
        col = f"NIFTY_RVol_{window}d"
        out[col] = (
            out[return_col]
            .rolling(window=window, min_periods=max(5, window // 2))
            .std()
            .mul(np.sqrt(252) * 100)          # annualise → %
        )
        log.info("Computed %d-day rolling vol → '%s'.", window, col)
        return out

    # ------------------------------------------------------------------ #
    # 3. Momentum
    # ------------------------------------------------------------------ #
    def _compute_momentum(
        self,
        df: pd.DataFrame,
        return_col: str = "NIFTY_Log_Return",
    ) -> pd.DataFrame:
        """Compute cumulative log-return momentum over multiple windows."""
        out = df.copy()
        for w in self.mom_windows:
            out[f"NIFTY_Mom_{w}d"] = (
                out[return_col]
                .rolling(window=w, min_periods=max(3, w // 4))
                .sum()
                .mul(100)                      # express as %
            )
        return out

    # ------------------------------------------------------------------ #
    # 4. Moving averages + price-to-MA ratio
    # ------------------------------------------------------------------ #
    def _compute_moving_averages(
        self,
        df: pd.DataFrame,
        price_col: str = "NIFTY_Close",
    ) -> pd.DataFrame:
        """Simple moving averages and a 252-day trend deviation ratio."""
        out = df.copy()
        for w in self.ma_windows:
            out[f"NIFTY_MA_{w}d"] = (
                out[price_col]
                .rolling(window=w, min_periods=max(10, w // 4))
                .mean()
            )
        ma252 = out.get("NIFTY_MA_252d")
        if ma252 is not None:
            out["NIFTY_Price_to_MA252"] = out[price_col] / ma252
        return out

    # ------------------------------------------------------------------ #
    # 5. Drawdown
    # ------------------------------------------------------------------ #
    def _compute_drawdown(
        self,
        df: pd.DataFrame,
        price_col: str = "NIFTY_Close",
    ) -> pd.DataFrame:
        """Current drawdown from expanding all-time high (expressed as %)."""
        out  = df.copy()
        peak = out[price_col].expanding().max()
        out["NIFTY_Drawdown_Pct"] = (out[price_col] - peak) / peak * 100
        return out

    # ------------------------------------------------------------------ #
    # 6. VIX-derived features
    # ------------------------------------------------------------------ #
    def _compute_vix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich VIX signal with moving average, z-score, and percentile rank."""
        out = df.copy()

        # 30-day MA of VIX
        out["VIX_30d_MA"] = (
            out["VIX_Close"]
            .rolling(window=30, min_periods=15)
            .mean()
        )

        # Rolling 252-day z-score  →  how unusual is today's VIX?
        roll = out["VIX_Close"].rolling(window=252, min_periods=63)
        out["VIX_Zscore"] = (out["VIX_Close"] - roll.mean()) / roll.std()

        # Rolling 252-day percentile rank  (0–100)
        out["VIX_Percentile_252d"] = (
            roll.rank(pct=True).mul(100)
        )

        log.info("Computed VIX-derived features (MA30, Z-score, Percentile).")
        return out

    # ------------------------------------------------------------------ #
    # 7. Educational risk score
    # ------------------------------------------------------------------ #
    def generate_educational_risk_score(
        self,
        volatility_series: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Map annualised volatility (%) to an integer Risk Score (1–10).

        This "what-if" scaling converts the continuous volatility signal into
        a categorical scale that is immediately interpretable by non-technical
        stakeholders — no quant background required.

        Threshold logic (preferred over ``pd.qcut`` which gives equal-frequency
        bins regardless of economic meaning):

        ┌────────────┬──────────────────────────────────┐
        │  Vol (%)   │  Score   Label                   │
        ├────────────┼──────────────────────────────────┤
        │ < 8        │   1    Ultra Safe                 │
        │ 8–11       │   2    Very Low                   │
        │ 11–14      │   3    Low                        │
        │ 14–17      │   4    Below Average              │
        │ 17–21      │   5    Moderate                   │
        │ 21–26      │   6    Elevated                   │
        │ 26–32      │   7    High                       │
        │ 32–40      │   8    Very High                  │
        │ 40–55      │   9    Extreme                    │
        │ ≥ 55       │  10    Panic / Crisis             │
        └────────────┴──────────────────────────────────┘

        NaN inputs (insufficient history for rolling window) are preserved
        as NaN in the output rather than forced to a score.

        Parameters
        ----------
        volatility_series : pd.Series
            Annualised realised volatility in **percent** (e.g. 18.5 means 18.5%).

        Returns
        -------
        (score_series, label_series) : tuple[pd.Series, pd.Series]
            ``score_series`` — Int64 Series with values 1-10 (or pd.NA).
            ``label_series`` — string Series with human-readable labels.
        """
        bins   = _VOL_BINS                           # 11 edges → 10 buckets
        labels = list(range(1, 11))                  # [1, 2, ..., 10]

        score_series: pd.Series = pd.cut(
            volatility_series,
            bins=bins,
            labels=labels,
            right=False,                             # [left, right)
            include_lowest=True,
        ).astype("Int64")                            # nullable integer → NaN preserved

        label_series: pd.Series = score_series.map(_RISK_LABELS)

        log.info(
            "Risk scores assigned — distribution:\n%s",
            score_series.value_counts().sort_index().to_string(),
        )
        return score_series, label_series

    # ------------------------------------------------------------------ #
    # 8. Master builder
    # ------------------------------------------------------------------ #
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the complete feature engineering pipeline.

        Execution order
        ~~~~~~~~~~~~~~~
        1. Log & simple returns
        2. Rolling volatility (all windows in *vol_windows*)
        3. Momentum (all windows in *mom_windows*)
        4. Moving averages + price/MA ratio
        5. Drawdown from all-time-high
        6. VIX-derived features
        7. Educational risk score (based on 30-day vol)
        8. Drop the initial NaN rows caused by rolling-window warm-up

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``LocalDataLoader.build_master_dataframe()``.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame ready for ML ingestion and Streamlit charts.
            Shape: (N_trading_days, original_cols + engineered_features).
        """
        log.info("Building features on DataFrame of shape %s …", df.shape)

        out = self.compute_log_returns(df)

        # Rolling volatility for every configured window
        for w in self.vol_windows:
            out = self.compute_rolling_volatility(out, window=w)

        out = self._compute_momentum(out)
        out = self._compute_moving_averages(out)
        out = self._compute_drawdown(out)
        out = self._compute_vix_features(out)

        # Educational risk score uses the primary 30-day volatility
        if "NIFTY_RVol_30d" in out.columns:
            out["Risk_Score"], out["Risk_Label"] = (
                self.generate_educational_risk_score(out["NIFTY_RVol_30d"])
            )

        # Drop rows where the primary return is NaN (first row only)
        out = out.dropna(subset=["NIFTY_Log_Return"])

        log.info(
            "Feature engineering complete — final shape: %s  (%d features added)",
            out.shape,
            out.shape[1] - df.shape[1],
        )
        return out
