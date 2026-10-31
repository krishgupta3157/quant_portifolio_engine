"""
src/data_loader.py
==================
LocalDataLoader — ingests and cleans the two raw NSE CSV exports:

  • "NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv"
  • "INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv"

Both files arrive with:
  - Values quoted (e.g. "21,466.40")          → strip surrounding quotes & commas
  - Dates as human-readable strings             → parse to DatetimeIndex
  - Rows in DESCENDING date order              → must be reversed
  - Volume using Indian numeral grouping        → same strip-commas treatment
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_loader")


# --------------------------------------------------------------------------- #
# Helper — strip Indian-numeral comma formatting from a Series
# --------------------------------------------------------------------------- #
def _strip_commas(series: pd.Series) -> pd.Series:
    """Remove all commas from a string Series and cast to float64.

    NSE exports use Indian numeral grouping (e.g. "3,21,74,72,081") for
    volume and thousands separators (e.g. "21,466.40") for prices.  A simple
    str.replace(',', '') handles both cases uniformly.

    Parameters
    ----------
    series : pd.Series
        Raw string column as read from CSV.

    Returns
    -------
    pd.Series
        float64 Series with NaN where the original value was non-numeric.
    """
    return (
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


# =========================================================================== #
# LocalDataLoader
# =========================================================================== #
class LocalDataLoader:
    """Load, clean, and merge the NIFTY 500 and India VIX local CSV exports.

    All methods return a ``pd.DataFrame`` with a ``DatetimeIndex`` sorted
    in **ascending** chronological order.  No data is mutated in place; each
    method returns a fresh copy.

    Typical usage
    -------------
    >>> loader = LocalDataLoader()
    >>> master = loader.build_master_dataframe(
    ...     nifty_path="NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv",
    ...     vix_path="INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv",
    ... )
    >>> print(master.tail())
    """

    # ------------------------------------------------------------------ #
    # NIFTY 500
    # ------------------------------------------------------------------ #
    def load_and_clean_nifty(self, filepath: str | Path) -> pd.DataFrame:
        """Read and clean the NIFTY 500 CSV export.

        Cleaning steps (all vectorised, no row-level loops):

        1. Read raw CSV (all columns arrive as quoted strings).
        2. Strip trailing ``%`` from *Change(%)* and cast to float.
        3. Remove comma separators from *Open, Close, High, Low, Volume*
           and cast to ``float64``.
        4. Parse *Date* strings → ``datetime64`` and set as index.
        5. Sort index ascending (raw file is newest-first).
        6. Drop duplicated dates, keeping the first occurrence.

        Parameters
        ----------
        filepath : str | Path
            Absolute or relative path to the NIFTY 500 CSV file.

        Returns
        -------
        pd.DataFrame
            Cleaned NIFTY DataFrame with DatetimeIndex, columns:
            ``Open, Close, High, Low, Volume, Change_Pct``.
        """
        path = Path(filepath)
        log.info("Loading NIFTY 500 from: %s", path.name)

        df: pd.DataFrame = pd.read_csv(
            path,
            dtype=str,          # read everything as string → we clean ourselves
            na_values=["", " ", "-", "N/A"],
        )

        # ── Normalise column names ──────────────────────────────────────── #
        df.columns = df.columns.str.strip()

        # ── Numeric price / volume columns ─────────────────────────────── #
        numeric_cols: list[str] = ["Open", "Close", "High", "Low", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = _strip_commas(df[col])

        # ── Percentage change column ────────────────────────────────────── #
        if "Change(%)" in df.columns:
            df["Change_Pct"] = (
                df["Change(%)"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
            df = df.drop(columns=["Change(%)"])

        # ── Date index ─────────────────────────────────────────────────── #
        df["Date"] = pd.to_datetime(df["Date"].str.strip(), dayfirst=True)
        df = (
            df.set_index("Date")
              .sort_index(ascending=True)          # raw data is descending
              .loc[~df.set_index("Date").index.duplicated(keep="first")]
        )

        log.info(
            "NIFTY 500 loaded — %d rows, %s → %s",
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )
        return df

    # ------------------------------------------------------------------ #
    # INDIA VIX
    # ------------------------------------------------------------------ #
    def load_and_clean_vix(self, filepath: str | Path) -> pd.DataFrame:
        """Read and clean the India VIX CSV export.

        India VIX values are already decimal floats (e.g. ``"18.72"``),
        but still arrive quoted.  The pipeline is identical to NIFTY except
        no Volume column exists.

        Parameters
        ----------
        filepath : str | Path
            Absolute or relative path to the India VIX CSV file.

        Returns
        -------
        pd.DataFrame
            Cleaned VIX DataFrame with DatetimeIndex, columns:
            ``Open, Close, High, Low, Change_Pct``.
        """
        path = Path(filepath)
        log.info("Loading India VIX from: %s", path.name)

        df: pd.DataFrame = pd.read_csv(
            path,
            dtype=str,
            na_values=["", " ", "-", "N/A"],
        )

        df.columns = df.columns.str.strip()

        numeric_cols: list[str] = ["Open", "Close", "High", "Low"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = _strip_commas(df[col])

        if "Change(%)" in df.columns:
            df["Change_Pct"] = (
                df["Change(%)"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
            df = df.drop(columns=["Change(%)"])

        df["Date"] = pd.to_datetime(df["Date"].str.strip(), dayfirst=True)
        df = (
            df.set_index("Date")
              .sort_index(ascending=True)
              .loc[~df.set_index("Date").index.duplicated(keep="first")]
        )

        log.info(
            "India VIX loaded — %d rows, %s → %s",
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )
        return df

    # ------------------------------------------------------------------ #
    # Master merge
    # ------------------------------------------------------------------ #
    def build_master_dataframe(
        self,
        nifty_path: str | Path,
        vix_path: str | Path,
    ) -> pd.DataFrame:
        """Merge NIFTY 500 and India VIX into a single master DataFrame.

        Steps
        -----
        1. Load & clean both sources independently.
        2. Rename price columns to avoid collisions:
           - NIFTY ``Close`` → ``NIFTY_Close``
           - VIX   ``Close`` → ``VIX_Close``
           (similarly for Open / High / Low)
        3. Outer-join on DatetimeIndex so no trading day is silently dropped.
        4. Forward-fill (``ffill``) gaps caused by instrument-specific
           holidays, then back-fill (``bfill``) any leading NaNs.
        5. Drop rows where **both** ``NIFTY_Close`` and ``VIX_Close``
           are NaN (i.e. true non-trading days such as weekends).

        Parameters
        ----------
        nifty_path : str | Path
            Path to the NIFTY 500 CSV.
        vix_path : str | Path
            Path to the India VIX CSV.

        Returns
        -------
        pd.DataFrame
            Master DataFrame with DatetimeIndex (ascending), columns:
            ``NIFTY_Open, NIFTY_Close, NIFTY_High, NIFTY_Low,
              NIFTY_Volume, NIFTY_Change_Pct,
              VIX_Open,   VIX_Close,   VIX_High,   VIX_Low,
              VIX_Change_Pct``.
        """
        nifty = self.load_and_clean_nifty(nifty_path).add_prefix("NIFTY_")
        vix   = self.load_and_clean_vix(vix_path).add_prefix("VIX_")

        master: pd.DataFrame = nifty.join(vix, how="outer")

        # Forward-fill holiday gaps, then back-fill leading NaNs
        master = master.ffill().bfill()

        # Drop rows that are pure non-trading days (weekends / public holidays
        # not covered by either dataset)
        master = master.dropna(subset=["NIFTY_Close", "VIX_Close"], how="all")

        log.info(
            "Master DataFrame built — %d rows, %d columns, %s → %s",
            len(master),
            master.shape[1],
            master.index.min().date(),
            master.index.max().date(),
        )
        return master


# =========================================================================== #
# __main__ — quick smoke test
# =========================================================================== #
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Assume CSVs live one level above src/
    base = Path(__file__).resolve().parent.parent

    nifty_csv = base / "NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv"
    vix_csv   = base / "INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv"

    for p in (nifty_csv, vix_csv):
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    loader = LocalDataLoader()
    master = loader.build_master_dataframe(nifty_csv, vix_csv)

    # Pipe through FeatureEngineer (must run after features.py is created)
    try:
        from features import FeatureEngineer
        fe = FeatureEngineer()
        master = fe.build_features(master)
        print("\n--- Feature-Engineered Master (tail 10) ---")
    except ImportError:
        print("\n--- Raw Master DataFrame (tail 10) ---")
        print("   (Run after src/features.py is in place for full output)")

    print(master.tail(10).to_string())
    print(f"\nShape : {master.shape}")
    print(f"Dtypes:\n{master.dtypes.to_string()}")
    print(f"\nMissing values:\n{master.isna().sum().to_string()}")
