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
           holidays, then drop leading rows missing either closing price.
        5. Drop rows where either closing price is still missing after
           forward-filling (including dates before one source begins).

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

        # Forward-fill only from observed values; never impute from the future.
        master = master.ffill()

        # Require both sources to have begun before retaining a date.
        master = master.dropna(subset=["NIFTY_Close", "VIX_Close"], how="any")

        log.info(
            "Master DataFrame built — %d rows, %d columns, %s → %s",
            len(master),
            master.shape[1],
            master.index.min().date(),
            master.index.max().date(),
        )
        return master


    # ------------------------------------------------------------------ #
    # Portfolio-wide asset matrix
    # ------------------------------------------------------------------ #
    def load_all_portfolio_assets(
        self,
        data_dir: str,
        missing_threshold: float = 0.30,
    ) -> pd.DataFrame:
        """Dynamically ingest every per-ticker CSV in *data_dir* and build a
        single, institution-grade wide price matrix aligned by date.

        Pipeline
        --------
        1. **Glob** – discover every ``*.csv`` file in *data_dir*; derive the
           ticker symbol from the filename stem.
        2. **Load** – extract ``Date`` and ``Close`` columns from each file,
           parse ``Date`` → ``DatetimeIndex``, sort ascending, deduplicate.
           Files that are empty or lack the required columns are skipped with
           a warning.
        3. **Outer-join** – merge all per-ticker Series into one wide DataFrame
           keyed on the shared DatetimeIndex.  Using an outer join preserves
           every trading day across the universe, regardless of IPO date.
        4. **Column filter** – drop any ticker whose column is missing more
           than *missing_threshold* (default 30 %) of all rows.  This cleanly
           removes recent IPOs that would introduce spurious short histories
           and distort long-run covariance estimates.
        5. **Row trim** – drop leading rows where *any* surviving column is
           still NaN (i.e. walk forward in time until every asset has a valid
           opening price).  This sets the unified start date.
        6. **Forward-fill** – apply ``ffill()`` to patch isolated daily gaps
           caused by market holidays or exchange-specific closures.

        Parameters
        ----------
        data_dir : str
            Path to the folder containing per-ticker CSV files.
            Each file must have at minimum ``Date`` and ``Close`` columns.
        missing_threshold : float, optional
            Maximum allowed fraction of missing rows per column.
            Columns exceeding this threshold are dropped.  Default is ``0.30``.

        Returns
        -------
        pd.DataFrame
            Wide price matrix with a sorted ``DatetimeIndex``.
            Columns are ticker symbols; values are adjusted close prices.

        Raises
        ------
        FileNotFoundError
            If *data_dir* does not exist.
        ValueError
            If no valid ticker files are found after filtering.
        """
        import glob
        import os

        data_path = Path(data_dir)
        if not data_path.exists():
            # Try resolving relative to repository root (parent of src/)
            base_dir = Path(__file__).resolve().parent.parent
            candidate = base_dir / data_dir
            if candidate.exists():
                data_path = candidate
                data_dir = str(candidate)
            elif (base_dir / "companies").exists() and ("raw" in data_dir or "companies" in data_dir):
                data_path = base_dir / "companies"
                data_dir = str(data_path)
            else:
                raise FileNotFoundError(f"Data directory not found: {data_path.resolve()}")

        # ── Step 1: Discover all CSV files ─────────────────────────────── #
        csv_files = sorted(glob.glob(os.path.join(str(data_path), "*.csv")))
        log.info("Found %d CSV files in '%s'", len(csv_files), data_path)

        # ── Steps 2 & 3: Load each ticker and outer-join ────────────────── #
        series_list: list[pd.Series] = []
        skipped: list[str] = []

        # Known non-ticker files to exclude (e.g., aggregated / metadata CSVs)
        _EXCLUDE_STEMS = {"nifty50_all", "stock_metadata"}

        for fpath in csv_files:
            ticker = Path(fpath).stem

            # Skip aggregate / metadata files
            if ticker.lower() in _EXCLUDE_STEMS:
                log.debug("Skipping non-ticker file: %s", ticker)
                continue

            try:
                raw = pd.read_csv(
                    fpath,
                    usecols=lambda c: c.strip() in ("Date", "Close"),
                    dtype=str,
                    na_values=["", " ", "-", "N/A"],
                )

                # Normalise column names
                raw.columns = raw.columns.str.strip()

                # Guard: require both Date and Close
                if "Date" not in raw.columns or "Close" not in raw.columns:
                    log.warning("Skipping '%s' — missing Date or Close column.", ticker)
                    skipped.append(ticker)
                    continue

                # Must have at least 2 data rows
                if len(raw.dropna(subset=["Close"])) < 2:
                    log.warning("Skipping '%s' — insufficient data rows.", ticker)
                    skipped.append(ticker)
                    continue

                # Parse and clean
                raw["Date"] = pd.to_datetime(raw["Date"].str.strip(), errors="coerce")
                raw["Close"] = (
                    raw["Close"]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                    .pipe(pd.to_numeric, errors="coerce")
                )

                # Set DatetimeIndex, sort ascending, deduplicate
                s: pd.Series = (
                    raw.dropna(subset=["Date"])
                       .set_index("Date")["Close"]
                       .sort_index()
                       .loc[lambda x: ~x.index.duplicated(keep="first")]
                       .rename(ticker)
                )

                series_list.append(s)

            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping '%s' — error during load: %s", ticker, exc)
                skipped.append(ticker)

        if not series_list:
            raise ValueError(
                f"No valid ticker files found in '{data_dir}'. "
                "Ensure CSVs contain at minimum 'Date' and 'Close' columns."
            )

        log.info(
            "Successfully parsed %d tickers; %d skipped: %s",
            len(series_list),
            len(skipped),
            skipped or "none",
        )

        # Outer join → wide DataFrame (all dates × all tickers)
        prices: pd.DataFrame = pd.concat(series_list, axis=1, join="outer")
        prices.index.name = "Date"
        prices.sort_index(inplace=True)

        log.info(
            "Raw wide matrix: %d rows × %d columns",
            prices.shape[0],
            prices.shape[1],
        )

        # ── Step 4: Drop columns exceeding the missing data threshold ─────── #
        missing_frac: pd.Series = prices.isna().mean()
        survivors = missing_frac[missing_frac <= missing_threshold].index.tolist()
        dropped_cols = missing_frac[missing_frac > missing_threshold].index.tolist()

        if dropped_cols:
            log.info(
                "Dropping %d tickers (>%.0f%% missing): %s",
                len(dropped_cols),
                missing_threshold * 100,
                dropped_cols,
            )

        prices = prices[survivors]

        # ── Step 5: Trim leading rows where any surviving column is NaN ────── #
        # Find the first date where ALL survivors have a valid price
        first_valid_row = prices.dropna(how="any").index.min()
        prices = prices.loc[first_valid_row:]

        log.info(
            "After trimming leading NaNs — start date: %s",
            first_valid_row.date() if pd.notna(first_valid_row) else "N/A",
        )

        # ── Step 6: Forward-fill isolated holiday gaps ─────────────────────── #
        prices = prices.ffill()

        log.info(
            "Final portfolio matrix: %d rows × %d columns | %s → %s",
            prices.shape[0],
            prices.shape[1],
            prices.index.min().date(),
            prices.index.max().date(),
        )

        return prices


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
