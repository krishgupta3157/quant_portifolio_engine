"""
src/optimization.py
===================
PortfolioOptimizer — Classical Modern Portfolio Theory (MPT) engine.

Uses PyPortfolioOpt to build:
  * 1/N Equal-Weight baseline
  * Max Sharpe Ratio portfolio
  * Minimum Volatility portfolio

All three methods accept the cleaned wide price matrix produced by
``LocalDataLoader.load_all_portfolio_assets()`` and return weights as a
plain dict[ticker -> weight], ready for downstream risk-attribution or
rebalancing logic.

Dependencies
------------
    pip install pyportfolioopt

Design Decisions
----------------
- We pass the **price** DataFrame directly (not returns) to PyPortfolioOpt;
  the library handles log-return calculation internally via
  ``mean_historical_return`` and Ledoit-Wolf covariance shrinkage.
- Weights are cleaned (round-tripped through ``clean_weights``) to drop
  dust positions below the 1-bp threshold, keeping the output human-readable
  and rebalancing-friendly.
- Ledoit-Wolf covariance shrinkage and L2 regularization reduce estimation
  instability for portfolios with highly correlated assets.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict

import pandas as pd

# PyPortfolioOpt -- graceful import error
try:
    from pypfopt import expected_returns, objective_functions, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
except ImportError as exc:
    raise ImportError(
        "PyPortfolioOpt is required.  Install it with:  pip install pyportfolioopt"
    ) from exc

# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimization")


class OptimizationError(RuntimeError):
    """Raised when an efficient-frontier optimization cannot be completed."""


# =========================================================================== #
# PortfolioOptimizer
# =========================================================================== #
class PortfolioOptimizer:
    """MPT-based portfolio optimizer built on top of PyPortfolioOpt.

    Parameters
    ----------
    asset_prices_df : pd.DataFrame
        Wide, cleaned price matrix with a ``DatetimeIndex`` (ascending) and
        one column per ticker.  Produced by
        ``LocalDataLoader.load_all_portfolio_assets()``.
    risk_free_rate : float, optional
        Annualised risk-free rate used in Sharpe ratio computations.
        Defaults to 0.065 (approx. current Indian 10-year G-Sec yield).
    frequency : int, optional
        Number of trading days per year used to annualise returns/volatility.
        Defaults to 252 (NSE convention).

    Attributes
    ----------
    prices : pd.DataFrame
        The cleaned price matrix passed at construction.
    tickers : list[str]
        Ordered list of surviving ticker symbols.
    n_assets : int
        Number of surviving assets.
    mu : pd.Series
        Annualised mean historical returns (calculated lazily).
    S : pd.DataFrame
        Sample covariance matrix (calculated lazily).

    Examples
    --------
    >>> loader = LocalDataLoader()
    >>> prices = loader.load_all_portfolio_assets("companies")
    >>> opt = PortfolioOptimizer(prices)
    >>> ew  = opt.get_equal_weights()
    >>> msr = opt.optimize_max_sharpe()
    >>> mv  = opt.optimize_min_volatility()
    """

    def __init__(
        self,
        asset_prices_df: pd.DataFrame,
        risk_free_rate: float = 0.065,
        frequency: int = 252,
    ) -> None:
        if asset_prices_df.empty:
            raise ValueError("asset_prices_df must not be empty.")

        self.prices: pd.DataFrame = asset_prices_df.copy()
        self.tickers: list[str] = list(self.prices.columns)
        self.n_assets: int = len(self.tickers)
        self.risk_free_rate: float = risk_free_rate
        self.frequency: int = frequency

        # Lazily computed -- populated on first call to _build_inputs()
        self._mu: pd.Series | None = None
        self._S: pd.DataFrame | None = None

        log.info(
            "PortfolioOptimizer initialised -- %d assets | %s -> %s",
            self.n_assets,
            self.prices.index.min().date(),
            self.prices.index.max().date(),
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _build_inputs(self) -> None:
        """Compute (and cache) expected returns ``mu`` and covariance ``S``.

        Uses PyPortfolioOpt's vectorised implementations:
        - ``mean_historical_return`` -- annualised compound growth rate
        - ``CovarianceShrinkage.ledoit_wolf`` -- shrunk covariance (annualised)

        Called automatically by optimisation methods on first invocation.
        """
        if self._mu is not None and self._S is not None:
            return  # Already computed

        log.info("Computing expected returns and covariance matrix ...")

        self._mu = expected_returns.mean_historical_return(
            self.prices,
            returns_data=False,      # prices, not pre-computed returns
            compounding=True,        # use log-return CAGR
            frequency=self.frequency,
        )

        self._S = risk_models.CovarianceShrinkage(
            self.prices,
            returns_data=False,
            frequency=self.frequency,
        ).ledoit_wolf()

        log.info("Inputs ready -- mu shape: %s, S shape: %s", self._mu.shape, self._S.shape)

    @property
    def mu(self) -> pd.Series:
        """Annualised mean historical returns (Series, ticker-indexed)."""
        self._build_inputs()
        return self._mu  # type: ignore[return-value]

    @property
    def S(self) -> pd.DataFrame:
        """Annualised Ledoit-Wolf covariance matrix (DataFrame)."""
        self._build_inputs()
        return self._S  # type: ignore[return-value]

    def _make_ef(self) -> EfficientFrontier:
        """Instantiate a fresh ``EfficientFrontier`` object.

        A new instance is required for each optimisation call because
        PyPortfolioOpt modifies the solver state in place.
        """
        return EfficientFrontier(
            self.mu,
            self.S,
            weight_bounds=(0, 1),   # long-only constraint
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_equal_weights(self) -> Dict[str, float]:
        """Return the 1/N equal-weight baseline portfolio.

        Assigns each of the N surviving assets an identical allocation of
        ``1 / N``, rounded to 4 decimal places.  Serves as the benchmark
        against which Sharpe-optimal and min-vol portfolios are evaluated.

        Returns
        -------
        dict[str, float]
            Mapping of ticker -> weight.  All weights sum to 1.0.

        Examples
        --------
        >>> opt.get_equal_weights()
        {'RELIANCE': 0.02, 'TCS': 0.02, ...}  # for N = 50
        """
        weight = round(1.0 / self.n_assets, 4)
        weights = {ticker: weight for ticker in self.tickers}

        log.info(
            "Equal-weight portfolio: %d assets @ %.4f each (sum = %.4f)",
            self.n_assets,
            weight,
            sum(weights.values()),
        )
        return weights

    def optimize_max_sharpe(self) -> Dict[str, float]:
        """Maximise the Sharpe ratio using the Efficient Frontier.

        Uses the SLSQP solver (scipy default) to find the portfolio on the
        efficient frontier that maximises the risk-adjusted return:

            Sharpe = (mu_p - r_f) / sigma_p

        where ``r_f`` is the ``risk_free_rate`` supplied at construction.

        The raw continuous weights are then passed through
        ``clean_weights(cutoff=1e-4, rounding=4)`` which:
        - Rounds weights to 4 decimal places.
        - Drops any position whose absolute weight is below 0.01% (dust).

        Returns
        -------
        dict[str, float]
            Cleaned mapping of ticker -> weight for the max-Sharpe portfolio.
            Only non-zero positions are included.

        Notes
        -----
        If the solver diverges (e.g., due to near-singular covariance), an
        ``OptimizationError`` is raised for the caller to handle explicitly.
        """
        try:
            ef = self._make_ef()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)

            raw_weights = ef.clean_weights(cutoff=1e-4, rounding=4)

            # Filter out zero-weight positions
            cleaned: Dict[str, float] = {
                k: v for k, v in raw_weights.items() if v > 0
            }

            perf = ef.portfolio_performance(
                verbose=False,
                risk_free_rate=self.risk_free_rate,
            )
            log.info(
                "Max-Sharpe: %.4f expected return | %.4f volatility | %.4f Sharpe",
                perf[0],
                perf[1],
                perf[2],
            )
            log.info("Non-zero positions: %d / %d", len(cleaned), self.n_assets)

            return cleaned

        except Exception as exc:  # noqa: BLE001
            raise OptimizationError(
                f"Max-Sharpe optimization failed: {exc}"
            ) from exc

    def optimize_min_volatility(self) -> Dict[str, float]:
        """Minimise portfolio volatility using the Efficient Frontier.

        Finds the Global Minimum Variance (GMV) portfolio -- the leftmost
        point on the efficient frontier.  This portfolio minimises:

            sigma_p = sqrt(w^T @ Sigma @ w)

        subject to the long-only constraint (all weights in [0, 1]).

        Returns
        -------
        dict[str, float]
            Cleaned mapping of ticker -> weight for the min-volatility portfolio.
            Only non-zero positions are included.

        Notes
        -----
        The GMV portfolio is often highly concentrated in low-beta, defensive
        sectors (FMCG, IT) for the NIFTY 50 universe.  Inspect the output
        alongside sector exposure before deploying in production.
        """
        try:
            ef = self._make_ef()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.min_volatility()

            raw_weights = ef.clean_weights(cutoff=1e-4, rounding=4)

            cleaned: Dict[str, float] = {
                k: v for k, v in raw_weights.items() if v > 0
            }

            perf = ef.portfolio_performance(
                verbose=False,
                risk_free_rate=self.risk_free_rate,
            )
            log.info(
                "Min-Vol: %.4f expected return | %.4f volatility | %.4f Sharpe",
                perf[0],
                perf[1],
                perf[2],
            )
            log.info("Non-zero positions: %d / %d", len(cleaned), self.n_assets)

            return cleaned

        except Exception as exc:  # noqa: BLE001
            raise OptimizationError(
                f"Minimum-volatility optimization failed: {exc}"
            ) from exc

    def rolling_backtest(
        self,
        lookback: int = 252,
        rebalance_every: int = 21,
    ) -> tuple[pd.DataFrame, dict[str, Dict[str, float]], int]:
        """Backtest optimized allocations using only prices before each rebalance.

        At each rebalance date after the initial lookback, weights are
        estimated using the preceding ``lookback`` prices. Those weights are
        applied only to subsequent close-to-close returns until the next
        rebalance. Failed optimization windows use equal weights for that
        strategy and are counted for caller notification.
        """
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer.")
        if rebalance_every <= 0:
            raise ValueError("rebalance_every must be a positive integer.")
        if len(self.prices) < 2:
            raise ValueError("At least two price rows are required for backtesting.")

        lookback = min(lookback, len(self.prices) - 1)
        ticker_count = len(self.tickers)
        equal_weights = {
            ticker: 1.0 / ticker_count
            for ticker in self.tickers
        }
        returns: list[dict[str, float]] = []
        latest_weights: dict[str, Dict[str, float]] = {}
        failures = 0

        def normalize(weights: Dict[str, float]) -> Dict[str, float]:
            aligned = {ticker: float(weights.get(ticker, 0.0)) for ticker in self.tickers}
            total = sum(aligned.values())
            if total <= 0:
                raise OptimizationError("Optimizer returned no positive portfolio weights.")
            return {ticker: weight / total for ticker, weight in aligned.items()}

        for rebalance_position in range(
            lookback,
            len(self.prices),
            rebalance_every,
        ):
            window_start = rebalance_position - lookback
            history = self.prices.iloc[window_start:rebalance_position]
            window_optimizer = PortfolioOptimizer(
                history,
                risk_free_rate=self.risk_free_rate,
                frequency=self.frequency,
            )
            try:
                max_sharpe_weights = normalize(
                    window_optimizer.optimize_max_sharpe()
                )
            except OptimizationError:
                max_sharpe_weights = equal_weights.copy()
                failures += 1
            try:
                min_volatility_weights = normalize(
                    window_optimizer.optimize_min_volatility()
                )
            except OptimizationError:
                min_volatility_weights = equal_weights.copy()
                failures += 1

            latest_weights = {
                "Max Sharpe": max_sharpe_weights,
                "Min Volatility": min_volatility_weights,
                "Equal Weight": equal_weights.copy(),
            }
            period_end = min(
                rebalance_position + rebalance_every,
                len(self.prices),
            )
            for position in range(rebalance_position, period_end):
                observed_return = (
                    self.prices.iloc[position]
                    .div(self.prices.iloc[position - 1])
                    .sub(1.0)
                )
                returns.append(
                    {
                        strategy: float(
                            observed_return.mul(pd.Series(weights)).sum()
                        )
                        for strategy, weights in latest_weights.items()
                    }
                )

        result = pd.DataFrame(
            returns,
            index=self.prices.index[lookback:],
            columns=["Max Sharpe", "Min Volatility", "Equal Weight"],
        )
        return result, latest_weights, failures


# =========================================================================== #
# __main__ -- end-to-end smoke test
# =========================================================================== #
if __name__ == "__main__":
    import sys
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(BASE_DIR / "src"))
    from data_loader import LocalDataLoader  # noqa: E402

    print("\n" + "=" * 65)
    print("  QUANTITATIVE PORTFOLIO RISK ENGINE -- MPT Optimisation Run")
    print("=" * 65)

    # 1. Load all portfolio assets from 'data/raw'
    loader = LocalDataLoader()
    prices = loader.load_all_portfolio_assets("data/raw")

    # 2. Instantiate PortfolioOptimizer
    opt = PortfolioOptimizer(prices)

    # 3. Print the requested key portfolio metrics
    print("\n" + "-" * 65)
    print("  PORTFOLIO MATRIX SUMMARY")
    print("-" * 65)
    print(f"  Total Surviving Assets : {opt.n_assets}")
    print(f"  Start Date             : {prices.index.min().date()}")
    print(f"  End Date               : {prices.index.max().date()}")
    print(f"  Total Trading Days     : {len(prices)}")

    # 4. Equal Weights baseline
    ew = opt.get_equal_weights()
    print("\n" + "-" * 65)
    print(f"  [EW]  Equal-Weight Baseline (1/{opt.n_assets} each)")
    print(f"        Sum of weights: {sum(ew.values()):.4f}")

    # 5. Max Sharpe Weights
    msr = opt.optimize_max_sharpe()
    print("\n" + "-" * 65)
    print("  [MSR] Final Max Sharpe Weights")
    print("-" * 65)
    print(f"  Allocated Tickers: {len(msr)} / {opt.n_assets}")
    for ticker, w in sorted(msr.items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 100)
        print(f"    {ticker:<12} : {w:>7.4f} ({w*100:5.2f}%)  {bar}")

    # 6. Min Volatility Weights
    mv = opt.optimize_min_volatility()
    print("\n" + "-" * 65)
    print("  [MV]  Min-Volatility Weights")
    print("-" * 65)
    print(f"  Allocated Tickers: {len(mv)} / {opt.n_assets}")
    for ticker, w in sorted(mv.items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 100)
        print(f"    {ticker:<12} : {w:>7.4f} ({w*100:5.2f}%)  {bar}")

    print("\n" + "=" * 65)
    print("  Run complete.")
    print("=" * 65 + "\n")
