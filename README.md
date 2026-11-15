# Quantitative Portfolio Risk Engine

A Streamlit application for historical portfolio simulation, classical portfolio optimization, market and asset diagnostics, and forward-volatility analysis. The project combines local NSE equity price files with pandas, NumPy, Plotly, PyPortfolioOpt, and LightGBM.

> **Scope:** This is an educational analytics project. Its historical results are not live investment performance, and its forecasts and allocations are not investment advice.

## Contents

- [What the application does](#what-the-application-does)
- [Application pages](#application-pages)
- [Data and coverage](#data-and-coverage)
- [Methods and calculations](#methods-and-calculations)
- [Run the application](#run-the-application)
- [Verification](#verification)
- [Limitations](#limitations)
- [Project layout](#project-layout)

## What the application does

The app has four interactive pages linked from `app/app.py`:

| Page | Purpose |
|---|---|
| Historical Simulator | Record dated, whole-share buys and sells and inspect the resulting portfolio history. |
| Classical Quant Engine | Compare equal-weight, minimum-volatility, and maximum-Sharpe portfolios; inspect allocations, risks, and an out-of-sample rolling backtest. |
| EDA & Statistics | Explore asset returns, correlations, sector comparisons, distributions, and rolling risk. |
| ML Risk Engine | Estimate forward market volatility, review a share-weighted portfolio, and calculate integer-share rebalance recommendations. |

The central analytical question is how historical asset behavior can inform portfolio risk and allocation, and whether trailing market features contain information about subsequent realized volatility. The application answers this through descriptive analysis, historical simulation, portfolio methods, and a volatility-regression workflow; it is not a crash-prediction system.

## Application pages

### 1. Historical Simulator

The transaction ledger records ticker, trade date, action, and cash amount. The simulator snaps a requested date to the next available trading date, looks up that close, and converts the amount to whole shares using `floor(amount / price)`. It rejects sells that exceed the shares held as of the execution date. Duplicate trades are kept as separate ledger events and combined when calculating holdings.

The page reports portfolio value, net capital (buys less sale proceeds), profit/loss, return, and portfolio composition over the recorded period. It also supports sector filtering and transaction history views.

### 2. Classical Quant Engine

The settings panel selects the asset universe and annual risk-free rate. The page offers 1-week through 5-year windows, Max, and custom dates. Selected assets without usable prices in the chosen interval are excluded and identified. Where available, up to 252 earlier trading sessions are used to initialize the walk-forward optimizer; displayed backtest returns are restricted to the selected interval.

The engine compares:

- **Equal weight:** one equal allocation per active asset.
- **Minimum volatility:** a long-only PyPortfolioOpt portfolio minimizing estimated variance.
- **Maximum Sharpe:** a long-only PyPortfolioOpt portfolio maximizing estimated excess return per unit of volatility, with L2 regularization (`gamma=0.1`).

The estimation uses annualized historical returns and Ledoit–Wolf shrunk covariance. In the rolling backtest, weights are estimated from prior observations and held between 21-session rebalances. The page presents the resulting equity curves, CAGR, realized Sharpe, drawdown, historical daily VaR, rolling volatility, pairwise correlations, allocations, and Monte Carlo risk/return portfolios. With one active asset, the allocation is necessarily 100% in that asset; multi-asset optimization is not meaningful in that case.

### 3. EDA & Statistics

The EDA page supports sector filtering and ticker selection. It provides cross-asset return correlations, sector-level risk/return views, and asset-level descriptive statistics and distribution plots. Price values that are nonpositive are excluded from return calculations. Annualized means and standard deviations use the 252-trading-day convention; sample statistics and rolling windows are shown over the observations available for the selected data.

### 4. ML Risk Engine

The ML page uses the locally loaded aligned asset matrix to engineer market-level features and train a LightGBM regressor. Its target is annualized realized volatility over the next 30 trading sessions. The page applies the model to the selected historical base date, scales the 30-session forecast to a chosen horizon using square-root-of-time scaling, and compares projected risk with a training-sample percentile threshold. This is a volatility forecast and risk summary; it does **not** predict market direction, time crashes, or identify crisis events.

Users enter current ticker/share holdings and select a defensive minimum-volatility or aggressive maximum-Sharpe objective. The optimizer runs on the entered holdings. It uses long-only MPT and a 6.5% annual risk-free rate (the optimizer default). Target weights are converted to whole-share targets using the portfolio value and prices on the effective date; recommendations are rounded to integer shares. The page also displays current holdings, sector concentration, correlation, historical portfolio performance, and risk summaries.

The page’s current LightGBM configuration is a regression model with 100 estimators, learning rate 0.05, maximum depth 5, 15 leaves, minimum child samples 30, row/column subsampling 0.85, L1 regularization 0.05, L2 regularization 0.10, and random seed 42.

#### Separate model-selection training script

`scripts/train_risk_model_standalone.py` is a separate experimentation pipeline, not the model served by the ML page. It engineers a 20-session forward-volatility target, uses a chronological train/validation/test split with target-horizon gaps, compares a naïve volatility baseline with LightGBM, Random Forest, and XGBoost, and tunes candidates with gapped `TimeSeriesSplit` cross-validation. It exports the selected model and diagnostic charts. The Streamlit ML page instead trains its own fixed-configuration LightGBM model for a 30-session target and does not load that exported artifact. Keep results from these two workflows separate when describing or presenting model performance.

## Data and coverage

### Equity universe

`src/data_loader.py` reads per-ticker CSV files from `companies/` or `data/raw/`, depending on the page. It parses dates and close prices, sorts observations, removes duplicate dates, merges tickers on the date index, filters assets above the configured missing-data threshold, and aligns usable histories. The current local dataset contains 52 CSV files; the established aligned matrix contains **42 assets and 3,994 dates**, from **23 March 2005 through 30 April 2021**. The exact number used by a page can depend on its data directory, filtering, selected universe, date window, and missing values.

### Other local data

Root-level NIFTY 500 and India VIX CSV files extend through 20 March 2026 and support standalone macro analysis scripts. They do not extend the per-company price history used by the four interactive portfolio pages.

### Data caveats

- The per-company price history is static and ends on 30 April 2021; the application does not fetch live prices.
- Company close series are not adjusted consistently for splits, bonuses, dividends, or other corporate actions. Corporate-action jumps can create artificial returns and distort volatility, covariance, drawdown, and optimization outputs.
- The retained universe reflects surviving/data-valid files and is not a point-in-time historical index membership dataset. Survivorship and membership bias are present.
- Different pages can apply different alignment and filtering steps. Check the date range, active assets, and warnings displayed by the page being used.

## Methods and calculations

### Portfolio optimization

`src/optimization.py` uses PyPortfolioOpt with long-only bounds (`0 ≤ wᵢ ≤ 1`, `Σwᵢ = 1`). Expected returns are estimated from historical prices and annualized at 252 sessions per year. Covariance is estimated with Ledoit–Wolf shrinkage and annualized on the same basis. The maximum-Sharpe objective includes an L2 penalty; the minimum-volatility objective minimizes portfolio variance. Optimized weights may be sparse after the library’s cleaning step, so selected assets can have zero allocation.

### Classical engine metrics

- **NAV:** starts at ₹100 and compounds realized daily portfolio returns.
- **CAGR:** annualizes the compounded out-of-sample return using the number of backtested trading sessions.
- **Realized Sharpe:** annualized mean daily return less the annual risk-free rate, divided by annualized sample volatility.
- **Maximum drawdown:** largest peak-to-trough decline from the NAV high-water mark, including the starting capital.
- **Historical 95% daily VaR:** based on the 5th percentile of realized daily returns and scaled to the displayed ₹10 lakh notional.
- **Monte Carlo portfolios:** random long-only allocations sampled from a Dirichlet distribution and evaluated using the estimated return and covariance inputs. The scatter is a simulation cloud, not a separately solved continuous efficient-frontier curve.

### ML features and target

The feature pipeline in `src/ml_risk_engine_test.py` creates nine trailing market-level features: daily equal-weight market return, 21-session momentum, price-to-63-session moving average, 30-session market volatility, 21- and 63-session market return volatility, mean and dispersion of 30-session asset volatilities, and average 30-session pairwise correlation. The target is forward 30-session annualized realized volatility. Feature/target alignment and the page’s training cutoff keep future target information out of the training sample.

The standalone evaluation uses a chronological split and a 30-session gap between train and test observations. Its 75th-percentile training-target threshold labels observations as normal or upper-volatility for evaluation; this label is a summary of forecast volatility, not a crash or crisis prediction. On the checked local test sample, the threshold labels had 87.73% overall accuracy, while recall for the upper-volatility label was 41.32% (the test set contains far more normal than upper-volatility observations). The regression forecast itself had MAE 0.052679 and RMSE 0.092757 in annualized-volatility units. Read the overall accuracy alongside class-specific recall: it does not mean the model detects 87.73% of high-volatility periods. These metrics are from one historical, corporate-action-unadjusted sample and are not guarantees of future performance.

## Run the application

Requirements are pinned in `requirements.txt` (Python 3.10 or newer is recommended).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard is served locally, normally at `http://localhost:8501`.

## Verification

There is no dedicated pytest suite in the repository. The following scripts exercise selected data, statistics, and model pipelines; some scripts write generated charts or artifacts, so inspect their source before running if preserving the workspace is important.

```powershell
python src/data_loader.py
python src/verify_eda.py
python src/ml_risk_engine_test.py
```

The Streamlit pages are interactive applications and should also be reviewed in the browser for navigation, input handling, charts, and the selected data range. Syntax checks alone do not validate visual layout or user interactions.

## Limitations

- Historical results are sensitive to the static sample, survivorship bias, and unadjusted corporate actions.
- Backtests omit brokerage, exchange fees, taxes, bid/ask spread, market impact, and slippage. Integer-share recommendations also ignore lot-size rules and may not be executable at the displayed close.
- The ML forecast concerns volatility magnitude. It does not forecast returns, prices, direction, or crash timing. Horizon scaling assumes square-root-of-time behavior and is an approximation.
- The optimizer’s expected returns and covariance are estimates from historical data. They can be unstable, especially for short samples or highly correlated assets.
- Risk-free rates are model inputs, not continuously sourced market observations. The Classical Quant Engine exposes a user setting; the ML rebalancing optimizer currently uses its 6.5% default.
- Outputs are analytical demonstrations, not personalized financial advice.

## Project layout

```text
app/
  app.py                         Streamlit landing page and navigation
  pages/
    1_Historical_Simulator.py    Trade ledger and historical portfolio simulation
    2_Quant_Engine.py            Classical MPT, backtest, metrics, and plots
    3_EDA_and_Stats.py           Return, risk, correlation, and distribution analysis
    4_ML_Risk_Engine.py           Volatility forecast and share rebalance analysis
  assets/                        Application images
src/
  data_loader.py                 Local CSV ingestion and price alignment
  optimization.py                PyPortfolioOpt portfolio methods/backtest
  ml_risk_engine_test.py         ML features, target, split, and evaluation
  features.py, ml_models.py      Additional feature/model utilities
  sector_mapping.py, theme.py    Shared page helpers
companies/                       Per-company historical CSV files
data/raw/                         Alternate raw ticker data directory
models/, charts/                  Existing generated model/chart artifacts
scripts/                          Standalone training and visualization scripts
requirements.txt                  Pinned Python dependencies
```
