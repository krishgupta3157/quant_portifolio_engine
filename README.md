# Quantitative Portfolio Risk Engine

A quantitative portfolio analytics, risk modeling, and asset allocation platform built with Python, PyPortfolioOpt, LightGBM, and Streamlit. The application models a 42-asset universe from India's NIFTY equity index, providing classical Mean-Variance optimization (Modern Portfolio Theory), statistical exploratory data analysis, interactive historical transaction simulation, and machine learning volatility forecasting.

---

## 1. Overview

The platform translates quantitative portfolio theory into interactive analytical workflows. It allows analysts and investors to explore historical asset dynamics, evaluate risk-return trade-offs, model mathematically regularized asset allocations, and forecast short-term market volatility regimes.

The application is structured into four primary modules accessible from a centralized landing page (`app/app.py`):

1. **Historical Simulator** (`pages/1_Historical_Simulator.py`): Interactive ledger-based portfolio construction with discrete whole-share trade execution.
2. **Classical Quant Engine** (`pages/2_Quant_Engine.py`): Modern Portfolio Theory (MPT) mean-variance optimization, Ledoit-Wolf covariance shrinkage, and walk-forward rolling backtesting.
3. **Exploratory Data Analysis & Statistics** (`pages/3_EDA_and_Stats.py`): Cross-sectional and single-asset statistical distributions, correlation structures, and tail-risk diagnostics.
4. **ML Dynamic Risk Engine** (`pages/4_ML_Risk_Engine.py`): Gradient-boosted volatility forecasting, regime classification, and target share rebalancing.

---

## 2. Implemented Modules

### Historical Simulator
- **Transaction Ledger**: Enables users to record sequential buy and sell transactions on specific historical trading dates.
- **Whole-Share Execution**: Calculates positions using whole shares (`amount // price`) based on actual historical daily closing prices.
- **Portfolio Tracking**: Evaluates dynamic portfolio market value, net invested capital, absolute profit/loss, cumulative return percentage, and current asset allocation breakdown.
- **Data Integrity**: Validates asset availability against historical trade dates and prevents sells that exceed current holdings.

### Classical Quant Engine
- **Mean-Variance Optimization**: Implements Markowitz portfolio optimization via `PyPortfolioOpt`. Solves for **Maximum Sharpe Ratio** and **Minimum Volatility** allocations.
- **Covariance Shrinkage**: Uses the Ledoit-Wolf shrinkage estimator to construct well-conditioned covariance matrices from historical daily returns.
- **Regularization & Bounds**: Enforces long-only constraints ($\sum w_i = 1, w_i \ge 0$) and incorporates L2 weight regularization ($\gamma = 0.1$) to prevent over-concentration.
- **Risk Metrics**: Calculates Annualized Return (compounded log CAGR), Annualized Volatility ($\times \sqrt{252}$), Value at Risk (Parametric and Historical 95% / 99% VaR), and Maximum Drawdown.
- **Efficient Frontier & Monte Carlo**: Generates the analytical efficient frontier curve alongside random Monte Carlo portfolio scatter distributions.
- **Rolling Walk-Forward Backtest**: Provides an out-of-sample rolling-origin backtest routine (`lookback = 252` trading days, rebalancing step = 21 trading days).

### Exploratory Data Analysis & Statistics
- **Macro Market Structure**: Interactive pairwise return correlation heatmap with dynamic ticker and sector filtering, plus sector-aggregated risk-return scatter plots.
- **Asset-Level Diagnostics**: Annualized mean return, annualized standard deviation, sample daily variance, skewness, and excess kurtosis.
- **Distribution Visualizations**: Probability density histograms with theoretical normal distribution overlays, 30-day and 90-day rolling annualized volatility trends, and box plots comparing individual asset dispersions against an equal-weighted market basket.

### ML Dynamic Risk Engine
- **Predictive Target**: Trains a gradient-boosted decision tree (`LightGBM`) regressor to predict 30-trading-day forward annualized market volatility ($\sigma_{t+30}$).
- **Feature Pipeline**: Constructs 9 market-level trailing features from the equal-weighted asset universe, including moving average ratios, trailing return volatilities, cross-asset dispersion, and average pairwise correlation.
- **Leakage Prevention**: Strictly offsets training features prior to the prediction timestamp and shifts forward targets to eliminate lookahead bias.
- **Regime Identification**: Flags elevated market risk when forecasted volatility exceeds the historical 75th-percentile quantile threshold of the training distribution.
- **Actionable Rebalancing**: Converts target optimal weights into discrete whole-share buy, sell, or hold orders based on current user holdings and asset prices.

---

## 3. Data Ingestion & Universe Coverage

### Data Source & Schema
- **Raw Files**: Located in `companies/` (mirrored in `data/raw/`). The data consists of raw National Stock Exchange of India (NSE) daily historical CSV exports.
- **File Schema**: Each company CSV contains 15 fields: `Date`, `Symbol`, `Series`, `Prev Close`, `Open`, `High`, `Low`, `Last`, `Close`, `VWAP`, `Volume`, `Turnover`, `Trades`, `Deliverable Volume`, and `%Deliverble`.

### Active Asset Universe & Date Range
The data ingestion pipeline (`LocalDataLoader.load_all_portfolio_assets` in `src/data_loader.py`):
1. Discovers 52 raw CSV files in `companies/`.
2. Skips files with insufficient observations (e.g., `INFRATEL.csv`, which has only 1 data row).
3. Evaluates missing-data thresholds and drops tickers with greater than 30% missing history (`ADANIPORTS`, `BAJAJ-AUTO`, `BAJAJFINSV`, `COALINDIA`, `NESTLEIND`, `POWERGRID`, `TECHM`).
4. Aligns prices to the first simultaneous trading date across all surviving assets and forward-fills isolated non-trading gaps.

- **Final Aligned Price Matrix**: **42 assets** across **3,994 trading days**.
- **Historical Coverage**: **March 23, 2005 through April 30, 2021**.

### Data Limitations & Known Gaps
- **Unadjusted Corporate Actions**: The raw NSE dataset contains unadjusted close prices. When a stock split or bonus issuance occurred, unadjusted prices reflect large single-day drops (e.g., Eicher Motors on 2020-08-24 dropped ~90% due to a 1:10 stock split; Titan on 2011-06-23 dropped ~94.7% due to a 1:10 split and 1:1 bonus; similar unadjusted splits affect ITC, JSW Steel, Asian Paints, Bajaj Finance, and SBI). These discontinuities introduce large artificial negative return spikes into historical volatility and covariance estimates for those specific periods.
- **Static Historical Window**: Per-company historical prices end on April 30, 2021. The platform does not ingest live, real-time, or post-2021 company market data. (The root-level `NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv` and `INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv` files contain macro index data through March 2026, but are only utilized in standalone macro analysis pipelines).
- **Survivorship Bias**: The 42-asset universe comprises companies active throughout the 2005–2021 window and does not account for historical index deletions, acquisitions, or corporate insolvencies.

---

## 4. Setup & Execution

### Prerequisites
- Python 3.10 or higher.
- Standard virtual environment (`venv`).

### Installation
From the `quant_portifolio_engine` root directory:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### Running the Application
Launch the Streamlit web dashboard:

```bash
streamlit run app/app.py
```

The application will start locally (default: `http://localhost:8501`).

### Standalone Verification Scripts
The repository contains standalone execution scripts that verify underlying pipelines:

```bash
# Verify data loading and matrix alignment
python src/data_loader.py

# Verify statistical calculations and EDA metrics
python src/verify_eda.py

# Run standalone ML model feature engineering and evaluation
python src/ml_risk_engine_test.py
```

---

## 5. Methodology Overview

### Descriptive Analysis vs. Forecasting
- **Descriptive Historical Analysis**: Metrics such as sample mean return, rolling volatility, historical drawdown, and distribution moments (skewness, excess kurtosis) quantify observed behavior over the recorded historical sample.
- **Predictive Modeling**: The machine learning engine models future volatility expectations. Point forecasts represent conditional expected 30-day volatility based on trailing market dispersion, not guaranteed future prices.

### Optimization Objectives & Constraints
The `PortfolioOptimizer` (`src/optimization.py`) provides:
1. **Max Sharpe Ratio**: $\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$, where $\mu$ is annualized historical mean log return, $\Sigma$ is the Ledoit-Wolf shrunk covariance matrix, and $R_f$ is the user-configured risk-free rate.
2. **Minimum Volatility**: $\min_w w^T \Sigma w$.
3. **Constraints**: Full investment ($\sum_{i=1}^n w_i = 1$) and long-only positions ($0 \le w_i \le 1$). An L2 penalty ($\gamma = 0.1$) is added to the objective to penalize extreme corner solutions and encourage diversification.
- *Note*: The optimizer does not directly minimize maximum drawdown via dynamic programming; drawdown metrics in the dashboard are computed ex-post on the resulting historical equity curve.

### Machine Learning Target & Evaluation
- **Target Variable**: Annualized realized volatility over the subsequent 30 trading days:
  $$\sigma_{t+30} = \text{std}(r_{t+1:t+30}) \times \sqrt{252}$$
- **Algorithm**: LightGBM regressor (`LGBMRegressor`, objective: `regression`, learning rate: `0.05`, estimators: `100`).
- **Evaluation**: The standalone evaluation script evaluates test-period Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$. Crisis regimes are evaluated post-hoc by comparing predicted volatility against the historical 75th-percentile threshold.

---

## 6. Limitations & Responsible Interpretation

1. **In-Sample Optimization Display**: Headline optimal weights and efficient frontier curves displayed on the main Quant Engine page are estimated over the selected historical slice (in-sample). While a walk-forward rolling-origin backtest routine is implemented in code (`PortfolioOptimizer.rolling_backtest`), static optimal allocations reflect historical parameter fitting rather than out-of-sample forward performance.
2. **Volatility Forecasting vs. Crash Timing**: The ML model forecasts continuous volatility magnitude. It does not predict market direction, trade timing, or individual security price shocks. High forecasted volatility indicates an environment of elevated historical dispersion, not a guaranteed market decline.
3. **Execution Frictions Omitted**: The historical simulator and rebalancing calculations do not incorporate brokerage commissions, exchange transaction charges, securities transaction tax (STT), market impact, bid-ask spread slippage, or dividend reinvestment. Real-world execution will deviate from simulated performance.
4. **No Financial or Investment Advice**: This software is an educational and quantitative engineering demonstration. It is not financial advice, investment advisory, or a recommendation to purchase or sell any security.
