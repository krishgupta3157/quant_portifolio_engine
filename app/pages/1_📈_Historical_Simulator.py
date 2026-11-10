"""Historical portfolio simulator page.

This page is intentionally isolated from ``app/app.py``.  It uses the same
local portfolio price matrix and lets a user build a dated trade ledger,
then values the resulting holdings against every historical trading day.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------------------------------- #
# Page setup and imports
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Historical Simulator · Quant Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.theme import apply_global_theme
from src.data_loader import LocalDataLoader
from src.sector_mapping import get_sector
apply_global_theme()


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg-app: #E8E3D7;
  --bg-card: #DCD5C6;
  --bg-card-sub: #D4CCBC;
  --border: #C8C0AE;
  --olive: #222B1C;
  --olive-mid: #3B4536;
  --text: #222B1C;
  --muted: #8B8578;
  --cream: #F5F0E8;
  --positive: #5C6B52;
  --negative: #8B4B4B;
  --amber: #8B6B2B;
}

*, *::before, *::after { box-sizing: border-box; }

.stApp {
  background: var(--bg-app) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, div[data-testid="stToolbar"],
div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {
  display: none !important;
}

header[data-testid="stHeader"] {
  background: rgba(232, 227, 215, 0.95) !important;
  backdrop-filter: blur(10px) !important;
  border-bottom: 1px solid var(--border) !important;
}

section.main > div.block-container {
  max-width: 100% !important;
  padding: 2.2rem 3.2rem 4rem !important;
}

[data-testid="stHorizontalBlock"] {
  gap: 1.25rem !important;
}

.sim-hero {
  padding: 0.8rem 0 1.4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}

.sim-eyebrow {
  color: var(--olive-mid);
  font-family: 'Inter', sans-serif;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
}

.sim-title {
  font-family: 'Playfair Display', Georgia, serif;
  color: var(--text);
  font-size: clamp(1.8rem, 3.5vw, 2.8rem);
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.1;
  margin: 0 0 0.45rem;
}

.sim-subtitle {
  color: var(--muted);
  font-family: 'Inter', sans-serif;
  font-size: 0.82rem;
  line-height: 1.65;
  margin: 0;
  max-width: 700px;
}

.section-title {
  color: var(--text);
  font-family: 'Playfair Display', serif;
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 1.7rem 0 0.8rem;
}

.section-title::before {
  background: var(--olive-mid);
  border-radius: 2px;
  content: "";
  display: inline-block;
  height: 17px;
  margin-right: 0.55rem;
  vertical-align: -3px;
  width: 3px;
}

.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  display: flex;
  flex: 1 1 0;
  flex-direction: column;
  flex-wrap: wrap;
  min-width: 0;
  overflow: visible;
  overflow-wrap: anywhere;
  padding: 1.4rem;
  transition: border-color .2s ease, transform .2s ease;
}

.kpi-card:hover {
  border-color: var(--olive);
  transform: translateY(-2px);
}

.kpi-label {
  color: var(--muted);
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  margin-bottom: 0.6rem;
  text-transform: uppercase;
}

.kpi-value {
  font-family: 'Playfair Display', serif;
  font-size: clamp(0.9rem, 1.45vw, 1.3rem);
  font-weight: 700;
  color: var(--text);
  line-height: 1.25;
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.kpi-note {
  color: var(--muted);
  font-family: 'Inter', sans-serif;
  font-size: 0.63rem;
  margin-top: 0.35rem;
  max-width: 100%;
  overflow-wrap: anywhere;
}

[data-testid="stHorizontalBlock"]:has(.kpi-card) {
  align-items: stretch;
  flex-wrap: wrap;
}

[data-testid="stMetric"] {
  min-width: 0 !important;
  overflow: visible !important;
  overflow-wrap: anywhere !important;
  padding: 1rem 0.5rem !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Playfair Display', serif !important;
  font-size: clamp(0.95rem, 1.8vw, 1.5rem) !important;
  color: var(--text) !important;
  max-width: 100% !important;
  overflow-wrap: anywhere !important;
  white-space: normal !important;
  word-break: break-word !important;
}

[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

.tone-blue { color: var(--olive); }
.tone-green { color: var(--positive); }
.tone-red { color: var(--negative); }
.tone-amber { color: var(--amber); }

.form-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.1rem 1.2rem 0.25rem;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stDateInput"] input,
div[data-testid="stNumberInput"] input {
  background: var(--bg-card-sub) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

div[data-testid="stForm"] {
  border: 0 !important;
  padding: 0 !important;
}

[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button {
  background: #222B1C !important;
  border: 1px solid #222B1C !important;
  border-radius: 24px !important;
  color: #FFFFFF !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.74rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}

[data-testid="stButton"] button *,
[data-testid="stFormSubmitButton"] button * {
  color: #FFFFFF !important;
  font-weight: 700 !important;
}

[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
  background: #3B4536 !important;
  border-color: #3B4536 !important;
  box-shadow: 0 4px 14px rgba(34, 43, 28, .22) !important;
  color: #FFFFFF !important;
  transform: translateY(-1px);
}

[data-testid="stButton"] button:hover *,
[data-testid="stFormSubmitButton"] button:hover * {
  color: #FFFFFF !important;
}

[data-testid="stPlotlyChart"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 0.5rem !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

.empty-state {
  background: var(--bg-card-sub);
  border: 1px dashed var(--border);
  border-radius: 12px;
  color: var(--muted);
  font-family: 'Inter', sans-serif;
  font-size: 0.8rem;
  padding: 1.6rem;
  text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data and state
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading historical NIFTY asset prices...")
def load_market_data() -> pd.DataFrame:
    """Load and cache the cleaned, aligned portfolio price matrix."""
    data_dir = BASE_DIR / "data" / "raw"
    if not data_dir.exists():
        data_dir = BASE_DIR / "companies"

    prices = LocalDataLoader().load_all_portfolio_assets(
        str(data_dir),
        missing_threshold=0.30,
    )
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    return prices


try:
    master_prices = load_market_data()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Unable to load the historical price matrix: {exc}")
    st.stop()

if master_prices.empty or master_prices.columns.empty:
    st.error("The historical price matrix is empty. Check the CSV files in `companies/`.")
    st.stop()

all_tickers = sorted(master_prices.columns.tolist())
available_sectors = sorted({get_sector(ticker) for ticker in all_tickers})
with st.sidebar:
    st.markdown("### Industry Filters")
    selected_sectors = st.multiselect(
        "Select Industry/Sector",
        options=available_sectors,
        key="simulator_sector_filter",
        help="Filter the trade-entry ticker dropdown by industry.",
    )
filtered_tickers = [
    ticker for ticker in all_tickers
    if not selected_sectors or get_sector(ticker) in selected_sectors
]
min_date = master_prices.index.min().date()
max_date = master_prices.index.max().date()

if "trade_ledger" not in st.session_state:
    st.session_state.trade_ledger = []


# --------------------------------------------------------------------------- #
# Ledger and simulation helpers
# --------------------------------------------------------------------------- #
LEDGER_COLUMNS = ["Date", "Ticker", "Action", "Shares", "Price", "Total Value"]


def ledger_dataframe() -> pd.DataFrame:
    """Return the session ledger with stable columns and dtypes."""
    ledger = pd.DataFrame(st.session_state.trade_ledger, columns=LEDGER_COLUMNS)
    if ledger.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    ledger["Date"] = pd.to_datetime(ledger["Date"]).dt.normalize()
    ledger["Ticker"] = ledger["Ticker"].astype(str)
    ledger["Action"] = ledger["Action"].astype(str)
    ledger["Shares"] = pd.to_numeric(ledger["Shares"], errors="coerce")
    ledger["Price"] = pd.to_numeric(ledger["Price"], errors="coerce")
    ledger["Total Value"] = pd.to_numeric(ledger["Total Value"], errors="coerce")
    return ledger.dropna(subset=["Date", "Shares", "Price", "Total Value"])


def snap_to_next_trading_day(requested_date: date) -> pd.Timestamp | None:
    """Return the first master index date on or after the requested date."""
    requested = pd.Timestamp(requested_date).normalize()
    position = master_prices.index.searchsorted(requested, side="left")
    if position >= len(master_prices.index):
        return None
    return pd.Timestamp(master_prices.index[position])


def shares_held_as_of(ledger: pd.DataFrame, ticker: str, as_of: pd.Timestamp) -> int:
    """Calculate the position available for a sell, including same-day trades."""
    matching = ledger[
        (ledger["Ticker"] == ticker) & (ledger["Date"] <= as_of)
    ]
    if matching.empty:
        return 0

    signed_shares = np.where(
        matching["Action"].eq("Buy"),
        matching["Shares"].to_numpy(dtype=int),
        -matching["Shares"].to_numpy(dtype=int),
    )
    return int(np.sum(signed_shares))


def build_daily_holdings(
    prices: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Build cumulative shares held for every date and every ticker."""
    daily_changes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if ledger.empty:
        return daily_changes

    events = ledger.copy()
    events["Signed Shares"] = np.where(
        events["Action"].eq("Buy"),
        events["Shares"],
        -events["Shares"],
    )
    changes = events.pivot_table(
        index="Date",
        columns="Ticker",
        values="Signed Shares",
        aggfunc="sum",
    )
    changes = changes.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return changes.cumsum()


def simulate_portfolio(
    prices: pd.DataFrame,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return daily holdings, position values, and total value series."""
    daily_holdings = build_daily_holdings(prices, ledger)
    position_values = daily_holdings.mul(prices, axis="index")
    total_value = position_values.sum(axis=1).rename("Total_Portfolio_Value")

    if not ledger.empty:
        first_trade = ledger["Date"].min()
        total_value = total_value.loc[first_trade:]
    else:
        total_value = total_value.iloc[0:0]
    return daily_holdings, position_values, total_value


def net_capital_invested(ledger: pd.DataFrame) -> float:
    """Return net external capital: buys less withdrawn sell proceeds."""
    if ledger.empty:
        return 0.0
    buy_value = ledger.loc[ledger["Action"].eq("Buy"), "Total Value"].sum()
    sell_value = ledger.loc[ledger["Action"].eq("Sell"), "Total Value"].sum()
    return float(buy_value - sell_value)


def asset_cost_basis(ledger: pd.DataFrame, ticker: str) -> tuple[int, float]:
    """Return remaining shares and weighted-average cost basis for one ticker."""
    matching = ledger[ledger["Ticker"].eq(ticker)].sort_values(
        "Date",
        kind="stable",
    )
    shares_held = 0
    cost_basis = 0.0

    for trade in matching.itertuples(index=False):
        shares = int(trade.Shares)
        price = float(trade.Price)
        if trade.Action == "Buy":
            shares_held += shares
            cost_basis += shares * price
        elif shares_held > 0:
            average_cost = cost_basis / shares_held
            shares_held -= shares
            cost_basis -= shares * average_cost

    return shares_held, max(float(cost_basis), 0.0)


def money(value: float) -> str:
    return f"₹{value:,.2f}"


def money_compact(value: float) -> str:
    """Format card values compactly while retaining commas for smaller values."""
    sign = "-" if value < 0 else ""
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000:
        return f"{sign}₹{magnitude / 1_000_000_000:,.2f}B"
    if magnitude >= 10_000_000:
        return f"{sign}₹{magnitude / 1_000_000:,.2f}M"
    return money(value)


def pct(value: float | None) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:,.2f}%"


def render_kpi(label: str, value: str, note: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {tone}">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(height: int) -> dict[str, Any]:
    return {
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 12, "r": 18, "t": 22, "b": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#3B4536"},
        "legend": {"bgcolor": "rgba(220,213,198,0.9)", "bordercolor": "rgba(168,160,149,.35)", "borderwidth": 1, "font": {"color": "#3B4536"}},
        "xaxis": {
            "gridcolor": "#C8C0AE",
            "linecolor": "#A8A095",
            "tickfont": {"color": "#8B8578"},
        },
        "yaxis": {
            "gridcolor": "#C8C0AE",
            "linecolor": "#A8A095",
            "tickfont": {"color": "#8B8578"},
            "tickprefix": "₹",
            "tickformat": ",.0f",
        },
    }


# --------------------------------------------------------------------------- #
# Header and trade form
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="sim-hero">
      <div class="sim-eyebrow">QUANT ENGINE / WHAT-IF TIME MACHINE</div>
      <h1 class="sim-title">Historical Portfolio Simulator</h1>
      <p class="sim-subtitle">
        Replay custom buy and sell decisions against {len(all_tickers)} aligned NIFTY assets
        from {min_date:%d %b %Y} to {max_date:%d %b %Y}.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("← Main Dashboard", key="back_to_dashboard"):
    st.switch_page("app.py")

st.markdown('<div class="section-title">Trade Ledger</div>', unsafe_allow_html=True)
form_col, reset_col = st.columns([5, 1])

with form_col:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    with st.form("trade_form", clear_on_submit=False):
        asset_col, date_col = st.columns(2)
        with asset_col:
            if not filtered_tickers:
                st.warning("No companies match the selected sectors. Clear the sector filter to trade.")
                selected_ticker = None
            else:
                selected_ticker = st.selectbox(
                    "Asset",
                    options=filtered_tickers,
                    format_func=lambda ticker: f"{ticker} · {get_sector(ticker)}",
                )
        with date_col:
            selected_date = st.date_input(
                "Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                help="Weekends and holidays are moved to the next available market date.",
            )

        action_col, amount_col, submit_col = st.columns([1, 1.3, 1])
        with action_col:
            selected_action = st.radio("Action", options=["Buy", "Sell"], horizontal=True)
        with amount_col:
            amount = st.number_input(
                "Amount (₹)",
                min_value=0.01,
                value=10000.0,
                step=100.0,
                format="%.2f",
            )
        with submit_col:
            st.markdown("<div style='height: 1.72rem'></div>", unsafe_allow_html=True)
            execute_trade = st.form_submit_button(
                "Execute Trade",
                type="primary",
                use_container_width=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

with reset_col:
    st.markdown("<div style='height: 1.72rem'></div>", unsafe_allow_html=True)
    if st.button("Reset Simulator", use_container_width=True):
        st.session_state.trade_ledger = []
        st.success("Trade ledger cleared.")


if execute_trade and selected_ticker is not None:
    requested_date = pd.Timestamp(selected_date).normalize()
    executed_date = snap_to_next_trading_day(selected_date)

    if executed_date is None:
        st.error("No trading date is available on or after the selected date.")
    elif executed_date != requested_date:
        st.info(
            f"{requested_date:%d %b %Y} was not a trading day. "
            f"Trade executed on {executed_date:%d %b %Y}."
        )

    if executed_date is not None:
        asset_price = master_prices.at[executed_date, selected_ticker]

        if pd.isna(asset_price) or not np.isfinite(float(asset_price)) or float(asset_price) <= 0:
            st.error(
                "Trade Failed: Company was not public or trading on this date."
            )
        elif amount <= 0:
            st.error("Trade amount must be greater than zero.")
        else:
            existing_ledger = ledger_dataframe()
            price = float(asset_price)
            shares = int(float(amount) // price)

            if shares < 1:
                st.warning(
                    "Amount too low to purchase a single whole share at current market price."
                )
            elif selected_action == "Sell":
                available_shares = shares_held_as_of(
                    existing_ledger,
                    selected_ticker,
                    executed_date,
                )
                if shares > available_shares:
                    st.error(
                        f"Trade Failed: You only own {available_shares} shares of "
                        f"{selected_ticker}. Cannot sell {shares}."
                    )
                else:
                    actual_value = float(shares * price)
                    st.session_state.trade_ledger.append(
                        {
                            "Date": executed_date,
                            "Ticker": selected_ticker,
                            "Action": selected_action,
                            "Shares": shares,
                            "Price": price,
                            "Total Value": actual_value,
                        }
                    )
                    st.success(
                        f"Executed {selected_action.lower()} of {shares:,} "
                        f"{selected_ticker} shares at {money(price)} each."
                    )
            else:
                actual_value = float(shares * price)
                st.session_state.trade_ledger.append(
                    {
                        "Date": executed_date,
                        "Ticker": selected_ticker,
                        "Action": selected_action,
                        "Shares": shares,
                        "Price": price,
                        "Total Value": actual_value,
                    }
                )
                st.success(
                    f"Executed {selected_action.lower()} of {shares:,} "
                    f"{selected_ticker} shares at {money(price)} each."
                )


# --------------------------------------------------------------------------- #
# Simulation output
# --------------------------------------------------------------------------- #
ledger = ledger_dataframe()
daily_holdings, position_values, portfolio_value = simulate_portfolio(
    master_prices,
    ledger,
)

capital_invested = net_capital_invested(ledger)
current_value = float(portfolio_value.iloc[-1]) if not portfolio_value.empty else 0.0
absolute_pnl = current_value - capital_invested
return_pct = (
    absolute_pnl / capital_invested * 100
    if capital_invested > 0
    else None
)
simulation_date = (
    portfolio_value.index[-1]
    if not portfolio_value.empty
    else master_prices.index[-1]
)
first_trade_date = ledger["Date"].min() if not ledger.empty else None
days_elapsed = (
    int((simulation_date - first_trade_date).days)
    if first_trade_date is not None
    else 0
)
cagr_pct = (
    ((current_value / capital_invested) ** (365.25 / days_elapsed) - 1) * 100
    if days_elapsed > 0 and capital_invested > 0 and current_value >= 0
    else 0.0
)

st.markdown('<div class="section-title">Portfolio Snapshot</div>', unsafe_allow_html=True)
kpi_cols = st.columns(5)
with kpi_cols[0]:
    render_kpi(
        "Net Capital Invested",
        money_compact(capital_invested),
        "Buy proceeds less withdrawals",
        "tone-blue",
    )
with kpi_cols[1]:
    render_kpi(
        "Current Portfolio Value",
        money_compact(current_value),
        "Marked at latest historical close",
        "tone-blue",
    )
with kpi_cols[2]:
    render_kpi(
        "Absolute Profit / Loss",
        money_compact(absolute_pnl),
        "Current value minus net capital",
        "tone-green" if absolute_pnl >= 0 else "tone-red",
    )
with kpi_cols[3]:
    render_kpi(
        "Percentage Return",
        pct(return_pct),
        "Relative to net capital invested",
        "tone-green" if return_pct is None or return_pct >= 0 else "tone-red",
    )
with kpi_cols[4]:
    render_kpi(
        "CAGR",
        pct(cagr_pct),
        f"Annualized over {days_elapsed} days",
        "tone-green" if cagr_pct >= 0 else "tone-red",
    )

st.markdown('<div class="section-title">Portfolio Value Through Time</div>', unsafe_allow_html=True)
if portfolio_value.empty:
    st.markdown(
        '<div class="empty-state">Execute a trade to start the historical simulation.</div>',
        unsafe_allow_html=True,
    )
else:
    chart_values = (
        pd.to_numeric(portfolio_value, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    value_fig = go.Figure()
    value_fig.add_trace(
        go.Scatter(
            x=chart_values.index,
            y=chart_values.to_numpy(dtype=float),
            mode="lines",
            name="Portfolio value",
            line={"color": "#3B4536", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(59, 69, 54, 0.1)",
            hovertemplate="%{x|%d %b %Y}<br>Value: ₹%{y:,.2f}<extra></extra>",
        )
    )

    buy_trades = ledger[ledger["Action"].eq("Buy")]
    sell_trades = ledger[ledger["Action"].eq("Sell")]

    if not buy_trades.empty:
        buy_dates = buy_trades["Date"]
        buy_values = chart_values.reindex(buy_dates).to_numpy(dtype=float)
        value_fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_values,
                mode="markers",
                name="Buy trades",
                marker={
                    "symbol": "triangle-up",
                    "color": "#5C6B52",
                    "size": 12,
                    "line": {"color": "#F5F0E8", "width": 1},
                },
                customdata=buy_trades[["Ticker", "Shares"]].to_numpy(),
                hovertemplate=(
                    "Bought %{customdata[1]:.0f} shares of %{customdata[0]}"
                    "<br>%{x|%d %b %Y}<extra></extra>"
                ),
            )
        )

    if not sell_trades.empty:
        sell_dates = sell_trades["Date"]
        sell_values = chart_values.reindex(sell_dates).to_numpy(dtype=float)
        value_fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_values,
                mode="markers",
                name="Sell trades",
                marker={
                    "symbol": "triangle-down",
                    "color": "#8B4B4B",
                    "size": 12,
                    "line": {"color": "#F5F0E8", "width": 1},
                },
                customdata=sell_trades[["Ticker", "Shares"]].to_numpy(),
                hovertemplate=(
                    "Sold %{customdata[1]:.0f} shares of %{customdata[0]}"
                    "<br>%{x|%d %b %Y}<extra></extra>"
                ),
            )
        )

    value_fig.update_layout(
        **chart_layout(470),
        hovermode="x unified",
        showlegend=True,
        uirevision=f"portfolio-{len(ledger)}",
        xaxis_autorange=True,
        yaxis_autorange=True,
        title={
            "text": (
                f"Simulation window: {portfolio_value.index.min():%d %b %Y}"
                f" → {portfolio_value.index.max():%d %b %Y}"
            ),
            "font": {"size": 13, "color": "#8B8578"},
            "x": 0.02,
        },
    )
    st.plotly_chart(value_fig, width="stretch", config={"displayModeBar": False})

st.markdown('<div class="section-title">Holdings Breakdown</div>', unsafe_allow_html=True)
holdings_col, transactions_col = st.columns([1, 1.55])

with holdings_col:
    current_holdings = (
        daily_holdings.iloc[-1].mul(master_prices.iloc[-1])
        if not daily_holdings.empty
        else pd.Series(dtype=float)
    )
    current_holdings = current_holdings[current_holdings > 1e-10].sort_values(ascending=False)

    if current_holdings.empty:
        st.markdown(
            '<div class="empty-state">No open holdings to display.</div>',
            unsafe_allow_html=True,
        )
    else:
        donut_fig = go.Figure(
            go.Pie(
                labels=current_holdings.index,
                values=current_holdings.values,
                hole=0.62,
                textinfo="label+percent",
                textfont={"color": "#3B4536", "size": 11},
                marker={
                    "colors": [
                        "#5C6B52",
                        "#7A8F6E",
                        "#3B4536",
                        "#8B9E7E",
                        "#B5A88E",
                        "#A8A095",
                        "#8B4B4B",
                        "#D4CCBC",
                    ]
                },
                hovertemplate="%{label}<br>Value: ₹%{value:,.2f}<extra></extra>",
            )
        )
        layout_args = chart_layout(390)
        layout_args.update(
            {
                "showlegend": False,
                "margin": {"l": 12, "r": 12, "t": 18, "b": 12},
            }
        )
        donut_fig.update_layout(**layout_args)
        st.plotly_chart(donut_fig, width="stretch", config={"displayModeBar": False})

        sector_holdings = (
            pd.DataFrame(
                {
                    "Sector": [get_sector(ticker) for ticker in current_holdings.index],
                    "Value": current_holdings.to_numpy(),
                }
            )
            .groupby("Sector", as_index=False)["Value"]
            .sum()
        )
        sector_donut = go.Figure(
            go.Pie(
                labels=sector_holdings["Sector"],
                values=sector_holdings["Value"],
                hole=0.62,
                textinfo="label+percent",
                textfont={"color": "#3B4536", "size": 10},
                hovertemplate="%{label}<br>Sector value: ₹%{value:,.2f}<extra></extra>",
            )
        )
        sector_donut.update_layout(**layout_args)
        sector_donut.update_layout(title="Sector Weight Allocation")
        st.plotly_chart(sector_donut, width="stretch", config={"displayModeBar": False})

with transactions_col:
    st.markdown("#### Transaction History")
    if ledger.empty:
        st.markdown(
            '<div class="empty-state">Your executed trades will appear here.</div>',
            unsafe_allow_html=True,
        )
    else:
        display_ledger = ledger.copy()
        display_ledger["Date"] = display_ledger["Date"].dt.date
        display_ledger["Shares"] = display_ledger["Shares"].astype("int64")
        st.dataframe(
            display_ledger.iloc[::-1],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                "Price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
                "Total Value": st.column_config.NumberColumn(
                    "Total Value",
                    format="₹ %.2f",
                ),
            },
        )

st.markdown(
    '<div class="section-title">Individual Holding Inspector</div>',
    unsafe_allow_html=True,
)
latest_holdings = (
    daily_holdings.iloc[-1]
    if not daily_holdings.empty
    else pd.Series(dtype=float)
)
held_tickers = sorted(
    ticker for ticker, shares in latest_holdings.items() if float(shares) > 0
)

if not held_tickers:
    st.markdown(
        '<div class="empty-state">No open holdings to inspect.</div>',
        unsafe_allow_html=True,
    )
else:
    inspected_ticker = st.selectbox(
        "Select an open holding",
        options=held_tickers,
        key="inspected_ticker",
    )
    st.markdown(
        f'<span style="display:inline-block;background:rgba(59,69,54,.1);border:1px solid rgba(59,69,54,.25);border-radius:999px;color:#5C6B52;font-family:Inter,sans-serif;font-size:.7rem;letter-spacing:.06em;padding:.3rem .7rem;text-transform:uppercase;">Sector · {get_sector(inspected_ticker)}</span>',
        unsafe_allow_html=True,
    )
    inspected_shares, invested_capital = asset_cost_basis(
        ledger,
        inspected_ticker,
    )
    current_price = float(master_prices.at[simulation_date, inspected_ticker])
    inspected_value = inspected_shares * current_price
    inspected_pnl = inspected_value - invested_capital
    inspected_return = (
        inspected_pnl / invested_capital * 100
        if invested_capital > 0
        else None
    )

    inspector_cols = st.columns(5)
    with inspector_cols[0]:
        st.metric("Shares Held", f"{inspected_shares:,}")
    with inspector_cols[1]:
        st.metric("Invested Capital", money_compact(invested_capital))
    with inspector_cols[2]:
        st.metric("Current Market Price", money_compact(current_price))
    with inspector_cols[3]:
        st.metric("Current Total Value", money_compact(inspected_value))
    with inspector_cols[4]:
        st.metric(
            "Net PnL",
            money_compact(inspected_pnl),
            delta=pct(inspected_return),
            delta_color="normal",
        )

st.markdown(
    '<div class="section-title">Transaction Ledger</div>',
    unsafe_allow_html=True,
)
if ledger.empty:
    st.markdown(
        '<div class="empty-state">Your executed trades will appear here.</div>',
        unsafe_allow_html=True,
    )
else:
    ledger_display = ledger.copy()
    ledger_display["Date"] = ledger_display["Date"].dt.strftime("%d %b %Y")
    ledger_display["Shares"] = ledger_display["Shares"].astype("int64")
    ledger_display["Price"] = ledger_display["Price"].round(2)
    ledger_display["Total Value"] = ledger_display["Total Value"].round(2)
    st.dataframe(
        ledger_display.iloc[::-1],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Shares": st.column_config.NumberColumn("Shares", format="%d"),
            "Price": st.column_config.NumberColumn("Price", format="₹ %.2f"),
            "Total Value": st.column_config.NumberColumn(
                "Total Value",
                format="₹ %.2f",
            ),
        },
    )
