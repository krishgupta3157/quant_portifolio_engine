"""Share-based machine-learning risk and portfolio rebalancing engine."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="ML Risk Engine · Quant Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import LocalDataLoader
from src.ml_risk_engine_test import engineer_features
from src.optimization import PortfolioOptimizer
from src.sector_mapping import get_sector
from src.theme import apply_global_theme


apply_global_theme()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

.ml-shell {
  border-bottom: 1px solid #C8C0AE;
  margin-bottom: 1.5rem;
  padding: .5rem 0 1.25rem;
}
.ml-eyebrow {
  color: #5C6B52;
  font-family: 'Inter', sans-serif;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .18em;
  margin-bottom: .65rem;
  text-transform: uppercase;
}
.ml-title {
  color: #3B4536 !important;
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(1.8rem, 3.5vw, 2.8rem);
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.08;
  margin: 0;
}
.ml-subtitle {
  color: #8B8578 !important;
  font-family: 'Inter', sans-serif;
  font-size: .82rem;
  line-height: 1.7;
  margin: .75rem 0 0;
  max-width: 860px;
}
.architecture-banner {
  background: rgba(212, 204, 188, 0.6);
  border: 1px solid #C8C0AE;
  border-left: 3px solid #5C6B52;
  border-radius: 10px;
  color: #8B8578;
  font-family: 'Inter', sans-serif;
  font-size: .76rem;
  line-height: 1.7;
  margin: 1rem 0 1.5rem;
  padding: 1rem 1.25rem;
}
.architecture-banner b { color: #222B1C; }
.section-heading {
  color: #3B4536 !important;
  font-family: 'Playfair Display', serif;
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 1.75rem 0 .85rem;
}
.section-heading span { color: #5C6B52; margin-right: .45rem; }
.section-copy {
  color: #8B8578 !important;
  font-family: 'Inter', sans-serif;
  font-size: .82rem;
  line-height: 1.65;
  margin: -.35rem 0 1rem;
}
.risk-kpi {
  background: #DCD5C6;
  border: 1px solid #C8C0AE;
  border-radius: 12px;
  min-height: 130px;
  padding: 1.25rem;
  transition: border-color .2s ease, transform .2s ease;
}
.risk-kpi:hover { border-color: #3B4536; transform: translateY(-2px); }
.risk-kpi-label {
  color: #8B8578;
  font-family: 'Inter', sans-serif;
  font-size: .64rem;
  letter-spacing: .08em;
  line-height: 1.4;
  text-transform: uppercase;
}
.risk-kpi-value {
  color: #3B4536;
  font-family: 'Playfair Display', serif;
  font-size: 1.45rem;
  font-weight: 700;
  margin-top: .75rem;
}
.risk-kpi-note { color: #8B8578; font-size: .7rem; margin-top: .35rem; }
.sector-badge {
  background: rgba(59,69,54,.1);
  border: 1px solid rgba(59,69,54,.25);
  border-radius: 999px;
  color: #5C6B52;
  display: inline-block;
  font-family: 'Inter', sans-serif;
  font-size: .68rem;
  letter-spacing: .06em;
  padding: .3rem .7rem;
  text-transform: uppercase;
}
div[data-testid="stPlotlyChart"] {
  background: #DCD5C6 !important;
  border: 1px solid #C8C0AE !important;
  border-radius: 16px !important;
  margin-bottom: .5rem;
  padding: 0.5rem !important;
}
section[data-testid="stSidebar"] {
  background: #D4CCBC !important;
  border-right: 1px solid #C8C0AE;
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
  box-shadow: none !important;
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
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the 42-asset price matrix...")
def load_live_prices() -> pd.DataFrame:
    prices = LocalDataLoader().load_all_portfolio_assets(
        str(BASE_DIR / "companies"),
        missing_threshold=0.30,
    )
    return (
        prices.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=1, how="all")
        .ffill()
        .dropna(how="any")
    )


@st.cache_data(show_spinner="Engineering market risk features...")
def build_live_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return engineer_features(prices)


@st.cache_resource(show_spinner="Training LightGBM on the chronological training set...")
def train_live_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> lgb.LGBMRegressor:
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
    return model


def chart_layout(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02, "font": {"family": "Inter, sans-serif", "size": 13, "color": "#8B8578"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#3B4536", "size": 12},
        margin={"l": 60, "r": 30, "t": 60, "b": 55},
        height=height,
        legend={
            "bgcolor": "rgba(220,213,198,0.9)",
            "bordercolor": "rgba(168,160,149,.35)",
            "borderwidth": 1,
            "font": {"color": "#3B4536", "size": 11},
        },
        xaxis={
            "gridcolor": "#C8C0AE",
            "linecolor": "#A8A095",
            "tickfont": {"color": "#8B8578"},
        },
        yaxis={
            "gridcolor": "#C8C0AE",
            "linecolor": "#A8A095",
            "tickfont": {"color": "#8B8578"},
        },
        hoverlabel={"bgcolor": "#DCD5C6", "font": {"color": "#3B4536"}},
    )
    return fig


def correlation_insight(correlation: pd.DataFrame) -> str:
    values = correlation.where(~np.eye(len(correlation), dtype=bool))
    pairs = values.stack().dropna()
    if pairs.empty:
        return "There are not enough distinct assets to calculate pairwise contagion."
    highest_pair = pairs.idxmax()
    lowest_pair = pairs.idxmin()
    high_a, high_b = highest_pair
    low_a, low_b = lowest_pair
    return (
        f"Highest linkage: **{high_a} / {high_b} ({float(pairs.max()):.2f})**. "
        f"Lowest linkage: **{low_a} / {low_b} ({float(pairs.min()):.2f})**; "
        "lower-linked holdings may improve diversification."
    )


def portfolio_metrics(prices: pd.DataFrame) -> tuple[pd.Series, float, float, float]:
    returns = prices.pct_change().dropna(how="all").mean(axis=1).fillna(0.0)
    cumulative = (1.0 + returns).cumprod()
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    annualized_return = float(cumulative.iloc[-1] ** (1.0 / years) - 1.0)
    drawdown = cumulative / cumulative.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    volatility = float(returns.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * returns.mean() / volatility) if volatility > 0 else 0.0
    return cumulative, annualized_return, max_drawdown, sharpe


def infer_style(volatility: float) -> tuple[str, str]:
    if volatility < 0.12:
        return "Conservative (Capital Preservation)", "info"
    if volatility <= 0.18:
        return "Balanced", "success"
    return "Aggressive (Max Alpha)", "warning"


def render_recommendation(style: str, volatility: float, threshold: float, horizon: str) -> None:
    if volatility > threshold:
        st.warning(
            f"Elevated crash risk detected over {horizon}. Your inferred "
            f"{style} profile should use disciplined position sizing."
        )
    else:
        st.success(f"Projected risk is within the model's normal range for {horizon}.")


def clean_ledger(data: pd.DataFrame, valid_tickers: list[str]) -> pd.DataFrame:
    """Validate and aggregate the editable share ledger."""
    if data.empty or not {"Ticker", "Shares Owned"}.issubset(data.columns):
        return pd.DataFrame(columns=["Ticker", "Shares Owned"])
    ledger = data.loc[:, ["Ticker", "Shares Owned"]].copy()
    ledger["Ticker"] = ledger["Ticker"].astype("string").str.strip()
    ledger["Shares Owned"] = pd.to_numeric(ledger["Shares Owned"], errors="coerce")
    ledger = ledger.dropna(subset=["Ticker", "Shares Owned"])
    ledger = ledger[ledger["Ticker"].isin(valid_tickers)]
    ledger = ledger[ledger["Shares Owned"] > 0]
    ledger["Shares Owned"] = np.floor(ledger["Shares Owned"]).astype(int)
    return (
        ledger.groupby("Ticker", as_index=False)["Shares Owned"]
        .sum()
        .sort_values("Ticker")
        .reset_index(drop=True)
    )


def main() -> None:
    prices = load_live_prices()
    if prices.empty or prices.shape[1] < 2:
        st.error("The price matrix does not contain enough usable assets.")
        return

    features, target = build_live_features(prices)
    split_at = int(len(features) * 0.70)
    if split_at <= 0 or split_at >= len(features):
        st.error("The historical feature set is too small for a chronological model split.")
        return
    model = train_live_model(features.iloc[:split_at], target.iloc[:split_at])
    crisis_threshold = float(target.iloc[:split_at].quantile(0.75))
    valid_tickers = sorted(prices.columns.tolist())

    st.markdown(
        """
        <div class="ml-shell">
          <div class="ml-eyebrow">// Feature 04 · Share-based robo-advisor</div>
          <h1 class="ml-title">ML Dynamic Risk Engine &amp;<br>Rebalancing Desk</h1>
          <p class="ml-subtitle">Enter your live holdings, let the engine infer your risk profile, and receive exact optimizer-driven buy or sell quantities.</p>
        </div>
        <div class="architecture-banner">
          <b>Model:</b> LightGBM &nbsp;|&nbsp; <b>Optimizer:</b> Long-only MPT
          &nbsp;|&nbsp; <b>Universe:</b> 42 data-valid NIFTY assets
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("share_based_risk_setup", clear_on_submit=False):
        st.markdown("### 1 · Live Ledger")
        st.caption("Add one row per holding. Duplicate tickers are consolidated automatically.")
        initial_ledger = pd.DataFrame(columns=["Ticker", "Shares Owned"])
        ledger_editor = st.data_editor(
            initial_ledger,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Ticker": st.column_config.SelectboxColumn(
                    "Ticker",
                    options=valid_tickers,
                    required=True,
                ),
                "Shares Owned": st.column_config.NumberColumn(
                    "Shares Owned",
                    min_value=1,
                    step=1,
                    format="%d",
                    required=True,
                ),
            },
            key="share_ledger_editor_v2",
        )
        setup_columns = st.columns(2)
        with setup_columns[0]:
            base_date = st.date_input(
                "Base date",
                value=features.index.max().date(),
                min_value=features.index.min().date(),
                max_value=prices.index.max().date(),
                help="Historical prices are taken precisely on the nearest available trading date on or before this date.",
            )
        with setup_columns[1]:
            goal = st.radio(
                "Rebalancing goal",
                [
                    "Minimize Crash Loss (Defensive Rebalance)",
                    "Maximize Risk-Adjusted Profit (Aggressive Rebalance)",
                ],
            )
        submitted = st.form_submit_button("Analyze portfolio and calculate rebalance", type="primary")

    if submitted:
        cleaned_ledger = clean_ledger(ledger_editor, valid_tickers)
        if len(cleaned_ledger) < 3:
            st.error("Please add at least 3 valid stocks with positive share counts before analyzing.")
            st.stop()
        st.session_state["ml_active_portfolio"] = cleaned_ledger
        st.session_state["share_risk_analysis_active"] = True
        st.session_state["share_risk_base_date"] = base_date
        st.session_state["share_risk_goal"] = goal

    if not st.session_state.get("share_risk_analysis_active", False):
        st.info("Enter your holdings above and submit the ledger to activate the robo-advisor.")
        return

    ledger = st.session_state.get(
        "ml_active_portfolio",
        pd.DataFrame(columns=["Ticker", "Shares Owned"]),
    ).copy()
    base_date = st.session_state.get("share_risk_base_date", base_date)
    goal = st.session_state.get("share_risk_goal", goal)
    if ledger.empty:
        st.warning("Add at least one ticker with a positive whole-number share count.")
        return

    requested_date = pd.Timestamp(base_date)
    eligible_prices = prices.loc[prices.index <= requested_date]
    eligible_features = features.loc[features.index <= requested_date]
    if eligible_prices.empty or eligible_features.empty:
        st.error("No historical prices or engineered features are available on or before the selected date.")
        return
    effective_date = eligible_prices.index[-1]
    feature_date = eligible_features.index[-1]
    selected_tickers = ledger["Ticker"].tolist()
    selected_prices = eligible_prices.loc[:, selected_tickers].dropna(how="all")
    selected_prices = selected_prices.ffill().dropna(how="any")
    if selected_prices.empty:
        st.error("The selected holdings do not have aligned historical price data.")
        return

    prices_at_date = prices.loc[effective_date, selected_tickers].astype(float)
    shares = ledger.set_index("Ticker")["Shares Owned"].astype(float).reindex(selected_tickers)
    holding_values = shares * prices_at_date
    total_value = float(holding_values.sum())
    if total_value <= 0 or not np.isfinite(total_value):
        st.error("The calculated portfolio value is zero or invalid. Check the share counts.")
        return
    current_weights = holding_values / total_value

    historical_returns = selected_prices.pct_change().dropna(how="all").fillna(0.0)
    weighted_returns = historical_returns.mul(current_weights, axis=1).sum(axis=1)
    historical_volatility = float(weighted_returns.std(ddof=1) * np.sqrt(252.0))
    inferred_style, style_kind = infer_style(historical_volatility)
    if style_kind == "info":
        st.info(f"Inferred Style: {inferred_style} · Historical annualized volatility: {historical_volatility:.2%}")
    elif style_kind == "success":
        st.success(f"Inferred Style: {inferred_style} · Historical annualized volatility: {historical_volatility:.2%}")
    else:
        st.warning(f"Inferred Style: {inferred_style} · Historical annualized volatility: {historical_volatility:.2%}")

    st.markdown('<div class="section-heading"><span>02</span> Portfolio Snapshot &amp; Sector Concentration</div>', unsafe_allow_html=True)
    snapshot_columns = st.columns(4)
    snapshot_values = [
        ("Current Portfolio Value", f"₹{total_value:,.2f}", f"Prices on {effective_date.date()}"),
        ("Historical Annualized Volatility", f"{historical_volatility:.2%}", "Weighted share-based basket"),
        ("Holdings", f"{len(selected_tickers)}", "Distinct positions"),
        ("Objective", "Defensive" if "Minimize" in goal else "Aggressive", "Optimizer target"),
    ]
    for column, (label, value, note) in zip(snapshot_columns, snapshot_values):
        with column:
            st.markdown(
                f'<div class="risk-kpi"><div class="risk-kpi-label">{label}</div><div class="risk-kpi-value">{value}</div><div class="risk-kpi-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    sector_concentration = (
        pd.DataFrame(
            {
                "Sector": [get_sector(ticker) for ticker in holding_values.index],
                "Value": holding_values.to_numpy(),
            }
        )
        .groupby("Sector", as_index=False)["Value"]
        .sum()
    )
    sector_concentration["Weight"] = sector_concentration["Value"] / total_value
    concentration_columns = st.columns([1.1, 1])
    with concentration_columns[0]:
        sector_fig = go.Figure(
            go.Pie(
                labels=sector_concentration["Sector"],
                values=sector_concentration["Value"],
                hole=0.62,
                textinfo="label+percent",
                marker={"line": {"color": "#E8E3D7", "width": 2}},
                hovertemplate="<b>%{label}</b><br>Value: ₹%{value:,.2f}<br>Weight: %{percent}<extra></extra>",
            )
        )
        sector_fig.add_annotation(
            text=f"<b>{len(sector_concentration)}</b><br><span style='font-size:9px;color:#8B8578'>SECTORS</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"family": "Playfair Display, serif", "size": 18, "color": "#3B4536"},
        )
        st.plotly_chart(
            chart_layout(sector_fig, "Current Sector Concentration", 380),
            width="stretch",
            config={"displayModeBar": False},
        )
    with concentration_columns[1]:
        display_concentration = sector_concentration.copy()
        display_concentration["Weight"] = display_concentration["Weight"].map(lambda value: f"{value:.2%}")
        display_concentration["Value"] = display_concentration["Value"].map(lambda value: f"₹{value:,.2f}")
        st.dataframe(display_concentration, hide_index=True, width="stretch")

    st.markdown('<div class="section-heading"><span>03</span> ML Multi-Horizon Risk</div>', unsafe_allow_html=True)
    selected_features = eligible_features.iloc[[-1]]
    base_predicted_volatility = max(0.0, float(model.predict(selected_features)[0]))
    horizon_options = {
        "1 month": 21,
        "3 months": 63,
        "6 months": 126,
        "1 year": 252,
        "2 years": 504,
        "3 years": 756,
        "4 years": 1008,
        "5 years": 1260,
        "Custom": None,
    }
    horizon_choice = st.selectbox(
        "Select prediction horizon",
        options=list(horizon_options),
        index=2,
        key="share_risk_horizon",
        help="Uses the 30-trading-day LightGBM forecast and square-root-of-time scaling.",
    )
    if horizon_choice == "Custom":
        horizon_days = int(
            st.number_input(
                "Custom horizon (trading days)",
                min_value=1,
                max_value=2520,
                value=378,
                step=1,
                key="share_risk_custom_days",
            )
        )
        horizon_label = f"Custom · {horizon_days:,} trading days"
    else:
        horizon_days = int(horizon_options[horizon_choice])
        horizon_label = horizon_choice
    horizon_volatility = base_predicted_volatility * np.sqrt(horizon_days / 30.0)
    risk_columns = st.columns(2)
    with risk_columns[0]:
        st.markdown(
            f'<div class="risk-kpi"><div class="risk-kpi-label">Predicted Volatility · {horizon_label}</div><div class="risk-kpi-value">{horizon_volatility:.2%}</div><div class="risk-kpi-note">Base 30D forecast: {base_predicted_volatility:.2%}</div></div>',
            unsafe_allow_html=True,
        )
    with risk_columns[1]:
        render_recommendation(inferred_style, horizon_volatility, crisis_threshold, horizon_label)

    st.markdown('<div class="section-heading"><span>04</span> Relationship Map</div>', unsafe_allow_html=True)
    asset_returns = selected_prices.pct_change().dropna(how="all")
    correlation = asset_returns.corr().fillna(0.0)
    heatmap = go.Figure(
        go.Heatmap(
            z=correlation.to_numpy(),
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#8B4B4B"], [0.5, "#DCD5C6"], [1, "#3B4536"]],
            texttemplate="%{z:.2f}",
            hovertemplate="%{y} ↔ %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar={"title": "Correlation"},
        )
    )
    st.plotly_chart(
        chart_layout(heatmap, "Current Holdings Correlation Matrix", 440),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.info(correlation_insight(correlation))

    st.markdown('<div class="section-heading"><span>05</span> Historical Portfolio X-Ray</div>', unsafe_allow_html=True)
    history = prices.loc[:effective_date]
    selected_cumulative, annualized_return, max_drawdown, sharpe = portfolio_metrics(history[selected_tickers])
    benchmark_cumulative, _, _, _ = portfolio_metrics(history)
    performance = pd.DataFrame(
        {
            "Current share-weighted basket": selected_cumulative,
            "NIFTY 42 equal-weight benchmark": benchmark_cumulative,
        }
    ).dropna()
    area = go.Figure()
    for column, color in zip(performance.columns, ["#3B4536", "#7A8F6E"]):
        area.add_trace(
            go.Scatter(
                x=performance.index,
                y=performance[column],
                mode="lines",
                name=column,
                line={"color": color, "width": 2},
            )
        )
    area.update_yaxes(title="Growth of ₹1", tickformat=".2f", gridcolor="#C8C0AE")
    area.update_xaxes(showgrid=False)
    st.plotly_chart(
        chart_layout(area, "Cumulative Returns Through Base Date", 420),
        width="stretch",
        config={"displayModeBar": False},
    )
    metric_columns = st.columns(3)
    for column, label, value, note in [
        (metric_columns[0], "Historical Annualized Return", f"{annualized_return:.2%}", "Share-weighted basket"),
        (metric_columns[1], "Maximum Drawdown", f"{max_drawdown:.2%}", "Peak-to-trough"),
        (metric_columns[2], "Sharpe Ratio", f"{sharpe:.2f}", "252-day annualization"),
    ]:
        with column:
            st.markdown(
                f'<div class="risk-kpi"><div class="risk-kpi-label">{label}</div><div class="risk-kpi-value">{value}</div><div class="risk-kpi-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-heading"><span>06</span> Actionable Rebalancing Recommendations</div>', unsafe_allow_html=True)
    optimizer = PortfolioOptimizer(selected_prices)
    try:
        optimal_weights = (
            optimizer.optimize_min_volatility()
            if "Minimize" in goal
            else optimizer.optimize_max_sharpe()
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        st.error(f"The optimizer could not produce a valid allocation for these holdings: {exc}")
        return

    optimal_weights = pd.Series(optimal_weights, dtype=float).reindex(selected_tickers, fill_value=0.0)
    optimal_weights = optimal_weights.clip(lower=0.0)
    weight_total = float(optimal_weights.sum())
    if weight_total <= 0:
        st.error("The optimizer returned no positive weights for the selected holdings.")
        return
    optimal_weights = optimal_weights / weight_total
    target_shares = np.rint(optimal_weights * total_value / prices_at_date).astype(int)
    share_delta = target_shares - shares.astype(int)
    delta_value = share_delta * prices_at_date
    recommendations = pd.DataFrame(
        {
            "Ticker": selected_tickers,
            "Sector": [get_sector(ticker) for ticker in selected_tickers],
            "Current Shares": shares.astype(int).to_numpy(),
            "Current Weight": current_weights.to_numpy(),
            "Target Weight": optimal_weights.to_numpy(),
            "Target Shares": target_shares.to_numpy(),
            "Share Difference": share_delta.to_numpy(),
            "Trade Value": delta_value.to_numpy(),
        }
    )
    recommendations["Action"] = np.where(
        recommendations["Share Difference"] > 0,
        "BUY",
        np.where(recommendations["Share Difference"] < 0, "SELL", "HOLD"),
    )
    recommendations["Instruction"] = recommendations.apply(
        lambda row: (
            f"Buy {int(row['Share Difference']):,} shares of {row['Ticker']}"
            if row["Share Difference"] > 0
            else f"Sell {abs(int(row['Share Difference'])):,} shares of {row['Ticker']}"
            if row["Share Difference"] < 0
            else f"Hold {row['Ticker']}"
        ),
        axis=1,
    )
    visible_recommendations = recommendations[recommendations["Action"] != "HOLD"].copy()
    if visible_recommendations.empty:
        st.success("Your current share counts already match the selected optimizer target after whole-share rounding.")
    else:
        st.dataframe(
            visible_recommendations[
                ["Instruction", "Sector", "Current Weight", "Target Weight", "Trade Value"]
            ].style.format(
                {
                    "Current Weight": "{:.2%}",
                    "Target Weight": "{:.2%}",
                    "Trade Value": "₹{:,.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    risk_by_ticker = asset_returns.std(ddof=1).mul(np.sqrt(252.0))
    weighted_asset_risk = float(risk_by_ticker.mul(current_weights).sum())
    if weighted_asset_risk > 0 and np.isfinite(weighted_asset_risk):
        ml_scaled_risk = base_predicted_volatility * risk_by_ticker / weighted_asset_risk
        risk_breakdown = (
            pd.DataFrame(
                {
                    "Sector": [get_sector(ticker) for ticker in selected_tickers],
                    "Predicted Volatility": ml_scaled_risk.reindex(selected_tickers).to_numpy(),
                }
            )
            .groupby("Sector", as_index=False)["Predicted Volatility"]
            .mean()
            .sort_values("Predicted Volatility", ascending=False)
        )
        risk_bar = go.Figure(
            go.Bar(
                x=risk_breakdown["Sector"],
                y=risk_breakdown["Predicted Volatility"],
                marker_color="#8B4B4B",
                hovertemplate="%{x}<br>ML-scaled volatility: %{y:.2%}<extra></extra>",
            )
        )
        risk_bar.update_xaxes(tickangle=-35)
        risk_bar.update_yaxes(title="ML-scaled 30D volatility", tickformat=".0%")
        st.plotly_chart(
            chart_layout(risk_bar, "Industry-wise Risk Breakdown", 390),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.caption(
        f"Prices and share deltas use the nearest available trading date {effective_date.date()}; "
        f"the ML feature row uses {feature_date.date()} to avoid future leakage."
    )


main()
