"""Interactive machine-learning risk engine for a user-defined NIFTY basket."""

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
from src.theme import apply_global_theme


apply_global_theme()

st.markdown(
    """
<style>
.ml-shell { border-bottom: 1px solid #334155; margin-bottom: 1.5rem; padding: .5rem 0 1.25rem; }
.ml-eyebrow { color: #38BDF8; font-family: 'JetBrains Mono', monospace; font-size: .7rem;
  font-weight: 700; letter-spacing: .16em; margin-bottom: .65rem; text-transform: uppercase; }
.ml-title { color: #F8FAFC !important; font-size: clamp(2rem, 4vw, 3.4rem); font-weight: 800;
  letter-spacing: -.05em; line-height: 1.08; margin: 0; }
.ml-subtitle { color: #94A3B8 !important; font-size: .9rem; line-height: 1.7; margin: .75rem 0 0; max-width: 820px; }
.architecture-banner { background: #172033; border: 1px solid #334155; border-left: 3px solid #38BDF8;
  border-radius: 10px; color: #CBD5E1; font-family: 'JetBrains Mono', monospace; font-size: .76rem;
  line-height: 1.7; margin: 1rem 0 1.5rem; padding: 1rem 1.25rem; }
.architecture-banner b { color: #F8FAFC; }
.section-heading { color: #F8FAFC !important; font-size: 1rem; font-weight: 800; letter-spacing: .08em;
  margin: 1.75rem 0 .85rem; text-transform: uppercase; }
.section-heading span { color: #38BDF8; margin-right: .45rem; }
.section-copy { color: #94A3B8 !important; font-size: .82rem; line-height: 1.65; margin: -.35rem 0 1rem; }
.risk-kpi { background: #1E293B; border: 1px solid #334155; border-radius: 12px; min-height: 130px; padding: 1.25rem; }
.risk-kpi-label { color: #94A3B8; font-family: 'JetBrains Mono', monospace; font-size: .64rem;
  letter-spacing: .08em; line-height: 1.4; text-transform: uppercase; }
.risk-kpi-value { color: #F8FAFC; font-size: 1.45rem; font-weight: 800; margin-top: .75rem; }
.risk-kpi-note { color: #64748B; font-size: .7rem; margin-top: .35rem; }
div[data-testid="stPlotlyChart"] { border-color: #334155 !important; margin-bottom: .5rem; }
section[data-testid="stSidebar"] { background: #172033 !important; border-right: 1px solid #334155; }
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
        template="plotly_dark",
        title={"text": title, "x": 0.02},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#F8FAFC"},
        margin={"l": 60, "r": 30, "t": 60, "b": 55},
        height=height,
    )
    return fig


def correlation_insight(correlation: pd.DataFrame) -> str:
    values = correlation.where(~np.eye(len(correlation), dtype=bool))
    pairs = values.stack().dropna()
    if pairs.empty:
        return "Select at least two assets with usable return history to inspect contagion."
    highest_pair = pairs.idxmax()
    lowest_pair = pairs.idxmin()
    highest = float(pairs.max())
    lowest = float(pairs.min())
    high_a, high_b = highest_pair
    low_a, low_b = lowest_pair
    return (
        f"Highest linkage: **{high_a} / {high_b} ({highest:.2f})**. "
        f"They may sell off together. Lowest linkage: **{low_a} / {low_b} "
        f"({lowest:.2f})**; the lower-linked asset may provide better diversification."
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


def recommendation(style: str, volatility: float, threshold: float, horizon: str) -> None:
    high_risk = volatility > threshold
    if high_risk and style.startswith("Aggressive"):
        st.warning(
            f"High crash risk detected over {horizon}. Aggressive growth exposure is "
            "currently heavy; consider a minimum-volatility allocation."
        )
    elif high_risk:
        st.warning(
            f"Elevated risk detected over {horizon}. Keep position sizes defensive "
            "and diversify away from highly correlated assets."
        )
    elif style.startswith("Conservative"):
        st.success(f"Risk is within the model's normal range for {horizon}; defensive sizing fits your style.")
    else:
        st.success(f"Risk is within the model's normal range for {horizon}; your selected style is supported.")


def main() -> None:
    prices = load_live_prices()
    if prices.empty or prices.shape[1] < 3:
        st.error("The price matrix does not contain at least three usable assets.")
        return

    features, target = build_live_features(prices)
    split_at = int(len(features) * 0.70)
    if split_at <= 0 or split_at >= len(features):
        st.error("The historical feature set is too small for a chronological model split.")
        return
    model = train_live_model(features.iloc[:split_at], target.iloc[:split_at])
    crisis_threshold = float(target.iloc[:split_at].quantile(0.75))

    st.markdown(
        """
        <div class="ml-shell">
          <div class="ml-eyebrow">// Feature 04 · Forward-looking risk intelligence</div>
          <h1 class="ml-title">ML Dynamic Risk Engine &amp;<br>Portfolio X-Ray</h1>
          <p class="ml-subtitle">Build a custom basket, inspect contagion, evaluate its history, and project model-driven risk across four time horizons.</p>
        </div>
        <div class="architecture-banner">
          <b>Model:</b> LightGBM &nbsp;|&nbsp; <b>Validation:</b> 70/30 chronological split
          &nbsp;|&nbsp; <b>Universe:</b> 42 data-valid NIFTY assets
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_tickers = sorted(prices.columns.tolist())
    with st.form("risk_engine_setup"):
        st.markdown("### 1 · Define your portfolio")
        selected_tickers = st.multiselect(
            "Company selection (choose at least 3)",
            options=available_tickers,
            default=available_tickers[:3],
            help="The analysis uses only these assets for the custom basket and relationship map.",
        )
        setup_columns = st.columns(2)
        with setup_columns[0]:
            trade_style = st.radio(
                "Trade style / risk tolerance",
                ["Conservative (Defensive)", "Balanced (Market Neutral)", "Aggressive (Growth/Max Sharpe)"],
                index=1,
            )
        with setup_columns[1]:
            base_date = st.date_input(
                "Base date",
                value=features.index.max().date(),
                min_value=features.index.min().date(),
                max_value=prices.index.max().date(),
                help="Weekend dates use the latest available trading day on or before the selection.",
            )
        submitted = st.form_submit_button("Run portfolio risk analysis", type="primary")

    if submitted:
        st.session_state["risk_engine_analysis_active"] = True
    analysis_active = submitted or bool(
        st.session_state.get("risk_engine_analysis_active", False)
    )
    if not analysis_active:
        st.info("Choose at least three assets and submit the setup to run the interactive analysis.")
        return
    if len(selected_tickers) < 3:
        st.error("Please select at least 3 stocks to form a portfolio.")
        return

    requested_date = pd.Timestamp(base_date)
    eligible_features = features.loc[features.index <= requested_date]
    if eligible_features.empty:
        st.error("No engineered feature row is available on or before the selected base date.")
        return
    effective_date = eligible_features.index[-1]
    selected_features = eligible_features.iloc[[-1]]
    predicted_volatility = max(0.0, float(model.predict(selected_features)[0]))

    selected_prices = prices.loc[:effective_date, selected_tickers].dropna(how="all")
    if selected_prices.shape[1] < 3 or selected_prices.empty:
        st.error("The selected assets do not have enough aligned historical prices.")
        return
    asset_returns = selected_prices.pct_change().dropna(how="all")
    correlation = asset_returns.corr().fillna(0.0)

    st.markdown('<div class="section-heading"><span>02</span> Inter-Asset Relationship &amp; Contagion</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-copy">Pairwise correlations for your {len(selected_tickers)}-asset basket through <b>{effective_date.date()}</b>.</div>',
        unsafe_allow_html=True,
    )
    heatmap = go.Figure(
        go.Heatmap(
            z=correlation.to_numpy(),
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#F87171"], [0.5, "#1E293B"], [1, "#38BDF8"]],
            texttemplate="%{z:.2f}",
            hovertemplate="%{y} ↔ %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar={"title": "Correlation"},
        )
    )
    st.plotly_chart(chart_layout(heatmap, "Selected Basket Correlation Matrix", 470), use_container_width=True, config={"displayModeBar": False})
    st.info(correlation_insight(correlation))

    st.markdown('<div class="section-heading"><span>03</span> Historical Portfolio X-Ray</div>', unsafe_allow_html=True)
    history = prices.loc[:effective_date]
    selected_cumulative, annualized_return, max_drawdown, sharpe = portfolio_metrics(history[selected_tickers])
    benchmark_cumulative, _, _, _ = portfolio_metrics(history)
    performance = pd.DataFrame({"Custom equal-weight basket": selected_cumulative, "NIFTY 42 equal-weight benchmark": benchmark_cumulative}).dropna()
    area = go.Figure()
    for column, color in zip(performance.columns, ["#38BDF8", "#34D399"]):
        area.add_trace(go.Scatter(x=performance.index, y=performance[column], mode="lines", name=column, line={"color": color, "width": 2}, fill="tonexty" if column == performance.columns[0] else None))
    area.update_yaxes(title="Growth of ₹1", tickformat=".2f", gridcolor="#334155")
    area.update_xaxes(showgrid=False)
    st.plotly_chart(chart_layout(area, "Cumulative Returns: Custom Basket vs NIFTY 42", 430), use_container_width=True, config={"displayModeBar": False})

    metric_columns = st.columns(3)
    for column, label, value, note in [
        (metric_columns[0], "Historical Annualized Return", f"{annualized_return:.2%}", "Equal-weight custom basket"),
        (metric_columns[1], "Maximum Drawdown", f"{max_drawdown:.2%}", "Peak-to-trough through base date"),
        (metric_columns[2], "Sharpe Ratio", f"{sharpe:.2f}", "252-day annualization"),
    ]:
        with column:
            st.markdown(f'<div class="risk-kpi"><div class="risk-kpi-label">{label}</div><div class="risk-kpi-value">{value}</div><div class="risk-kpi-note">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading"><span>04</span> ML Multi-Horizon Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-copy">Base-date LightGBM forecast: <b>{effective_date.date()}</b>. Longer horizons use the square-root-of-time rule.</div>',
        unsafe_allow_html=True,
    )
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
        help="Volatility is projected from the model's 30-trading-day forecast using the square-root-of-time rule.",
    )
    if horizon_choice == "Custom":
        custom_days = st.number_input(
            "Custom horizon (trading days)",
            min_value=1,
            max_value=2520,
            value=378,
            step=1,
            help="Enter the horizon in trading days. 252 trading days is approximately one year.",
        )
        horizon_days = int(custom_days)
        horizon_label = f"Custom · {horizon_days:,} trading days"
    else:
        horizon_days = horizon_options[horizon_choice]
        assert horizon_days is not None
        horizon_label = horizon_choice

    horizon_volatility = predicted_volatility * np.sqrt(horizon_days / 30.0)
    horizon_card, horizon_note = st.columns([1.25, 1], gap="large")
    with horizon_card:
        st.markdown(
            f"""
            <div class="risk-kpi">
              <div class="risk-kpi-label">Predicted Volatility · {horizon_label}</div>
              <div class="risk-kpi-value">{horizon_volatility:.2%}</div>
              <div class="risk-kpi-note">Scaled from the 30-day LightGBM forecast across {horizon_days:,} trading days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with horizon_note:
        recommendation(trade_style, horizon_volatility, crisis_threshold, horizon_label)

    st.caption("Features are evaluated only through the selected base date; no future observations are used in the prediction row.")


main()
