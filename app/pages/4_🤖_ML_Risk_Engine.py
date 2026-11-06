"""UI shell for the machine-learning dynamic risk engine."""

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
from src.ml_risk_engine_test import (
    engineer_features,
)
from src.optimization import PortfolioOptimizer
from src.theme import apply_global_theme


apply_global_theme()

st.markdown(
    """
<style>
.ml-shell {
  border-bottom: 1px solid #334155;
  margin-bottom: 1.75rem;
  padding: .5rem 0 1.5rem;
}
.ml-eyebrow {
  color: #38BDF8;
  font-family: 'JetBrains Mono', monospace;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .16em;
  margin-bottom: .65rem;
  text-transform: uppercase;
}
.ml-title {
  color: #F8FAFC !important;
  font-size: clamp(2rem, 4vw, 3.4rem);
  font-weight: 800;
  letter-spacing: -.05em;
  line-height: 1.08;
  margin: 0;
}
.ml-subtitle {
  color: #94A3B8 !important;
  font-size: .9rem;
  line-height: 1.7;
  margin: .75rem 0 0;
  max-width: 820px;
}
.architecture-banner {
  background: #172033;
  border: 1px solid #334155;
  border-left: 3px solid #38BDF8;
  border-radius: 10px;
  color: #CBD5E1;
  font-family: 'JetBrains Mono', monospace;
  font-size: .76rem;
  line-height: 1.7;
  margin: 1rem 0 1.75rem;
  padding: 1rem 1.25rem;
}
.architecture-banner b { color: #F8FAFC; }
.section-heading {
  color: #F8FAFC !important;
  font-size: 1rem;
  font-weight: 800;
  letter-spacing: .08em;
  margin: 1.75rem 0 .85rem;
  text-transform: uppercase;
}
.section-heading span { color: #38BDF8; margin-right: .45rem; }
.section-copy {
  color: #94A3B8 !important;
  font-size: .82rem;
  line-height: 1.65;
  margin: -.35rem 0 1rem;
}
.risk-kpi {
  background: #1E293B;
  border: 1px solid #334155;
  border-radius: 12px;
  min-height: 135px;
  padding: 1.35rem 1.5rem;
  transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
}
.risk-kpi:hover {
  border-color: #38BDF8;
  box-shadow: 0 8px 20px rgba(56, 189, 248, .08);
  transform: translateY(-2px);
}
.risk-kpi-label {
  color: #94A3B8;
  font-family: 'JetBrains Mono', monospace;
  font-size: .64rem;
  letter-spacing: .08em;
  line-height: 1.4;
  text-transform: uppercase;
}
.risk-kpi-value {
  color: #F8FAFC;
  font-size: 1.55rem;
  font-weight: 800;
  margin-top: .75rem;
}
.risk-kpi-note { color: #64748B; font-size: .7rem; margin-top: .35rem; }
.status-pill {
  background: rgba(52, 211, 153, .10);
  border: 1px solid rgba(52, 211, 153, .35);
  border-radius: 999px;
  color: #34D399;
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem;
  letter-spacing: .08em;
  padding: .3rem .65rem;
  text-transform: uppercase;
}
div[data-testid="stPlotlyChart"] {
  border-color: #334155 !important;
  margin-bottom: .5rem;
}
section[data-testid="stSidebar"] {
  background: #172033 !important;
  border-right: 1px solid #334155;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the 42-asset price matrix...")
def load_live_prices() -> pd.DataFrame:
    """Load the same cleaned multi-asset matrix used by the test script."""
    prices = LocalDataLoader().load_all_portfolio_assets(
        str(BASE_DIR / "companies"),
        missing_threshold=0.30,
    )
    return prices.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(axis=1, how="all").ffill().dropna(how="any")


@st.cache_data(show_spinner="Engineering market risk features...")
def build_live_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the validated feature matrix and future-volatility target."""
    return engineer_features(prices)


@st.cache_resource(show_spinner="Training LightGBM on the chronological training set...")
def train_live_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> lgb.LGBMRegressor:
    """Fit the validated conservative LightGBM configuration once per session."""
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


prices = load_live_prices()
features, target = build_live_features(prices)
split_at = int(len(features) * 0.70)
x_train, x_test = features.iloc[:split_at], features.iloc[split_at:]
y_train, y_test = target.iloc[:split_at], target.iloc[split_at:]
model = train_live_model(x_train, y_train)
crisis_threshold = float(y_train.quantile(0.75))

test_start = x_test.index.min().date()
test_end = x_test.index.max().date()

with st.sidebar:
    st.markdown("### Simulation Controls")
    st.caption("Run the trained LightGBM risk model at a historical test-set date.")
    selected_date = st.date_input(
        "Historical analysis date",
        value=test_start,
        min_value=test_start,
        max_value=test_end,
        help="Weekend selections use the nearest available trading-day features.",
    )
    st.divider()
    st.markdown("**Scenario status**")
    st.markdown('<span class="status-pill">Live Model Active</span>', unsafe_allow_html=True)
    st.caption(f"Test window: {test_start} → {test_end}")

selected_timestamp = pd.Timestamp(selected_date)
feature_lookup = features.reindex(
    features.index.union(pd.DatetimeIndex([selected_timestamp]))
).sort_index().ffill().bfill()
selected_features = feature_lookup.loc[[selected_timestamp]]
effective_date = selected_features.index[0]
predicted_volatility = float(model.predict(selected_features)[0])
is_crisis = predicted_volatility > crisis_threshold
regime = "DEFENSE (Min Volatility)" if is_crisis else "OFFENSE (Max Sharpe)"
regime_color = "#F87171" if is_crisis else "#34D399"

price_history = prices.loc[:effective_date]
optimizer = PortfolioOptimizer(price_history)
max_sharpe_weights = optimizer.optimize_max_sharpe()
min_volatility_weights = optimizer.optimize_min_volatility()
selected_weights = min_volatility_weights if is_crisis else max_sharpe_weights
selected_portfolio_label = "Min Volatility" if is_crisis else "Max Sharpe"

recent_prices = price_history.tail(30)
recent_returns = recent_prices.pct_change().dropna()
recent_corr = recent_returns.corr().fillna(0.0)
heatmap_assets = list(selected_weights)[:12]
heatmap_assets = [asset for asset in heatmap_assets if asset in recent_corr.columns]
if len(heatmap_assets) < 2:
    heatmap_assets = list(recent_corr.columns[:5])
correlation_matrix = recent_corr.loc[heatmap_assets, heatmap_assets]
contagion_score = float(np.clip(correlation_matrix.where(
    ~np.eye(len(correlation_matrix), dtype=bool)
).stack().mean() * 10, 0, 10))
rolling_peak = price_history.tail(30).max()
recent_drawdown = float((price_history.iloc[-1] / rolling_peak - 1.0).min())

st.markdown(
    """
<div class="ml-shell">
  <div class="ml-eyebrow">// Feature 04 · Forward-looking risk intelligence</div>
  <h1 class="ml-title">ML Dynamic Risk Engine &amp;<br>Asset Relationship Analyzer</h1>
  <p class="ml-subtitle">
    Simulate macro shocks, inspect predicted contagion behavior, and compare
    offense-versus-defense portfolio positioning.
  </p>
</div>
<div class="architecture-banner">
  <b>Model Architecture:</b> LightGBM
  &nbsp;|&nbsp; <b>Validation:</b> 70/30 Chronological Split
  &nbsp;|&nbsp; <b>Status:</b> Live Model Active
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading"><span>01</span> ML Predicted Risk KPIs</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="section-copy">Live LightGBM output for the nearest trading day '
    f'<b>{effective_date.date()}</b>; crisis threshold: {crisis_threshold:.2%}.</div>',
    unsafe_allow_html=True,
)

kpi_columns = st.columns(4, gap="large")
kpis = [
    ("Predicted 30-Day Market Volatility", f"{predicted_volatility:.2%}", "Live LightGBM forecast"),
    ("Recommended Portfolio Regime", regime, f"Threshold {crisis_threshold:.2%}"),
    ("Recent Maximum Drawdown", f"{recent_drawdown:.2%}", "Trailing 30 trading days"),
    ("Contagion Risk Score", f"{contagion_score:.1f} / 10", "Live 30-day correlation"),
]
for column, (label, value, note) in zip(kpi_columns, kpis):
    with column:
        st.markdown(
            f"""
            <div class="risk-kpi">
              <div class="risk-kpi-label">{label}</div>
              <div class="risk-kpi-value">{value}</div>
              <div class="risk-kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="section-heading"><span>02</span> Inter-Asset Relationship &amp; Contagion Map</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-copy">Trailing 30-day realized correlations across the '
    'assets receiving non-zero optimizer weights.</div>',
    unsafe_allow_html=True,
)

heatmap = go.Figure(
    go.Heatmap(
        z=correlation_matrix.to_numpy(),
        x=heatmap_assets,
        y=heatmap_assets,
        zmin=-1,
        zmax=1,
        colorscale=[[0, "#F87171"], [.5, "#1E293B"], [1, "#38BDF8"]],
        texttemplate="%{z:.2f}",
        textfont={"color": "#F8FAFC", "size": 12},
        hovertemplate="%{y} ↔ %{x}<br>30-day correlation: %{z:.2f}<extra></extra>",
        colorbar={"title": "Correlation", "tickfont": {"color": "#94A3B8"}},
    )
)
heatmap.update_layout(
    template="plotly_dark",
    title={"text": "Live 30-Day Asset Correlation Map", "x": .02},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"family": "Inter, sans-serif", "color": "#F8FAFC"},
    margin={"l": 70, "r": 30, "t": 60, "b": 65},
    height=470,
)
st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    '<div class="section-heading"><span>03</span> Dynamic Portfolio Rebalancing</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="section-copy">The model selected <b style="color:{regime_color}">'
    f'{selected_portfolio_label}</b> weights for {effective_date.date()}.</div>',
    unsafe_allow_html=True,
)

weights = pd.Series(selected_weights, dtype=float).sort_values(ascending=False)
allocation_assets = list(weights.head(12).index)
other_weight = max(0.0, 1.0 - float(weights.head(12).sum()))
if other_weight > 1e-4:
    allocation_assets.append("OTHER")
current_weights = pd.Series(1.0 / len(optimizer.tickers), index=optimizer.tickers)
current_other_weight = max(
    0.0,
    1.0 - float(current_weights.reindex(weights.head(12).index).sum()),
)
current_allocation = [
    current_other_weight if asset == "OTHER" else float(current_weights.get(asset, 0.0))
    for asset in allocation_assets
]
optimized_allocation = [
    float(weights.get(asset, other_weight if asset == "OTHER" else 0.0))
    for asset in allocation_assets
]
allocation_chart = go.Figure()
for asset_index, asset in enumerate(allocation_assets):
    allocation_chart.add_trace(
        go.Bar(
            name=asset,
            x=["Current Allocation", "ML Optimized Allocation"],
            y=[current_allocation[asset_index], optimized_allocation[asset_index]],
            marker_color="#38BDF8" if asset_index % 2 == 0 else "#34D399",
            hovertemplate=f"{asset}<br>Weight: %{{y:.2%}}<extra></extra>",
        )
    )
allocation_chart.update_layout(
    template="plotly_dark",
    barmode="stack",
    title={"text": "Current vs ML Optimized Allocation", "x": .02},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"family": "Inter, sans-serif", "color": "#F8FAFC"},
    xaxis={"showgrid": False, "tickfont": {"color": "#94A3B8"}},
    yaxis={"title": "Portfolio weight", "tickformat": ".0%", "gridcolor": "#334155", "tickfont": {"color": "#94A3B8"}},
    legend={"font": {"color": "#F8FAFC"}},
    margin={"l": 60, "r": 30, "t": 60, "b": 55},
    height=420,
)
st.plotly_chart(allocation_chart, use_container_width=True, config={"displayModeBar": False})

st.caption("Live LightGBM prediction and PortfolioOptimizer output. Features are evaluated only through the selected historical date.")
