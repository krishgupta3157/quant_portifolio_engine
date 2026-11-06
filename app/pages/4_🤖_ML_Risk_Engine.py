"""UI shell for the machine-learning dynamic risk engine."""

from __future__ import annotations

import sys
from pathlib import Path

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


TOP_NIFTY_STOCKS = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"]
AVAILABLE_ASSETS = [
    "RELIANCE",
    "HDFCBANK",
    "INFY",
    "TCS",
    "ICICIBANK",
    "BHARTIARTL",
    "ITC",
    "LT",
    "SBIN",
    "KOTAKBANK",
]

with st.sidebar:
    st.markdown("### Simulation Controls")
    st.caption("Static UI mockup inputs — model wiring is intentionally deferred.")
    selected_assets = st.multiselect(
        "Portfolio selector",
        options=AVAILABLE_ASSETS,
        default=TOP_NIFTY_STOCKS,
        help="Choose the assets to include in the simulated risk scenario.",
    )
    shock_scenario = st.selectbox(
        "Macro shock scenario",
        options=[
            "Normal Market",
            "VIX Spike / Panic",
            "Tech Sector Crash",
            "Interest Rate Hike",
        ],
    )
    risk_tolerance = st.slider(
        "Target maximum annualized volatility",
        min_value=5,
        max_value=40,
        value=20,
        step=1,
        format="%d%%",
    )
    st.divider()
    st.markdown("**Scenario status**")
    st.markdown('<span class="status-pill">UI Mockup Active</span>', unsafe_allow_html=True)
    st.caption(f"{len(selected_assets)} assets selected · {shock_scenario}")

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
  &nbsp;|&nbsp; <b>Status:</b> UI Mockup Phase
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading"><span>01</span> ML Predicted Risk KPIs</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="section-copy">Illustrative outputs for <b>{shock_scenario}</b> with a '
    f'{risk_tolerance}% target volatility ceiling.</div>',
    unsafe_allow_html=True,
)

kpi_columns = st.columns(4, gap="large")
kpis = [
    ("Predicted 30-Day Market Volatility", "18.5%", "Mock forecast · annualized"),
    ("Recommended Portfolio Regime", "DEFENSIVE", "Signal confidence · 82%"),
    ("Predicted Maximum Drawdown", "-12.4%", "Stress-window estimate"),
    ("Contagion Risk Score", "7 / 10", "Cross-asset transmission"),
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
    '<div class="section-copy">This mockup will visualize how a crash in one stock '
    'can affect the risk profile of the remaining assets.</div>',
    unsafe_allow_html=True,
)

assets_for_chart = selected_assets or TOP_NIFTY_STOCKS
mock_correlation = [
    [1.00, .62, .48, .55, .66],
    [.62, 1.00, .41, .58, .72],
    [.48, .41, 1.00, .76, .35],
    [.55, .58, .76, 1.00, .39],
    [.66, .72, .35, .39, 1.00],
]
heatmap_assets = assets_for_chart[:5]
heatmap = go.Figure(
    go.Heatmap(
        z=[row[: len(heatmap_assets)] for row in mock_correlation[: len(heatmap_assets)]],
        x=heatmap_assets,
        y=heatmap_assets,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#0F172A"], [.5, "#256D8F"], [1, "#38BDF8"]],
        texttemplate="%{z:.2f}",
        textfont={"color": "#F8FAFC", "size": 12},
        hovertemplate="%{y} ↔ %{x}<br>Predicted correlation: %{z:.2f}<extra></extra>",
        colorbar={"title": "Shift", "tickfont": {"color": "#94A3B8"}},
    )
)
heatmap.update_layout(
    template="plotly_dark",
    title={"text": "Predicted Asset Correlation Shift under Stress", "x": .02},
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
    '<div class="section-copy">Compare the current portfolio mix with the '
    'illustrative ML-optimized allocation for the selected stress scenario.</div>',
    unsafe_allow_html=True,
)

allocation_assets = assets_for_chart[:5]
current_allocation = [24, 21, 20, 19, 16][: len(allocation_assets)]
optimized_allocation = [18, 27, 24, 16, 15][: len(allocation_assets)]
allocation_chart = go.Figure()
allocation_chart.add_trace(
    go.Bar(
        name="Current Allocation",
        x=allocation_assets,
        y=current_allocation,
        marker_color="#64748B",
        hovertemplate="%{x}<br>Current: %{y}%<extra></extra>",
    )
)
allocation_chart.add_trace(
    go.Bar(
        name="ML Optimized Allocation",
        x=allocation_assets,
        y=optimized_allocation,
        marker_color="#34D399",
        hovertemplate="%{x}<br>Optimized: %{y}%<extra></extra>",
    )
)
allocation_chart.update_layout(
    template="plotly_dark",
    barmode="group",
    title={"text": "Current vs ML Optimized Allocation", "x": .02},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"family": "Inter, sans-serif", "color": "#F8FAFC"},
    xaxis={"showgrid": False, "tickfont": {"color": "#94A3B8"}},
    yaxis={"title": "Portfolio weight (%)", "gridcolor": "#334155", "tickfont": {"color": "#94A3B8"}},
    legend={"font": {"color": "#F8FAFC"}},
    margin={"l": 60, "r": 30, "t": 60, "b": 55},
    height=420,
)
st.plotly_chart(allocation_chart, use_container_width=True, config={"displayModeBar": False})

st.caption("All values and visualizations on this page are static placeholders for the upcoming ML implementation.")
