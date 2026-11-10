"""Exploratory data analysis and statistical diagnostics for the asset universe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="EDA & Stats · Quant Engine",
    page_icon="📊",
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
  --cyan: #5C6B52;
  --purple: #3B4536;
  --green: #7A8F6E;
  --red: #8B4B4B;
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

.eda-hero {
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
  padding: .8rem 0 1.5rem;
}

.eda-eyebrow {
  color: var(--olive-mid);
  font-family: 'Inter', sans-serif;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .18em;
  margin-bottom: .55rem;
  text-transform: uppercase;
}

.eda-title {
  color: var(--text);
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(1.8rem, 3.5vw, 2.8rem);
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.1;
  margin: 0 0 .45rem;
}

.eda-subtitle {
  color: var(--muted);
  font-family: 'Inter', sans-serif;
  font-size: .82rem;
  line-height: 1.65;
  margin: 0;
  max-width: 850px;
}

.section-title {
  color: var(--text);
  font-family: 'Playfair Display', serif;
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 1.7rem 0 .8rem;
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

.section-copy {
  color: var(--muted);
  font-size: .82rem;
  line-height: 1.65;
  margin: -.25rem 0 1rem;
}

.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  min-height: 118px;
  padding: 1.05rem 1.15rem;
  transition: border-color .2s ease, transform .2s ease;
}

.kpi-card:hover {
  border-color: var(--olive);
  transform: translateY(-2px);
}

.kpi-label {
  color: var(--muted);
  font-size: .62rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
}

.kpi-value {
  font-family: 'Playfair Display', serif;
  font-size: clamp(1.1rem, 1.55vw, 1.4rem);
  font-weight: 700;
  margin-top: .55rem;
}

.kpi-note {
  color: var(--muted);
  font-family: 'Inter', sans-serif;
  font-size: .65rem;
  margin-top: .25rem;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background: var(--bg-card-sub) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

[data-testid="stPlotlyChart"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: .5rem !important;
}

.insight {
  background: rgba(59, 69, 54, .07);
  border-left: 3px solid var(--olive-mid);
  border-radius: 0 8px 8px 0;
  color: var(--muted);
  font-size: .78rem;
  line-height: 1.65;
  margin: .7rem 0 1rem;
  padding: .7rem .9rem;
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
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the 42-asset price matrix...")
def load_market_data() -> pd.DataFrame:
    """Load and cache the cleaned, aligned close-price matrix."""
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
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices.dropna(axis=1, how="all").dropna(axis=0, how="all")


def plot_layout(title: str, height: int = 480) -> dict[str, object]:
    """Return a consistent earthy Plotly layout for every chart on this page."""
    return {
        "template": "plotly_white",
        "title": {"text": title, "x": 0.02, "xanchor": "left", "font": {"family": "Inter, sans-serif", "size": 13, "color": "#8B8578"}},
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#3B4536"},
        "margin": {"l": 55, "r": 24, "t": 62, "b": 52},
        "hoverlabel": {"bgcolor": "#DCD5C6", "font": {"color": "#3B4536"}},
        "legend": {
            "bgcolor": "rgba(220,213,198,0.9)",
            "bordercolor": "rgba(168,160,149,.35)",
            "borderwidth": 1,
            "font": {"color": "#3B4536"},
        },
        "xaxis": {
            "gridcolor": "#C8C0AE",
            "linecolor": "#A8A095",
            "tickfont": {"color": "#8B8578"},
        },
        "yaxis": {
            "gridcolor": "#C8C0AE",
            "linecolor": "#A8A095",
            "tickfont": {"color": "#8B8578"},
        },
    }


def format_percent(value: float) -> str:
    """Format a finite decimal return as a percentage."""
    return f"{value:.2%}" if np.isfinite(value) else "N/A"


def format_statistic(value: float) -> str:
    """Format a distribution statistic, including small-sample NaN values."""
    return f"{value:.3f}" if np.isfinite(value) else "N/A"


try:
    prices_df = load_market_data()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Unable to load the asset price matrix: {exc}")
    st.stop()

if prices_df.empty or prices_df.shape[1] < 2:
    st.error("At least two valid asset price series are required for EDA.")
    st.stop()

# Invalid and non-positive prices cannot produce meaningful percentage or log
# returns. They are treated as missing before calculating both return series.
valid_prices = prices_df.mask(prices_df <= 0)
pct_returns = valid_prices.pct_change(fill_method=None)
log_returns = np.log(valid_prices / valid_prices.shift(1))
pct_returns = pct_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")

if log_returns.empty or log_returns.shape[1] < 2:
    st.error("Insufficient valid observations to calculate asset returns.")
    st.stop()

asset_returns = log_returns.dropna(axis=1, how="all")
asset_returns = asset_returns.loc[:, asset_returns.count() >= 2]
if asset_returns.shape[1] < 2:
    st.error("At least two assets with two valid return observations are required.")
    st.stop()

assets = sorted(asset_returns.columns.tolist())
available_sectors = sorted({get_sector(asset) for asset in assets})
with st.sidebar:
    st.markdown("### Industry Filters")
    selected_sectors = st.multiselect(
        "Select Industry/Sector",
        options=available_sectors,
        key="eda_sector_filter",
        help="Choose sectors to narrow the ticker universe used by the EDA views.",
    )
sector_assets = [
    asset for asset in assets
    if not selected_sectors or get_sector(asset) in selected_sectors
]
if not sector_assets:
    st.warning("No companies match the selected sectors. Clear the sector filter to restore the full universe.")
    st.stop()
asset_returns = asset_returns.loc[:, sector_assets]
assets = sector_assets
daily_mean = asset_returns.mean()
daily_std = asset_returns.std(ddof=1)
annualized_return = daily_mean * 252
annualized_volatility = daily_std * np.sqrt(252)
market_return = asset_returns.mean(axis=1).dropna()

st.markdown(
    """
<div class="eda-hero">
  <div class="eda-eyebrow">03 / Exploratory Data Analysis &amp; Stats</div>
  <h1 class="eda-title">Measure the market's hidden shape.</h1>
  <p class="eda-subtitle">A statistical control room for return, risk, dependence, distribution shape, and regime changes across the local asset universe.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Macro view / market structure</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Use the universe-level views to identify common risk factors before inspecting an individual asset.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")
with left:
    default_assets = assets[: min(12, len(assets))]
    selected_assets = st.multiselect(
        "Select Assets/Tickers (up to 15 recommended)",
        options=assets,
        default=default_assets,
        key="correlation_assets",
    )
    if len(selected_assets) < 2:
        st.info("Select at least two assets to render the correlation heatmap.")
    else:
        heatmap_returns = asset_returns[selected_assets]
        correlation = heatmap_returns.corr(min_periods=2)
        heatmap = go.Figure(
            go.Heatmap(
                z=correlation.to_numpy(),
                x=correlation.columns,
                y=correlation.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar={"title": "ρ"},
                hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )
        heatmap.update_layout(**plot_layout("Return correlation", height=530))
        heatmap.update_xaxes(tickangle=-45)
        st.plotly_chart(heatmap, use_container_width=True)
    st.markdown(
        '<div class="insight"><b>Why correlation matters:</b> assets with lower correlation tend to react differently to the same shocks. Combining them can reduce portfolio-level volatility, even when each asset is individually risky.</div>',
        unsafe_allow_html=True,
    )

with right:
    sectors = pd.Series(assets, index=assets).map(get_sector)
    sector_palette = {
        "Financial Services": "#3B4536",
        "IT": "#7A8F6E",
        "Energy/Oil & Gas": "#8B6B2B",
        "FMCG": "#5C6B52",
        "Healthcare": "#8B4B4B",
        "Automobile": "#B5A88E",
        "Metals & Mining": "#A8A095",
        "Cement/Construction": "#8B9E7E",
    }
    scatter = go.Figure(
        go.Scatter(
            x=annualized_volatility,
            y=annualized_return,
            mode="markers+text",
            text=assets,
            textposition="top center",
            textfont={"size": 9, "color": "#8B8578"},
            marker={
                "size": 10,
                "color": [sector_palette.get(sector, "#CBD5E1") for sector in sectors],
                "showscale": False,
                "line": {"width": 1, "color": "#C8C0AE"},
            },
            customdata=np.column_stack(
                [
                    annualized_return.to_numpy(),
                    annualized_volatility.to_numpy(),
                    sectors.to_numpy(),
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>Annualized return: %{customdata[0]:.2%}"
                "<br>Annualized volatility: %{customdata[1]:.2%}"
                "<br>Sector: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    for sector, color in sector_palette.items():
        if sector in set(sectors):
            scatter.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=sector,
                    marker={"size": 9, "color": color},
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
    scatter.update_layout(**plot_layout("Risk-return profile", height=530))
    scatter.update_xaxes(title="Annualized volatility", tickformat=".0%")
    scatter.update_yaxes(title="Annualized mean return", tickformat=".0%")
    st.plotly_chart(scatter, use_container_width=True)
    sector_summary = pd.DataFrame(
        {
            "Sector": [get_sector(asset) for asset in assets],
            "Annualized Return": annualized_return.to_numpy(),
            "Annualized Volatility": annualized_volatility.to_numpy(),
        }
    ).groupby("Sector", as_index=False).mean()
    sector_bar = go.Figure()
    sector_bar.add_trace(
        go.Bar(
            x=sector_summary["Sector"],
            y=sector_summary["Annualized Return"],
            name="Average return",
            marker_color="#3B4536",
            hovertemplate="%{x}<br>Average return: %{y:.2%}<extra></extra>",
        )
    )
    sector_bar.add_trace(
        go.Bar(
            x=sector_summary["Sector"],
            y=sector_summary["Annualized Volatility"],
            name="Average volatility",
            marker_color="#B5A88E",
            hovertemplate="%{x}<br>Average volatility: %{y:.2%}<extra></extra>",
        )
    )
    sector_bar.update_layout(barmode="group")
    sector_bar.update_xaxes(tickangle=-35)
    sector_bar.update_yaxes(tickformat=".0%")
    sector_bar.update_layout(**plot_layout("Average Return and Volatility by Sector", height=420))
    st.plotly_chart(sector_bar, use_container_width=True)
    st.markdown(
        '<div class="insight">The upward direction is the familiar risk-reward trade-off: higher expected return may come with greater dispersion. Outliers deserve further investigation rather than automatic selection.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Micro view / individual asset</div>', unsafe_allow_html=True)
selected_asset = st.selectbox("Select an asset for deep analysis", options=assets, index=0)
selected_returns = asset_returns[selected_asset].dropna()

if selected_returns.size < 2:
    st.warning(f"{selected_asset} does not have enough observations for statistical diagnostics.")
    st.stop()

mean_return = float(selected_returns.mean())
volatility = float(selected_returns.std(ddof=1))
variance = float(selected_returns.var(ddof=1))
skewness = float(selected_returns.skew())
kurtosis = float(selected_returns.kurt())

kpis = [
    ("Annualized mean return", format_percent(mean_return * 252), "daily mean × 252", "cyan"),
    ("Annualized volatility / SD", format_percent(volatility * np.sqrt(252)), "daily SD × √252", "purple"),
    ("Daily variance", f"{variance:.6f}", "sample variance", "green"),
    ("Skewness", format_statistic(skewness), "negative = crash risk", "amber"),
    ("Kurtosis", format_statistic(kurtosis), "excess; fat tails / black swans", "red"),
]

kpi_columns = st.columns(len(kpis), gap="small")
for column, (label, value, note, tone) in zip(kpi_columns, kpis):
    with column:
        st.markdown(
            f"""
<div class="kpi-card" style="--accent: var(--{tone});">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color: var(--{tone});">{value}</div>
  <div class="kpi-note">{note}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="insight">Skewness measures asymmetry: a negative value indicates a longer or heavier downside tail. Pandas kurtosis reports excess kurtosis, where values above zero indicate heavier tails than a normal distribution.</div>',
    unsafe_allow_html=True,
)

histogram = go.Figure()
histogram.add_trace(
    go.Histogram(
        x=selected_returns,
        histnorm="probability density",
        nbinsx=60,
        name="Daily log returns",
        marker_color="#3B4536",
        opacity=.72,
        hovertemplate="Return: %{x:.3%}<br>Density: %{y:.3f}<extra></extra>",
    )
)

if np.isfinite(volatility) and volatility > 0:
    curve_x = np.linspace(
        float(selected_returns.min()),
        float(selected_returns.max()),
        300,
    )
    curve_y = (
        np.exp(-0.5 * ((curve_x - mean_return) / volatility) ** 2)
        / (volatility * np.sqrt(2 * np.pi))
    )
    histogram.add_trace(
        go.Scatter(
            x=curve_x,
            y=curve_y,
            mode="lines",
            name="Normal curve",
            line={"color": "#8B6B2B", "width": 2},
            hovertemplate="Normal density: %{y:.3f}<extra></extra>",
        )
    )
histogram.update_layout(**plot_layout(f"{selected_asset} · daily log-return distribution"))
histogram.update_xaxes(title="Daily log return", tickformat=".2%")
histogram.update_yaxes(title="Probability density")

rolling = pd.DataFrame(
    {
        "30-day rolling volatility": selected_returns.rolling(30, min_periods=30).std()
        * np.sqrt(252),
        "90-day rolling volatility": selected_returns.rolling(90, min_periods=90).std()
        * np.sqrt(252),
    }
)
rolling_chart = go.Figure()
rolling_chart.add_trace(
    go.Scatter(
        x=rolling.index,
        y=rolling["30-day rolling volatility"],
        mode="lines",
        name="30-day",
        line={"color": "#3B4536", "width": 2},
        connectgaps=False,
    )
)
rolling_chart.add_trace(
    go.Scatter(
        x=rolling.index,
        y=rolling["90-day rolling volatility"],
        mode="lines",
        name="90-day",
        line={"color": "#B5A88E", "width": 2},
        connectgaps=False,
    )
)
rolling_chart.update_layout(**plot_layout(f"{selected_asset} · rolling annualized volatility"))
rolling_chart.update_xaxes(title="Date")
rolling_chart.update_yaxes(title="Annualized volatility", tickformat=".0%")

box_chart = go.Figure()
box_chart.add_trace(
    go.Box(
        y=selected_returns,
        name=selected_asset,
        marker_color="#3B4536",
        boxmean=True,
        hovertemplate=f"{selected_asset}<br>Return: %{{y:.3%}}<extra></extra>",
    )
)
box_chart.add_trace(
    go.Box(
        y=market_return,
        name="Equal-weight market",
        marker_color="#B5A88E",
        boxmean=True,
        hovertemplate="Equal-weight market<br>Return: %{y:.3%}<extra></extra>",
    )
)
box_chart.update_layout(**plot_layout("Return distribution comparison", height=480))
box_chart.update_yaxes(title="Daily log return", tickformat=".2%")

chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.plotly_chart(histogram, use_container_width=True)
with chart_right:
    st.plotly_chart(rolling_chart, use_container_width=True)
st.plotly_chart(box_chart, use_container_width=True)

st.caption(
    f"Universe: {asset_returns.shape[1]} assets · "
    f"{len(asset_returns):,} trading-day rows · "
    f"{asset_returns.index.min().date()} to {asset_returns.index.max().date()} · "
    "Returns are calculated from cleaned close prices; annualization uses 252 trading days."
)
