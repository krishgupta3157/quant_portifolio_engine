"""
src/visualize_optimization.py
=============================
Generates an interactive institutional-grade HTML dashboard
(`optimization_report.html`) for the MPT Portfolio Optimization Engine.

Includes 4 interactive Plotly panels:
  1. Efficient Frontier & Capital Allocation Line (with simulated portfolios,
     Max Sharpe, Min Volatility, and 1/N Equal Weight).
  2. Asset Allocation Breakdown (Max Sharpe vs Min Vol vs Equal Weight).
  3. Historical Wealth Growth ($10,000 backtest, 2005 - 2021).
  4. Historical Drawdowns Comparison across crisis periods (2008 GFC, 2020 Covid).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from data_loader import LocalDataLoader
from optimization import PortfolioOptimizer

# ── Color Palette (Dark Mode Institutional Theme) ───────────────────────────
BG        = "#050813"
SURF      = "#0A0F1E"
GRID      = "rgba(79, 142, 247, 0.08)"
FONT_FAM  = "Plus Jakarta Sans, -apple-system, sans-serif"
MONO      = "JetBrains Mono, monospace"

C_MAX_SHARPE = "#34D399"   # emerald green
C_MIN_VOL    = "#38BDF8"   # sky blue
C_EQUAL_WT   = "#A78BFA"   # purple / violet
C_CAL        = "#FBBF24"   # gold
C_MUTED      = "#64748B"
C_TEXT       = "#E2E8F0"


def generate_portfolio_dashboard(
    prices: pd.DataFrame,
    opt: PortfolioOptimizer,
    n_simulations: int = 2500,
) -> go.Figure:
    """Build the 4-panel interactive Plotly portfolio optimization dashboard."""
    returns = prices.pct_change().dropna()
    mu = opt.mu
    S = opt.S
    rf = opt.risk_free_rate

    # 1. Optimal Weights
    ew_weights = opt.get_equal_weights()
    ms_weights = opt.optimize_max_sharpe()
    mv_weights = opt.optimize_min_volatility()

    # Convert to full weight vectors
    tickers = list(prices.columns)
    w_ew = np.array([ew_weights.get(t, 0.0) for t in tickers])
    w_ms = np.array([ms_weights.get(t, 0.0) for t in tickers])
    w_mv = np.array([mv_weights.get(t, 0.0) for t in tickers])

    # Portfolio Stats
    def get_stats(w: np.ndarray):
        ret = float(np.sum(w * mu))
        vol = float(np.sqrt(np.dot(w.T, np.dot(S, w))))
        sharpe = (ret - rf) / vol if vol > 0 else 0
        return ret, vol, sharpe

    r_ew, v_ew, s_ew = get_stats(w_ew)
    r_ms, v_ms, s_ms = get_stats(w_ms)
    r_mv, v_mv, s_mv = get_stats(w_mv)

    # 2. Monte Carlo Simulation for cloud
    np.random.seed(42)
    sim_vols = []
    sim_rets = []
    sim_sharpes = []

    for _ in range(n_simulations):
        # Dirichlet distribution for random positive weights summing to 1
        w_rand = np.random.dirichlet(np.ones(len(tickers)))
        r_p = float(np.sum(w_rand * mu))
        v_p = float(np.sqrt(np.dot(w_rand.T, np.dot(S, w_rand))))
        sh_p = (r_p - rf) / v_p if v_p > 0 else 0
        sim_rets.append(r_p)
        sim_vols.append(v_p)
        sim_sharpes.append(sh_p)

    # 3. Cumulative Growth of $10,000 Backtest
    daily_ew = returns.dot(w_ew)
    daily_ms = returns.dot(w_ms)
    daily_mv = returns.dot(w_mv)

    cum_ew = 10000 * (1 + daily_ew).cumprod()
    cum_ms = 10000 * (1 + daily_ms).cumprod()
    cum_mv = 10000 * (1 + daily_mv).cumprod()

    # 4. Drawdowns
    def get_dd(cum_series: pd.Series):
        peak = cum_series.cummax()
        return (cum_series - peak) / peak * 100

    dd_ew = get_dd(cum_ew)
    dd_ms = get_dd(cum_ms)
    dd_mv = get_dd(cum_mv)

    # ─────────────────────────────────────────────────────────────────────────
    # SUBPLOTS
    # ─────────────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>Efficient Frontier & Capital Allocation Line</b>",
            "<b>Asset Allocation Weights (% of Portfolio)</b>",
            "<b>Historical Wealth Growth ($10,000 Invested)</b>",
            "<b>Peak-to-Trough Drawdown (%)</b>",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # ── Panel 1: Efficient Frontier ──────────────────────────────────────────
    # Simulated portfolios cloud
    fig.add_trace(
        go.Scatter(
            x=sim_vols,
            y=sim_rets,
            mode="markers",
            marker=dict(
                size=4,
                color=sim_sharpes,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text="Sharpe", font=dict(family=MONO, size=11, color=C_TEXT)),
                    len=0.42,
                    x=0.44,
                    y=0.82,
                    thickness=12,
                    tickfont=dict(color=C_TEXT, family=MONO, size=9),
                ),
                opacity=0.35,
            ),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Capital Allocation Line (from rf to Max Sharpe and beyond)
    cal_x = [0, v_ms * 1.35]
    cal_y = [rf, rf + s_ms * (v_ms * 1.35)]
    fig.add_trace(
        go.Scatter(
            x=cal_x,
            y=cal_y,
            mode="lines",
            line=dict(color=C_CAL, width=2, dash="dash"),
            name=f"CAL (Rf={rf*100:.1f}%)",
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra>CAL</extra>",
        ),
        row=1, col=1,
    )

    # Max Sharpe point
    fig.add_trace(
        go.Scatter(
            x=[v_ms],
            y=[r_ms],
            mode="markers+text",
            marker=dict(symbol="star", size=17, color=C_MAX_SHARPE, line=dict(color="#FFFFFF", width=1.5)),
            name=f"Max Sharpe ({s_ms:.2f})",
            text=[" Max Sharpe"],
            textposition="top right",
            textfont=dict(family=MONO, size=11, color=C_MAX_SHARPE),
            hovertemplate="<b>Max Sharpe Portfolio</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: " + f"{s_ms:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Min Vol point
    fig.add_trace(
        go.Scatter(
            x=[v_mv],
            y=[r_mv],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color=C_MIN_VOL, line=dict(color="#FFFFFF", width=1.5)),
            name=f"Min Vol ({v_mv:.2%})",
            text=[" Min Volatility"],
            textposition="bottom right",
            textfont=dict(family=MONO, size=11, color=C_MIN_VOL),
            hovertemplate="<b>Min Volatility Portfolio</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: " + f"{s_mv:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Equal Weight point
    fig.add_trace(
        go.Scatter(
            x=[v_ew],
            y=[r_ew],
            mode="markers+text",
            marker=dict(symbol="circle", size=12, color=C_EQUAL_WT, line=dict(color="#FFFFFF", width=1.5)),
            name=f"Equal Weight (1/{len(tickers)})",
            text=[" 1/N Equal Weight"],
            textposition="top left",
            textfont=dict(family=MONO, size=11, color=C_EQUAL_WT),
            hovertemplate="<b>1/N Equal Weight Portfolio</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: " + f"{s_ew:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Panel 2: Asset Allocation Weights (Bar Chart) ────────────────────────
    # Filter tickers that have at least 1% weight in Max Sharpe or Min Vol
    active_tickers = [
        t for t in tickers if ms_weights.get(t, 0) >= 0.01 or mv_weights.get(t, 0) >= 0.03
    ]
    active_tickers.sort(key=lambda t: max(ms_weights.get(t, 0), mv_weights.get(t, 0)), reverse=True)

    fig.add_trace(
        go.Bar(
            x=active_tickers,
            y=[ms_weights.get(t, 0) * 100 for t in active_tickers],
            name="Max Sharpe",
            marker_color=C_MAX_SHARPE,
            hovertemplate="%{x}: %{y:.2f}%<extra>Max Sharpe</extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=active_tickers,
            y=[mv_weights.get(t, 0) * 100 for t in active_tickers],
            name="Min Volatility",
            marker_color=C_MIN_VOL,
            hovertemplate="%{x}: %{y:.2f}%<extra>Min Volatility</extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=active_tickers,
            y=[ew_weights.get(t, 0) * 100 for t in active_tickers],
            name="Equal Weight",
            marker_color=C_EQUAL_WT,
            hovertemplate="%{x}: %{y:.2f}%<extra>1/N Baseline</extra>",
        ),
        row=1, col=2,
    )

    # ── Panel 3: Wealth Growth ───────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=cum_ms.index,
            y=cum_ms.values,
            mode="lines",
            line=dict(color=C_MAX_SHARPE, width=2.5),
            name="Max Sharpe ($10k)",
            hovertemplate="%{x|%Y-%m-%d}<br>Value: ₹%{y:,.0f}<extra>Max Sharpe</extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cum_mv.index,
            y=cum_mv.values,
            mode="lines",
            line=dict(color=C_MIN_VOL, width=2),
            name="Min Vol ($10k)",
            hovertemplate="%{x|%Y-%m-%d}<br>Value: ₹%{y:,.0f}<extra>Min Vol</extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cum_ew.index,
            y=cum_ew.values,
            mode="lines",
            line=dict(color=C_EQUAL_WT, width=1.8, dash="dot"),
            name="Equal Weight ($10k)",
            hovertemplate="%{x|%Y-%m-%d}<br>Value: ₹%{y:,.0f}<extra>Equal Weight</extra>",
        ),
        row=2, col=1,
    )

    # ── Panel 4: Drawdowns ───────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dd_ms.index,
            y=dd_ms.values,
            mode="lines",
            line=dict(color=C_MAX_SHARPE, width=1.5),
            name="Max Sharpe DD",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra>Max Sharpe DD</extra>",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_mv.index,
            y=dd_mv.values,
            mode="lines",
            fill="tozeroy",
            line=dict(color=C_MIN_VOL, width=1.5),
            name="Min Vol DD",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra>Min Vol DD</extra>",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_ew.index,
            y=dd_ew.values,
            mode="lines",
            line=dict(color=C_EQUAL_WT, width=1.2, dash="dot"),
            name="Equal Weight DD",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra>Equal Weight DD</extra>",
        ),
        row=2, col=2,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT & STYLING
    # ─────────────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=SURF,
        font=dict(family=FONT_FAM, color=C_TEXT),
        title=dict(
            text="<b>PORTFOLIO OPTIMIZATION RISK ENGINE</b> · Modern Portfolio Theory (NIFTY 50 Universe)",
            font=dict(family=FONT_FAM, size=18, color="#FFFFFF"),
            x=0.03, y=0.98,
        ),
        height=940,
        margin=dict(l=55, r=40, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.99,
            font=dict(size=11, family=MONO),
            bgcolor="rgba(10, 15, 30, 0.7)",
            bordercolor="rgba(79, 142, 247, 0.2)",
            borderwidth=1,
        ),
        barmode="group",
    )

    # Format Axes
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)

    # Specialized axes
    fig.update_xaxes(title_text="Annualized Volatility (σ)", tickformat=".1%", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Expected Return (μ)", tickformat=".1%", row=1, col=1)

    fig.update_xaxes(title_text="Surviving Tickers (Top Allocations)", tickangle=-35, row=1, col=2)
    fig.update_yaxes(title_text="Weight (%)", ticksuffix="%", row=1, col=2)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value (₹, Log Scale)", type="log", tickprefix="₹", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=2)

    return fig


if __name__ == "__main__":
    print("[1/2] Loading data and computing optimizations ...")
    loader = LocalDataLoader()
    prices = loader.load_all_portfolio_assets("data/raw")
    opt = PortfolioOptimizer(prices)

    print("[2/2] Generating interactive Plotly figure ...")
    fig = generate_portfolio_dashboard(prices, opt)

    out_file = BASE_DIR / "optimization_report.html"
    fig.write_html(
        str(out_file),
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )
    print(f"\n[DONE] Dashboard saved to: {out_file}")
    print(f"       Total Assets : {opt.n_assets}")
    print(f"       Start Date   : {prices.index.min().date()}")
