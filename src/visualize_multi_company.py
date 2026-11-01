"""
src/visualize_multi_company.py
==============================
Visualizes the multi-asset universe of all 50+ individual NIFTY companies.

Generates `nifty50_portfolio_visualizer.html` containing:
  1. Asset Ingestion & IPO Data Availability (all 50 individual stocks + status).
  2. 42x42 Pairwise Correlation Matrix Heatmap across all surviving stocks.
  3. Normalized Stock Price Trends (Base = 100 on 2005-03-23).
  4. MPT Portfolio Allocation: Max Sharpe vs Min Vol vs 1/N Equal Weight.
"""

from __future__ import annotations

import glob
import os
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

# ── Aesthetic Theme (Dark Mode Institutional Theme) ──────────────────────────
BG       = "#050813"
SURF     = "#0A0F1E"
GRID     = "rgba(79, 142, 247, 0.08)"
FONT_FAM = "Plus Jakarta Sans, -apple-system, sans-serif"
MONO     = "JetBrains Mono, monospace"
C_TEXT   = "#E2E8F0"


def inspect_raw_company_timeline(data_dir: str):
    """Scan all CSV files in data_dir and build metadata on date ranges and missing rates."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    records = []
    _EXCLUDE = {"nifty50_all", "stock_metadata"}

    for fpath in csv_files:
        stem = Path(fpath).stem
        if stem.lower() in _EXCLUDE:
            continue

        try:
            df = pd.read_csv(fpath, usecols=lambda c: c.strip() in ("Date", "Close"))
            df.columns = df.columns.str.strip()
            if "Date" not in df.columns or "Close" not in df.columns or len(df.dropna()) < 2:
                records.append({
                    "ticker": stem,
                    "status": "Skipped (Insufficient Data)",
                    "start": None,
                    "end": None,
                    "rows": 0,
                })
                continue

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", "Close"])
            records.append({
                "ticker": stem,
                "start": df["Date"].min(),
                "end": df["Date"].max(),
                "rows": len(df),
            })
        except Exception:
            records.append({
                "ticker": stem,
                "status": "Error",
                "start": None,
                "end": None,
                "rows": 0,
            })

    return pd.DataFrame(records)


def build_dashboard():
    print("[1/3] Loading cleaned price matrix of all portfolio assets ...")
    loader = LocalDataLoader()
    prices = loader.load_all_portfolio_assets("data/raw")
    opt = PortfolioOptimizer(prices)

    tickers = sorted(prices.columns.tolist())
    returns = prices.pct_change().dropna()
    corr = returns.corr()

    # Get portfolio weights
    ew_weights = opt.get_equal_weights()
    ms_weights = opt.optimize_max_sharpe()
    mv_weights = opt.optimize_min_volatility()

    # Raw timeline info
    print("[2/3] Analyzing raw company files ...")
    timeline_df = inspect_raw_company_timeline("data/raw")

    # Mark survival
    timeline_df["survived"] = timeline_df["ticker"].apply(
        lambda t: "Surviving (Active in Matrix)" if t in tickers
        else ("Skipped (Empty File)" if t == "INFRATEL" else "Dropped (>30% Missing / Late IPO)")
    )
    timeline_df = timeline_df.sort_values(by=["survived", "ticker"], ascending=[False, True])

    print("[3/3] Generating multi-panel interactive Plotly visualization ...")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>1. NIFTY 50 Universe Ingestion & IPO Availability</b>",
            "<b>2. 42x42 Asset Correlation Matrix Heatmap</b>",
            "<b>3. Historical Cumulative Performance of Surviving Stocks (Base=100)</b>",
            "<b>4. MPT Asset Allocation Weights (% by Company)</b>",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.09,
    )

    # ── Panel 1: Data Availability Timeline ──────────────────────────────────
    color_map = {
        "Surviving (Active in Matrix)": "#34D399",
        "Dropped (>30% Missing / Late IPO)": "#FB7185",
        "Skipped (Empty File)": "#64748B",
    }

    for status, group in timeline_df.groupby("survived"):
        valid = group.dropna(subset=["start", "end"])
        fig.add_trace(
            go.Bar(
                y=valid["ticker"],
                x=(valid["end"] - valid["start"]).dt.days / 365.25,
                orientation="h",
                name=status,
                marker=dict(color=color_map.get(status, "#64748B")),
                hovertemplate="<b>%{y}</b><br>Years of Data: %{x:.1f} yrs<br>Status: " + status + "<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Panel 2: 42x42 Correlation Heatmap ───────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmin=-0.2,
            zmax=1.0,
            colorbar=dict(
                title=dict(text="Corr", font=dict(family=MONO, size=10, color=C_TEXT)),
                len=0.42,
                x=1.01,
                y=0.82,
                thickness=12,
                tickfont=dict(color=C_TEXT, family=MONO, size=8),
            ),
            hovertemplate="Asset 1: %{y}<br>Asset 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        ),
        row=1, col=2,
    )

    # ── Panel 3: Normalized Cumulative Performance (Base = 100) ──────────────
    # Highlight top holdings and benchmark average
    norm_prices = (prices / prices.iloc[0]) * 100
    top_picks = ["SHREECEM", "BAJFINANCE", "HINDUNILVR", "RELIANCE", "TCS", "INFY"]

    for col in norm_prices.columns:
        is_top = col in top_picks
        fig.add_trace(
            go.Scatter(
                x=norm_prices.index,
                y=norm_prices[col],
                mode="lines",
                line=dict(
                    width=2.5 if is_top else 0.7,
                    color="#38BDF8" if col == "BAJFINANCE"
                    else ("#34D399" if col == "SHREECEM"
                    else ("#FBBF24" if col == "HINDUNILVR"
                    else ("#A78BFA" if col == "RELIANCE"
                    else ("#F472B6" if col == "TCS" else "rgba(100, 116, 139, 0.25)")))),
                ),
                name=col if is_top else "Other Stocks",
                showlegend=is_top,
                hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Index: %{{y:,.1f}}<extra></extra>",
            ),
            row=2, col=1,
        )

    # ── Panel 4: Asset Allocation by Company ─────────────────────────────────
    active_stocks = [t for t in tickers if ms_weights.get(t, 0) > 0.005 or mv_weights.get(t, 0) > 0.025]
    active_stocks.sort(key=lambda t: max(ms_weights.get(t, 0), mv_weights.get(t, 0)), reverse=True)

    fig.add_trace(
        go.Bar(
            x=active_stocks,
            y=[ms_weights.get(t, 0) * 100 for t in active_stocks],
            name="Max Sharpe",
            marker_color="#34D399",
            hovertemplate="%{x}: %{y:.2f}%<extra>Max Sharpe</extra>",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=active_stocks,
            y=[mv_weights.get(t, 0) * 100 for t in active_stocks],
            name="Min Volatility",
            marker_color="#38BDF8",
            hovertemplate="%{x}: %{y:.2f}%<extra>Min Volatility</extra>",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=active_stocks,
            y=[ew_weights.get(t, 0) * 100 for t in active_stocks],
            name="1/N Equal Weight (2.38%)",
            marker_color="#A78BFA",
            hovertemplate="%{x}: %{y:.2f}%<extra>1/N Baseline</extra>",
        ),
        row=2, col=2,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=SURF,
        font=dict(family=FONT_FAM, color=C_TEXT),
        title=dict(
            text="<b>NIFTY 50 MULTI-ASSET PORTFOLIO ENGINE</b> · 50 Individual Companies Ingestion & MPT Allocation",
            font=dict(family=FONT_FAM, size=18, color="#FFFFFF"),
            x=0.03, y=0.98,
        ),
        height=1000,
        margin=dict(l=60, r=60, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.99,
            font=dict(size=10, family=MONO),
            bgcolor="rgba(10, 15, 30, 0.7)",
            bordercolor="rgba(79, 142, 247, 0.2)",
            borderwidth=1,
        ),
        barmode="group",
    )

    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)

    fig.update_xaxes(title_text="Years of Trading History", row=1, col=1)
    fig.update_yaxes(title_text="Ticker", tickfont=dict(size=7), row=1, col=1)

    fig.update_xaxes(title_text="Asset Ticker", tickangle=-90, tickfont=dict(size=7), row=1, col=2)
    fig.update_yaxes(title_text="Asset Ticker", tickfont=dict(size=7), row=1, col=2)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Indexed Price (Base=100, Log)", type="log", row=2, col=1)

    fig.update_xaxes(title_text="Stock Ticker", tickangle=-35, row=2, col=2)
    fig.update_yaxes(title_text="Allocation Weight (%)", ticksuffix="%", row=2, col=2)

    return fig


if __name__ == "__main__":
    fig = build_dashboard()
    out_file = BASE_DIR / "nifty50_portfolio_visualizer.html"
    fig.write_html(
        str(out_file),
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )
    print(f"\n[SUCCESS] Multi-company visualizer saved to: {out_file}")
