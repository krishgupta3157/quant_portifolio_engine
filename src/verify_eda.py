"""
src/verify_eda.py
=================
Exploratory Data Analysis — visual integrity verification of the
engineered feature pipeline.

Three-panel interactive Plotly dashboard (dark theme):
  Row 1 — NIFTY 500 price with Crisis regime overlays (Risk_Score ≥ 8)
  Row 2 — Drawdown (%) — verifies peak-to-trough maths
  Row 3 — VIX_Close (line) vs Risk_Score (bar) — verifies educational scaling

Run
---
    python -X utf8 src/verify_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── resolve paths so the script can be run from any cwd ─────────────────────
SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_loader import LocalDataLoader
from features    import FeatureEngineer

# ── palette & shared style constants ────────────────────────────────────────
BG        = "rgba(0,0,0,0)"
SURF      = "rgba(0,0,0,0)"
GRID      = "rgba(71,85,105,0.12)"
FONT_FAM  = "Plus Jakarta Sans, Inter, sans-serif"
MONO      = "JetBrains Mono, Consolas, monospace"

C_PRICE   = "#2563EB"   # deep blue price line
C_CRISIS  = "#FB7185"   # rose  — crisis zones
C_DD      = "#FB7185"   # drawdown fill
C_VIX     = "#FBBF24"   # amber — VIX line
C_RISK    = "#4F8EF7"   # cornflower — risk score bars
C_MUTED   = "#64748B"

# Risk-score colour ramp for the bar chart (1=green … 10=deep red)
_RISK_PALETTE = {
    1:  "#34D399",   # emerald
    2:  "#6EE7B7",
    3:  "#A7F3D0",
    4:  "#FCD34D",
    5:  "#FBBF24",
    6:  "#F97316",
    7:  "#EF4444",
    8:  "#DC2626",
    9:  "#991B1B",
    10: "#7F1D1D",   # deep crimson
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline() -> pd.DataFrame:
    nifty_csv = BASE_DIR / "NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv"
    vix_csv   = BASE_DIR / "INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv"

    for p in (nifty_csv, vix_csv):
        if not p.exists():
            sys.exit(f"[ERROR] File not found: {p}")

    loader  = LocalDataLoader()
    master  = loader.build_master_dataframe(nifty_csv, vix_csv)
    fe      = FeatureEngineer()
    df      = fe.build_features(master)

    # Drop rolling warm-up NaNs (252-day windows -> first ~year)
    df = df.dropna(subset=["NIFTY_MA_252d", "Risk_Score"])
    print(f"[EDA] Dataset ready -- {len(df):,} rows  "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _crisis_shapes(df: pd.DataFrame, threshold: int = 8) -> list[dict]:
    """
    Build Plotly layout shapes that shade every contiguous block of days
    where Risk_Score >= threshold across ALL three rows (y-axis spans 0→1
    in paper coordinates).
    """
    crisis       = df["Risk_Score"] >= threshold
    in_crisis    = False
    start        = None
    shapes: list[dict] = []

    for dt, val in crisis.items():
        if val and not in_crisis:
            in_crisis = True
            start = dt
        elif not val and in_crisis:
            in_crisis = False
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=str(start.date()), x1=str(dt.date()),
                y0=0, y1=1,
                fillcolor="rgba(251,113,133,0.07)",
                line=dict(width=0), layer="below",
            ))

    if in_crisis:                                # close open block at end
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=str(start.date()), x1=str(df.index[-1].date()),
            y0=0, y1=1,
            fillcolor="rgba(251,113,133,0.07)",
            line=dict(width=0), layer="below",
        ))
    return shapes


def _bar_colours(risk_series: pd.Series) -> list[str]:
    """Map Risk_Score integers to hex colours."""
    return [_RISK_PALETTE.get(int(v) if pd.notna(v) else 5, C_MUTED)
            for v in risk_series]


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def build_eda_figure(df: pd.DataFrame) -> go.Figure:

    crisis_df = df[df["Risk_Score"] >= 8]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.25, 0.30],
        vertical_spacing=0.04,
        subplot_titles=[
            "NIFTY 500 — Price & Crisis Regimes  (Risk Score ≥ 8)",
            "Drawdown from All-Time High (%)",
            "India VIX  vs  Educational Risk Score (1–10)",
        ],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
    )

    # ── ROW 1 : NIFTY price + crisis scatter ─────────────────────────────── #
    fig.add_trace(go.Scatter(
        x=df.index, y=df["NIFTY_Close"],
        name="NIFTY 500", mode="lines",
        line=dict(color=C_PRICE, width=1.6),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Close: ₹%{y:,.1f}<extra></extra>",
    ), row=1, col=1)

    # Crisis-day scatter dots on top of price line
    fig.add_trace(go.Scatter(
        x=crisis_df.index,
        y=crisis_df["NIFTY_Close"],
        name="Crisis Day (Score ≥ 8)",
        mode="markers",
        marker=dict(color=C_CRISIS, size=4, symbol="circle",
                    line=dict(width=0)),
        hovertemplate=(
            "<b>CRISIS — %{x|%d %b %Y}</b><br>"
            "Close: ₹%{y:,.1f}<br>"
            "Risk Score: %{customdata}<extra></extra>"
        ),
        customdata=crisis_df["Risk_Score"],
    ), row=1, col=1)

    # ── ROW 2 : Drawdown ─────────────────────────────────────────────────── #
    fig.add_trace(go.Scatter(
        x=df.index, y=df["NIFTY_Drawdown_Pct"],
        name="Drawdown %",
        fill="tozeroy",
        fillcolor="rgba(251,113,133,0.12)",
        line=dict(color=C_DD, width=1.4),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    # Annotate the single worst drawdown
    min_idx = df["NIFTY_Drawdown_Pct"].idxmin()
    min_val = df.loc[min_idx, "NIFTY_Drawdown_Pct"]
    fig.add_annotation(
        x=min_idx, y=min_val,
        text=f"Max DD<br>{min_val:.1f}%",
        showarrow=True, arrowhead=2, arrowcolor=C_CRISIS,
        font=dict(color=C_CRISIS, size=10, family=MONO),
        bgcolor="rgba(251,113,133,0.12)", bordercolor=C_CRISIS,
        borderwidth=1, borderpad=4, ay=-38,
        xref="x3", yref="y3",
    )

    # ── ROW 3 : Risk Score bars (primary y2) ─────────────────────────────── #
    fig.add_trace(go.Bar(
        x=df.index, y=df["Risk_Score"],
        name="Risk Score (1–10)",
        marker=dict(color=_bar_colours(df["Risk_Score"]), line=dict(width=0)),
        opacity=0.85,
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Risk Score: %{y}<extra></extra>"
        ),
        yaxis="y5",          # secondary y on row 3
    ), row=3, col=1, secondary_y=True)

    # VIX line (primary y on row 3)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["VIX_Close"],
        name="India VIX",
        line=dict(color=C_VIX, width=1.8),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>VIX: %{y:.2f}<extra></extra>",
    ), row=3, col=1, secondary_y=False)

    # VIX 30d MA overlay
    fig.add_trace(go.Scatter(
        x=df.index, y=df["VIX_30d_MA"],
        name="VIX 30d MA",
        line=dict(color="#F97316", width=1.2, dash="dot"),
        hovertemplate="VIX MA30: %{y:.2f}<extra></extra>",
    ), row=3, col=1, secondary_y=False)

    # Horizontal reference lines on risk score panel
    for lvl, col, lbl in [
        (8,  "rgba(251,113,133,0.5)",  "Crisis threshold"),
        (5,  "rgba(251,187,36,0.35)",  "Moderate threshold"),
    ]:
        fig.add_hline(
            y=lvl, line=dict(color=col, width=1, dash="dot"),
            annotation_text=lbl,
            annotation_font=dict(color=col, size=9, family=MONO),
            annotation_position="top right",
            row=3, col=1, secondary_y=True,
        )

    # ── Crisis background shading across all rows ─────────────────────────── #
    shapes = _crisis_shapes(df, threshold=8)
    fig.update_layout(shapes=shapes)

    # ── Global layout ────────────────────────────────────────────────────── #
    common_axis = dict(
        showgrid=True, gridcolor=GRID, zeroline=False,
        showspikes=True, spikecolor=C_RISK, spikethickness=1, spikedash="dot",
        tickfont=dict(family=MONO, size=10, color=C_MUTED),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=SURF,
        font=dict(family=FONT_FAM, color=C_MUTED, size=12),
        height=980,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10,15,35,0.92)", bordercolor="rgba(79,142,247,0.3)",
            font=dict(family=MONO, size=11, color="#F8FAFC"),
        ),
        legend=dict(
            bgcolor="rgba(10,15,30,0.85)", bordercolor="rgba(79,142,247,0.18)",
            borderwidth=1, font=dict(size=11, color="#F8FAFC"),
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
        ),
        margin=dict(l=60, r=80, t=100, b=40),
        title=dict(
            text=(
                "<b>NIFTY 500 · EDA Integrity Check</b>"
                "<br><sup style='color:#64748B;font-family:JetBrains Mono'>"
                "Pipeline: LocalDataLoader → FeatureEngineer · "
                "4,023 trading days · 2011→2026"
                "</sup>"
            ),
            font=dict(size=18, color="#F8FAFC", family=FONT_FAM),
            x=0.01,
        ),
    )

    # Shared spike/grid keys without tickfont (avoid duplicate kwarg)
    _grid_spike = {
        "showgrid":      common_axis["showgrid"],
        "gridcolor":     common_axis["gridcolor"],
        "zeroline":      common_axis["zeroline"],
        "showspikes":    common_axis["showspikes"],
        "spikecolor":    common_axis["spikecolor"],
        "spikethickness":common_axis["spikethickness"],
        "spikedash":     common_axis["spikedash"],
    }

    # Per-row axis styling
    fig.update_xaxes(**{**common_axis, "showgrid": False})

    fig.update_yaxes(
        title_text="NIFTY 500 (₹)",
        title_font=dict(size=11, color=C_MUTED, family=MONO),
        tickprefix="₹", tickformat=",.0f",
        **common_axis, row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Drawdown (%)",
        title_font=dict(size=11, color=C_MUTED, family=MONO),
        ticksuffix="%",
        zeroline=True, zerolinecolor="rgba(79,142,247,0.2)", zerolinewidth=1,
        **{k: v for k, v in common_axis.items() if k not in ("zeroline", "tickfont")},
        tickfont=dict(family=MONO, size=10, color=C_MUTED),
        row=2, col=1,
    )
    fig.update_yaxes(
        title_text="VIX",
        title_font=dict(size=11, color=C_VIX, family=MONO),
        tickfont=dict(family=MONO, size=10, color=C_VIX),
        **_grid_spike, row=3, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Risk Score",
        title_font=dict(size=11, color=C_RISK, family=MONO),
        tickfont=dict(family=MONO, size=10, color=C_RISK),
        range=[0, 11], dtick=1,
        showgrid=False, zeroline=False,
        row=3, col=1, secondary_y=True,
    )

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font.update(size=11, color="#94A3B8", family=MONO)
        ann.update(x=0.01, xanchor="left")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRINT SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────
def print_crisis_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("  EDA CRISIS REGIME SUMMARY")
    print("=" * 62)

    crisis = df[df["Risk_Score"] >= 8].copy()
    crisis["year"] = crisis.index.year
    by_year = crisis.groupby("year").size().rename("crisis_days")

    print(f"  Total crisis days (Score >= 8) : {len(crisis):>5,}")
    print(f"  Worst drawdown overall         : "
          f"{df['NIFTY_Drawdown_Pct'].min():.2f}%")
    print(f"  Peak VIX reading               : "
          f"{df['VIX_Close'].max():.2f}")
    print(f"  Median VIX (full history)      : "
          f"{df['VIX_Close'].median():.2f}")
    print()
    print("  Crisis days by year:")
    for yr, cnt in by_year.items():
        bar = "#" * min(cnt, 40)
        print(f"    {yr}  {bar}  {cnt}")

    print()
    print("  Risk Score distribution:")
    dist = (
        df["Risk_Score"]
        .value_counts()
        .sort_index()
        .rename("days")
    )
    for score, days in dist.items():
        from features import _RISK_LABELS
        pct  = days / len(df) * 100
        bar  = "#" * max(1, int(pct))
        lbl  = _RISK_LABELS.get(int(score), "")
        print(f"    [{score:>2}] {lbl:<16} {bar:<42} {days:>4d}d  ({pct:4.1f}%)")

    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df  = load_pipeline()
    print_crisis_summary(df)

    fig = build_eda_figure(df)

    # Save as a portable, self-contained HTML file
    out_path = BASE_DIR / "eda_report.html"
    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",   # ~3 KB stub; loads Plotly from CDN
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )
    print(f"\n[EDA] Chart saved -> {out_path}")
    print(f"      Open this file in any browser to view it interactively.")
