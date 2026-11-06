"""Shared visual theme for the Streamlit application."""

from __future__ import annotations

import streamlit as st


def apply_global_theme() -> None:
    """Apply the shared high-contrast slate visual system."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --qe-bg: #0F172A;
  --qe-surface: #1E293B;
  --qe-surface-soft: #172033;
  --qe-border: #334155;
  --qe-border-strong: #475569;
  --qe-text: #F8FAFC;
  --qe-muted: #94A3B8;
  --qe-accent: #38BDF8;
  --qe-accent-strong: #0EA5E9;
  --qe-positive: #34D399;
  --qe-negative: #F87171;
  --qe-warning: #FBBF24;
}

*, *::before, *::after { box-sizing: border-box; }

body .stApp,
.stApp {
  background: var(--qe-bg) !important;
  color: var(--qe-text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp::before { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none !important; }

section.main > div.block-container,
.block-container {
  padding-top: 2rem !important;
  padding-bottom: 3rem !important;
}

h1, h2, h3, h4, h5, h6, p, label,
.stMarkdown, .stText, [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
  color: var(--qe-text) !important;
}

.stButton > button,
button[kind="primary"] {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  color: var(--qe-text) !important;
  font-weight: 600 !important;
  min-height: 2.45rem !important;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
}
.stButton > button:hover,
button[kind="primary"]:hover {
  border-color: var(--qe-accent) !important;
  box-shadow: 0 8px 20px rgba(56, 189, 248, .08) !important;
  transform: translateY(-2px);
}
.stButton > button:focus {
  border-color: var(--qe-accent) !important;
  box-shadow: 0 0 0 3px rgba(56, 189, 248, .18) !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea, input {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 10px !important;
  color: var(--qe-text) !important;
}
label, .stTextInput label, .stSelectbox label, .stNumberInput label,
.stDateInput label, .stSlider label {
  color: var(--qe-muted) !important;
  font-size: .78rem !important;
}

div[data-testid="stMetric"],
div[data-testid="metric-container"],
div[data-testid="stVerticalBlockBorderWrapper"],
.css-1r6slb0, .kpi-card, .form-card, .info-panel {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease !important;
}
div[data-testid="stMetric"],
.kpi-card, .form-card, .info-panel {
  padding: 1.75rem !important;
}
div[data-testid="stMetric"]:hover,
div[data-testid="stVerticalBlockBorderWrapper"]:hover,
.kpi-card:hover, .form-card:hover, .info-panel:hover {
  border-color: var(--qe-accent) !important;
  box-shadow: 0 8px 20px rgba(56, 189, 248, .08) !important;
  transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] { color: var(--qe-muted) !important; }
div[data-testid="stMetricValue"] { color: var(--qe-text) !important; }
div[data-testid="stMetricDelta"] svg { fill: currentColor !important; }

div[data-testid="stDataFrame"],
div[data-testid="stTable"],
[data-testid="stPlotlyChart"] {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  overflow: hidden !important;
}

details, div[data-testid="stExpander"] {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 12px !important;
}
button[data-baseweb="tab"] { color: var(--qe-muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--qe-accent) !important; }
div[data-testid="stAlert"] { border-radius: 12px !important; }

.positive, .gain, .up { color: var(--qe-positive) !important; }
.negative, .loss, .down { color: var(--qe-negative) !important; }
.muted { color: var(--qe-muted) !important; }
</style>
        """,
        unsafe_allow_html=True,
    )
