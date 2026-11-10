"""Shared visual theme for the Streamlit application — Earthy Premium Edition."""

from __future__ import annotations

import streamlit as st


def apply_global_theme() -> None:
    """Apply the shared warm, earthy premium visual system inspired by the design reference."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --qe-bg: #E8E3D7;
  --qe-surface: #DCD5C6;
  --qe-surface-soft: #D4CCBC;
  --qe-border: #C8C0AE;
  --qe-border-strong: #A8A095;
  --qe-text: #222B1C;
  --qe-muted: #7A7568;
  --qe-accent: #222B1C;
  --qe-accent-strong: #151C11;
  --qe-olive-mid: #3B4536;
  --qe-sage: #5C6B52;
  --qe-positive: #3B4536;
  --qe-negative: #8B4B4B;
  --qe-warning: #8B6B2B;
  --qe-white: #FFFFFF;
  --qe-cream: #F5F0E8;
}

*, *::before, *::after { box-sizing: border-box; }

body .stApp,
.stApp {
  background: var(--qe-bg) !important;
  color: var(--qe-text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp::before,
.stApp::after { display: none !important; }

header[data-testid="stHeader"] {
  background: rgba(232, 227, 215, 0.96) !important;
  backdrop-filter: blur(10px) !important;
  border-bottom: 1px solid var(--qe-border) !important;
}
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none !important; }

section.main > div.block-container,
.block-container {
  padding-top: 2rem !important;
  padding-bottom: 3.5rem !important;
}

h1, h2, h3 {
  font-family: 'Playfair Display', Georgia, serif !important;
  color: var(--qe-text) !important;
  letter-spacing: -0.01em !important;
}

h4, h5, h6, p, label,
.stMarkdown, .stText, [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
  color: var(--qe-text) !important;
}

/* ═══════ BUTTONS & PILLS (HIGH-CONTRAST DEEP GREEN + STARK WHITE TEXT) ═══════ */
.stButton > button,
button[kind="primary"],
button[kind="secondary"],
[data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
  background: #222B1C !important;
  border: 1px solid #222B1C !important;
  border-radius: 24px !important;
  box-shadow: none !important;
  color: #FFFFFF !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.74rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  min-height: 2.6rem !important;
  text-transform: uppercase !important;
  transition: transform .18s ease, box-shadow .18s ease, background .18s ease !important;
}

.stButton > button *,
button[kind="primary"] *,
button[kind="secondary"] *,
[data-testid="stFormSubmitButton"] > button *,
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-secondary"] * {
  color: #FFFFFF !important;
  font-weight: 700 !important;
}

.stButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  background: #3B4536 !important;
  border-color: #3B4536 !important;
  box-shadow: 0 4px 14px rgba(34, 43, 28, .22) !important;
  color: #FFFFFF !important;
  transform: translateY(-1px);
}

.stButton > button:hover *,
button[kind="primary"]:hover *,
button[kind="secondary"]:hover *,
[data-testid="stFormSubmitButton"] > button:hover * {
  color: #FFFFFF !important;
}

.stButton > button:focus {
  border-color: #222B1C !important;
  box-shadow: 0 0 0 3px rgba(34, 43, 28, .18) !important;
}

[data-baseweb="tag"] {
  background: #222B1C !important;
  border: 1px solid #222B1C !important;
  border-radius: 16px !important;
  color: #FFFFFF !important;
  font-weight: 700 !important;
}
[data-baseweb="tag"] * {
  color: #FFFFFF !important;
  font-weight: 700 !important;
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
  font-size: .72rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  font-weight: 600 !important;
}

div[data-testid="stMetric"],
div[data-testid="metric-container"],
div[data-testid="stVerticalBlockBorderWrapper"],
.css-1r6slb0, .kpi-card, .form-card, .info-panel {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 16px !important;
  box-shadow: none !important;
  transition: transform .2s ease, border-color .2s ease !important;
}
div[data-testid="stMetric"],
.kpi-card, .form-card, .info-panel {
  padding: 1.5rem !important;
}
div[data-testid="stMetric"]:hover,
div[data-testid="stVerticalBlockBorderWrapper"]:hover,
.kpi-card:hover, .form-card:hover, .info-panel:hover {
  border-color: var(--qe-border-strong) !important;
}
div[data-testid="stMetricLabel"] {
  color: var(--qe-muted) !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  font-size: 0.68rem !important;
  font-weight: 600 !important;
}
div[data-testid="stMetricValue"] {
  color: var(--qe-text) !important;
  font-family: 'Playfair Display', serif !important;
  font-size: 1.5rem !important;
  font-weight: 700 !important;
}
div[data-testid="stMetricDelta"] svg { fill: currentColor !important; }

div[data-testid="stDataFrame"],
div[data-testid="stTable"],
[data-testid="stPlotlyChart"] {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 16px !important;
  box-shadow: none !important;
  overflow: hidden !important;
}

details, div[data-testid="stExpander"] {
  background: var(--qe-surface) !important;
  border: 1px solid var(--qe-border) !important;
  border-radius: 12px !important;
}
button[data-baseweb="tab"] {
  color: var(--qe-muted) !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  font-size: 0.72rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--qe-accent) !important;
  border-bottom: 2px solid var(--qe-accent) !important;
}
div[data-testid="stAlert"] { border-radius: 12px !important; }

.positive, .gain, .up { color: var(--qe-positive) !important; }
.negative, .loss, .down { color: var(--qe-negative) !important; }
.muted { color: var(--qe-muted) !important; }

section[data-testid="stSidebar"] {
  background: var(--qe-surface-soft) !important;
  border-right: 1px solid var(--qe-border) !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h3 {
  color: var(--qe-text) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )
