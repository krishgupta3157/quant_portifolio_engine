"""
Quantitative Portfolio Risk Engine & Dynamic Asset Allocation Optimizer
Frontend · Streamlit + Plotly · Powered by Real NIFTY 50 Historical Backend
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Engine · NIFTY 50",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & BACKEND IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from data_loader import LocalDataLoader
from optimization import PortfolioOptimizer

# ─────────────────────────────────────────────────────────────────────────────
# CSS + NAVBAR + AMBIENT BLOBS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg:      #050813;
  --surf:    #0A0F1E;
  --bdr:     rgba(79,142,247,0.12);
  --bdr-h:   rgba(79,142,247,0.32);
  --blue:    #4F8EF7;
  --violet:  #A78BFA;
  --green:   #34D399;
  --rose:    #FB7185;
  --amber:   #FBBF24;
  --cyan:    #22D3EE;
  --txt:     #E2E8F0;
  --muted:   #64748B;
  --dim:     #1E293B;
  --nav-h:   58px;
}

*,*::before,*::after { box-sizing: border-box; }

.stApp {
  background: var(--bg) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--txt) !important;
}

/* Dot-grid texture */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(rgba(79,142,247,0.10) 1px, transparent 1px);
  background-size: 32px 32px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 50%, transparent 100%);
}

/* Hide Streamlit menu and footer, but keep header and sidebar toggle interactive */
#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {
  display: none !important;
}

header[data-testid="stHeader"] {
  background: transparent !important;
  height: var(--nav-h) !important;
  z-index: 100000 !important;
  pointer-events: none !important;
  display: flex !important;
  align-items: center !important;
  padding-left: 10px !important;
}

/* Ensure sidebar collapse & expand chevron buttons are always visible & prominent */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button,
button[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
  display: flex !important;
  pointer-events: auto !important;
  z-index: 100002 !important;
  cursor: pointer !important;
}

[data-testid="collapsedControl"] {
  position: fixed !important;
  top: 10px !important;
  left: 12px !important;
  background: rgba(10, 15, 30, 0.95) !important;
  border: 1px solid rgba(79, 142, 247, 0.5) !important;
  border-radius: 8px !important;
  padding: 4px 8px !important;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.7), 0 0 10px rgba(79, 142, 247, 0.3) !important;
  transition: all 0.2s ease !important;
}

[data-testid="collapsedControl"]:hover {
  border-color: #4F8EF7 !important;
  box-shadow: 0 4px 22px rgba(79, 142, 247, 0.5) !important;
  transform: scale(1.05);
}

[data-testid="collapsedControl"] svg {
  color: #4F8EF7 !important;
  stroke: #4F8EF7 !important;
  fill: #4F8EF7 !important;
}

.block-container {
  padding: calc(var(--nav-h) + 1.2rem) 2.2rem 2.5rem 2.2rem !important;
  max-width: 100% !important; position: relative; z-index: 1;
}

/* ── Ambient gradient blobs ── */
.blob { position: fixed; border-radius: 50%; pointer-events: none; z-index: 0; }
.blob-1 {
  width: 700px; height: 700px; top: -220px; left: -180px;
  background: radial-gradient(circle, rgba(79,142,247,0.10) 0%, transparent 65%);
  filter: blur(70px); animation: blobFloat 14s ease-in-out infinite;
}
.blob-2 {
  width: 550px; height: 550px; bottom: 5%; right: -120px;
  background: radial-gradient(circle, rgba(167,139,250,0.08) 0%, transparent 65%);
  filter: blur(70px); animation: blobFloat 18s ease-in-out infinite reverse;
}
.blob-3 {
  width: 420px; height: 420px; top: 38%; left: 42%;
  background: radial-gradient(circle, rgba(52,211,153,0.07) 0%, transparent 65%);
  filter: blur(70px); animation: blobFloat 22s ease-in-out infinite 5s;
}
@keyframes blobFloat {
  0%,100% { transform: translate(0,0) scale(1); }
  33%      { transform: translate(38px,-42px) scale(1.07); }
  66%      { transform: translate(-28px,30px) scale(0.93); }
}

/* ═══════ NAVBAR ═══════ */
.qp-nav {
  position: fixed; top: 0; left: 0; right: 0; height: var(--nav-h);
  z-index: 9999; display: flex; align-items: center; padding: 0 2rem;
  background: rgba(5,8,19,0.88);
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border-bottom: 1px solid var(--bdr);
}
.qp-brand {
  font-size: 0.78rem; font-weight: 800; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--muted); margin-right: 2.5rem;
  flex-shrink: 0; white-space: nowrap;
}
.qp-brand .hi { color: var(--blue); }
.qp-links { display: flex; gap: 0.2rem; flex: 1; }
.nav-btn {
  display: flex; align-items: center; gap: 0.38rem;
  padding: 0.32rem 0.88rem; border-radius: 8px;
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
  color: var(--muted); cursor: pointer; text-decoration: none;
  transition: color 0.18s, background 0.18s, border-color 0.18s;
  border: 1px solid transparent; background: none;
  white-space: nowrap; user-select: none;
}
.nav-btn:hover  { color: var(--txt); background: rgba(79,142,247,0.08); border-color: var(--bdr); }
.nav-btn.active { color: var(--blue); background: rgba(79,142,247,0.13); border-color: rgba(79,142,247,0.28); }
.nav-btn .ni { font-size: 0.8rem; opacity: 0.8; }
.qp-right { margin-left: auto; display: flex; align-items: center; gap: 0.85rem; }
.regime-chip {
  display: flex; align-items: center; gap: 0.42rem;
  padding: 0.22rem 0.72rem; border-radius: 20px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; font-weight: 600;
  background: rgba(52,211,153,0.10); border: 1px solid rgba(52,211,153,0.25);
  color: var(--green); white-space: nowrap;
}
.rdot {
  width: 5px; height: 5px; border-radius: 50%; background: var(--green);
  animation: rdotPulse 1.6s infinite;
}
@keyframes rdotPulse {
  0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }
  60%      { opacity:0.4; box-shadow: 0 0 0 5px rgba(52,211,153,0); }
}

/* ═══════ FLIP CARDS ═══════ */
.flip-wrap {
  perspective: 1200px; height: 184px;
  cursor: pointer; user-select: none;
}
.flip-wrap:active { opacity: 0.92; }
.flip-inner {
  position: relative; width: 100%; height: 100%;
  transform-style: preserve-3d;
  transition: transform 0.65s cubic-bezier(0.35,0.15,0.15,1);
}
.flip-wrap.flipped .flip-inner { transform: rotateY(180deg); }
.flip-face {
  position: absolute; inset: 0;
  backface-visibility: hidden; -webkit-backface-visibility: hidden;
  border-radius: 16px; border: 1px solid var(--bdr);
  transition: border-color 0.25s, box-shadow 0.25s;
}
.flip-front {
  background: linear-gradient(145deg, rgba(10,15,35,0.97) 0%, rgba(6,10,22,0.95) 100%);
  overflow: hidden; padding: 0 1.2rem 1rem 1.2rem;
}
.flip-back {
  background: linear-gradient(145deg, rgba(10,18,48,0.98) 0%, rgba(7,12,32,0.97) 100%);
  transform: rotateY(180deg);
  padding: 1rem 1.2rem; display: flex; flex-direction: column;
  border-color: rgba(79,142,247,0.18);
}
.flip-wrap:hover .flip-face {
  border-color: var(--bdr-h);
  box-shadow: 0 0 24px rgba(79,142,247,0.10), 0 8px 32px rgba(0,0,0,0.45);
}
.fc-bar { height: 2.5px; margin: 0 -1.2rem 0.85rem -1.2rem; }
.fc-row1 { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.08rem; }
.fc-lbl  { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.fc-ico  { font-size: 1.25rem; opacity: 0.2; }
.fc-val  { font-family: 'JetBrains Mono', monospace; font-size: 1.95rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; margin-bottom: 0.38rem; }
.fc-dlt  { font-family: 'JetBrains Mono', monospace; font-size: 0.67rem; color: var(--muted); }
.fc-dlt .up   { color: var(--green); }
.fc-dlt .down { color: var(--rose);  }
.fc-hint { position: absolute; bottom: 0.6rem; right: 0.85rem; font-family: 'JetBrains Mono', monospace; font-size: 0.52rem; color: #334155; letter-spacing: 0.08em; text-transform: uppercase; }
.fc-back-tag  { font-size: 0.57rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.35rem; }
.fc-back-name { font-size: 0.84rem; font-weight: 700; color: var(--txt); margin-bottom: 0.45rem; }
.fc-back-desc { font-size: 0.69rem; color: #94A3B8; line-height: 1.6; flex: 1; }
.fc-back-eq   { font-family: 'JetBrains Mono', monospace; font-size: 0.59rem; color: var(--cyan); background: rgba(34,211,238,0.07); border: 1px solid rgba(34,211,238,0.14); border-radius: 6px; padding: 0.28rem 0.6rem; margin-top: 0.5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Color helpers */
.c-blue   { color: #4F8EF7 !important; }
.c-green  { color: #34D399 !important; }
.c-rose   { color: #FB7185 !important; }
.c-amber  { color: #FBBF24 !important; }
.c-violet { color: #A78BFA !important; }

/* ═══════ SECTION HEADERS & INFO ═══════ */
.sec-anchor { scroll-margin-top: calc(var(--nav-h) + 18px); display: block; }
.sec-hdr { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.8rem; }
.sec-left { display: flex; align-items: center; gap: 0.55rem; flex: 1; }
.sec-pip  { width: 3px; height: 16px; border-radius: 2px; flex-shrink: 0; }
.sec-title { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }
.sec-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; font-weight: 600; padding: 2px 9px; border-radius: 20px; background: rgba(79,142,247,0.09); border: 1px solid rgba(79,142,247,0.2); color: var(--blue); }
.info-btn {
  width: 22px; height: 22px; border-radius: 50%;
  border: 1px solid var(--bdr); background: rgba(79,142,247,0.05);
  color: var(--muted); cursor: pointer; font-size: 0.7rem;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.18s; flex-shrink: 0; padding: 0; font-weight: 700;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
.info-btn:hover { border-color: var(--blue); color: var(--blue); background: rgba(79,142,247,0.1); }

.info-panel {
  background: rgba(8,13,32,0.97); border: 1px solid rgba(79,142,247,0.18);
  border-radius: 12px; padding: 0.9rem 1.15rem; margin-bottom: 0.85rem; display: none;
}
.info-panel.open { display: block; animation: ipSlide 0.22s ease; }
@keyframes ipSlide {
  from { opacity:0; transform:translateY(-7px); }
  to   { opacity:1; transform:translateY(0); }
}
.info-panel h4 { font-size: 0.75rem; font-weight: 800; color: var(--blue); margin: 0 0 0.42rem 0; letter-spacing: 0.04em; }
.info-panel p  { font-size: 0.71rem; color: #94A3B8; line-height: 1.65; margin: 0 0 0.42rem 0; }
.info-panel .eq { font-family: 'JetBrains Mono', monospace; font-size: 0.63rem; color: var(--cyan); background: rgba(34,211,238,0.06); border: 1px solid rgba(34,211,238,0.13); border-radius: 6px; padding: 0.32rem 0.65rem; display: inline-block; margin-top: 0.1rem; }

/* Glow divider */
.glow-div { height: 1px; margin: 0.2rem 0 1.6rem 0; background: linear-gradient(90deg, transparent, rgba(79,142,247,0.3) 30%, rgba(167,139,250,0.15) 65%, transparent); }

/* Hero */
.hero-wrap { padding: 0.3rem 0 0.8rem 0; }
.hero-title { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.025em; background: linear-gradient(115deg, #E2E8F0 0%, #4F8EF7 55%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 0.22rem 0; line-height: 1.15; }
.hero-sub  { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted); margin: 0 0 0.65rem 0; }
.hero-chips { display: flex; gap: 1.2rem; flex-wrap: wrap; }
.hero-chip  { font-family: 'JetBrains Mono', monospace; font-size: 0.63rem; color: var(--muted); display: flex; align-items: center; gap: 0.32rem; }
.hero-chip .cd { width: 5px; height: 5px; border-radius: 50%; }

/* Sidebar and toggle button styles */
.nav-sidebar-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.34rem 0.82rem;
  border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #4F8EF7;
  background: rgba(79, 142, 247, 0.12);
  border: 1px solid rgba(79, 142, 247, 0.35);
  cursor: pointer;
  transition: all 0.18s ease;
  user-select: none;
  white-space: nowrap;
}
.nav-sidebar-btn:hover {
  background: rgba(79, 142, 247, 0.22);
  border-color: #4F8EF7;
  box-shadow: 0 0 12px rgba(79, 142, 247, 0.35);
  transform: translateY(-1px);
}

/* On-page segmented control for time horizon */
[data-testid="stSegmentedControl"] {
  background: rgba(10, 15, 30, 0.75) !important;
  border: 1px solid rgba(79, 142, 247, 0.22) !important;
  border-radius: 10px !important;
  padding: 3px !important;
  gap: 2px !important;
}
[data-testid="stSegmentedControl"] button {
  background: transparent !important;
  color: #94A3B8 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.74rem !important;
  font-weight: 600 !important;
  border: 1px solid transparent !important;
  border-radius: 7px !important;
  padding: 0.38rem 0.9rem !important;
  transition: all 0.18s ease !important;
}
[data-testid="stSegmentedControl"] button:hover {
  color: #E2E8F0 !important;
  background: rgba(79, 142, 247, 0.12) !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-checked="true"],
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] button[aria-selected="true"] {
  background: rgba(79, 142, 247, 0.25) !important;
  color: #FFFFFF !important;
  border-color: rgba(79, 142, 247, 0.5) !important;
  box-shadow: 0 0 12px rgba(79, 142, 247, 0.35) !important;
  font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(5,8,19,0.98) !important; border-right: 1px solid var(--bdr) !important; }
[data-testid="stSidebar"] * { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stMultiSelect [data-baseweb="tag"] { background: rgba(79,142,247,0.14) !important; border: 1px solid rgba(79,142,247,0.28) !important; color: #4F8EF7 !important; }
.stRadio label { color: #E2E8F0 !important; font-size: 0.82rem !important; }
.stSelectbox label, .stDateInput label { color: #94A3B8 !important; font-size: 0.78rem !important; font-weight: 600; }

/* Material dashboard override */
.stApp { background: #E9EDF2 !important; color: #263238 !important; }
.stApp::before { display: none; }
.block-container { padding: 5.5rem 3.25rem 3.5rem !important; max-width: 100% !important; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1218 0%, #14232D 58%, #081017 100%) !important;
  border-right: 2px solid #263E4D !important;
  box-shadow: 5px 0 24px rgba(15, 23, 42, 0.28) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #E5EEF4 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  color: #FFFFFF !important;
  background: linear-gradient(135deg, #075985, #0284C7) !important;
  border: 1px solid #38BDF8 !important;
  border-radius: 10px !important;
  padding: 0.85rem 0.9rem !important;
  margin: 0.2rem 0 1.15rem !important;
  box-shadow: 0 5px 14px rgba(2, 132, 199, 0.30) !important;
  letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] hr {
  border-color: #385363 !important;
  opacity: 0.9 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
  color: #DCEAF2 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="slider"] {
  border-color: #476475 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #1B2D39 !important;
  color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
  background: #075985 !important;
  border: 1px solid #38BDF8 !important;
  color: #FFFFFF !important;
}
.qp-nav {
  background: #FFFFFF;
  border-bottom: 2px solid #CBD5E1;
  box-shadow: 0 3px 12px rgba(15,23,42,0.14);
}
.nav-btn { color: #334155 !important; border: 1px solid transparent !important; }
.nav-btn:hover { color: #075985 !important; background: #E0F2FE !important; border-color: #7DD3FC !important; }
.nav-btn.active { color: #FFFFFF !important; background: #0284C7 !important; border-color: #0369A1 !important; box-shadow: 0 4px 10px rgba(2,132,199,.25); }
.qp-right .regime-chip { border-color: #7DD3FC; background: #E0F2FE; color: #075985; }
.qp-brand, .sec-title, .hero-sub, .hero-chip, .fc-lbl, .fc-dlt { color: #64748B !important; }
.hero-title { background: linear-gradient(115deg, #263238 0%, #0288D1 70%, #00ACC1 100%); -webkit-background-clip: text; }
.hero-sub, .hero-chip { color: #475569 !important; }
.glow-div { background: linear-gradient(90deg, transparent, #94A3B8 35%, #CBD5E1 70%, transparent); }
.sec-hdr { margin-top: 1.8rem; margin-bottom: 1rem; }
.sec-badge { background: #E0F2FE; border-color: #BAE6FD; color: #0288D1; }
.flip-wrap { height: 190px; }
.flip-face {
  border: 1px solid #CBD5E1;
  border-radius: 10px;
  box-shadow: 0 5px 12px rgba(15,23,42,0.14);
}
.flip-front { background: #FFFFFF; }
.flip-back { background: #FFFFFF; border: 1px solid #7DD3FC; }
.flip-wrap:hover .flip-face { border-color: #0284C7; box-shadow: 0 8px 20px rgba(15,23,42,0.20); }
.metric-explanation {
  min-height: 145px;
  padding: 1rem 1.2rem;
  border: 1px solid #0284C7;
  border-radius: 10px;
  background: #E0F2FE;
  box-shadow: 0 5px 12px rgba(15,23,42,0.14);
}
.metric-explanation .fc-back-name { color: #0F172A; }
.metric-explanation .fc-back-desc { color: #334155; }
.info-panel { background: #FFFFFF; border: 1px solid #94A3B8; box-shadow: 0 5px 12px rgba(15,23,42,0.12); }
.info-panel p, .info-panel h4 { color: #475569; }
[data-testid="stPlotlyChart"] {
  background: #F1F5F9;
  border: 1px solid #94A3B8;
  border-radius: 10px;
  padding: 1rem 1rem 0.4rem;
  margin-bottom: 1.75rem;
  box-shadow: 0 5px 12px rgba(15,23,42,0.18);
}
[data-testid="stHorizontalBlock"] { gap: 1.4rem !important; }
[data-testid="stButton"] button {
  border-radius: 8px !important;
  border: 1px solid #0284C7 !important;
  background: #F0F9FF !important;
  color: #075985 !important;
  box-shadow: 0 3px 7px rgba(15,23,42,0.10) !important;
  font-weight: 700 !important;
}
[data-testid="stButton"] button:hover { border-color: #0369A1 !important; background: #E0F2FE !important; }
</style>

<!-- Ambient blobs -->
<div class="blob blob-1"></div>
<div class="blob blob-2"></div>
<div class="blob blob-3"></div>

<!-- ═══ NAVBAR ═══ -->
<nav class="qp-nav">
  <a class="nav-btn active" href="#sec-kpi" style="margin-right:1.3rem;"><span class="ni">☰</span> DASHBOARD</a>
  <div class="qp-brand">QUANT <span class="hi">ENGINE</span></div>
  <div class="qp-links">
    <a class="nav-btn active" href="#sec-kpi"><span class="ni">◈</span>Overview</a>
    <a class="nav-btn" href="#sec-perf"><span class="ni">↗</span>Performance</a>
    <a class="nav-btn" href="#sec-analytics"><span class="ni">◎</span>Efficient Frontier</a>
    <a class="nav-btn" href="#sec-risk"><span class="ni">⚡</span>Risk & Correlation</a>
    <a class="nav-btn" href="#sec-portfolio"><span class="ni">◉</span>Asset Weights</a>
  </div>
  <div class="qp-right">
    <span class="regime-chip">⚙ FILTERS IN SIDEBAR</span>
    <div class="regime-chip"><span class="rdot"></span>BACKEND CONNECTED</div>
  </div>
</nav>

""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA CACHING & INGESTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Ingesting NIFTY 50 multi-asset market data from raw storage...")
def load_market_data() -> pd.DataFrame:
    """Load, clean, and cache the unified price matrix of all surviving assets.
    
    Dynamically scans data/raw (with fallback to companies/), drops tickers with
    >30% missing history, aligns to first simultaneous trading date, and ffills gaps.
    """
    loader = LocalDataLoader()
    raw_dir = BASE_DIR / "data" / "raw"
    if not raw_dir.exists():
        raw_dir = BASE_DIR / "companies"

    prices = loader.load_all_portfolio_assets(str(raw_dir), missing_threshold=0.30)
    return prices


# Load master cached price matrix
try:
    master_prices = load_market_data()
except Exception as exc:
    st.error(f"❌ Failed to load market data: {exc}")
    st.stop()

if master_prices.empty:
    st.error("❌ Master price matrix is empty. Please verify that CSV files exist in data/raw or companies/.")
    st.stop()

all_surviving_tickers = sorted(master_prices.columns.tolist())
master_min_date = master_prices.index.min().date()
master_max_date = master_prices.index.max().date()


# ─────────────────────────────────────────────────────────────────────────────
# 2. HERO PLACEHOLDER & UNIVERSAL TIME HORIZON TOOLBAR
# ─────────────────────────────────────────────────────────────────────────────
hero_placeholder = st.empty()

# Dedicated On-Page Time Horizon Toolbar (rendered directly underneath the Hero title)
HORIZONS = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "Max", "Custom"]

with st.container():
    settings_popover = st.popover("⚙ Manage Companies & Rate", use_container_width=False)
    with settings_popover:
        st.markdown("#### Strategy & Filter Parameters")
        selected_assets = st.multiselect(
            "Companies / Asset Universe",
            options=all_surviving_tickers,
            default=all_surviving_tickers,
            help="Select the companies included in the portfolio calculations.",
            key="sb_universe_select",
        )
        if not selected_assets:
            selected_assets = all_surviving_tickers[:10]
            st.info("Defaulted to top 10 assets because selection was empty.")
        rf_rate_pct = st.slider(
            "Risk-Free Rate (Rf %)",
            min_value=2.0,
            max_value=10.0,
            value=6.5,
            step=0.25,
            help="Annualised risk-free rate used for Sharpe ratio calculation.",
            key="sb_rf_slider",
        )
        st.caption("Changes apply to the portfolio calculations and charts.")
    rf_rate = rf_rate_pct / 100.0

    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin:0.1rem 0 0.5rem 0;flex-wrap:wrap;gap:0.7rem;">
      <div style="display:flex;align-items:center;gap:0.6rem;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;font-weight:700;color:#4F8EF7;letter-spacing:0.06em;background:rgba(79,142,247,0.12);padding:0.25rem 0.65rem;border-radius:6px;border:1px solid rgba(79,142,247,0.3);">
          ⏱ TIME HORIZON
        </span>
        <span style="font-size:0.73rem;color:#94A3B8;">Click any horizon below for instant portfolio recalculation &amp; chart updates:</span>
      </div>
      <span class="nav-btn" style="margin:0;font-size:0.72rem;padding:0.32rem 0.85rem;">⚙ Settings &amp; Universe in sidebar</span>
    </div>
    """, unsafe_allow_html=True)

    time_horizon = st.segmented_control(
        "Time Horizon",
        options=HORIZONS,
        default="Max",
        label_visibility="collapsed",
        key="main_page_horizon"
    ) or "Max"

    custom_start: date | None = None
    custom_end: date | None = None

    if time_horizon == "Custom":
        col_c1, col_c2, col_c3 = st.columns([2, 2, 3])
        with col_c1:
            custom_start = st.date_input(
                "Start Date",
                value=master_min_date,
                min_value=master_min_date,
                max_value=master_max_date,
                key="custom_start_in",
            )
        with col_c2:
            custom_end = st.date_input(
                "End Date",
                value=master_max_date,
                min_value=master_min_date,
                max_value=master_max_date,
                key="custom_end_in",
            )
        with col_c3:
            st.markdown(f"<div style='font-size:0.74rem;color:#94A3B8;padding-top:1.7rem;'>Available History: <b>{master_min_date}</b> → <b>{master_max_date}</b></div>", unsafe_allow_html=True)

        if custom_start and custom_end and custom_start > custom_end:
            st.error("❌ **Invalid Date Range**: Start Date cannot be after End Date.")
            st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 2B. SIDEBAR STRATEGY PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Strategy & Filter Parameters")
    st.markdown("---")
    st.markdown(f"**Selected Time Horizon:** `{time_horizon}`")
    st.markdown("<span style='font-size:0.75rem;color:#94A3B8;'>You can also click any horizon directly in the toolbar on the main dashboard.</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Use **Manage Companies & Rate** at the top-left to update the asset universe and risk-free rate.")
    st.metric("Risk-Free Rate", f"{rf_rate_pct:.2f}%")
    st.metric("Selected Companies", len(selected_assets))

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.62rem;color:#475569;line-height:1.8;font-family:'JetBrains Mono',monospace;">
      <b>PORTFOLIO ENGINE v3.0</b><br>
      Surviving Universe: {len(all_surviving_tickers)} Assets<br>
      Selected Assets: {len(selected_assets)} Assets<br>
      History: {master_min_date} → {master_max_date}<br>
      Total Trading Days: {len(master_prices):,}
    </div>
    <br>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SLICING LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def slice_price_matrix(
    df: pd.DataFrame,
    horizon: str,
    assets: list[str],
    c_start: date | None = None,
    c_end: date | None = None,
) -> pd.DataFrame:
    """Strictly slice cached master price matrix down to the selected date range and asset subset."""
    max_d = df.index.max()
    min_d = df.index.min()

    if horizon == "1 Week":
        start_ts = max_d - pd.Timedelta(days=7)
        end_ts = max_d
    elif horizon == "1 Month":
        start_ts = max_d - pd.DateOffset(months=1)
        end_ts = max_d
    elif horizon == "3 Months":
        start_ts = max_d - pd.DateOffset(months=3)
        end_ts = max_d
    elif horizon == "6 Months":
        start_ts = max_d - pd.DateOffset(months=6)
        end_ts = max_d
    elif horizon == "1 Year":
        start_ts = max_d - pd.DateOffset(years=1)
        end_ts = max_d
    elif horizon == "3 Years":
        start_ts = max_d - pd.DateOffset(years=3)
        end_ts = max_d
    elif horizon == "5 Years":
        start_ts = max_d - pd.DateOffset(years=5)
        end_ts = max_d
    elif horizon == "Custom" and c_start and c_end:
        start_ts = pd.Timestamp(c_start)
        end_ts = pd.Timestamp(c_end)
    else:  # "Max"
        start_ts = min_d
        end_ts = max_d

    # Slice dates
    sub = df.loc[(df.index >= start_ts) & (df.index <= end_ts), assets].copy()

    # Drop columns that have no active prices in this slice
    sub = sub.dropna(axis=1, how="all")
    # Forward-fill any internal gaps
    sub = sub.ffill().bfill()
    return sub


sliced_prices = slice_price_matrix(
    master_prices, time_horizon, selected_assets, custom_start, custom_end
)

if len(sliced_prices) < 5:
    st.error(f"❌ **Insufficient Historical Data**: Only {len(sliced_prices)} trading days found for the selected horizon. Please choose a broader interval (at least 5 trading days are required to calculate returns and covariance).")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DYNAMIC MPT OPTIMIZATION & MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
opt = PortfolioOptimizer(sliced_prices, risk_free_rate=rf_rate, frequency=252)

# Calculate dynamic weights
ms_weights = opt.optimize_max_sharpe()
mv_weights = opt.optimize_min_volatility()
ew_weights = opt.get_equal_weights()

active_tickers = list(sliced_prices.columns)
n_active = len(active_tickers)

# Vectorized weight arrays
w_ms = np.array([ms_weights.get(t, 0.0) for t in active_tickers])
w_mv = np.array([mv_weights.get(t, 0.0) for t in active_tickers])
w_ew = np.array([ew_weights.get(t, 0.0) for t in active_tickers])

# Daily returns of the active universe
daily_rets = sliced_prices.pct_change().dropna()

# Portfolio daily returns series
ret_ms_series = daily_rets.dot(w_ms) if len(daily_rets) > 0 else pd.Series(dtype=float)
ret_mv_series = daily_rets.dot(w_mv) if len(daily_rets) > 0 else pd.Series(dtype=float)
ret_ew_series = daily_rets.dot(w_ew) if len(daily_rets) > 0 else pd.Series(dtype=float)

# Cumulative NAV trajectories (Base $100)
nav_ms = 100.0 * (1.0 + ret_ms_series).cumprod()
nav_mv = 100.0 * (1.0 + ret_mv_series).cumprod()
nav_ew = 100.0 * (1.0 + ret_ew_series).cumprod()


# ── Monte Carlo Simulation for Efficient Frontier ────────────────────────────
@st.cache_data(show_spinner=False)
def simulate_efficient_frontier(
    mu_dict: dict[str, float],
    cov_matrix: pd.DataFrame,
    rf: float,
    n_sims: int = 3500,
) -> pd.DataFrame:
    """Generate Monte Carlo portfolios across the risk-return spectrum."""
    tickers = list(cov_matrix.columns)
    mu_arr = np.array([mu_dict[t] for t in tickers])
    cov_arr = cov_matrix.values
    n = len(tickers)

    np.random.seed(42)
    # Dirichlet distribution ensures all weights >= 0 and sum(w) = 1.0
    W = np.random.dirichlet(np.ones(n), size=n_sims)
    rets = np.dot(W, mu_arr)
    vols = np.sqrt(np.sum(W * np.dot(W, cov_arr), axis=1))
    sharpes = np.where(vols > 0, (rets - rf) / vols, 0.0)

    return pd.DataFrame({"v": vols, "r": rets, "s": sharpes})


# Compute MC cloud from dynamic mu & covariance
frontier_df = simulate_efficient_frontier(
    opt.mu.to_dict(), opt.S, rf_rate, n_sims=3500
)

# Portfolio specific stats
def calc_portfolio_stats(w_vec: np.ndarray, mu_s: pd.Series, cov_df: pd.DataFrame, rf: float):
    ret = float(np.sum(w_vec * mu_s.values))
    vol = float(np.sqrt(np.dot(w_vec.T, np.dot(cov_df.values, w_vec))))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe

ms_ret, ms_vol, ms_sharpe = calc_portfolio_stats(w_ms, opt.mu, opt.S, rf_rate)
mv_ret, mv_vol, mv_sharpe = calc_portfolio_stats(w_mv, opt.mu, opt.S, rf_rate)
ew_ret, ew_vol, ew_sharpe = calc_portfolio_stats(w_ew, opt.mu, opt.S, rf_rate)


# ── KPI Calculations ─────────────────────────────────────────────────────────
n_days = len(ret_ms_series)
days_per_year = 252

# Annualized CAGR
if n_days >= 20:
    years = n_days / days_per_year
    cagr_ms = (nav_ms.iloc[-1] / 100.0) ** (1.0 / max(years, 0.08)) - 1.0
    cagr_ew = (nav_ew.iloc[-1] / 100.0) ** (1.0 / max(years, 0.08)) - 1.0
else:
    cagr_ms = (nav_ms.iloc[-1] / 100.0) - 1.0 if len(nav_ms) > 0 else 0.0
    cagr_ew = (nav_ew.iloc[-1] / 100.0) - 1.0 if len(nav_ew) > 0 else 0.0

# Max Drawdown
cummax_ms = nav_ms.cummax()
dd_ms_series = (nav_ms - cummax_ms) / cummax_ms * 100.0
max_dd_ms = float(dd_ms_series.min()) if len(dd_ms_series) > 0 else 0.0

cummax_ew = nav_ew.cummax()
dd_ew_series = (nav_ew - cummax_ew) / cummax_ew * 100.0
max_dd_ew = float(dd_ew_series.min()) if len(dd_ew_series) > 0 else 0.0

# 95% Daily Historical VaR
var_95_pct = float(np.percentile(ret_ms_series, 5)) if len(ret_ms_series) > 0 else 0.0
var_95_dollar = abs(var_95_pct) * 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#F1F5F9",
    plot_bgcolor="#E8EEF4",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#1E293B", size=12),
    margin=dict(l=15, r=15, t=38, b=15),
    legend=dict(
        bgcolor="#F8FAFC", bordercolor="#94A3B8",
        borderwidth=1, font=dict(size=11, color="#1E293B"),
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
    ),
    xaxis=dict(showgrid=False, zeroline=False, showspikes=True,
               tickfont=dict(color="#334155", size=11),
               title_font=dict(color="#1E293B", size=12),
               spikecolor="#4F8EF7", spikethickness=1, spikedash="dot"),
    yaxis=dict(showgrid=True, gridcolor="#B8C5D1",
               tickfont=dict(color="#334155", size=11),
               title_font=dict(color="#1E293B", size=12),
               zeroline=False, showspikes=True,
               spikecolor="#4F8EF7", spikethickness=1, spikedash="dot"),
)

def apply_layout(fig, h=385, **kw):
    fig.update_layout(**{**PLAYOUT, "height": h, **kw})
    fig.update_xaxes(
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#1E293B", size=12),
        linecolor="#64748B",
        linewidth=1,
    )
    fig.update_yaxes(
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#1E293B", size=12),
        gridcolor="#B8C5D1",
        linecolor="#64748B",
        linewidth=1,
    )
    return fig

def anchor(sec_id):
    st.markdown(f'<div id="{sec_id}" class="sec-anchor"></div>', unsafe_allow_html=True)

def _toggle_session_value(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)

def sec_header(title, badge="", color="#4F8EF7", info_id=None, info_html=""):
    if info_id:
        st.session_state.setdefault(info_id, False)
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="sec-hdr">
      <div class="sec-left">
        <div class="sec-pip" style="background:{color};box-shadow:0 0 8px {color}55;"></div>
        <span class="sec-title">{title}</span>
        {badge_html}
      </div>
    </div>
    """, unsafe_allow_html=True)
    if info_id:
        st.button(
            "ⓘ Details",
            key=f"{info_id}_button",
            on_click=_toggle_session_value,
            args=(info_id,),
        )
        if st.session_state[info_id]:
            st.markdown(f'<div class="info-panel open">{info_html}</div>', unsafe_allow_html=True)

def flip_card(card_id, bar_css, lbl, icon, val_html, val_cls,
              dlt_html, back_tag, back_tag_cls, back_name, desc, eq):
    state_key = f"card_{card_id}_details"
    st.session_state.setdefault(state_key, False)
    st.button(
        "↔ " + ("Show metric" if st.session_state[state_key] else "Explain metric"),
        key=f"{card_id}_toggle",
        on_click=_toggle_session_value,
        args=(state_key,),
        use_container_width=True,
    )
    if st.session_state[state_key]:
        st.markdown(f"""
        <div class="metric-explanation">
          <div class="fc-back-tag {back_tag_cls}">{back_tag}</div>
          <div class="fc-back-name">{back_name}</div>
          <div class="fc-back-desc">{desc}</div>
          <div class="fc-back-eq">{eq}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        icon_colors = {
            "cagr": "#43A047",
            "sharpe": "#1E88E5",
            "maxdd": "#E53935",
            "var": "#FB8C00",
        }
        icon_color = icon_colors.get(card_id, "#0288D1")
        st.markdown(f"""
        <div class="flip-face flip-front" style="position:relative;min-height:145px;padding:0 1.2rem 1rem;">
          <div class="fc-bar" style="background:{bar_css};"></div>
          <div class="fc-row1"><span class="fc-lbl">{lbl}</span><span class="fc-ico" style="background:{icon_color};color:#fff;opacity:1;border-radius:8px;padding:.45rem .55rem;box-shadow:0 5px 10px rgba(15,23,42,.18);">{icon}</span></div>
          <div class="fc-val {val_cls}">{val_html}</div>
          <div class="fc-dlt">{dlt_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
delta_days = (sliced_prices.index.max() - sliced_prices.index.min()).days
start_str = sliced_prices.index.min().strftime('%d %b %Y')
end_str = sliced_prices.index.max().strftime('%d %b %Y')

hero_placeholder.markdown(f"""
<div class="hero-wrap">
  <div class="hero-title">Quantitative Portfolio Risk Engine</div>
  <div class="hero-sub">MODERN PORTFOLIO THEORY &nbsp;·&nbsp; REAL NIFTY 50 MULTI-ASSET ENGINE</div>
  <div class="hero-chips">
    <div class="hero-chip"><div class="cd" style="background:#4F8EF7;"></div>{n_active} Active Assets</div>
    <div class="hero-chip"><div class="cd" style="background:#34D399;"></div>Horizon: {time_horizon} ({n_days:,} Trading Days)</div>
    <div class="hero-chip"><div class="cd" style="background:#A78BFA;"></div>Range: {start_str} → {end_str}</div>
    <div class="hero-chip"><div class="cd" style="background:#FB7185;"></div>Risk-Free Rate: {rf_rate_pct:.1f}%</div>
  </div>
</div>
<div class="glow-div"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROW 1 — REAL PERFORMANCE KPI FLIP CARDS
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-kpi")
sec_header(
    "Performance Snapshot",
    badge=f"REAL DATA · {time_horizon.upper()}",
    color="#4F8EF7",
    info_id="info-kpi",
    info_html="""
<h4>📊 Performance Snapshot (Dynamic Time Horizon)</h4>
<p>Four primary risk-adjusted metrics calculated dynamically on the <b>Max Sharpe Portfolio</b> over your chosen time window. Figures are annualized via standard NSE 252-day convention.</p>
<span class="eq">Base: INR · Confidence: 95% · Currency: Multi-Asset NIFTY 50</span>
""",
)

diff_cagr = (cagr_ms - cagr_ew) * 100.0
diff_sharpe = ms_sharpe - ew_sharpe
diff_dd = max_dd_ms - max_dd_ew

c1, c2, c3, c4 = st.columns(4)

with c1:
    flip_card(
        "cagr",
        bar_css="linear-gradient(90deg,#34D399,#059669)",
        lbl="Max Sharpe CAGR",
        icon="📈",
        val_html=f"{cagr_ms*100:+.1f}%",
        val_cls="c-green" if cagr_ms >= 0 else "c-rose",
        dlt_html=f'<span class="{"up" if diff_cagr >= 0 else "down"}">{"▲" if diff_cagr >= 0 else "▼"} {diff_cagr:+.1f}pp vs Equal Weight</span>',
        back_tag="COMPOUND RETURN",
        back_tag_cls="c-green",
        back_name="Compound Annual Growth Rate",
        desc=f"Annualized compounded return over the selected {time_horizon} horizon. Evaluates the long-term wealth compounding power of the risk-adjusted optimal asset weights.",
        eq="CAGR = (NAV_end / NAV_start)^(252 / N) − 1",
    )

with c2:
    flip_card(
        "sharpe",
        bar_css="linear-gradient(90deg,#4F8EF7,#2563EB)",
        lbl="Sharpe Ratio",
        icon="⚡",
        val_html=f"{ms_sharpe:.2f}",
        val_cls="c-blue",
        dlt_html=f'<span class="{"up" if diff_sharpe >= 0 else "down"}">{"▲" if diff_sharpe >= 0 else "▼"} {diff_sharpe:+.2f} vs Equal Weight</span>',
        back_tag="RISK-ADJUSTED",
        back_tag_cls="c-blue",
        back_name="Sharpe Ratio",
        desc=f"Excess return generated per unit of total annualized volatility above the {rf_rate_pct:.1f}% risk-free rate for this exact time period.",
        eq=f"Sharpe = (R_p − {rf_rate_pct:.2f}%) / σ_p × √252",
    )

with c3:
    flip_card(
        "maxdd",
        bar_css="linear-gradient(90deg,#FB7185,#BE123C)",
        lbl="Max Drawdown",
        icon="📉",
        val_html=f"{max_dd_ms:.1f}%",
        val_cls="c-rose",
        dlt_html=f'<span class="{"up" if diff_dd >= 0 else "down"}">{"▲" if diff_dd >= 0 else "▼"} {abs(diff_dd):.1f}pp vs Equal Weight</span>',
        back_tag="DOWNSIDE RISK",
        back_tag_cls="c-rose",
        back_name="Maximum Drawdown",
        desc="The single deepest peak-to-trough decline experienced by the Max Sharpe portfolio across the selected interval.",
        eq="MDD = min_t [ (NAV_t − max_{τ≤t} NAV_τ) / max NAV ]",
    )

with c4:
    flip_card(
        "var",
        bar_css="linear-gradient(90deg,#FBBF24,#B45309)",
        lbl="Daily 95% VaR",
        icon="🎯",
        val_html=f"₹{var_95_dollar:,.0f}",
        val_cls="c-amber",
        dlt_html=f'<span class="down">▼ {abs(var_95_pct)*100:.2f}% per ₹10L Notional</span>',
        back_tag="TAIL RISK",
        back_tag_cls="c-amber",
        back_name="Historical Value at Risk (95%)",
        desc="The minimum expected loss on the worst 5% of trading days during this time window, scaled to ₹10 Lakhs portfolio notional.",
        eq="VaR₀.₀₅ = Percentile(R_daily, 5%) × Notional",
    )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROW 2 — REAL EQUITY CURVE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-perf")
sec_header(
    f"Equity Curve Comparison ({start_str} → {end_str})",
    badge=f"{len(ret_ms_series):,} TRADING DAYS",
    color="#4F8EF7",
    info_id="info-perf",
    info_html="""
<h4>📈 Equity Curve Comparison (Base ₹100)</h4>
<p>Cumulative compounded performance of the three canonical MPT allocation strategies over your selected time window:</p>
<p>• <b style="color:#34D399;">Max Sharpe Portfolio</b> — Mean-variance optimal tangential allocation<br>
• <b style="color:#38BDF8;">Minimum Volatility Portfolio</b> — Global Minimum Variance (left-most frontier)<br>
• <b style="color:#A78BFA;">Equal Weight (1/N)</b> — Benchmark allocation across all surviving assets</p>
""",
)

fig_main = go.Figure()

fig_main.add_trace(go.Scatter(
    x=nav_ew.index, y=nav_ew.values, name="Equal Weight (1/N)",
    line=dict(color="#A78BFA", width=1.8, dash="dot"),
    hovertemplate="<b>Equal Weight</b><br>%{x|%d %b %Y}<br>NAV: ₹%{y:,.2f}<extra></extra>",
))

fig_main.add_trace(go.Scatter(
    x=nav_mv.index, y=nav_mv.values, name="Min Volatility",
    line=dict(color="#38BDF8", width=2.2),
    hovertemplate="<b>Min Volatility</b><br>%{x|%d %b %Y}<br>NAV: ₹%{y:,.2f}<extra></extra>",
))

fig_main.add_trace(go.Scatter(
    x=nav_ms.index, y=nav_ms.values, name="Max Sharpe Portfolio",
    line=dict(color="#34D399", width=3.2),
    fill="tozeroy", fillcolor="rgba(52,211,153,0.05)",
    hovertemplate="<b>Max Sharpe</b><br>%{x|%d %b %Y}<br>NAV: ₹%{y:,.2f}<extra></extra>",
))

apply_layout(
    fig_main, h=410,
    yaxis=dict(tickprefix="₹", showgrid=True, gridcolor="rgba(79,142,247,0.06)", zeroline=False),
    hovermode="x unified",
)
st.plotly_chart(fig_main, width="stretch", config={"displayModeBar": False})
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROW 3 — MONTE CARLO EFFICIENT FRONTIER + ASSET ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-analytics")
col_ef, col_alloc = st.columns([1.1, 0.9], gap="large")

with col_ef:
    sec_header(
        "Monte Carlo Efficient Frontier",
        badge=f"{len(frontier_df):,} SIMULATIONS",
        color="#A78BFA",
        info_id="info-ef",
        info_html="""
<h4>◎ Dynamic Monte Carlo Efficient Frontier</h4>
<p>Thousands of random Dirichlet-weighted portfolios mapped in risk-return space using the active covariance matrix for the selected time window.</p>
<p>• <b style="color:#34D399;">★ Green Star</b> = Max Sharpe Tangency Portfolio<br>
• <b style="color:#38BDF8;">◆ Blue Diamond</b> = Global Minimum Volatility Portfolio<br>
• <b style="color:#A78BFA;">● Violet Circle</b> = 1/N Equal Weight Baseline<br>
• <b style="color:#FBBF24;">-- Yellow Dashed</b> = Capital Allocation Line (CAL) from Rf</p>
""",
    )

    fig_ef = go.Figure()

    # Cloud of simulated portfolios
    fig_ef.add_trace(go.Scatter(
        x=frontier_df["v"] * 100,
        y=frontier_df["r"] * 100,
        mode="markers",
        marker=dict(
            color=frontier_df["s"],
            colorscale="Viridis",
            size=3.5,
            opacity=0.45,
            colorbar=dict(
                title=dict(text="Sharpe", font=dict(family="JetBrains Mono", size=10, color="#64748B")),
                tickfont=dict(family="JetBrains Mono", size=8, color="#64748B"),
                thickness=8, len=0.6,
            ),
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
    ))

    # Capital Allocation Line (CAL)
    cal_max_vol = max(ms_vol * 1.35, 0.35)
    cal_x = [0.0, cal_max_vol * 100]
    cal_y = [rf_rate * 100, (rf_rate + ms_sharpe * cal_max_vol) * 100]
    fig_ef.add_trace(go.Scatter(
        x=cal_x, y=cal_y,
        mode="lines",
        line=dict(color="#FBBF24", width=1.8, dash="dash"),
        name=f"CAL (Rf={rf_rate_pct:.1f}%)",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>CAL</extra>",
    ))

    # Optimal Markers
    fig_ef.add_trace(go.Scatter(
        x=[ms_vol * 100], y=[ms_ret * 100],
        mode="markers+text",
        marker=dict(symbol="star", size=18, color="#34D399", line=dict(color="#FFFFFF", width=1.5)),
        text=[" Max Sharpe"], textposition="top right",
        textfont=dict(family="JetBrains Mono", size=10, color="#34D399"),
        name=f"Max Sharpe (SR={ms_sharpe:.2f})",
        hovertemplate="<b>Max Sharpe Portfolio</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: " + f"{ms_sharpe:.2f}<extra></extra>",
    ))

    fig_ef.add_trace(go.Scatter(
        x=[mv_vol * 100], y=[mv_ret * 100],
        mode="markers+text",
        marker=dict(symbol="diamond", size=13, color="#38BDF8", line=dict(color="#FFFFFF", width=1.2)),
        text=[" Min Vol"], textposition="bottom right",
        textfont=dict(family="JetBrains Mono", size=10, color="#38BDF8"),
        name=f"Min Vol (σ={mv_vol*100:.1f}%)",
        hovertemplate="<b>Min Volatility Portfolio</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    fig_ef.add_trace(go.Scatter(
        x=[ew_vol * 100], y=[ew_ret * 100],
        mode="markers+text",
        marker=dict(symbol="circle", size=11, color="#A78BFA", line=dict(color="#FFFFFF", width=1.2)),
        text=[" Equal Weight"], textposition="top left",
        textfont=dict(family="JetBrains Mono", size=9, color="#A78BFA"),
        name="Equal Weight (1/N)",
        hovertemplate="<b>1/N Equal Weight</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    apply_layout(
        fig_ef, h=390,
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Expected Return (%)",
        xaxis=dict(ticksuffix="%", showgrid=False),
        yaxis=dict(ticksuffix="%", gridcolor="rgba(79,142,247,0.06)"),
        legend=dict(orientation="v", x=0.02, y=0.98, xanchor="left", yanchor="top", font=dict(size=9.5)),
    )
    st.plotly_chart(fig_ef, width="stretch", config={"displayModeBar": False})

with col_alloc:
    sec_header(
        "Optimal Asset Allocation Breakdown",
        badge=f"{len([k for k, v in ms_weights.items() if v > 0])} ALLOCATED",
        color="#34D399",
        info_id="info-alloc",
        info_html="""
<h4>◉ Dynamic Max Sharpe Weights</h4>
<p>Optimal weight distribution computed by PyPortfolioOpt specifically for the chosen time horizon.</p>
<p>Weights are rounded to 4 decimals, with positions below 0.01% filtered out to avoid transaction dust.</p>
""",
    )

    # Sort allocations for visualization
    top_tickers = [t for t in active_tickers if ms_weights.get(t, 0) > 0.005 or mv_weights.get(t, 0) > 0.03]
    top_tickers.sort(key=lambda t: max(ms_weights.get(t, 0), mv_weights.get(t, 0)), reverse=True)

    if not top_tickers:
        top_tickers = active_tickers[:10]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=top_tickers,
        y=[ms_weights.get(t, 0.0) * 100 for t in top_tickers],
        name="Max Sharpe",
        marker_color="#34D399",
        hovertemplate="<b>%{x}</b><br>Max Sharpe: %{y:.2f}%<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        x=top_tickers,
        y=[mv_weights.get(t, 0.0) * 100 for t in top_tickers],
        name="Min Volatility",
        marker_color="#38BDF8",
        hovertemplate="<b>%{x}</b><br>Min Vol: %{y:.2f}%<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        x=top_tickers,
        y=[ew_weights.get(t, 0.0) * 100 for t in top_tickers],
        name="Equal Weight",
        marker_color="#A78BFA",
        hovertemplate="<b>%{x}</b><br>Equal Weight: %{y:.2f}%<extra></extra>",
    ))

    apply_layout(
        fig_bar, h=390,
        barmode="group",
        yaxis_title="Weight (%)",
        yaxis=dict(ticksuffix="%", gridcolor="rgba(79,142,247,0.06)"),
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=9.5, family="JetBrains Mono")),
        legend=dict(orientation="h", x=1.0, y=1.02, xanchor="right", font=dict(size=9.5)),
    )
    st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ROW 4 — ROLLING RISK & CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-risk")
col_rv, col_corr = st.columns([1.05, 0.95], gap="large")

with col_rv:
    sec_header(
        "Realized Rolling Volatility",
        badge="21D & 63D WINDOWS",
        color="#34D399",
        info_id="info-vol",
        info_html="""
<h4>⚡ Realized Rolling Risk</h4>
<p>Rolling volatility computed directly on the real historical returns of the Max Sharpe portfolio across the selected window:</p>
<p>• <b style="color:#4F8EF7;">21-Day Window</b> — High-responsiveness short-term volatility<br>
• <b style="color:#A78BFA;">63-Day Window</b> — Quarterly structural volatility trend</p>
""",
    )

    roll_window_short = min(21, max(3, len(ret_ms_series) // 4))
    roll_window_long  = min(63, max(5, len(ret_ms_series) // 2))

    vol_21 = ret_ms_series.rolling(roll_window_short).std() * np.sqrt(252) * 100
    vol_63 = ret_ms_series.rolling(roll_window_long).std() * np.sqrt(252) * 100

    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=vol_21.index, y=vol_21.values, name=f"{roll_window_short}d Realized Vol",
        line=dict(color="#4F8EF7", width=2.2),
        hovertemplate="%{x|%d %b %Y}<br>21d Vol: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig_rv.add_trace(go.Scatter(
        x=vol_63.index, y=vol_63.values, name=f"{roll_window_long}d Realized Vol",
        line=dict(color="#A78BFA", width=2.0, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>63d Vol: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig_rv.add_hline(
        y=ms_vol * 100, line=dict(color="#34D399", width=1.5, dash="dot"),
        annotation_text=f"Sample Vol ({ms_vol*100:.1f}%)", annotation_position="right",
        annotation_font=dict(color="#34D399", size=9, family="JetBrains Mono"),
    )

    apply_layout(
        fig_rv, h=325,
        yaxis=dict(ticksuffix="%", gridcolor="rgba(79,142,247,0.06)"),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
    )
    st.plotly_chart(fig_rv, width="stretch", config={"displayModeBar": False})

with col_corr:
    sec_header(
        "Pairwise Correlation Matrix",
        badge="REAL TIME-WINDOW COV",
        color="#22D3EE",
        info_id="info-corr",
        info_html="""
<h4>◈ Cross-Asset Pearson Correlation</h4>
<p>Pairwise return correlations for the primary stocks in your selection, computed across this specific time slice.</p>
""",
    )

    # Pick top 12 assets to keep heatmap clean and readable
    corr_assets = top_tickers[:12] if len(top_tickers) >= 4 else active_tickers[:12]
    corr_matrix = daily_rets[corr_assets].corr()

    fig_hm = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale=[[0, "#FB7185"], [0.35, "#3B1B26"], [0.5, "#0A0F1E"], [0.75, "#1E3A5F"], [1, "#4F8EF7"]],
        zmin=-0.3, zmax=1.0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=8, family="JetBrains Mono", color="#E2E8F0"),
        hovertemplate="%{y} vs %{x}<br>Correlation: <b>%{z:.3f}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="ρ", font=dict(size=10, color="#64748B")),
            tickfont=dict(family="JetBrains Mono", size=8, color="#64748B"),
            thickness=8, len=0.8,
        ),
    ))

    apply_layout(
        fig_hm, h=325,
        xaxis=dict(showgrid=False, tickfont=dict(size=8.5, family="JetBrains Mono"), tickangle=-45),
        yaxis=dict(showgrid=False, tickfont=dict(size=8.5, family="JetBrains Mono")),
        margin=dict(l=5, r=40, t=35, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig_hm, width="stretch", config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ROW 5 — ALLOCATION DONUT + HISTORICAL DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-portfolio")
col_donut, col_dd = st.columns([0.85, 1.15], gap="large")

with col_donut:
    sec_header(
        "Max Sharpe Allocation Composition",
        badge="ACTIVE HOLDINGS",
        color="#34D399",
        info_id="info-pie",
        info_html="""
<h4>◉ Portfolio Allocation Composition</h4>
<p>Proportional allocation of capital among surviving stocks under the Max Sharpe objective function.</p>
""",
    )

    non_zero_alloc = {k: v for k, v in ms_weights.items() if v > 0.001}
    if not non_zero_alloc:
        non_zero_alloc = {t: round(1.0 / len(active_tickers), 4) for t in active_tickers[:8]}

    donut_df = pd.DataFrame(list(non_zero_alloc.items()), columns=["ticker", "weight"]).sort_values("weight", ascending=False)

    PALETTE = ["#34D399", "#4F8EF7", "#FBBF24", "#FB7185", "#A78BFA",
               "#22D3EE", "#F97316", "#E879F9", "#38BDF8", "#84CC16", "#FCD34D", "#6EE7B7"]

    fig_pie = go.Figure(go.Pie(
        labels=donut_df["ticker"],
        values=donut_df["weight"],
        hole=0.62,
        marker=dict(colors=PALETTE[:len(donut_df)], line=dict(color="#050813", width=2)),
        textinfo="label+percent",
        textfont=dict(family="JetBrains Mono", size=9.5, color="#E2E8F0"),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        direction="clockwise",
        sort=True,
    ))

    fig_pie.add_annotation(
        text=f"<b>{len(donut_df)}</b><br><span style='font-size:9px;color:#64748B'>STOCKS</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="JetBrains Mono", size=19, color="#E2E8F0"),
    )

    apply_layout(fig_pie, h=330, showlegend=False, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

with col_dd:
    sec_header(
        "Historical Underwater Drawdown (%)",
        badge=f"MAX DD: {max_dd_ms:.1f}%",
        color="#FB7185",
        info_id="info-dd",
        info_html="""
<h4>📉 Peak-to-Trough Drawdown Analysis</h4>
<p>Measures the continuous drawdown percentage from prior high-water marks for the Max Sharpe portfolio across this specific time horizon.</p>
""",
    )

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_ms_series.index, y=dd_ms_series.values,
        fill="tozeroy", fillcolor="rgba(251,113,133,0.08)",
        line=dict(color="#FB7185", width=1.8), name="Max Sharpe DD",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: <b>%{y:.2f}%</b><extra></extra>",
    ))

    if len(dd_ms_series) > 0:
        min_idx = dd_ms_series.idxmin()
        fig_dd.add_annotation(
            x=min_idx, y=dd_ms_series.loc[min_idx],
            text=f"Trough: {max_dd_ms:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor="#FB7185",
            font=dict(color="#FB7185", size=9.5, family="JetBrains Mono"),
            bgcolor="rgba(251,113,133,0.12)", bordercolor="#FB7185",
            borderwidth=1, borderpad=4, ay=-35,
        )

    apply_layout(
        fig_dd, h=330,
        yaxis=dict(ticksuffix="%", gridcolor="rgba(79,142,247,0.06)", zeroline=True, zerolinecolor="#1E293B", zerolinewidth=1),
        xaxis=dict(showgrid=False),
        showlegend=False,
        hovermode="x",
    )
    st.plotly_chart(fig_dd, width="stretch", config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glow-div" style="margin-top:2.5rem;"></div>
<div style="display:flex;justify-content:space-between;align-items:center;padding:0 0 1.2rem 0;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#475569;">
    Quantitative Portfolio Risk Engine &nbsp;·&nbsp; Real Backend Connected (PyPortfolioOpt) &nbsp;·&nbsp; NIFTY 50 Universe
  </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#475569;">
    Streamlit + Plotly &nbsp;·&nbsp; IDSE Quant Lab © 2026
  </span>
</div>
""", unsafe_allow_html=True)
