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
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & BACKEND IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.theme import apply_global_theme
from src.data_loader import LocalDataLoader
from src.optimization import PortfolioOptimizer
from src.sector_mapping import get_sector
apply_global_theme()

# ─────────────────────────────────────────────────────────────────────────────
# 1B. DRAWER STATE & SLIDE-OUT MECHANICS
# ─────────────────────────────────────────────────────────────────────────────
if "show_settings_panel" not in st.session_state:
    st.session_state.show_settings_panel = False

def _open_settings_panel():
    st.session_state.show_settings_panel = True

def _close_settings_panel():
    st.session_state.show_settings_panel = False

is_panel_open = bool(st.session_state.get("show_settings_panel", False))
panel_transform = "translateX(0)" if is_panel_open else "translateX(-105%)"
backdrop_css = "opacity: 1; visibility: visible; pointer-events: auto;" if is_panel_open else "opacity: 0; visibility: hidden; pointer-events: none;"

# ─────────────────────────────────────────────────────────────────────────────
# CSS + NAVBAR + MODERN SLEEK DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
  --bg-app:     #E8E3D7;
  --bg-card:    #DCD5C6;
  --bg-card-sub:#D4CCBC;
  --bdr:        #C8C0AE;
  --bdr-hover:  #A8A095;
  --primary:    #222B1C;
  --primary-bg: rgba(34, 43, 28, 0.10);
  --primary-dk: #151C11;
  --txt-dark:   #3B4536;
  --txt-body:   #3B4536;
  --txt-muted:  #8B8578;
  --green:      #5C6B52;
  --green-bg:   rgba(92, 107, 82, 0.15);
  --rose:       #8B4B4B;
  --rose-bg:    rgba(139, 75, 75, 0.15);
  --amber:      #8B6B2B;
  --amber-bg:   rgba(139, 107, 43, 0.15);
  --nav-h:      58px;
}}

*, *::before, *::after {{ box-sizing: border-box; }}

/* ── Global Background & Typography ── */
.stApp {{
  background: var(--bg-app) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--txt-body) !important;
}}

/* Hide default Streamlit clutter */
#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
  display: none !important;
}}

header[data-testid="stHeader"] {{
  background: transparent !important;
  height: var(--nav-h) !important;
  z-index: 9999 !important;
  pointer-events: none !important;
  display: flex !important;
  align-items: center !important;
  padding-left: 12px !important;
}}

/* Hide native Streamlit sidebar collapse control buttons */
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"] {{
  display: none !important;
}}

/* Main content spacing & breathing room */
section.main,
.block-container,
div[data-testid="stAppViewContainer"] > section.main {{
  margin-left: 0 !important;
  margin-right: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  padding: calc(var(--nav-h) + 1.8rem) 3.2rem 4rem 3.2rem !important;
  transition: none !important;
}}

/* ═══════ CLEAN PRO TOP NAVBAR ═══════ */
.qp-nav {{
  position: fixed; top: 0; left: 0; right: 0; height: var(--nav-h);
  z-index: 9998; display: flex; align-items: center; padding: 0 2rem;
  background: #3B4536;
  border-bottom: 1px solid #5C6B52;
  box-shadow: 0 2px 12px rgba(59, 69, 54, 0.2);
}}
.qp-brand {{
  font-family: 'Playfair Display', serif;
  font-size: 0.95rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #F5F0E8; margin-left: 1rem; margin-right: 2.5rem;
  flex-shrink: 0; white-space: nowrap; display: flex; align-items: center; gap: 0.5rem;
}}
.qp-brand .hi {{ color: #DCD5C6; font-weight: 400; font-style: italic; }}
.qp-links {{
  display: flex; gap: 0.6rem; align-items: center; flex: 1;
}}
.nav-btn {{
  display: flex; align-items: center; gap: 0.45rem;
  padding: 0.42rem 1.05rem; border-radius: 20px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #D4CCBC; text-decoration: none;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid transparent; background: transparent;
  white-space: nowrap; user-select: none;
}}
.nav-btn:hover {{
  color: #F5F0E8; background: rgba(245, 240, 232, 0.12);
  border-color: #5C6B52;
  transform: translateY(-1px);
}}
.nav-btn.active {{
  color: #F5F0E8 !important;
  background: #5C6B52 !important;
  border-color: #7A8F6E !important;
}}
.nav-btn .ni {{ font-size: 0.82rem; opacity: 0.9; }}

.simulator-nav-button {{
  position: fixed !important;
  top: 9px;
  right: 2rem;
  z-index: 10001;
}}

.simulator-nav-button [data-testid="stButton"] button {{
  padding: 0.42rem 1.05rem !important;
  background: #222B1C !important;
  border: 1px solid #222B1C !important;
  border-radius: 20px !important;
  color: #FFFFFF !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}}

.simulator-nav-button [data-testid="stButton"] button * {{
  color: #FFFFFF !important;
  font-weight: 700 !important;
}}

.simulator-nav-button [data-testid="stButton"] button:hover {{
  background: #3B4536 !important;
  border-color: #3B4536 !important;
  color: #FFFFFF !important;
  transform: translateY(-1px) !important;
}}

.simulator-nav-button [data-testid="stButton"] button:hover * {{
  color: #FFFFFF !important;
}}

/* ═══════ OFF-CANVAS SETTINGS DRAWER ═══════ */
section[data-testid="stSidebar"],
div[data-testid="stSidebarContent"],
div[data-testid="stSidebarUserContent"] {{
  background-color: #D4CCBC !important;
  color: #3B4536 !important;
}}

section[data-testid="stSidebar"] {{
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  bottom: 0 !important;
  height: 100vh !important;
  width: 480px !important;
  max-width: 90vw !important;
  z-index: 1000000 !important;
  background-color: #D4CCBC !important;
  border-right: 1px solid #C8C0AE !important;
  box-shadow: 16px 0 50px rgba(59, 69, 54, 0.25) !important;
  transform: {panel_transform} !important;
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  overflow-y: auto !important;
  display: block !important;
  visibility: visible !important;
}}

div[data-testid="stSidebarContent"] {{
  padding: 1.4rem 1.4rem 2.5rem 1.4rem !important;
}}

[data-testid="stSidebar"] * {{
  font-family: 'Inter', sans-serif !important;
}}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] b,
[data-testid="stSidebar"] h3 {{
  color: #3B4536 !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] > div {{
  background: #E8E3D7 !important;
  border: 1px solid #C8C0AE !important;
  border-radius: 8px !important;
  color: #3B4536 !important;
}}

[data-testid="stSidebar"] [data-baseweb="tag"],
[data-baseweb="tag"] {{
  background: #222B1C !important;
  border: 1px solid #222B1C !important;
  color: #FFFFFF !important;
  border-radius: 16px !important;
  font-weight: 700 !important;
  font-size: 0.74rem !important;
}}

[data-testid="stSidebar"] [data-baseweb="tag"] *,
[data-baseweb="tag"] * {{
  color: #FFFFFF !important;
  font-weight: 700 !important;
}}

/* BaseWeb Dropdown Menus */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {{
  background-color: #DCD5C6 !important;
  border: 1px solid #C8C0AE !important;
  color: #3B4536 !important;
}}

li[role="option"] {{
  background-color: #DCD5C6 !important;
  color: #3B4536 !important;
}}

li[role="option"]:hover,
li[aria-selected="true"] {{
  background-color: #D4CCBC !important;
  color: #3B4536 !important;
}}

/* Backdrop overlay */
.settings-backdrop {{
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(59, 69, 54, 0.45);
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  z-index: 999990;
  {backdrop_css}
  transition: opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1), visibility 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* ═══════ BUTTONS (HIGH-CONTRAST DEEP GREEN + STARK WHITE TEXT) ═══════ */
[data-testid="stButton"] button,
[data-testid="stSidebar"] [data-testid="stButton"] button {{
  border-radius: 24px !important;
  border: 1px solid #222B1C !important;
  background: #222B1C !important;
  color: #FFFFFF !important;
  font-weight: 700 !important;
  font-size: 0.74rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.45rem 1.1rem !important;
  box-shadow: none !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  white-space: nowrap !important;
  word-break: keep-all !important;
}}

[data-testid="stButton"] button *,
[data-testid="stSidebar"] [data-testid="stButton"] button * {{
  color: #FFFFFF !important;
  font-weight: 700 !important;
}}

[data-testid="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
  border-color: #3B4536 !important;
  background: #3B4536 !important;
  color: #FFFFFF !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(34, 43, 28, 0.22) !important;
}}

[data-testid="stButton"] button:hover *,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover * {{
  color: #FFFFFF !important;
}}

/* ═══════ TIME HORIZON TOOLBAR ═══════ */
[data-testid="stSegmentedControl"] {{
  background: #DCD5C6 !important;
  border: 1px solid #C8C0AE !important;
  border-radius: 24px !important;
  padding: 4px !important;
  gap: 4px !important;
}}

[data-testid="stSegmentedControl"] button {{
  background: transparent !important;
  color: #8B8578 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  border: 1px solid transparent !important;
  border-radius: 20px !important;
  padding: 0.42rem 0.95rem !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

[data-testid="stSegmentedControl"] button:hover {{
  color: #3B4536 !important;
  background: rgba(59, 69, 54, 0.08) !important;
}}

[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-checked="true"],
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] button[aria-selected="true"] {{
  background: #222B1C !important;
  color: #FFFFFF !important;
  border-color: #222B1C !important;
  font-weight: 700 !important;
}}

/* ═══════ FLIP CARDS & METRIC WIDGETS ═══════ */
.flip-wrap {{
  height: 195px;
}}

.flip-face {{
  border: 1px solid #C8C0AE;
  border-radius: 16px;
  background: #DCD5C6;
  transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}}

.flip-wrap:hover .flip-face {{
  border-color: #3B4536;
  box-shadow: 0 6px 20px rgba(59, 69, 54, 0.15);
}}

.flip-front {{
  position: relative;
  min-height: 150px;
  padding: 0 1.3rem 1.1rem;
}}

.fc-bar {{
  height: 3px;
  margin: 0 -1.3rem 0.9rem -1.3rem;
  border-radius: 16px 16px 0 0;
}}

.fc-row1 {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.15rem;
}}

.fc-lbl {{
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--txt-muted);
}}

.fc-ico {{
  font-size: 1.15rem;
}}

.fc-val {{
  font-family: 'Playfair Display', serif;
  font-size: 2.05rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: #3B4536;
  margin-bottom: 0.4rem;
}}

.fc-dlt {{
  font-family: 'Inter', sans-serif;
  font-size: 0.7rem;
  color: var(--txt-muted);
}}

.fc-dlt .up   {{ color: #5C6B52; font-weight: 700; }}
.fc-dlt .down {{ color: #8B4B4B; font-weight: 700; }}

.metric-explanation {{
  min-height: 150px;
  padding: 1.1rem 1.3rem;
  border: 1px solid #C8C0AE;
  border-radius: 16px;
  background: #D4CCBC;
  display: flex;
  flex-direction: column;
}}

.fc-back-tag {{
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}}

.fc-back-name {{
  font-family: 'Playfair Display', serif;
  font-size: 0.9rem;
  font-weight: 700;
  color: #3B4536;
  margin-bottom: 0.4rem;
}}

.fc-back-desc {{
  font-size: 0.72rem;
  color: #8B8578;
  line-height: 1.6;
  flex: 1;
}}

.fc-back-eq {{
  font-family: 'Inter', monospace;
  font-size: 0.62rem;
  color: #3B4536;
  background: #E8E3D7;
  border: 1px solid #C8C0AE;
  border-radius: 6px;
  padding: 0.3rem 0.65rem;
  margin-top: 0.5rem;
}}

/* Color helpers */
.c-blue   {{ color: #3B4536 !important; }}
.c-green  {{ color: #5C6B52 !important; }}
.c-rose   {{ color: #8B4B4B !important; }}
.c-amber  {{ color: #8B6B2B !important; }}

/* ═══════ SECTION HEADERS & DETAILS PANEL ═══════ */
.sec-anchor {{
  scroll-margin-top: calc(var(--nav-h) + 24px);
  display: block;
}}

.sec-hdr {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 2.4rem;
  margin-bottom: 1.15rem;
}}

.sec-left {{
  display: flex;
  align-items: center;
  gap: 0.65rem;
  flex: 1;
}}

.sec-pip {{
  width: 4px;
  height: 18px;
  border-radius: 2px;
  flex-shrink: 0;
  background: #5C6B52;
}}

.sec-title {{
  font-family: 'Playfair Display', serif;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: #3B4536;
}}

.sec-badge {{
  font-family: 'Inter', sans-serif;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 20px;
  background: rgba(59, 69, 54, 0.1);
  border: 1px solid rgba(59, 69, 54, 0.25);
  color: #5C6B52;
}}

.info-panel {{
  background: #DCD5C6;
  border: 1px solid #C8C0AE;
  border-radius: 16px;
  padding: 1.1rem 1.35rem;
  margin-bottom: 1.25rem;
  animation: ipSlide 0.22s ease;
}}

@keyframes ipSlide {{
  from {{ opacity: 0; transform: translateY(-6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

.info-panel h4 {{
  font-family: 'Playfair Display', serif;
  font-size: 0.95rem;
  font-weight: 700;
  color: #3B4536;
  margin: 0 0 0.45rem 0;
}}

.info-panel p {{
  font-size: 0.74rem;
  color: #8B8578;
  line-height: 1.65;
  margin: 0 0 0.45rem 0;
}}

.info-panel .eq {{
  font-family: 'Inter', monospace;
  font-size: 0.65rem;
  color: #3B4536;
  background: #E8E3D7;
  border: 1px solid #C8C0AE;
  border-radius: 6px;
  padding: 0.35rem 0.7rem;
  display: inline-block;
  margin-top: 0.15rem;
}}

/* ═══════ PLOTLY CHARTS & CONTAINERS ═══════ */
[data-testid="stPlotlyChart"] {{
  background: #DCD5C6 !important;
  border: 1px solid #C8C0AE !important;
  border-radius: 16px !important;
  padding: 1.25rem 1.25rem 0.5rem !important;
  margin-bottom: 2rem !important;
}}

[data-testid="stHorizontalBlock"] {{
  gap: 1.8rem !important;
}}

/* Dividers & Hero */
.glow-div {{
  height: 1px;
  margin: 1.5rem 0;
  background: #C8C0AE;
}}

.hero-wrap {{
  padding: 0.4rem 0 0.9rem 0;
}}

.hero-title {{
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(1.8rem, 3.5vw, 2.8rem);
  font-weight: 700;
  letter-spacing: -0.025em;
  color: #3B4536;
  margin: 0 0 0.25rem 0;
  line-height: 1.2;
}}

.hero-sub {{
  font-family: 'Inter', sans-serif;
  font-size: 0.82rem;
  color: #8B8578;
  margin: 0 0 0.75rem 0;
}}

.hero-chips {{
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}}

.hero-chip {{
  font-family: 'Inter', sans-serif;
  font-size: 0.68rem;
  font-weight: 600;
  color: #3B4536;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  background: #DCD5C6;
  padding: 0.3rem 0.7rem;
  border-radius: 20px;
  border: 1px solid #C8C0AE;
}}

.hero-chip .cd {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
}}

/* Sidebar Active Badge Chips */
.asset-badge {{
  display: inline-block;
  background: #E8E3D7;
  color: #3B4536;
  border: 1px solid #C8C0AE;
  border-radius: 4px;
  padding: 2px 6px;
  margin: 2px;
  font-family: 'Inter', sans-serif;
  font-size: 0.68rem;
  font-weight: 600;
}}
</style>

<!-- ═══ CLEAN PRO TOP NAVBAR ═══ -->
<nav class="qp-nav">
  <div class="qp-brand">QUANT <span class="hi">ENGINE</span></div>
  <div class="qp-links">
    <a class="nav-btn active" href="#sec-kpi"><span class="ni">◈</span>Overview</a>
    <a class="nav-btn" href="#sec-perf"><span class="ni">↗</span>Performance</a>
    <a class="nav-btn" href="#sec-analytics"><span class="ni">◎</span>Efficient Frontier</a>
    <a class="nav-btn" href="#sec-risk">Risk & Correlation</a>
    <a class="nav-btn" href="#sec-portfolio"><span class="ni">◉</span>Asset Weights</a>
  </div>
</nav>

<!-- ═══ SEMI-TRANSPARENT SETTINGS BACKDROP ═══ -->
<div class="settings-backdrop" onclick="const b = Array.from(document.querySelectorAll('button')).find(el => el.innerText.includes('Close') || el.innerText.includes('✖')); if(b) b.click();"></div>
""", unsafe_allow_html=True)

with st.container(key="simulator_nav_button"):
    if st.button("Historical Simulator", key="nav_historical_simulator"):
        st.switch_page("pages/1_Historical_Simulator.py")


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
    st.error(f"Failed to load market data: {exc}")
    st.stop()

if master_prices.empty:
    st.error("Master price matrix is empty. Please verify that CSV files exist in data/raw or companies/.")
    st.stop()

all_surviving_tickers = sorted(master_prices.columns.tolist())
master_min_date = master_prices.index.min().date()
master_max_date = master_prices.index.max().date()


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE INITIALIZATION & PRESET CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
if "sb_universe_select" not in st.session_state or not st.session_state.get("sb_universe_select"):
    st.session_state["sb_universe_select"] = all_surviving_tickers.copy()

if "sb_rf_slider" not in st.session_state:
    st.session_state["sb_rf_slider"] = 6.5

def _preset_select_all():
    st.session_state["sb_universe_select"] = all_surviving_tickers.copy()

def _preset_select_top10():
    preferred_top10 = [
        "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS",
        "KOTAKBANK", "HINDUNILVR", "ITC", "LT", "AXISBANK"
    ]
    matched = [t for t in preferred_top10 if t in all_surviving_tickers]
    if len(matched) < 10:
        remainder = [t for t in all_surviving_tickers if t not in matched]
        matched.extend(remainder[: 10 - len(matched)])
    st.session_state["sb_universe_select"] = sorted(matched)

def _preset_select_top5():
    preferred_top5 = ["RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS"]
    matched = [t for t in preferred_top5 if t in all_surviving_tickers]
    if len(matched) < 5:
        remainder = [t for t in all_surviving_tickers if t not in matched]
        matched.extend(remainder[: 5 - len(matched)])
    st.session_state["sb_universe_select"] = sorted(matched)

def _preset_reset_default():
    st.session_state["sb_universe_select"] = all_surviving_tickers.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR (OVERLAY SETTINGS PANEL FOR COMPANIES & RISK-FREE RATE)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    col_head, col_close = st.columns([3.2, 1.4])
    with col_head:
        st.markdown("""
        <div style="padding-top:2px;">
          <h3 style="margin:0;font-size:1.1rem;font-weight:800;color:#FFFFFF;letter-spacing:-0.01em;white-space:nowrap;">
            Settings Panel
          </h3>
          <p style="margin:2px 0 0 0;font-size:0.74rem;color:#94A3B8;">
            Asset Universe &amp; Risk Benchmark
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col_close:
        st.button("✖ Close", key="btn_close_panel", on_click=_close_settings_panel, width="stretch")

    st.markdown("<div style='height:1px;background:#3A3F4D;margin:0.6rem 0 0.9rem 0;'></div>", unsafe_allow_html=True)

    st.markdown("<b style='font-size:0.8rem;color:#FFFFFF;letter-spacing:0.02em;'>Quick Universe Presets</b>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.button("✦ All (42 Stocks)", on_click=_preset_select_all, width="stretch", key="sb_btn_all")
        st.button("✦ Top 5 Core", on_click=_preset_select_top5, width="stretch", key="sb_btn_top5")
    with col_p2:
        st.button("✦ Top 10 Leaders", on_click=_preset_select_top10, width="stretch", key="sb_btn_top10")
        st.button("↺ Reset Universe", on_click=_preset_reset_default, width="stretch", key="sb_btn_reset")

    st.markdown("<div style='margin-top:0.9rem;'><b style='font-size:0.8rem;color:#FFFFFF;letter-spacing:0.02em;'>Select / Deselect Individual Companies</b></div>", unsafe_allow_html=True)
    selected_assets = st.multiselect(
        "Select / Deselect Individual Companies",
        options=all_surviving_tickers,
        key="sb_universe_select",
        label_visibility="collapsed",
        placeholder="Search or select companies...",
    )

    st.markdown(
        f"<div style='margin-top:1.1rem;display:flex;justify-content:space-between;align-items:center;'>"
        f"<b style='font-size:0.8rem;color:#FFFFFF;letter-spacing:0.02em;'>Risk-Free Rate (Rf %)</b>"
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.82rem;font-weight:700;color:#10B981;'>{float(st.session_state.get('sb_rf_slider', 6.5)):.2f}%</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    rf_rate_pct = st.slider(
        "Risk-Free Rate (Rf %)",
        min_value=2.0,
        max_value=10.0,
        step=0.25,
        label_visibility="collapsed",
        key="sb_rf_slider",
    )
    rf_rate = rf_rate_pct / 100.0

    n_sel = len(selected_assets)
    n_tot = len(all_surviving_tickers)
    pct_active = (n_sel / n_tot) * 100.0 if n_tot > 0 else 0.0

    st.markdown(f"""
    <div style="background:#1A1D24;border:1px solid #3A3F4D;border-radius:10px;padding:0.75rem 0.9rem;margin:1.1rem 0 0.5rem 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
        <span style="font-size:0.7rem;font-weight:700;color:#38BDF8;letter-spacing:0.08em;text-transform:uppercase;">Portfolio Universe</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:700;color:#FFFFFF;">{n_sel} / {n_tot} ({pct_active:.0f}%)</span>
      </div>
      <div style="background:#2E3440;border-radius:4px;height:5px;width:100%;overflow:hidden;">
        <div style="background:linear-gradient(90deg,#0284C7,#38BDF8);height:100%;width:{pct_active:.1f}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if n_sel > 0:
        with st.expander(f"View Active Holdings ({n_sel})", expanded=False):
            chips_html = "".join([f'<span class="asset-badge">{t}</span>' for t in selected_assets])
            st.markdown(f'<div style="max-height:160px;overflow-y:auto;line-height:1.9;">{chips_html}</div>', unsafe_allow_html=True)
    else:
        st.error("No companies selected! Please choose at least 2 companies or click Top 10.")


# ─────────────────────────────────────────────────────────────────────────────
# 3B. HERO PLACEHOLDER & DASHBOARD TOOLBAR WITH TOGGLE BUTTON
# ─────────────────────────────────────────────────────────────────────────────
hero_placeholder = st.empty()

# Dedicated On-Page Time Horizon Toolbar with Dashboard Toggle Button
HORIZONS = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "Max", "Custom"]

with st.container():
    col_tbar1, col_tbar2, col_tbar3 = st.columns([1.5, 1.2, 1.2])
    with col_tbar1:
        st.button("Configure Universe & Risk", key="btn_open_panel", on_click=_open_settings_panel, width="stretch")
    with col_tbar2:
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:700;color:#38BDF8;background:#252830;border:1px solid #3A3F4D;border-radius:8px;padding:0.46rem 0.9rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.25);">
          Active: <b>{n_sel}</b> / {n_tot} ({pct_active:.0f}%)
        </div>
        """, unsafe_allow_html=True)
    with col_tbar3:
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:700;color:#10B981;background:#252830;border:1px solid #3A3F4D;border-radius:8px;padding:0.46rem 0.9rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.25);">
          Risk-Free: <b>{rf_rate_pct:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin:0.8rem 0 0.5rem 0;flex-wrap:wrap;gap:0.7rem;">
      <div style="display:flex;align-items:center;gap:0.65rem;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;font-weight:700;color:#38BDF8;letter-spacing:0.06em;background:rgba(56,189,248,0.15);padding:0.28rem 0.7rem;border-radius:6px;border:1px solid #0284C7;">
          ⏱ TIME HORIZON
        </span>
        <span style="font-size:0.75rem;color:#94A3B8;font-weight:500;">Select an investment window for dynamic portfolio rebalancing:</span>
      </div>
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
            st.error("**Invalid Date Range**: Start Date cannot be after End Date.")
            st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 3C. EDGE CASE HANDLING FOR SELECTION
# ─────────────────────────────────────────────────────────────────────────────
if n_sel == 0:
    st.markdown("""
    <div id="sec-kpi"></div>
    <div id="sec-perf"></div>
    <div id="sec-analytics"></div>
    <div id="sec-risk"></div>
    <div id="sec-portfolio"></div>
    <div style="background:#252830;border:2px dashed #F43F5E;border-radius:14px;padding:2.5rem 2rem;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.4);max-width:850px;margin:2rem auto;">
      <h3 style="font-size:1.3rem;font-weight:800;color:#F43F5E;margin:0 0 0.4rem 0;">No Companies Selected</h3>
      <p style="font-size:0.86rem;color:#94A3B8;max-width:560px;margin:0 auto 1.5rem auto;line-height:1.6;">
        Quantitative portfolio optimization, covariance estimation, and risk analytics require active assets. Open the settings drawer or select a preset below to instantly build your portfolio:
      </p>
    </div>
    """, unsafe_allow_html=True)
    col_ec1, col_ec2, col_ec3 = st.columns([1.2, 1.2, 1.2])
    with col_ec1:
        st.button("Open Universe Settings", on_click=_open_settings_panel, width="stretch", key="ec_btn_open")
    with col_ec2:
        st.button("✦ Load Top 10 Core Leaders", on_click=_preset_select_top10, width="stretch", key="ec_btn_top10")
    with col_ec3:
        st.button("✦ Load Full Universe (42)", on_click=_preset_select_all, width="stretch", key="ec_btn_all")
    st.stop()

if n_sel == 1:
    single_name = selected_assets[0]
    st.markdown(f"""
    <div id="sec-kpi"></div>
    <div id="sec-perf"></div>
    <div id="sec-analytics"></div>
    <div id="sec-risk"></div>
    <div id="sec-portfolio"></div>
    <div style="background:#252830;border:2px dashed #F59E0B;border-radius:14px;padding:2.5rem 2rem;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.4);max-width:850px;margin:2rem auto;">
      <h3 style="font-size:1.3rem;font-weight:800;color:#F59E0B;margin:0 0 0.4rem 0;">Only 1 Asset Selected ({single_name})</h3>
      <p style="font-size:0.86rem;color:#94A3B8;max-width:560px;margin:0 auto 1.5rem auto;line-height:1.6;">
        Modern Portfolio Theory optimization (Sharpe ratio maximization, minimum variance, and pairwise cross-asset correlation) mathematically requires at least <b>2 distinct assets</b> to construct a non-singular covariance matrix.
      </p>
    </div>
    """, unsafe_allow_html=True)
    col_ec1, col_ec2, col_ec3 = st.columns([1.2, 1.2, 1.2])
    with col_ec1:
        st.button("Open Universe Settings", on_click=_open_settings_panel, width="stretch", key="ec_btn_open_1")
    with col_ec2:
        st.button("✦ Add Top 5 Core Mega-Caps", on_click=_preset_select_top5, width="stretch", key="ec_btn_top5_1")
    with col_ec3:
        st.button("✦ Load Full Universe (42)", on_click=_preset_select_all, width="stretch", key="ec_btn_all_1")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SLICING LOGIC
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
    # Forward-fill only from prior observations and remove leading gaps.
    sub = sub.ffill().dropna(how="any")
    return sub


sliced_prices = slice_price_matrix(
    master_prices, time_horizon, selected_assets, custom_start, custom_end
)

if len(sliced_prices) < 5:
    st.error(f"**Insufficient Historical Data**: Only {len(sliced_prices)} trading days found for the selected horizon. Please choose a broader interval (at least 5 trading days are required to calculate returns and covariance).")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 5. DYNAMIC MPT OPTIMIZATION & MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running rolling-origin portfolio backtest...")
def run_rolling_backtest(
    prices: pd.DataFrame,
    risk_free_rate: float,
    lookback: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], int]:
    """Cache the rolling-origin backtest for the selected market window."""
    optimizer = PortfolioOptimizer(
        prices,
        risk_free_rate=risk_free_rate,
        frequency=252,
    )
    return optimizer.rolling_backtest(lookback)


backtest_lookback = min(252, len(sliced_prices) - 1)
rolling_returns, rolling_weights, _ = run_rolling_backtest(
    sliced_prices,
    rf_rate,
    backtest_lookback,
)

# Use only the most recent completed training window for allocation/frontier
# statistics; the final day's return is excluded from parameter estimation.
opt = PortfolioOptimizer(
    sliced_prices.iloc[-backtest_lookback - 1:-1],
    risk_free_rate=rf_rate,
    frequency=252,
)

ms_weights = rolling_weights["Max Sharpe"]
mv_weights = rolling_weights["Min Volatility"]
ew_weights = rolling_weights["Equal Weight"]

active_tickers = list(sliced_prices.columns)
n_active = len(active_tickers)

# Vectorized weight arrays
w_ms = np.array([ms_weights.get(t, 0.0) for t in active_tickers])
w_mv = np.array([mv_weights.get(t, 0.0) for t in active_tickers])
w_ew = np.array([ew_weights.get(t, 0.0) for t in active_tickers])

# Daily returns are used for descriptive correlation views, not optimized
# portfolio performance.
daily_rets = sliced_prices.pct_change().dropna()

# Each optimized return uses weights estimated strictly before that date.
ret_ms_series = rolling_returns["Max Sharpe"]
ret_mv_series = rolling_returns["Min Volatility"]
ret_ew_series = rolling_returns["Equal Weight"]

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
# 6. SHARED PLOTLY THEME (MODERN ELEVATED DARK SURFACES)
# ─────────────────────────────────────────────────────────────────────────────
PLAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#3B4536", size=12),
    margin=dict(l=18, r=18, t=42, b=18),
    legend=dict(
        bgcolor="rgba(220, 213, 198, 0.95)", bordercolor="rgba(168, 160, 149, .35)",
        borderwidth=1, font=dict(size=11, color="#3B4536"),
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
    ),
    xaxis=dict(showgrid=False, zeroline=False, showspikes=True,
               tickfont=dict(color="#8B8578", size=11),
               title_font=dict(color="#3B4536", size=12, family="Inter"),
               spikecolor="#5C6B52", spikethickness=1, spikedash="dot"),
    yaxis=dict(showgrid=True, gridcolor="#C8C0AE",
               tickfont=dict(color="#8B8578", size=11),
               title_font=dict(color="#3B4536", size=12, family="Inter"),
               zeroline=False, showspikes=True,
               spikecolor="#5C6B52", spikethickness=1, spikedash="dot"),
)

def apply_layout(fig, h=390, **kw):
    fig.update_layout(**{**PLAYOUT, "height": h, **kw})
    fig.update_xaxes(
        tickfont=dict(color="#8B8578", size=11),
        title_font=dict(color="#3B4536", size=12),
        linecolor="#A8A095",
        linewidth=1,
    )
    fig.update_yaxes(
        tickfont=dict(color="#8B8578", size=11),
        title_font=dict(color="#3B4536", size=12),
        gridcolor="#C8C0AE",
        linecolor="#A8A095",
        linewidth=1,
    )
    return fig

def anchor(sec_id):
    st.markdown(f'<div id="{sec_id}" class="sec-anchor"></div>', unsafe_allow_html=True)

def _toggle_session_value(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)

def sec_header(title, badge="", color="#0284C7", info_id=None, info_html=""):
    if info_id:
        st.session_state.setdefault(info_id, False)
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="sec-hdr">
      <div class="sec-left">
        <div class="sec-pip" style="background:{color};box-shadow:0 0 10px {color}55;"></div>
        <span class="sec-title">{title}</span>
        {badge_html}
      </div>
    </div>
    """, unsafe_allow_html=True)
    if info_id:
        st.button(
            "ⓘ Metric Details & Formula" if not st.session_state[info_id] else "✕ Close Details",
            key=f"{info_id}_button",
            on_click=_toggle_session_value,
            args=(info_id,),
        )
        if st.session_state[info_id]:
            st.markdown(f'<div class="info-panel">{info_html}</div>', unsafe_allow_html=True)

def flip_card(card_id, bar_css, lbl, val_html, val_cls,
              dlt_html, back_tag, back_tag_cls, back_name, desc, eq):
    state_key = f"card_{card_id}_details"
    st.session_state.setdefault(state_key, False)
    
    is_explained = st.session_state[state_key]
    st.button(
        "↔ " + ("Show Metric Value" if is_explained else "Explain Metric Formula"),
        key=f"{card_id}_toggle",
        on_click=_toggle_session_value,
        args=(state_key,),
        width="stretch",
    )
    if is_explained:
        st.markdown(f"""
        <div class="metric-explanation">
          <div class="fc-back-tag {back_tag_cls}">{back_tag}</div>
          <div class="fc-back-name">{back_name}</div>
          <div class="fc-back-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="flip-face flip-front">
          <div class="fc-bar" style="background:{bar_css};"></div>
          <div class="fc-row1">
            <span class="fc-lbl">{lbl}</span>
          </div>
          <div class="fc-val {val_cls}">{val_html}</div>
          <div class="fc-dlt">{dlt_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
start_str = sliced_prices.index.min().strftime('%d %b %Y')
end_str = sliced_prices.index.max().strftime('%d %b %Y')

hero_placeholder.markdown(f"""
<div class="hero-wrap">
  <div class="hero-title">Quantitative Portfolio Risk Engine</div>
  <div class="hero-sub">MODERN PORTFOLIO THEORY &nbsp;·&nbsp; REAL NIFTY 50 MULTI-ASSET ENGINE</div>
  <div class="hero-chips">
    <div class="hero-chip"><div class="cd" style="background:#38BDF8;"></div><b>{n_active}</b> Active Assets</div>
    <div class="hero-chip"><div class="cd" style="background:#10B981;"></div>Horizon: <b>{time_horizon}</b> ({n_days:,} Trading Days)</div>
    <div class="hero-chip"><div class="cd" style="background:#A855F7;"></div>History: <b>{start_str}</b> → <b>{end_str}</b></div>
    <div class="hero-chip"><div class="cd" style="background:#F43F5E;"></div>Risk-Free Rate: <b>{rf_rate_pct:.2f}%</b></div>
  </div>
</div>
<div class="glow-div"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ROW 1 — REAL PERFORMANCE KPI FLIP CARDS
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-kpi")
sec_header(
    "Performance Snapshot",
    badge=f"REAL DATA · {time_horizon.upper()}",
    color="#0284C7",
    info_id="info-kpi",
    info_html="""
<h4>Performance Snapshot (Dynamic Time Horizon)</h4>
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
        bar_css="linear-gradient(90deg,#059669,#10B981)",
        lbl="Max Sharpe CAGR",
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
        bar_css="linear-gradient(90deg,#0284C7,#38BDF8)",
        lbl="Sharpe Ratio",
        val_html=f"{ms_sharpe:.2f}",
        val_cls="c-blue",
        dlt_html=f'<span class="{"up" if diff_sharpe >= 0 else "down"}">{"▲" if diff_sharpe >= 0 else "▼"} {diff_sharpe:+.2f} vs Equal Weight</span>',
        back_tag="RISK-ADJUSTED",
        back_tag_cls="c-blue",
        back_name="Sharpe Ratio",
        desc=f"Excess return generated per unit of total annualized volatility above the {rf_rate_pct:.2f}% risk-free rate for this exact time period.",
        eq=f"Sharpe = (R_p − {rf_rate_pct:.2f}%) / σ_p × √252",
    )

with c3:
    flip_card(
        "maxdd",
        bar_css="linear-gradient(90deg,#E11D48,#F43F5E)",
        lbl="Max Drawdown",
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
        bar_css="linear-gradient(90deg,#D97706,#FBBF24)",
        lbl="Daily 95% VaR",
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
# 9. ROW 2 — REAL EQUITY CURVE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-perf")
sec_header(
    f"Equity Curve Comparison ({start_str} → {end_str})",
    badge=f"{len(ret_ms_series):,} TRADING DAYS",
    color="#0284C7",
    info_id="info-perf",
    info_html="""
<h4>Equity Curve Comparison (Base ₹100)</h4>
<p>Cumulative compounded performance of the three canonical MPT allocation strategies over your selected time window:</p>
<p>• <b style="color:#10B981;">Max Sharpe Portfolio</b> — Mean-variance optimal tangential allocation<br>
• <b style="color:#38BDF8;">Minimum Volatility Portfolio</b> — Global Minimum Variance (left-most frontier)<br>
• <b style="color:#A855F7;">Equal Weight (1/N)</b> — Benchmark allocation across all active assets</p>
""",
)

fig_main = go.Figure()

fig_main.add_trace(go.Scatter(
    x=nav_ew.index, y=nav_ew.values, name="Equal Weight (1/N)",
    line=dict(color="#B5A88E", width=2.0, dash="dot"),
    hovertemplate="<b>Equal Weight</b><br>%{x|%d %b %Y}<br>NAV: ₹%{y:,.2f}<extra></extra>",
))

fig_main.add_trace(go.Scatter(
    x=nav_mv.index, y=nav_mv.values, name="Min Volatility",
    line=dict(color="#3B4536", width=2.4),
    hovertemplate="<b>Min Volatility</b><br>%{x|%d %b %Y}<br>NAV: ₹%{y:,.2f}<extra></extra>",
))

fig_main.add_trace(go.Scatter(
    x=nav_ms.index, y=nav_ms.values, name="Max Sharpe Portfolio",
    line=dict(color="#5C6B52", width=3.4),
    fill="tozeroy", fillcolor="rgba(92, 107, 82, 0.12)",
    hovertemplate="<b>Max Sharpe</b><br>%{x|%d %b %Y}<br>NAV: ₹%{y:,.2f}<extra></extra>",
))

apply_layout(
    fig_main, h=420,
    yaxis=dict(tickprefix="₹", showgrid=True, gridcolor="#C8C0AE", zeroline=False),
    hovermode="x unified",
)
st.plotly_chart(fig_main, width="stretch", config={"displayModeBar": False})
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. ROW 3 — MONTE CARLO EFFICIENT FRONTIER + ASSET ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-analytics")
col_ef, col_alloc = st.columns([1.1, 0.9], gap="large")

with col_ef:
    sec_header(
        "Monte Carlo Efficient Frontier",
        badge=f"{len(frontier_df):,} SIMULATIONS",
        color="#7C3AED",
        info_id="info-ef",
        info_html="""
<h4>◎ Dynamic Monte Carlo Efficient Frontier</h4>
<p>Thousands of random Dirichlet-weighted portfolios mapped in risk-return space using the active covariance matrix for the selected time window.</p>
<p>• <b style="color:#10B981;">★ Green Star</b> = Max Sharpe Tangency Portfolio<br>
• <b style="color:#38BDF8;">◆ Blue Diamond</b> = Global Minimum Volatility Portfolio<br>
• <b style="color:#A855F7;">● Violet Circle</b> = 1/N Equal Weight Baseline<br>
• <b style="color:#F59E0B;">-- Amber Dashed</b> = Capital Allocation Line (CAL) from Rf</p>
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
            colorscale=[[0, "#DCD5C6"], [0.5, "#7A8F6E"], [1, "#3B4536"]],
            size=3.8,
            opacity=0.45,
            colorbar=dict(
                title=dict(text="Sharpe", font=dict(family="JetBrains Mono", size=10, color="#94A3B8")),
                tickfont=dict(family="JetBrains Mono", size=9, color="#94A3B8"),
                thickness=9, len=0.65,
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
        line=dict(color="#8B6B2B", width=2.0, dash="dash"),
        name=f"CAL (Rf={rf_rate_pct:.1f}%)",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>CAL</extra>",
    ))

    # Optimal Markers
    fig_ef.add_trace(go.Scatter(
        x=[ms_vol * 100], y=[ms_ret * 100],
        mode="markers+text",
        marker=dict(symbol="star", size=19, color="#5C6B52", line=dict(color="#F5F0E8", width=1.5)),
        text=[" Max Sharpe"], textposition="top right",
        textfont=dict(family="Inter", size=10, color="#5C6B52"),
        name=f"Max Sharpe (SR={ms_sharpe:.2f})",
        hovertemplate="<b>Max Sharpe Portfolio</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: " + f"{ms_sharpe:.2f}<extra></extra>",
    ))

    fig_ef.add_trace(go.Scatter(
        x=[mv_vol * 100], y=[mv_ret * 100],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color="#3B4536", line=dict(color="#F5F0E8", width=1.3)),
        text=[" Min Vol"], textposition="bottom right",
        textfont=dict(family="Inter", size=10, color="#3B4536"),
        name=f"Min Vol (σ={mv_vol*100:.1f}%)",
        hovertemplate="<b>Min Volatility Portfolio</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    fig_ef.add_trace(go.Scatter(
        x=[ew_vol * 100], y=[ew_ret * 100],
        mode="markers+text",
        marker=dict(symbol="circle", size=12, color="#B5A88E", line=dict(color="#F5F0E8", width=1.3)),
        text=[" Equal Weight"], textposition="top left",
        textfont=dict(family="Inter", size=9.5, color="#B5A88E"),
        name="Equal Weight (1/N)",
        hovertemplate="<b>1/N Equal Weight</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    apply_layout(
        fig_ef, h=400,
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Expected Return (%)",
        xaxis=dict(ticksuffix="%", showgrid=False),
        yaxis=dict(ticksuffix="%", gridcolor="#C8C0AE"),
        legend=dict(orientation="v", x=0.02, y=0.98, xanchor="left", yanchor="top", font=dict(size=9.5, color="#E2E8F0")),
    )
    st.plotly_chart(fig_ef, width="stretch", config={"displayModeBar": False})

with col_alloc:
    sec_header(
        "Optimal Asset Allocation Breakdown",
        badge=f"{len([k for k, v in ms_weights.items() if v > 0])} ALLOCATED",
        color="#059669",
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
        marker_color="#5C6B52",
        hovertemplate="<b>%{x}</b><br>Max Sharpe: %{y:.2f}%<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        x=top_tickers,
        y=[mv_weights.get(t, 0.0) * 100 for t in top_tickers],
        name="Min Volatility",
        marker_color="#3B4536",
        hovertemplate="<b>%{x}</b><br>Min Vol: %{y:.2f}%<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        x=top_tickers,
        y=[ew_weights.get(t, 0.0) * 100 for t in top_tickers],
        name="Equal Weight",
        marker_color="#B5A88E",
        hovertemplate="<b>%{x}</b><br>Equal Weight: %{y:.2f}%<extra></extra>",
    ))

    apply_layout(
        fig_bar, h=400,
        barmode="group",
        yaxis_title="Weight (%)",
        yaxis=dict(ticksuffix="%", gridcolor="#C8C0AE"),
        xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=9.5, family="JetBrains Mono", color="#94A3B8")),
        legend=dict(orientation="h", x=1.0, y=1.02, xanchor="right", font=dict(size=9.5, color="#E2E8F0")),
    )
    st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11. ROW 4 — ROLLING RISK & CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-risk")
col_rv, col_corr = st.columns([1.05, 0.95], gap="large")

with col_rv:
    sec_header(
        "Realized Rolling Volatility",
        badge="21D & 63D WINDOWS",
        color="#059669",
        info_id="info-vol",
        info_html="""
<h4>Realized Rolling Risk</h4>
<p>Rolling volatility computed directly on the real historical returns of the Max Sharpe portfolio across the selected window:</p>
<p>• <b style="color:#38BDF8;">21-Day Window</b> — High-responsiveness short-term volatility<br>
• <b style="color:#A855F7;">63-Day Window</b> — Quarterly structural volatility trend</p>
""",
    )

    roll_window_short = min(21, max(3, len(ret_ms_series) // 4))
    roll_window_long  = min(63, max(5, len(ret_ms_series) // 2))

    vol_21 = ret_ms_series.rolling(roll_window_short).std() * np.sqrt(252) * 100
    vol_63 = ret_ms_series.rolling(roll_window_long).std() * np.sqrt(252) * 100

    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=vol_21.index, y=vol_21.values, name=f"{roll_window_short}d Realized Vol",
        line=dict(color="#3B4536", width=2.4),
        hovertemplate="%{x|%d %b %Y}<br>21d Vol: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig_rv.add_trace(go.Scatter(
        x=vol_63.index, y=vol_63.values, name=f"{roll_window_long}d Realized Vol",
        line=dict(color="#B5A88E", width=2.2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>63d Vol: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig_rv.add_hline(
        y=ms_vol * 100, line=dict(color="#5C6B52", width=1.8, dash="dot"),
        annotation_text=f"Sample Vol ({ms_vol*100:.1f}%)", annotation_position="right",
        annotation_font=dict(color="#5C6B52", size=9.5, family="Inter"),
    )

    apply_layout(
        fig_rv, h=340,
        yaxis=dict(ticksuffix="%", gridcolor="#C8C0AE"),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
    )
    st.plotly_chart(fig_rv, width="stretch", config={"displayModeBar": False})

with col_corr:
    sec_header(
        "Pairwise Correlation Matrix",
        badge="REAL TIME-WINDOW COV",
        color="#0284C7",
        info_id="info-corr",
        info_html="""
<h4>◈ Cross-Asset Pearson Correlation</h4>
<p>Pairwise return correlations for the primary stocks in your selection, computed across this specific time slice.</p>
""",
    )

    corr_assets = top_tickers[:12] if len(top_tickers) >= 4 else active_tickers[:12]
    corr_matrix = daily_rets[corr_assets].corr()

    fig_hm = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale=[[0, "#8B4B4B"], [0.5, "#DCD5C6"], [1, "#3B4536"]],
        zmin=-0.3, zmax=1.0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=8.5, family="Inter", color="#3B4536"),
        hovertemplate="%{y} vs %{x}<br>Correlation: <b>%{z:.3f}</b><extra></extra>",
        colorbar=dict(
            title=dict(text="ρ", font=dict(size=10, color="#94A3B8")),
            tickfont=dict(family="JetBrains Mono", size=8.5, color="#94A3B8"),
            thickness=9, len=0.8,
        ),
    ))

    apply_layout(
        fig_hm, h=340,
        xaxis=dict(showgrid=False, tickfont=dict(size=8.5, family="JetBrains Mono", color="#94A3B8"), tickangle=-45),
        yaxis=dict(showgrid=False, tickfont=dict(size=8.5, family="JetBrains Mono", color="#94A3B8")),
        margin=dict(l=5, r=40, t=35, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig_hm, width="stretch", config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 12. ROW 5 — ALLOCATION DONUT + HISTORICAL DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-portfolio")
col_donut, col_dd = st.columns([0.85, 1.15], gap="large")

with col_donut:
    sec_header(
        "Max Sharpe Allocation Composition",
        badge="ACTIVE HOLDINGS",
        color="#059669",
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

    PALETTE = ["#3B4536", "#5C6B52", "#7A8F6E", "#8B9E7E", "#B5A88E", "#8B6B2B", "#A8A095", "#8B4B4B", "#D4CCBC", "#6E7A68", "#9E8B70", "#4A5644"]

    fig_pie = go.Figure(go.Pie(
        labels=donut_df["ticker"],
        values=donut_df["weight"],
        hole=0.62,
        marker=dict(colors=PALETTE[:len(donut_df)], line=dict(color="#E8E3D7", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=9.5, color="#F5F0E8"),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        direction="clockwise",
        sort=True,
    ))

    fig_pie.add_annotation(
        text=f"<b>{len(donut_df)}</b><br><span style='font-size:9px;color:#94A3B8'>STOCKS</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="Playfair Display", size=19, color="#3B4536"),
    )

    apply_layout(fig_pie, h=345, showlegend=False, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

    sector_allocations = (
        pd.DataFrame(
            {
                "sector": [get_sector(ticker) for ticker in non_zero_alloc],
                "weight": list(non_zero_alloc.values()),
            }
        )
        .groupby("sector", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )
    sector_colors = ["#3B4536", "#5C6B52", "#7A8F6E", "#8B9E7E", "#B5A88E", "#8B6B2B", "#A8A095", "#8B4B4B", "#D4CCBC", "#6E7A68", "#9E8B70"]
    sector_fig = go.Figure(
        go.Pie(
            labels=sector_allocations["sector"],
            values=sector_allocations["weight"],
            hole=0.62,
            marker=dict(
                colors=sector_colors[: len(sector_allocations)],
                line=dict(color="#E8E3D7", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(family="JetBrains Mono", size=9, color="#FFFFFF"),
            hovertemplate="<b>%{label}</b><br>Sector weight: %{percent}<extra></extra>",
            sort=True,
        )
    )
    sector_fig.add_annotation(
        text=f"<b>{len(sector_allocations)}</b><br><span style='font-size:9px;color:#94A3B8'>SECTORS</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family="Playfair Display", size=19, color="#3B4536"),
    )
    apply_layout(sector_fig, h=345, showlegend=False, margin=dict(l=10, r=10, t=35, b=10))
    st.markdown("#### Sector Allocation Breakdown")
    st.plotly_chart(sector_fig, width="stretch", config={"displayModeBar": False})

with col_dd:
    sec_header(
        "Historical Underwater Drawdown (%)",
        badge=f"MAX DD: {max_dd_ms:.1f}%",
        color="#E11D48",
        info_id="info-dd",
        info_html="""
<h4>Peak-to-Trough Drawdown Analysis</h4>
<p>Measures the continuous drawdown percentage from prior high-water marks for the Max Sharpe portfolio across this specific time horizon.</p>
""",
    )

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_ms_series.index, y=dd_ms_series.values,
        fill="tozeroy", fillcolor="rgba(139,75,75,0.12)",
        line=dict(color="#8B4B4B", width=2.0), name="Max Sharpe DD",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: <b>%{y:.2f}%</b><extra></extra>",
    ))

    if len(dd_ms_series) > 0:
        min_idx = dd_ms_series.idxmin()
        fig_dd.add_annotation(
            x=min_idx, y=dd_ms_series.loc[min_idx],
            text=f"Trough: {max_dd_ms:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor="#8B4B4B",
            font=dict(color="#8B4B4B", size=9.5, family="Inter"),
            bgcolor="#DCD5C6", bordercolor="#8B4B4B",
            borderwidth=1, borderpad=4, ay=-35,
        )

    apply_layout(
        fig_dd, h=345,
        yaxis=dict(ticksuffix="%", gridcolor="#C8C0AE", zeroline=True, zerolinecolor="#475569", zerolinewidth=1),
        xaxis=dict(showgrid=False),
        showlegend=False,
        hovermode="x",
    )
    st.plotly_chart(fig_dd, width="stretch", config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# 13. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glow-div" style="margin-top:3rem;"></div>
<div style="display:flex;justify-content:space-between;align-items:center;padding:0 0 1.5rem 0;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8B8578;">
    Quantitative Portfolio Risk Engine &nbsp;·&nbsp; Modern Portfolio Theory &nbsp;·&nbsp; NIFTY 50 Universe
  </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8B8578;">
    Streamlit + Plotly &nbsp;·&nbsp; IDSE Quant Lab © 2026
  </span>
</div>
""", unsafe_allow_html=True)
