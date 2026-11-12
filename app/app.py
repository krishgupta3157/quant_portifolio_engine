"""landing page for the Quantitative Portfolio Risk Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.theme import apply_global_theme


st.set_page_config(
    page_title="Quant Terminal | Portfolio Risk Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_global_theme()


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

html {
  scroll-behavior: smooth !important;
}

:root {
  --ink: #E8E3D7;
  --panel: #DCD5C6;
  --panel-deep: #D4CCBC;
  --line: #C8C0AE;
  --text: #222B1C;
  --muted: #7A7568;
  --olive: #222B1C;
  --olive-mid: #3B4536;
  --sage: #7A8F6E;
  --cream: #F5F0E8;
  --terra: #8B4B4B;
}

* { box-sizing: border-box; }
.stApp {
  background: var(--ink) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}
header[data-testid="stHeader"] {
  display: none !important;
}
section[data-testid="stSidebar"] { display: none; }

/* Generous top clearance so fixed top navbar never overlaps page content */
.block-container {
  max-width: 1180px;
  padding: 6.2rem 1.5rem 3.5rem !important;
}

/* ═══════ GLOBAL TOP NAVBAR (DEEP OLIVE GREEN) ═══════ */
.fixed-top-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 64px;
  background: #222B1C;
  border-bottom: 1px solid rgba(200, 192, 174, 0.25);
  box-shadow: 0 4px 18px rgba(34, 43, 28, 0.25);
  z-index: 999999;
  display: flex;
  align-items: center;
  justify-content: center;
}

.nav-inner {
  width: 100%;
  max-width: 1180px;
  padding: 0 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 2.2rem;
}

.nav-brand {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: #FFFFFF !important;
  text-decoration: none !important;
  letter-spacing: 0.04em;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 1.8rem;
}

.nav-link {
  font-family: 'Inter', sans-serif;
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #D4CCBC !important;
  text-decoration: none !important;
  transition: color 0.18s ease;
}

.nav-link:hover {
  color: #FFFFFF !important;
}

.nav-right {
  display: flex;
  align-items: center;
}

.nav-cta-btn {
  background: #E8E3D7 !important;
  color: #222B1C !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.45rem 1.15rem !important;
  border-radius: 20px !important;
  text-decoration: none !important;
  transition: all 0.2s ease !important;
  border: 1px solid #E8E3D7 !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.nav-cta-btn:hover {
  background: #DCD5C6 !important;
  color: #171E12 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

/* ═══════ HERO HEADER ═══════ */
.hero { max-width: 920px; padding-bottom: 2.2rem; }
div[data-testid="stHorizontalBlock"]:has(.hero) {
  align-items: center;
  gap: 2.5rem !important;
}
div[data-testid="stHorizontalBlock"]:has(.hero) > div[data-testid="stColumn"] img {
  border-radius: 22px;
  box-shadow: 0 18px 44px rgba(34, 43, 28, .10);
  mix-blend-mode: multiply;
  opacity: .9;
  transform: scale(1.08);
  -webkit-mask-image: radial-gradient(ellipse at center, #000 62%, rgba(0, 0, 0, .92) 78%, transparent 100%);
  mask-image: radial-gradient(ellipse at center, #000 62%, rgba(0, 0, 0, .92) 78%, transparent 100%);
}
.kicker {
  color: var(--olive-mid);
  font-family: 'Inter', sans-serif;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .18em;
  margin-bottom: 1.25rem;
  text-transform: uppercase;
}
.hero h1 {
  color: var(--text);
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(2.8rem, 6.5vw, 5.5rem);
  font-weight: 700;
  letter-spacing: -.03em;
  line-height: 1.05;
  margin: 0;
}
.hero h1 span {
  color: var(--olive-mid);
  font-style: italic;
  font-weight: 400;
}
.hero-copy {
  color: var(--muted);
  font-size: .95rem;
  line-height: 1.75;
  margin: 1.4rem 0 1.4rem;
  max-width: 640px;
}
.scroll-target {
  scroll-margin-top: 84px;
  display: block;
}

.section-label {
  color: var(--muted);
  font-family: 'Inter', sans-serif;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .16em;
  margin: 1.5rem 0 1rem;
  text-transform: uppercase;
}
.section-label span { color: var(--olive); font-weight: 800; }

/* ═══════ ABOUT SECTION (CLEAN MINIMALIST WRAPPER) ═══════ */
.about-section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 2.2rem 2.4rem;
  margin-bottom: 3rem;
}
.about-heading {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(1.4rem, 2.4vw, 1.9rem);
  font-weight: 700;
  letter-spacing: -0.015em;
  color: var(--text);
  margin: 0.4rem 0 1.25rem;
  line-height: 1.25;
}
.about-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2.2rem;
}
.about-body {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.75;
  margin: 0;
}
.about-body strong {
  color: var(--text);
  font-weight: 600;
}

/* ═══════ FEATURE CARDS ═══════ */
div[data-testid="stHorizontalBlock"]:has(.card-content) {
  gap: 2.5rem !important;
}
div[data-testid="stColumn"]:has(.card-content) > div[data-testid="stVerticalBlock"] {
  gap: 2rem !important;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  box-shadow: none;
  min-height: 295px;
  overflow: hidden;
  position: relative;
  padding: 1.6rem 1.6rem 1.4rem !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  transition: border-color .25s ease, translate .25s ease, box-shadow .25s ease;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) {
  animation: card-enter .65s cubic-bezier(.2, .75, .25, 1) both;
}
div[data-testid="stColumn"]:nth-child(2) div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) {
  animation-delay: .12s;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content):hover {
  border-color: var(--olive);
  box-shadow: 0 12px 28px rgba(34, 43, 28, .10);
  translate: 0 -5px;
}

@keyframes card-enter {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}

.card-content {
  padding-top: 0.85rem;
  position: relative;
  border-top: 3px solid var(--card-color);
}
.cyan { --card-color: var(--olive); }
.emerald { --card-color: var(--olive-mid); }
.purple { --card-color: #8B8578; }
.crimson { --card-color: var(--terra); }

.card-title {
  color: var(--text);
  font-family: 'Playfair Display', serif;
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 .55rem 0;
  line-height: 1.2;
}
.card-description {
  color: var(--muted);
  font-size: .86rem;
  line-height: 1.65;
  min-height: 64px;
  margin: 0 0 .8rem 0;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) .stButton {
  position: relative;
  z-index: 2;
  margin-top: 1rem;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) .stButton > button,
.stButton > button {
  background: #222B1C !important;
  border: 1px solid #222B1C !important;
  border-radius: 24px !important;
  color: #FFFFFF !important;
  font-family: 'Inter', sans-serif !important;
  font-size: .74rem !important;
  font-weight: 700 !important;
  letter-spacing: .1em !important;
  min-height: 2.6rem !important;
  text-transform: uppercase !important;
  transition: all .2s ease;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) .stButton > button *,
.stButton > button * {
  color: #FFFFFF !important;
  font-weight: 700 !important;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) .stButton > button:hover,
.stButton > button:hover {
  background: #3B4536 !important;
  border-color: #3B4536 !important;
  box-shadow: 0 4px 14px rgba(34, 43, 28, .22) !important;
  color: #FFFFFF !important;
  transform: translateY(-1px);
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) .stButton > button:hover *,
.stButton > button:hover * {
  color: #FFFFFF !important;
}

.footer {
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-family: 'Inter', sans-serif;
  font-size: .68rem;
  letter-spacing: .06em;
  margin-top: 2.5rem;
  padding: 1.5rem 0 1rem;
  text-align: center;
}

@media (max-width: 768px) {
  .block-container { padding: 5.5rem 1rem 2rem !important; }
  div[data-testid="stHorizontalBlock"]:has(.card-content) { gap: 1.25rem !important; }
  div[data-testid="stColumn"]:has(.card-content) > div[data-testid="stVerticalBlock"] { gap: 1.5rem !important; }
  .nav-inner { padding: 0 1rem; }
  .nav-left { gap: 1rem; }
  .nav-links { gap: 1rem; }
  .nav-brand { font-size: 0.9rem; }
  .nav-link { font-size: 0.68rem; }
  .nav-cta-btn { font-size: 0.65rem !important; padding: 0.35rem 0.8rem !important; }
  .about-grid { grid-template-columns: 1fr; gap: 1.2rem; }
  .about-section { padding: 1.6rem 1.4rem; }
  .metrics { flex-wrap: wrap; }
  .metric { flex-basis: 50%; }
  .metric:nth-child(3) { border-left: 0; padding-left: 0; }
}

@media (max-width: 480px) {
  .nav-links { display: none !important; }
}

@media (prefers-reduced-motion: reduce) {
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) {
    animation: none;
  }
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .card-content) {
    transition: none;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<!-- Fixed Global Top Navbar -->
<nav class="fixed-top-nav">
  <div class="nav-inner">
    <div class="nav-left">
      <a href="#" class="nav-brand">QUANT TERMINAL</a>
      <div class="nav-links">
        <a href="#about" class="nav-link">About</a>
        <a href="#features" class="nav-link">Features</a>
      </div>
    </div>
  </div>
</nav>

""",
    unsafe_allow_html=True,
)

hero_copy_column, hero_image_column = st.columns([1.2, 0.8], gap="large")
with hero_copy_column:
    st.markdown(
        """
<div class="hero">
  <div class="kicker">Quantitative Intelligence Platform</div>
  <h1>See risk.<br><span>Build conviction.</span></h1>
  <p class="hero-copy">
    Institutional-grade portfolio intelligence, made transparent. Explore
    historical scenarios, optimize allocations, and turn market noise into
    decisions you can defend.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
with hero_image_column:
    st.image(
        str(BASE_DIR / "app" / "assets" / "market-insight.png"),
        width="stretch",
    )

st.markdown(
    """

<!-- Informational About Section -->
<div id="about" class="scroll-target"></div>
<div class="about-section">
  <div class="section-label"><span>01</span> / About the Platform</div>
  <h2 class="about-heading">Institutional rigor, simplified for modern portfolio architecture.</h2>
  <div class="about-grid">
    <div class="about-col">
      <p class="about-body">
        The Quant Portfolio Risk Engine is designed to bridge the gap between academic portfolio theory and tactical execution. Built around the <strong>NIFTY 50</strong> universe, the platform unifies over 2.1 million observations to measure empirical risk, covariance structures, and fat-tailed distribution dynamics.
      </p>
    </div>
    <div class="about-col">
      <p class="about-body">
        By combining Markowitz Mean-Variance optimization with gradient-boosted volatility forecasting, the platform allows investors to test what-if scenarios across historical market regimes, establish mathematically optimal asset allocations, and quantify tail risk before volatility spikes occur.
      </p>
    </div>
  </div>
</div>

<!-- Features Section Header -->
<div id="features" class="scroll-target"></div>
<div class="section-label"><span>02</span> / Core Analytical Modules</div>
""",
    unsafe_allow_html=True,
)


def open_page(page: str) -> None:
    st.switch_page(page)


left_column, right_column = st.columns(2, gap="large")

with left_column:
    with st.container(border=True):
        st.markdown(
            """
<div class="card-content cyan">
  <div class="card-title">Historical Simulator</div>
  <div class="card-description">Travel back in time. Simulate custom investments, build your own portfolio with real historical data, and track your PnL.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Launch simulator  →", key="historical_simulator", use_container_width=True):
            open_page("pages/1_Historical_Simulator.py")

    with st.container(border=True):
        st.markdown(
            """
<div class="card-content purple">
  <div class="card-title">Exploratory Data Analysis &amp; Stats</div>
  <div class="card-description">Dive deep into the raw numbers. Visualize volatility, price distributions, moving averages, and asset correlations.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open EDA & stats  →", key="eda", use_container_width=True):
            st.switch_page("pages/3_EDA_and_Stats.py")

with right_column:
    with st.container(border=True):
        st.markdown(
            """
<div class="card-content emerald">
  <div class="card-title">Classical Quant Engine</div>
  <div class="card-description">Discover how Wall Street balances risk and reward using Modern Portfolio Theory and Markowitz optimization.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open quant engine  →", key="quant_engine", use_container_width=True):
            open_page("pages/2_Quant_Engine.py")

    with st.container(border=True):
        st.markdown(
            """
<div class="card-content crimson">
  <div class="card-title">ML Dynamic Risk Engine</div>
  <div class="card-description">The brain of the platform. Watch machine learning models predict market crashes and dynamically rebalance your assets to safety.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open ML risk engine  →", key="ml_engine", use_container_width=True):
            st.switch_page("pages/4_ML_Risk_Engine.py")


st.markdown(
    """
<div class="footer">
  Quantitative Portfolio Risk Engine
</div>
""",
    unsafe_allow_html=True,
)
