"""Premium landing page for the Quantitative Portfolio Risk Engine."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Quant Terminal | Portfolio Risk Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --ink: #0B0F19;
  --panel: rgba(20, 25, 35, .72);
  --line: rgba(148, 163, 184, .16);
  --text: #E8EEF7;
  --muted: #8290A6;
  --cyan: #22D3EE;
  --emerald: #34D399;
  --purple: #A78BFA;
  --crimson: #FB7185;
}

* { box-sizing: border-box; }
.stApp {
  background:
    radial-gradient(ellipse at 10% 0%, rgba(34, 211, 238, .10), transparent 34%),
    radial-gradient(ellipse at 90% 15%, rgba(124, 58, 237, .12), transparent 30%),
    var(--ink) !important;
  color: var(--text) !important;
  font-family: 'Space Grotesk', sans-serif !important;
}
.stApp:before {
  background-image: linear-gradient(rgba(34,211,238,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(34,211,238,.035) 1px, transparent 1px);
  background-size: 44px 44px;
  content: "";
  inset: 0;
  mask-image: linear-gradient(to bottom, black, transparent 80%);
  pointer-events: none;
  position: fixed;
}
header[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1180px; padding: 3.25rem 1.5rem 2rem; }

.terminal-nav {
  align-items: center;
  display: flex;
  justify-content: space-between;
  margin-bottom: 4.5rem;
}
.brand {
  color: #fff;
  font-family: 'DM Mono', monospace;
  font-size: .78rem;
  letter-spacing: .14em;
}
.brand b { color: var(--cyan); font-size: 1.1rem; }
.nav-status {
  color: var(--emerald);
  font-family: 'DM Mono', monospace;
  font-size: .68rem;
  letter-spacing: .1em;
}
.nav-status:before { content: "●"; margin-right: .5rem; }

.hero { max-width: 920px; padding-bottom: 3.2rem; }
.kicker {
  color: var(--cyan);
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  letter-spacing: .18em;
  margin-bottom: 1.25rem;
  text-transform: uppercase;
}
.hero h1 {
  color: #fff;
  font-size: clamp(2.6rem, 7vw, 6.4rem);
  font-weight: 700;
  letter-spacing: -.075em;
  line-height: .96;
  margin: 0;
}
.hero h1 span {
  background: linear-gradient(100deg, #fff 20%, var(--cyan), #A78BFA);
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
}
.hero-copy {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.7;
  margin: 1.6rem 0 1.5rem;
  max-width: 650px;
}
.type-line {
  animation: type-in 3.2s steps(58, end) both;
  border-right: 1px solid var(--cyan);
  color: var(--cyan);
  display: inline-block;
  font-family: 'DM Mono', monospace;
  font-size: .76rem;
  max-width: 100%;
  overflow: hidden;
  padding-right: .35rem;
  white-space: nowrap;
}
@keyframes type-in { from { max-width: 0; } to { max-width: 100%; } }

.metrics {
  border-bottom: 1px solid var(--line);
  border-top: 1px solid var(--line);
  display: flex;
  gap: 0;
  margin-bottom: 2rem;
  max-width: 800px;
}
.metric { flex: 1; padding: 1rem 1.25rem 1rem 0; }
.metric + .metric { border-left: 1px solid var(--line); padding-left: 1.25rem; }
.metric-label {
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.metric-value { color: #fff; font-size: 1rem; font-weight: 600; margin-top: .35rem; }

.section-label {
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  font-size: .68rem;
  letter-spacing: .16em;
  margin: 1rem 0 1rem;
  text-transform: uppercase;
}
.section-label span { color: var(--cyan); }

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel);
  backdrop-filter: blur(14px);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 16px 45px rgba(0,0,0,.24);
  min-height: 325px;
  overflow: hidden;
  position: relative;
  transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(34, 211, 238, .7);
  box-shadow: 0 22px 55px rgba(0,0,0,.42), 0 0 28px rgba(34,211,238,.10);
  transform: translateY(-8px);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.emerald):hover {
  border-color: rgba(52, 211, 153, .75);
  box-shadow: 0 22px 55px rgba(0,0,0,.42), 0 0 28px rgba(52,211,153,.12);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.purple):hover {
  border-color: rgba(167, 139, 250, .75);
  box-shadow: 0 22px 55px rgba(0,0,0,.42), 0 0 28px rgba(167,139,250,.12);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.crimson):hover {
  border-color: rgba(251, 113, 133, .75);
  box-shadow: 0 22px 55px rgba(0,0,0,.42), 0 0 28px rgba(251,113,133,.12);
}
.card-content { padding-top: .15rem; position: relative; }
.card-content:before {
  background: var(--card-color);
  box-shadow: 0 0 18px var(--card-color);
  content: "";
  height: 3px;
  left: -1rem;
  position: absolute;
  right: -1rem;
  top: -1.05rem;
}
.cyan { --card-color: var(--cyan); }
.emerald { --card-color: var(--emerald); }
.purple { --card-color: var(--purple); }
.crimson { --card-color: var(--crimson); }
.card-icon {
  align-items: center;
  background: color-mix(in srgb, var(--card-color) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--card-color) 35%, transparent);
  border-radius: 10px;
  color: var(--card-color);
  display: flex;
  font-size: 1.35rem;
  height: 43px;
  justify-content: center;
  margin: .25rem 0 1.25rem;
  width: 43px;
}
.card-title { color: #fff; font-size: 1.15rem; font-weight: 600; margin-bottom: .65rem; }
.card-description { color: var(--muted); font-size: .86rem; line-height: 1.65; min-height: 72px; }
.card-meta {
  color: var(--card-color);
  font-family: 'DM Mono', monospace;
  font-size: .63rem;
  letter-spacing: .1em;
  margin: 1.1rem 0 .75rem;
  text-transform: uppercase;
}
.badge {
  background: rgba(148,163,184,.08);
  border: 1px solid rgba(148,163,184,.18);
  border-radius: 4px;
  color: #AAB7CA;
  display: inline-block;
  font-family: 'DM Mono', monospace;
  font-size: .62rem;
  margin-right: .35rem;
  padding: .25rem .4rem;
}
.formula {
  color: rgba(167,139,250,.10);
  font-family: Georgia, serif;
  font-size: 2.45rem;
  font-style: italic;
  position: absolute;
  right: -.2rem;
  top: 3.3rem;
  transition: color .25s ease;
  white-space: nowrap;
  z-index: 0;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover .formula { color: rgba(167,139,250,.24); }

div[data-testid="stVerticalBlockBorderWrapper"] .stButton { position: relative; z-index: 2; }
div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
  background: rgba(255,255,255,.025);
  border: 1px solid rgba(148,163,184,.24);
  border-radius: 7px;
  color: #DCE5F2;
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  min-height: 2.55rem;
  transition: all .2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:hover {
  background: rgba(34,211,238,.1);
  border-color: var(--cyan);
  color: #fff;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.emerald) .stButton > button:hover {
  background: rgba(52,211,153,.1);
  border-color: var(--emerald);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.purple) .stButton > button:hover {
  background: rgba(167,139,250,.1);
  border-color: var(--purple);
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.crimson) .stButton > button:hover {
  background: rgba(251,113,133,.1);
  border-color: var(--crimson);
}
div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:disabled {
  background: rgba(255,255,255,.02);
  border-color: rgba(148,163,184,.12);
  color: #5D687A;
}

.terminal-title {
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  font-size: .66rem;
  letter-spacing: .14em;
  margin: 2rem 0 .5rem;
  text-transform: uppercase;
}
.footer {
  border-top: 1px solid var(--line);
  color: #65748A;
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  letter-spacing: .06em;
  margin-top: 2rem;
  padding: 1.25rem 0 1rem;
  text-align: center;
}
.stack { color: #AAB7CA; margin-top: .5rem; }
.stack b { color: var(--cyan); font-weight: 400; }
@media (max-width: 640px) {
  .block-container { padding: 2rem 1rem; }
  .terminal-nav { margin-bottom: 3rem; }
  .metrics { flex-wrap: wrap; }
  .metric { flex-basis: 50%; }
  .metric:nth-child(3) { border-left: 0; padding-left: 0; }
  .type-line { font-size: .62rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="terminal-nav">
  <div class="brand"><b>◈</b> QP / RISK-ENGINE</div>
  <div class="nav-status">SYSTEM ONLINE · v1.0.0</div>
</div>
<div class="hero">
  <div class="kicker">// Quantitative intelligence platform</div>
  <h1>See risk.<br><span>Build conviction.</span></h1>
  <p class="hero-copy">
    Institutional-grade portfolio intelligence, made transparent. Explore
    historical scenarios, optimize allocations, and turn market noise into
    decisions you can defend.
  </p>
  <div class="type-line">&gt; Initializing Multi-Asset Risk Engine... Model: LightGBM... Status: Online █</div>
</div>
<div class="metrics">
  <div class="metric"><div class="metric-label">Data points</div><div class="metric-value">2.1M+</div></div>
  <div class="metric"><div class="metric-label">Compute</div><div class="metric-value">Vectorized</div></div>
  <div class="metric"><div class="metric-label">Universe</div><div class="metric-value">NIFTY 50</div></div>
</div>
<div class="section-label"><span>01</span> / Select a module</div>
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
  <div class="card-icon">↗</div>
  <div class="card-title">Historical Simulator</div>
  <div class="card-description">Travel back in time. Simulate custom investments, build your own portfolio with real historical data, and track your PnL.</div>
  <div class="card-meta">● Live module</div>
  <span class="badge">Plotly</span><span class="badge">Pandas</span><span class="badge">NIFTY 50</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Launch simulator  →", key="historical_simulator", use_container_width=True):
            open_page("pages/1_📈_Historical_Simulator.py")

    with st.container(border=True):
        st.markdown(
            """
<div class="card-content purple">
  <div class="card-icon">⌁</div>
  <div class="card-title">Exploratory Data Analysis &amp; Stats</div>
  <div class="card-description">Dive deep into the raw numbers. Visualize volatility, price distributions, moving averages, and asset correlations.</div>
  <div class="card-meta">● Live module</div>
  <span class="badge">Plotly</span><span class="badge">NumPy</span><span class="badge">Statistics</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open EDA & stats  →", key="eda", use_container_width=True):
            st.switch_page("pages/3_📊_EDA_and_Stats.py")

with right_column:
    with st.container(border=True):
        st.markdown(
            """
<div class="card-content emerald">
  <div class="formula">σ² = wᵀΣw</div>
  <div class="card-icon">∑</div>
  <div class="card-title">Classical Quant Engine</div>
  <div class="card-description">Discover how Wall Street balances risk and reward using Modern Portfolio Theory and Markowitz optimization.</div>
  <div class="card-meta">● Live module</div>
  <span class="badge">SciPy</span><span class="badge">MPT</span><span class="badge">Markowitz</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open quant engine  →", key="quant_engine", use_container_width=True):
            open_page("pages/2_⚙️_Quant_Engine.py")

    with st.container(border=True):
        st.markdown(
            """
<div class="card-content crimson">
  <div class="card-icon">⌬</div>
  <div class="card-title">ML Dynamic Risk Engine</div>
  <div class="card-description">The brain of the platform. Watch machine learning models predict market crashes and dynamically rebalance your assets to safety.</div>
  <div class="card-meta">○ Pipeline queued</div>
  <span class="badge">LightGBM</span><span class="badge">Scikit-Learn</span><span class="badge">ML</span>
</div>
""",
            unsafe_allow_html=True,
        )
        st.button("Module in development", key="ml_engine", disabled=True, use_container_width=True)


st.markdown('<div class="terminal-title">// system activity log</div>', unsafe_allow_html=True)
components.html(
    """
<style>
body { background: transparent; margin: 0; }
.log { background: rgba(3, 7, 14, .92); border: 1px solid #1E2A3B; border-radius: 8px; color: #738399; font: 11px/1.8 'Courier New', monospace; height: 106px; overflow-y: auto; padding: 10px 14px; }
.ok { color: #34D399; } .info { color: #22D3EE; } .warn { color: #FBBF24; }
.cursor { animation: blink 1s steps(2) infinite; color: #22D3EE; } @keyframes blink { 50% { opacity: 0; } }
</style>
<div class="log" id="log"><span class="info">&gt; boot sequence initiated...</span><br></div>
<script>
const lines = [
  ['info', '[INFO] Loading covariance matrices...'],
  ['warn', '[WARN] High volatility detected in Tech sector...'],
  ['info', '[INFO] Vectorized price matrix: 2,104,887 rows'],
  ['ok', '[OK] MPT Optimizer ready.'],
  ['ok', '[OK] Risk engine accepting requests.']
];
let i = 0;
function writeLine() {
  if (i >= lines.length) return;
  const [kind, text] = lines[i++];
  document.getElementById('log').innerHTML += `<span class="${kind}">${text}</span><br>`;
  setTimeout(writeLine, 520);
}
setTimeout(writeLine, 420);
</script>
""",
    height=130,
    scrolling=False,
)

st.markdown(
    """
<div class="footer">
  QUANTITATIVE PORTFOLIO RISK ENGINE
  <div class="stack"><b>Python</b> · Pandas · LightGBM · Streamlit · Scikit-Learn · SciPy</div>
</div>
""",
    unsafe_allow_html=True,
)
