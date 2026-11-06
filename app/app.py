"""Landing page for the Quantitative Portfolio Risk Engine."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Quantitative Portfolio Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
  --background: #1A1C23;
  --card: #252830;
  --border: #3A3F4D;
  --accent: #38BDF8;
  --accent-dark: #0284C7;
  --text: #E2E8F0;
  --muted: #94A3B8;
}

*, *::before, *::after {
  box-sizing: border-box;
}

.stApp {
  background:
    radial-gradient(circle at 50% -10%, rgba(56, 189, 248, 0.13), transparent 38rem),
    var(--background) !important;
  color: var(--text) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

header[data-testid="stHeader"] {
  background: transparent !important;
}

section[data-testid="stSidebar"] {
  display: none;
}

.block-container {
  max-width: 1160px;
  padding: 5rem 2rem 4rem;
}

.hero {
  margin: 0 auto 3.5rem;
  max-width: 850px;
  text-align: center;
}

.eyebrow {
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  margin-bottom: 1.1rem;
  text-transform: uppercase;
}

.hero h1 {
  color: #FFFFFF;
  font-size: clamp(2.3rem, 5vw, 4.4rem);
  font-weight: 800;
  letter-spacing: -0.055em;
  line-height: 1.05;
  margin: 0;
}

.hero p {
  color: var(--muted);
  font-size: 1.08rem;
  line-height: 1.75;
  margin: 1.5rem auto 0;
  max-width: 690px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
  min-height: 286px;
  transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--accent);
  box-shadow: 0 18px 42px rgba(14, 165, 233, 0.16);
  transform: translateY(-5px);
}

.feature-icon {
  align-items: center;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-radius: 10px;
  color: var(--accent);
  display: flex;
  font-size: 1.45rem;
  height: 45px;
  justify-content: center;
  margin-bottom: 1.25rem;
  width: 45px;
}

.feature-title {
  color: #FFFFFF;
  font-size: 1.2rem;
  font-weight: 700;
  margin: 0 0 0.75rem;
}

.feature-description {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.65;
  margin: 0 0 1.25rem;
  min-height: 74px;
}

.status {
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
  background: var(--accent-dark);
  border: 1px solid var(--accent);
  border-radius: 7px;
  color: #FFFFFF;
  font-weight: 700;
  min-height: 2.55rem;
  transition: background 180ms ease, transform 180ms ease;
}

div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:hover {
  background: #0369A1;
  transform: translateY(-1px);
}

div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:disabled {
  background: #303541;
  border-color: #444B5B;
  color: #7F8A9D;
}

.footer {
  color: #64748B;
  font-size: 0.78rem;
  margin-top: 3.5rem;
  text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Data · Risk · Intelligence</div>
  <h1>Quantitative Portfolio<br>Risk Engine</h1>
  <p>
    Demystifying institutional quantitative finance through data science,
    machine learning, and interactive simulation.
  </p>
</div>
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
<div class="feature-icon">📈</div>
<div class="feature-title">Historical Simulator</div>
<p class="feature-description">
  Travel back in time. Simulate custom investments, build your own portfolio
  with real historical data, and track your PnL.
</p>
<div class="status">● Active</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Historical Simulator", key="historical_simulator", use_container_width=True):
            open_page("pages/1_📈_Historical_Simulator.py")

    with st.container(border=True):
        st.markdown(
            """
<div class="feature-icon">🔬</div>
<div class="feature-title">Exploratory Data Analysis &amp; Stats</div>
<p class="feature-description">
  Dive deep into the raw numbers. Visualize volatility, price distributions,
  moving averages, and asset correlations.
</p>
<div class="status">Next phase</div>
""",
            unsafe_allow_html=True,
        )
        st.button(
            "Coming Soon",
            key="eda",
            disabled=True,
            use_container_width=True,
        )

with right_column:
    with st.container(border=True):
        st.markdown(
            """
<div class="feature-icon">⚙️</div>
<div class="feature-title">Classical Quant Engine</div>
<p class="feature-description">
  Discover how Wall Street balances risk and reward using Modern Portfolio
  Theory (MPT) and Markowitz optimization.
</p>
<div class="status">● Active</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Quant Engine", key="quant_engine", use_container_width=True):
            open_page("pages/2_⚙️_Quant_Engine.py")

    with st.container(border=True):
        st.markdown(
            """
<div class="feature-icon">🧠</div>
<div class="feature-title">ML Dynamic Risk Engine</div>
<p class="feature-description">
  The brain of the platform. Watch machine learning models predict market
  crashes and dynamically rebalance your assets to safety.
</p>
<div class="status">Next phase</div>
""",
            unsafe_allow_html=True,
        )
        st.button(
            "Coming Soon",
            key="ml_engine",
            disabled=True,
            use_container_width=True,
        )


st.markdown(
    '<div class="footer">Built for curious investors, aspiring quants, and better decisions.</div>',
    unsafe_allow_html=True,
)
