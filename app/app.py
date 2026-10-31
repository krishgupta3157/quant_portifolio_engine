"""
Quantitative Portfolio Risk Engine & Dynamic Asset Allocation Optimizer
Frontend · Streamlit + Plotly · All mock data — backend not yet built
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS + NAVBAR + AMBIENT BLOBS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

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

/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden !important; height: 0 !important; }
.block-container {
  padding: calc(var(--nav-h) + 1.8rem) 2.2rem 2.5rem 2.2rem !important;
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
  background: rgba(5,8,19,0.80);
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  border-bottom: 1px solid var(--bdr);
}
.qp-brand {
  font-size: 0.75rem; font-weight: 800; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--muted); margin-right: 2.5rem;
  flex-shrink: 0; white-space: nowrap;
}
.qp-brand .hi { color: var(--blue); }
.qp-links { display: flex; gap: 0.15rem; flex: 1; }
.nav-btn {
  display: flex; align-items: center; gap: 0.38rem;
  padding: 0.3rem 0.88rem; border-radius: 8px;
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
  color: var(--muted); cursor: pointer;
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
  background: rgba(251,113,133,0.10); border: 1px solid rgba(251,113,133,0.22);
  color: var(--rose); white-space: nowrap;
}
.rdot {
  width: 5px; height: 5px; border-radius: 50%; background: var(--rose);
  animation: rdotPulse 1.6s infinite;
}
@keyframes rdotPulse {
  0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(251,113,133,0.5); }
  60%      { opacity:0.4; box-shadow: 0 0 0 5px rgba(251,113,133,0); }
}

/* ═══════ FLIP CARDS ═══════ */
.flip-wrap {
  perspective: 1200px; height: 182px;
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
.fc-ico  { font-size: 1.25rem; opacity: 0.13; }
.fc-val  { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1.05; margin-bottom: 0.38rem; }
.fc-dlt  { font-family: 'JetBrains Mono', monospace; font-size: 0.67rem; color: var(--muted); }
.fc-dlt .up   { color: var(--green); }
.fc-dlt .down { color: var(--rose);  }
.fc-hint { position: absolute; bottom: 0.6rem; right: 0.85rem; font-family: 'JetBrains Mono', monospace; font-size: 0.52rem; color: #1E293B; letter-spacing: 0.08em; text-transform: uppercase; }
.fc-back-tag  { font-size: 0.57rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.35rem; }
.fc-back-name { font-size: 0.84rem; font-weight: 700; color: var(--txt); margin-bottom: 0.45rem; }
.fc-back-desc { font-size: 0.69rem; color: #94A3B8; line-height: 1.65; flex: 1; }
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
.sec-title { font-size: 0.68rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }
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

/* Info panel */
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
.hero-wrap { padding: 0.5rem 0 1rem 0; }
.hero-title { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.025em; background: linear-gradient(115deg, #E2E8F0 0%, #4F8EF7 55%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 0.22rem 0; line-height: 1.15; }
.hero-sub  { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted); margin: 0 0 0.65rem 0; }
.hero-chips { display: flex; gap: 1.2rem; flex-wrap: wrap; }
.hero-chip  { font-family: 'JetBrains Mono', monospace; font-size: 0.63rem; color: var(--muted); display: flex; align-items: center; gap: 0.32rem; }
.hero-chip .cd { width: 5px; height: 5px; border-radius: 50%; }

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(5,8,19,0.98) !important; border-right: 1px solid var(--bdr) !important; }
[data-testid="stSidebar"] * { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stMultiSelect [data-baseweb="tag"] { background: rgba(79,142,247,0.14) !important; border: 1px solid rgba(79,142,247,0.28) !important; color: #4F8EF7 !important; }
.stRadio label { color: #E2E8F0 !important; font-size: 0.82rem !important; }
</style>

<!-- Ambient blobs -->
<div class="blob blob-1"></div>
<div class="blob blob-2"></div>
<div class="blob blob-3"></div>

<!-- ═══ NAVBAR ═══ -->
<nav class="qp-nav">
  <div class="qp-brand">RISK <span class="hi">ENGINE</span></div>
  <div class="qp-links">
    <span class="nav-btn active" data-sec="sec-kpi"       onclick="navTo(this,'sec-kpi')"><span class="ni">◈</span>Overview</span>
    <span class="nav-btn"        data-sec="sec-perf"      onclick="navTo(this,'sec-perf')"><span class="ni">↗</span>Performance</span>
    <span class="nav-btn"        data-sec="sec-analytics" onclick="navTo(this,'sec-analytics')"><span class="ni">◎</span>Analytics</span>
    <span class="nav-btn"        data-sec="sec-risk"      onclick="navTo(this,'sec-risk')"><span class="ni">⚡</span>Risk</span>
    <span class="nav-btn"        data-sec="sec-portfolio" onclick="navTo(this,'sec-portfolio')"><span class="ni">◉</span>Portfolio</span>
  </div>
  <div class="qp-right">
    <div class="regime-chip"><span class="rdot"></span>HIGH VOL REGIME</div>
  </div>
</nav>

<script>
function navTo(el, id) {
  document.querySelectorAll('.nav-btn').forEach(function(b){ b.classList.remove('active'); });
  el.classList.add('active');
  var t = document.getElementById(id);
  if (t) t.scrollIntoView({ behavior:'smooth', block:'start' });
}
function toggleInfo(id) {
  var el = document.getElementById(id);
  if (!el) return;
  var open = el.classList.contains('open');
  el.classList.toggle('open', !open);
  if (!open) { el.style.animation='none'; void el.offsetHeight; el.style.animation=''; }
}
// Scroll-spy: highlight active nav item
setTimeout(function() {
  var ids = ['sec-kpi','sec-perf','sec-analytics','sec-risk','sec-portfolio'];
  var obs = new IntersectionObserver(function(entries) {
    entries.forEach(function(e) {
      if (e.isIntersecting) {
        var sid = e.target.id;
        document.querySelectorAll('.nav-btn').forEach(function(b) {
          b.classList.toggle('active', b.getAttribute('data-sec') === sid);
        });
      }
    });
  }, { threshold: 0.4, rootMargin: '-56px 0px -45% 0px' });
  ids.forEach(function(id){ var el=document.getElementById(id); if(el) obs.observe(el); });
}, 1400);
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","JPM","XOM","UNH","V","META","BRK.B","LLY"]

PLAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#64748B", size=11),
    margin=dict(l=10, r=10, t=38, b=10),
    legend=dict(
        bgcolor="rgba(10,15,30,0.82)", bordercolor="rgba(79,142,247,0.15)",
        borderwidth=1, font=dict(size=11, color="#E2E8F0"),
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
    ),
    xaxis=dict(showgrid=False, zeroline=False, showspikes=True,
               spikecolor="#4F8EF7", spikethickness=1, spikedash="dot"),
    yaxis=dict(showgrid=True, gridcolor="rgba(79,142,247,0.06)",
               zeroline=False, showspikes=True,
               spikecolor="#4F8EF7", spikethickness=1, spikedash="dot"),
)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def gen_equity(start, end, risk, assets_t):
    np.random.seed(42)
    dates = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    n = len(dates)
    if n < 10:
        n = 1260; dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    ew  = np.random.normal(0.00038, 0.0082, n)
    mpt = np.random.normal(0.00055, 0.0075*(1-risk/200), n)
    gbm = np.random.normal(0.00072, 0.0065*(1-risk/300), n)
    cs, ce = int(n*0.38), int(n*0.48)
    ew [cs:ce] = np.random.normal(-0.0055, 0.018, ce-cs)
    mpt[cs:ce] = np.random.normal(-0.0045, 0.017, ce-cs)
    gbm[cs:ce] = np.random.normal(-0.0035, 0.016, ce-cs)
    df = pd.DataFrame({
        "Date":             dates[:n],
        "Equal Weight":     100*np.cumprod(1+ew),
        "Markowitz MPT":    100*np.cumprod(1+mpt),
        "LightGBM Dynamic": 100*np.cumprod(1+gbm),
    })
    return df, (dates[cs], dates[ce-1])

@st.cache_data(show_spinner=False)
def gen_kpis(df):
    r  = df["LightGBM Dynamic"].pct_change().dropna()
    n  = len(r)
    rm = df["LightGBM Dynamic"].cummax()
    dd = (df["LightGBM Dynamic"] - rm) / rm
    return {
        "cagr":   (df["LightGBM Dynamic"].iloc[-1]/100)**(252/n)-1,
        "sharpe": r.mean()/r.std()*np.sqrt(252),
        "maxdd":  dd.min(),
        "var95":  np.percentile(r, 5)*df["LightGBM Dynamic"].iloc[-1]/100,
    }

@st.cache_data(show_spinner=False)
def gen_frontier(n=8000):
    np.random.seed(7)
    vols = np.random.beta(2,5,n)*0.35+0.06
    sh   = np.random.beta(3,2,n)*3.2+0.3
    ret  = np.clip(vols*sh+np.random.normal(0,0.008,n), 0.03, 0.55)
    mask = (ret > vols*0.35)|(vols < 0.13)
    return pd.DataFrame({"v":vols[mask],"r":ret[mask],"s":sh[mask]})

@st.cache_data(show_spinner=False)
def gen_shap():
    feats = ["VIX 30d Moving Avg","Tech/Energy Corr","Asset Momentum 12M",
             "Yield Curve Slope","Credit Spread Delta","FX Vol Index",
             "Macro Surprise","Liquidity Ratio","Sector Rotation","Earnings Rev."]
    np.random.seed(99)
    vals = sorted(np.random.randn(len(feats))*0.18, key=abs)
    return pd.DataFrame({"f":feats,"v":vals})

@st.cache_data(show_spinner=False)
def gen_corr(assets_t):
    assets = list(assets_t); np.random.seed(21); n = len(assets)
    c = np.corrcoef(np.random.randn(n,n)); np.fill_diagonal(c,1.)
    return pd.DataFrame(c, index=assets, columns=assets)

@st.cache_data(show_spinner=False)
def gen_risk(df):
    r = df.set_index("Date")["LightGBM Dynamic"].pct_change()
    return pd.DataFrame({
        "Date": df["Date"],
        "v21":  r.rolling(21).std()*np.sqrt(252)*100,
        "v63":  r.rolling(63).std()*np.sqrt(252)*100,
    })


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def apply_layout(fig, h=385, **kw):
    fig.update_layout(**{**PLAYOUT, "height": h, **kw})
    return fig

def anchor(sec_id):
    st.markdown(f'<div id="{sec_id}" class="sec-anchor"></div>', unsafe_allow_html=True)

def sec_header(title, badge="", color="#4F8EF7", info_id=None, info_html=""):
    info_btn_html = ""
    panel_html    = ""
    if info_id:
        info_btn_html = f'<button class="info-btn" onclick="toggleInfo(\'{info_id}\')">ⓘ</button>'
        panel_html    = f'<div id="{info_id}" class="info-panel">{info_html}</div>'
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="sec-hdr">
      <div class="sec-left">
        <div class="sec-pip" style="background:{color};box-shadow:0 0 8px {color}55;"></div>
        <span class="sec-title">{title}</span>
        {badge_html}
      </div>
      {info_btn_html}
    </div>
    {panel_html}
    """, unsafe_allow_html=True)

def flip_card(card_id, bar_css, lbl, icon, val_html, val_cls,
              dlt_html, back_tag, back_tag_cls, back_name, desc, eq):
    st.markdown(f"""
    <div class="flip-wrap" id="fc-{card_id}" onclick="this.classList.toggle('flipped')">
      <div class="flip-inner">
        <div class="flip-face flip-front">
          <div class="fc-bar" style="background:{bar_css};"></div>
          <div class="fc-row1">
            <span class="fc-lbl">{lbl}</span>
            <span class="fc-ico">{icon}</span>
          </div>
          <div class="fc-val {val_cls}">{val_html}</div>
          <div class="fc-dlt">{dlt_html}</div>
          <div class="fc-hint">CLICK TO EXPLAIN</div>
        </div>
        <div class="flip-face flip-back">
          <div class="fc-back-tag {back_tag_cls}">{back_tag}</div>
          <div class="fc-back-name">{back_name}</div>
          <div class="fc-back-desc">{desc}</div>
          <div class="fc-back-eq">{eq}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Strategy Parameters")
    st.markdown("---")
    selected_assets = st.multiselect(
        "Asset Universe", options=TICKERS,
        default=["AAPL","MSFT","NVDA","JPM","XOM","GOOGL","V","META"],
    )
    if not selected_assets:
        selected_assets = TICKERS[:8]
    st.markdown("---")
    today      = date.today()
    start_date = st.date_input("Start Date", value=today - timedelta(days=5*365))
    end_date   = st.date_input("End Date",   value=today)
    st.markdown("---")
    risk_tol  = st.slider("Target Max Volatility (%)", 5, 40, 18, 1)
    st.markdown("---")
    view_mode = st.radio("View Mode",
                         ["📊 Backtest Performance", "🔮 Live Regime Forecast"], index=0)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.6rem;color:#1E293B;text-align:center;line-height:2;
                font-family:'JetBrains Mono',monospace;">
        v2.1.0 · mock data<br>backend pending<br>© 2025 IDSE Quant Lab
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE DATA
# ─────────────────────────────────────────────────────────────────────────────
at       = tuple(selected_assets)
eq_df, reg = gen_equity(start_date, end_date, risk_tol, at)
kpis     = gen_kpis(eq_df)
front    = gen_frontier()
shap_df  = gen_shap()
corr_df  = gen_corr(at)
risk_df  = gen_risk(eq_df)
is_bt    = "Backtest" in view_mode

PALETTE  = ["#4F8EF7","#34D399","#FBBF24","#FB7185","#A78BFA",
            "#22D3EE","#F97316","#E879F9","#38BDF8","#84CC16","#FCD34D","#6EE7B7"]


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-title">Quant Portfolio Risk Engine</div>
  <div class="hero-sub">DYNAMIC ASSET ALLOCATION &nbsp;·&nbsp; ML-DRIVEN RISK MANAGEMENT</div>
  <div class="hero-chips">
    <div class="hero-chip"><div class="cd" style="background:#4F8EF7;"></div>{len(selected_assets)} Assets</div>
    <div class="hero-chip"><div class="cd" style="background:#34D399;"></div>Vol Cap: {risk_tol}%</div>
    <div class="hero-chip"><div class="cd" style="background:#A78BFA;"></div>{"Backtest" if is_bt else "Forecast"}</div>
    <div class="hero-chip"><div class="cd" style="background:#FB7185;"></div>{date.today().strftime('%d %b %Y')}</div>
  </div>
</div>
<div class="glow-div"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — KPI FLIP CARDS
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-kpi")
sec_header("Performance Snapshot", badge="LIVE MOCK", color="#4F8EF7",
           info_id="info-kpi", info_html="""
<h4>📊 Performance Snapshot</h4>
<p>Four key risk-adjusted performance indicators for the <b>LightGBM Dynamic</b> strategy over
the selected backtest window. All figures are annualised. <b>Click any card</b> to flip it and
see the full definition and formula.</p>
<span class="eq">Base currency: USD &nbsp;·&nbsp; Confidence: 95% &nbsp;·&nbsp; Calendar: business days</span>
""")

c1, c2, c3, c4 = st.columns(4)
with c1:
    flip_card("cagr",
        bar_css="linear-gradient(90deg,#34D399,#059669)",
        lbl="Strategy CAGR", icon="📈",
        val_html=f"{kpis['cagr']*100:+.1f}%", val_cls="c-green",
        dlt_html='<span class="up">▲ +4.2pp vs SPY</span>',
        back_tag="RETURN METRIC", back_tag_cls="c-green",
        back_name="Compound Annual Growth Rate",
        desc="CAGR smooths compounded returns over the full horizon, showing the hypothetical constant annual growth rate. Higher CAGR relative to benchmarks signals consistent alpha generation.",
        eq="CAGR = (NAV_end / NAV_start)^(252/N) − 1")

with c2:
    flip_card("sharpe",
        bar_css="linear-gradient(90deg,#4F8EF7,#2563EB)",
        lbl="Sharpe Ratio", icon="⚡",
        val_html=f"{kpis['sharpe']:.2f}", val_cls="c-blue",
        dlt_html='<span class="up">▲ 0.38 vs Equal Weight</span>',
        back_tag="RISK-ADJUSTED", back_tag_cls="c-blue",
        back_name="Sharpe Ratio",
        desc="Measures excess return earned per unit of total risk (σ). A ratio above 1.0 is good; above 2.0 is excellent. Assumes risk-free rate ≈ 0 for simplicity in backtest mode.",
        eq="Sharpe = (R_p − R_f) / σ_p × √252")

with c3:
    flip_card("maxdd",
        bar_css="linear-gradient(90deg,#FB7185,#BE123C)",
        lbl="Max Drawdown", icon="📉",
        val_html=f"{kpis['maxdd']*100:.1f}%", val_cls="c-rose",
        dlt_html='<span class="up">▲ 6.1pp better vs EW</span>',
        back_tag="DOWNSIDE RISK", back_tag_cls="c-rose",
        back_name="Maximum Drawdown",
        desc="The worst peak-to-trough decline in portfolio value during the backtest. Quantifies the maximum loss an investor would have sustained buying at the peak and selling at the nadir.",
        eq="MDD = min_t [ (NAV_t − max_{τ≤t} NAV_τ) / max NAV ]")

with c4:
    flip_card("var",
        bar_css="linear-gradient(90deg,#FBBF24,#B45309)",
        lbl="Next-Day 95% VaR", icon="🎯",
        val_html=f"${abs(kpis['var95'])*1e6:,.0f}", val_cls="c-amber",
        dlt_html='<span class="down">▼ Per $1M notional</span>',
        back_tag="TAIL RISK", back_tag_cls="c-amber",
        back_name="Value at Risk (95%)",
        desc="Historical simulation VaR: the minimum expected loss on the worst 5% of days, scaled to $1M notional. This is the next-day forecast from the rolling empirical return distribution.",
        eq="VaR₀.₀₅ = Percentile(R_daily, 5%) × Notional")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — EQUITY CURVES
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-perf")
sec_header(
    "Equity Curve Comparison" if is_bt else "Live Regime Forecast",
    badge="BACKTEST" if is_bt else "FORECAST",
    color="#4F8EF7",
    info_id="info-perf", info_html="""
<h4>📈 Equity Curve Comparison</h4>
<p>Normalised NAV curves (base $100) for three construction methodologies over the backtest window.
The <b style="color:#FB7185;">shaded red zone</b> marks a simulated crash regime detected by the
Hidden Markov Model classifier.</p>
<p><b>Equal Weight</b> — naive 1/N benchmark &nbsp;·&nbsp; <b>Markowitz MPT</b> — mean-variance
optimal &nbsp;·&nbsp; <b>LightGBM Dynamic</b> — ML tactical allocation (hero strategy)</p>
<span class="eq">Hover for cross-hair &nbsp;·&nbsp; Regime: 2-state HMM on daily returns</span>
""")

fig_main = go.Figure()
fig_main.add_vrect(
    x0=str(reg[0].date()), x1=str(reg[1].date()),
    fillcolor="rgba(251,113,133,0.05)", layer="below", line_width=0,
    annotation_text="⚠ CRASH REGIME", annotation_position="top left",
    annotation=dict(font=dict(color="#FB7185",size=9,family="JetBrains Mono"),
                    bgcolor="rgba(251,113,133,0.1)", bordercolor="rgba(251,113,133,0.25)",
                    borderwidth=1, borderpad=4),
)
fig_main.add_trace(go.Scatter(
    x=eq_df["Date"], y=eq_df["Equal Weight"], name="Equal Weight",
    line=dict(color="#334155",width=1.6,dash="dot"),
    hovertemplate="<b>Equal Weight</b><br>%{x|%d %b %Y}<br>$%{y:.1f}<extra></extra>",
))
fig_main.add_trace(go.Scatter(
    x=eq_df["Date"], y=eq_df["Markowitz MPT"], name="Markowitz MPT",
    line=dict(color="#FBBF24",width=2.2),
    hovertemplate="<b>Markowitz MPT</b><br>%{x|%d %b %Y}<br>$%{y:.1f}<extra></extra>",
))
fig_main.add_trace(go.Scatter(
    x=eq_df["Date"], y=eq_df["LightGBM Dynamic"], name="LightGBM Dynamic",
    line=dict(color="#4F8EF7",width=3.2),
    fill="tozeroy", fillcolor="rgba(79,142,247,0.04)",
    hovertemplate="<b>LightGBM Dynamic</b><br>%{x|%d %b %Y}<br>$%{y:.1f}<extra></extra>",
))
apply_layout(fig_main, h=415,
             yaxis=dict(tickprefix="$",showgrid=True,
                        gridcolor="rgba(79,142,247,0.06)",zeroline=False),
             hovermode="x unified")
st.plotly_chart(fig_main, width='stretch', config={"displayModeBar": False})
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — EFFICIENT FRONTIER + SHAP
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-analytics")
col_ef, col_sh = st.columns([1,1], gap="large")

with col_ef:
    sec_header("Monte Carlo Efficient Frontier", badge="8K SIMS", color="#A78BFA",
               info_id="info-ef", info_html="""
<h4>◎ Monte Carlo Efficient Frontier</h4>
<p>8,000 randomly-weighted portfolios plotted in risk-return space, colored by Sharpe Ratio.
The upper-left boundary defines the <b>Efficient Frontier</b> — maximum return for given risk.</p>
<p><b style="color:#FB7185;">★ Red</b> = Markowitz optimal point &nbsp;·&nbsp;
<b style="color:#4F8EF7;">★ Blue</b> = current ML strategy allocation</p>
<span class="eq">Weights via Dirichlet sampling &nbsp;·&nbsp; Cov: rolling 252-day window</span>
""")
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=front["v"]*100, y=front["r"]*100, mode="markers",
        marker=dict(
            color=front["s"],
            colorscale=[[0,"#0A0F1E"],[0.3,"#1E3A5F"],[0.6,"#2563EB"],[1,"#4F8EF7"]],
            size=3, opacity=0.6,
            colorbar=dict(title=dict(text="Sharpe",font=dict(size=10,color="#64748B")),
                          tickfont=dict(family="JetBrains Mono",size=9,color="#64748B"),
                          thickness=9, len=0.65),
        ),
        name="Portfolios",
        hovertemplate="Vol:%{x:.1f}%  Ret:%{y:.1f}%<extra></extra>",
    ))
    oi = front["s"].idxmax()
    ov, or_ = front.loc[oi,"v"]*100, front.loc[oi,"r"]*100
    for x,y,col,txt,pos in [
        (ov,  or_,     "#FB7185","MPT Optimal","top right"),
        (ov*0.87, or_*1.09, "#4F8EF7","ML Portfolio","top left"),
    ]:
        fig_ef.add_trace(go.Scatter(
            x=[x],y=[y],mode="markers+text",
            marker=dict(symbol="star",size=18,color=col,line=dict(color="#fff",width=1.4)),
            text=[txt],textposition=pos,
            textfont=dict(family="JetBrains Mono",size=9,color=col),
            name=txt,
            hovertemplate=f"<b>{txt}</b><br>Vol:%{{x:.1f}}%<extra></extra>",
        ))
    apply_layout(fig_ef, h=375,
                 xaxis_title="Annualised Volatility (%)",
                 yaxis_title="Expected Return (%)",
                 xaxis=dict(ticksuffix="%",showgrid=False),
                 yaxis=dict(ticksuffix="%",gridcolor="rgba(79,142,247,0.06)"),
                 legend=dict(orientation="v",x=0.01,y=0.99,
                             xanchor="left",yanchor="top",font=dict(size=10)))
    st.plotly_chart(fig_ef, width='stretch', config={"displayModeBar": False})

with col_sh:
    sec_header("LightGBM Allocation Drivers", badge="LIVE SHAP", color="#22D3EE",
               info_id="info-shap", info_html="""
<h4>⚡ SHAP Feature Importance</h4>
<p>SHAP values decompose the model's current allocation decision into individual feature
contributions. <b style="color:#34D399;">Green</b> = increases risk allocation;
<b style="color:#FB7185;">Red</b> = reduces risk allocation.</p>
<p>Features include macro signals, momentum factors, and cross-asset correlation metrics
updated in real-time by the LightGBM scoring pipeline.</p>
<span class="eq">SHAP: phi_i = E[f(x)] with feature i vs without feature i</span>
""")
    sd = shap_df.sort_values("v")
    fig_sh = go.Figure()
    fig_sh.add_trace(go.Bar(
        x=sd["v"], y=sd["f"], orientation="h",
        marker=dict(color=["#FB7185" if v<0 else "#34D399" for v in sd["v"]],
                    opacity=0.82, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in sd["v"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono",size=10,color="#E2E8F0"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig_sh.add_vline(x=0, line_color="#334155", line_width=1.5)
    apply_layout(fig_sh, h=375,
                 xaxis_title="SHAP Value",
                 yaxis=dict(showgrid=False,tickfont=dict(family="Plus Jakarta Sans",size=10)),
                 xaxis=dict(showgrid=True,gridcolor="rgba(79,142,247,0.06)"),
                 showlegend=False, margin=dict(l=170,r=60,t=38,b=40))
    st.plotly_chart(fig_sh, width='stretch', config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — ROLLING RISK + CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-risk")
col_rv, col_hm = st.columns([1.1,0.9], gap="large")

with col_rv:
    sec_header("Rolling Risk Decomposition", badge="DYNAMIC", color="#34D399",
               info_id="info-risk", info_html="""
<h4>⚡ Rolling Volatility Decomposition</h4>
<p>Realised portfolio volatility on <b style="color:#4F8EF7;">21-day</b> (responsive) and
<b style="color:#A78BFA;">63-day</b> (stable) rolling windows, annualised by ×√252.
The <b style="color:#FB7185;">dashed line</b> is your volatility target; the band above is the breach zone.</p>
<p>The optimizer uses the 63-day estimate as a hard constraint. Breaches trigger automatic
portfolio rebalancing toward lower-risk assets.</p>
<span class="eq">σ_ann = σ_daily × √252 &nbsp;·&nbsp; EWMA alternative configurable</span>
""")
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(x=risk_df["Date"],y=risk_df["v21"],name="21d Realised Vol",
        line=dict(color="#4F8EF7",width=2.2),
        hovertemplate="21d: <b>%{y:.1f}%</b><extra></extra>"))
    fig_rv.add_trace(go.Scatter(x=risk_df["Date"],y=risk_df["v63"],name="63d Realised Vol",
        line=dict(color="#A78BFA",width=2.2,dash="dash"),
        hovertemplate="63d: <b>%{y:.1f}%</b><extra></extra>"))
    fig_rv.add_hrect(y0=risk_tol,y1=risk_tol+5,fillcolor="rgba(251,113,133,0.05)",line_width=0)
    fig_rv.add_hline(y=risk_tol,line=dict(color="#FB7185",width=1.5,dash="dot"),
                     annotation_text=f"Target {risk_tol}%",annotation_position="right",
                     annotation_font=dict(color="#FB7185",size=9,family="JetBrains Mono"))
    apply_layout(fig_rv,h=318,
                 yaxis=dict(ticksuffix="%",gridcolor="rgba(79,142,247,0.06)"),
                 xaxis=dict(showgrid=False),hovermode="x unified")
    st.plotly_chart(fig_rv, width='stretch', config={"displayModeBar": False})

with col_hm:
    sec_header("Asset Correlation Matrix", badge=f"{len(selected_assets)} ASSETS",
               color="#22D3EE", info_id="info-corr", info_html="""
<h4>◈ Asset Correlation Matrix</h4>
<p>Pairwise Pearson correlations on rolling 252-day windows.
<b style="color:#4F8EF7;">Blue (+1.0)</b> = high co-movement;
<b style="color:#FB7185;">Red (−1.0)</b> = inverse relationship.</p>
<p>Low inter-asset correlation is the foundation of Markowitz diversification.
The optimizer exploits these relationships to construct efficient portfolios.</p>
<span class="eq">ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y) ∈ [−1, 1]</span>
""")
    fig_hm = go.Figure(go.Heatmap(
        z=corr_df.values, x=list(corr_df.columns), y=list(corr_df.index),
        colorscale=[[0,"#FB7185"],[0.25,"#5B1A28"],[0.5,"#0A0F1E"],[0.75,"#1E3A5F"],[1,"#4F8EF7"]],
        zmin=-1, zmax=1,
        text=np.round(corr_df.values,2), texttemplate="%{text:.2f}",
        textfont=dict(size=8,family="JetBrains Mono",color="#E2E8F0"),
        hovertemplate="%{y} vs %{x}<br>rho = <b>%{z:.3f}</b><extra></extra>",
        colorbar=dict(title=dict(text="rho",font=dict(size=11,color="#64748B")),
                      tickfont=dict(family="JetBrains Mono",size=9,color="#64748B"),
                      thickness=9,len=0.8),
    ))
    apply_layout(fig_hm,h=318,
                 xaxis=dict(showgrid=False,tickfont=dict(size=9,family="JetBrains Mono"),tickangle=-45),
                 yaxis=dict(showgrid=False,tickfont=dict(size=9,family="JetBrains Mono")),
                 margin=dict(l=8,r=58,t=38,b=62),showlegend=False)
    st.plotly_chart(fig_hm, width='stretch', config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 — ALLOCATION DONUT + DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────
anchor("sec-portfolio")
col_pie, col_dd = st.columns([0.82,1.18], gap="large")

with col_pie:
    sec_header("Current ML Allocation", badge="OPTIMAL", color="#4F8EF7",
               info_id="info-alloc", info_html="""
<h4>◉ Current ML Portfolio Weights</h4>
<p>Optimal weights output by the LightGBM allocation model for the current market regime.
Constraints: min 2% per asset, max 40%, Σwᵢ = 100%. Rebalanced <b>weekly</b>.</p>
<p>The model uses regime classification plus factor signals (momentum, value, quality)
to dynamically tilt toward higher-conviction positions.</p>
<span class="eq">w* = argmax CAGR &nbsp; s.t. &nbsp; σ_p ≤ target, Σwᵢ = 1, wᵢ ≥ 0</span>
""")
    np.random.seed(55)
    wts = np.random.dirichlet(np.ones(len(selected_assets))*2)
    adf = pd.DataFrame({"a":selected_assets,"w":wts}).sort_values("w",ascending=False)
    fig_pie = go.Figure(go.Pie(
        labels=adf["a"], values=adf["w"], hole=0.64,
        marker=dict(colors=PALETTE[:len(selected_assets)],line=dict(color="#050813",width=2)),
        textinfo="label+percent",
        textfont=dict(family="JetBrains Mono",size=9,color="#E2E8F0"),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        direction="clockwise", sort=True,
    ))
    fig_pie.add_annotation(
        text=f"<b>{len(selected_assets)}</b><br><span style='font-size:9px;color:#64748B'>ASSETS</span>",
        x=0.5,y=0.5,showarrow=False,
        font=dict(family="JetBrains Mono",size=20,color="#E2E8F0"),
    )
    apply_layout(fig_pie,h=320,showlegend=False,margin=dict(l=8,r=8,t=38,b=8))
    st.plotly_chart(fig_pie, width='stretch', config={"displayModeBar": False})

with col_dd:
    sec_header("Drawdown Analysis", badge="UNDERWATER", color="#FB7185",
               info_id="info-dd", info_html="""
<h4>📉 Drawdown Analysis</h4>
<p>The percentage decline from the previous all-time portfolio high at each point in time.
At 0% the portfolio is at a new peak; deeper troughs represent larger unrealised losses from prior highs.</p>
<p>The annotation marks the <b style="color:#FB7185;">Maximum Drawdown</b> — the single worst
peak-to-trough decline during the entire backtest period.</p>
<span class="eq">DD_t = (NAV_t − max_{τ≤t} NAV_τ) / max_{τ≤t} NAV_τ × 100</span>
""")
    rm  = eq_df["LightGBM Dynamic"].cummax()
    dd  = (eq_df["LightGBM Dynamic"] - rm) / rm * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=eq_df["Date"],y=dd,
        fill="tozeroy", fillcolor="rgba(251,113,133,0.09)",
        line=dict(color="#FB7185",width=1.8),name="Drawdown",
        hovertemplate="<b>Drawdown</b><br>%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    mi = int(dd.idxmin())
    fig_dd.add_annotation(
        x=eq_df["Date"].iloc[mi], y=dd.iloc[mi],
        text=f"Max DD: {dd.iloc[mi]:.1f}%",
        showarrow=True,arrowhead=2,arrowcolor="#FB7185",
        font=dict(color="#FB7185",size=10,family="JetBrains Mono"),
        bgcolor="rgba(251,113,133,0.12)",bordercolor="#FB7185",
        borderwidth=1,borderpad=4,ay=-38,
    )
    apply_layout(fig_dd,h=320,
                 yaxis=dict(ticksuffix="%",gridcolor="rgba(79,142,247,0.06)",
                            zeroline=True,zerolinecolor="#1E293B",zerolinewidth=1),
                 xaxis=dict(showgrid=False),showlegend=False,hovermode="x")
    st.plotly_chart(fig_dd, width='stretch', config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glow-div" style="margin-top:2.5rem;"></div>
<div style="display:flex;justify-content:space-between;align-items:center;padding:0 0 1.2rem 0;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.59rem;color:#1E293B;">
    Portfolio Risk Engine &nbsp;·&nbsp; Mock Data &nbsp;·&nbsp; Backend Integration Pending
  </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.59rem;color:#1E293B;">
    Streamlit + Plotly &nbsp;·&nbsp; IDSE Quant Lab © 2025
  </span>
</div>
""", unsafe_allow_html=True)
