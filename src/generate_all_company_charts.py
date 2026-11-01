"""
src/generate_all_company_charts.py
==================================
Generates an institutional-grade, dark-themed interactive visual dashboard
(`all_companies_charts.html`) displaying individual charts for all 52 assets
(50 NIFTY companies + NIFTY 500 index + INDIA VIX).

Features:
  - 52 Individual Interactive Stock Cards with Sparkline Trajectories
  - Search by Ticker or Company Name
  - Sector Filtering (IT, Financial Services, Auto, Pharma, Energy, Metals, etc.)
  - Status Filtering (MPT Active, Late IPO, Benchmark)
  - Sorting (Highest Return, Lowest Drawdown, Name, Price)
  - Interactive Modal Deep Dive for every single company
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
COMPANIES_DIR = BASE_DIR / "companies"
METADATA_PATH = COMPANIES_DIR / "stock_metadata.csv"
OUT_PATH = BASE_DIR / "all_companies_charts.html"


def load_metadata() -> dict[str, dict]:
    """Load sector, full company name, and ISIN from stock_metadata.csv."""
    meta = {}
    if METADATA_PATH.exists():
        df_meta = pd.read_csv(METADATA_PATH)
        for _, row in df_meta.iterrows():
            sym = str(row.get("Symbol", "")).strip()
            meta[sym] = {
                "name": str(row.get("Company Name", sym)).strip(),
                "industry": str(row.get("Industry", "OTHER")).strip(),
                "isin": str(row.get("ISIN Code", "")).strip(),
            }
    return meta


def process_all_assets() -> list[dict]:
    """Process all 50 company CSV files + NIFTY 500 + INDIA VIX."""
    meta = load_metadata()
    csv_files = sorted(glob.glob(str(COMPANIES_DIR / "*.csv")))
    
    # 7 known late IPOs dropped by the 30% rule
    DROPPED_IPOS = {"ADANIPORTS", "BAJAJ-AUTO", "BAJAJFINSV", "COALINDIA", "NESTLEIND", "POWERGRID", "TECHM"}
    
    results = []

    for fpath in csv_files:
        stem = Path(fpath).stem
        if stem.lower() in ("nifty50_all", "stock_metadata"):
            continue

        ticker = stem
        company_info = meta.get(ticker, {
            "name": ticker,
            "industry": "EQUITY",
            "isin": "",
        })

        # Read CSV
        try:
            df = pd.read_csv(fpath, usecols=lambda c: c.strip() in ("Date", "Close", "Volume"))
            df.columns = df.columns.str.strip()

            if "Date" not in df.columns or "Close" not in df.columns or len(df.dropna(subset=["Close"])) < 2:
                # Thin or empty file (e.g. INFRATEL)
                results.append({
                    "ticker": ticker,
                    "name": company_info["name"],
                    "industry": company_info["industry"],
                    "status": "Insufficient Data",
                    "status_color": "gray",
                    "rows": 0,
                    "start_date": "N/A",
                    "end_date": "N/A",
                    "first_price": 0,
                    "last_price": 0,
                    "total_return_pct": 0,
                    "max_drawdown_pct": 0,
                    "ath": 0,
                    "atl": 0,
                    "points": [],
                })
                continue

            # Clean and parse
            df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), errors="coerce")
            df["Close"] = (
                df["Close"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

            first_p = float(df["Close"].iloc[0])
            last_p = float(df["Close"].iloc[-1])
            ath = float(df["Close"].max())
            atl = float(df["Close"].min())
            total_return = ((last_p - first_p) / first_p) * 100.0 if first_p > 0 else 0.0

            # Peak-to-trough max drawdown
            cummax = df["Close"].cummax()
            dd = (df["Close"] - cummax) / cummax * 100.0
            max_dd = float(dd.min())

            # Status
            if ticker in DROPPED_IPOS:
                status = "Late IPO (>30% Missing)"
                status_color = "rose"
            else:
                status = "MPT Active Survivor"
                status_color = "emerald"

            # Downsample to ~120-150 evenly spaced points for high-performance sparkline & chart
            step = max(1, len(df) // 120)
            downsampled = df.iloc[::step].copy()
            if downsampled.iloc[-1]["Date"] != df.iloc[-1]["Date"]:
                downsampled = pd.concat([downsampled, df.iloc[[-1]]])

            points = [
                {"d": d.strftime("%Y-%m-%d"), "c": round(p, 2)}
                for d, p in zip(downsampled["Date"], downsampled["Close"])
            ]

            results.append({
                "ticker": ticker,
                "name": company_info["name"],
                "industry": company_info["industry"],
                "status": status,
                "status_color": status_color,
                "rows": len(df),
                "start_date": df["Date"].iloc[0].strftime("%b %Y"),
                "end_date": df["Date"].iloc[-1].strftime("%b %Y"),
                "first_price": round(first_p, 2),
                "last_price": round(last_p, 2),
                "total_return_pct": round(total_return, 1),
                "max_drawdown_pct": round(max_dd, 1),
                "ath": round(ath, 2),
                "atl": round(atl, 2),
                "points": points,
            })

        except Exception as exc:
            print(f"[WARN] Error reading {ticker}: {exc}")

    # Also add the 2 Benchmark files from project root if present
    benchmarks = [
        ("NIFTY 500", BASE_DIR / "NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv", "BENCHMARK INDEX"),
        ("INDIA VIX", BASE_DIR / "INDIA VIX 01 Jan 2010 - 20 Mar 2026.csv", "VOLATILITY INDEX"),
    ]
    for b_name, b_path, b_industry in benchmarks:
        if b_path.exists():
            try:
                b_df = pd.read_csv(b_path, usecols=lambda c: c.strip() in ("Date", "Close"))
                b_df.columns = b_df.columns.str.strip()
                b_df["Date"] = pd.to_datetime(b_df["Date"].astype(str).str.strip(), dayfirst=True, errors="coerce")
                b_df["Close"] = (
                    b_df["Close"]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                    .pipe(pd.to_numeric, errors="coerce")
                )
                b_df = b_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

                first_p = float(b_df["Close"].iloc[0])
                last_p = float(b_df["Close"].iloc[-1])
                ath = float(b_df["Close"].max())
                atl = float(b_df["Close"].min())
                total_return = ((last_p - first_p) / first_p) * 100.0 if first_p > 0 else 0.0

                cummax = b_df["Close"].cummax()
                dd = (b_df["Close"] - cummax) / cummax * 100.0
                max_dd = float(dd.min())

                step = max(1, len(b_df) // 120)
                downsampled = b_df.iloc[::step].copy()
                if downsampled.iloc[-1]["Date"] != b_df.iloc[-1]["Date"]:
                    downsampled = pd.concat([downsampled, b_df.iloc[[-1]]])

                points = [
                    {"d": d.strftime("%Y-%m-%d"), "c": round(p, 2)}
                    for d, p in zip(downsampled["Date"], downsampled["Close"])
                ]

                results.append({
                    "ticker": b_name,
                    "name": f"{b_name} Index",
                    "industry": b_industry,
                    "status": "Market Benchmark",
                    "status_color": "sky",
                    "rows": len(b_df),
                    "start_date": b_df["Date"].iloc[0].strftime("%b %Y"),
                    "end_date": b_df["Date"].iloc[-1].strftime("%b %Y"),
                    "first_price": round(first_p, 2),
                    "last_price": round(last_p, 2),
                    "total_return_pct": round(total_return, 1),
                    "max_drawdown_pct": round(max_dd, 1),
                    "ath": round(ath, 2),
                    "atl": round(atl, 2),
                    "points": points,
                })
            except Exception as exc:
                print(f"[WARN] Error reading benchmark {b_name}: {exc}")

    return results


def generate_html(assets: list[dict]):
    """Generate self-contained, high-performance HTML dashboard."""
    json_data = json.dumps(assets)
    industries = sorted(list({a["industry"] for a in assets if a["industry"]}))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NIFTY 50 Multi-Asset Universe · Individual Stock Charts</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

  <style>
    :root {{
      --bg: #050813;
      --surf: #0A0F1E;
      --card-bg: rgba(10, 15, 30, 0.85);
      --card-hover: rgba(14, 22, 45, 0.95);
      --bdr: rgba(79, 142, 247, 0.12);
      --bdr-h: rgba(79, 142, 247, 0.35);
      --blue: #4F8EF7;
      --green: #34D399;
      --rose: #FB7185;
      --amber: #FBBF24;
      --cyan: #22D3EE;
      --violet: #A78BFA;
      --txt: #E2E8F0;
      --muted: #64748B;
      --nav-h: 60px;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      font-family: 'Plus Jakarta Sans', sans-serif;
      color: var(--txt);
      min-height: 100vh;
      overflow-x: hidden;
      padding-bottom: 4rem;
    }}

    /* Subtle background grid */
    body::before {{
      content: '';
      position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background-image: radial-gradient(rgba(79, 142, 247, 0.08) 1px, transparent 1px);
      background-size: 28px 28px;
    }}

    /* Header Nav */
    header {{
      position: sticky; top: 0; z-index: 100;
      height: var(--nav-h);
      background: rgba(5, 8, 19, 0.85);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--bdr);
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 2rem;
    }}
    .brand {{
      font-size: 0.85rem; font-weight: 800; letter-spacing: 0.15em;
      text-transform: uppercase; color: var(--muted);
      display: flex; align-items: center; gap: 0.5rem;
    }}
    .brand .hl {{ color: var(--blue); }}
    .badge {{
      font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600;
      padding: 3px 10px; border-radius: 20px;
      background: rgba(79,142,247,0.1); border: 1px solid rgba(79,142,247,0.25);
      color: var(--blue);
    }}

    .container {{
      max-width: 1560px; margin: 0 auto; padding: 2rem 2rem;
      position: relative; z-index: 1;
    }}

    /* Top stats banner */
    .hero-banner {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem; margin-bottom: 2rem;
    }}
    .stat-pill {{
      background: var(--card-bg);
      border: 1px solid var(--bdr);
      border-radius: 14px; padding: 1.1rem 1.3rem;
      transition: all 0.2s;
    }}
    .stat-pill:hover {{ border-color: var(--bdr-h); transform: translateY(-2px); }}
    .stat-lbl {{ font-size: 0.68rem; font-weight: 700; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem; }}
    .stat-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #FFFFFF; }}
    .stat-sub {{ font-size: 0.68rem; color: var(--muted); margin-top: 0.2rem; }}

    /* Filters Bar */
    .filter-section {{
      background: var(--card-bg);
      border: 1px solid var(--bdr);
      border-radius: 16px; padding: 1.25rem 1.5rem;
      margin-bottom: 2rem;
      display: flex; flex-direction: column; gap: 1rem;
    }}
    .filter-row1 {{
      display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
    }}
    .search-box {{
      flex: 1; min-width: 260px; position: relative;
    }}
    .search-input {{
      width: 100%; height: 42px;
      background: rgba(5, 8, 19, 0.7);
      border: 1px solid var(--bdr);
      border-radius: 10px; padding: 0 1rem 0 2.5rem;
      font-family: inherit; font-size: 0.85rem; color: var(--txt);
      outline: none; transition: border-color 0.2s;
    }}
    .search-input:focus {{ border-color: var(--blue); box-shadow: 0 0 10px rgba(79,142,247,0.2); }}
    .search-icon {{
      position: absolute; left: 0.9rem; top: 50%; transform: translateY(-50%);
      color: var(--muted); font-size: 0.9rem;
    }}

    .sort-select {{
      height: 42px; background: rgba(5, 8, 19, 0.7);
      border: 1px solid var(--bdr); border-radius: 10px;
      color: var(--txt); font-family: inherit; font-size: 0.82rem;
      padding: 0 1rem; outline: none; cursor: pointer;
    }}

    .filter-pills {{
      display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center;
    }}
    .filter-btn {{
      padding: 0.32rem 0.85rem; border-radius: 8px;
      font-size: 0.72rem; font-weight: 600; cursor: pointer;
      background: rgba(79, 142, 247, 0.05); border: 1px solid var(--bdr);
      color: var(--muted); transition: all 0.15s;
    }}
    .filter-btn:hover {{ color: var(--txt); background: rgba(79, 142, 247, 0.1); border-color: var(--bdr-h); }}
    .filter-btn.active {{
      background: rgba(79, 142, 247, 0.2);
      border-color: var(--blue); color: #FFFFFF;
    }}

    /* Grid of Company Cards */
    .grid-container {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
      gap: 1.25rem;
    }}

    .stock-card {{
      background: var(--card-bg);
      border: 1px solid var(--bdr);
      border-radius: 16px;
      padding: 1.25rem 1.4rem;
      cursor: pointer;
      transition: transform 0.22s cubic-bezier(0.16,1,0.3,1), border-color 0.22s, box-shadow 0.22s;
      position: relative; overflow: hidden;
      display: flex; flex-direction: column; justify-content: space-between;
    }}
    .stock-card:hover {{
      transform: translateY(-4px);
      border-color: var(--bdr-h);
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(79, 142, 247, 0.12);
    }}

    .card-top {{
      display: flex; justify-content: space-between; align-items: flex-start;
      margin-bottom: 0.6rem;
    }}
    .ticker-title {{
      font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 800;
      color: #FFFFFF; letter-spacing: -0.02em;
    }}
    .comp-name {{
      font-size: 0.72rem; color: var(--muted);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 210px;
    }}
    .sector-badge {{
      font-size: 0.58rem; font-weight: 700; letter-spacing: 0.08em;
      text-transform: uppercase; padding: 2px 8px; border-radius: 6px;
      background: rgba(167, 139, 250, 0.1); border: 1px solid rgba(167, 139, 250, 0.25);
      color: var(--violet); white-space: nowrap;
    }}

    .card-stats {{
      display: grid; grid-template-columns: 1fr 1fr 1fr;
      margin: 0.7rem 0 0.9rem 0;
      border-top: 1px solid rgba(255,255,255,0.04);
      border-bottom: 1px solid rgba(255,255,255,0.04);
      padding: 0.55rem 0;
    }}
    .cstat-item {{ display: flex; flex-direction: column; }}
    .cstat-lbl {{ font-size: 0.56rem; font-weight: 600; color: var(--muted); text-transform: uppercase; }}
    .cstat-val {{ font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: var(--txt); }}
    .cstat-val.up {{ color: var(--green); }}
    .cstat-val.down {{ color: var(--rose); }}

    .chart-container {{
      height: 95px; width: 100%; position: relative;
    }}

    .card-bottom {{
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 0.65rem; font-size: 0.62rem; color: var(--muted);
    }}
    .status-tag {{
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 0.62rem; font-weight: 600;
    }}
    .status-dot {{ width: 6px; height: 6px; border-radius: 50%; }}
    .dot-emerald {{ background: var(--green); box-shadow: 0 0 8px var(--green); }}
    .dot-rose    {{ background: var(--rose); box-shadow: 0 0 8px var(--rose); }}
    .dot-gray    {{ background: #64748B; }}
    .dot-sky     {{ background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }}

    /* Modal Deep Dive */
    .modal-overlay {{
      position: fixed; inset: 0; z-index: 1000;
      background: rgba(2, 4, 10, 0.88);
      backdrop-filter: blur(14px);
      display: none; align-items: center; justify-content: center;
      padding: 1.5rem;
    }}
    .modal-overlay.open {{ display: flex; animation: modalIn 0.2s cubic-bezier(0.16,1,0.3,1); }}
    @keyframes modalIn {{
      from {{ opacity: 0; transform: scale(0.96); }}
      to   {{ opacity: 1; transform: scale(1); }}
    }}
    .modal-card {{
      background: #0A0F22;
      border: 1px solid var(--bdr-h);
      border-radius: 20px;
      width: 100%; max-width: 1050px;
      max-height: 90vh; overflow-y: auto;
      box-shadow: 0 25px 60px rgba(0,0,0,0.8), 0 0 40px rgba(79,142,247,0.15);
      padding: 2rem; position: relative;
    }}
    .modal-close {{
      position: absolute; top: 1.5rem; right: 1.5rem;
      background: rgba(255,255,255,0.06); border: 1px solid var(--bdr);
      width: 32px; height: 32px; border-radius: 50%;
      color: var(--txt); cursor: pointer; font-size: 1rem;
      display: flex; align-items: center; justify-content: center;
      transition: background 0.15s;
    }}
    .modal-close:hover {{ background: rgba(255,255,255,0.15); }}
    .modal-header {{ margin-bottom: 1.5rem; }}
    .modal-ticker {{ font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; color: #FFFFFF; }}
    .modal-stats-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 0.8rem; margin: 1.2rem 0;
    }}
    .mstat-box {{
      background: rgba(5, 8, 19, 0.7); border: 1px solid var(--bdr);
      border-radius: 10px; padding: 0.7rem 0.9rem;
    }}
    .mstat-lbl {{ font-size: 0.6rem; font-weight: 700; color: var(--muted); text-transform: uppercase; }}
    .mstat-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-top: 0.2rem; }}
    .modal-chart-wrap {{ height: 420px; width: 100%; position: relative; margin-top: 1rem; }}
  </style>
</head>
<body>

  <!-- Top Nav -->
  <header>
    <div class="brand">
      <span>QUANT PORTFOLIO RISK ENGINE</span>
      <span class="hl">/</span>
      <span style="color:#FFFFFF;">ALL 52 ASSETS EXPLORER</span>
    </div>
    <div style="display:flex; align-items:center; gap:1rem;">
      <span class="badge">NIFTY 50 UNIVERSE</span>
      <span class="badge" style="color:var(--green); border-color:rgba(52,211,153,0.3); background:rgba(52,211,153,0.08);">52 CHARTS ACTIVE</span>
    </div>
  </header>

  <div class="container">

    <!-- Stat banner -->
    <div class="hero-banner">
      <div class="stat-pill">
        <div class="stat-lbl">Total Assets Tracked</div>
        <div class="stat-val">{len(assets)}</div>
        <div class="stat-sub">50 Equities + 2 Benchmarks</div>
      </div>
      <div class="stat-pill">
        <div class="stat-lbl">MPT Surviving Universe</div>
        <div class="stat-val" style="color:var(--green);">42 Stocks</div>
        <div class="stat-sub">Active in Mean-Variance Matrix</div>
      </div>
      <div class="stat-pill">
        <div class="stat-lbl">Filtered Recent IPOs</div>
        <div class="stat-val" style="color:var(--rose);">7 Stocks</div>
        <div class="stat-sub">>30% History Missing (Protected)</div>
      </div>
      <div class="stat-pill">
        <div class="stat-lbl">Data Coverage Span</div>
        <div class="stat-val" style="color:var(--cyan);">2000 – 2021</div>
        <div class="stat-sub">Multi-Decade Tick-Aligned Clean Series</div>
      </div>
    </div>

    <!-- Filter Bar -->
    <div class="filter-section">
      <div class="filter-row1">
        <div class="search-box">
          <span class="search-icon">🔍</span>
          <input type="text" id="searchInput" class="search-input" placeholder="Search company e.g. TCS, Reliance, Infosys, HDFC..." oninput="filterStocks()">
        </div>

        <select id="sortSelect" class="sort-select" onchange="filterStocks()">
          <option value="name_asc">Sort: Name (A → Z)</option>
          <option value="return_desc">Sort: Highest Return %</option>
          <option value="dd_asc">Sort: Lowest Drawdown</option>
          <option value="price_desc">Sort: Highest Price</option>
        </select>
      </div>

      <!-- Sector Pills -->
      <div class="filter-pills" id="sectorPills">
        <button class="filter-btn active" data-sector="ALL" onclick="setSector('ALL')">All Sectors ({len(assets)})</button>
"""
    for ind in industries:
        count = sum(1 for a in assets if a["industry"] == ind)
        html_content += f"""        <button class="filter-btn" data-sector="{ind}" onclick="setSector('{ind}')">{ind} ({count})</button>\n"""

    html_content += f"""      </div>
    </div>

    <!-- Grid of 52 Cards -->
    <div class="grid-container" id="stocksGrid"></div>

  </div>

  <!-- Deep Dive Modal -->
  <div class="modal-overlay" id="modalOverlay" onclick="closeModalOnBg(event)">
    <div class="modal-card">
      <button class="modal-close" onclick="closeModal()">✕</button>
      <div class="modal-header">
        <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem;">
          <span class="modal-ticker" id="mTicker">TCS</span>
          <span class="sector-badge" id="mSector">IT</span>
          <span class="badge" id="mStatus">MPT Active</span>
        </div>
        <div style="color:var(--muted); font-size:0.9rem;" id="mName">Tata Consultancy Services Ltd.</div>
      </div>

      <div class="modal-stats-grid">
        <div class="mstat-box">
          <div class="mstat-lbl">Latest Close</div>
          <div class="mstat-val" id="mClose">₹0.00</div>
        </div>
        <div class="mstat-box">
          <div class="mstat-lbl">All-Time Return</div>
          <div class="mstat-val" id="mReturn">+0.0%</div>
        </div>
        <div class="mstat-box">
          <div class="mstat-lbl">Max Drawdown</div>
          <div class="mstat-val" style="color:var(--rose);" id="mDD">-0.0%</div>
        </div>
        <div class="mstat-box">
          <div class="mstat-lbl">All-Time High</div>
          <div class="mstat-val" id="mATH">₹0.00</div>
        </div>
        <div class="mstat-box">
          <div class="mstat-lbl">Trading Days</div>
          <div class="mstat-val" id="mRows">0</div>
        </div>
      </div>

      <div class="modal-chart-wrap">
        <canvas id="modalCanvas"></canvas>
      </div>
    </div>
  </div>

  <script>
    const STOCKS_DATA = {json_data};
    let currentSector = "ALL";
    let modalChartInstance = null;
    const sparklineCharts = {{}};

    function init() {{
      renderCards(STOCKS_DATA);
    }}

    function setSector(sec) {{
      currentSector = sec;
      document.querySelectorAll('#sectorPills .filter-btn').forEach(b => {{
        b.classList.toggle('active', b.dataset.sector === sec);
      }});
      filterStocks();
    }}

    function filterStocks() {{
      const query = document.getElementById('searchInput').value.trim().toLowerCase();
      const sortBy = document.getElementById('sortSelect').value;

      let filtered = STOCKS_DATA.filter(s => {{
        const matchSector = (currentSector === "ALL" || s.industry === currentSector);
        const matchQuery = (
          s.ticker.toLowerCase().includes(query) ||
          s.name.toLowerCase().includes(query) ||
          s.industry.toLowerCase().includes(query)
        );
        return matchSector && matchQuery;
      }});

      // Sorting
      if (sortBy === "return_desc") {{
        filtered.sort((a,b) => b.total_return_pct - a.total_return_pct);
      }} else if (sortBy === "dd_asc") {{
        filtered.sort((a,b) => b.max_drawdown_pct - a.max_drawdown_pct);
      }} else if (sortBy === "price_desc") {{
        filtered.sort((a,b) => b.last_price - a.last_price);
      }} else {{
        filtered.sort((a,b) => a.ticker.localeCompare(b.ticker));
      }}

      renderCards(filtered);
    }}

    function renderCards(list) {{
      const container = document.getElementById('stocksGrid');
      container.innerHTML = "";

      // Cleanup old charts
      Object.values(sparklineCharts).forEach(c => c.destroy());

      if (list.length === 0) {{
        container.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 4rem; color:var(--muted);">No matching assets found.</div>`;
        return;
      }}

      list.forEach((stock, idx) => {{
        const card = document.createElement('div');
        card.className = "stock-card";
        card.onclick = () => openModal(stock);

        const isPositive = stock.total_return_pct >= 0;
        const retColor = isPositive ? "up" : "down";
        const retSign = isPositive ? "+" : "";

        let dotClass = "dot-emerald";
        if (stock.status_color === "rose") dotClass = "dot-rose";
        if (stock.status_color === "gray") dotClass = "dot-gray";
        if (stock.status_color === "sky")  dotClass = "dot-sky";

        card.innerHTML = `
          <div>
            <div class="card-top">
              <div>
                <div class="ticker-title">${{stock.ticker}}</div>
                <div class="comp-name" title="${{stock.name}}">${{stock.name}}</div>
              </div>
              <span class="sector-badge">${{stock.industry}}</span>
            </div>

            <div class="card-stats">
              <div class="cstat-item">
                <span class="cstat-lbl">Last Close</span>
                <span class="cstat-val">₹${{stock.last_price.toLocaleString()}}</span>
              </div>
              <div class="cstat-item">
                <span class="cstat-lbl">Return</span>
                <span class="cstat-val ${{retColor}}">${{retSign}}${{stock.total_return_pct}}%</span>
              </div>
              <div class="cstat-item">
                <span class="cstat-lbl">Drawdown</span>
                <span class="cstat-val down">${{stock.max_drawdown_pct}}%</span>
              </div>
            </div>

            <div class="chart-container">
              <canvas id="spark_${{stock.ticker.replace(/[^a-zA-Z0-9]/g, '_')}}"></canvas>
            </div>
          </div>

          <div class="card-bottom">
            <span class="status-tag">
              <span class="status-dot ${{dotClass}}"></span>
              ${{stock.status}}
            </span>
            <span>${{stock.start_date}} → ${{stock.end_date}}</span>
          </div>
        `;
        container.appendChild(card);

        // Render sparkline
        setTimeout(() => {{
          renderSparkline(stock);
        }}, 0);
      }});
    }}

    function renderSparkline(stock) {{
      const canvasId = "spark_" + stock.ticker.replace(/[^a-zA-Z0-9]/g, '_');
      const canvas = document.getElementById(canvasId);
      if (!canvas || stock.points.length === 0) return;

      const labels = stock.points.map(p => p.d);
      const data = stock.points.map(p => p.c);

      const isPositive = stock.total_return_pct >= 0;
      let strokeColor = isPositive ? "#34D399" : "#FB7185";
      if (stock.status_color === "sky") strokeColor = "#38BDF8";

      const ctx = canvas.getContext('2d');
      const gradient = ctx.createLinearGradient(0, 0, 0, 95);
      gradient.addColorStop(0, strokeColor + '40');
      gradient.addColorStop(1, strokeColor + '00');

      sparklineCharts[stock.ticker] = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: labels,
          datasets: [{{
            data: data,
            borderColor: strokeColor,
            borderWidth: 1.8,
            pointRadius: 0,
            fill: true,
            backgroundColor: gradient,
            tension: 0.25,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              mode: 'index',
              intersect: false,
              backgroundColor: 'rgba(10, 15, 30, 0.9)',
              titleColor: '#94A3B8',
              bodyColor: '#FFFFFF',
              borderColor: 'rgba(79,142,247,0.3)',
              borderWidth: 1,
              padding: 6,
              displayColors: false,
              callbacks: {{
                label: (ctx) => `₹${{ctx.parsed.y.toLocaleString()}}`
              }}
            }}
          }},
          scales: {{
            x: {{ display: false }},
            y: {{ display: false }}
          }}
        }}
      }});
    }}

    function openModal(stock) {{
      document.getElementById('mTicker').innerText = stock.ticker;
      document.getElementById('mName').innerText = stock.name;
      document.getElementById('mSector').innerText = stock.industry;
      document.getElementById('mStatus').innerText = stock.status;
      document.getElementById('mClose').innerText = `₹${{stock.last_price.toLocaleString()}}`;
      document.getElementById('mReturn').innerText = `${{stock.total_return_pct >= 0 ? '+' : ''}}${{stock.total_return_pct}}%`;
      document.getElementById('mReturn').style.color = stock.total_return_pct >= 0 ? 'var(--green)' : 'var(--rose)';
      document.getElementById('mDD').innerText = `${{stock.max_drawdown_pct}}%`;
      document.getElementById('mATH').innerText = `₹${{stock.ath.toLocaleString()}}`;
      document.getElementById('mRows').innerText = `${{stock.rows.toLocaleString()}} days`;

      document.getElementById('modalOverlay').classList.add('open');

      if (modalChartInstance) modalChartInstance.destroy();

      const ctx = document.getElementById('modalCanvas').getContext('2d');
      const labels = stock.points.map(p => p.d);
      const data = stock.points.map(p => p.c);

      const strokeColor = stock.total_return_pct >= 0 ? "#34D399" : "#FB7185";
      const gradient = ctx.createLinearGradient(0, 0, 0, 420);
      gradient.addColorStop(0, strokeColor + '40');
      gradient.addColorStop(1, strokeColor + '00');

      modalChartInstance = new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: labels,
          datasets: [{{
            label: stock.ticker + ' Price',
            data: data,
            borderColor: strokeColor,
            borderWidth: 2.2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: strokeColor,
            fill: true,
            backgroundColor: gradient,
            tension: 0.2,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: 'rgba(10, 15, 30, 0.95)',
              titleColor: '#94A3B8',
              bodyColor: '#FFFFFF',
              borderColor: 'rgba(79, 142, 247, 0.3)',
              borderWidth: 1,
              padding: 10,
              callbacks: {{
                label: (ctx) => `Close: ₹${{ctx.parsed.y.toLocaleString()}}`
              }}
            }}
          }},
          scales: {{
            x: {{
              grid: {{ color: 'rgba(79, 142, 247, 0.08)' }},
              ticks: {{ color: '#64748B', maxTicksLimit: 10 }}
            }},
            y: {{
              grid: {{ color: 'rgba(79, 142, 247, 0.08)' }},
              ticks: {{
                color: '#64748B',
                callback: (val) => '₹' + val.toLocaleString()
              }}
            }}
          }}
        }}
      }});
    }}

    function closeModal() {{
      document.getElementById('modalOverlay').classList.remove('open');
      if (modalChartInstance) modalChartInstance.destroy();
    }}

    function closeModalOnBg(e) {{
      if (e.target.id === "modalOverlay") closeModal();
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === "Escape") closeModal();
    }});

    window.onload = init;
  </script>
</body>
</html>
"""
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[SUCCESS] Generated all 52 charts dashboard: {OUT_PATH}")
    print(f"          Total Assets: {len(assets)}")


if __name__ == "__main__":
    print("[1/2] Processing all 52 assets ...")
    assets = process_all_assets()
    print(f"[2/2] Generating HTML dashboard for {len(assets)} assets ...")
    generate_html(assets)
