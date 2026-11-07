"""Central sector classifications for the supported NIFTY asset universe."""

from __future__ import annotations


NIFTY_SECTORS: dict[str, str] = {
    "HDFCBANK": "Financial Services",
    "ICICIBANK": "Financial Services",
    "KOTAKBANK": "Financial Services",
    "SBIN": "Financial Services",
    "AXISBANK": "Financial Services",
    "BAJFINANCE": "Financial Services",
    "INDUSINDBK": "Financial Services",
    "HDFC": "Financial Services",
    "TCS": "IT",
    "INFY": "IT",
    "HCLTECH": "IT",
    "WIPRO": "IT",
    "TECHM": "IT",
    "RELIANCE": "Energy/Oil & Gas",
    "ONGC": "Energy/Oil & Gas",
    "BPCL": "Energy/Oil & Gas",
    "IOC": "Energy/Oil & Gas",
    "GAIL": "Energy/Oil & Gas",
    "NTPC": "Energy/Oil & Gas",
    "POWERGRID": "Energy/Oil & Gas",
    "ITC": "FMCG",
    "HINDUNILVR": "FMCG",
    "BRITANNIA": "FMCG",
    "NESTLEIND": "FMCG",
    "TITAN": "Consumer Durables",
    "SUNPHARMA": "Healthcare",
    "DRREDDY": "Healthcare",
    "CIPLA": "Healthcare",
    "DIVISLAB": "Healthcare",
    "MARUTI": "Automobile",
    "TATAMOTORS": "Automobile",
    "MM": "Automobile",
    "HEROMOTOCO": "Automobile",
    "EICHERMOT": "Automobile",
    "TATASTEEL": "Metals & Mining",
    "HINDALCO": "Metals & Mining",
    "JSWSTEEL": "Metals & Mining",
    "VEDL": "Metals & Mining",
    "LT": "Cement/Construction",
    "ULTRACEMCO": "Cement/Construction",
    "SHREECEM": "Cement/Construction",
    "GRASIM": "Cement/Construction",
    "ASIANPAINT": "Consumer Materials",
    "BHARTIARTL": "Telecommunication",
    "UPL": "Chemicals",
    "ZEEL": "Media & Entertainment",
}


def get_sector(ticker: str) -> str:
    """Return the sector for a ticker, defaulting to ``Other``."""
    return NIFTY_SECTORS.get(str(ticker).strip().upper(), "Other")
