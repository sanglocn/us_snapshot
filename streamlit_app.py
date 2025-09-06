import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io
import base64
import re
from typing import List, Dict, Tuple

# ---------------------------------
# Configuration
# ---------------------------------
st.set_page_config(page_title="US Market Snapshot", layout="wide")

# Constants
DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv",
    "holdings": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_holdings.csv",
}
LOOKBACK_DAYS = 21
GROUP_ORDER = [
    "Market",
    "Sector",
    "Commodity",
    "Crypto",
    "Country",
    "Theme",
    "Leader"
]

# ---------------------------------
# Data Loading
# ---------------------------------
@st.cache_data(ttl=900)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs = pd.read_csv(DATA_URLS["rs"])
    df_etf["date"] = pd.to_datetime(df_etf["date"])
    df_rs["date"] = pd.to_datetime(df_rs["date"])
    return df_etf, df_rs

@st.cache_data(ttl=900, show_spinner=False)
def load_holdings_csv(url: str = DATA_URLS["holdings"]) -> pd.DataFrame:
    """
    Expected columns: fund_ticker, fund_name, security_name, security_weight, ingest_date
    """
    df = pd.read_csv(url)
    need = {"fund_ticker", "fund_name", "security_name", "security_weight", "ingest_date"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[holdings] Missing columns: {sorted(missing)}")
    df["ingest_date"] = pd.to_datetime(df["ingest_date"], errors="coerce")
    df["security_weight"] = pd.to_numeric(df["security_weight"], errors="coerce")
    # Optional: readable casing
    df["fund_name"] = df["fund_name"].astype(str).str.title()
    df["security_name"] = df["security_name"].astype(str).str.title()
    return df

# ---------------------------------
# Data Processing
# ---------------------------------
def process_data(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK_DAYS)
    return latest, rs_last_n

def compute_threshold_counts(df_etf: pd.DataFrame) -> pd.DataFrame:
    if "rs_rank_21d" not in df_etf.columns:
        return pd.DataFrame(columns=["date", "count_over_85", "count_under_50", "date_str"])

    tmp = df_etf.copy()
    tmp["over_85"] = tmp["rs_rank_21d"] >= 0.85
    tmp["under_50"] = tmp["rs_rank_21d"] < 0.50

    daily = (
        tmp.groupby("date", as_index=False)
           .agg(count_over_85=("over_85", "sum"),
                count_under_50=("under_50", "sum"))
           .sort_values("date")
    )
    daily = daily.dropna(subset=["count_over_85", "count_under_50"])

    last_21_dates = daily["date"].drop_duplicates().sort_values().tail(21)
    daily_21 = daily[daily["date"].isin(last_21_dates)].copy().sort_values("date")

    daily_21["date_str"] = daily_21["date"].dt.strftime("%Y-%m-%d")
    return daily_21

# ---------------------------------
# Tooltip helpers (HTML/CSS)
# ---------------------------------
def _escape(s) -> str:
    import html
    return html.escape("" if pd.isna(s) else str(s))

def make_tooltip_card_for_ticker(holdings_df: pd.DataFrame, ticker: str, max_rows: int = 15) -> str:
    sub = holdings_df[holdings_df["fund_ticker"] == ticker]
    if sub.empty:
        return ""
    last_date = sub["ingest_date"].max()
    if pd.notna(last_date):
        sub = sub[sub["ingest_date"] == last_date]

    fund_name = _escape(sub["fund_name"].iloc[0])
    last_update_str = _escape(last_date.strftime("%Y-%m-%d %H:%M") if pd.notna(last_date) else "N/A")

    topn = (
        sub[["security_name", "security_weight"]]
        .dropna(subset=["security_name"])
        .sort_values("security_weight", ascending=False)
        .head(max_rows)
    )

    rows = []
    for _, r in topn.iterrows():
        sec = _escape(r["security_name"])
        wt = "" if pd.isna(r["security_weight"]) else f"{float(r['security_weight']):.2f}%"
        rows.append(f"<tr><td class='tt-sec'>{sec}</td><td class='tt-wt'>{wt}</td></tr>")

    table_html = (
        "<table class='tt-table'><thead><tr><th>Security</th><th>Weight</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    return f"""
    <div class="tt-card">
      <div class="tt-title">{fund_name}</div>
      <div class="tt-sub">Last update: <span class="tt-date">{last_update_str}</span></div>
      {table_html}
    </div>
    """

def make_ticker_chip_with_tooltip(ticker: str, card_html: str) -> str:
    t = _escape(ticker)
    return f"""
    <span class="tt-chip">{t}
      {card_html}
    </span>
    """

TOOLTIP_CSS = """
<style>
.tt-chip { position: relative; display: inline-block; padding: 4px 8px;
  border-radius: 9999px; border: 1px solid rgba(0,0,0,0.10); background: #f7f7f9;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px; cursor: default; white-space: nowrap; box-shadow: 0 1px 2px rgba(0,0,0,0.04);}
.tt-chip:hover { background: #eef2ff; border-color: rgba(59,130,246,0.30);}
.tt-chip .tt-card { visibility: hidden; opacity: 0; transform: translateY(6px);
  transition: opacity .18s ease, transform .18s ease; position: absolute; left: 0; top: calc(100% + 8px);
  z-index: 1000; width: min(520px, 90vw); background: #ffffff; border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 12px 28px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.06); border-radius: 12px; padding: 12px;}
.tt-chip:hover .tt-card { visibility: visible; opacity: 1; transform: translateY(0);}
.tt-card .tt-title { font-weight: 700; font-size: 14px; margin-bottom: 4px;}
.tt-card .tt-sub { color: #667085; font-size: 12px; margin-bottom: 8px;}
.tt-card .tt-date { font-variant-numeric: tabular-nums;}
.tt-table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed;}
.tt-table thead th { text-align: left; padding: 6px 6px; border-bottom: 1px solid #eee;}
.tt-table tbody td { padding: 6px 6px; border-bottom: 1px dashed #f0f0f0; vertical-align: top; word-wrap: break-word;}
.tt-table tbody tr:last-child td { border-bottom: none;}
.tt-sec { width: 80%;}
.tt-wt { width: 20%; text-align: right; font-variant-numeric: tabular-nums;}
</style>
"""
