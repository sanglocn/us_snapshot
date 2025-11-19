import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
import io
import base64
import re
import html
from typing import List

# ---------------------------------
# Configuration
# ---------------------------------
st.set_page_config(page_title="US Market Daily Snapshot", layout="wide")

DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv",
    "heat": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_heat.csv",
    "stage": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price_weekly.csv",
}

LOOKBACK_DAYS = 21
GROUP_ORDER = ["Market", "Sector", "Commodity", "Crypto", "Country", "Theme", "Leader"]

GROUP_PALETTE = {
    "Market":   ("#0284c7", "#075985"),
    "Sector":   ("#16a34a", "#166534"),
    "Commodity":("#b45309", "#7c2d12"),
    "Crypto":   ("#7c3aed", "#4c1d95"),
    "Country":  ("#ea580c", "#9a3412"),
    "Theme":    ("#2563eb", "#1e40af"),
    "Leader":   ("#db2777", "#9d174d"),
}

use_group_colors = True

# ---------------------------------
# Helpers
# ---------------------------------
def _clean_text_series(s: pd.Series):
    return s.astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

def _clean_ticker_series(s: pd.Series):
    return s.astype(str).str.replace(r"[\r\n\t\s]+", "", regex=True).str.upper().str.strip()

def _escape(s) -> str:
    return html.escape("" if pd.isna(s) else str(s))

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')

# ---------------------------------
# Sparkline for Tooltip
# ---------------------------------
def create_sparkline_base64(values: List[float], width: int = 180, height: int = 50) -> str:
    if not values or len(values) < 2:
        return ""
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_axes([0,0,1,1])
    ax.plot(values, color="#16a34a", linewidth=2)
    ax.fill_between(range(len(values)), values, min(values), color="#16a34a22")
    ax.plot(len(values)-1, values[-1], "o", color="#166534", markersize=5)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ---------------------------------
# Advanced Tooltip (RS + Key Metrics)
# ---------------------------------
def make_advanced_tooltip(row: pd.Series, rs_series: List[float]) -> str:
    spark_img = create_sparkline_base64(rs_series)
    stage_raw = row.get("stage_label_adj", "Unknown")
    stage = str(stage_raw).lower()
    stage_emoji = {
        "stage 1": "🟡", "stage 2": "🟢", "stage 3": "🟠", "stage 4": "🔴"
    }.get(stage, "⚪")

    return f"""
    <div class="tt-card">
        <div class="tt-title">{_escape(row.name)}</div>
        {f'<img src="{spark_img}" style="width:100%;border-radius:8px;margin:10px 0;"/>' if spark_img else ''}
        <table class="tt-table">
            <tr><td>1W Return</td><td>{row.get('ret_1w', 0):+.1f}%</td></tr>
            <tr><td>1M Return</td><td>{row.get('ret_1m', 0):+.1f}%</td></tr>
            <tr><td>Below 52W High</td><td>{row.get('pct_below_high', 0):.1f}%</td></tr>
            <tr><td>Above 52W Low</td><td>{row.get('pct_above_low', 0):.1f}%</td></tr>
            <tr><td>Above SMA5</td><td>{'✅' if row.get('above_sma5') == 'yes' else '❌'}</td></tr>
            <tr><td>Above SMA10</td><td>{'✅' if row.get('above_sma10') == 'yes' else '❌'}</td></tr>
            <tr><td>Modified Stage</td><td>{stage_emoji} {stage_raw if pd.notna(stage_raw) else 'N/A'}</td></tr>
        </table>
    </div>
    """

def make_ticker_chip(ticker: str, tooltip_html: str, group_name: str | None) -> str:
    group_class = f" chip--{slugify(group_name)}" if use_group_colors and group_name else ""
    return f'<span class="tt-chip{group_class}">{_escape(ticker)}{tooltip_html}</span>'

# ---------------------------------
# CSS
# ---------------------------------
def build_chip_css() -> str:
    base = """
    .tt-chip{position:relative;display:inline-block;padding:6px 12px;margin:2px;border-radius:9999px;
      background:linear-gradient(135deg,#2563eb22,#07598522);border:1px solid #2563eb55;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;font-weight:700;
      color:#1e3a8a;cursor:default;transition:all .2s;box-shadow:0 1px 2px rgba(0,0,0,.05);}
    .tt-chip:hover{background:linear-gradient(135deg,#2563eb33,#1d4ed833);border-color:#2563eb88;
      box-shadow:0 4px 16px rgba(37,99,235,.35);}
    .tt-card{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%) translateY(6px);z-index:999999;
      width:min(380px,92vw);max-height:82vh;overflow:auto;background:#fff;border:1px solid rgba(0,0,0,.08);
      box-shadow:0 20px 40px rgba(0,0,0,.15);border-radius:16px;padding:16px;visibility:hidden;opacity:0;
      transition:all .22s;font-size:13px;}
    .tt-chip:hover .tt-card{visibility:visible;opacity:1;transform:translate(-50%,-50%);}
    .tt-title{font-weight:800;font-size:16px;margin-bottom:8px;color:#1e293b;}
    .tt-table{width:100%;border-collapse:collapse;margin-top:8px;}
    .tt-table td{padding:6px 0;border-bottom:1px dashed #e2e8f0;}
    .tt-table td:first-child{color:#64748b;width:62%;}
    .tt-table td:last-child{text-align:right;font-weight:600;color:#1e293b;}
    .tt-table tr:last-child td{border:none;}
    @media(max-width:768px){.tt-card{width:95vw;padding:14px;font-size:12px;}.tt-title{font-size:15px;}}
    """
    groups = ""
    if use_group_colors:
        for g, (b, d) in GROUP_PALETTE.items():
            s = slugify(g)
            groups += f".tt-chip.chip--{s}{{background:linear-gradient(135deg,{b}22,{d}22);border-color:{b}55;color:{d};}}"
            groups += f".tt-chip.chip--{s}:hover{{background:linear-gradient(135deg,{b}33,{d}33);border-color:{b}88;box-shadow:0 4px 16px {b}55;}}"
    return "<style>" + base + groups + "</style>"

# ---------------------------------
# Data Loading
# ---------------------------------
@st.cache_data(ttl=900)
def load_data():
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs  = pd.read_csv(DATA_URLS["rs"])
    df_etf["date"] = pd.to_datetime(df_etf["date"])
    df_rs["date"]  = pd.to_datetime(df_rs["date"])
    df_etf["ticker"] = _clean_ticker_series(df_etf["ticker"])
    df_rs["ticker"]  = _clean_ticker_series(df_rs["ticker"])
    if "group" in df_etf.columns:
        df_etf["group"] = _clean_text_series(df_etf["group"])
    return df_etf, df_rs

@st.cache_data(ttl=900)
def load_heat_csv():
    df = pd.read_csv(DATA_URLS["heat"])
    colmap = {c.lower(): c for c in df.columns}
    rename = {colmap.get(k, k): v for k, v in zip(['date','ticker','code','pricefactor','volumefactor'],
                                                 ['date','ticker','code','PriceFactor','VolumeFactor'])}
    df = df.rename(columns=rename)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=900)
def load_stage_csv():
    df = pd.read_csv(DATA_URLS["stage"])
    colmap = {c.lower(): c for c in df.columns}
    rename = {colmap.get(k, k): v for k, v in zip(['ticker','date','stage_label_core','stage_label_adj'],
                                                 ['ticker','date','stage_label_core','stage_label_adj'])}
    df = df.rename(columns=rename)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.sort_values(['ticker', 'date']).groupby('ticker').tail(1)

# ---------------------------------
# Formatting Functions (unchanged)
# ---------------------------------
def format_rank(v): 
    if pd.isna(v): return '<span style="display:block;text-align:right;">-</span>'
    p = int(round(v*100))
    if p >= 85: bg, bor = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)"
    elif p < 50: bg, bor = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"
    else: bg, bor = "rgba(156,163,175,.25)", "rgba(156,163,175,.35)"
    return f'<span style="display:block;text-align:right;padding:2px 6px;border-radius:6px;background:{bg};border:1px solid {bor};">{p}%</span>'

def format_performance(v):
    if pd.isna(v): return '<span style="display:block;text-align:right;">-</span>'
    return f'<span style="display:block;text-align:right;">{v:.1f}%</span>'

def format_performance_intraday(v):
    if pd.isna(v): return '<span style="display:block;text-align:right;">-</span>'
    txt = f"{v:.1f}%"
    if v > 0: bg, bor = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)"
    elif v < 0: bg, bor = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"
    else: bg, bor = "rgba(156,163,175,.25)", "rgba(156,163,175,.35)"
    return f'<span style="display:block;text-align:right;padding:2px 6px;border-radius:6px;background:{bg};border:1px solid {bor};">{txt}</span>'

def format_52w_high(v):
    if pd.isna(v): return '<span style="display:block;text-align:right;">-</span>'
    if v > -3: return '<span style="display:block;text-align:center;">🚀</span>'
    return f'<span style="display:block;text-align:right;">{v:.1f}%</span>'

def format_52w_low(v):
    if pd.isna(v): return '<span style="display:block;text-align:right;">-</span>'
    if v < 3: return '<span style="display:block;text-align:center;">🐌</span>'
    return f'<span style="display:block;text-align:right;">{v:.1f}%</span>'

def format_volume_alert(val, rs252):
    if not isinstance(val, str): return '<span style="display:block;text-align:center;">-</span>'
    val = val.strip().lower()
    try: strong_rs = float(rs252) >= 0.80
    except: strong_rs = False
    if val == "positive" and strong_rs: return '<span style="display:block;text-align:center;font-size:16px;">💎</span>'
    if val == "positive": return '<span style="display:block;text-align:center;font-size:16px;">🟩</span>'
    if val == "negative": return '<span style="display:block;text-align:center;font-size:16px;">🟥</span>'
    return '<span style="display:block;text-align:center;">-</span>'

def format_volatility(v):
    v = str(v).strip().lower()
    if v == "compression": return '<span style="color:green;display:block;text-align:center;">🎯</span>'
    if v == "spike": return '<span style="color:red;display:block;text-align:center;">⚠️</span>'
    return '<span style="display:block;text-align:center;">-</span>'

def format_multiple(v):
    try: v = float(v)
    except: return '<span style="display:block;text-align:right;">-</span>'
    txt = f"{v:.2f}"
    if v >= 10: bg, bor = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"
    elif v >= 4: bg, bor = "rgba(234,179,8,.22)", "rgba(234,179,8,.35)"
    elif v > 0: bg, bor = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)"
    else: bg, bor = "rgba(156,163,175,.18)", "rgba(156,163,175,.30)"
    return f'<span style="display:block;text-align:right;padding:2px 6px;border-radius:6px;background:{bg};border:1px solid {bor};">{txt}</span>'

def format_stage_label(v):
    v = str(v).lower()
    if "stage 1" in v: return '<span style="display:block;text-align:center;">🟡</span>'
    if "stage 2" in v: return '<span style="display:block;text-align:center;">🟢</span>'
    if "stage 3" in v: return '<span style="display:block;text-align:center;">🟠</span>'
    if "stage 4" in v: return '<span style="display:block;text-align:center;">🔴</span>'
    return '<span style="display:block;text-align:center;">⚪</span>'

# ---------------------------------
# Dashboard
# ---------------------------------
def render_dashboard(df_etf: pd.DataFrame, df_rs: pd.DataFrame):
    st.title("US Market Daily Snapshot")
    st.markdown(build_chip_css(), unsafe_allow_html=True)

    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK_DAYS)

    st.caption(f"Latest Update: {latest['date'].max().date()}")

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        hide_rs = st.toggle("Hide RS Rank (1M) < 85%", False)
        hide_ext = st.toggle("Hide Extension Multiple > 4", False)
        hide_pv = st.toggle("Hide Weak Price/Volume", False)
        st.markdown("---")
        for g in GROUP_ORDER:
            st.markdown(f'[📌 {g}](#{slugify(g)})', unsafe_allow_html=True)
        st.markdown('[✏️ Breadth Gauge](#breadth-gauge)')
        st.markdown('[🧠 Price & Volume](#price-volume)')

    # Merge stage data
    df_stage = load_stage_csv()
    if not df_stage.empty:
        latest = latest.reset_index().merge(df_stage[['ticker', 'stage_label_core', 'stage_label_adj']],
                                           on='ticker', how='left').set_index('ticker')

    # Render each group
    for group in GROUP_ORDER:
        anchor = slugify(group)
        st.markdown(f'<div id="{anchor}" style="padding-top:80px;margin-top:-80px;"></div>', unsafe_allow_html=True)
        st.header(f"📌 {group}")

        if "group" not in latest.columns or group not in latest["group"].values:
            st.info("No data")
            continue

        tickers = latest[latest["group"] == group].index.tolist()

        # Apply filters
        if hide_rs and "rs_rank_21d" in latest.columns:
            tickers = [t for t in tickers if latest.loc[t, "rs_rank_21d"] >= 0.85]
        if hide_ext and "ratio_pct_dist_to_atr_pct" in latest.columns:
            tickers = [t for t in tickers if latest.loc[t, "ratio_pct_dist_to_atr_pct"] <= 4.0]

        if not tickers:
            st.info("No tickers after filters")
            continue

        rows = []
        for t in tickers:
            r = latest.loc[t]
            rs_series = rs_last[rs_last["ticker"] == t]["rs_to_spy"].tolist()
            tooltip = make_advanced_tooltip(r, rs_series)
            chip = make_ticker_chip(t, tooltip, group)

            rows.append({
                "Ticker": chip,
                "RS Rank (1M)": format_rank(r.get("rs_rank_21d")),
                "RS Rank (1Y)": format_rank(r.get("rs_rank_252d")),
                "Vol Alert": format_volume_alert(r.get("volume_alert", ""), r.get("rs_rank_252d")),
                "Volatility": format_volatility(r.get("volatility_signal")),
                "Intraday": format_performance_intraday(r.get("ret_intraday")),
                "1D": format_performance(r.get("ret_1d")),
                "1W": format_performance(r.get("ret_1w")),
                "1M": format_performance(r.get("ret_1m")),
                "52W High": format_52w_high(r.get("pct_below_high")),
                "52W Low": format_52w_low(r.get("pct_above_low")),
                "Extension": format_multiple(r.get("ratio_pct_dist_to_atr_pct")),
                "Core": format_stage_label(r.get("stage_label_core")),
                "Mod": format_stage_label(r.get("stage_label_adj"))
            })

        st.markdown(pd.DataFrame(rows).to_html(escape=False, index=False), unsafe_allow_html=True)

    # Breadth & Heat sections remain unchanged (you already have them in your original code)

# ---------------------------------
# Main
# ---------------------------------
def main():
    df_etf, df_rs = load_data()
    render_dashboard(df_etf, df_rs)

if __name__ == "__main__":
    main()
