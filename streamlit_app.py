import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import altair as alt
import io
import base64
import re
import html
from typing import List, Dict, Tuple

# ---------------------------------
# Configuration
# ---------------------------------
st.set_page_config(page_title="US Market Daily Snapshot", layout="wide")

# Constants
DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv",
    "heat": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_heat.csv",
    "stage": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price_weekly.csv",
}
LOOKBACK_DAYS = 21
GROUP_ORDER = ["Market", "Sector", "Commodity", "Crypto", "Country", "Theme", "Leader"]

# Palette per group
GROUP_PALETTE = {
    "Market":     ("#0284c7", "#075985"),
    "Sector":     ("#16a34a", "#166534"),
    "Commodity":  ("#b45309", "#7c2d12"),
    "Crypto":     ("#7c3aed", "#4c1d95"),
    "Country":    ("#ea580c", "#9a3412"),
    "Theme":      ("#2563eb", "#1e40af"),
    "Leader":     ("#db2777", "#9d174d"),
}

# Settings
use_group_colors = True

# ---------------------------------
# Small Helpers
# ---------------------------------
def _clean_text_series(s: pd.Series, title_case: bool = False) -> pd.Series:
    s = s.astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    if title_case:
        s = s.str.title()
    return s

def _clean_ticker_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[\r\n\t\s]+", "", regex=True).str.upper().str.strip()

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')

# ---------------------------------
# Data Loading
# ---------------------------------
@st.cache_data(ttl=900)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs = pd.read_csv(DATA_URLS["rs"])

    # Validate
    if not {"date", "ticker"}.issubset(df_etf.columns):
        raise ValueError("ETF price CSV missing required columns.")
    if not {"date", "ticker", "rs_to_spy"}.issubset(df_rs.columns):
        raise ValueError("RS CSV missing required columns.")

    df_etf["date"] = pd.to_datetime(df_etf["date"], errors="coerce")
    df_rs["date"] = pd.to_datetime(df_rs["date"], errors="coerce")

    df_etf["ticker"] = _clean_ticker_series(df_etf["ticker"])
    df_rs["ticker"] = _clean_ticker_series(df_rs["ticker"])
    if "group" in df_etf.columns:
        df_etf["group"] = _clean_text_series(df_etf["group"])
    if "group" in df_rs.columns:
        df_rs["group"] = _clean_text_series(df_rs["group"])

    return df_etf, df_rs

@st.cache_data(ttl=900)
def load_heat_csv() -> pd.DataFrame:
    df = pd.read_csv(DATA_URLS["heat"])
    colmap = {c.lower(): c for c in df.columns}
    required = ['date', 'ticker', 'code', 'pricefactor', 'volumefactor']
    missing = [r for r in required if r not in colmap]
    if missing:
        raise ValueError(f"Missing heat columns: {missing}")

    df = df.rename(columns={
        colmap['date']: 'date',
        colmap['ticker']: 'ticker',
        colmap['code']: 'code',
        colmap['pricefactor']: 'PriceFactor',
        colmap['volumefactor']: 'VolumeFactor'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=900)
def load_stage_csv() -> pd.DataFrame:
    df = pd.read_csv(DATA_URLS["stage"])
    colmap = {c.lower(): c for c in df.columns}
    required = ['ticker', 'date', 'stage_label_core', 'stage_label_adj']
    missing = [r for r in required if r not in colmap]
    if missing:
        raise ValueError(f"Missing stage columns: {missing}")

    df = df.rename(columns={
        colmap['ticker']: 'ticker',
        colmap['date']: 'date',
        colmap['stage_label_core']: 'stage_label_core',
        colmap['stage_label_adj']: 'stage_label_adj'
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['ticker', 'date']).groupby('ticker').tail(1)
    return df

# ---------------------------------
# Data Processing
# ---------------------------------
def process_data(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK_DAYS)
    return latest, rs_last_n

def compute_threshold_counts(df_etf: pd.DataFrame) -> pd.DataFrame:
    if "count_over_85" not in df_etf.columns or "count_under_50" not in df_etf.columns:
        return pd.DataFrame()
    daily = df_etf[["date", "count_over_85", "count_under_50"]].drop_duplicates().sort_values("date").dropna()
    daily_21 = daily.tail(21).copy()
    daily_21["date_str"] = daily_21["date"].dt.strftime("%Y-%m-%d")
    return daily_21

# ---------------------------------
# Sparkline
# ---------------------------------
def create_sparkline(values: List[float], width: int = 155, height: int = 36) -> str:
    if not values:
        return ""
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(values, linewidth=1.5, color="#16a34a")
    ax.plot(len(values)-1, values[-1], "o", color="#166534", markersize=4)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}" />'

# ---------------------------------
# Formatting Helpers
# ---------------------------------
def format_rank(value: float) -> str:
    if pd.isna(value):
        return "-"
    pct = int(round(value * 100))
    if pct >= 85:
        color = "#166534"
    elif pct < 50:
        color = "#991b1b"
    else:
        color = "#4b5563"
    return f'<span style="color:{color};font-weight:700;">{pct}%</span>'

def format_performance(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1f}%"

def format_performance_intraday(value: float) -> str:
    if pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    color = "#166534" if value > 0 else "#991b1b" if value < 0 else "#4b5563"
    return f'<span style="color:{color};">{sign}{value:.1f}%</span>'

def format_52w_high(value: float) -> str:
    if pd.isna(value) or value > -3:
        return "🚀"
    return f"{value:.1f}%"

def format_52w_low(value: float) -> str:
    if pd.isna(value) or value < 3:
        return "🐌"
    return f"{value:.1f}%"

def format_indicator(value: str) -> str:
    v = str(value).strip().lower()
    return "✅" if v == "yes" else "❌" if v == "no" else "-"

def format_volume_alert(value: str, rs_rank_252d) -> str:
    if not isinstance(value, str):
        return "-"
    v = value.strip().lower()
    try:
        rs = float(rs_rank_252d) >= 0.80
    except:
        rs = False
    if v == "positive" and rs:
        return "💎"
    if v == "positive":
        return "🟩"
    if v == "negative":
        return "🟥"
    return "-"

def format_volatility(value: str) -> str:
    v = str(value).strip().lower()
    if v == "compression":
        return "🎯"
    if v == "spike":
        return "⚠️"
    return "-"

def format_multiple(value) -> str:
    try:
        v = float(value)
        txt = f"{v:.2f}x"
        if v >= 10:
            color = "#991b1b"
        elif v >= 4:
            color = "#d97706"
        elif v > 0:
            color = "#166534"
        else:
            color = "#4b5563"
        return f'<span style="color:{color};font-weight:700;">{txt}</span>'
    except:
        return "-"

def format_stage_label(value: str) -> str:
    v = str(value).strip().lower()
    if v == "stage 1": return "🟡"
    if v == "stage 2": return "🟢"
    if v == "stage 3": return "🟠"
    if v == "stage 4": return "🔴"
    return "⚪"

# ---------------------------------
# Simple Chip CSS (no tooltip)
# ---------------------------------
def build_simple_css() -> str:
    base = """
    <style>
    .ticker-chip {
      display: inline-block;
      padding: 4px 10px;
      margin: 2px;
      border-radius: 8px;
      font-weight: 700;
      font-family: ui-monospace, monospace;
      font-size: 13px;
      white-space: nowrap;
    }
    """
    parts = []
    for g, (light, dark) in GROUP_PALETTE.items():
        slug = slugify(g)
        parts.append(f"""
        .ticker-chip.chip-{slug} {{
          background: {light}22;
          border: 1px solid {light}66;
          color: {dark};
        }}""")
    return base + "\n".join(parts) + "</style>"

def make_ticker_chip(ticker: str, group: str | None) -> str:
    cls = f"ticker-chip chip-{slugify(group)}" if group and use_group_colors else "ticker-chip"
    return f'<span class="{cls}">{ticker}</span>'

# ---------------------------------
# Breadth Chart
# ---------------------------------
def breadth_chart(df: pd.DataFrame, col: str, color: str) -> alt.Chart:
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%b %d")
    return alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X("date:N", sort=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{col}:Q"),
        tooltip=[alt.Tooltip("date", title="Date"), alt.Tooltip(col, title=col.replace("_", " ").title())]
    ).properties(height=300)

# ---------------------------------
# Heat / P&V Visualizations
# ---------------------------------
def render_heat_section(df_heat: pd.DataFrame):
    if df_heat.empty:
        st.warning("Heat data not available.")
        return

    latest = df_heat.sort_values('date').groupby('ticker').tail(1)
    latest_date = df_heat['date'].max().strftime("%b %d, %Y")

    st.subheader("🧠 Price & Volume Factor Analysis")
    st.caption(f"Data as of {latest_date}")

    fig_scatter = px.scatter(latest, x='VolumeFactor', y='PriceFactor', color='code',
                            hover_data=['ticker'], height=560,
                            labels={"VolumeFactor":"Volume Factor", "PriceFactor":"Price Factor"})
    fig_scatter.update_traces(marker=dict(size=12, opacity=0.8))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Heatmaps
    dates = sorted(df_heat['date'].unique())
    tickers = df_heat.groupby(['ticker','code'])['date'].min().reset_index().sort_values(['code','ticker'])['ticker'].tolist()

    vol_pivot = df_heat.pivot(index='ticker', columns='date', values='VolumeFactor').reindex(tickers, dates)
    price_pivot = df_heat.pivot(index='ticker', columns='date', values='PriceFactor').reindex=tickers, columns=dates)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(go.Figure(go.Heatmap(z=vol_pivot.values, x=[d.strftime("%m-%d") for d in dates],
                                             y=vol_pivot.index, colorscale='RdYlGn', showscale=False)), use_container_width=True)
        st.caption("Volume Factor")
    with col2:
        st.plotly_chart(go.Figure(go.Heatmap(z=price_pivot.values, x=[d.strftime("%m-%d") for d in dates],
                                             y=price_pivot.index, colorscale='RdYlGn', showscale=False)), use_container_width=True)
        st.caption("Price Factor")

# ---------------------------------
# Main Dashboard
# ---------------------------------
def render_dashboard(df_etf: pd.DataFrame, df_rs: pd.DataFrame):
    st.title("US Market Daily Snapshot")
    st.markdown(build_simple_css(), unsafe_allow_html=True)

    latest, rs_last_n = process_data(df_etf, df_rs)
    latest_date = latest["date"].max().date() if "date" in latest.columns else "N/A"
    st.caption(f"Latest data: **{latest_date}**")

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        hide_weak_rs = st.toggle("Hide RS < 85%", False)
        hide_overextended = st.toggle("Hide Extension ≥4x", False)
        hide_weak_pv = st.toggle("Hide weak Price/Volume", False)

    # Load optional data
    try:
        df_heat = load_heat_csv()
    except:
        df_heat = pd.DataFrame()

    try:
        df_stage = load_stage_csv()
        latest = latest.reset_index().merge(df_stage[['ticker','stage_label_core','stage_label_adj']], on='ticker', how='left').set_index('ticker')
    except:
        pass

    # Group tables
    for group in GROUP_ORDER:
        st.markdown(f"### 📌 {group}")
        if group not in latest.group.values:
            st.info("No data")
            continue

        df_group = latest[latest["group"] == group].copy()

        if hide_weak_rs and "rs_rank_21d" in df_group.columns:
            df_group = df_group[df_group["rs_rank_21d"] >= 0.85]
        if hide_overextended and "ratio_pct_dist_to_atr_pct" in df_group.columns:
            df_group = df_group[df_group["ratio_pct_dist_to_atr_pct"] <= 4]

        if df_group.empty:
            st.info("No tickers after filters")
            continue

        # Sort by RS
        if "rs_rank_21d" in df_group.columns:
            df_group = df_group.sort_values("rs_rank_21d", ascending=False)

        rows = []
        for t, row in df_group.iterrows():
            spark = rs_last_n[rs_last_n["ticker"] == t]["rs_to_spy"].tolist()
            chip = make_ticker_chip(t, row.get("group"))

            rows.append({
                "Ticker": chip,
                "RS Spark": create_sparkline(spark),
                "RS 1M": format_rank(row.get("rs_rank_21d")),
                "RS 1Y": format_rank(row.get("rs_rank_252d")),
                "Vol Alert": format_volume_alert(row.get("volume_alert", ""), row.get("rs_rank_252d")),
                "Volatility": format_volatility(row.get("volatility_signal", "")),
                "Intraday": format_performance_intraday(row.get("ret_intraday")),
                "1D": format_performance(row.get("ret_1d")),
                "1W": format_performance(row.get("ret_1w")),
                "1M": format_performance(row.get("ret_1m")),
                "52W High": format_52w_high(row.get("pct_below_high")),
                "52W Low": format_52w_low(row.get("pct_above_low")),
                "Ext Multiple": format_multiple(row.get("ratio_pct_dist_to_atr_pct")),
                "SMA5": format_indicator(row.get("above_sma5")),
                "SMA10": format_indicator(row.get("above_sma10")),
                "SMA20": format_indicator(row.get("above_sma20")),
                "Core": format_stage_label(row.get("stage_label_core", "")),
                "Adj": format_stage_label(row.get("stage_label_adj", ""))
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Breadth
    counts = compute_threshold_counts(df_etf)
    if not counts.empty:
        st.markdown("### ✏️ Breadth Gauge (last 21 days)")
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(breadth_chart(counts, "count_over_85", "green"), use_container_width=True)
        with c2:
            st.altair_chart(breadth_chart(counts, "count_under_50", "red"), use_container_width=True)

    # Heat section
    render_heat_section(df_heat)

# ---------------------------------
# Main
# ---------------------------------
def main():
    try:
        df_etf, df_rs = load_data()
        render_dashboard(df_etf, df_rs)
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
