import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import re
from typing import List, Dict

# Configuration
st.set_page_config(page_title="US Market Snapshot", layout="wide")

# Constants
DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv"
}
LOOKBACK_DAYS = 21

# Custom group order for visualization
GROUP_ORDER = [
    "Market",
    "Market Weight Sector",
    "Crypto",
    "Commodity",
    "Foreign Market",
    "Theme",
    "Stock"
]

# Data Loading
@st.cache_data(ttl=900)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess ETF and RS data from CSV files."""
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs = pd.read_csv(DATA_URLS["rs"])
    df_etf["date"] = pd.to_datetime(df_etf["date"])
    df_rs["date"] = pd.to_datetime(df_rs["date"])
    return df_etf, df_rs

# Data Processing
def process_data(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process ETF and RS data for dashboard display."""
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK_DAYS)
    return latest, rs_last_n

# Visualization
def create_sparkline(series_vals: List[float], width: int = 120, height: int = 36) -> str:
    """Generate a sparkline image from a series of values."""
    if not series_vals:
        return ""
    
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(range(len(series_vals)), series_vals, linewidth=1.5, color="green")
    ax.plot(len(series_vals)-1, series_vals[-1], "o", color="darkgreen", markersize=4)
    ax.axis("off")
    
    y_min, y_max = min(series_vals), max(series_vals)
    padding = (y_max - y_min) * 0.05 or 0.01
    ax.set_ylim(y_min - padding, y_max + padding)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}" alt="sparkline" />'

# Formatting Helpers
def format_rank(value: float) -> str:
    """Format rank value as percentage."""
    return f"{int(round(float(value) * 100))}%" if pd.notna(value) else ""

def format_perf(value: float) -> str:
    """Format performance value as percentage."""
    return f"{float(value):.1f}%" if pd.notna(value) else ""

def tick_icon(value: str) -> str:
    """Return checkmark or cross icon with centered styling."""
    value = str(value).strip().lower()
    if value == "yes":
        return '<span style="color:green; display:block; text-align:center;">✅</span>'
    elif value == "no":
        return '<span style="color:red; display:block; text-align:center;">❌</span>'
    return '<span style="display:block; text-align:center;">-</span>'

def volume_alert_format(value: str) -> str:
    """Format volume alert value with centered styling."""
    value = str(value).strip()
    return f'<span style="display:block; text-align:center;">{value}</span>'

def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')

# Table Rendering
def render_group_table(group_name: str, rows: List[Dict]) -> None:
    """Render a styled table for a group of tickers."""
    table_id = f"tbl-{slugify(group_name)}"
    html = pd.DataFrame(rows).to_html(escape=False, index=False)
    
    st.markdown(
        f"""
        <div id="{table_id}">
        <style>
        #{table_id} table {{
            width: 100%;
            border-collapse: collapse;
        }}
        #{table_id} table th {{
            text-align: center !important;
        }}
        #{table_id} table td:nth-child(5),
        #{table_id} table td:nth-child(11),
        #{table_id} table td:nth-child(12),
        #{table_id} table td:nth-child(13) {{
            text-align: center !important;
        }}
        </style>
        {html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Dashboard Rendering
def render_dashboard(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> None:
    """Render the complete US Market Snapshot dashboard."""
    st.title("US Market Daily Snapshot")
    
    latest, rs_last_n = process_data(df_etf, df_rs)
    st.caption(f"Latest data date: {latest['date'].max().date()}")
    
    # Get all groups and their tickers
    group_tickers = latest.groupby("group").groups
    
    # Iterate through groups in the specified order
    for group_name in GROUP_ORDER:
        if group_name not in group_tickers:
            continue  # Skip if group is not in the data
        st.header(f"📌 {group_name}")
        rows = []
        
        for ticker in group_tickers[group_name]:
            row = latest.loc[ticker]
            spark_series = rs_last_n.loc[rs_last_n["ticker"] == ticker, "rs_to_spy"].tolist()
            rows.append({
                "Ticker": ticker,
                "Relative Strength": create_sparkline(spark_series),
                "RS Rank (1M)": format_rank(row.get("rs_rank_21d")),
                "RS Rank (1Y)": format_rank(row.get("rs_rank_252d")),
                "Volume Alert": volume_alert_format(row.get("volume_alert", "-")),
                " ": "",
                "1D Return": format_perf(row.get("ret_1d")),
                "1W Return": format_perf(row.get("ret_1w")),
                "1M Return": format_perf(row.get("ret_1m")),
                "  ": "",
                "Above SMA5": tick_icon(row.get("above_sma5")),
                "Above SMA10": tick_icon(row.get("above_sma10")),
                "Above SMA20": tick_icon(row.get("above_sma20")),
            })
        
        render_group_table(group_name, rows)

# Main Execution
if __name__ == "__main__":
    df_etf, df_rs = load_data()
    render_dashboard(df_etf, df_rs)
