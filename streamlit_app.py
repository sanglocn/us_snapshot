import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io
import base64
import re
from typing import List, Dict, Tuple

# Configuration
st.set_page_config(page_title="US Market Snapshot", layout="wide")

# Constants
DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv"
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

# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data(ttl=900)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess ETF and RS data from CSV files."""
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs = pd.read_csv(DATA_URLS["rs"])
    df_etf["date"] = pd.to_datetime(df_etf["date"])
    df_rs["date"] = pd.to_datetime(df_rs["date"])
    return df_etf, df_rs


# ---------------------------
# Data Processing
# ---------------------------
def process_data(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process ETF and RS data for dashboard display."""
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK_DAYS)
    return latest, rs_last_n


def compute_threshold_counts(df_etf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day counts:
      - count_over_85: number of tickers with rs_rank_21d >= 0.85
      - count_under_50: number of tickers with rs_rank_21d < 0.50
    Keep only the last 21 trading days present in the data.
    """
    if "rs_rank_21d" not in df_etf.columns:
        return pd.DataFrame(columns=["date", "count_over_85", "count_under_50"])

    tmp = df_etf.copy()
    # Comparisons with NaN yield False, which is fine for counting
    tmp["over_85"] = tmp["rs_rank_21d"] >= 0.85
    tmp["under_50"] = tmp["rs_rank_21d"] < 0.50

    daily = (
        tmp.groupby("date", as_index=False)
           .agg(count_over_85=("over_85", "sum"),
                count_under_50=("under_50", "sum"))
           .sort_values("date")
    )

    # Limit to last 21 trading days available
    last_21_dates = daily["date"].drop_duplicates().sort_values().tail(21)
    daily_21 = daily[daily["date"].isin(last_21_dates)].copy().sort_values("date")
    return daily_21


# ---------------------------
# Visualization Helpers
# ---------------------------
def create_sparkline(values: List[float], width: int = 155, height: int = 36) -> str:
    """Generate a sparkline image from a series of values."""
    if not values:
        return ""

    fig = plt.figure(figsize=(width / 96, height / 96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(range(len(values)), values, linewidth=1.5, color="green")
    ax.plot(len(values) - 1, values[-1], "o", color="darkgreen", markersize=4)
    ax.axis("off")

    y_min, y_max = min(values), max(values)
    padding = (y_max - y_min) * 0.05 or 0.01
    ax.set_ylim(y_min - padding, y_max + padding)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}" alt="sparkline" />'


def breadth_column_chart(df: pd.DataFrame, value_col: str, title: str) -> alt.Chart:
    """Altair column chart helper for breadth counts."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{value_col}:Q", title=title),
            tooltip=[alt.Tooltip("date:T", title="Date"),
                     alt.Tooltip(f"{value_col}:Q", title=title)]
        )
        .properties(height=320, title=title)
    )


# ---------------------------
# Formatting Helpers
# ---------------------------
def format_rank(value: float) -> str:
    """Format rank value as percentage with right-aligned styling."""
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    return f'<span style="display:block; text-align:right;">{int(round(value * 100))}%</span>'


def format_performance(value: float) -> str:
    """Format performance value as percentage with right-aligned styling."""
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    return f'<span style="display:block; text-align:right;">{value:.1f}%</span>'


def format_indicator(value: str) -> str:
    """Return checkmark or cross icon with centered styling based on input."""
    value = str(value).strip().lower()
    if value == "yes":
        return '<span style="color:green; display:block; text-align:center;">✅</span>'
    if value == "no":
        return '<span style="color:red; display:block; text-align:center;">❌</span>'
    return '<span style="display:block; text-align:center;">-</span>'


def format_volume_alert(value: str) -> str:
    """Format volume alert value with centered styling."""
    value = str(value).strip()
    return f'<span style="display:block; text-align:center;">{value}</span>'


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')


# ---------------------------
# Table Rendering
# ---------------------------
def render_group_table(group_name: str, rows: List[Dict]) -> None:
    """Render a styled table for a group of tickers."""
    table_id = f"tbl-{slugify(group_name)}"
    html = pd.DataFrame(rows).to_html(escape=False, index=False)

    css = f"""
        #{table_id} table {{
            width: 100%;
            border-collapse: collapse;
        }}
        #{table_id} table th {{
            text-align: center !important;
        }}
        #{table_id} table td:nth-child(3),
        #{table_id} table td:nth-child(4),
        #{table_id} table td:nth-child(5),
        #{table_id} table td:nth-child(7),
        #{table_id} table td:nth-child(8),
        #{table_id} table td:nth-child(9),
        #{table_id} table td:nth-child(11),
        #{table_id} table td:nth-child(12),
        #{table_id} table td:nth-child(13) {{
            text-align: center !important;
        }}
        #{table_id} table td:nth-child(3),
        #{table_id} table td:nth-child(4),
        #{table_id} table td:nth-child(7),
        #{table_id} table td:nth-child(8),
        #{table_id} table td:nth-child(9) {{
            text-align: right !important;
        }}
    """
    st.markdown(f'<div id="{table_id}"><style>{css}</style>{html}</div>', unsafe_allow_html=True)


# ---------------------------
# Dashboard Rendering
# ---------------------------
def render_dashboard(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> None:
    """Render the complete US Market Snapshot dashboard."""
    st.title("US Market Daily Snapshot")

    # Top breadth charts (last 21 trading days)
    counts_21 = compute_threshold_counts(df_etf)
    if not counts_21.empty:
        start_date = counts_21["date"].min().date()
        end_date = counts_21["date"].max().date()
        st.subheader("Breadth: RS Threshold Counts (Last 21 Trading Days)")
        st.caption(f"From {start_date} to {end_date}")

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                breadth_column_chart(counts_21, "count_over_85", "Count ≥ 0.85"),
                use_container_width=True
            )
        with c2:
            st.altair_chart(
                breadth_column_chart(counts_21, "count_under_50", "Count < 0.50"),
                use_container_width=True
            )
    else:
        st.info("`rs_rank_21d` not found in ETF data — breadth charts skipped.")

    # Rest of dashboard
    latest, rs_last_n = process_data(df_etf, df_rs)
    st.caption(f"Latest data date: {latest['date'].max().date()}")

    group_tickers = latest.groupby("group").groups
    for group_name in GROUP_ORDER:
        if group_name not in group_tickers:
            continue
        st.header(f"📌 {group_name}")
        rows = []
        tickers_in_group = group_tickers[group_name]

        for ticker in tickers_in_group:
            row = latest.loc[ticker]
            spark_series = rs_last_n.loc[rs_last_n["ticker"] == ticker, "rs_to_spy"].tolist()
            rows.append({
                "Ticker": ticker,
                "Relative Strength": create_sparkline(spark_series),
                "RS Rank (1M)": format_rank(row.get("rs_rank_21d")),
                "RS Rank (1Y)": format_rank(row.get("rs_rank_252d")),
                "Volume Alert": format_volume_alert(row.get("volume_alert", "-")),
                " ": "",
                "Intraday": format_performance(row.get("ret_intraday")),
                "1D Return": format_performance(row.get("ret_1d")),
                "1W Return": format_performance(row.get("ret_1w")),
                "1M Return": format_performance(row.get("ret_1m")),
                "  ": "",
                "Above SMA5": format_indicator(row.get("above_sma5")),
                "Above SMA10": format_indicator(row.get("above_sma10")),
                "Above SMA20": format_indicator(row.get("above_sma20")),
            })

        render_group_table(group_name, rows)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    df_etf, df_rs = load_data()
    render_dashboard(df_etf, df_rs)
