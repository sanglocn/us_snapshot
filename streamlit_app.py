import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
import re

st.set_page_config(page_title="US Market Snapshot", layout="wide")

# Constants
DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv"
}
LOOKBACK = 21

# Load and preprocess data
@st.cache_data(ttl=900)
def load_data():
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs = pd.read_csv(DATA_URLS["rs"])
    df_etf["date"] = pd.to_datetime(df_etf["date"])
    df_rs["date"] = pd.to_datetime(df_rs["date"])
    return df_etf, df_rs

# Data processing
def get_processed_data(df_etf, df_rs):
    latest = (
        df_etf.sort_values("date")
        .groupby("ticker")
        .tail(1)
        .set_index("ticker")
    )
    rs_last_n = (
        df_rs.sort_values(["ticker", "date"])
        .groupby("ticker")
        .tail(LOOKBACK)
    )
    return latest, rs_last_n

# Sparkline generation
def create_sparkline(series_vals, width=160, height=40):
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
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img style="display:block;margin:0 auto;max-width:100%;" src="data:image/png;base64,{b64}" alt="sparkline" />'

# Formatting helpers
def format_rank(value):
    return f"{int(round(float(value) * 100))}%" if pd.notna(value) else ""

def format_perf(value):
    return f"{float(value):.1f}%" if pd.notna(value) else ""

def tick_icon(value):
    v = str(value).strip().lower()
    if v == "yes":
        return '<span style="color:green;">✅</span>'
    elif v == "no":
        return '<span style="color:red;">❌</span>'
    else:
        return "-"

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

# Single group renderer with alignment + responsive spacer hiding
def render_group_table(group_name, rows):
    df = pd.DataFrame(rows)
    tbl_id = f"tbl-{slugify(group_name)}"

    st.markdown(
        f"""
        <style>
        /* table base */
        #{tbl_id} {{
          table-layout: fixed;
          width: 100%;
          border-collapse: collapse;
        }}
        #{tbl_id} th {{
          text-align: center !important;
          vertical-align: middle;
        }}
        #{tbl_id} td, #{tbl_id} th {{
          padding: 6px 8px;
          vertical-align: middle;
          font-size: 0.95rem;
          line-height: 1.25rem;
        }}

        /* widen Relative Strength (2nd column), narrow Ticker (1st) */
        #{tbl_id} td:nth-child(1), #{tbl_id} th:nth-child(1) {{ width: 90px; }}
        #{tbl_id} td:nth-child(2), #{tbl_id} th:nth-child(2) {{ width: 200px; }}

        /* hide spacer columns (6th and 10th) on small screens */
        @media (max-width: 768px) {{
            #{tbl_id} td:nth-child(6), #{tbl_id} th:nth-child(6),
            #{tbl_id} td:nth-child(10), #{tbl_id} th:nth-child(10) {{
                display: none;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Alignment: right by default; center Ticker, Relative Strength, SMA ticks
    center_cols = ["Ticker", "Relative Strength", "Above SMA5", "Above SMA10", "Above SMA20"]
    styler = (
        df.style
          .hide_index()
          .set_table_attributes(f'id="{tbl_id}"')
          .set_properties(subset=df.columns, **{"text-align": "right"})
          .set_properties(subset=center_cols, **{"text-align": "center"})
    )

    st.markdown(styler.to_html(escape=False), unsafe_allow_html=True)

# Render dashboard
def render_dashboard(df_etf, df_rs):
    st.title("US Market Daily Snapshot")
    latest, rs_last_n = get_processed_data(df_etf, df_rs)
    st.caption(f"Latest data date: {latest['date'].max().date()}")

    # vertical groups (original layout)
    for group_name, tickers in latest.groupby("group").groups.items():
        st.header(f"📌 {group_name}")
        rows = []
        for ticker in tickers:
            row = latest.loc[ticker]
            spark_series = rs_last_n.loc[rs_last_n["ticker"] == ticker, "rs_to_spy"].tolist()
            rows.append({
                "Ticker": ticker,                                     # centered
                "Relative Strength": create_sparkline(spark_series),  # centered + wider
                "RS Rank (1M)": format_rank(row.get("rs_rank_21d")),  # right
                "RS Rank (1Y)": format_rank(row.get("rs_rank_252d")), # right
                "Volume Alert": row.get("volume_alert", "-"),         # right (words/numbers)
                " ": "",                                              # spacer (hidden on small screens)
                "1D Return": format_perf(row.get("ret_1d")),          # right
                "1W Return": format_perf(row.get("ret_1w")),          # right
                "1M Return": format_perf(row.get("ret_1m")),          # right
                "  ": "",                                             # spacer (hidden on small screens)
                "Above SMA5": tick_icon(row.get("above_sma5")),       # centered
                "Above SMA10": tick_icon(row.get("above_sma10")),     # centered
                "Above SMA20": tick_icon(row.get("above_sma20")),     # centered
            })
        render_group_table(group_name, rows)

# Run app
if __name__ == "__main__":
    df_etf, df_rs = load_data()
    render_dashboard(df_etf, df_rs)
