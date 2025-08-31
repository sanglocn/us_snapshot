import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

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
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK)
    return latest, rs_last_n

# Sparkline generation
def create_sparkline(series_vals, width=120, height=36):
    if not series_vals:
        return ""
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(range(len(series_vals)), series_vals, linewidth=1.5, color="green")
    ax.plot(len(series_vals)-1, series_vals[-1], "o", color="darkgreen", markersize=4)
    ax.axis("off")
    # Set y-axis limits based on the series' own min and max
    if series_vals:
        y_min, y_max = min(series_vals), max(series_vals)
        padding = (y_max - y_min) * 0.05 or 0.01
        ax.set_ylim(y_min - padding, y_max + padding)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}" alt="sparkline" />'

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

# Render dashboard
def render_dashboard(df_etf, df_rs):
    st.title("US Market Daily Snapshot")
    latest, rs_last_n = get_processed_data(df_etf, df_rs)
    st.caption(f"Latest data date: {latest['date'].max().date()}")

    for group_name, tickers in latest.groupby("group").groups.items():
        st.header(f"📌 {group_name}")
        rows = []
        for ticker in tickers:
            row = latest.loc[ticker]
            spark_series = rs_last_n.loc[rs_last_n["ticker"] == ticker, "rs_to_spy"].tolist()
            rows.append({
                "Ticker": ticker,
                "Relative Strength": create_sparkline(spark_series),
                "RS Rank (1M)": format_rank(row.get("rs_rank_21d")),
                "RS Rank (1Y)": format_rank(row.get("rs_rank_252d")),
                "Volume Alert": row.get("volume_alert", "-"),
                " ": "",
                "1D Return": format_perf(row.get("ret_1d")),
                "1W Return": format_perf(row.get("ret_1w")),
                "1M Return": format_perf(row.get("ret_1m")),
                "  ": "",
                "Above SMA5": tick_icon(row.get("above_sma5")),
                "Above SMA10": tick_icon(row.get("above_sma10")),
                "Above SMA20": tick_icon(row.get("above_sma20")),
            })
        table_html = pd.DataFrame(rows).to_html(escape=False, index=False)
        st.markdown(
            f"""
            <style>
            table th {{
                text-align: center !important;
            }}
            </style>
            {table_html}
            """,
            unsafe_allow_html=True,
        )

# Run app
if __name__ == "__main__":
    df_etf, df_rs = load_data()
    render_dashboard(df_etf, df_rs)
