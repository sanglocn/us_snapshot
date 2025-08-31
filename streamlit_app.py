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
    y_min, y_max = float(rs_last_n["rs_to_spy"].min()), float(rs_last_n["rs_to_spy"].max()) if not rs_last_n.empty else (0.9, 1.1)
    return latest, rs_last_n, (y_min, y_max)

# Sparkline generation
def create_sparkline(series_vals, y_domain=None, width=120, height=36):
    if not series_vals:
        return ""
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(range(len(series_vals)), series_vals, linewidth=1.5, color="green")
    ax.plot(len(series_vals)-1, series_vals[-1], "o", color="darkgreen", markersize=4)
    ax.axis("off")
    if y_domain:
        ax.set_ylim(y_domain)
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
    return "✅" if bool(value) else "❌"

# Render dashboard
def render_dashboard(df_etf, df_rs):
    st.title("US Market Daily Snapshot")
    latest, rs_last_n, y_domain = get_processed_data(df_etf, df_rs)
    st.caption(f"Latest data date: {latest['date'].max().date()}")
    
    for group_name, tickers in latest.groupby("group").groups.items():
        st.header(f"📌 {group_name}")
        rows = []
        for ticker in tickers:
            row = latest.loc[ticker]
            spark_series = rs_last_n.loc[rs_last_n["ticker"] == ticker, "rs_to_spy"].tolist()
            rows.append({
                "Ticker": ticker,
                "RS Sparkline": create_sparkline(spark_series, y_domain),
                "RS Rank (21D)": format_rank(row.get("rs_rank_21d")),
                "RS Rank (252D)": format_rank(row.get("rs_rank_252d")),
                "Volume Alert": row.get("volume_alert", "-"),
                " ": "",
                "1D": format_perf(row.get("ret_1d")),
                "1W": format_perf(row.get("ret_1w")),
                "1M": format_perf(row.get("ret_1m")),
                "  ": "",
                "SMA5": tick_icon(row.get("above_sma5")),
                "SMA10": tick_icon(row.get("above_sma10")),
                "SMA20": tick_icon(row.get("above_sma20")),
            })
        st.write(pd.DataFrame(rows).to_html(escape=False, index=False), unsafe_allow_html=True)

# Run app
if __name__ == "__main__":
    df_etf, df_rs = load_data()
    render_dashboard(df_etf, df_rs)
