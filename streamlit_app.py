import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

st.set_page_config(page_title="US Market Snapshot", layout="wide")

URL_ETF = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv"
URL_RS  = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv"

# --- Load data ---
@st.cache_data(ttl=900)
def load_data():
    df_etf = pd.read_csv(URL_ETF)
    df_rs  = pd.read_csv(URL_RS)
    df_etf["date"] = pd.to_datetime(df_etf["date"])
    df_rs["date"]  = pd.to_datetime(df_rs["date"])
    return df_etf, df_rs

df_etf, df_rs = load_data()

# latest snapshot per ticker
latest = (
    df_etf.sort_values("date")
          .groupby("ticker", as_index=False)
          .tail(1)
          .set_index("ticker")
)

# last 21 trading days for sparklines
LOOKBACK = 21
df_rs = df_rs.sort_values(["ticker", "date"])
rs_lastN = df_rs.groupby("ticker").tail(LOOKBACK)

RS_MIN, RS_MAX = rs_lastN["rs_to_spy"].min(), rs_lastN["rs_to_spy"].max()

# --- Sparkline helper ---
def sparkbar_img(series_vals, width=120, height=30):
    if not series_vals:
        return ""
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_axes([0,0,1,1])
    colors = ["lightgreen"] * len(series_vals)
    colors[-1] = "darkgreen"
    ax.bar(range(len(series_vals)), series_vals, color=colors)
    ax.axis("off")
    ax.set_ylim(RS_MIN, RS_MAX)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" alt="sparkline" />'

# --- Page ---
st.title("US Market Daily Snapshot")
st.caption(f"Latest data date: {latest['date'].max().date()}")

# --- Render by group (no Ungrouped handling needed) ---
for group_name, tickers in latest.groupby("group").groups.items():
    st.header(f"📌 {group_name}")
    rows = []
    for ticker in tickers:
        row = latest.loc[ticker]
        series = rs_lastN.loc[rs_lastN["ticker"]==ticker, "rs_to_spy"].tolist()
        spark = sparkbar_img(series)

        rows.append({
            "Ticker": ticker,
            "RS Sparkline": spark,
            "RS Rank (21D)": f"{int(round(row['rs_rank_21d']))}%" if pd.notnull(row['rs_rank_21d']) else "",
            "RS Rank (252D)": f"{int(round(row['rs_rank_252d']))}%" if pd.notnull(row['rs_rank_252d']) else "",
            "1D": f"{row['ret_1d']*100:.1f}%" if pd.notnull(row['ret_1d']) else "",
            "1W": f"{row['ret_1w']*100:.1f}%" if pd.notnull(row['ret_1w']) else "",
            "1M": f"{row['ret_1m']*100:.1f}%" if pd.notnull(row['ret_1m']) else "",
            "SMA10": "✅" if row.get("above_sma10") else "❌",
            "SMA20": "✅" if row.get("above_sma20") else "❌",
            "SMA50": "✅" if row.get("above_sma50") else "❌",
        })
    disp = pd.DataFrame(rows)
    st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)
