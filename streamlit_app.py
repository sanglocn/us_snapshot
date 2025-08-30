import streamlit as st
import pandas as pd

# GitHub raw URLs
url_etf = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/us_snapshot_etf_price.csv"
url_rs  = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/us_snapshot_rs_sparkline.csv"

# Load data
df_etf = pd.read_csv(url_etf)
df_rs  = pd.read_csv(url_rs)

st.title("US ETF Snapshot Dashboard")

# --- Section 1: ETF Price Table ---
st.header("ETF Prices & Metrics")
st.dataframe(
    df_etf[[
        "date","ticker","adj_close","ret_1d","ret_1w","ret_1m",
        "ret_3m","ret_6m","ret_1y","ret_ytd",
        "rs_to_spy","rs_rank_21d","rs_rank_252d","volume_alert"
    ]]
)

# --- Section 2: Sparkline Relative Strength ---
st.header("Relative Strength vs SPY (Sparklines)")

tickers = df_rs["ticker"].unique().tolist()
selected = st.multiselect("Select tickers", tickers, default=tickers[:3])

for t in selected:
    df_sub = df_rs[df_rs["ticker"] == t]
    st.line_chart(df_sub.set_index("date")["rs_to_spy"], height=120)

# --- Section 3: Filters and Drilldown ---
st.header("Drilldown")
ticker_choice = st.selectbox("Choose a ticker", df_etf["ticker"].unique())
st.write(df_etf[df_etf["ticker"] == ticker_choice])
