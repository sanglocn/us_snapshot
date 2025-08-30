import streamlit as st
import pandas as pd
import altair as alt

# Load GitHub CSVs
price_url = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv"
rs_url = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv"

df_price = pd.read_csv(price_url)
df_rs = pd.read_csv(rs_url)

st.title("📊 US Market Daily Snapshot")

# 1. Relative Strength
st.header("1️⃣ Relative Strength vs SPY")
st.dataframe(df_rs[["ticker","rs_to_spy"]])
# Example sparkline chart
sparkline = alt.Chart(df_rs).mark_line().encode(
    x="day", y="rs_ratio", color="ticker"
).properties(width=150, height=60)
st.altair_chart(sparkline, use_container_width=True)

# 2. Price Performance
st.header("2️⃣ Price Performance Snapshot")
cols_perf = ["ticker","1D","1W","1M","3M","6M","1Y","YTD","intraday","pct_from_52w_high","pct_from_52w_low"]
st.dataframe(df_price[cols_perf])

# 3. Trend Check
st.header("3️⃣ Trend Check vs SMAs")
cols_sma = ["ticker","sma5_check","sma10_check","sma20_check","sma50_check","sma100_check"]
st.dataframe(df_price[cols_sma])

# 4. Volume Alerts
st.header("4️⃣ Volume Alerts")
alerts = df_price[df_price["volume_alert"] != "-"]
st.dataframe(alerts[["ticker","volume_alert"]])

# 5. Group Strength Rotation
st.header("5️⃣ Group Strength Rotation")
# Example bar chart by group
bar = alt.Chart(df_rs).mark_bar().encode(
    x="rs_rank_change_1d", y="ticker", color="group", tooltip=["ticker","group","rs_rank_change_1d"]
)
st.altair_chart(bar, use_container_width=True)
