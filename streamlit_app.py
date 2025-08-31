import streamlit as st
import pandas as pd
import io, base64
import matplotlib.pyplot as plt

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

# Latest snapshot per ticker
latest = (
    df_etf.sort_values("date")
          .groupby("ticker", as_index=False)
          .tail(1)
          .set_index("ticker")
)

# Last 21 trading days for sparklines
LOOKBACK = 21
df_rs_sorted = df_rs.sort_values(["ticker", "date"])
rs_lastN = df_rs_sorted.groupby("ticker").tail(LOOKBACK)

# Global y-scale for comparable sparklines
RS_MIN = float(rs_lastN["rs_to_spy"].min()) if not rs_lastN.empty else 0.9
RS_MAX = float(rs_lastN["rs_to_spy"].max()) if not rs_lastN.empty else 1.1

# --- Helpers ---
def sparkline_img(series_vals, width=120, height=36, y_domain=None):
    """Matplotlib line sparkline with last point highlighted, returned as <img> tag."""
    if not series_vals:
        return ""
    fig = plt.figure(figsize=(width/96, height/96), dpi=96)
    ax = fig.add_axes([0,0,1,1])
    ax.plot(range(len(series_vals)), series_vals, linewidth=1.5, color="green")
    ax.plot([len(series_vals)-1], [series_vals[-1]], "o", color="darkgreen", markersize=4)
    ax.axis("off")
    if y_domain and len(y_domain) == 2:
        ax.set_ylim(y_domain[0], y_domain[1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" alt="sparkline" />'

def fmt_rank_percent(val):
    """RS rank stored as 0–1 → integer % (e.g., 0.87 -> 87%)."""
    if pd.isnull(val):
        return ""
    try:
        return f"{int(round(float(val) * 100))}%"
    except Exception:
        return ""

def tick_icon(v):
    return "✅" if bool(v) else "❌"

# --- Page ---
st.title("US Market Daily Snapshot")
st.caption(f"Latest data date: {latest['date'].max().date()}")

# --- Render by group (assumes 'group' exists) ---
for group_name, tickers in latest.groupby("group").groups.items():
    st.header(f"📌 {group_name}")
    rows = []
    for ticker in tickers:
        row = latest.loc[ticker]
        series = rs_lastN.loc[rs_lastN["ticker"] == ticker, "rs_to_spy"].tolist()
        spark = sparkline_img(series, y_domain=(RS_MIN, RS_MAX))

        rows.append({
            "Ticker": ticker,
            "RS Sparkline": spark,
            "RS Rank (21D)": fmt_rank_percent(row.get("rs_rank_21d")),
            "RS Rank (252D)": fmt_rank_percent(row.get("rs_rank_252d")),
            "Volume Alert": row.get("volume_alert", "-"),

            " ": "",  # spacer between ranks and performance

            # Performance already in %, so no *100
            "1D": f"{row['ret_1d']:.1f}%" if pd.notnull(row['ret_1d']) else "",
            "1W": f"{row['ret_1w']:.1f}%" if pd.notnull(row['ret_1w']) else "",
            "1M": f"{row['ret_1m']:.1f}%" if pd.notnull(row['ret_1m']) else "",

            "  ": "",  # spacer between performance and SMA

            "SMA5":  tick_icon(row.get("above_sma5")),
            "SMA10": tick_icon(row.get("above_sma10")),
            "SMA20": tick_icon(row.get("above_sma20")),
        })
    disp = pd.DataFrame(rows)
    st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)
