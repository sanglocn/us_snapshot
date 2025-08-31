import streamlit as st
import pandas as pd
import base64

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

# last 21 trading days for sparklines (global scale)
LOOKBACK = 21
df_rs = df_rs.sort_values(["ticker", "date"])
rs_lastN = df_rs.groupby("ticker").tail(LOOKBACK)
RS_MIN = float(rs_lastN["rs_to_spy"].min()) if not rs_lastN.empty else 0.9
RS_MAX = float(rs_lastN["rs_to_spy"].max()) if not rs_lastN.empty else 1.1

# --- Helpers ---
def sparkline_img(series_vals, width=120, height=36, y_domain=None):
    """Altair line sparkline with last point highlighted; returns <img> tag via PNG export."""
    if not series_vals:
        return ""
    import altair as alt
    alt.renderers.set_embed_options(actions=False)
    df_tmp = pd.DataFrame({"x": list(range(len(series_vals))), "y": series_vals})
    df_tmp["is_current"] = df_tmp["x"] == df_tmp["x"].max()

    y_scale = alt.Scale(zero=False)
    if y_domain and len(y_domain) == 2:
        y_scale = alt.Scale(zero=False, domain=list(y_domain))

    line = (
        alt.Chart(df_tmp)
           .mark_line()
           .encode(
               x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, max(0, len(series_vals)-1)])),
               y=alt.Y("y:Q", axis=None, scale=y_scale),
               color=alt.value("green"),
           )
           .properties(width=width, height=height)
    )
    last_pt = (
        alt.Chart(df_tmp[df_tmp["is_current"]])
           .mark_point(size=40, filled=True, color="darkgreen")
           .encode(x="x:Q", y="y:Q")
    )
    chart = line + last_pt
    png_bytes = chart.save(format="png")  # requires vl-convert-python
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" alt="sparkline" />'

def fmt_rank_percent(val):
    """Robustly show rank as integer % (no decimals). Accepts 0–1 or 0–100 inputs."""
    if pd.isnull(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    # if it looks like a fraction, convert to percent
    if 0 <= v <= 1:
        v *= 100.0
    # clamp for safety then round
    v = max(0, min(100, v))
    return f"{int(round(v))}%"

def fmt_pct(val):
    """Performance already stored as percent, just show with 1 decimal and % if numeric; pass through if already str%."""
    if pd.isnull(val):
        return ""
    if isinstance(val, str) and "%" in val:
        return val
    try:
        return f"{float(val):.1f}%"
    except Exception:
        return str(val)

def tick_icon(v):
    return "✅" if bool(v) else "❌"

# --- Page ---
st.title("US Market Daily Snapshot")
st.caption(f"Latest data date: {latest['date'].max().date()}")

# group safely (fallback to "All")
groups = latest.groupby("group").groups if "group" in latest.columns else {"All": latest.index}

# --- Render by group ---
for group_name, tickers in groups.items():
    st.header(f"📌 {group_name}")
    rows = []
    for ticker in tickers:
        row = latest.loc[ticker]
        series = rs_lastN.loc[rs_lastN["ticker"] == ticker, "rs_to_spy"].tolist()
        spark = sparkline_img(series, y_domain=(RS_MIN, RS_MAX))

        rows.append({
            "Ticker": ticker,
            "RS Sparkline": spark,
            # ✅ use the correct CSV columns: rs_rank_21d and rs_rank_252d
            "RS Rank (21D)": fmt_rank_percent(row.get("rs_rank_21d")),
            "RS Rank (252D)": fmt_rank_percent(row.get("rs_rank_252d")),
            "Volume Alert": row.get("volume_alert", "-"),

            " ": "",  # spacer

            # Performance (already % values)
            "1D": fmt_pct(row.get("ret_1d")),
            "1W": fmt_pct(row.get("ret_1w")),
            "1M": fmt_pct(row.get("ret_1m")),

            "  ": "",  # spacer

            # SMA checks
            "SMA5":  tick_icon(row.get("above_sma5")),
            "SMA10": tick_icon(row.get("above_sma10")),
            "SMA20": tick_icon(row.get("above_sma20")),
        })
    disp = pd.DataFrame(rows)
    st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)
