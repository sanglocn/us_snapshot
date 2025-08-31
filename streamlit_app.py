import streamlit as st
import pandas as pd
import altair as alt
import base64

st.set_page_config(page_title="US Market Daily Snapshot", layout="wide")

# --------------------------------------------------
# Config: GitHub CSV locations
# --------------------------------------------------
URL_ETF = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv"
URL_RS  = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv"

# --------------------------------------------------
# Fetch data
# --------------------------------------------------
@st.cache_data(ttl=600)
def load_data():
    df_etf = pd.read_csv(URL_ETF)
    df_rs  = pd.read_csv(URL_RS)
    # coerce dtypes
    df_etf["date"] = pd.to_datetime(df_etf["date"])
    df_rs["date"]  = pd.to_datetime(df_rs["date"])
    return df_etf, df_rs

df_etf, df_rs = load_data()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Controls")

# Group filter
groups = sorted(df_etf["group"].dropna().unique().tolist()) if "group" in df_etf.columns else ["All"]
selected_groups = st.sidebar.multiselect("Groups", groups, default=groups)

# Sparkline lookback (trading days)
lookback = st.sidebar.selectbox("Sparkline lookback (trading days)", [21, 63, 252], index=0)

# Toggle RS columns
show_rs21  = st.sidebar.checkbox("Show RS Rank (21D)", value=True)
show_rs252 = st.sidebar.checkbox("Show RS Rank (252D)", value=True)

# --------------------------------------------------
# Prepare base frames
# --------------------------------------------------
# latest snapshot per ticker for metrics
latest = (
    df_etf.sort_values("date")
          .groupby("ticker", as_index=False)
          .tail(1)
          .set_index("ticker")
)

if "group" in latest.columns:
    latest = latest[latest["group"].isin(selected_groups)] if selected_groups else latest

# Sparkline source: keep last N by ticker
df_rs = df_rs.sort_values(["ticker", "date"])
rs_lastN = df_rs.groupby("ticker").tail(lookback)

# Global absolute scale for sparklines (so charts are comparable)
if not rs_lastN.empty:
    RS_MIN = float(rs_lastN["rs_to_spy"].min())
    RS_MAX = float(rs_lastN["rs_to_spy"].max())
else:
    RS_MIN, RS_MAX = 0.9, 1.1  # fallback

# --------------------------------------------------
# Helpers: sparkline + styling
# --------------------------------------------------
def sparkbar_img(series_vals, width=120, height=36):
    """Return a base64 <img> tag for a green bar sparkline (last bar darker)."""
    if not isinstance(series_vals, (list, tuple)) or len(series_vals) == 0:
        return ""
    df_tmp = pd.DataFrame({"x": range(len(series_vals)), "y": series_vals})
    df_tmp["is_current"] = df_tmp["x"] == df_tmp["x"].max()

    chart = (
        alt.Chart(df_tmp)
        .mark_bar()
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, max(0, len(series_vals)-1)])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, domain=[RS_MIN, RS_MAX])),
            color=alt.condition(alt.datum.is_current, alt.value("darkgreen"), alt.value("lightgreen")),
            tooltip=[alt.Tooltip("y:Q", title="RS to SPY", format=".4f")],
        )
        .properties(width=width, height=height)
    )

    img_bytes = chart.to_image(format="png")
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" alt="sparkline" />'

def highlight_rs_cell(val):
    """Full-cell background for RS Rank thresholds."""
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 85:
        bg = "green"
    elif v >= 50:
        bg = "yellow"
    else:
        bg = "red"
    return f"background-color: {bg}; color: black; font-weight: bold"

def highlight_volume_cell(val):
    if val == "Positive":
        return "background-color: green; color: black; font-weight: bold"
    if val == "Negative":
        return "background-color: red; color: black; font-weight: bold"
    return "background-color: white; color: black;"

def shade_performance(val):
    """Gradient fill for returns (fraction)."""
    try:
        v = float(val)
    except Exception:
        return ""
    cap = 0.20  # ±20% clamp for color intensity
    if v > 0:
        intensity = 255 - int(min(v, cap) / cap * 95)  # 255..160
        return f"background-color: rgb({intensity}, 255, {intensity});"
    elif v < 0:
        intensity = 255 - int(min(-v, cap) / cap * 95)
        return f"background-color: rgb(255, {intensity}, {intensity});"
    else:
        return "background-color: white;"

def sma_icon(val):
    truthy = {True, "True", "YES", "Yes", "yes", "Y", "1", 1}
    return "✅" if val in truthy else "❌"

def color_sma_text(val):
    return "color: green; font-weight: 700" if val == "✅" else "color: red; font-weight: 700"

# --------------------------------------------------
# Page header
# --------------------------------------------------
st.title("US Market Daily Snapshot")
if not latest.empty:
    st.caption(f"Latest data date: {latest['date'].max().date()}  •  Sparkline lookback: {lookback} trading days")

# --------------------------------------------------
# Render by group
# --------------------------------------------------
grouped = latest.groupby("group") if "group" in latest.columns else [("All", latest.index)]

for group_name, idx in (grouped.groups.items() if isinstance(grouped, pd.core.groupby.generic.DataFrameGroupBy) else [("All", latest.index)]):
    st.header(f"📌 {group_name}")

    rows = []
    for ticker in idx:
        row = latest.loc[ticker]

        # Sparkline series for this ticker (last N)
        series = rs_lastN.loc[rs_lastN["ticker"] == ticker, "rs_to_spy"].tolist()
        spark_img = sparkbar_img(series) if series else ""

        # Build table row
        record = {
            "Ticker": ticker,
            "RS Sparkline": spark_img,
        }
        if show_rs21:
            record["RS Rank (21D)"] = row.get("rs_rank_21d", None)
        if show_rs252:
            record["RS Rank (252D)"] = row.get("rs_rank_252d", None)

        record.update({
            "": "",  # spacer
            "Volume Alert": row.get("volume_alert", "-"),
            "  ": "",  # spacer
            # performance (fractions -> format later)
            "1D": row.get("ret_1d", None),
            "1W": row.get("ret_1w", None),
            "1M": row.get("ret_1m", None),
            "3M": row.get("ret_3m", None),
            "6M": row.get("ret_6m", None),
            "1Y": row.get("ret_1y", None),
            "YTD": row.get("ret_ytd", None),
            "   ": "",  # spacer
            # SMA checks
            "SMA5":  sma_icon(row.get("above_sma5", 0)),
            "SMA10": sma_icon(row.get("above_sma10", 0)),
            "SMA20": sma_icon(row.get("above_sma20", 0)),
            "SMA50": sma_icon(row.get("above_sma50", 0)),
            "SMA100":sma_icon(row.get("above_sma100", 0)),
        })
        rows.append(record)

    disp = pd.DataFrame(rows)

    # Performance columns formatting
    perf_cols = ["1D","1W","1M","3M","6M","1Y","YTD"]
    numeric_for_style = disp[perf_cols].copy()
    for c in perf_cols:
        disp[c] = numeric_for_style[c].map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")

    # Build subset lists depending on toggles
    rs_cols = [c for c in ["RS Rank (21D)", "RS Rank (252D)"] if c in disp.columns]

    # Styler with all visual rules
    styler = (
        disp.style
        .hide(axis="index")
        .format(na_rep="", subset=disp.columns)
        .applymap(highlight_rs_cell, subset=rs_cols)
        .applymap(highlight_volume_cell, subset=["Volume Alert"])
        .apply(lambda s: [shade_performance(v) for v in numeric_for_style[s.name]]
               , subset=perf_cols, axis=0)
        .applymap(color_sma_text, subset=["SMA5","SMA10","SMA20","SMA50","SMA100"])
    )

    # Render as HTML (so sparkline <img> is preserved)
    st.write(styler.to_html(escape=False), unsafe_allow_html=True)
