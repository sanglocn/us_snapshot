import streamlit as st
import pandas as pd
import base64
import io
import matplotlib.pyplot as plt  # fallback for sparkline export

st.set_page_config(page_title="US Market Daily Snapshot", layout="wide")

# -------------------------------------------------------------------
# Data locations (GitHub raw CSVs)
# -------------------------------------------------------------------
URL_ETF = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv"
URL_RS  = "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv"

# -------------------------------------------------------------------
# Data loading (cached)
# -------------------------------------------------------------------
@st.cache_data(ttl=900)
def load_data():
    df_etf = pd.read_csv(URL_ETF)
    df_rs  = pd.read_csv(URL_RS)
    # dtypes
    df_etf["date"] = pd.to_datetime(df_etf["date"], errors="coerce")
    df_rs["date"]  = pd.to_datetime(df_rs["date"], errors="coerce")
    # normalize group/ticker text (defensive)
    if "group" in df_etf.columns:
        df_etf["group"] = df_etf["group"].fillna("Ungrouped")
    df_etf["ticker"] = df_etf["ticker"].astype(str)
    df_rs["ticker"]  = df_rs["ticker"].astype(str)
    return df_etf, df_rs

df_etf, df_rs = load_data()

# -------------------------------------------------------------------
# Prep: latest snapshot & sparkline slice (fixed 21 days)
# -------------------------------------------------------------------
LOOKBACK = 21  # fixed (last 21 trading days for sparkline)

# latest row per ticker for table metrics
latest = (
    df_etf.sort_values("date")
          .groupby("ticker", as_index=False)
          .tail(1)
          .set_index("ticker")
)

# sparkline data: last 21 per ticker, global absolute Y scale
df_rs = df_rs.sort_values(["ticker", "date"])
rs_lastN = df_rs.groupby("ticker").tail(LOOKBACK)

if not rs_lastN.empty:
    RS_MIN = float(rs_lastN["rs_to_spy"].min())
    RS_MAX = float(rs_lastN["rs_to_spy"].max())
else:
    RS_MIN, RS_MAX = 0.9, 1.1

# -------------------------------------------------------------------
# Helpers: sparkline + styling
# -------------------------------------------------------------------
def sparkbar_img(series_vals, width=120, height=36, y_domain=None):
    """
    Return a base64 <img> tag for a mini bar sparkline.
    Tries Altair PNG export (requires vl-convert-python). If that fails,
    falls back to Matplotlib (no extra backend needed).
    """
    if not isinstance(series_vals, (list, tuple)) or len(series_vals) == 0:
        return ""

    # ---------- Try Altair first ----------
    try:
        import altair as alt
        alt.renderers.set_embed_options(actions=False)

        df_tmp = pd.DataFrame({"x": range(len(series_vals)), "y": series_vals})
        df_tmp["is_current"] = df_tmp["x"] == df_tmp["x"].max()

        y_scale = alt.Scale(zero=False)
        if y_domain and len(y_domain) == 2:
            y_scale = alt.Scale(zero=False, domain=list(y_domain))

        chart = (
            alt.Chart(df_tmp)
            .mark_bar()
            .encode(
                x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, max(0, len(series_vals)-1)])),
                y=alt.Y("y:Q", axis=None, scale=y_scale),
                color=alt.condition(alt.datum.is_current, alt.value("darkgreen"), alt.value("lightgreen")),
                tooltip=[alt.Tooltip("y:Q", title="RS to SPY", format=".4f")],
            )
            .properties(width=width, height=height)
        )

        # Altair v5 + vl-convert backend; may raise if backend missing
        png_bytes = chart.save(format="png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return f'<img src="data:image/png;base64,{b64}" alt="sparkline" />'

    except Exception:
        # ---------- Fallback: Matplotlib ----------
        fig = plt.figure(figsize=(width/96, height/96), dpi=96)
        ax = fig.add_axes([0, 0, 1, 1])

        colors = ["lightgreen"] * len(series_vals)
        colors[-1] = "darkgreen"

        ax.bar(range(len(series_vals)), series_vals, color=colors)
        ax.axis("off")
        if y_domain and len(y_domain) == 2:
            ax.set_ylim(y_domain[0], y_domain[1])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
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
    """Gradient for returns (fraction): red (neg) -> white -> green (pos)."""
    try:
        v = float(val)
    except Exception:
        return ""
    cap = 0.20  # clamp at +/-20% for color intensity
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

# -------------------------------------------------------------------
# Page header
# -------------------------------------------------------------------
st.title("US Market Daily Snapshot")
if not latest.empty:
    st.caption(f"Latest data date: {latest['date'].max().date()} • Sparkline lookback fixed at {LOOKBACK} trading days")

# -------------------------------------------------------------------
# Render by group (all groups, no controls)
# -------------------------------------------------------------------
if "group" in latest.columns:
    group_iter = latest.groupby("group").groups.items()
else:
    group_iter = [("All", latest.index)]

for group_name, tickers in group_iter:
    st.header(f"📌 {group_name}")

    rows = []
    for ticker in tickers:
        row = latest.loc[ticker]

        # RS sparkline series for this ticker (last 21)
        series = rs_lastN.loc[rs_lastN["ticker"] == ticker, "rs_to_spy"].tolist()
        spark_img = sparkbar_img(series, y_domain=(RS_MIN, RS_MAX)) if series else ""

        record = {
            "Ticker": ticker,
            "RS Sparkline": spark_img,
            "RS Rank (21D)": row.get("rs_rank_21d", None),
            "RS Rank (252D)": row.get("rs_rank_252d", None),
            "": "",  # spacer
            "Volume Alert": row.get("volume_alert", "-"),
            "  ": "",  # spacer
            # performance (fractions -> format after)
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
        }
        rows.append(record)

    disp = pd.DataFrame(rows)

    # format performance columns as % strings but keep numeric copy for color shading
    perf_cols = ["1D","1W","1M","3M","6M","1Y","YTD"]
    numeric_for_style = disp[perf_cols].copy()
    for c in perf_cols:
        disp[c] = numeric_for_style[c].map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "")

    # format RS ranks as integer percentages (no decimals)
    for col in ["RS Rank (21D)", "RS Rank (252D)"]:
        if col in disp.columns:
            disp[col] = disp[col].map(lambda x: f"{int(round(x))}%" if pd.notnull(x) else "")

    styler = (
        disp.style
        .hide(axis="index")
        .format(na_rep="", subset=disp.columns)
        .applymap(highlight_rs_cell, subset=rs_cols)
        .applymap(highlight_volume_cell, subset=["Volume Alert"])
        .apply(lambda s: [shade_performance(v) for v in numeric_for_style[s.name]], subset=perf_cols, axis=0)
        .applymap(color_sma_text, subset=["SMA5","SMA10","SMA20","SMA50","SMA100"])
    )

    # Render as HTML (keeps <img> for sparkline)
    st.write(styler.to_html(escape=False), unsafe_allow_html=True)