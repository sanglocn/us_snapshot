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

# global y-scale for comparable sparklines
RS_MIN = float(rs_lastN["rs_to_spy"].min()) if not rs_lastN.empty else 0.9
RS_MAX = float(rs_lastN["rs_to_spy"].max()) if not rs_lastN.empty else 1.1

# --- Sparkline helper: Altair line (fallback to Matplotlib line) ---
def sparkline_img(series_vals, width=120, height=36, y_domain=None):
    if not series_vals:
        return ""
    # Try Altair first
    try:
        import altair as alt
        import pandas as pd
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

    except Exception:
        # Matplotlib fallback (line)
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

# --- Page ---
st.title("US Market Daily Snapshot")
st.caption(f"Latest data date: {latest['date'].max().date()}")

# group safely (fallback to "All" if missing)
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
            "RS Rank (21D)": f"{int(round(row['rs_rank_21d']))}%" if pd.notnull(row['rs_rank_21d']) else "",
            "RS Rank (252D)": f"{int(round(row['rs_rank_252d']))}%" if pd.notnull(row['rs_rank_252d']) else "",
            "Volume Alert": row.get("volume_alert", "-"),

            " ": "",  # spacer between ranks and performance

            # Performance already in %, so no *100
            "1D": f"{row['ret_1d']:.1f}%" if pd.notnull(row['ret_1d']) else "",
            "1W": f"{row['ret_1w']:.1f}%" if pd.notnull(row['ret_1w']) else "",
            "1M": f"{row['ret_1m']:.1f}%" if pd.notnull(row['ret_1m']) else "",

            "  ": "",  # spacer between performance and SMA

            "SMA5":  "✅" if row.get("above_sma5")  else "❌",
            "SMA10": "✅" if row.get("above_sma10") else "❌",
            "SMA20": "✅" if row.get("above_sma20") else "❌",
        })
    disp = pd.DataFrame(rows)
    st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)
