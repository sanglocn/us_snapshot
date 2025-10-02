import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import io
import base64
import re
from typing import List, Dict, Tuple

# ---------------------------------
# Configuration
# ---------------------------------
st.set_page_config(page_title="US Market Snapshot", layout="wide")

# Constants
DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv",
    "holdings": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_holdings.csv",
    "chart": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_chart.csv",
}
LOOKBACK_DAYS = 21
GROUP_ORDER = ["Market","Sector","Commodity","Crypto","Country","Theme","Leader"]

# Palette per group (base, darker)
GROUP_PALETTE = {
    "Market":   ("#0284c7", "#075985"),  # cyan
    "Sector":   ("#16a34a", "#166534"),  # green
    "Commodity":("#b45309", "#7c2d12"),  # amber/brown
    "Crypto":   ("#7c3aed", "#4c1d95"),  # purple
    "Country":  ("#ea580c", "#9a3412"),  # orange
    "Theme":    ("#2563eb", "#1e40af"),  # blue
    "Leader":   ("#db2777", "#9d174d"),  # pink
}

# --- settings (no sidebar) ---
use_group_colors = True        # color-code ticker chips by group
max_holdings_rows = 10         # rows shown in tooltip table

# ---------------------------------
# Small helpers
# ---------------------------------
def _clean_text_series(s: pd.Series, title_case: bool = False) -> pd.Series:
    s = (
        s.astype(str)
         .str.replace(r"[\r\n\t]+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )
    if title_case:
        s = s.str.title()
    return s

def _fix_acronyms_in_name(s: pd.Series) -> pd.Series:
    """Preserve CSV case but normalize common acronyms (ETF, USD, USA, REIT, AI, S&P, US)."""
    s = s.astype(str)
    s = s.str.replace(r"\betf\b", "ETF", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\busd\b", "USD", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\busa\b", "USA", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\breit\b", "REIT", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\bai\b", "AI", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\bus\b", "US", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\bS\s*(?:&|and|-)\s*P\b", "S&P", regex=True, flags=re.IGNORECASE)
    return s

def _clean_ticker_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[\r\n\t\s]+", "", regex=True)
         .str.upper()
         .str.strip()
    )

def _escape(s) -> str:
    import html
    return html.escape("" if pd.isna(s) else str(s))

# ---------------------------------
# Data Loading
# ---------------------------------
@st.cache_data(ttl=900)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs = pd.read_csv(DATA_URLS["rs"])

    if "date" not in df_etf.columns or "ticker" not in df_etf.columns:
        raise ValueError("ETF price CSV must include 'date' and 'ticker'.")
    if "date" not in df_rs.columns or "ticker" not in df_rs.columns or "rs_to_spy" not in df_rs.columns:
        raise ValueError("RS CSV must include 'date', 'ticker', 'rs_to_spy'.")

    df_etf["date"] = pd.to_datetime(df_etf["date"], errors="coerce")
    df_rs["date"] = pd.to_datetime(df_rs["date"], errors="coerce")

    if "ticker" in df_etf.columns:
        df_etf["ticker"] = _clean_ticker_series(df_etf["ticker"])
    if "group" in df_etf.columns:
        df_etf["group"] = _clean_text_series(df_etf["group"])
    if "ticker" in df_rs.columns:
        df_rs["ticker"] = _clean_ticker_series(df_rs["ticker"])
    if "group" in df_rs.columns:
        df_rs["group"] = _clean_text_series(df_rs["group"])
    
    return df_etf, df_rs

@st.cache_data(ttl=900, show_spinner=False)
def load_holdings_csv(url: str = DATA_URLS["holdings"]) -> pd.DataFrame:
    """
    Expected columns:
      fund_ticker, fund_name, security_name, security_ticker, security_weight, ingest_date
    """
    df = pd.read_csv(url)
    need = {"fund_ticker","fund_name","security_name","security_ticker","security_weight","ingest_date"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[holdings] Missing columns: {sorted(missing)}")

    df["ingest_date"] = pd.to_datetime(df["ingest_date"], errors="coerce")
    df["security_weight"] = pd.to_numeric(df["security_weight"], errors="coerce")

    df["fund_ticker"]     = _clean_ticker_series(df["fund_ticker"])
    # Preserve CSV casing + fix acronyms; do NOT title-case fund_name
    df["fund_name"]       = _fix_acronyms_in_name(_clean_text_series(df["fund_name"], title_case=False))
    # Security name can be title-cased but also fix acronyms
    df["security_name"]   = _fix_acronyms_in_name(_clean_text_series(df["security_name"], title_case=True))
    df["security_ticker"] = _clean_ticker_series(df["security_ticker"])
    return df

@st.cache_data(ttl=900, show_spinner=False)
def load_chart_csv(url: str = DATA_URLS["chart"]) -> pd.DataFrame:
    """
    Expected columns:
      ticker, group, date, adj_open, adj_close, adj_volume, adj_high, adj_low, sma5, sma10, sma20, sma50
    """
    df = pd.read_csv(url)
    need = {
        "ticker","group","date","adj_open","adj_close","adj_volume","adj_high","adj_low",
        "sma5","sma10","sma20","sma50"
    }
    missing = need - set(df.columns)
    # Soft requirement for SMA cols: allow chart without them
    soft_missing = {"sma5","sma10","sma20","sma50"} - set(df.columns)
    if missing - {"sma5","sma10","sma20","sma50"}:
        raise ValueError(f"[chart] Missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = _clean_ticker_series(df["ticker"])
    df["group"] = _clean_text_series(df["group"])
    # Coerce numerics
    for c in ["adj_open","adj_close","adj_high","adj_low","adj_volume","sma5","sma10","sma20","sma50"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.attrs["sma_missing"] = sorted(list(soft_missing))
    return df

# ---------------------------------
# Data Processing
# ---------------------------------
def process_data(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker","date"]).groupby("ticker").tail(LOOKBACK_DAYS)
    return latest, rs_last_n

def compute_threshold_counts(df_etf: pd.DataFrame) -> pd.DataFrame:
    if "count_over_85" not in df_etf.columns or "count_under_50" not in df_etf.columns:
        return pd.DataFrame(columns=["date","count_over_85","count_under_50","date_str"])
    daily = df_etf[["date","count_over_85","count_under_50"]].drop_duplicates().sort_values("date")
    daily = daily.dropna(subset=["count_over_85","count_under_50"])
    last_21_dates = daily["date"].tail(21)
    daily_21 = daily[daily["date"].isin(last_21_dates)].copy().sort_values("date")
    daily_21["date_str"] = daily_21["date"].dt.strftime("%Y-%m-%d")
    return daily_21

# ---------------------------------
# Chips + Tooltip (HTML/CSS)
# ---------------------------------
def make_tooltip_card_for_ticker(holdings_df: pd.DataFrame, ticker: str, max_rows: int) -> str:
    sub = holdings_df[holdings_df["fund_ticker"] == ticker]
    if sub.empty:
        return ""
    last_date = sub["ingest_date"].max()
    if pd.notna(last_date):
        sub = sub[sub["ingest_date"] == last_date]

    fund_name = _escape(sub["fund_name"].iloc[0])  # uses fixed casing (ETF not Etf)
    last_update_str = _escape(last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "N/A")  # Changed to show only date

    topn = (
        sub[["security_name","security_ticker","security_weight"]]
        .dropna(subset=["security_name"])
        .sort_values("security_weight", ascending=False)
        .head(max_rows)
    )

    rows = []
    for _, r in topn.iterrows():
        sec = _escape(r["security_name"])
        tk  = _escape(r.get("security_ticker", ""))
        wt  = "" if pd.isna(r["security_weight"]) else f"{float(r['security_weight']):.2f}%"
        rows.append(
            "<tr>"
            f"<td class='tt-sec' title='{sec}'>{sec}</td>"
            f"<td class='tt-tk' title='{tk}'>{tk}</td>"
            f"<td class='tt-wt' title='{wt}'>{wt}</td>"
            "</tr>"
        )

    table_html = (
        "<table class='tt-table'>"
        "<thead><tr><th>Security</th><th>Ticker</th><th>Weight</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return (
        f'<div class="tt-card"><div class="tt-title">{fund_name}</div>'
        f'<div class="tt-sub">Last update: <span class="tt-date">{last_update_str}</span></div>'
        f'{table_html}</div>'
    )

def make_ticker_chip_with_tooltip(ticker: str, card_html: str, group_name: str | None) -> str:
    t = _escape(ticker)
    group_class = ""
    if use_group_colors and group_name:
        group_slug = re.sub(r'[^a-z0-9]+', '-', group_name.lower()).strip('-')
        group_class = f" chip--{group_slug}"
    return f'<span class="tt-chip{group_class}">{t}{card_html}</span>'

def build_chip_css() -> str:
    base_css = """
/* Chip base */
.tt-chip {
  position: relative;
  display: inline-block;
  padding: 6px 12px;
  margin: 2px;
  border-radius: 9999px;
  background: linear-gradient(135deg, #2563eb22, #07598522);
  border: 1px solid #2563eb55;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;
  font-size: 13px;
  font-weight: 700;
  color: #1e3a8a;
  cursor: default;
  white-space: nowrap;
  transition: all .2s ease-in-out;
  box-shadow: 0 1px 2px rgba(0,0,0,.05);
}
.tt-chip:hover {
  background: linear-gradient(135deg, #2563eb33, #1d4ed833);
  border-color: #2563eb88;
  color: #1e40af;
  box-shadow: 0 2px 8px rgba(37,99,235,.28);
}

/* Tooltip card to the RIGHT; scroll if tall */
.tt-chip .tt-card {
  position: absolute;
  left: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%) translateX(6px);
  z-index: 999999;
  width: min(520px, 90vw);
  max-height: 60vh;
  overflow: auto;
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 12px 28px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.06);
  border-radius: 12px;
  padding: 12px 12px 8px 12px;
  visibility: hidden;
  opacity: 0;
  transition: opacity .18s ease, transform .18s ease;
  pointer-events: none;
}
.tt-chip:hover .tt-card {
  visibility: visible;
  opacity: 1;
  transform: translateY(-50%) translateX(0);
}

/* Card text */
.tt-card .tt-title { font-weight: 700; font-size: 14px; margin-bottom: 4px; }
.tt-card .tt-sub   { color: #667085; font-size: 12px; margin-bottom: 8px; }
.tt-card .tt-date  { font-variant-numeric: tabular-nums; }

/* Tooltip table */
.tt-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  table-layout: auto;       /* let browser auto-size columns based on content */
}
.tt-table thead th {
  text-align: left;
  padding: 6px 6px;
  border-bottom: 1px solid #eee;
}
.tt-table tbody td {
  padding: 6px 6px;
  border-bottom: 1px dashed #f0f0f0;
  vertical-align: top;
  word-break: break-word;   /* wrap long security names nicely */
}
.tt-table tbody tr:last-child td { border-bottom: none; }

/* Column behaviors:
   - Security flexes and wraps
   - Ticker and Weight stay compact (width:1% trick) and don't wrap
*/
.tt-sec { width: auto; }
.tt-tk  {
  width: 1%;
  white-space: nowrap;
  font-family: ui-monospace, Menlo, Consolas, monospace;
}
.tt-wt  {
  width: 1%;
  white-space: nowrap;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* Mobile fallback: show above chip */
@media (max-width: 768px) {
  .tt-chip .tt-card {
    left: 0;
    top: auto;
    bottom: calc(100% + 8px);
    transform: translateY(6px);
    max-height: 50vh;
  }
  .tt-chip:hover .tt-card {
    visibility: visible;
    transform: translateY(0);
  }
}
"""
    group_css_parts = []
    if use_group_colors:
        for g, (base, dark) in GROUP_PALETTE.items():
            slug = re.sub(r'[^a-z0-9]+', '-', g.lower()).strip('-')
            group_css_parts.append(f"""
.tt-chip.chip--{slug} {{
  background: linear-gradient(135deg, {base}22, {dark}22);
  border-color: {base}55;
  color: {dark};
}}
.tt-chip.chip--{slug}:hover {{
  background: linear-gradient(135deg, {base}33, {dark}33);
  border-color: {base}88;
  color: {dark};
  box-shadow: 0 2px 8px {base}55;
}}
""")
    return "<style>" + base_css + "\n".join(group_css_parts) + "</style>"

# ---------------------------------
# Visualization Helpers
# ---------------------------------
def create_sparkline(values: List[float], width: int = 155, height: int = 36) -> str:
    if not values:
        return ""
    fig = plt.figure(figsize=(width / 96, height / 96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(range(len(values)), values, linewidth=1.5, color="green")
    ax.plot(len(values) - 1, values[-1], "o", color="darkgreen", markersize=4)
    ax.axis("off")
    y_min, y_max = min(values), max(values)
    padding = (y_max - y_min) * 0.05 or 0.01
    ax.set_ylim(y_min - padding, y_max + padding)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}" alt="sparkline" />'

def breadth_column_chart(df: pd.DataFrame, value_col: str, bar_color: str = "steelblue"):
    """
    Create an Altair bar chart that uses the datetime 'date' as the x-axis (temporal).
    df must contain 'date' as pd.Timestamp; returns Altair Chart.
    """
    if df.empty:
        # return an empty chart with correct schema
        return (
            alt.Chart(pd.DataFrame({"date": [], value_col: []}))
            .mark_bar()
            .encode(
                x=alt.X("date:T", axis=alt.Axis(title=None)),
                y=alt.Y(f"{value_col}:Q", axis=alt.Axis(title=None)),
            )
        )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("date:T",
                    axis=alt.Axis(title=None, format="%Y-%m-%d", labelAngle=-45)),
            y=alt.Y(f"{value_col}:Q", axis=alt.Axis(title=None)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip(f"{value_col}:Q", title=value_col),
            ],
            color=alt.value(bar_color),
        )
        .properties(height=240)
    )

    return chart

def format_chart_link(ticker: str) -> str:
    """Returns an HTML link with a chart emoji that sets ?chart=<ticker> in URL."""
    t = _escape(ticker)
    return (
        f'<a href="?chart={t}" target="_self" '
        f'style="text-decoration:none; display:block; text-align:center; font-size:18px;" '
        f'title="Open chart for {t}">📈</a>'
    )

# ---------------------------------
# Formatting Helpers
# ---------------------------------
def format_rank(value: float) -> str:
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    pct = int(round(value * 100))
    if pct >= 85:
        bg, border = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)"
    elif pct < 50:
        bg, border = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"
    else:
        bg, border = "rgba(156,163,175,.25)", "rgba(156,163,175,.35)"
    return (f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
            f'background-color:{bg}; border:1px solid {border}; color:inherit;">{pct}%</span>')

def format_performance(value: float) -> str:
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    return f'<span style="display:block; text-align:right;">{value:.1f}%</span>'

def format_performance_intraday(value: float) -> str:
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    pct_text = f"{value:.1f}%"
    if value > 0:
        bg, border = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)"
    elif value < 0:
        bg, border = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"
    else:
        bg, border = "rgba(156,163,175,.25)", "rgba(156,163,175,.35)"
    return (f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
            f'background-color:{bg}; border:1px solid {border}; color:inherit;">{pct_text}</span>')

def format_indicator(value: str) -> str:
    value = str(value).strip().lower()
    if value == "yes": return '<span style="color:green; display:block; text-align:center;">✅</span>'
    if value == "no":  return '<span style="color:red; display:block; text-align:center;">❌</span>'
    return '<span style="display:block; text-align:center;">-</span>'

def format_volume_alert(value: str, rs_rank_252d) -> str:
    if not isinstance(value, str):
        return '<span style="display:block; text-align:center;">-</span>'
    val = value.strip().lower()
    try:
        rs_val = float(rs_rank_252d)
    except (ValueError, TypeError):
        rs_val = None
    if val == "positive" and rs_val is not None and rs_val >= 0.80:
        return '<span style="display:block; text-align:center; font-size:16px;">💎</span>'
    elif val == "positive":
        return '<span style="display:block; text-align:center; font-size:16px;">🟩</span>'
    elif val == "negative":
        return '<span style="display:block; text-align:center; font-size:16px;">🟥</span>'
    else:
        return '<span style="display:block; text-align:center;">-</span>'

def format_multiple(value) -> str:
    """Render numeric multiple like 1.75 with a subtle badge."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '<span style="display:block; text-align:right;">-</span>'
    txt = f"{v:.2f}"
    if v >= 10:
        bg, border = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"   # red
    elif v >= 4:
        bg, border = "rgba(234,179,8,.22)", "rgba(234,179,8,.35)"   # yellow/amber
    elif v > 0:
        bg, border = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)" # green
    else:
        bg, border = "rgba(156,163,175,.18)", "rgba(156,163,175,.30)" # grey
    return (f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
            f'background-color:{bg}; border:1px solid {border}; color:inherit;">{txt}</span>')

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')

# ---------------------------------
# Plotly Chart Builder
# ---------------------------------
# --- ADDED ---
def make_ticker_figure(df_chart: pd.DataFrame, ticker: str, max_bars: int = 180) -> go.Figure:
    sub = df_chart[df_chart["ticker"] == ticker].sort_values("date")
    if sub.empty:
        raise ValueError(f"No chart data for {ticker}.")

    # Keep valid sessions only
    sub = sub[
        sub["adj_open"].notna() &
        sub["adj_high"].notna() &
        sub["adj_low"].notna() &
        sub["adj_close"].notna()
    ].copy()

    if len(sub) > max_bars:
        sub = sub.tail(max_bars)

    sub = sub.reset_index(drop=True)
    date_str = sub["date"].dt.strftime("%Y-%m-%d")

    # --- detect market holidays (weekdays with no trading) ---
    trading_days = pd.to_datetime(sub["date"].dt.normalize().unique())
    all_weekdays = pd.bdate_range(
        start=sub["date"].min().normalize(),
        end=sub["date"].max().normalize(),
        freq="B"
    )
    closed_days = sorted(set(all_weekdays) - set(trading_days))
    closed_days_str = [d.strftime("%Y-%m-%d") for d in closed_days]

    # ----- traces: use real dates on x -----
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.72, 0.28]
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=sub["date"],
            open=sub["adj_open"], high=sub["adj_high"],
            low=sub["adj_low"],  close=sub["adj_close"],
            name="Price",
            hovertext=[
                f"Date: {d}<br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
                for d, o, h, l, c in zip(date_str, sub["adj_open"], sub["adj_high"], sub["adj_low"], sub["adj_close"])
            ],
            hoverinfo="text"
        ),
        row=1, col=1
    )

    # SMAs
    for sma_col, name in [("sma5","SMA 5"), ("sma10","SMA 10"), ("sma20","SMA 20"), ("sma50","SMA 50")]:
        if sma_col in sub.columns and sub[sma_col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=sub["date"], y=sub[sma_col],
                    mode="lines", name=name, line=dict(width=1.2),
                    hovertext=[f"Date: {d}<br>{name}: {y:.2f}" for d, y in zip(date_str, sub[sma_col])],
                    hoverinfo="text"
                ),
                row=1, col=1
            )

    # Volume
    fig.add_trace(
        go.Bar(
            x=sub["date"], y=sub["adj_volume"], name="Volume", opacity=0.9,
            hovertext=[f"Date: {d}<br>Volume: {int(v):,}" for d, v in zip(date_str, sub["adj_volume"].fillna(0))],
            hoverinfo="text"
        ),
        row=2, col=1
    )

    # ----- layout -----
    fig.update_layout(
        autosize=True,
        height=600,
        margin=dict(l=20, r=20, t=50, b=90),
        title=dict(text=f"{ticker} — Candlestick with SMA & Volume", x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        bargap=0.3
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # ----- AUTO monthly ticks + hide weekends + holidays -----
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b-%Y",
        ticklabelmode="period",
        rangebreaks=[
            dict(bounds=["sat", "mon"]),    # weekends
            dict(values=closed_days_str),   # holidays
        ],
        showgrid=True,
        row=1, col=1
    )
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b-%Y",
        ticklabelmode="period",
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=closed_days_str),
        ],
        showgrid=True,
        row=2, col=1
    )

    return fig

# --- ADDED ---
def open_chart_ui(ticker: str, df_chart: pd.DataFrame):
    """Opens a modal if available; falls back to sidebar."""
    try:
        fig = make_ticker_figure(df_chart, ticker)
    except Exception as e:
        if hasattr(st, "dialog"):
            @st.dialog(f"Chart — {ticker}")
            def _dlg():
                st.error(str(e))
            _dlg()
        else:
            with st.sidebar:
                st.header(f"Chart — {ticker}")
                st.error(str(e))
        return

    if hasattr(st, "dialog"):
        @st.dialog(f"Chart — {ticker}")
        def _dlg():
            # Soft notice if SMA columns were missing
            sma_missing = df_chart.attrs.get("sma_missing", [])
            if sma_missing:
                st.caption(f"Note: Missing SMA columns in source: {', '.join(sma_missing)}")
            st.plotly_chart(fig, use_container_width=True)
        _dlg()
    else:
        with st.sidebar:
            sma_missing = df_chart.attrs.get("sma_missing", [])
            st.header(f"Chart — {ticker}")
            if sma_missing:
                st.caption(f"Note: Missing SMA columns in source: {', '.join(sma_missing)}")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Table Rendering
# ---------------------------------
def render_group_table(group_name: str, rows: List[Dict]) -> None:
    table_id = f"tbl-{slugify(group_name)}"
    html = pd.DataFrame(rows).to_html(escape=False, index=False)

    css = f"""
        #{table_id} table {{
            width: 100%;
            border-collapse: collapse;
            border-spacing: 0;
            border: none;
            border-radius: 8px;
        }}
        #{table_id} table thead th {{
            text-align: center !important;
            border-bottom: 2px solid rgba(156, 163, 175, 0.6);
            border-left: none !important;
            border-right: none !important;
            padding: 6px 8px;
        }}
        #{table_id} table tbody td {{
            border-bottom: none;
            border-left: none !important;
            border-right: none !important;
            padding: 6px 8px;
            position: relative;
        }}
        #{table_id} table tbody tr:last-child td {{ border-bottom: none; }}
        /* Right align numeric-ish columns */
        #{table_id} table td:nth-child(3),
        #{table_id} table td:nth-child(4),
        #{table_id} table td:nth-child(7),
        #{table_id} table td:nth-child(8),
        #{table_id} table td:nth-child(9),
        #{table_id} table td:nth-child(10) {{ text-align: right !important; }}
        #{table_id} table td:nth-child(5),
        #{table_id} table td:nth-child(11),
        #{table_id} table td:nth-child(12),
        #{table_id} table td:nth-child(13),
        #{table_id} table td:nth-child(14) {{ text-align: center !important; }}
        /* Keep Ticker column tight and on one line */
        #{table_id} table td:nth-child(1) {{ white-space: nowrap; line-height: 1.25; }}
    """
    st.markdown(f'<div id="{table_id}"><style>{css}</style>{html}</div>', unsafe_allow_html=True)

# ---------------------------------
# Dashboard Rendering
# ---------------------------------
def render_dashboard(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> None:
    st.title("US Market Daily Snapshot")

    # Inject CSS for chips/tooltips
    st.markdown(build_chip_css(), unsafe_allow_html=True)

    latest, rs_last_n = process_data(df_etf, df_rs)
    if "date" in latest.columns:
        st.caption(f"Latest Update: {pd.to_datetime(latest['date']).max().date()}")
    else:
        st.caption("Latest Update: N/A")

    try:
        df_holdings = load_holdings_csv(DATA_URLS["holdings"])
    except Exception as e:
        df_holdings = pd.DataFrame()
        st.warning(f"Holdings tooltips disabled — {e}")

    # --- ADDED ---
    # Load chart data once; used by modal/sidebar when user clicks 📈
    try:
        df_chart = load_chart_csv(DATA_URLS["chart"])
    except Exception as e:
        df_chart = pd.DataFrame()
        st.warning(f"Chart data unavailable — {e}")

    if "group" not in latest.columns:
        st.error("Column 'group' is missing in ETF dataset — cannot render grouped tables.")
        return

    # --- ADDED ---
    # Detect URL param ?chart=TICKER (works with our link-based 📈 cells)
    qp = st.query_params if hasattr(st, "query_params") else {}
    selected_chart_ticker = None
    if qp:
        # Streamlit may return str or list; handle both
        val = qp.get("chart", None)
        if isinstance(val, list):
            selected_chart_ticker = (val[0] if val else None)
        else:
            selected_chart_ticker = val
        if selected_chart_ticker:
            selected_chart_ticker = str(selected_chart_ticker).upper().strip()

    if "group" not in latest.columns:
        st.error("Column 'group' is missing in ETF dataset — cannot render grouped tables.")
        return

    group_tickers = latest.groupby("group").groups
    for group_name in GROUP_ORDER:
        if group_name not in group_tickers:
            continue
        st.header(f"📌 {group_name}")
        tickers_in_group = group_tickers[group_name]

        if "rs_rank_21d" in latest.columns:
            tickers_in_group = sorted(
                tickers_in_group,
                key=lambda t: latest.loc[t, "rs_rank_21d"] if not pd.isna(latest.loc[t, "rs_rank_21d"]) else -1,
                reverse=True,
            )

        rows = []
        for ticker in tickers_in_group:
            row = latest.loc[ticker]
            spark_series = rs_last_n.loc[rs_last_n["ticker"] == ticker, "rs_to_spy"].tolist()

            chip = ticker
            if not df_holdings.empty:
                card_html = make_tooltip_card_for_ticker(df_holdings, ticker, max_rows=max_holdings_rows)
                if card_html:
                    chip = make_ticker_chip_with_tooltip(ticker, card_html, group_name)

            rows.append({
                "Ticker": chip,
                "Relative Strength": create_sparkline(spark_series),
                "RS Rank (1M)": format_rank(row.get("rs_rank_21d")),
                "RS Rank (1Y)": format_rank(row.get("rs_rank_252d")),
                "Volume Alert": format_volume_alert(row.get("volume_alert", "-"), row.get("rs_rank_252d")),
                " ": "",
                "Intraday": format_performance_intraday(row.get("ret_intraday")),
                "1D Return": format_performance(row.get("ret_1d")),
                "1W Return": format_performance(row.get("ret_1w")),
                "1M Return": format_performance(row.get("ret_1m")),
                "  ": "",
                "Extension Multiple": format_multiple(row.get("ratio_pct_dist_to_atr_pct")),
                "Above SMA5": format_indicator(row.get("above_sma5")),
                "Above SMA10": format_indicator(row.get("above_sma10")),
                "Above SMA20": format_indicator(row.get("above_sma20")),
                "  ": "",
                "Chart": format_chart_link(ticker),
            })

        render_group_table(group_name, rows)

    # Breadth charts
    counts_21 = compute_threshold_counts(df_etf)
    if not counts_21.empty:
        start_date = counts_21["date"].min().date()
        end_date = counts_21["date"].max().date()
        
        st.subheader("Breadth Gauge")
        st.caption("Green = No. of tickers gaining momentum · Red = No. of tickers losing momentum")
        st.caption(f"From {start_date} to {end_date}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(breadth_column_chart(counts_21, "count_over_85", bar_color="green"),
                            use_container_width=True)
        with c2:
            st.altair_chart(breadth_column_chart(counts_21, "count_under_50", bar_color="red"),
                            use_container_width=True)
    else:
        st.info("`count_over_85` and `count_under_50` not found in ETF data — breadth charts skipped.")

    # --- ADDED ---
    # Open chart UI (modal or sidebar) if the user clicked a 📈 cell
    if selected_chart_ticker:
        if df_chart.empty:
            st.warning("Chart data not available.")
        else:
            open_chart_ui(selected_chart_ticker, df_chart)

# ---------------------------------
# Main
# ---------------------------------
try:
    df_etf, df_rs = load_data()
except Exception as e:
    st.error(f"Failed to load price/RS data — {e}")
else:
    try:
        render_dashboard(df_etf, df_rs)
    except Exception as e:
        st.exception(e)
