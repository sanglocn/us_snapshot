import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
}
LOOKBACK_DAYS = 21
GROUP_ORDER = [
    "Market",
    "Sector",
    "Commodity",
    "Crypto",
    "Country",
    "Theme",
    "Leader",
]

# ---------------------------------
# Small helpers
# ---------------------------------
def _clean_text_series(s: pd.Series, title_case: bool = False) -> pd.Series:
    """Remove newlines/tabs, collapse spaces, strip. Optionally title-case."""
    s = (
        s.astype(str)
         .str.replace(r"[\r\n\t]+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )
    if title_case:
        s = s.str.title()
    return s

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

    # Basic schema checks
    if "date" not in df_etf.columns or "ticker" not in df_etf.columns:
        raise ValueError("ETF price CSV must include columns: 'date', 'ticker'.")
    if "date" not in df_rs.columns or "ticker" not in df_rs.columns or "rs_to_spy" not in df_rs.columns:
        raise ValueError("RS sparkline CSV must include columns: 'date', 'ticker', 'rs_to_spy'.")

    # Types
    df_etf["date"] = pd.to_datetime(df_etf["date"], errors="coerce")
    df_rs["date"] = pd.to_datetime(df_rs["date"], errors="coerce")

    # Clean textual fields to avoid \n artifacts
    for col in [c for c in ["ticker", "group"] if c in df_etf.columns]:
        df_etf[col] = _clean_text_series(df_etf[col], title_case=False)
    for col in [c for c in ["ticker"] if c in df_rs.columns]:
        df_rs[col] = _clean_text_series(df_rs[col], title_case=False)

    return df_etf, df_rs

@st.cache_data(ttl=900, show_spinner=False)
def load_holdings_csv(url: str = DATA_URLS["holdings"]) -> pd.DataFrame:
    """
    Expected columns: fund_ticker, fund_name, security_name, security_weight, ingest_date
    """
    df = pd.read_csv(url)
    need = {"fund_ticker", "fund_name", "security_name", "security_weight", "ingest_date"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[holdings] Missing columns: {sorted(missing)}")

    # Types
    df["ingest_date"] = pd.to_datetime(df["ingest_date"], errors="coerce")
    df["security_weight"] = pd.to_numeric(df["security_weight"], errors="coerce")

    # Clean text fields (remove \n / multiple spaces). Title-case names.
    df["fund_ticker"]   = _clean_text_series(df["fund_ticker"], title_case=False)
    df["fund_name"]     = _clean_text_series(df["fund_name"], title_case=True)
    df["security_name"] = _clean_text_series(df["security_name"], title_case=True)
    return df

# ---------------------------------
# Data Processing
# ---------------------------------
def process_data(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK_DAYS)
    return latest, rs_last_n

def compute_threshold_counts(df_etf: pd.DataFrame) -> pd.DataFrame:
    if "rs_rank_21d" not in df_etf.columns:
        return pd.DataFrame(columns=["date", "count_over_85", "count_under_50", "date_str"])
    tmp = df_etf.copy()
    tmp["over_85"] = tmp["rs_rank_21d"] >= 0.85
    tmp["under_50"] = tmp["rs_rank_21d"] < 0.50
    daily = (
        tmp.groupby("date", as_index=False)
           .agg(count_over_85=("over_85", "sum"), count_under_50=("under_50", "sum"))
           .sort_values("date")
    )
    daily = daily.dropna(subset=["count_over_85", "count_under_50"])
    last_21_dates = daily["date"].drop_duplicates().sort_values().tail(21)
    daily_21 = daily[daily["date"].isin(last_21_dates)].copy().sort_values("date")
    daily_21["date_str"] = daily_21["date"].dt.strftime("%Y-%m-%d")
    return daily_21

# ---------------------------------
# Tooltip (HTML/CSS)
# ---------------------------------
def make_tooltip_card_for_ticker(holdings_df: pd.DataFrame, ticker: str, max_rows: int = 15) -> str:
    sub = holdings_df[holdings_df["fund_ticker"] == ticker]
    if sub.empty:
        return ""
    last_date = sub["ingest_date"].max()
    if pd.notna(last_date):
        sub = sub[sub["ingest_date"] == last_date]

    # Title / date
    fund_name = _escape(sub["fund_name"].iloc[0])
    last_update_str = _escape(last_date.strftime("%Y-%m-%d %H:%M") if pd.notna(last_date) else "N/A")

    # Top holdings by weight
    topn = (
        sub[["security_name", "security_weight"]]
        .dropna(subset=["security_name"])
        .sort_values("security_weight", ascending=False)
        .head(max_rows)
    )

    rows = []
    for _, r in topn.iterrows():
        sec = _escape(r["security_name"])
        wt = "" if pd.isna(r["security_weight"]) else f"{float(r['security_weight']):.2f}%"
        rows.append(f"<tr><td class='tt-sec'>{sec}</td><td class='tt-wt'>{wt}</td></tr>")

    table_html = (
        "<table class='tt-table'><thead><tr><th>Security</th><th>Weight</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    return f"""
    <div class="tt-card">
      <div class="tt-title">{fund_name}</div>
      <div class="tt-sub">Last update: <span class="tt-date">{last_update_str}</span></div>
      {table_html}
    </div>
    """

def make_ticker_chip_with_tooltip(ticker: str, card_html: str) -> str:
    t = _escape(ticker)
    return f"""
    <span class="tt-chip">{t}
      {card_html}
    </span>
    """

TOOLTIP_CSS = """
<style>
/* Chip */
.tt-chip {
  position: relative;
  display: inline-block;
  padding: 4px 8px;
  border-radius: 9999px;
  border: 1px solid rgba(0,0,0,0.10);
  background: #f7f7f9;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  cursor: default;
  white-space: nowrap;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.tt-chip:hover { background: #eef2ff; border-color: rgba(59,130,246,0.30); }

/* Tooltip card */
.tt-chip .tt-card {
  visibility: hidden;
  opacity: 0;
  transform: translateY(6px);
  transition: opacity .18s ease, transform .18s ease;
  position: absolute;
  left: 0;
  top: calc(100% + 8px);
  z-index: 1000;
  width: min(520px, 90vw);
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow:
    0 12px 28px rgba(0,0,0,0.08),
    0 2px 4px rgba(0,0,0,0.06);
  border-radius: 12px;
  padding: 12px 12px 8px 12px;
}
.tt-chip:hover .tt-card { visibility: visible; opacity: 1; transform: translateY(0); }

/* Card typography */
.tt-card .tt-title { font-weight: 700; font-size: 14px; margin-bottom: 4px; }
.tt-card .tt-sub { color: #667085; font-size: 12px; margin-bottom: 8px; }
.tt-card .tt-date { font-variant-numeric: tabular-nums; }

/* Table */
.tt-table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
.tt-table thead th { text-align: left; padding: 6px 6px; border-bottom: 1px solid #eee; }
.tt-table tbody td { padding: 6px 6px; border-bottom: 1px dashed #f0f0f0; vertical-align: top; word-wrap: break-word; }
.tt-table tbody tr:last-child td { border-bottom: none; }
.tt-sec { width: 80%; }
.tt-wt { width: 20%; text-align: right; font-variant-numeric: tabular-nums; }
</style>
"""

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

def breadth_column_chart(df: pd.DataFrame, value_col: str, bar_color: str) -> alt.Chart:
    df = df.copy()
    df["date_label"] = df["date"].dt.strftime("%b %d")
    return (
        alt.Chart(df)
        .mark_bar(color=bar_color)
        .encode(
            x=alt.X("date_label:N", axis=alt.Axis(title=None, labelAngle=-45)),
            y=alt.Y(f"{value_col}:Q", title=None),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{value_col}:Q", title=value_col.replace("_", " ").title()),
            ],
        )
        .properties(height=320)
    )

# ---------------------------------
# Formatting Helpers
# ---------------------------------
def format_rank(value: float) -> str:
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    pct = int(round(value * 100))
    if pct >= 85:
        bg = "rgba(16, 185, 129, 0.22)"
        border = "rgba(16, 185, 129, 0.35)"
    elif pct < 50:
        bg = "rgba(239, 68, 68, 0.22)"
        border = "rgba(239, 68, 68, 0.35)"
    else:
        bg = "rgba(156, 163, 175, 0.25)"
        border = "rgba(156, 163, 175, 0.35)"
    return (
        f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
        f'background-color:{bg}; border:1px solid {border}; color:inherit;">{pct}%</span>'
    )

def format_performance(value: float) -> str:
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    return f'<span style="display:block; text-align:right;">{value:.1f}%</span>'

def format_performance_intraday(value: float) -> str:
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    pct_text = f"{value:.1f}%"
    if value > 0:
        bg = "rgba(16, 185, 129, 0.22)"
        border = "rgba(16, 185, 129, 0.35)"
    elif value < 0:
        bg = "rgba(239, 68, 68, 0.22)"
        border = "rgba(239, 68, 68, 0.35)"
    else:
        bg = "rgba(156, 163, 175, 0.25)"
        border = "rgba(156, 163, 175, 0.35)"
    return (
        f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
        f'background-color:{bg}; border:1px solid {border}; color:inherit;">{pct_text}</span>'
    )

def format_indicator(value: str) -> str:
    value = str(value).strip().lower()
    if value == "yes":
        return '<span style="color:green; display:block; text-align:center;">✅</span>'
    if value == "no":
        return '<span style="color:red; display:block; text-align:center;">❌</span>'
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

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')

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
            border: 2px solid rgba(156, 163, 175, 0.7);
            border-radius: 8px;
            overflow: hidden;
        }}
        #{table_id} table thead th {{
            text-align: center !important;
            border-bottom: 2px solid rgba(156, 163, 175, 0.6);
            border-left: none !important;
            border-right: none !important;
            padding: 6px 8px;
        }}
        #{table_id} table tbody td {{
            border-bottom: 1px solid rgba(156, 163, 175, 0.22);
            border-left: none !important;
            border-right: none !important;
            padding: 6px 8px;
        }}
        #{table_id} table tbody tr:last-child td {{
            border-bottom: none;
        }}
        #{table_id} table td:nth-child(3),
        #{table_id} table td:nth-child(4),
        #{table_id} table td:nth-child(7),
        #{table_id} table td:nth-child(8),
        #{table_id} table td:nth-child(9),
        #{table_id} table td:nth-child(10) {{
            text-align: right !important;
        }}
        #{table_id} table td:nth-child(5),
        #{table_id} table td:nth-child(11),
        #{table_id} table td:nth-child(12),
        #{table_id} table td:nth-child(13),
        #{table_id} table td:nth-child(14) {{
            text-align: center !important;
        }}
    """
    st.markdown(f'<div id="{table_id}"><style>{css}</style>{html}</div>', unsafe_allow_html=True)

# ---------------------------------
# Dashboard Rendering
# ---------------------------------
def render_dashboard(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> None:
    st.title("US Market Daily Snapshot")

    # Inject tooltip CSS (once)
    st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

    latest, rs_last_n = process_data(df_etf, df_rs)
    if "date" in latest.columns:
        st.caption(f"Latest Update: {pd.to_datetime(latest['date']).max().date()}")
    else:
        st.caption("Latest Update: N/A")

    # Load holdings for tooltips
    try:
        df_holdings = load_holdings_csv(DATA_URLS["holdings"])
    except Exception as e:
        df_holdings = pd.DataFrame()
        st.warning(f"Holdings tooltips disabled — {e}")

    if "group" not in latest.columns:
        st.error("Column 'group' is missing in ETF dataset — cannot render grouped tables.")
        return

    group_tickers = latest.groupby("group").groups
    for group_name in GROUP_ORDER:
        if group_name not in group_tickers:
            continue
        st.header(f"📌 {group_name}")
        tickers_in_group = group_tickers[group_name]

        # Sort by rs_rank_21d if present
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

            ticker_cell = ticker  # default plain text
            if not df_holdings.empty:
                card_html = make_tooltip_card_for_ticker(df_holdings, ticker, max_rows=15)
                if card_html:
                    ticker_cell = make_ticker_chip_with_tooltip(ticker, card_html)

            rows.append({
                "Ticker": ticker_cell,
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
                "Above SMA5": format_indicator(row.get("above_sma5")),
                "Above SMA10": format_indicator(row.get("above_sma10")),
                "Above SMA20": format_indicator(row.get("above_sma20")),
            })

        render_group_table(group_name, rows)

    # ---- Breadth charts at bottom ----
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
        st.info("`rs_rank_21d` not found in ETF data — breadth charts skipped.")

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
