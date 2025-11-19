import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
import io
import base64
import re
import html
from typing import List, Dict, Tuple

# ---------------------------------
# Configuration
# ---------------------------------
st.set_page_config(page_title="US Market Daily Snapshot", layout="wide")

# Constants
DATA_URLS = {
    "etf": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price.csv",
    "rs": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_rs_sparkline.csv",
    "holdings": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_holdings.csv",
    "heat": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_heat.csv",
    "stage": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_etf_price_weekly.csv",
}
LOOKBACK_DAYS = 21
GROUP_ORDER = ["Market", "Sector", "Commodity", "Crypto", "Country", "Theme", "Leader"]

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

# Settings
use_group_colors = True        # color-code ticker chips by group
max_holdings_rows = 10         # rows shown in tooltip table (ignored now, kept for compatibility)

# ---------------------------------
# Small Helpers
# ---------------------------------
def _clean_text_series(s: pd.Series, title_case: bool = False) -> pd.Series:
    """Clean text series by removing extra whitespace and optionally title-casing."""
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
    replacements = [
        (r"\betf\b", "ETF"),
        (r"\busd\b", "USD"),
        (r"\busa\b", "USA"),
        (r"\breit\b", "REIT"),
        (r"\bai\b", "AI"),
        (r"\bus\b", "US"),
        (r"\bS\s*(?:&|and|-)\s*P\b", "S&P"),
    ]
    for pattern, repl in replacements:
        s = s.str.replace(pattern, repl, regex=True, flags=re.IGNORECASE)
    return s

def _clean_ticker_series(s: pd.Series) -> pd.Series:
    """Clean ticker series by removing whitespace and uppercasing."""
    return (
        s.astype(str)
        .str.replace(r"[\r\n\t\s]+", "", regex=True)
        .str.upper()
        .str.strip()
    )

def _escape(s) -> str:
    """HTML-escape a string, handling NaN."""
    return html.escape("" if pd.isna(s) else str(s))

def slugify(text: str) -> str:
    """Convert text to a slug for CSS/IDs."""
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')

# ---------------------------------
# Data Loading
# ---------------------------------
@st.cache_data(ttl=900)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean ETF price and RS data."""
    df_etf = pd.read_csv(DATA_URLS["etf"])
    df_rs = pd.read_csv(DATA_URLS["rs"])

    # Validate required columns
    etf_req = {"date", "ticker"}
    rs_req = {"date", "ticker", "rs_to_spy"}
    if not etf_req.issubset(df_etf.columns):
        raise ValueError(f"ETF price CSV must include {etf_req}.")
    if not rs_req.issubset(df_rs.columns):
        raise ValueError(f"RS CSV must include {rs_req}.")

    # Convert dates
    df_etf["date"] = pd.to_datetime(df_etf["date"], errors="coerce")
    df_rs["date"] = pd.to_datetime(df_rs["date"], errors="coerce")

    # Clean common columns
    for df, cols in [(df_etf, ["ticker"]), (df_rs, ["ticker"])]:
        for col in cols:
            if col in df.columns:
                df[col] = _clean_ticker_series(df[col])
    for df, cols in [(df_etf, ["group"]), (df_rs, ["group"])]:
        for col in cols:
            if col in df.columns:
                df[col] = _clean_text_series(df[col])

    return df_etf, df_rs

@st.cache_data(ttl=900, show_spinner=False)
def load_holdings_csv(url: str = DATA_URLS["holdings"]) -> pd.DataFrame:
    """Load and clean ETF holdings data."""
    df = pd.read_csv(url)
    required = {"fund_ticker", "fund_name", "security_name", "security_ticker", "security_weight", "ingest_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Holdings CSV missing columns: {sorted(missing)}")

    df["ingest_date"] = pd.to_datetime(df["ingest_date"], errors="coerce")
    df["security_weight"] = pd.to_numeric(df["security_weight"], errors="coerce")

    df["fund_ticker"] = _clean_ticker_series(df["fund_ticker"])
    df["fund_name"] = _fix_acronyms_in_name(_clean_text_series(df["fund_name"], title_case=False))
    df["security_name"] = _fix_acronyms_in_name(_clean_text_series(df["security_name"], title_case=True))
    df["security_ticker"] = _clean_ticker_series(df["security_ticker"])
    return df

@st.cache_data(ttl=900)
def load_heat_csv(url: str = DATA_URLS["heat"]) -> pd.DataFrame:
    """Load and normalize heat data."""
    df = pd.read_csv(url)
    # Normalize column names to lowercase keys mapping to original name
    colmap = {c.lower(): c for c in df.columns}
    # Required fields
    required = ['date', 'ticker', 'code', 'pricefactor', 'volumefactor']
    missing = [r for r in required if r not in colmap]
    if missing:
        raise ValueError(f"Missing columns in heat CSV: {missing}. Available: {list(df.columns)}")
    # Rename to canonical names
    df = df.rename(columns={
        colmap['date']: 'date',
        colmap['ticker']: 'ticker',
        colmap['code']: 'code',
        colmap['pricefactor']: 'PriceFactor',
        colmap['volumefactor']: 'VolumeFactor'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=900)
def load_stage_csv(url: str = DATA_URLS["stage"]) -> pd.DataFrame:
    """Load and normalize weekly stage data (latest per ticker only)."""
    df = pd.read_csv(url)

    # Normalize column names to lowercase keys mapping to original names
    colmap = {c.lower(): c for c in df.columns}

    # Required fields
    required = ['ticker', 'date', 'stage_label_core', 'stage_label_adj']
    missing = [r for r in required if r not in colmap]
    if missing:
        raise ValueError(f"Missing columns in stage CSV: {missing}. Available: {list(df.columns)}")

    # Rename to canonical names
    df = df.rename(columns={
        colmap['ticker']: 'ticker',
        colmap['date']: 'date',
        colmap['stage_label_core']: 'stage_label_core',
        colmap['stage_label_adj']: 'stage_label_adj'
    })

    # Ensure date is datetime type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Keep only the most recent record per ticker
    df = df.sort_values(['ticker', 'date']).groupby('ticker', as_index=False).tail(1)

    return df

# ---------------------------------
# Data Processing
# ---------------------------------
def process_data(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process ETF and RS data to latest snapshot and recent RS series."""
    latest = df_etf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rs_last_n = df_rs.sort_values(["ticker", "date"]).groupby("ticker").tail(LOOKBACK_DAYS)
    return latest, rs_last_n

def compute_threshold_counts(df_etf: pd.DataFrame) -> pd.DataFrame:
    """Compute daily counts of tickers over 85% and under 50% RS rank for last 21 days."""
    if "count_over_85" not in df_etf.columns or "count_under_50" not in df_etf.columns:
        return pd.DataFrame(columns=["date", "count_over_85", "count_under_50", "date_str"])
    daily = df_etf[["date", "count_over_85", "count_under_50"]].drop_duplicates().sort_values("date")
    daily = daily.dropna(subset=["count_over_85", "count_under_50"])
    last_21_dates = daily["date"].tail(21)
    daily_21 = daily[daily["date"].isin(last_21_dates)].copy().sort_values("date")
    daily_21["date_str"] = daily_21["date"].dt.strftime("%Y-%m-%d")
    return daily_21

# ---------------------------------
# Chips + Tooltip (HTML/CSS)
# ---------------------------------
def make_tooltip_card_for_ticker(holdings_df: pd.DataFrame, ticker: str, latest_row=None, max_rows: int = 10) -> str:
    """Generate HTML tooltip card for a ticker.

    Shows two sections only:
      - Performance: 1W Return and 1M Return
      - Moving Average: Above SMA5 and Above SMA10

    holdings_df is accepted for compatibility but ignored (we deliberately DO NOT show holdings).
    latest_row should be the latest row/series/dict for that ticker (contains ret_1w, ret_1m, above_sma5, above_sma10).
    """
    # Title (use ticker as primary title — holdings removed)
    title = _escape(ticker)

    # Last update date: try common date fields from latest_row if present
    last_update_str = "N/A"
    if latest_row is not None:
        # latest_row can be pd.Series or dict
        try:
            date_val = latest_row.get("date") if isinstance(latest_row, dict) else latest_row.get("date", None)
        except Exception:
            try:
                date_val = latest_row["date"]
            except Exception:
                date_val = None
        if pd.notna(date_val):
            try:
                last_update_str = _escape(pd.to_datetime(date_val).strftime("%Y-%m-%d"))
            except Exception:
                last_update_str = _escape(str(date_val))

    # Helper to safely get values from latest_row
    def _val(key):
        if latest_row is None:
            return None
        try:
            return latest_row.get(key) if isinstance(latest_row, dict) else latest_row.get(key, None)
        except Exception:
            try:
                return latest_row[key]
            except Exception:
                return None

    # Use existing formatters so visual style matches table cells elsewhere
    ret_1w_html = format_performance(_val("ret_1w"))
    ret_1m_html = format_performance(_val("ret_1m"))
    above_sma5_html = format_indicator(_val("above_sma5"))
    above_sma10_html = format_indicator(_val("above_sma10"))

    # Build compact "Performance" and "Moving Average" tables
    perf_table = (
        "<table class='tt-table small'><thead><tr><th colspan='2'>Performance</th></tr></thead>"
        "<tbody>"
        f"<tr><td>1W Return</td><td style='text-align:right'>{ret_1w_html}</td></tr>"
        f"<tr><td>1M Return</td><td style='text-align:right'>{ret_1m_html}</td></tr>"
        "</tbody></table>"
    )

    ma_table = (
        "<table class='tt-table small'><thead><tr><th colspan='2'>Moving Average</th></tr></thead>"
        "<tbody>"
        f"<tr><td>Above SMA5</td><td style='text-align:center'>{above_sma5_html}</td></tr>"
        f"<tr><td>Above SMA10</td><td style='text-align:center'>{above_sma10_html}</td></tr>"
        "</tbody></table>"
    )

    return (
        f'<div class="tt-card">'
        f'<div class="tt-title">{title}</div>'
        f'<div class="tt-sub">Last update: <span class="tt-date">{last_update_str}</span></div>'
        f'<div class="tt-sections">{perf_table}{ma_table}</div>'
        f'</div>'
    )

def make_ticker_chip_with_tooltip(ticker: str, card_html: str, group_name: str | None) -> str:
    """Generate HTML chip for ticker with optional tooltip."""
    t = _escape(ticker)
    group_class = ""
    if use_group_colors and group_name:
        group_slug = slugify(group_name)
        group_class = f" chip--{group_slug}"
    return f'<span class="tt-chip{group_class}">{t}{card_html}</span>'

def build_chip_css() -> str:
    """Generate CSS for ticker chips and tooltips."""
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

/* Tooltip card centered on screen; scroll if tall */
.tt-chip .tt-card {
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%) translateY(6px);
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
  transform: translate(-50%, -50%);
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

/* Mobile fallback: adjust for smaller screen */
@media (max-width: 768px) {
  .tt-chip .tt-card {
    position: fixed;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 95vw;
    max-height: 70vh;
    max-width: none;
    overflow: auto;
    padding: 12px;
  }
  .tt-chip:hover .tt-card {
    visibility: visible;
    opacity: 1;
    transform: translate(-50%, -50%);
  }
  /* Table adjustments for mobile */
  .tt-table {
    font-size: 11px;
  }
  .tt-table thead th,
  .tt-table tbody td {
    padding: 4px 4px;
  }
  .tt-sec {
    font-size: 10px;
  }
}

/* Navigation links in sidebar: no underline */
.stSidebar a {
    text-decoration: none;
}

/* Tooltip sections layout: side-by-side on wide screens, stacked on mobile */
.tt-card .tt-sections { display:flex; gap:10px; margin-bottom:8px; align-items:flex-start; flex-wrap:wrap; }
.tt-card .tt-table.small { width: 48%; font-size:12px; margin:0; border-collapse:collapse; }
.tt-card .tt-table.small thead th { font-weight:700; font-size:12px; padding-bottom:6px; }
.tt-card .tt-table.small tbody td { padding:4px 6px; border-bottom:none; vertical-align:middle; }
@media (max-width:768px) {
  .tt-card .tt-table.small { width:100%; }
}
"""
    group_css_parts = []
    if use_group_colors:
        for g, (base, dark) in GROUP_PALETTE.items():
            slug = slugify(g)
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
    """Create a base64-encoded PNG sparkline image."""
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
    """Create an Altair bar chart for breadth counts."""
    df = df.copy()
    df["date_label"] = df["date"].dt.strftime("%b %d")

    return (
        alt.Chart(df)
        .mark_bar(color=bar_color)
        .encode(
            x=alt.X(
                "date_label:N",
                sort=None,
                axis=alt.Axis(title=None, labelAngle=-45)
            ),
            y=alt.Y(f"{value_col}:Q", title=None),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{value_col}:Q", title=value_col.replace("_", " ").title())
            ]
        )
        .properties(height=320)
    )

# ---------------------------------
# Formatting Helpers
# ---------------------------------
def format_rank(value: float) -> str:
    """Format RS rank as a colored percentage badge."""
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    pct = int(round(value * 100))
    if pct >= 85:
        bg, border = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)"
    elif pct < 50:
        bg, border = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"
    else:
        bg, border = "rgba(156,163,175,.25)", "rgba(156,163,175,.35)"
    return (
        f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
        f'background-color:{bg}; border:1px solid {border}; color:inherit;">{pct}%</span>'
    )

def format_performance(value: float) -> str:
    """Format performance return as percentage."""
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    return f'<span style="display:block; text-align:right;">{value:.1f}%</span>'

def format_performance_intraday(value: float) -> str:
    """Format intraday return as a colored percentage badge."""
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    pct_text = f"{value:.1f}%"
    if value > 0:
        bg, border = "rgba(16,185,129,.22)", "rgba(16,185,129,.35)"
    elif value < 0:
        bg, border = "rgba(239,68,68,.22)", "rgba(239,68,68,.35)"
    else:
        bg, border = "rgba(156,163,175,.25)", "rgba(156,163,175,.35)"
    return (
        f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
        f'background-color:{bg}; border:1px solid {border}; color:inherit;">{pct_text}</span>'
    )

def format_52w_high(value: float) -> str:
    """Format 52-week pct_below_high with rocket emoji if value > -3."""
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    if value > -3:
        return '<span style="display:block; text-align:center;">🚀</span>'
    formatted_value = f'{value:.1f}%'
    return f'<span style="display:block; text-align:right;">{formatted_value}</span>'

def format_52w_low(value: float) -> str:
    """Format 52-week pct_above_low with snail emoji if value < 3."""
    if pd.isna(value):
        return '<span style="display:block; text-align:right;">-</span>'
    if value < 3:
        return '<span style="display:block; text-align:center;">🐌</span>'
    formatted_value = f'{value:.1f}%'
    return f'<span style="display:block; text-align:right;">{formatted_value}</span>'

def format_indicator(value: str) -> str:
    """Format yes/no indicator as emoji."""
    value = str(value).strip().lower()
    if value == "yes":
        return '<span style="color:green; display:block; text-align:center;">✅</span>'
    if value == "no":
        return '<span style="color:red; display:block; text-align:center;">❌</span>'
    return '<span style="display:block; text-align:center;">-</span>'

def format_volume_alert(value: str, rs_rank_252d) -> str:
    """Format volume alert with diamond for strong RS."""
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
    """Format extension multiple as a colored badge."""
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
    return (
        f'<span style="display:block; text-align:right; padding:2px 6px; border-radius:6px; '
        f'background-color:{bg}; border:1px solid {border}; color:inherit;">{txt}</span>'
    )

def format_volatility(value: str) -> str:
    """Format ratio indicator as directional emoji combo: up=warning, down=on-target."""
    try:
        value = str(value).strip().lower()
        if value == "compression":
            return '<span style="color:green; display:block; text-align:center;">🎯</span>'  
        elif value == "spike":
            return '<span style="color:red; display:block; text-align:center;">⚠️</span>'
        else:
            return '<span style="display:block; text-align:center;">-</span>'
    except ValueError:
        return '<span style="display:block; text-align:center;">-</span>'

def format_stage_label(value: str) -> str:
    """Format stage label with colored dot emoji."""
    try:
        value = str(value).strip().lower()
        if value == "stage 1":
            return '<span style="display:block; text-align:center;">🟡</span>'  # yellow
        elif value == "stage 2":
            return '<span style="display:block; text-align:center;">🟢</span>'  # green
        elif value == "stage 3":
            return '<span style="display:block; text-align:center;">🟠</span>'  # orange
        elif value == "stage 4":
            return '<span style="display:block; text-align:center;">🔴</span>'  # red
        else:
            return '<span style="display:block; text-align:center;">⚪</span>'  # grey/other
    except Exception:
        return '<span style="display:block; text-align:center;">⚪</span>'

# ---------------------------------
# Table Rendering
# ---------------------------------
def render_group_table(group_name: str, rows: List[Dict]) -> None:
    """Render a group table as styled HTML."""
    table_id = f"tbl-{slugify(group_name)}"
    html_str = f'<div style="padding-bottom: 40px; overflow: visible;">{pd.DataFrame(rows).to_html(escape=False, index=False)}</div>'

    css = f"""
        #{table_id} table {{
            width: 100%;
            border-collapse: collapse;
            border-spacing: 0;
            border: none;
            border-radius: 8px;
            overflow: visible;
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
            padding: 8px 8px;
            position: relative;
            overflow: visible;
        }}
        #{table_id} table tbody tr:last-child td {{ 
            border-bottom: none; 
            padding-bottom: 12px;
        }}
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
        #{table_id} table td:nth-child(1) {{ white-space: nowrap; line-height: 1.25; overflow: visible; }}
        /* Ensure chart links have space */
        #{table_id} table td:nth-child(14) a {{ display: block; margin: 0 auto; }}
    """
    st.markdown(f'<div id="{table_id}"><style>{css}</style>{html_str}</div>', unsafe_allow_html=True)

# ---------------------------------
# Heatmap Rendering Helpers
# ---------------------------------
def render_heat_scatter(df_latest: pd.DataFrame, latest_date: str) -> None:
    """Render scatter plot of latest PriceFactor vs VolumeFactor."""
    st.subheader("🧠 Price & Volume Analysis")
    st.caption("High Volume Factor = Volume Accumulation · High Price Factor = Price Compression")
    st.caption(f"Data as of {latest_date}")
    
    if df_latest.empty:
        st.warning("No data available after filtering.")
        return

    # Add formatted date string (without time) for tooltip
    df_latest = df_latest.copy()
    df_latest['date_str'] = df_latest['date'].dt.strftime('%b %d, %Y')
    
    fig = px.scatter(
        df_latest,
        x='VolumeFactor',
        y='PriceFactor',
        color='code',
        custom_data=['date_str', 'ticker', 'PriceFactor', 'VolumeFactor'],
        height=550,
    )
    fig.update_traces(
        marker=dict(size=14, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "<i>%{customdata[0]}</i><br>"
            "Price Factor: %{customdata[2]:.2f}<br>"
            "Volume Factor: %{customdata[3]:.2f}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="Volume Factor",
        yaxis_title="Price Factor",
        hovermode='closest',
        template='plotly_white',
    )
    st.plotly_chart(fig, use_container_width=True)

def render_heat_heatmaps(df_heat: pd.DataFrame) -> None:
    """Render side-by-side heatmaps for VolumeFactor and PriceFactor over time."""
    if df_heat.empty:
        st.warning("No heatmap data available.")
        return
    
    # Order tickers by code
    ticker_order_df = df_heat.groupby(['ticker', 'code'])['date'].min().reset_index().sort_values(['code', 'ticker'])
    ticker_list = ticker_order_df['ticker'].tolist()
    
    # Sorted dates
    dates_sorted = sorted(df_heat['date'].unique())
    
    # Pivot tables
    vol_pivot = df_heat.pivot_table(index='ticker', columns='date', values='VolumeFactor', aggfunc='last').reindex(index=ticker_list, columns=dates_sorted)
    price_pivot = df_heat.pivot_table(index='ticker', columns='date', values='PriceFactor', aggfunc='last').reindex(index=ticker_list, columns=dates_sorted)
    
    # Ticker to code map
    code_map = df_heat.groupby('ticker')['code'].first().to_dict()
    
    if vol_pivot.empty or price_pivot.empty:
        st.warning("No heatmap data available.")
        return
    
    # Customdata for codes
    vol_customdata = [[code_map.get(t, "")] * vol_pivot.shape[1] for t in vol_pivot.index]
    price_customdata = [[code_map.get(t, "")] * price_pivot.shape[1] for t in price_pivot.index]
    
    # X labels
    x_labels = [d.strftime("%Y-%m-%d") for d in vol_pivot.columns]
    
    # Dynamic height: base + ~20px per ticker (adjust as needed)
    num_tickers = len(ticker_list)
    dynamic_height = max(520, 100 + num_tickers * 20)  # Minimum 520px, scales with tickers
    
    # Heatmaps in columns
    col1, col2 = st.columns(2)
    
    with col1:
        fig_vol = go.Figure(
            data=go.Heatmap(
                z=vol_pivot.values,
                x=x_labels,
                y=vol_pivot.index.tolist(),
                colorscale='RdYlGn',
                showscale=False,
                colorbar=dict(title='VolumeFactor'),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Date: %{x}<br>"
                    "Volume Factor: %{z:.2f}<br>"
                    "Code: %{customdata}<extra></extra>"
                ),
                customdata=vol_customdata
            )
        )
        fig_vol.update_layout(height=dynamic_height, margin=dict(t=40, b=40), title="Volume Factor", hoverlabel=dict(align='left'))
        fig_vol.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        fig_price = go.Figure(
            data=go.Heatmap(
                z=price_pivot.values,
                x=x_labels,
                y=price_pivot.index.tolist(),
                colorscale='RdYlGn',
                showscale=False,
                colorbar=dict(title='PriceFactor'),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Date: %{x}<br>"
                    "Price Factor: %{z:.2f}<br>"
                    "Code: %{customdata}<extra></extra>"
                ),
                customdata=price_customdata
            )
        )
        fig_price.update_layout(height=dynamic_height, margin=dict(t=40, b=40), title="Price Factor", hoverlabel=dict(align='left'))
        fig_price.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------
# Dashboard Rendering
# ---------------------------------
def render_dashboard(df_etf: pd.DataFrame, df_rs: pd.DataFrame) -> None:
    """Render the full dashboard."""
    st.title("US Market Daily Snapshot")

    # Inject CSS
    st.markdown(build_chip_css(), unsafe_allow_html=True)

    latest, rs_last_n = process_data(df_etf, df_rs)
    latest_date = latest["date"].max().date() if "date" in latest.columns else "N/A"
    st.caption(f"Latest Update: {latest_date}")

    # Sidebar for global filters
    with st.sidebar:
        st.header("Filters")
        hide_rs = st.toggle('Hide RS', value=False, help="Hide all tickers with RS Rank (1M) below 85%")
        hide_extension = st.toggle('Hide Multiple', value=False, help="Hide all tickers with Extension Multiple above 4")
        hide_pv = st.toggle('Hide Price & Volume', value=False, help="Hide all tickers where Price Factor is below 0.60 or Volume Factor is below 0.65 (based on latest values)")
        st.markdown("---")
        st.markdown("## Navigation")
        for group_name in GROUP_ORDER:
            st.markdown(f'[📌 {group_name}](#{slugify(group_name)})', unsafe_allow_html=True)
        st.markdown(f'[✏️ Breadth Gauge](#breadth-gauge)', unsafe_allow_html=True)
        st.markdown(f'[🧠 Price & Volume Analysis](#price-volume)', unsafe_allow_html=True)

    # Load optional data with fallbacks
    try:
        df_holdings = load_holdings_csv()
    except Exception as e:
        df_holdings = pd.DataFrame()
        st.warning(f"Holdings tooltips disabled — {e}")

    try:
        df_heat = load_heat_csv()
    except Exception as e:
        df_heat = pd.DataFrame()
        st.warning(f"Heat data unavailable — {e}")

    try:
        df_stage = load_stage_csv()
    except Exception as e:
        df_stage = pd.DataFrame()
        st.warning(f"Stage data unavailable — {e}")

    # Merge stage labels from weekly data into latest (if available)
    if not df_stage.empty:
        latest = (
            latest
            .reset_index()
            .merge(
                df_stage[['ticker', 'stage_label_core', 'stage_label_adj']],
                on='ticker',
                how='left'
            )
            .set_index('ticker')
        )

    if "group" not in latest.columns:
        st.error("Column 'group' is missing in ETF dataset — cannot render grouped tables.")
        return

    # Render group tables
    group_tickers = latest.groupby("group").groups
    for group_name in GROUP_ORDER:
        anchor_id = slugify(group_name)
        st.markdown(f'<div id="{anchor_id}" style="padding-top: 80px; margin-top: -80px;"></div>', unsafe_allow_html=True)
        st.header(f"📌 {group_name}")
        if group_name not in group_tickers:
            st.info(f"No data available for {group_name}.")
            continue
        tickers_in_group = group_tickers[group_name]

        # Filter by RS rank if toggle is on
        if hide_rs and "rs_rank_21d" in latest.columns:
            tickers_in_group = [t for t in tickers_in_group if latest.loc[t, "rs_rank_21d"] >= 0.85]
            if not tickers_in_group:
                st.info(f"No tickers meet the RS threshold for {group_name}.")
                continue

        # Filter by Extension Multiple if toggle is on
        if hide_extension and "ratio_pct_dist_to_atr_pct" in latest.columns:
            tickers_in_group = [t for t in tickers_in_group if latest.loc[t, "ratio_pct_dist_to_atr_pct"] <= 4]
            if not tickers_in_group:
                st.info(f"No tickers meet the Extension Multiple threshold for {group_name}.")
                continue

        # Sort by RS rank if available
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
            # Always call tooltip generator (holdings_df accepted but now ignored).
            # Pass the latest row so tooltip can render ret_1w, ret_1m, above_sma5, above_sma10
            card_html = make_tooltip_card_for_ticker(df_holdings, ticker, latest_row=row, max_rows=max_holdings_rows)
            if card_html:
                chip = make_ticker_chip_with_tooltip(ticker, card_html, group_name)

            rows.append({
                "Ticker": chip,
                "Relative Strength": create_sparkline(spark_series),
                "RS Rank (1M)": format_rank(row.get("rs_rank_21d")),
                "RS Rank (1Y)": format_rank(row.get("rs_rank_252d")),
                " ": "",
                "Volume Alert": format_volume_alert(row.get("volume_alert", "-"), row.get("rs_rank_252d")),
                "Volatility Alert": format_volatility(row.get("volatility_signal")),
                " ": "",
                "Intraday": format_performance_intraday(row.get("ret_intraday")),
                "1D Return": format_performance(row.get("ret_1d")),
                "1W Return": format_performance(row.get("ret_1w")),
                "1M Return": format_performance(row.get("ret_1m")),
                "52W High": format_52w_high(row.get("pct_below_high")),
                "52W Low": format_52w_low(row.get("pct_above_low")),
                "  ": "",
                "Extension Multiple": format_multiple(row.get("ratio_pct_dist_to_atr_pct")),
                "  ": "",
                "Above SMA5": format_indicator(row.get("above_sma5")),
                "Above SMA10": format_indicator(row.get("above_sma10")),
                "Above SMA20": format_indicator(row.get("above_sma20")),
                "  ": "",
                "Core Model": format_stage_label(row.get("stage_label_core")),
                "Modified Model": format_stage_label(row.get("stage_label_adj"))
            })

        render_group_table(group_name, rows)

    # Breadth charts
    counts_21 = compute_threshold_counts(df_etf)
    st.markdown('<div id="breadth-gauge" style="padding-top: 80px; margin-top: -80px;"></div>', unsafe_allow_html=True)
    st.subheader("✏️ Breadth Gauge")
    if not counts_21.empty:
        start_date = counts_21["date"].min().date()
        end_date = counts_21["date"].max().date()
        
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

    # Heat data visualizations
    st.markdown('<div id="price-volume" style="padding-top: 80px; margin-top: -80px;"></div>', unsafe_allow_html=True)
    if not df_heat.empty:
        df_heat_latest = df_heat.sort_values('date').groupby('ticker').tail(1)
        df_heat_latest_date = df_heat['date'].max().strftime("%Y-%m-%d")
        
        if hide_pv:
            mask = (df_heat_latest['PriceFactor'] >= 0.60) & (df_heat_latest['VolumeFactor'] >= 0.65)
            if mask.sum() == 0:
                st.warning("No tickers meet the Price & Volume threshold.")
            else:
                df_heat_filtered = df_heat[df_heat['ticker'].isin(df_heat_latest[mask]['ticker'])]
                render_heat_scatter(df_heat_latest[mask], df_heat_latest_date)
                render_heat_heatmaps(df_heat_filtered)
        else:
            render_heat_scatter(df_heat_latest, df_heat_latest_date)
            render_heat_heatmaps(df_heat)
    else:
        st.warning("Heat data not available — skipping price/volume analysis.")

# ---------------------------------
# Main
# ---------------------------------
def main():
    try:
        df_etf, df_rs = load_data()
    except Exception as e:
        st.error(f"Failed to load price/RS data — {e}")
        st.stop()
    try:
        render_dashboard(df_etf, df_rs)
    except Exception as e:
        st.exception(e)

if __name__ == "__main__":
    main()
