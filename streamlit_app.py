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
max_holdings_rows = 15         # rows shown in tooltip table

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
    s = s.astype(str)
    s = s.str.replace(r"\betf\b", "ETF", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\busd\b", "USD", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\busa\b", "USA", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\breit\b", "REIT", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\bai\b", "AI", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"\bus\b", "US", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(r"s&?p", "S&P", regex=True, flags=re.IGNORECASE)
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
    df = pd.read_csv(url)
    need = {"fund_ticker","fund_name","security_name","security_ticker","security_weight","ingest_date"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[holdings] Missing columns: {sorted(missing)}")

    df["ingest_date"] = pd.to_datetime(df["ingest_date"], errors="coerce")
    df["security_weight"] = pd.to_numeric(df["security_weight"], errors="coerce")

    df["fund_ticker"]     = _clean_ticker_series(df["fund_ticker"])
    df["fund_name"]       = _fix_acronyms_in_name(_clean_text_series(df["fund_name"], title_case=False))
    df["security_name"]   = _fix_acronyms_in_name(_clean_text_series(df["security_name"], title_case=True))
    df["security_ticker"] = _clean_ticker_series(df["security_ticker"])
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

    fund_name = _escape(sub["fund_name"].iloc[0])
    last_update_str = _escape(last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "N/A")

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

/* Tooltip card: auto-size full table */
.tt-chip .tt-card {
  position: absolute;
  left: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%) translateX(6px);
  z-index: 999999;
  width: min(520px, 90vw);

  max-height: 90vh;   /* soft cap so it doesn’t overflow screen */
  overflow: visible;

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

def breadth_column_chart(df: pd.DataFrame, value_col: str, bar_color: str) -> alt.Chart:
    df = df.copy()
    df["date_label"] = df["date"].dt.strftime("%b %d")
    return (
        alt.Chart(df)
        .mark_bar(color=bar_color)
        .encode(
            x=alt.X("date_label:N", axis=alt.Axis(title=None, labelAngle=-45)),
            y=alt.Y(f"{value_col}:Q", title=None),
            tooltip=[alt.Tooltip("date:T", title="Date"),
                     alt.Tooltip(f"{value_col}:Q", title=value_col.replace("_"," ").title())]
        )
        .properties(height=320)
    )

# ---------------------------------
# Formatting Helpers (shortened here for brevity)
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
    if value == "yes": return '<span style="color:green; display:block; text-align:center;"></span>'
    if value == "no":  return '<span style="color:red; display:block; text-align:center;"></span>'
    return '<span style="display:block; text-align:center;">-</span>'

def format_volume_alert(value: str, rs_rank_252d) -> str:
    if not isinstance(value, str):
        return '<span style="display:block; text-align:center;">-</span>'
    return '<span style="display:block; text-align:center; font-size:16px;"></span>'

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
        border: none;
        border-radius: 8px;
    }}
    #{table_id} th, #{table_id} td {{
        padding: 6px 10px;
        font-size: 13px;
        border: none;
    }}
    #{table_id} thead th {{
        border-bottom: 1px solid #ddd;
        text-align: left;
        background: #f9fafb;
    }}
    #{table_id} tbody tr:hover {{
        background: #f9fafb;
    }}
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(f"<div id='{table_id}'>{html}</div>", unsafe_allow_html=True)

# ---------------------------------
# Main App
# ---------------------------------
def main():
    st.markdown(build_chip_css(), unsafe_allow_html=True)
    st.title("US Market Snapshot")

    df_etf, df_rs = load_data()
    df_holdings = load_holdings_csv()
    latest, rs_last_n = process_data(df_etf, df_rs)
    df_threshold = compute_threshold_counts(df_etf)

    if not df_threshold.empty:
        col1, col2 = st.columns(2)
        with col1: st.altair_chart(breadth_column_chart(df_threshold,"count_over_85","#10b981"), use_container_width=True)
        with col2: st.altair_chart(breadth_column_chart(df_threshold,"count_under_50","#ef4444"), use_container_width=True)

    for g in GROUP_ORDER:
        if g not in latest["group"].unique(): continue
        st.subheader(g)
        rows = []
        group_tickers = latest[latest["group"] == g].index.tolist()
        for tk in group_tickers:
            snap = latest.loc[tk]
            rs_slice = rs_last_n[rs_last_n["ticker"] == tk]
            spark = create_sparkline(rs_slice["rs_to_spy"].tolist()) if not rs_slice.empty else ""
            card_html = make_tooltip_card_for_ticker(df_holdings, tk, max_holdings_rows)
            chip = make_ticker_chip_with_tooltip(tk, card_html, g)
            rows.append({
                "Ticker": chip,
                "RS Rank": format_rank(snap.get("rs_rank_252d")),
                "RS vs SPY": spark,
                "Perf (21d)": format_performance(snap.get("perf_21d")),
                "Perf (5d)": format_performance(snap.get("perf_5d")),
                "Perf (1d)": format_performance_intraday(snap.get("perf_1d")),
                "Stage": snap.get("stage","-"),
                "ETF25": format_indicator(snap.get("in_top_25")),
                "Volume": format_volume_alert(snap.get("volume_alert"), snap.get("rs_rank_252d"))
            })
        render_group_table(g, rows)

if __name__ == "__main__":
    main()
