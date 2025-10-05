# full_app_with_click_to_filter.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.express as px
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
    "heat": "https://raw.githubusercontent.com/sanglocn/us_snapshot/main/data/us_snapshot_heat.csv",
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

@st.cache_data
def load_and_normalize(url):
    df = pd.read_csv(url)
    # normalize column names to lowercase keys mapping to original name
    colmap = {c.lower(): c for c in df.columns}
    # required fields
    required = ['date', 'ticker', 'code', 'pricefactor', 'volumefactor']
    missing = [r for r in required if r not in colmap]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available columns: {list(df.columns)}")
    # rename to canonical names
    df = df.rename(columns={
        colmap['date']: 'date',
        colmap['ticker']: 'ticker',
        colmap['code']: 'code',
        colmap['pricefactor']: 'PriceFactor',
        colmap['volumefactor']: 'VolumeFactor'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df_heat = load_and_normalize(DATA_URLS["heat"])
except Exception as e:
    st.error(f"Failed to load/normalize data: {e}")
    st.stop()

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

    fund_name = _escape(sub["fund_name"].iloc(0)) if not sub.empty else ""
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

# ... (I kept the rest of your helper functions -- create_sparkline, format helpers, etc. unchanged)
# For brevity I'm not repeating them here; they remain exactly as in your original file.

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

    # --- chart data for modal/sidebar ---
    try:
        df_chart = load_chart_csv(DATA_URLS["chart"])
    except Exception as e:
        df_chart = pd.DataFrame()
        st.warning(f"Chart data unavailable — {e}")

    # ... (rows / grouped table rendering remains the same)
    # [unchanged code omitted for brevity: group tables and breadth charts]

    # --- Load heat CSV (we already loaded earlier as df_heat via load_and_normalize) ---
    # Ensure we have canonical columns (PriceFactor, VolumeFactor, code, date, ticker)
    df_heat_local = df_heat.copy()
    df_heat_local['date'] = pd.to_datetime(df_heat_local['date'])

    # --- Scatter (most recent point per ticker) ---
    df_heat_latest = df_heat_local.sort_values('date').groupby('ticker').tail(1).reset_index(drop=True)
    df_heat_latest_date = df_heat_local['date'].max().strftime("%Y-%m-%d")

    st.subheader("🧠 Price & Volume Analysis")
    st.caption(f"Data as of {df_heat_latest_date}")

    # Ensure numeric types for plotting (coerce bad values -> NaN)
    df_heat_latest['PriceFactor'] = pd.to_numeric(df_heat_latest['PriceFactor'], errors='coerce')
    df_heat_latest['VolumeFactor'] = pd.to_numeric(df_heat_latest['VolumeFactor'], errors='coerce')

    # Drop rows missing x or y (can't plot)
    n_before = len(df_heat_latest)
    df_heat_latest = df_heat_latest.dropna(subset=['PriceFactor', 'VolumeFactor'])
    if df_heat_latest.empty:
        st.warning("No valid scatter data (PriceFactor/VolumeFactor missing or non-numeric).")
        return
    if len(df_heat_latest) < n_before:
        st.info(f"Dropped {n_before - len(df_heat_latest)} invalid latest rows with missing PriceFactor/VolumeFactor.")

    # Build robust customdata for each point: [date_str, ticker, code, PriceFactor, VolumeFactor]
    customdata = [
        [r['date'].strftime("%Y-%m-%d"), r['ticker'], str(r.get('code', '')), float(r['PriceFactor']), float(r['VolumeFactor'])]
        for _, r in df_heat_latest.iterrows()
    ]

    # Use your original scatter creation but attach customdata
    fig = px.scatter(
        df_heat_latest,
        x='VolumeFactor',
        y='PriceFactor',
        color='code',
        custom_data=['date', 'ticker', 'PriceFactor', 'VolumeFactor'],
        height=550,
    )
    fig.update_traces(
        marker=dict(size=14, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"      # ticker
            "<i>%{customdata[0]}</i><br>"      # date
            "Price Factor: %{customdata[2]:.2f}<br>"
            "Volume Factor: %{customdata[3]:.2f}<extra></extra>"
        ),
        customdata=customdata  # attach the robust customdata used for event parsing
    )
    fig.update_layout(
        xaxis_title="Volume Factor",
        yaxis_title="Price Factor",
        hovermode='closest',
        template='plotly_white',
    )

    # Render scatter (guarantee it renders first) and then capture events
    st.plotly_chart(fig, use_container_width=True)

    # Capture click / select events (single click OR box/lasso)
    events = plotly_events(fig, click_event=True, select_event=True, override_height=560, key="scatter_events")

    # Parse selected tickers from events
    selected_tickers = []
    if events:
        for ev in events:
            cd = ev.get("customdata")
            if cd and isinstance(cd, (list, tuple)) and len(cd) >= 2:
                selected_tickers.append(cd[1])
            else:
                pn = ev.get("pointNumber")
                if pn is not None and 0 <= pn < len(df_heat_latest):
                    selected_tickers.append(df_heat_latest.iloc[int(pn)]['ticker'])

    # Reset selection UI + show current selection
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Reset Selection"):
            selected_tickers = []
    with col_b:
        if selected_tickers:
            st.info(f"Filtering heatmaps to: {', '.join(sorted(set(selected_tickers)))}")
        else:
            st.write("No selection — showing all tickers")

    # --- Filter heatmap data based on selection (if any) ---
    if selected_tickers:
        df_for_heat = df_heat_local[df_heat_local['ticker'].isin(selected_tickers)].copy()
    else:
        df_for_heat = df_heat_local.copy()

    # Rebuild ticker order grouped by code, using the filtered dataset so rows align with selection
    if df_for_heat.empty:
        st.warning("No heatmap data available for selected tickers.")
        return

    ticker_order_df = df_for_heat.groupby(['ticker', 'code'])['date'].min().reset_index().sort_values(['code', 'ticker'])
    ticker_list = ticker_order_df['ticker'].tolist()

    # Ensure dates are sorted (oldest -> newest) using the filtered dataset
    dates_sorted = sorted(df_for_heat['date'].unique())

    # Build pivot matrices using filtered data and consistent ordering
    vol_pivot = df_for_heat.pivot_table(index='ticker', columns='date', values='VolumeFactor', aggfunc='last').reindex(index=ticker_list, columns=dates_sorted)
    price_pivot = df_for_heat.pivot_table(index='ticker', columns='date', values='PriceFactor', aggfunc='last').reindex(index=ticker_list, columns=dates_sorted)

    # Build small map ticker -> code for customdata
    code_map = df_for_heat.groupby('ticker')['code'].first().to_dict()

    # If there is no data, warn and stop
    if vol_pivot.empty or price_pivot.empty:
        st.warning("No heatmap data available.")
        return

    # Precompute customdata arrays (repeat code per column so hover shows code)
    vol_customdata = [[code_map.get(t, "")] * vol_pivot.shape[1] for t in vol_pivot.index]
    price_customdata = [[code_map.get(t, "")] * price_pivot.shape[1] for t in price_pivot.index]

    # Format x labels as strings
    x_labels = [d.strftime("%Y-%m-%d") for d in vol_pivot.columns]

    # Plot heatmaps side-by-side in two columns
    col1, col2 = st.columns(2)

    with col1:
        fig_vol = go.Figure(
            data=go.Heatmap(
                z=vol_pivot.values,
                x=x_labels,
                y=vol_pivot.index.tolist(),
                colorscale='Viridis',
                colorbar=dict(title='VolumeFactor'),
                hovertemplate=(
                    "Ticker: %{y}<br>"
                    "Date: %{x}<br>"
                    "Volumefactor: %{z:.4f}<br>"
                    "Code: %{customdata}<extra></extra>"
                ),
                customdata=vol_customdata
            )
        )
        fig_vol.update_layout(height=520, margin=dict(t=40, b=40), title="Volume Factor")
        fig_vol.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        fig_price = go.Figure(
            data=go.Heatmap(
                z=price_pivot.values,
                x=x_labels,
                y=price_pivot.index.tolist(),
                colorscale='Plasma',
                colorbar=dict(title='PriceFactor'),
                hovertemplate=(
                    "Ticker: %{y}<br>"
                    "Date: %{x}<br>"
                    "Pricefactor: %{z:.4f}<br>"
                    "Code: %{customdata}<extra></extra>"
                ),
                customdata=price_customdata
            )
        )
        fig_price.update_layout(height=520, margin=dict(t=40, b=40), title="Price Factor")
        fig_price.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_price, use_container_width=True)

    # Open chart UI (modal or sidebar) if the user clicked a 📈 cell
    if 'selected_chart_ticker' in locals() and selected_chart_ticker:
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
