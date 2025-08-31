# --- Render by group ---
for group_name, tickers in latest.groupby("group").groups.items():
    st.header(f"📌 {group_name}")
    rows = []
    for ticker in tickers:
        row = latest.loc[ticker]
        series = rs_lastN.loc[rs_lastN["ticker"]==ticker, "rs_to_spy"].tolist()
        spark = sparkbar_img(series)

        rows.append({
            "Ticker": ticker,
            "RS Sparkline": spark,
            "RS Rank (21D)": f"{int(round(row['rs_rank_21d']))}%" if pd.notnull(row['rs_rank_21d']) else "",
            "RS Rank (252D)": f"{int(round(row['rs_rank_252d']))}%" if pd.notnull(row['rs_rank_252d']) else "",

            " ": "",  # small spacer between rank & performance

            "1D": f"{row['ret_1d']*100:.1f}%" if pd.notnull(row['ret_1d']) else "",
            "1W": f"{row['ret_1w']*100:.1f}%" if pd.notnull(row['ret_1w']) else "",
            "1M": f"{row['ret_1m']*100:.1f}%" if pd.notnull(row['ret_1m']) else "",

            "  ": "",  # small spacer between performance & SMA

            "SMA5": "✅" if row.get("above_sma5") else "❌",
            "SMA10": "✅" if row.get("above_sma10") else "❌",
            "SMA20": "✅" if row.get("above_sma20") else "❌",
        })
    disp = pd.DataFrame(rows)
    st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)