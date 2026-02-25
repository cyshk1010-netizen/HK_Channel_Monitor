import os
import pandas as pd
import streamlit as st

OUTDIR = "outputs"
CSV_SIGNALS = os.path.join(OUTDIR, "signals_latest.csv")
CSV_OPPS = os.path.join(OUTDIR, "opportunities_latest.csv")

st.set_page_config(page_title="HK Opportunity Scanner", layout="wide")

st.title("HK Opportunity Scanner")
st.caption("BUY/SELL opportunities ranked by score (Entry = Close, Stop = ATR-based)")

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---- Sidebar controls
st.sidebar.header("View")
view = st.sidebar.selectbox("Dataset", ["Opportunities", "Signals (All)"], index=0)

search = st.sidebar.text_input("Search ticker (optional)", value="").strip().upper()

if view == "Opportunities":
    st.sidebar.subheader("Opportunity Filters")
    min_score = st.sidebar.slider("Min score", 0, 100, 50, 1)
    action_filter = st.sidebar.multiselect("Action", ["BUY", "SELL"], default=["BUY", "SELL"])
    top_n = st.sidebar.number_input("Top N (Today)", min_value=1, max_value=100, value=10, step=1)
else:
    min_score = 0
    action_filter = ["BUY", "SELL"]
    top_n = 10

# ---- Load data
if view == "Opportunities":
    df = load_csv(CSV_OPPS)
    if df.empty:
        st.warning("opportunities_latest.csv not found or empty. Run: python3 -u scan_daily.py")
        st.stop()

    # types
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    for c in ["entry", "stop", "t1", "t2", "t3", "close", "atr14"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # filters
    if "score" in df.columns:
        df = df[df["score"].fillna(-1) >= min_score]
    if "action" in df.columns:
        df = df[df["action"].isin(action_filter)]
    if search and "ticker" in df.columns:
        df = df[df["ticker"].astype(str).str.upper().str.contains(search)]

    # sort
    if "score" in df.columns:
        df = df.sort_values(["score", "ticker"], ascending=[False, True])

    # ---------- TOP N TODAY + EXPORT ----------
    st.subheader("Top Opportunities Today")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows (filtered)", len(df))
    with c2:
        st.metric("BUY", int((df["action"] == "BUY").sum()) if "action" in df.columns else 0)
    with c3:
        st.metric("SELL", int((df["action"] == "SELL").sum()) if "action" in df.columns else 0)
    with c4:
        st.metric("Min score", min_score)

    top_df = df.head(int(top_n)).copy()

    # Friendly columns for team share
    preferred_cols = [
        "date", "ticker", "action", "setup", "score",
        "entry", "stop", "t1", "t2", "t3", "reason", "atr14"
    ]
    cols = [c for c in preferred_cols if c in top_df.columns]
    if cols:
        top_df = top_df[cols + [c for c in top_df.columns if c not in cols]]

    st.dataframe(top_df, width="stretch", height=360, hide_index=True)

    # Download filtered or Top N
    dcol1, dcol2 = st.columns([1, 1])
    with dcol1:
        st.download_button(
            label="Download TOP N as CSV",
            data=to_csv_bytes(top_df),
            file_name="top_opportunities_today.csv",
            mime="text/csv",
        )
    with dcol2:
        st.download_button(
            label="Download FILTERED list as CSV",
            data=to_csv_bytes(df),
            file_name="opportunities_filtered.csv",
            mime="text/csv",
        )

    st.divider()

    # ---------- FULL TABLE ----------
    st.subheader("Opportunities (Filtered)")
    st.dataframe(df, width="stretch", height=420, hide_index=True)

    # ---------- CHART & TRADE PLAN ----------
    st.divider()
    st.subheader("Chart & Trade Plan")

    tickers = df["ticker"].astype(str).tolist() if "ticker" in df.columns else []
    if not tickers:
        st.info("No rows match your filters.")
        st.stop()

    selected = st.selectbox("Select a ticker", tickers, index=0)
    row = df[df["ticker"] == selected].iloc[0].to_dict()

    cA, cB = st.columns([1, 1])

    with cA:
        st.markdown("### Trade Plan")
        st.write(
            {
                "Ticker": row.get("ticker", ""),
                "Action": row.get("action", ""),
                "Setup": row.get("setup", ""),
                "Score": int(row.get("score", 0)) if pd.notna(row.get("score", None)) else "",
                "Entry (Close)": row.get("entry", ""),
                "Stop (ATR)": row.get("stop", ""),
                "Target 1": row.get("t1", ""),
                "Target 2": row.get("t2", ""),
                "Target 3": row.get("t3", ""),
                "Reason": row.get("reason", ""),
            }
        )

    with cB:
        st.markdown("### Chart")
        chart_path = row.get("chart", "")
        if isinstance(chart_path, str) and chart_path and os.path.exists(chart_path):
            st.image(chart_path, width="stretch")
        else:
            st.info("Chart file not found for this ticker (or not generated yet).")

else:
    df = load_csv(CSV_SIGNALS)
    if df.empty:
        st.warning("signals_latest.csv not found or empty. Run: python3 -u scan_daily.py")
        st.stop()

    if search and "ticker" in df.columns:
        df = df[df["ticker"].astype(str).str.upper().str.contains(search)]

    st.subheader("Signals (All)")
    st.dataframe(df, width="stretch", height=650, hide_index=True)

    st.download_button(
        label="Download Signals as CSV",
        data=to_csv_bytes(df),
        file_name="signals_latest.csv",
        mime="text/csv",
    )