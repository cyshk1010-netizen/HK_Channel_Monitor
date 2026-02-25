from __future__ import annotations

import os
import math
import datetime as dt
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# =========================
# SETTINGS (edit these later)
# =========================
DATA_YEARS = 3

# "Loose" proximity threshold (1%)
NEAR_PCT = 0.01

# Moving average regime
MA_FAST = 50
MA_SLOW = 250
MA_CROSS_LOOKBACK_DAYS = 20  # trading days

# 52-week low
LOW_52W_WINDOW = 252

# RSI settings
RSI_PERIODS = [6, 12, 24]
PIVOT_L = 3
PIVOT_R = 3
MIN_SWING_PCT = 0.03  # 3% swing requirement for divergence to reduce noise

# Volume spike
VOL_AVG_WINDOW = 20
VOL_SPIKE_RATIO = 2.0

# ATR-based stop
ATR_PERIOD = 14
ATR_MULT_STOP = 1.0

# Opportunity threshold
MIN_SCORE_TO_LIST = 20   # <- testing friendly; later increase to 50/60

# Bulk download chunk size (important for 80–100 tickers)
DOWNLOAD_CHUNK = 20

# Output folders
OUTDIR = "outputs"
CHART_DIR = os.path.join(OUTDIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LEVEL_NAME_ORDER = ["Lower", "LowerMid", "Mid", "UpperMid", "Upper"]


# =========================
# IO
# =========================
def normalize_ticker(t: str) -> str:
    t = t.strip()
    if not t or t.startswith("#"):
        return ""
    if t.lower().endswith(".hk"):
        return t[:-3].upper() + ".HK"
    t_up = t.upper()
    if t_up.endswith(".HK"):
        return t_up
    if t.isdigit():
        return f"{t.zfill(4)}.HK"
    return t_up

def read_tickers(path="tickers.txt") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        tickers = [normalize_ticker(x) for x in f.readlines()]
    tickers = [t for t in tickers if t]
    # de-dupe preserve order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# =========================
# INDICATORS / HELPERS
# =========================
def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def find_pivots(df: pd.DataFrame, left: int, right: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    highs = df["High"].values
    lows = df["Low"].values
    ph, pl = [], []
    for i in range(left, len(df) - right):
        if highs[i] == np.max(highs[i-left:i+right+1]):
            ph.append((i, float(highs[i])))
        if lows[i] == np.min(lows[i-left:i+right+1]):
            pl.append((i, float(lows[i])))
    return ph, pl

def fit_line(points: List[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
    if len(points) < 2:
        return None
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    b = np.polyfit(x, y, 1)[0]
    a = y.mean() - b * x.mean()
    return a, b

def build_parallel_channel(df: pd.DataFrame) -> Optional[Tuple[Dict[str, np.ndarray], bool]]:
    n = len(df)
    x = np.arange(n, dtype=float)
    close = df["Close"].values

    b_trend = np.polyfit(x, close, 1)[0]
    uptrend = b_trend >= 0

    ph, pl = find_pivots(df, PIVOT_L, PIVOT_R)
    anchor = pl if uptrend else ph
    other = ph if uptrend else pl

    cutoff = max(0, n - 180)
    anchor = [p for p in anchor if p[0] >= cutoff]
    other = [p for p in other if p[0] >= cutoff]

    if len(anchor) < 2 or len(other) < 2:
        return None

    line = fit_line(anchor)
    if line is None:
        return None
    a, b = line
    base = a + b * x

    other_x = np.array([p[0] for p in other], dtype=int)
    other_y = np.array([p[1] for p in other], dtype=float)
    base_at_other = a + b * other_x
    deltas = other_y - base_at_other
    offset = float(np.max(deltas) if uptrend else np.min(deltas))

    if uptrend:
        lower = base
        upper = base + offset
    else:
        upper = base
        lower = base + offset

    mid = (lower + upper) / 2.0
    lower_mid = (lower + mid) / 2.0
    upper_mid = (mid + upper) / 2.0

    lines = {"Lower": lower, "LowerMid": lower_mid, "Mid": mid, "UpperMid": upper_mid, "Upper": upper}
    return lines, uptrend

def near_pct(a: float, b: float, pct: float) -> bool:
    return abs(a - b) / max(1e-12, abs(b)) <= pct

def ma_regime_and_cross(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[int], float, float]:
    close = df["Close"]
    ma_fast = close.rolling(MA_FAST).mean()
    ma_slow = close.rolling(MA_SLOW).mean()

    f = float(ma_fast.iloc[-1])
    s = float(ma_slow.iloc[-1])
    regime = "Bull" if f > s else "Bear" if f < s else "Flat"

    diff = (ma_fast - ma_slow).dropna()
    cross_type = None
    days_ago = None
    if len(diff) > MA_CROSS_LOOKBACK_DAYS + 2:
        tail = diff.iloc[-(MA_CROSS_LOOKBACK_DAYS + 2):]
        sign = np.sign(tail.values)
        for k in range(len(sign) - 1, 0, -1):
            if sign[k] == 0 or sign[k-1] == 0:
                continue
            if sign[k] != sign[k-1]:
                days_ago = (len(sign) - 1) - k
                cross_type = "Golden" if sign[k] > sign[k-1] else "Death"
                break
    return regime, cross_type, days_ago, f, s

def low_52w_info(df: pd.DataFrame) -> Tuple[float, float, str]:
    win = df.tail(LOW_52W_WINDOW)
    low_52w = float(win["Low"].min())
    close = float(df["Close"].iloc[-1])
    dist = (close - low_52w) / max(1e-12, low_52w)
    bucket = ""
    if dist <= 0.10:
        bucket = "<=10%"
    elif dist <= 0.20:
        bucket = "<=20%"
    return low_52w, dist, bucket

def rsi_divergence_flags(df: pd.DataFrame) -> List[str]:
    ph, pl = find_pivots(df, PIVOT_L, PIVOT_R)
    if len(ph) < 2 or len(pl) < 2:
        return []
    hits = []
    for p in RSI_PERIODS:
        r = compute_rsi(df["Close"], p)

        (i1, p1), (i2, p2) = pl[-2], pl[-1]
        swing = abs(p2 - p1) / max(1e-12, p1)
        if (p2 < p1) and (float(r.iloc[i2]) > float(r.iloc[i1])) and swing >= MIN_SWING_PCT:
            hits.append(f"Bullish RSI{p}")

        (j1, h1), (j2, h2) = ph[-2], ph[-1]
        swingh = abs(h2 - h1) / max(1e-12, h1)
        if (h2 > h1) and (float(r.iloc[j2]) < float(r.iloc[j1])) and swingh >= MIN_SWING_PCT:
            hits.append(f"Bearish RSI{p}")

    return hits


# =========================
# OPPORTUNITY LOGIC
# =========================
def classify_opportunity(
    close: float,
    regime: str,
    cross_type: Optional[str],
    cross_days_ago: Optional[int],
    bucket52: str,
    div_hits: List[str],
    vol_spike: bool,
    lines: Dict[str, np.ndarray],
    atr: float
) -> Optional[Dict[str, object]]:
    L = float(lines["Lower"][-1])
    LM = float(lines["LowerMid"][-1])
    M = float(lines["Mid"][-1])
    UM = float(lines["UpperMid"][-1])
    U = float(lines["Upper"][-1])

    near_lower = near_pct(close, L, NEAR_PCT) or near_pct(close, LM, NEAR_PCT)
    near_upper = near_pct(close, U, NEAR_PCT) or near_pct(close, UM, NEAR_PCT)

    breakout = close > U * 1.002
    breakdown = close < L * 0.998

    has_div = len(div_hits) > 0
    has_ma_cross = (cross_type is not None and cross_days_ago is not None and cross_days_ago <= MA_CROSS_LOOKBACK_DAYS)

    def base_score(direction: str) -> int:
        s = 0
        if direction == "BUY":
            s += 15 if regime == "Bull" else -15 if regime == "Bear" else 0
            if has_ma_cross and cross_type == "Golden": s += 10
            if bucket52: s += 10
            if has_div and any("bullish" in d.lower() for d in div_hits): s += 15
        else:
            s += 15 if regime == "Bear" else -15 if regime == "Bull" else 0
            if has_ma_cross and cross_type == "Death": s += 10
            if has_div and any("bearish" in d.lower() for d in div_hits): s += 15
        if vol_spike: s += 10
        return s

    # A) BUY reversal near Lower/LowerMid
    if near_lower and (has_div or vol_spike or bucket52):
        score = 35 + base_score("BUY")
        return {
            "action": "BUY", "setup": "A", "score": score,
            "entry": close, "stop": L - ATR_MULT_STOP * atr,
            "t1": M, "t2": UM, "t3": U,
            "reason": "Reversal near channel support"
        }

    # B) SELL reversal near Upper/UpperMid
    if near_upper and (has_div or vol_spike):
        score = 35 + base_score("SELL")
        return {
            "action": "SELL", "setup": "B", "score": score,
            "entry": close, "stop": U + ATR_MULT_STOP * atr,
            "t1": M, "t2": LM, "t3": L,
            "reason": "Reversal near channel resistance"
        }

    # C) BUY breakout above Upper
    if breakout:
        score = 30 + base_score("BUY")
        height = U - L
        return {
            "action": "BUY", "setup": "C", "score": score,
            "entry": close, "stop": U - ATR_MULT_STOP * atr,
            "t1": close + 0.5 * height, "t2": close + 1.0 * height, "t3": close + 1.5 * height,
            "reason": "Breakout above channel upper"
        }

    # D) SELL breakdown below Lower
    if breakdown:
        score = 30 + base_score("SELL")
        height = U - L
        return {
            "action": "SELL", "setup": "D", "score": score,
            "entry": close, "stop": L + ATR_MULT_STOP * atr,
            "t1": close - 0.5 * height, "t2": close - 1.0 * height, "t3": close - 1.5 * height,
            "reason": "Breakdown below channel lower"
        }

    return None


# =========================
# PLOTTING
# =========================
def plot_chart(ticker: str, df: pd.DataFrame, lines: Dict[str, np.ndarray], outpath: str):
    view = df.tail(260)
    n0 = len(df) - len(view)

    plt.figure(figsize=(12, 6))
    plt.plot(view.index, view["Close"].values, label="Close")
    for name in LEVEL_NAME_ORDER:
        plt.plot(view.index, lines[name][n0:], label=name)
    plt.title(f"{ticker} - Channel + Midlines")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# =========================
# BULK DOWNLOAD (CHUNKED)
# =========================
def download_chunk(tickers: List[str], start: dt.date, end: dt.date) -> Dict[str, pd.DataFrame]:
    joined = " ".join(tickers)
    df = yf.download(
        joined,
        start=str(start),
        end=str(end + dt.timedelta(days=1)),
        progress=False,
        group_by="ticker",
        threads=True,
        auto_adjust=False,
    )
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t in df.columns.get_level_values(0):
                sub = df[t].copy()
                sub.columns = [c.title() for c in sub.columns]
                needed = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in sub.columns for c in needed):
                    out[t] = sub[needed].dropna()
    else:
        # single ticker fallback
        sub = df.copy()
        sub.columns = [c.title() for c in sub.columns]
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if all(c in sub.columns for c in needed):
            out[tickers[0]] = sub[needed].dropna()

    return out


# =========================
# MAIN
# =========================
def main():
    tickers = read_tickers()
    if not tickers:
        print("tickers.txt is empty.")
        return

    today = dt.date.today()
    start = today - dt.timedelta(days=365 * DATA_YEARS)

    rows = []
    opps = []

    print(f"Scanning {len(tickers)} tickers... (chunk={DOWNLOAD_CHUNK})", flush=True)

    for i in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[i:i + DOWNLOAD_CHUNK]
        print(f"Downloading {i+1}-{i+len(chunk)} / {len(tickers)} ...", flush=True)

        data_map = download_chunk(chunk, start, today)

        for t in chunk:
            df = data_map.get(t)
            if df is None or df.empty or len(df) < (MA_SLOW + 30):
                continue

            close = float(df["Close"].iloc[-1])

            ch_pack = build_parallel_channel(df)
            if ch_pack is None:
                continue
            lines, uptrend = ch_pack

            atr = float(compute_atr(df, ATR_PERIOD).iloc[-1])
            if not np.isfinite(atr) or atr <= 0:
                continue

            regime, cross_type, cross_days_ago, ma50, ma250 = ma_regime_and_cross(df)
            low52, dist52, bucket52 = low_52w_info(df)

            div_hits = rsi_divergence_flags(df)

            vol_avg = float(df["Volume"].rolling(VOL_AVG_WINDOW).mean().iloc[-1])
            vol_ratio = float(df["Volume"].iloc[-1] / max(1.0, vol_avg))
            vol_spike = vol_ratio >= VOL_SPIKE_RATIO

            # Save chart
            chart_path = os.path.join(CHART_DIR, f"{t}_{today.isoformat()}.png")
            plot_chart(t, df, lines, chart_path)

            rows.append({
                "date": today.isoformat(),
                "ticker": t,
                "close": close,
                "regime": regime,
                "ma50": ma50,
                "ma250": ma250,
                "ma_cross_type": cross_type or "",
                "ma_cross_days_ago": "" if cross_days_ago is None else cross_days_ago,
                "low_52w": low52,
                "dist_from_52w_low_pct": dist52 * 100.0,
                "near_52w_low_bucket": bucket52,
                "vol_ratio_20": vol_ratio,
                "vol_spike": bool(vol_spike),
                "rsi_divergences": ", ".join(div_hits),
                "channel_upper": float(lines["Upper"][-1]),
                "channel_upper_mid": float(lines["UpperMid"][-1]),
                "channel_mid": float(lines["Mid"][-1]),
                "channel_lower_mid": float(lines["LowerMid"][-1]),
                "channel_lower": float(lines["Lower"][-1]),
                "atr14": atr,
                "chart": chart_path,
            })

            opp = classify_opportunity(
                close=close,
                regime=regime,
                cross_type=cross_type,
                cross_days_ago=cross_days_ago,
                bucket52=bucket52,
                div_hits=div_hits,
                vol_spike=vol_spike,
                lines=lines,
                atr=atr
            )

            if opp is not None and int(opp["score"]) >= MIN_SCORE_TO_LIST:
                opps.append({
                    "date": today.isoformat(),
                    "ticker": t,
                    "action": opp["action"],
                    "setup": opp["setup"],
                    "score": int(opp["score"]),
                    "entry": float(opp["entry"]),
                    "stop": float(opp["stop"]),
                    "t1": float(opp["t1"]),
                    "t2": float(opp["t2"]),
                    "t3": float(opp["t3"]),
                    "reason": opp["reason"],
                    "close": close,
                    "atr14": atr,
                    "chart": chart_path,
                })

    # Always save outputs (even if empty)
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "signals_latest.csv"), index=False)

    df_opps = pd.DataFrame(opps)
    if not df_opps.empty:
        df_opps = df_opps.sort_values(["score", "ticker"], ascending=[False, True])
    df_opps.to_csv(os.path.join(OUTDIR, "opportunities_latest.csv"), index=False)

    print("Saved:", flush=True)
    print(" - outputs/signals_latest.csv", flush=True)
    print(" - outputs/opportunities_latest.csv", flush=True)
    print(" - outputs/charts/*.png", flush=True)
    if df_opps.empty:
        print("NOTE: opportunities_latest.csv is empty today (no setups met the rules).", flush=True)
    else:
        print(f"NOTE: {len(df_opps)} opportunities found.", flush=True)


if __name__ == "__main__":
    main()