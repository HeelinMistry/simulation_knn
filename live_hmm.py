"""
live_hmm.py  -  live test of the frozen pooled HMM on Binance candles
─────────────────────────────────────────────────────────────────────
Reuses pooled_hmm_pipeline.py (frozen model, manifest check, forward filter,
min-dwell) and data/preprocessing.py (the feature registry), so a live label is
produced by the SAME code path as a research label.

Flow per run
  1. GET /api/v3/uiKlines (XRPUSDT, 4h, up to 1000 candles).
  2. Drop the still-forming candle (close_time >= server time). Only CLOSED
     candles are ever labelled.
  3. Map to the raw-master schema -> compute_features() on the whole window
     (1000 candles is enough: the longest window is 250).
  4. Forward-filter from the start of the window (the filter forgets its
     initial state within a few dozen candles) -> State / State_Conf / State_HC.
  5. Append new closed candles to a prediction log (CSV).

Modes
  python live_hmm.py                 # label the latest closed candle once
  python live_hmm.py --watch         # loop: wake after each 4h close, label, log
  python live_hmm.py --backfill 200  # log the last 200 closed candles (causal labels)
  python live_hmm.py --verify        # compare live-computed features vs your processed master
  python live_hmm.py --score         # realised forward returns by state, from the log

Notes
  * The HMM was fit on pooled coins; this runs ONE coin (XRPUSDT) live. --score
    therefore gives descriptive numbers, not the panel inference of the research
    script. With few candles the noise is large, so don't read much into it.
  * Live features differ from master features only through path-dependent EMAs
    (MACD) in the first candles of the window; the rolling features are exact.
    --verify shows the size of that difference on the overlap.
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from pooled_hmm_pipeline import (
    Config, _tag, _paths, _check_manifest, forward_filter, apply_min_dwell,
    CANDLE, VOL_WIN, VOL_MIN, HORIZONS, OUT_DIR,
)
from data.preprocessing import compute_features, RAW_KEEP

BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
LIMIT = 1000
LOG_DIR = os.path.join(OUT_DIR, "live")
KLINE_COLS = ["open_ms", "Open", "High", "Low", "Close", "Volume", "close_ms",
              "Quote_asset_volume", "Num_trades", "Taker_buy_base_vol",
              "Taker_buy_quote_vol", "Ignore"]


# ── Binance ───────────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None):
    url = BASE_URL + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "live-hmm/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def server_time_ms() -> int:
    return int(_get("/api/v3/time")["serverTime"])


def fetch_closed_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT) -> pd.DataFrame:
    """Raw-master-shaped frame of CLOSED candles only (UTC, naive Open_time)."""
    rows = _get("/api/v3/uiKlines", {"symbol": symbol, "interval": interval, "limit": limit})
    now = server_time_ms()
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = df[df["close_ms"] < now].copy()                      # drop the forming candle
    df["Open_time"] = pd.to_datetime(df["open_ms"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in RAW_KEEP:
        if c != "Open_time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[RAW_KEEP].drop_duplicates("Open_time", keep="last").reset_index(drop=True)
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

def load_frozen(cfg: Config, model_path: str | None = None) -> dict:
    feats = cfg.features()
    path = model_path or _paths(cfg.timeframe, _tag(cfg, feats))[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"No frozen model at {path}. Run: python pooled_hmm_pipeline.py --refit")
    art = joblib.load(path)
    if list(art["feature_cols"]) != list(feats):
        raise RuntimeError(f"Model features {art['feature_cols']} != active {feats}.")
    _check_manifest(art, feats, path)       # fails loudly if feature definitions changed
    print(f"✓ frozen model {path}  (fit {art['fit_utc']}, train cutoff {art['cutoff']})")
    return art


def label_live(raw: pd.DataFrame, art: dict, min_dwell: int, thr: float) -> pd.DataFrame:
    """Features -> causal forward filter on the latest contiguous segment -> labels."""
    feats = art["feature_cols"]
    fdf = compute_features(raw, label=f"{SYMBOL}/live")
    fdf["Open_time"] = pd.to_datetime(fdf["Open_time"])
    fdf = fdf.sort_values("Open_time").reset_index(drop=True)
    seg = (fdf["Open_time"].diff() > CANDLE * 1.5).cumsum()
    fdf = fdf[seg == seg.iloc[-1]].reset_index(drop=True)     # latest gap-free run only
    if len(fdf) < 50:
        raise RuntimeError(f"Only {len(fdf)} usable candles after warm-up/gaps.")

    model, K = art["model"], art["model"].n_components
    post = forward_filter(model, fdf[feats].values)[:, art["canon_to_raw"]]
    raw_state = post.argmax(1)
    state = apply_min_dwell(raw_state, min_dwell)
    fdf["State_Raw"] = raw_state
    fdf["State"] = state
    fdf["State_Conf"] = post[np.arange(len(fdf)), state]
    fdf["State_HC"] = np.where(fdf["State_Conf"] >= thr, state, -1)
    for k in range(K):
        fdf[f"P_State_{k}"] = post[:, k]
    return fdf


def describe(row: pd.Series, K: int, thr: float) -> str:
    probs = "  ".join(f"S{k}={row[f'P_State_{k}']:.2f}" for k in range(K))
    call = f"S{int(row['State_HC'])}" if row["State_HC"] >= 0 else "no call"
    return (f"{row['Open_time']:%Y-%m-%d %H:%M} UTC  close={row['Close']:.4f}  "
            f"state=S{int(row['State'])} conf={row['State_Conf']:.2f}  "
            f"HC(>= {thr})={call}   [{probs}]")


# ── Prediction log ────────────────────────────────────────────────────────────

def log_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"{SYMBOL}_{INTERVAL}_live_log.csv")


def append_log(lab: pd.DataFrame, K: int) -> int:
    cols = ["Open_time", "Close", "State", "State_Raw", "State_Conf", "State_HC"] + \
           [f"P_State_{k}" for k in range(K)]
    p = log_path()
    old = pd.read_csv(p, parse_dates=["Open_time"]) if os.path.exists(p) else None
    new = lab[cols] if old is None else lab.loc[~lab["Open_time"].isin(old["Open_time"]), cols]
    if new.empty:
        return 0
    out = new if old is None else pd.concat([old, new], ignore_index=True)
    out.sort_values("Open_time").to_csv(p, index=False)
    return len(new)


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_once(cfg, art, backfill: int = 0) -> pd.DataFrame:
    lab = label_live(fetch_closed_klines(), art, cfg.min_dwell, cfg.conf_threshold)
    K = art["model"].n_components
    tail = lab.tail(max(backfill, 1))
    print(f"\n{describe(lab.iloc[-1], K, cfg.conf_threshold)}")
    n = append_log(tail, K)
    print(f"logged {n} new candle(s) → {log_path()}")
    return lab


def watch(cfg, art, delay_s: int = 8) -> None:
    """Sleep until just after each 4h boundary, then label the newly closed candle."""
    step_ms = int(CANDLE.total_seconds() * 1000)
    last_seen = None
    while True:
        try:
            lab = run_once(cfg, art)
            last_seen = lab["Open_time"].iloc[-1]
        except Exception as exc:                              # network hiccup: retry shortly
            print(f"⚠ {type(exc).__name__}: {exc}  (retrying in 30s)")
            time.sleep(30)
            continue
        now = server_time_ms()
        next_close = (now // step_ms + 1) * step_ms
        wait = (next_close - now) / 1000 + delay_s
        print(f"next close {datetime.fromtimestamp(next_close / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC "
              f"(sleeping {wait / 60:.0f} min)")
        time.sleep(max(wait, 1))
        for _ in range(10):                                   # wait until the new candle is listed
            try:
                latest = fetch_closed_klines(limit=3)["Open_time"].iloc[-1]
                if pd.Timestamp(latest) > last_seen:
                    break
            except Exception:
                pass
            time.sleep(5)


def verify(cfg, art) -> None:
    """Live-computed features vs the cached processed master on the overlapping candles."""
    from data.data_manager import update_master_data
    feats = art["feature_cols"]
    lab = label_live(fetch_closed_klines(), art, cfg.min_dwell, cfg.conf_threshold)
    ref = update_master_data(cfg.timeframe, SYMBOL, frozen=True)
    ref["Open_time"] = pd.to_datetime(ref["Open_time"])
    m = lab.merge(ref[["Open_time"] + feats], on="Open_time", suffixes=("", "_ref"))
    if m.empty:
        print("No overlap between live window and processed master — run a data refresh first.")
        return
    print(f"\noverlap: {len(m)} candles  {m['Open_time'].min()} → {m['Open_time'].max()}")
    rows = []
    for f in feats:
        d = (m[f] - m[f + "_ref"]).abs()
        rows.append({"feature": f, "max_abs_diff": d.max(), "mean_abs_diff": d.mean(),
                     "max_diff_last_100": d.tail(100).max()})
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    print("\nSmall (<~1e-3) is expected. Persistent large diffs mean the live raw columns differ "
          "from your raw master (e.g. a unit or column mismatch).")


def score(cfg, art) -> None:
    """Realised forward return by logged state (sigma-normalised, XRPUSDT only)."""
    p = log_path()
    if not os.path.exists(p):
        print("No log yet — run --backfill N or --watch first.")
        return
    lg = pd.read_csv(p, parse_dates=["Open_time"])
    raw = fetch_closed_klines()
    px = raw.assign(Open_time=pd.to_datetime(raw["Open_time"])).set_index("Open_time")["Close"]
    r1 = np.log(px).diff()
    vol = r1.rolling(VOL_WIN, min_periods=VOL_MIN).std()
    K = art["model"].n_components
    d = lg.set_index("Open_time")
    rows = []
    for h in HORIZONS:
        fwd = np.log(px.shift(-h) / px)
        ok = (px.index.to_series().shift(-h) - px.index.to_series()) == CANDLE * h
        fwd, z = fwd.where(ok), fwd.where(ok) / (vol * np.sqrt(h))
        j = d.join(pd.DataFrame({"fwd": fwd, "z": z}), how="left").dropna(subset=["z"])
        if j.empty:
            continue
        base_z, base_hit = z.mean(), (fwd.dropna() > 0).mean()   # all candles in the fetched window
        for name, col in (("all", "State"), (f"HC>={cfg.conf_threshold}", "State_HC")):
            for k in range(K):
                g = j[j[col] == k]
                if g.empty:
                    continue
                rows.append({"h": h, "basis": name, "state": k, "n": len(g),
                             "mean_bps": g["fwd"].mean() * 1e4,
                             "mean_z": g["z"].mean(), "excess_z": g["z"].mean() - base_z,
                             "hit": (g["fwd"] > 0).mean(), "hit_excess": (g["fwd"] > 0).mean() - base_hit})
    if not rows:
        print("Not enough elapsed candles to score any horizon yet.")
        return
    print(f"\nLOGGED CALLS: {len(lg)}  ({lg['Open_time'].min()} → {lg['Open_time'].max()})")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nDescriptive only: one coin, overlapping horizons (n is not independent at h>1), no "
          "multiple-testing control. Compare the sign/ordering of excess_z by state with your OOS "
          "table from pooled_hmm_pipeline.py rather than reading the p-value-free numbers alone.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--backfill", type=int, default=0, help="log the last N closed candles")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", default=None, help="path to a frozen .joblib (default: derived from config)")
    ap.add_argument("--conf-threshold", type=float, default=0.75)
    ap.add_argument("--min-dwell", type=int, default=1)
    ap.add_argument("--cov-type", choices=["diag", "full"], default="diag")
    ap.add_argument("--sticky", type=float, default=0.0)
    a = ap.parse_args()

    cfg = Config(conf_threshold=a.conf_threshold, min_dwell=a.min_dwell,
                 cov_type=a.cov_type, sticky=a.sticky)
    art = load_frozen(cfg, a.model)
    if a.verify:
        verify(cfg, art)
    elif a.score:
        score(cfg, art)
    elif a.watch:
        watch(cfg, art)
    else:
        run_once(cfg, art, backfill=a.backfill)


if __name__ == "__main__":
    main()