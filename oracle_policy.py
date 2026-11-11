"""
oracle_policy.py

Hindsight-oracle supervised trading policy, pooled across coins and timeframes.

Pipeline (one section per step of the plan)
───────────────────────────────────────────
 1. ORACLE      Bellman backward induction over a 2-state machine (Flat / In
                position) that maximises net terminal log-wealth after fee +
                slippage on every entry and exit. Optional holding penalty
                (--hold-penalty) makes the oracle prefer shorter, higher-
                conviction holds. The DP also returns the per-candle decision
                MARGIN (how much value is lost by taking the wrong action).
 2. STATE       The policy sees Position_State (0/1) as a feature. The oracle
                yields the optimal action for BOTH hypothetical states at every
                candle, so every candle produces two training rows and no
                recursive rollout is needed.
 3. POOLING     All <symbol>/<timeframe> feature masters are stacked into one
                training frame (features are unit-free and per-asset scaled).
 4. MODEL       LightGBM multiclass, CUDA if the build supports it (falls back
                to CPU). Row weight = inverse-frequency class weight
                (--weight-power) x oracle-margin weight (--margin-weight).
 5. SIMULATION  Cost-aware backtest of the policy. Decision thresholds are
                either ABSOLUTE probabilities (--th-mode abs, tuned on a
                chronological validation split) or RELATIVE rolling quantiles
                of the model's own recent confidence (--th-mode quantile,
                default: trade only when p is in the top 1-q of the trailing
                --q-window candles; no per-fold tuning needed).
 6. VALIDATION  Leave-one-coin-out test (all timeframes of the held-out coin)
                plus a circular-block-bootstrap null: synthetic price paths that
                keep the coin's drift and local volatility structure, run through
                the SAME features, model and threshold rule.

Label convention (3 classes, state-conditioned)
  0 = DO NOTHING (stay flat when flat, hold when in position)
  1 = BUY        (only ever the label for Position_State == 0)
  2 = CLOSE      (only ever the label for Position_State == 1)

Timing: features at candle t include candle t's close. An action is taken at
close_t and earns the return close_t -> close_{t+1}. Rolling-quantile thresholds
use only probabilities from candles BEFORE t (shifted by one), so they are causal.

Usage
  python oracle_policy.py --holdout XRPUSDT
  python oracle_policy.py --holdout XRPUSDT --th-mode quantile --q-buy 0.95 --q-close 0.95
  python oracle_policy.py --holdout XRPUSDT --th-mode abs            # old behaviour
  python oracle_policy.py --holdout XRPUSDT --margin-weight 1 --hold-penalty 0.002
  python oracle_policy.py --holdout XRPUSDT --weight-power 0.5       # or 0
  python oracle_policy.py --all-folds --n-boot 100
  python oracle_policy.py --holdout XRPUSDT --cutoff 2024-01-01   # also removes
      the time overlap between training coins and the held-out coin (recommended:
      crypto coins are strongly correlated, so same-period training leaks).

Requires: lightgbm, numpy, pandas. Optional: numba (much faster simulation).
"""

import argparse
import contextlib
import io
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.data_manager import SYMBOLS, update_master_data
from data.preprocessing import compute_features, feature_columns, feature_hashes

try:
    from numba import njit
except ImportError:                                   # pure-python fallback
    def njit(f):
        return f

TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240}
NOTHING, BUY, CLOSE = 0, 1, 2
CLASS_NAMES = ["NOTHING", "BUY", "CLOSE"]

MARGIN_FLOOR, MARGIN_CEIL = 0.05, 10.0     # normalised margin weights are clipped to this


def periods_per_year(tf: str) -> float:
    return 525_600 / TF_MINUTES[tf]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Hindsight optimal target generator
# ══════════════════════════════════════════════════════════════════════════════

@njit
def _oracle_dp(r, lc):
    """
    r  : log returns used for labelling, length n-1 (may include a holding penalty)
    lc : ln(1 - cost), paid on every entry and every exit

    V0[t] / V1[t] = best achievable future log-wealth when entering candle t
    flat / in position (before acting at close_t).
      flat : stay -> V0[t+1]              | buy   -> lc + r[t] + V1[t+1]
      held : hold -> r[t] + V1[t+1]       | close -> lc + V0[t+1]
    Terminal: V0[n-1] = 0, V1[n-1] = lc (forced liquidation on the last candle).

    Also returns m0[t] = |buy - stay| and m1[t] = |close - hold|: the log-wealth
    lost by choosing the wrong action at t (the decision margin).
    """
    n = r.shape[0] + 1
    V0 = np.zeros(n)
    V1 = np.zeros(n)
    a0 = np.zeros(n, np.int8)
    a1 = np.zeros(n, np.int8)
    m0 = np.zeros(n)
    m1 = np.zeros(n)
    V1[n - 1] = lc
    for t in range(n - 2, -1, -1):
        stay = V0[t + 1]
        buy = lc + r[t] + V1[t + 1]
        m0[t] = abs(buy - stay)
        if buy > stay:
            V0[t] = buy
            a0[t] = 1
        else:
            V0[t] = stay
        hold = r[t] + V1[t + 1]
        cls = lc + V0[t + 1]
        m1[t] = abs(cls - hold)
        if cls > hold:
            V1[t] = cls
            a1[t] = 2
        else:
            V1[t] = hold
    return a0, a1, m0, m1


def segment_ids(open_time: pd.Series, tf: str) -> np.ndarray:
    """Contiguous-data segment id; a gap > 2 candles starts a new segment."""
    ts = pd.to_datetime(open_time)
    broken = ts.diff() > pd.Timedelta(minutes=2 * TF_MINUTES[tf])
    return broken.cumsum().to_numpy()


def oracle_labels(close: np.ndarray, seg: np.ndarray, cost: float, hold_pen: float = 0.0):
    """
    Returns (a0, a1, valid, r, m0, m1):
      a0[t] optimal action if FLAT at t (0 or BUY); a1[t] if IN POSITION (0 or CLOSE)
      valid[t] False on the last candle of each segment (no next return)
      r[t]     REAL log return close_t -> close_{t+1} (0 where invalid)
      m0, m1   raw decision margins (log-wealth lost by the wrong action)
    hold_pen is a per-candle log-return penalty applied ONLY inside the oracle
    (labels); the simulator and r always use the real return.
    """
    n = len(close)
    lc = math.log1p(-cost)
    lp = np.log(close.astype(float))
    a0 = np.zeros(n, np.int8)
    a1 = np.zeros(n, np.int8)
    m0 = np.zeros(n)
    m1 = np.zeros(n)
    valid = np.zeros(n, bool)
    r = np.zeros(n)
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        if len(idx) < 3:
            continue
        rr = np.diff(lp[idx])
        x0, x1, y0, y1 = _oracle_dp(rr - hold_pen, lc)
        a0[idx], a1[idx] = x0, x1
        m0[idx], m1[idx] = y0, y1
        valid[idx[:-1]] = True
        r[idx[:-1]] = rr
    return a0, a1, valid, r, m0, m1


# ══════════════════════════════════════════════════════════════════════════════
# Steps 2 & 3 — state feature, pooled dataset
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Series:
    sym: str
    tf: str
    df: pd.DataFrame
    a0: np.ndarray
    a1: np.ndarray
    valid: np.ndarray
    r: np.ndarray
    m0: np.ndarray          # normalised, clipped decision margin if flat
    m1: np.ndarray          # normalised, clipped decision margin if in position

    @property
    def tfm(self) -> int:
        return TF_MINUTES[self.tf]

    def n(self) -> int:
        return len(self.df)


def load_all_series(symbols, timeframes, cost, hold_pen_per_day=0.0) -> list:
    out = []
    for sym in symbols:
        for tf in timeframes:
            try:
                df = update_master_data(tf, symbol=sym, frozen=True)
            except FileNotFoundError as exc:
                print(f"  ⚠ skipping {sym}/{tf}: {exc}")
                continue
            df = df.sort_values("Open_time").reset_index(drop=True)
            seg = segment_ids(df["Open_time"], tf)
            hp = hold_pen_per_day * TF_MINUTES[tf] / 1440.0
            a0, a1, valid, r, m0, m1 = oracle_labels(df["Close"].to_numpy(float), seg, cost, hp)
            # per-series normalisation so every coin/timeframe has mean margin 1
            mm = np.concatenate([m0[valid], m1[valid]])
            scale = float(mm.mean()) if mm.size and mm.mean() > 1e-12 else 1.0
            m0 = np.clip(m0 / scale, MARGIN_FLOOR, MARGIN_CEIL)
            m1 = np.clip(m1 / scale, MARGIN_FLOOR, MARGIN_CEIL)
            out.append(Series(sym, tf, df, a0, a1, valid, r, m0, m1))
            tr = (a0[valid] == BUY).mean() * 100
            print(f"  ✓ {sym}/{tf}: {len(df):,} candles, oracle flat->BUY on {tr:.1f}% of candles")
    if not out:
        raise RuntimeError("No (symbol, timeframe) series could be loaded.")
    return out


def design_matrix(df, feat_cols, tfm, state, use_tf) -> np.ndarray:
    parts = [df[feat_cols].to_numpy(np.float32)]
    if use_tf:
        parts.append(np.full((len(df), 1), math.log(tfm), np.float32))
    parts.append(np.full((len(df), 1), state, np.float32))
    return np.hstack(parts)


def design_names(feat_cols, use_tf):
    return [*feat_cols, *(["LogTF"] if use_tf else []), "Position_State"]


def stacked_xy(s: Series, lo: int, hi: int, feat_cols, use_tf):
    """Both hypothetical states for every valid candle in [lo, hi). Returns (X, y, margin)."""
    idx = np.arange(lo, hi)
    idx = idx[s.valid[idx]]
    if len(idx) == 0:
        return None, None, None
    d = s.df.iloc[idx]
    X = np.vstack([design_matrix(d, feat_cols, s.tfm, 0, use_tf),
                   design_matrix(d, feat_cols, s.tfm, 1, use_tf)])
    y = np.concatenate([s.a0[idx], s.a1[idx]]).astype(np.int32)
    m = np.concatenate([s.m0[idx], s.m1[idx]])
    return X, y, m


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — LightGBM (CUDA) multiclass policy
# ══════════════════════════════════════════════════════════════════════════════

def class_weights(y, power: float) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    return (len(y) / (3.0 * counts)) ** power


def train_policy(Xtr, ytr, mtr, Xva, yva, mva, names, device, weight_power,
                 margin_weight, rounds, lr, seed):
    """
    Row weight = class weight ** weight_power  x  oracle margin ** margin_weight.
    The margin factor is the cost-aware part of the loss: rows where the optimal
    action barely beats the alternative after fees (marginal trades) carry little
    weight, rows where the choice is worth a lot carry more.
    """
    import lightgbm as lgb
    cw = class_weights(ytr, weight_power)
    print(f"  class share (train): "
          + ", ".join(f"{CLASS_NAMES[i]} {c:.2%}" for i, c in enumerate(np.bincount(ytr, minlength=3) / len(ytr)))
          + f"  | class weights: {np.round(cw, 2).tolist()}"
          + f"  | margin-weight power {margin_weight}")
    wtr = cw[ytr] * (mtr ** margin_weight)
    wva = cw[yva] * (mva ** margin_weight)
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr)
    base = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                learning_rate=lr, num_leaves=63, min_data_in_leaf=200,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, seed=seed, verbose=-1)
    devices = ["cuda", "cpu"] if device == "auto" else [device]
    last_exc = None
    for dev in devices:
        try:
            t0 = time.time()
            model = lgb.train({**base, "device_type": dev}, dtr, num_boost_round=rounds,
                              valid_sets=[dva],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(100)])
            print(f"  ✓ LightGBM trained on device_type='{dev}' in {time.time()-t0:.0f}s, "
                  f"best iteration {model.best_iteration}")
            return model
        except Exception as exc:                      # CUDA build missing, etc.
            last_exc = exc
            print(f"  ⚠ device_type='{dev}' failed ({exc}); trying next option.")
    raise RuntimeError(f"LightGBM training failed: {last_exc}")


def predict_pair(model, df, feat_cols, tfm, use_tf):
    """(P(BUY | flat), P(CLOSE | in position)) for every candle of df."""
    it = model.best_iteration or None
    p0 = model.predict(design_matrix(df, feat_cols, tfm, 0, use_tf), num_iteration=it)
    p1 = model.predict(design_matrix(df, feat_cols, tfm, 1, use_tf), num_iteration=it)
    return p0[:, BUY].astype(float), p1[:, CLOSE].astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — thresholds (absolute or rolling-quantile), simulation, tuning
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ThSpec:
    mode: str        # "abs": value is a probability | "quantile": value is a quantile in (0,1)
    buy: float
    close: float
    window: int = 1000

    def describe(self) -> str:
        if self.mode == "quantile":
            return (f"BUY when p > rolling p{self.buy*100:g}, CLOSE when p > rolling "
                    f"p{self.close*100:g} (window {self.window})")
        return f"BUY > {self.buy:.2f}, CLOSE > {self.close:.2f}"


def rolling_quantile_threshold(p: np.ndarray, q: float, window: int) -> np.ndarray:
    """
    Causal rolling q-quantile of p over the previous `window` candles (the
    current candle is excluded via shift(1)). Until enough history exists the
    threshold is +inf, i.e. no trade is triggered during warm-up.
    """
    minp = max(50, window // 5)
    s = pd.Series(p).rolling(window, min_periods=minp).quantile(q).shift(1).to_numpy()
    return np.where(np.isnan(s), np.inf, s)


def th_array(p: np.ndarray, value: float, mode: str, window: int) -> np.ndarray:
    if mode == "quantile":
        return rolling_quantile_threshold(p, value, window)
    return np.full(len(p), value, dtype=float)


def resolve_thresholds(pb, pc, spec: ThSpec):
    return th_array(pb, spec.buy, spec.mode, spec.window), th_array(pc, spec.close, spec.mode, spec.window)


@njit
def _simulate(pb, pc, r, valid, th_b, th_c, lc):
    n = r.shape[0]
    out = np.zeros(n)
    pos = 0
    trades = 0
    inpos = 0
    for t in range(n):
        if not valid[t]:                  # end of a data segment: liquidate
            if pos == 1:
                out[t] += lc
                pos = 0
                trades += 1
            continue
        if pos == 0:
            if pb[t] > th_b[t]:
                pos = 1
                out[t] += lc
                trades += 1
        else:
            if pc[t] > th_c[t]:
                pos = 0
                out[t] += lc
                trades += 1
        if pos == 1:
            out[t] += r[t]
            inpos += 1
    return out, trades, inpos


def sharpe(x: np.ndarray, ppy: float) -> float:
    sd = x.std()
    return float(x.mean() / sd * math.sqrt(ppy)) if sd > 1e-12 else 0.0


def sim_metrics(pb, pc, r, valid, th_b, th_c, cost, ppy) -> dict:
    """th_b / th_c: scalar probabilities or per-candle threshold arrays."""
    n = len(r)
    th_b = np.full(n, th_b, dtype=float) if np.isscalar(th_b) else np.ascontiguousarray(th_b, dtype=float)
    th_c = np.full(n, th_c, dtype=float) if np.isscalar(th_c) else np.ascontiguousarray(th_c, dtype=float)
    out, trades, inpos = _simulate(pb, pc, r, valid, th_b, th_c, math.log1p(-cost))
    cum = np.cumsum(out)
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
    return dict(sharpe=sharpe(out, ppy), net_logret=float(out.sum()), trades=int(trades),
                exposure=float(inpos / max(len(out), 1)), max_dd_log=dd)


def tune_thresholds(val_items, grid, cost, mode="abs", window=1000):
    """val_items: list of (pb, pc, r, valid, ppy). Maximise mean Sharpe over the grid."""
    cache_b, cache_c = {}, {}

    def tb_arr(i, x):
        if (i, x) not in cache_b:
            cache_b[(i, x)] = th_array(val_items[i][0], x, mode, window)
        return cache_b[(i, x)]

    def tc_arr(i, x):
        if (i, x) not in cache_c:
            cache_c[(i, x)] = th_array(val_items[i][1], x, mode, window)
        return cache_c[(i, x)]

    rows, best = [], (-1e9, grid[0], grid[0])
    for tb in grid:
        for tc in grid:
            sc = np.mean([sim_metrics(pb, pc, r, v, tb_arr(i, tb), tc_arr(i, tc), cost, ppy)["sharpe"]
                          for i, (pb, pc, r, v, ppy) in enumerate(val_items)])
            rows.append(dict(th_buy=tb, th_close=tc, val_sharpe=float(sc)))
            if sc > best[0]:
                best = (sc, tb, tc)
    return best[1], best[2], best[0], pd.DataFrame(rows)


def eval_fixed_thresholds(val_items, spec: ThSpec, cost) -> float:
    """Mean validation Sharpe of one fixed (untuned) threshold spec."""
    scores = []
    for pb, pc, r, v, ppy in val_items:
        tb, tc = resolve_thresholds(pb, pc, spec)
        scores.append(sim_metrics(pb, pc, r, v, tb, tc, cost, ppy)["sharpe"])
    return float(np.mean(scores))


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — LOGO test + circular block bootstrap null
# ══════════════════════════════════════════════════════════════════════════════

def synth_frame(df: pd.DataFrame, rng, block: int) -> pd.DataFrame:
    """
    Circular block bootstrap of CANDLES. Each candle is kept intact relative to
    the previous close (Open/High/Low/Close ratios) together with its volume
    columns; blocks are drawn with wrap-around and chained into a new price
    path. Drift and short-range volatility structure survive; longer-range
    ordering (trends, regimes) is destroyed.
    """
    d = df.reset_index(drop=True)
    pc = d["Close"].to_numpy(float)
    prev = pc[:-1]
    m = len(prev)
    ratios = {k: d[k].to_numpy(float)[1:] / prev for k in ("Open", "High", "Low", "Close")}
    starts = rng.integers(0, m, math.ceil(m / block))
    idx = ((starts[:, None] + np.arange(block)[None, :]) % m).ravel()[:m]
    path = pc[0] * np.cumprod(ratios["Close"][idx])
    pprev = np.concatenate([[pc[0]], path[:-1]])
    out = {"Open_time": d["Open_time"].iloc[1:].to_numpy()}
    for k in ("Open", "High", "Low", "Close"):
        out[k] = ratios[k][idx] * pprev
    for c in d.columns:
        if c not in out and c != "Open_time" and not c.endswith("_Scaled"):
            out[c] = d[c].to_numpy(float)[1:][idx]
    return pd.DataFrame(out)


def null_sharpes(model, df, tf, feat_cols, use_tf, spec: ThSpec, cost, n_boot, block, seed):
    rng = np.random.default_rng(seed)
    ppy, tfm = periods_per_year(tf), TF_MINUTES[tf]
    res = []
    for _ in range(n_boot):
        syn = synth_frame(df, rng, block)
        with contextlib.redirect_stdout(io.StringIO()):
            feats = compute_features(syn, label="null")
        if len(feats) < 100:
            continue
        lp = np.log(feats["Close"].to_numpy(float))
        r = np.append(np.diff(lp), 0.0)
        valid = np.ones(len(feats), bool)
        valid[-1] = False
        pb, pc = predict_pair(model, feats, feat_cols, tfm, use_tf)
        tb, tc = resolve_thresholds(pb, pc, spec)        # recomputed on the synthetic path
        res.append(sim_metrics(pb, pc, r, valid, tb, tc, cost, ppy)["sharpe"])
    return np.array(res)


# ══════════════════════════════════════════════════════════════════════════════
# One fold: train on all other coins, test on the held-out coin
# ══════════════════════════════════════════════════════════════════════════════

def run_fold(all_series, holdout, args, feat_cols, cost, outdir):
    print(f"\n{'='*70}\n  FOLD: hold out {holdout}\n{'='*70}")
    use_tf = not args.no_tf_feature
    train_s = [s for s in all_series if s.sym != holdout]
    test_s = [s for s in all_series if s.sym == holdout]
    if not train_s or not test_s:
        print("  ⚠ need both training and held-out series; skipping fold.")
        return None

    def cut_index(s):
        if not args.cutoff:
            return s.n()
        return int(np.searchsorted(s.df["Open_time"].to_numpy(), args.cutoff))

    # chronological train / validation split inside each training series
    Xtr, ytr, mtr, Xva, yva, mva, val_items = [], [], [], [], [], [], []
    for s in train_s:
        n_eff = cut_index(s)
        if n_eff < 1000:
            continue
        va_lo = int(n_eff * (1 - args.val_frac))
        tr_hi = max(va_lo - args.embargo, 0)
        a = stacked_xy(s, 0, tr_hi, feat_cols, use_tf)
        b = stacked_xy(s, va_lo, n_eff, feat_cols, use_tf)
        if a[0] is None or b[0] is None:
            continue
        Xtr.append(a[0]); ytr.append(a[1]); mtr.append(a[2])
        Xva.append(b[0]); yva.append(b[1]); mva.append(b[2])
        val_items.append((s, va_lo, n_eff))
    if not Xtr:
        raise RuntimeError("No training data after applying cutoff / split.")
    Xtr, ytr, mtr = np.vstack(Xtr), np.concatenate(ytr), np.concatenate(mtr)
    Xva, yva, mva = np.vstack(Xva), np.concatenate(yva), np.concatenate(mva)
    print(f"  pooled train rows {len(ytr):,} | validation rows {len(yva):,} "
          f"| series used {len(val_items)}")

    names = design_names(feat_cols, use_tf)
    model = train_policy(Xtr, ytr, mtr, Xva, yva, mva, names, args.device,
                         args.weight_power, args.margin_weight, args.rounds, args.lr, args.seed)
    imp = sorted(zip(names, model.feature_importance("gain")), key=lambda x: -x[1])
    print("  top features (gain): " + ", ".join(f"{n}={g:,.0f}" for n, g in imp[:6]))

    # thresholds on validation slices
    vitems = []
    for s, lo, hi in val_items:
        pb, pc = predict_pair(model, s.df.iloc[lo:hi], feat_cols, s.tfm, use_tf)
        vitems.append((pb, pc, s.r[lo:hi], s.valid[lo:hi], periods_per_year(s.tf)))

    if args.th_mode == "abs":
        grid = [round(x, 2) for x in np.arange(args.th_lo, args.th_hi + 1e-9, args.th_step)]
        th_b, th_c, vscore, grid_df = tune_thresholds(vitems, grid, cost, "abs", args.q_window)
    elif args.q_tune:
        grid = [round(x, 3) for x in args.q_grid]
        th_b, th_c, vscore, grid_df = tune_thresholds(vitems, grid, cost, "quantile", args.q_window)
    else:                                              # fixed quantiles: nothing is fitted per fold
        th_b, th_c = args.q_buy, args.q_close
        spec0 = ThSpec("quantile", th_b, th_c, args.q_window)
        vscore = eval_fixed_thresholds(vitems, spec0, cost)
        grid_df = pd.DataFrame([dict(th_buy=th_b, th_close=th_c, val_sharpe=vscore)])
    spec = ThSpec(args.th_mode, th_b, th_c, args.q_window)
    print(f"  thresholds [{args.th_mode}]: {spec.describe()} (val Sharpe {vscore:.2f})")

    # held-out coin
    results = []
    for s in test_s:
        lo = cut_index(s) if args.cutoff else 0
        hi = s.n()
        if hi - lo < 500:
            print(f"  ⚠ {s.sym}/{s.tf}: <500 test candles, skipped.")
            continue
        sub = s.df.iloc[lo:hi]
        ppy = periods_per_year(s.tf)
        pb, pc = predict_pair(model, sub, feat_cols, s.tfm, use_tf)
        r, v = s.r[lo:hi], s.valid[lo:hi]
        tb, tc = resolve_thresholds(pb, pc, spec)
        m = sim_metrics(pb, pc, r, v, tb, tc, cost, ppy)
        base = sim_metrics(pb, pc, r, v, 0.5, 0.5, cost, ppy)       # untuned argmax-ish
        bh = r[v]
        row = dict(symbol=s.sym, timeframe=s.tf, test_rows=int(hi - lo),
                   sharpe=m["sharpe"], net_logret=m["net_logret"], trades=m["trades"],
                   exposure=m["exposure"], max_dd_log=m["max_dd_log"],
                   sharpe_th050=base["sharpe"], buyhold_sharpe=sharpe(bh, ppy),
                   buyhold_logret=float(bh.sum()))
        if args.n_boot > 0:
            ns = null_sharpes(model, sub, s.tf, feat_cols, use_tf, spec, cost,
                              args.n_boot, args.block, args.seed)
            row.update(null_mean=float(ns.mean()), null_p95=float(np.quantile(ns, 0.95)),
                       p_value=float((1 + (ns >= m["sharpe"]).sum()) / (1 + len(ns))),
                       n_null=int(len(ns)))
        results.append(row)
        print(f"  {s.sym}/{s.tf}: Sharpe {m['sharpe']:+.2f} | net logret {m['net_logret']:+.3f} "
              f"| trades {m['trades']} | exposure {m['exposure']:.0%} | B&H Sharpe "
              f"{row['buyhold_sharpe']:+.2f}"
              + (f" | null mean {row['null_mean']:+.2f}, p={row['p_value']:.3f}"
                 if args.n_boot > 0 else ""))

    fold_dir = os.path.join(outdir, f"holdout_{holdout}")
    os.makedirs(fold_dir, exist_ok=True)
    model.save_model(os.path.join(fold_dir, "policy_lgbm.txt"))
    grid_df.to_csv(os.path.join(fold_dir, "threshold_grid.csv"), index=False)
    report = dict(holdout=holdout, cost=cost,
                  thresholds=dict(mode=spec.mode, buy=th_b, close=th_c, window=spec.window,
                                  val_sharpe=vscore),
                  training=dict(weight_power=args.weight_power, margin_weight=args.margin_weight,
                                hold_penalty_per_day=args.hold_penalty),
                  feature_columns=feat_cols, design_columns=names, use_tf_feature=use_tf,
                  feature_hashes=feature_hashes(feat_cols), cutoff=args.cutoff,
                  train_symbols=sorted({s.sym for s in train_s}), results=results)
    with open(os.path.join(fold_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ saved model + report → {fold_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Oracle-labelled pooled trading policy.")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"])
    ap.add_argument("--holdout", default="XRPUSDT", help="coin to hold out (LOGO)")
    ap.add_argument("--all-folds", action="store_true", help="run LOGO for every coin")
    ap.add_argument("--cutoff", default=None,
                    help="YYYY-MM-DD: train only before it, test held-out coin from it")
    ap.add_argument("--fee", type=float, default=0.001, help="per-side fee (0.001 = 10 bp)")
    ap.add_argument("--slippage", type=float, default=0.0005, help="per-side slippage")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    # Option C — class weights
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="class weight = (N/(3*count))^power; 0 disables (try 0.5 or 0)")
    # Option B — cost-aware training
    ap.add_argument("--margin-weight", type=float, default=0.0,
                    help="row weight x (oracle decision margin)^this. The margin is the "
                         "log-wealth lost by the wrong action after costs, so marginal "
                         "trades are down-weighted. 0 disables, 1 = linear.")
    ap.add_argument("--hold-penalty", type=float, default=0.0,
                    help="log-return penalty PER DAY of holding, applied only to the oracle "
                         "labels (scaled to the timeframe). Makes labels sparser/shorter.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--embargo", type=int, default=100,
                    help="candles dropped between train and validation (oracle labels look ahead)")
    # Option A — thresholds
    ap.add_argument("--th-mode", choices=["quantile", "abs"], default="quantile",
                    help="quantile: trade when p exceeds its own rolling quantile; "
                         "abs: tuned absolute probability cutoffs (old behaviour)")
    ap.add_argument("--q-window", type=int, default=1000,
                    help="rolling window (candles) for the quantile thresholds")
    ap.add_argument("--q-buy", type=float, default=0.95, help="BUY quantile (0.95 = top 5%%)")
    ap.add_argument("--q-close", type=float, default=0.95, help="CLOSE quantile")
    ap.add_argument("--q-tune", action="store_true",
                    help="tune the quantiles on validation over --q-grid instead of fixing them")
    ap.add_argument("--q-grid", type=float, nargs="+", default=[0.80, 0.90, 0.95, 0.98])
    ap.add_argument("--th-lo", type=float, default=0.50, help="abs mode: grid start")
    ap.add_argument("--th-hi", type=float, default=0.90, help="abs mode: grid end")
    ap.add_argument("--th-step", type=float, default=0.05, help="abs mode: grid step")
    ap.add_argument("--n-boot", type=int, default=50, help="0 disables the null test")
    ap.add_argument("--block", type=int, default=48, help="bootstrap block length (candles)")
    ap.add_argument("--no-tf-feature", action="store_true", help="drop the LogTF feature")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/oracle_policy")
    args = ap.parse_args()

    cost = args.fee + args.slippage
    feat_cols = feature_columns(default_only=True)
    print(f"Round-trip cost {2*cost:.3%} | features: {feat_cols}")
    all_series = load_all_series(args.symbols, args.timeframes, cost, args.hold_penalty)
    for c in feat_cols:
        if any(c not in s.df.columns for s in all_series):
            raise RuntimeError(f"Feature column {c} missing from a master — rebuild the masters.")

    folds = sorted({s.sym for s in all_series}) if args.all_folds else [args.holdout]
    summary = []
    for h in folds:
        res = run_fold(all_series, h, args, feat_cols, cost, args.out)
        summary += res or []
    if summary:
        df = pd.DataFrame(summary)
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
        pd.set_option("display.width", 200)
        print("\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()