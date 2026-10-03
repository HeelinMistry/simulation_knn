"""
pooled_hmm_pipeline.py
─────────────────────────────────────────────────────────────
Pooled GaussianHMM across symbols — STEP 1: do the states predict anything
out-of-sample?

CHANGES IN THIS REVISION
────────────────────────
* Feature set now includes GK_Scaled (Garman-Klass range-based volatility).
  All features are produced by preprocessing.py as trailing, per-asset
  rolling z-scores (see that file), so pooled symbols share a comparable
  "relative to my own recent regime" scale.
* Frozen-model safety: if a saved artifact was fit on a different feature
  list than FEATURE_COLS (e.g. the old 6-feature model), loading is refused
  with a clear message — pass --refit. The old artifact is NOT silently
  reused with mismatched columns.
* Data safety: if a processed master lacks any FEATURE_COLS (masters built
  before this revision), loading fails loudly with rebuild instructions
  instead of a KeyError deep in pandas.

* BB_Scaled removed from FEATURE_COLS (duplicate of MeanDev_Scaled).
* N_STATES default is now 5 (--n-states to override). Artifacts are saved as
  pooled_hmm_<tf>_k<K>.joblib so 4- and 5-state models can be compared
  side by side. A heuristic tag (chop/low-vol, up, down) is printed per state.

Unchanged from the previous revision: no look-ahead (forward filtering),
fit-once-then-freeze, one global calendar train/test cutoff, canonical
state ordering, non-overlapping-subsample significance tests.

Run:
    python pooled_hmm_pipeline.py                # fit once (or load), evaluate
    python pooled_hmm_pipeline.py --refit        # force retrain (required once
                                                 # after this feature change)
    python pooled_hmm_pipeline.py --rescan-raw   # rebuild masters from raw files
    python pooled_hmm_pipeline.py --train-frac 0.6 --primary-horizon 12
"""

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats
from scipy.stats import multivariate_normal

from data.data_manager import update_master_data, SYMBOLS

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "RSI_Scaled", "MACD_Scaled",
    "OBV_Scaled", "ATR_Scaled", "GK_Scaled", "MeanDev_Scaled",
]
CANON_FEATURE = "MeanDev_Scaled"        # states are ordered by this feature's mean
TIMEFRAME     = "4h"
CANDLE        = pd.Timedelta(hours=4)
N_STATES      = 5                       # 5 = room for an explicit low-vol chop state
N_INIT        = 5                       # random restarts; best train log-lik wins
HORIZONS      = (1, 3, 6, 12, 24)       # candles (4h -> 4h, 12h, 1d, 2d, 4d)
MODEL_DIR     = "models"
OUT_DIR       = "data/processed"


def _paths(timeframe: str, n_states: int = N_STATES):
    # n_states in the filename so 4- and 5-state artifacts coexist for comparison
    return (
        os.path.join(MODEL_DIR, f"pooled_hmm_{timeframe}_k{n_states}.joblib"),
        os.path.join(MODEL_DIR, f"pooled_hmm_{timeframe}_k{n_states}_profile.json"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Causal inference (reusable live)
# ══════════════════════════════════════════════════════════════════════════════

def emission_logprob(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """(T, K) log p(x_t | state k). Uses public attributes only."""
    covs = model.covars_
    if covs.ndim == 2:                       # (K, F) diag stored compactly
        covs = np.array([np.diag(c) for c in covs])
    out = np.empty((len(X), model.n_components))
    for k in range(model.n_components):
        out[:, k] = multivariate_normal(
            mean=model.means_[k], cov=covs[k], allow_singular=True
        ).logpdf(X)
    return out


def filter_step(model: GaussianHMM, alpha_prev, log_b_t: np.ndarray) -> np.ndarray:
    """
    One forward-filtering update.
      alpha_prev : P(s_{t-1} | x_1..t-1), or None for the first observation
      log_b_t    : (K,) emission log-probs for x_t
    Returns P(s_t | x_1..t). Uses ONLY past + current observation.
    """
    pred = model.startprob_ if alpha_prev is None else alpha_prev @ model.transmat_
    a = pred * np.exp(log_b_t - log_b_t.max())      # max-shift for stability
    s = a.sum()
    return a / s if s > 0 else np.full_like(a, 1.0 / len(a))


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """(T, K) filtered posteriors P(s_t | x_1..t). No look-ahead."""
    log_b = emission_logprob(model, X)
    post = np.empty_like(log_b)
    alpha = None
    for t in range(len(X)):
        alpha = filter_step(model, alpha, log_b[t])
        post[t] = alpha
    return post


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_symbol_frames(symbols, timeframe=TIMEFRAME, frozen=True) -> dict:
    """
    Research-time loader (reads the processed masters). frozen=True means no
    raw-file scan / master rewrite. NOT for live use.
    Each frame gets a `seg` id that increments at any gap > 1.5 candles, so
    the HMM never treats a data hole as consecutive candles.
    """
    frames = {}
    for sym in symbols:
        try:
            df = update_master_data(timeframe=timeframe, symbol=sym, frozen=frozen)
        except FileNotFoundError as exc:
            print(f"  ⚠ Skipping {sym}: {exc}")
            continue

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Master CSV for {sym}/{timeframe} is missing feature column(s) "
                f"{missing} — it was built before the rolling-standardization / "
                f"Garman-Klass change. Rebuild it from raw files "
                f"(back up the old master first, then run with --rescan-raw)."
            )

        df = df.dropna(subset=FEATURE_COLS).copy()
        if df.empty:
            continue
        df["Open_time"] = pd.to_datetime(df["Open_time"])
        df = df.sort_values("Open_time").drop_duplicates("Open_time").reset_index(drop=True)
        df["Symbol"] = sym
        df["seg"] = (df["Open_time"].diff() > CANDLE * 1.5).cumsum()
        df["t_idx"] = np.arange(len(df))
        frames[sym] = df
        print(f"  ✓ {sym}: {len(df):,} candles  "
              f"{df['Open_time'].iloc[0]:%Y-%m-%d} → {df['Open_time'].iloc[-1]:%Y-%m-%d}  "
              f"({df['seg'].nunique()} segment(s))")
    return frames


def global_cutoff(frames: dict, train_frac: float) -> pd.Timestamp:
    all_ts = np.sort(np.concatenate([f["Open_time"].values for f in frames.values()]))
    return pd.Timestamp(all_ts[int(len(all_ts) * train_frac)])


def build_train_matrix(frames: dict, cutoff: pd.Timestamp):
    """Stack train rows (Open_time < cutoff); lengths = one per (symbol, segment)."""
    X_parts, lengths = [], []
    for df in frames.values():
        tr = df[df["Open_time"] < cutoff]
        for _, seg_df in tr.groupby("seg", sort=True):
            if len(seg_df) < 50:                    # ignore tiny fragments
                continue
            X_parts.append(seg_df[FEATURE_COLS].values)
            lengths.append(len(seg_df))
    return np.vstack(X_parts), lengths


# ══════════════════════════════════════════════════════════════════════════════
# Fit / freeze / load
# ══════════════════════════════════════════════════════════════════════════════

def fit_best_hmm(X, lengths, n_states=N_STATES, n_init=N_INIT) -> GaussianHMM:
    best, best_ll = None, -np.inf
    for seed in range(n_init):
        m = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=300, tol=1e-4, random_state=42 + seed)
        m.fit(X, lengths=lengths)
        ll = m.score(X, lengths=lengths)
        print(f"    init {seed}: train log-lik = {ll:,.1f}  "
              f"(converged={m.monitor_.converged})")
        if ll > best_ll:
            best, best_ll = m, ll
    print(f"  ✓ Best train log-lik: {best_ll:,.1f}")
    return best


def canonical_order(model: GaussianHMM) -> np.ndarray:
    """canon_to_raw[k] = raw state index that becomes canonical state k."""
    j = FEATURE_COLS.index(CANON_FEATURE)
    return np.argsort(model.means_[:, j])


def state_profile(model: GaussianHMM, canon_to_raw: np.ndarray) -> dict:
    prof = {}
    for k, raw in enumerate(canon_to_raw):
        prof[str(k)] = {
            "raw_state": int(raw),
            **{f: round(float(model.means_[raw, i]), 4) for i, f in enumerate(FEATURE_COLS)},
        }
    return prof


def fit_or_load(frames, cutoff, timeframe=TIMEFRAME, refit=False, n_states=N_STATES):
    model_path, prof_path = _paths(timeframe, n_states)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(model_path) and not refit:
        art = joblib.load(model_path)
        if list(art["feature_cols"]) != list(FEATURE_COLS):
            raise RuntimeError(
                f"Frozen model {model_path} was fit on features "
                f"{list(art['feature_cols'])}, but FEATURE_COLS is now "
                f"{FEATURE_COLS}. The old model cannot score the new feature "
                f"set. Re-run with --refit to train and freeze a new one."
            )
        print(f"  ✓ Loaded FROZEN model ← {model_path}  "
              f"(fit {art['fit_utc']}, train cutoff {art['cutoff']})")
        if pd.Timestamp(art["cutoff"]) != cutoff:
            print(f"  ⚠ Frozen model's cutoff ({art['cutoff']}) != requested "
                  f"cutoff ({cutoff}). Evaluating against the FROZEN cutoff.")
        return art

    X, lengths = build_train_matrix(frames, cutoff)
    print(f"  Fitting pooled HMM on TRAIN only: {len(X):,} obs, "
          f"{len(lengths)} sequence(s), cutoff {cutoff}")
    model = fit_best_hmm(X, lengths, n_states=n_states)
    order = canonical_order(model)
    art = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "canon_to_raw": order,
        "raw_to_canon": np.argsort(order),
        "cutoff": str(cutoff),
        "timeframe": timeframe,
        "symbols": list(frames.keys()),
        "fit_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(art, model_path)
    with open(prof_path, "w") as f:
        json.dump({k: v for k, v in art.items() if k not in ("model", "canon_to_raw", "raw_to_canon")}
                  | {"state_profile": state_profile(model, order)}, f, indent=2)
    print(f"  ✓ Saved frozen model → {model_path}\n  ✓ Saved profile      → {prof_path}")
    return art


# ══════════════════════════════════════════════════════════════════════════════
# Labelling (causal) + diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def label_frames(frames: dict, art: dict) -> pd.DataFrame:
    """
    Per symbol, per segment: causal filtered posteriors -> hard state (argmax)
    in CANONICAL numbering, plus a one-off Viterbi state for the look-ahead
    diagnostic. Filtering runs through train AND test continuously (backward
    looking only); the caller slices to test for evaluation.
    """
    model, r2c = art["model"], art["raw_to_canon"]
    K = model.n_components
    out = []
    for sym, df in frames.items():
        df = df.copy()
        post = np.zeros((len(df), K))
        vit = np.zeros(len(df), dtype=int)
        for _, idx in df.groupby("seg").indices.items():
            X = df.iloc[idx][FEATURE_COLS].values
            post[idx] = forward_filter(model, X)
            vit[idx] = model.predict(X)                 # DIAGNOSTIC ONLY (look-ahead)
        canon_post = post[:, art["canon_to_raw"]]       # columns in canonical order
        df["State"] = canon_post.argmax(1)
        df["State_Conf"] = canon_post.max(1)
        df["State_Viterbi"] = r2c[vit]
        for k in range(K):
            df[f"P_State_{k}"] = canon_post[:, k]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def add_forward_returns(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """fwd_h = log(Close[t+h] / Close[t]); NaN if the h-th next candle isn't
    exactly h*CANDLE later (gap) or runs past the end. Never crosses symbols."""
    parts = []
    for _, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Open_time").copy()
        for h in horizons:
            ok = (g["Open_time"].shift(-h) - g["Open_time"]) == CANDLE * h
            r = np.log(g["Close"].shift(-h) / g["Close"])
            g[f"fwd_{h}"] = r.where(ok)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def evaluate_states(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    rows = []
    for h in horizons:
        col = f"fwd_{h}"
        d = df.dropna(subset=[col]).copy()
        d["ex"] = d[col] - d.groupby("Symbol")[col].transform("mean")
        sub = d[d["t_idx"] % h == 0]                   # non-overlapping subsample
        groups = [g["ex"].values for _, g in sub.groupby("State") if len(g) > 5]
        kw_p = stats.kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
        for s, g in d.groupby("State"):
            gs = sub[sub["State"] == s]["ex"]
            t = (gs.mean() / (gs.std(ddof=1) / np.sqrt(len(gs)))) if len(gs) > 5 else np.nan
            rows.append({
                "horizon": h, "state": s, "n": len(g), "n_nonoverlap": len(gs),
                "mean_bps": g[col].mean() * 1e4,
                "excess_bps": g["ex"].mean() * 1e4,
                "median_bps": g[col].median() * 1e4,
                "hit_rate": (g[col] > 0).mean(),
                "t_excess": t,
                "KW_p_all_states": kw_p,
            })
    return pd.DataFrame(rows)


def dwell_times(df: pd.DataFrame, K: int) -> pd.Series:
    runs = {k: [] for k in range(K)}
    for _, g in df.groupby(["Symbol", "seg"]):
        s = g.sort_values("Open_time")["State"].values
        if len(s) == 0:
            continue
        start = 0
        for i in range(1, len(s) + 1):
            if i == len(s) or s[i] != s[start]:
                runs[s[start]].append(i - start)
                start = i
    return pd.Series({k: (np.mean(v) if v else np.nan) for k, v in runs.items()},
                     name="avg_dwell_candles")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def tag_states(model, canon_to_raw) -> dict:
    """
    Heuristic labels from the TRAIN-fit means (diagnostic only):
      chop     = lowest GK_Scaled (quietest realised range) among states whose
                 |MeanDev| is below the median |MeanDev| of all states
      (falls back to the lowest-GK state if none qualify)
      directional states are tagged by MeanDev sign: down / up.
    """
    gk = FEATURE_COLS.index("GK_Scaled")
    md = FEATURE_COLS.index("MeanDev_Scaled")
    K = len(canon_to_raw)
    gk_m = np.array([model.means_[r, gk] for r in canon_to_raw])
    md_m = np.array([model.means_[r, md] for r in canon_to_raw])
    near = np.abs(md_m) <= np.median(np.abs(md_m))
    cand = np.where(near)[0] if near.any() else np.arange(K)
    chop = int(cand[np.argmin(gk_m[cand])])
    return {k: ("chop/low-vol" if k == chop else ("up" if md_m[k] > 0 else "down"))
            for k in range(K)}


def run_pooled_hmm(train_frac=0.70, refit=False, primary_h=6, timeframe=TIMEFRAME,
                   frozen_data=True, n_states=N_STATES):
    print(f"\n{'=' * 64}\n  POOLED HMM — STEP 1: OUT-OF-SAMPLE STATE CHECK\n{'=' * 64}")
    frames = load_symbol_frames(SYMBOLS, timeframe, frozen=frozen_data)
    if not frames:
        print("❌ No data loaded.")
        return

    cutoff = global_cutoff(frames, train_frac)
    print(f"\n  Global train/test cutoff: {cutoff}  (train_frac={train_frac})")

    art = fit_or_load(frames, cutoff, timeframe, refit=refit, n_states=n_states)
    cutoff = pd.Timestamp(art["cutoff"])                # trust the frozen cutoff
    model = art["model"]
    K = model.n_components

    print(f"\n  [Canonical state profile — TRAIN-fit means, K={K}]")
    prof_df = pd.DataFrame(state_profile(model, art["canon_to_raw"])).T.drop(columns="raw_state")
    prof_df["tag"] = pd.Series(tag_states(model, art["canon_to_raw"]))
    print(prof_df.round(3).to_string())

    print("\n  [Transition matrix, canonical order]")
    o = art["canon_to_raw"]
    print(pd.DataFrame(model.transmat_[np.ix_(o, o)],
                       index=[f"S{i}" for i in range(K)],
                       columns=[f"S{i}" for i in range(K)]).round(3).to_string())

    # Causal labels over the full series, then split
    lab = add_forward_returns(label_frames(frames, art))
    train = lab[lab["Open_time"] < cutoff]
    test = lab[lab["Open_time"] >= cutoff]
    print(f"\n  Train rows: {len(train):,}   Test rows: {len(test):,}")

    # Diagnostics
    print("\n  [State occupancy (causal)]  train vs test")
    occ = pd.concat([train["State"].value_counts(normalize=True).sort_index().rename("train"),
                     test["State"].value_counts(normalize=True).sort_index().rename("test")], axis=1)
    print(occ.round(3).to_string())
    print("\n  [Average dwell time, test, in candles]  (low => flickering)")
    print(dwell_times(test, K).round(1).to_string())
    agree = (test["State"] == test["State_Viterbi"]).mean()
    print(f"\n  Causal vs Viterbi agreement on test: {agree:.1%}  "
          f"(gap from 100% = how much look-ahead was changing your labels)")

    # The actual question
    ev_test = evaluate_states(test)
    ev_train = evaluate_states(train)
    show = ["horizon", "state", "n", "n_nonoverlap", "mean_bps", "excess_bps",
            "hit_rate", "t_excess", "KW_p_all_states"]
    print(f"\n{'─' * 64}\n  OUT-OF-SAMPLE (test) — forward returns by causal state\n{'─' * 64}")
    print(ev_test[show].round(3).to_string(index=False))
    print(f"\n{'─' * 64}\n  IN-SAMPLE (train) — for comparison only\n{'─' * 64}")
    print(ev_train[show].round(3).to_string(index=False))

    # Per-symbol consistency at the primary horizon
    col = f"fwd_{primary_h}"
    d = test.dropna(subset=[col]).copy()
    d["ex"] = d[col] - d.groupby("Symbol")[col].transform("mean")
    piv = (d.pivot_table(index="Symbol", columns="State", values="ex", aggfunc="mean") * 1e4)
    piv.columns = [f"S{c}" for c in piv.columns]
    print(f"\n{'─' * 64}\n  Per-symbol excess return (bps), test, horizon={primary_h} candles\n"
          f"  (an edge worth trusting has the SAME sign pattern across most symbols)\n{'─' * 64}")
    print(piv.round(1).to_string())

    # Compact pass/fail read at the primary horizon
    p = ev_test[ev_test["horizon"] == primary_h]
    print(f"\n{'─' * 64}\n  READ (horizon={primary_h}): "
          f"KW p={p['KW_p_all_states'].iloc[0]:.4f}  |  "
          f"excess spread best-worst = {p['excess_bps'].max() - p['excess_bps'].min():.1f} bps\n"
          f"  Proceed to the next step only if: spread is economically meaningful vs costs,\n"
          f"  |t| is large for at least one state on non-overlapping data, the pattern\n"
          f"  holds across symbols, AND it survives from train to test (same sign/order).\n{'─' * 64}")

    os.makedirs(OUT_DIR, exist_ok=True)
    ev_test.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_test.csv"), index=False)
    ev_train.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_train.csv"), index=False)
    keep = ["Open_time", "Symbol", "Close", "State", "State_Conf", "State_Viterbi"] + \
           [f"P_State_{k}" for k in range(K)] + [f"fwd_{h}" for h in HORIZONS]
    lab[keep].assign(Split=np.where(lab["Open_time"] < cutoff, "train", "test")) \
        .to_csv(os.path.join(OUT_DIR, "universal_crypto_pooled_hmm_states_causal.csv"), index=False)
    print(f"\n  ✓ Saved eval tables + causal labelled dataset to {OUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="retrain and overwrite frozen model")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--primary-horizon", type=int, default=6, choices=HORIZONS)
    ap.add_argument("--timeframe", default=TIMEFRAME)
    ap.add_argument("--n-states", type=int, default=N_STATES,
                    help="number of HMM states (4 vs 5 artifacts are saved separately)")
    ap.add_argument("--rescan-raw", action="store_true",
                    help="let update_master_data() rescan raw files (default: frozen masters)")
    a = ap.parse_args()
    run_pooled_hmm(a.train_frac, a.refit, a.primary_horizon, a.timeframe,
                   frozen_data=not a.rescan_raw, n_states=a.n_states)

# Run standard evaluation using the frozen model or fit if missing
# python pooled_hmm_pipeline.py

# Force a complete retrain of the pooled model and update the frozen artifacts
# python pooled_hmm_pipeline.py --refit

# Rebuild masters from raw files, then retrain (needed once after this change)
# python pooled_hmm_pipeline.py --rescan-raw --refit