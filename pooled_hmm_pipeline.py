"""
pooled_hmm_pipeline.py
─────────────────────────────────────────────────────────────
Pooled GaussianHMM across symbols — STEP 1: do the states predict anything
out-of-sample?

CHANGES IN THIS REVISION — COIN-AGNOSTIC, RETURN-BASED, HONEST STATISTICS
─────────────────────────────────────────────────────────────────────────
A. FEATURES COME FROM THE REGISTRY (single source of truth)
   FEATURE_COLS / OPTIONAL_FEATURES are derived from data.preprocessing.FEATURES
   (the `default` flag), no hand-kept lists. Each frozen model stores the content
   hash of every feature it was fit on; loading it after a feature's definition
   changed fails loudly (--refit). Processed masters carry the same hashes
   (see data_manager.py), so a stale master is rebuilt, never silently used.
   --feature-check now also runs the unit-free contract test (rescales prices
   and volumes; a coin-specific feature fails) and a cross-coin scaling report.
   --legacy-features reproduces the old six-feature set (ATR in, no returns).

B. PRICE CHANGE IS IN THE MODEL AND IN THE TARGET
   * New default features Ret1 / Mom12 / RealVol (per-asset z-scored returns =
     volatility-normalised returns). ATR is opt-in (--extra-features).
   * New targets per horizon h (vol = trailing 60-candle std of 1-candle log
     returns, causal, computed on the full history):
         fwd_h  = log(C[t+h] / C[t])                       raw log return
         fwdz_h = fwd_h / (vol_t * sqrt(h))                sigma-normalised
         absz_h = |fwdz_h|                                 volatility effect
     Evaluation defaults to fwdz, so a 4-bps move in BTC and a 40-bps move in
     SOL carry the same weight. Direction (fwdz) and volatility (absz) are
     reported SEPARATELY (the HMM sees GK/RealVol, so states are expected to
     separate on volatility; that is not a tradable direction).

C. SIGNIFICANCE THAT RESPECTS CROSS-COIN DEPENDENCE
   Rows from different coins at the same timestamp are NOT independent, so
   per-row (or per-symbol) subsampled tests overstate significance. Now:
     * data are arranged as a time x coin PANEL; for each state the statistic is
       the time-average of the cross-sectional mean excess (one number per
       timestamp, so one observation per timestamp, not per coin);
     * `t_boot`, `ex_lo`, `ex_hi`: moving-block bootstrap over TIME
       (block = max(2h, 12) candles; handles overlapping horizons);
     * `p_null`, `p_spread`: circular-shift null. All coins' state sequences are
       shifted by ONE common random offset in time (keeps within-coin persistence
       AND the cross-coin co-movement of states; breaks only the link between
       states and subsequent returns). `p_spread` is for best-minus-worst, so the
       selection bias of max-min is built into the null; `p_null` is two-sided
       per state;
     * `p_adj_BH`: Benjamini-Hochberg across every (horizon, state) row of a
       table. Treat the primary horizon (--primary-horizon) as the headline and
       the rest as exploratory;
     * `hit_excess`: hit rate minus each coin's own baseline hit rate (drift).
   Baselines are per-symbol means over ALL candles with a valid target (never
   over a filtered subset).

D. CONFIG / CLI AGREE
   One COV_TYPE constant ("diag") is the default for both Config and argparse
   (the CLI used to default to "full", giving different artifact names
   depending on how you launched). Override with --cov-type full.

Unchanged: no look-ahead (forward filtering), fit-once-then-freeze, one global
calendar train/test cutoff, canonical state ordering, posterior-confidence
filtering, walk-forward refits with Hungarian label alignment, sticky/min-dwell
duration control, drop-one ablation on an inner validation slice.

Run:
    python pooled_hmm_pipeline.py --refit
    python pooled_hmm_pipeline.py --feature-check
    python pooled_hmm_pipeline.py --conf-threshold 0.8
    python pooled_hmm_pipeline.py --walk-forward
    python pooled_hmm_pipeline.py --ablate
    python pooled_hmm_pipeline.py --legacy-features --refit
    python pooled_hmm_pipeline.py --target fwd --n-null 2000      # raw returns
    python pooled_hmm_pipeline.py --cov-type full --min-covar 1e-3 --refit
    python pooled_hmm_pipeline.py --sticky 500 --min-dwell 2 --refit
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

from data.data_manager import update_master_data, SYMBOLS
from data.preprocessing import (
    feature_columns, feature_hashes,
    cross_coin_scaling_report, verify_feature_contract,
)

# ── Config ────────────────────────────────────────────────────────────────────
# Derived from the registry — add an indicator there, not here.
FEATURE_COLS      = feature_columns(default_only=True)
OPTIONAL_FEATURES = [c for c in feature_columns() if c not in FEATURE_COLS]
LEGACY_FEATURES   = ["RSI_Scaled", "MACD_Scaled", "OBV_Scaled",
                     "ATR_Scaled", "GK_Scaled", "MeanDev_Scaled"]
_unknown = [c for c in LEGACY_FEATURES if c not in feature_columns()]
assert not _unknown, f"LEGACY_FEATURES not in the registry: {_unknown}"

CANON_FEATURE = "MeanDev_Scaled"        # states are ordered by this feature's mean
TIMEFRAME     = "4h"
CANDLE        = pd.Timedelta(hours=4)
N_STATES      = 4
N_INIT        = 5                       # random restarts; best train log-lik wins
COV_TYPE      = "diag"                  # ONE default for Config and the CLI
HORIZONS      = (1, 3, 6, 12, 24)       # candles (4h -> 4h, 12h, 1d, 2d, 4d)
CONF_SWEEP    = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
WF_WARMUP     = 200                     # candles of filter warm-up before each WF fold
MODEL_DIR     = "models"
OUT_DIR       = "data/processed"

TARGET        = "fwdz"                  # sigma-normalised forward return
VOL_WIN       = 60                      # trailing window for the vol normaliser
VOL_MIN       = 30
N_NULL        = 1000                    # circular-shift draws
N_BOOT        = 500                     # block-bootstrap draws
MIN_TS_STATE  = 30                      # a state needs this many timestamps to enter the spread


@dataclass
class Config:
    timeframe: str = TIMEFRAME
    n_states: int = N_STATES
    n_init: int = N_INIT
    train_frac: float = 0.70
    primary_h: int = 6
    refit: bool = False
    frozen_data: bool = True
    cov_type: str = COV_TYPE
    min_covar: float = 1e-3
    sticky: float = 0.0
    min_dwell: int = 1
    conf_threshold: float = 0.75
    walk_forward: bool = False
    wf_step_months: int = 6
    wf_train_months: int = 0            # 0 = expanding window
    drop_features: tuple = ()
    extra_features: bool = False
    legacy_features: bool = False
    feature_check: bool = False
    ablate: bool = False
    target: str = TARGET                # fwdz | fwd
    n_null: int = N_NULL
    n_boot: int = N_BOOT

    def features(self) -> list:
        feats = list(LEGACY_FEATURES if self.legacy_features else FEATURE_COLS)
        if self.extra_features:
            feats += [f for f in OPTIONAL_FEATURES if f not in feats]
        drop = {d if d.endswith("_Scaled") else f"{d}_Scaled" for d in self.drop_features}
        feats = [f for f in feats if f not in drop]
        if len(feats) < 2:
            raise ValueError("Need at least 2 features after --drop-features.")
        return feats


def _tag(cfg: Config, feats: list) -> str:
    """Artifact tag. Default settings -> 'k<K>'."""
    t = f"k{cfg.n_states}"
    if cfg.cov_type != COV_TYPE:
        t += f"_{cfg.cov_type}"
    if cfg.min_covar != 1e-3:
        t += f"_mc{cfg.min_covar:g}"
    if cfg.sticky > 0:
        t += f"_sticky{cfg.sticky:g}"
    if feats != FEATURE_COLS:
        t += "_f" + hashlib.md5(",".join(feats).encode()).hexdigest()[:6]
    return t


def _paths(timeframe: str, tag: str):
    return (
        os.path.join(MODEL_DIR, f"pooled_hmm_{timeframe}_{tag}.joblib"),
        os.path.join(MODEL_DIR, f"pooled_hmm_{timeframe}_{tag}_profile.json"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Causal inference (reusable live)
# ══════════════════════════════════════════════════════════════════════════════

def emission_logprob(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """(T, K) log p(x_t | state k). Supports 'diag' and 'full' covariances."""
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
    """One forward-filtering update: P(s_t | x_1..t). Uses ONLY past + current."""
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


def apply_min_dwell(raw: np.ndarray, m: int) -> np.ndarray:
    """
    Causal hysteresis. out[t] depends only on raw[:t+1]. A switch away from the
    current state is accepted only once the SAME new state has been the argmax
    for m consecutive candles. m <= 1 returns raw unchanged.
    """
    if m <= 1 or len(raw) == 0:
        return raw.copy()
    out = np.empty_like(raw)
    cur, pend, cnt = raw[0], -1, 0
    for i, s in enumerate(raw):
        if s == cur:
            pend, cnt = -1, 0
        elif s == pend:
            cnt += 1
        else:
            pend, cnt = s, 1
        if pend != -1 and cnt >= m:
            cur, pend, cnt = pend, -1, 0
        out[i] = cur
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_symbol_frames(symbols, timeframe=TIMEFRAME, frozen=True, feats=None) -> dict:
    """
    Research-time loader (reads the processed masters; stale feature caches are
    rebuilt from the raw master by data_manager). Each frame gets:
      seg   : id that increments at any gap > 1.5 candles
      tg    : GLOBAL calendar index (candles since the earliest timestamp over
              all symbols) — aligns coins into a time x coin panel
      vol   : causal trailing std of 1-candle log returns (target normaliser)
    """
    feats = list(FEATURE_COLS) if feats is None else feats
    frames = {}
    for sym in symbols:
        try:
            df = update_master_data(timeframe=timeframe, symbol=sym, frozen=frozen)
        except FileNotFoundError as exc:
            print(f"  ⚠ Skipping {sym}: {exc}")
            continue

        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Processed master for {sym}/{timeframe} lacks feature column(s) "
                f"{missing}. They are not produced by the indicator registry "
                f"(data/preprocessing.py) — e.g. TakerImb needs Taker_buy_base_vol "
                f"in the raw files. Fix the registry/raw data or drop them."
            )

        df = df.dropna(subset=feats).copy()
        if df.empty:
            continue
        df["Open_time"] = pd.to_datetime(df["Open_time"])
        df = df.sort_values("Open_time").drop_duplicates("Open_time").reset_index(drop=True)
        df["Symbol"] = sym
        df["seg"] = (df["Open_time"].diff() > CANDLE * 1.5).cumsum()
        df["t_idx"] = np.arange(len(df))
        r1 = np.log(df["Close"]).diff().where(df["Open_time"].diff() == CANDLE)
        df["vol"] = r1.rolling(VOL_WIN, min_periods=VOL_MIN).std()
        frames[sym] = df
        print(f"  ✓ {sym}: {len(df):,} candles  "
              f"{df['Open_time'].iloc[0]:%Y-%m-%d} → {df['Open_time'].iloc[-1]:%Y-%m-%d}  "
              f"({df['seg'].nunique()} segment(s))")

    if frames:
        t0 = min(f["Open_time"].min() for f in frames.values())
        for df in frames.values():
            df["tg"] = ((df["Open_time"] - t0) // CANDLE).astype(int)
    return frames


def global_cutoff(frames: dict, train_frac: float) -> pd.Timestamp:
    all_ts = np.sort(np.concatenate([f["Open_time"].values for f in frames.values()]))
    return pd.Timestamp(all_ts[int(len(all_ts) * train_frac)])


def build_train_matrix(frames: dict, cutoff: pd.Timestamp, feats: list, start=None):
    """Stack train rows (start <= Open_time < cutoff); one length per (symbol, segment)."""
    X_parts, lengths = [], []
    for df in frames.values():
        tr = df[df["Open_time"] < cutoff]
        if start is not None:
            tr = tr[tr["Open_time"] >= start]
        for _, seg_df in tr.groupby("seg", sort=True):
            if len(seg_df) < 50:                    # ignore tiny fragments
                continue
            X_parts.append(seg_df[feats].values)
            lengths.append(len(seg_df))
    if not X_parts:
        raise RuntimeError(f"No training data before {cutoff} (start={start}).")
    return np.vstack(X_parts), lengths


# ══════════════════════════════════════════════════════════════════════════════
# Fit / freeze / load
# ══════════════════════════════════════════════════════════════════════════════

def make_hmm(cfg: Config, seed: int) -> GaussianHMM:
    kw = {}
    if cfg.sticky > 0:
        # Dirichlet MAP prior: +sticky pseudo-counts on the diagonal
        kw["transmat_prior"] = np.ones((cfg.n_states, cfg.n_states)) + cfg.sticky * np.eye(cfg.n_states)
    return GaussianHMM(n_components=cfg.n_states, covariance_type=cfg.cov_type,
                       min_covar=cfg.min_covar, n_iter=300, tol=1e-4,
                       random_state=42 + seed, **kw)


def fit_best_hmm(X, lengths, cfg: Config, verbose=True):
    best, best_ll = None, -np.inf
    for seed in range(cfg.n_init):
        m = make_hmm(cfg, seed)
        m.fit(X, lengths=lengths)
        ll = m.score(X, lengths=lengths)
        if verbose:
            print(f"    init {seed}: train log-lik = {ll:,.1f}  "
                  f"(converged={m.monitor_.converged})")
        if ll > best_ll:
            best, best_ll = m, ll
    if verbose:
        print(f"  ✓ Best train log-lik: {best_ll:,.1f}")
    return best, best_ll


def canonical_order(model: GaussianHMM, feats: list) -> np.ndarray:
    """canon_to_raw[k] = raw state index that becomes canonical state k."""
    j = feats.index(CANON_FEATURE) if CANON_FEATURE in feats else 0
    return np.argsort(model.means_[:, j])


def state_profile(model: GaussianHMM, canon_to_raw: np.ndarray, feats: list) -> dict:
    prof = {}
    for k, raw in enumerate(canon_to_raw):
        prof[str(k)] = {
            "raw_state": int(raw),
            **{f: round(float(model.means_[raw, i]), 4) for i, f in enumerate(feats)},
        }
    return prof


def fit_art(frames, cutoff, feats, cfg: Config, start=None, verbose=True,
            timeframe=None) -> dict:
    """Fit on rows in [start, cutoff) and return an artifact dict (not saved)."""
    X, lengths = build_train_matrix(frames, cutoff, feats, start)
    if verbose:
        print(f"  Fitting pooled HMM on TRAIN only: {len(X):,} obs, "
              f"{len(lengths)} sequence(s), cutoff {cutoff}  "
              f"[cov={cfg.cov_type}, min_covar={cfg.min_covar:g}, sticky={cfg.sticky:g}]")
    model, ll = fit_best_hmm(X, lengths, cfg, verbose)
    order = canonical_order(model, feats)
    return {
        "model": model,
        "feature_cols": list(feats),
        "feature_hashes": feature_hashes(feats),      # manifest: definition of each feature
        "canon_to_raw": order,
        "raw_to_canon": np.argsort(order),
        "cutoff": str(cutoff),
        "train_start": None if start is None else str(start),
        "timeframe": timeframe or cfg.timeframe,
        "symbols": list(frames.keys()),
        "cov_type": cfg.cov_type,
        "min_covar": cfg.min_covar,
        "sticky": cfg.sticky,
        "n_train": int(len(X)),
        "train_ll_per_obs": float(ll) / len(X),
        "fit_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def align_art(art: dict, ref_means: np.ndarray) -> dict:
    """
    Re-label a freshly fitted model's states to match a reference (K, F) set of
    canonical means via minimum-distance assignment, so state k means the same
    thing across walk-forward folds. Features are all in [-1, 1], so plain
    Euclidean distance is comparable across dimensions.
    """
    cur = art["model"].means_[art["canon_to_raw"]]
    cost = np.linalg.norm(ref_means[:, None, :] - cur[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    new = art["canon_to_raw"][c]
    art = dict(art)
    art["canon_to_raw"] = new
    art["raw_to_canon"] = np.argsort(new)
    art["align_dist"] = float(cost[r, c].mean())
    return art


def _check_manifest(art: dict, feats: list, model_path: str) -> None:
    """A frozen model is only valid for the feature DEFINITIONS it was trained on."""
    saved = art.get("feature_hashes")
    if saved is None:
        raise RuntimeError(
            f"Frozen model {model_path} predates the feature manifest, so its "
            f"features cannot be verified. Re-run with --refit."
        )
    cur = feature_hashes(feats)
    changed = [f for f in feats if saved.get(f) != cur[f]]
    if changed:
        raise RuntimeError(
            f"Frozen model {model_path}: definition of {changed} changed since it was "
            f"fit (indicator code, transform, scaler, window or global constants). "
            f"Re-run with --refit."
        )


def fit_or_load(frames, cutoff, cfg: Config, feats: list):
    tag = _tag(cfg, feats)
    model_path, prof_path = _paths(cfg.timeframe, tag)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(model_path) and not cfg.refit:
        art = joblib.load(model_path)
        if list(art["feature_cols"]) != list(feats):
            raise RuntimeError(
                f"Frozen model {model_path} was fit on features "
                f"{list(art['feature_cols'])}, but the active feature set is "
                f"{feats}. Re-run with --refit to train and freeze a new one."
            )
        _check_manifest(art, feats, model_path)
        print(f"  ✓ Loaded FROZEN model ← {model_path}  "
              f"(fit {art['fit_utc']}, train cutoff {art['cutoff']})")
        if pd.Timestamp(art["cutoff"]) != cutoff:
            print(f"  ⚠ Frozen model's cutoff ({art['cutoff']}) != requested "
                  f"cutoff ({cutoff}). Evaluating against the FROZEN cutoff.")
        return art

    art = fit_art(frames, cutoff, feats, cfg)
    joblib.dump(art, model_path)
    with open(prof_path, "w") as f:
        json.dump({k: v for k, v in art.items()
                   if k not in ("model", "canon_to_raw", "raw_to_canon")}
                  | {"state_profile": state_profile(art["model"], art["canon_to_raw"], feats)},
                  f, indent=2)
    print(f"  ✓ Saved frozen model → {model_path}\n  ✓ Saved profile      → {prof_path}")
    return art


# ══════════════════════════════════════════════════════════════════════════════
# Labelling (causal) + targets
# ══════════════════════════════════════════════════════════════════════════════

def label_frames(frames: dict, art: dict, min_dwell: int = 1,
                 with_viterbi: bool = True) -> pd.DataFrame:
    """
    Per symbol, per segment: causal filtered posteriors -> hard state in
    CANONICAL numbering.
      State_Raw  : argmax of the filtered posterior
      State      : State_Raw after optional causal min-dwell hysteresis
      State_Conf : filtered posterior of the reported State
    State_Viterbi (look-ahead diagnostic only) is computed when with_viterbi.
    """
    model, r2c, feats = art["model"], art["raw_to_canon"], art["feature_cols"]
    K = model.n_components
    out = []
    for sym, df in frames.items():
        if df.empty:
            continue
        df = df.copy()
        post = np.zeros((len(df), K))
        vit = np.zeros(len(df), dtype=int)
        seg_idx = df.groupby("seg").indices
        for _, idx in seg_idx.items():
            X = df.iloc[idx][feats].values
            post[idx] = forward_filter(model, X)
            if with_viterbi:
                vit[idx] = model.predict(X)             # DIAGNOSTIC ONLY (look-ahead)
        canon_post = post[:, art["canon_to_raw"]]       # columns in canonical order
        raw_state = canon_post.argmax(1)
        state = raw_state.copy()
        if min_dwell > 1:
            for _, idx in seg_idx.items():
                state[idx] = apply_min_dwell(raw_state[idx], min_dwell)
        df["State_Raw"] = raw_state
        df["State"] = state
        df["State_Conf"] = canon_post[np.arange(len(df)), state]
        if with_viterbi:
            df["State_Viterbi"] = r2c[vit]
        for k in range(K):
            df[f"P_State_{k}"] = canon_post[:, k]
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_signal(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """State_HC = State where State_Conf >= thr, else -1 (no call)."""
    df = df.copy()
    df["State_HC"] = np.where(df["State_Conf"] >= thr, df["State"], -1)
    return df


def add_forward_returns(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """
    Per symbol, for each horizon h (NaN if the h-th next candle isn't exactly
    h*CANDLE later, or runs past the end; never crosses symbols):
      fwd_h  = log(Close[t+h] / Close[t])
      fwdz_h = fwd_h / (vol_t * sqrt(h))     sigma-normalised, comparable across coins
      absz_h = |fwdz_h|                       volatility effect
    vol_t is the causal trailing std computed on the full history in
    load_symbol_frames (so walk-forward slices keep their warm-up).
    """
    if "vol" not in df.columns:
        raise KeyError("'vol' column missing — build frames with load_symbol_frames().")
    parts = []
    for _, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Open_time").copy()
        vol = g["vol"].where(g["vol"] > 0)
        for h in horizons:
            ok = (g["Open_time"].shift(-h) - g["Open_time"]) == CANDLE * h
            r = np.log(g["Close"].shift(-h) / g["Close"]).where(ok)
            g[f"fwd_{h}"] = r
            z = r / (vol * np.sqrt(h))
            g[f"fwdz_{h}"] = z
            g[f"absz_{h}"] = z.abs()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation: time x coin panel, block bootstrap, circular-shift null, BH-FDR
# ══════════════════════════════════════════════════════════════════════════════

def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs ignored and preserved)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    fin = np.where(np.isfinite(p))[0]
    if fin.size == 0:
        return out
    order = fin[np.argsort(p[fin])]
    m = len(order)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0, 1)
    return out


def _panel(df_all: pd.DataFrame, valid: pd.DataFrame, state_col: str, ycol: str, base: pd.Series):
    """
    S[T, N] int state per (global time, coin), -1 = no call / no data.
    Y[T, N] excess target per (global time, coin), NaN = no valid target.
    S is built from ALL rows (so the shift null can move states onto any slot);
    Y only from rows with a valid target.
    """
    syms = sorted(df_all["Symbol"].unique())
    si = {s: i for i, s in enumerate(syms)}
    t0 = int(df_all["tg"].min())
    T = int(df_all["tg"].max()) - t0 + 1
    S = np.full((T, len(syms)), -1, dtype=np.int16)
    S[df_all["tg"].values - t0, df_all["Symbol"].map(si).values] = df_all[state_col].values
    Y = np.full((T, len(syms)), np.nan)
    ex = valid[ycol].values - valid["Symbol"].map(base).values
    Y[valid["tg"].values - t0, valid["Symbol"].map(si).values] = ex
    return S, Y


def _group_means(S: np.ndarray, Y0: np.ndarray, F: np.ndarray, K: int):
    """
    Per state: time-average of the cross-sectional mean excess. One number per
    timestamp (mean over the coins in that state then), so one observation per
    timestamp however many coins agree. Returns (means[K], n_timestamps[K]).
    """
    means = np.full(K, np.nan)
    nts = np.zeros(K, dtype=int)
    for k in range(K):
        m = (S == k) & F
        cnt = m.sum(1)
        ok = cnt > 0
        n = int(ok.sum())
        if n:
            means[k] = ((Y0 * m).sum(1)[ok] / cnt[ok]).mean()
            nts[k] = n
    return means, nts


def _null_draws(S, Y0, F, K, h, n_null, rng, min_ts):
    """
    Circular-shift null. ONE common random offset moves every coin's state
    sequence in time (rows of S), leaving returns fixed: within-coin state
    persistence and cross-coin co-movement of states are preserved, only the
    state -> future-return link is broken. Returns (null_means[B,K], null_spread[B]).
    """
    T = S.shape[0]
    lo = max(5 * h, int(0.05 * T))
    hi = T - lo
    if n_null <= 0 or hi <= lo:
        return None, None
    offs = rng.integers(lo, hi, size=n_null)
    nm = np.full((n_null, K), np.nan)
    sp = np.full(n_null, np.nan)
    for b, o in enumerate(offs):
        m, nts = _group_means(np.roll(S, int(o), axis=0), Y0, F, K)
        nm[b] = m
        ok = nts >= min_ts
        if ok.sum() >= 2:
            sp[b] = np.nanmax(m[ok]) - np.nanmin(m[ok])
    return nm, sp


def _boot_means(S, Y0, F, K, block, n_boot, rng):
    """Moving-block bootstrap over TIME of each state's per-timestamp series -> (n_boot, K)."""
    T = S.shape[0]
    if n_boot <= 0 or T < 2 * block:
        return None
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :T]
    out = np.full((n_boot, K), np.nan)
    for k in range(K):
        m = (S == k) & F
        cnt = m.sum(1)
        ts = np.where(cnt > 0, (Y0 * m).sum(1) / np.maximum(cnt, 1), np.nan)
        samp = ts[idx]
        fin = np.isfinite(samp)
        c = fin.sum(1)
        out[:, k] = np.where(c > 0, np.where(fin, samp, 0.0).sum(1) / np.maximum(c, 1), np.nan)
    return out


def evaluate_states(df: pd.DataFrame, horizons=HORIZONS, state_col: str = "State",
                    prefix: str = TARGET, n_null: int = N_NULL, n_boot: int = N_BOOT,
                    seed: int = 0, min_ts: int = MIN_TS_STATE) -> pd.DataFrame:
    """
    Forward-target table by state, with dependence-aware inference.

    prefix : 'fwdz' (sigma-normalised, default), 'fwd' (raw log return) or
             'absz' (|sigma-normalised| = volatility effect). Clustered columns
             below are in that target's units; mean_bps / excess_bps /
             hit_rate always use the raw log return fwd_h.

    Columns
      n, n_ts, n_eff    rows, distinct timestamps in the state, n_ts / h
      mean_bps, excess_bps, hit_rate, hit_excess   pooled per-row, raw returns
      ex, ex_lo, ex_hi  time-average cross-sectional excess + 95% block-bootstrap CI
      t_boot            ex / bootstrap SE
      p_null            two-sided circular-shift p for that state
      p_adj_BH          BH-adjusted p_null across every (horizon, state) row
      spread, p_spread  best-minus-worst over states with >= min_ts timestamps,
                        and its circular-shift p (selection bias included)
    Rows with state_col < 0 (no-call) are excluded from groups; baselines are
    per-symbol means over ALL rows with a valid target.
    """
    if "tg" not in df.columns:
        raise KeyError("'tg' (global time index) missing — build frames with load_symbol_frames().")
    rng = np.random.default_rng(seed)
    present = [int(s) for s in df[state_col].unique() if s >= 0]
    if not present:
        return pd.DataFrame()
    K = max(present) + 1

    rows = []
    for h in horizons:
        ycol, rcol = f"{prefix}_{h}", f"fwd_{h}"
        if ycol not in df.columns:
            continue
        valid = df.dropna(subset=[ycol, rcol])
        if valid.empty:
            continue
        base = valid.groupby("Symbol")[ycol].mean()
        base_raw = valid.groupby("Symbol")[rcol].mean()
        base_hit = valid.groupby("Symbol")[rcol].apply(lambda s: float((s > 0).mean()))

        S, Y = _panel(df, valid, state_col, ycol, base)
        F = np.isfinite(Y)
        Y0 = np.where(F, Y, 0.0)
        means, nts = _group_means(S, Y0, F, K)
        ok = nts >= min_ts
        spread = float(np.nanmax(means[ok]) - np.nanmin(means[ok])) if ok.sum() >= 2 else np.nan

        nm, sp = _null_draws(S, Y0, F, K, h, n_null, rng, min_ts)
        p_spread = np.nan
        if sp is not None and np.isfinite(spread):
            fin = np.isfinite(sp)
            p_spread = (1 + (sp[fin] >= spread).sum()) / (1 + fin.sum())

        bm = _boot_means(S, Y0, F, K, max(2 * h, 12), n_boot, rng)

        for k in range(K):
            if nts[k] == 0:
                continue
            dk = valid[valid[state_col] == k]
            if dk.empty:
                continue
            r = dk[rcol]
            sym_base = dk["Symbol"]
            hit = float((r > 0).mean())
            p_k = np.nan
            if nm is not None and np.isfinite(means[k]):
                col = nm[:, k]
                fin = np.isfinite(col)
                p_k = (1 + (np.abs(col[fin]) >= abs(means[k])).sum()) / (1 + fin.sum())
            lo = hi = t_b = np.nan
            if bm is not None and np.isfinite(bm[:, k]).sum() > 10:
                b = bm[:, k][np.isfinite(bm[:, k])]
                lo, hi = np.percentile(b, [2.5, 97.5])
                se = b.std(ddof=1)
                t_b = means[k] / se if se > 0 else np.nan
            rows.append({
                "horizon": h, "state": k, "n": len(dk), "n_ts": int(nts[k]),
                "n_eff": nts[k] / h,
                "mean_bps": r.mean() * 1e4,
                "excess_bps": (r - sym_base.map(base_raw)).mean() * 1e4,
                "hit_rate": hit,
                "hit_excess": hit - sym_base.map(base_hit).mean(),
                "ex": means[k], "ex_lo": lo, "ex_hi": hi, "t_boot": t_b,
                "p_null": p_k, "spread": spread, "p_spread": p_spread,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj_BH"] = bh_adjust(out["p_null"].values)
    return out


def confidence_sweep(df: pd.DataFrame, h: int, thresholds=CONF_SWEEP,
                     prefix: str = TARGET, n_null: int = 300, seed: int = 0) -> pd.DataFrame:
    """For each threshold: coverage and state separation at horizon h (no bootstrap)."""
    rows = []
    for thr in thresholds:
        d = df.assign(State_HC=np.where(df["State_Conf"] >= thr, df["State"], -1))
        ev = evaluate_states(d, (h,), "State_HC", prefix, n_null=n_null, n_boot=0, seed=seed)
        if ev.empty:
            continue
        pool = ev[ev["n_ts"] >= MIN_TS_STATE]
        pool = ev if pool.empty else pool
        top, bot = pool.loc[pool["ex"].idxmax()], pool.loc[pool["ex"].idxmin()]
        rows.append({
            "conf_thr": thr,
            "coverage": float((d["State_HC"] >= 0).mean()),
            "p_spread": ev["p_spread"].iloc[0], "spread": ev["spread"].iloc[0],
            "best_state": int(top["state"]), "best_ex": top["ex"],
            "best_hit_excess": top["hit_excess"], "best_n_eff": top["n_eff"],
            "worst_state": int(bot["state"]), "worst_ex": bot["ex"],
            "worst_hit_excess": bot["hit_excess"],
        })
    return pd.DataFrame(rows)


def per_symbol_table(df: pd.DataFrame, h: int, state_col: str = "State",
                     prefix: str = TARGET) -> pd.DataFrame:
    """Mean excess target by symbol x state (target units: sigma for fwdz)."""
    ycol = f"{prefix}_{h}"
    valid = df.dropna(subset=[ycol])
    base = valid.groupby("Symbol")[ycol].mean()
    d = valid[valid[state_col] >= 0].copy()
    if d.empty:
        return pd.DataFrame()
    d["ex"] = d[ycol] - d["Symbol"].map(base)
    piv = d.pivot_table(index="Symbol", columns=state_col, values="ex", aggfunc="mean")
    piv.columns = [f"S{c}" for c in piv.columns]
    return piv


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


def tag_states(model, canon_to_raw, feats) -> dict:
    """
    Heuristic labels from the TRAIN-fit means (diagnostic only):
      chop = lowest GK_Scaled among states whose |MeanDev| is below the median
      |MeanDev| of all states; others tagged up/down by MeanDev sign.
    Falls back to generic names if GK_Scaled / MeanDev_Scaled were dropped.
    """
    K = len(canon_to_raw)
    if "GK_Scaled" not in feats or "MeanDev_Scaled" not in feats:
        return {k: f"state{k}" for k in range(K)}
    gk = feats.index("GK_Scaled")
    md = feats.index("MeanDev_Scaled")
    gk_m = np.array([model.means_[r, gk] for r in canon_to_raw])
    md_m = np.array([model.means_[r, md] for r in canon_to_raw])
    near = np.abs(md_m) <= np.median(np.abs(md_m))
    cand = np.where(near)[0] if near.any() else np.arange(K)
    chop = int(cand[np.argmin(gk_m[cand])])
    return {k: ("chop/low-vol" if k == chop else ("up" if md_m[k] > 0 else "down"))
            for k in range(K)}


# ══════════════════════════════════════════════════════════════════════════════
# Feature diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def feature_redundancy_report(frames: dict, cutoff: pd.Timestamp, feats: list) -> None:
    """Contract + cross-coin scaling + correlation / VIF / PCA (train rows only)."""
    print(f"\n{'─' * 64}\n  UNIT-FREE CONTRACT (prices x1000, volumes x7 => features unchanged)\n{'─' * 64}")
    sample = next(iter(frames.values())).tail(4000)
    rep = verify_feature_contract(sample)
    print(rep.round(8).to_string(index=False))
    print(f"\n{'─' * 64}\n  CROSS-COIN SCALING (each coin vs the pool)\n{'─' * 64}")
    cc = cross_coin_scaling_report(frames, feats)
    if not cc.empty:
        print(cc.pivot(index="symbol", columns="feature", values="std").round(3).to_string())

    X = np.vstack([df.loc[df["Open_time"] < cutoff, feats].values for df in frames.values()])
    corr = pd.DataFrame(X, columns=feats).corr()
    print(f"\n{'─' * 64}\n  FEATURE REDUNDANCY (train rows only, n={len(X):,})\n{'─' * 64}")
    print("\n  [Correlation matrix]")
    print(corr.round(2).to_string())

    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:]
             if abs(corr.loc[a, b]) >= 0.8]
    print("\n  [Pairs with |rho| >= 0.8]")
    print("  " + ("none" if not pairs else
                  "\n  ".join(f"{a} ~ {b}: {r:+.2f}" for a, b, r in pairs)))

    vifs = {}
    for j, f in enumerate(feats):
        y = X[:, j]
        A = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - (y - A @ beta).var() / max(y.var(), 1e-12)
        vifs[f] = 1.0 / max(1.0 - r2, 1e-9)
    print("\n  [VIF]  (>5 notable, >10 severe)")
    print(pd.Series(vifs, name="VIF").round(2).to_string())

    eig = np.linalg.eigvalsh(corr.values)[::-1]
    print("\n  [PCA spectrum of the correlation matrix]")
    print(pd.DataFrame({"explained": eig / eig.sum(),
                        "cumulative": np.cumsum(eig) / eig.sum()},
                       index=[f"PC{i + 1}" for i in range(len(eig))]).round(3).to_string())
    print(f"  Condition number: {eig[0] / max(eig[-1], 1e-12):,.1f}")
    print("\n  Note: with covariance_type='diag' the HMM treats features as\n"
          "  independent, so correlated features double-count evidence — redundancy\n"
          "  matters more there than under --cov-type full.")


def _summ(df, h, thr, prefix=TARGET, n_null=200) -> dict:
    s = confidence_sweep(df, h, (thr,), prefix=prefix, n_null=n_null)
    if s.empty:
        return {"p_spread": np.nan, "spread": np.nan, "coverage": np.nan}
    r = s.iloc[0]
    return {"p_spread": r["p_spread"], "spread": r["spread"], "coverage": r["coverage"]}


def run_ablation(frames: dict, cutoff: pd.Timestamp, feats: list, cfg: Config) -> None:
    """
    Drop-one-feature refits. Fit on the first 70% of TRAIN, score on the last
    30% of TRAIN (inner validation) — the test set is untouched. Log-likelihood
    isn't comparable across feature counts, so the score is the downstream
    separation of the target at the primary horizon (spread + its null p).
    """
    h = cfg.primary_h
    fr = {s: df[df["Open_time"] < cutoff].copy() for s, df in frames.items()}
    tr_ts = np.sort(np.concatenate([df["Open_time"].values for df in fr.values()]))
    val_cut = pd.Timestamp(tr_ts[int(len(tr_ts) * 0.7)])
    sub_cfg = replace(cfg, n_init=min(cfg.n_init, 3))
    print(f"\n{'─' * 64}\n  DROP-ONE ABLATION  fit < {val_cut}  |  validate {val_cut} → {cutoff}"
          f"\n  horizon={h}, target={cfg.target}, conf_thr={cfg.conf_threshold}\n{'─' * 64}")

    variants = [("ALL", feats)] + [(f"-{f}", [x for x in feats if x != f]) for f in feats]
    rows = []
    for name, fs in variants:
        print(f"  fitting {name} …")
        art = fit_art(fr, val_cut, fs, sub_cfg, verbose=False)
        lab = add_signal(add_forward_returns(
            label_frames(fr, art, cfg.min_dwell, with_viterbi=False)), cfg.conf_threshold)
        val = lab[lab["Open_time"] >= val_cut]
        a, hc = _summ(val, h, 0.0, cfg.target), _summ(val, h, cfg.conf_threshold, cfg.target)
        rows.append({"variant": name,
                     "p_spread": a["p_spread"], "spread": a["spread"],
                     "avg_dwell": dwell_times(val, cfg.n_states).mean(),
                     "HC_coverage": hc["coverage"], "HC_p_spread": hc["p_spread"],
                     "HC_spread": hc["spread"]})
    print()
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n  Read: a feature whose removal leaves spread / p_spread about the same (or better)\n"
          "  is a drop candidate. Differences within random-restart noise should not be\n"
          "  over-read. Apply with --drop-features X --refit and look at test ONCE.")


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_label(frames: dict, start: pd.Timestamp, cfg: Config, feats: list,
                       ref_means):
    """
    Refit every cfg.wf_step_months on [train_lo, fold_start) (expanding if
    wf_train_months == 0), label the next fold causally with that fold's model.
    Returns (labelled_df_without_forward_returns, folds_df, artifacts).
    """
    last_ts = max(df["Open_time"].max() for df in frames.values())
    step = pd.DateOffset(months=cfg.wf_step_months)
    parts, info, arts = [], [], []
    s, i = start, 0
    while s <= last_ts:
        e = s + step
        lo = None if cfg.wf_train_months <= 0 else s - pd.DateOffset(months=cfg.wf_train_months)
        print(f"\n  [WF fold {i}] train {lo if lo is not None else 'start'} → {s}   "
              f"test {s} → {min(e, last_ts)}")
        art = fit_art(frames, s, feats, cfg, start=lo, verbose=False)
        if ref_means is not None:
            art = align_art(art, ref_means)
        warm = s - WF_WARMUP * CANDLE
        sub = {sym: df[(df["Open_time"] >= warm) & (df["Open_time"] < e)]
               for sym, df in frames.items()}
        lab = label_frames(sub, art, cfg.min_dwell, with_viterbi=False)
        if lab.empty:
            s, i = e, i + 1
            continue
        lab = lab[lab["Open_time"] >= s].copy()
        lab["Fold"] = i
        parts.append(lab)
        tags = tag_states(art["model"], art["canon_to_raw"], feats)
        info.append({
            "fold": i, "train_start": str(lo) if lo is not None else "start",
            "train_end": str(s), "test_end": str(min(e, last_ts)),
            "n_train": art["n_train"], "n_test": len(lab),
            "train_ll_per_obs": round(art["train_ll_per_obs"], 4),
            "align_dist": round(art.get("align_dist", 0.0), 4),
            "tags": " | ".join(f"S{k}:{v}" for k, v in tags.items()),
        })
        print(f"      n_train={art['n_train']:,}  n_test={len(lab):,}  "
              f"align_dist={art.get('align_dist', 0.0):.3f}  {info[-1]['tags']}")
        arts.append(art)
        s, i = e, i + 1
    if not parts:
        raise RuntimeError("Walk-forward produced no folds.")
    return pd.concat(parts, ignore_index=True), pd.DataFrame(info), arts


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

_SHOW = ["horizon", "state", "n", "n_ts", "n_eff", "mean_bps", "excess_bps",
         "hit_rate", "hit_excess", "ex", "ex_lo", "ex_hi", "t_boot",
         "p_null", "p_adj_BH", "spread", "p_spread"]


def _print_eval(title: str, ev: pd.DataFrame, note: str = "") -> None:
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}")
    if note:
        print(f"  {note}")
    print("  (no rows)" if ev.empty else ev[_SHOW].round(3).to_string(index=False))


def run_pooled_hmm(cfg: Config):
    feats = cfg.features()
    unit = "sigma units" if cfg.target == "fwdz" else "log-return units"
    print(f"\n{'=' * 64}\n  POOLED HMM — STEP 1: OUT-OF-SAMPLE STATE CHECK\n{'=' * 64}")
    print(f"  features={feats}\n  K={cfg.n_states} cov={cfg.cov_type} "
          f"min_covar={cfg.min_covar:g} sticky={cfg.sticky:g} "
          f"min_dwell={cfg.min_dwell} conf_thr={cfg.conf_threshold}\n"
          f"  target={cfg.target} ({unit})  null draws={cfg.n_null}  boot draws={cfg.n_boot}")
    frames = load_symbol_frames(SYMBOLS, cfg.timeframe, frozen=cfg.frozen_data, feats=feats)
    if not frames:
        print("❌ No data loaded.")
        return

    cutoff = global_cutoff(frames, cfg.train_frac)
    print(f"\n  Global train/test cutoff: {cutoff}  (train_frac={cfg.train_frac})")

    if cfg.feature_check:
        feature_redundancy_report(frames, cutoff, feats)
        return
    cross_coin_scaling_report(frames, feats)            # prints only flagged coins/features
    if cfg.ablate:
        run_ablation(frames, cutoff, feats, cfg)
        return

    art = fit_or_load(frames, cutoff, cfg, feats)
    cutoff = pd.Timestamp(art["cutoff"])                # trust the frozen cutoff
    model = art["model"]
    K = model.n_components
    h = cfg.primary_h
    thr = cfg.conf_threshold
    ev_kw = dict(prefix=cfg.target, n_null=cfg.n_null, n_boot=cfg.n_boot)
    clus = (f"ex / ex_lo / ex_hi in {unit}; per-timestamp cross-sectional means, "
            f"block-bootstrap CI, circular-shift p; p_adj_BH over this table.")

    print(f"\n  [Canonical state profile — TRAIN-fit means, K={K}]")
    prof_df = pd.DataFrame(state_profile(model, art["canon_to_raw"], feats)).T.drop(columns="raw_state")
    tags_ = tag_states(model, art["canon_to_raw"], feats)
    prof_df["tag"] = [tags_[k] for k in range(K)]
    print(prof_df.round(3).to_string())

    print("\n  [Transition matrix, canonical order]")
    o = art["canon_to_raw"]
    print(pd.DataFrame(model.transmat_[np.ix_(o, o)],
                       index=[f"S{i}" for i in range(K)],
                       columns=[f"S{i}" for i in range(K)]).round(3).to_string())

    # Causal labels over the full series, then split
    lab = add_signal(add_forward_returns(label_frames(frames, art, cfg.min_dwell)), thr)
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
    agree = (test["State_Raw"] == test["State_Viterbi"]).mean()
    print(f"\n  Causal argmax vs Viterbi agreement on test: {agree:.1%}  "
          f"(gap from 100% = how much look-ahead was changing your labels)")

    print(f"\n  [High-conviction coverage, thr={thr}]  share of candles with a call")
    cov = pd.concat([(train["State_HC"] >= 0).groupby(train["State"]).mean().rename("train"),
                     (test["State_HC"] >= 0).groupby(test["State"]).mean().rename("test")], axis=1)
    cov.loc["ALL"] = [(train["State_HC"] >= 0).mean(), (test["State_HC"] >= 0).mean()]
    print(cov.round(3).to_string())

    # The actual question: direction, unfiltered vs high-conviction
    ev_test, ev_test_hc = evaluate_states(test, **ev_kw), evaluate_states(test, state_col="State_HC", **ev_kw)
    ev_train, ev_train_hc = evaluate_states(train, **ev_kw), evaluate_states(train, state_col="State_HC", **ev_kw)
    _print_eval("OUT-OF-SAMPLE (test) — ALL candles, causal state — DIRECTION", ev_test, clus)
    _print_eval(f"OUT-OF-SAMPLE (test) — HIGH-CONVICTION only (conf >= {thr}) — DIRECTION", ev_test_hc, clus)
    _print_eval("IN-SAMPLE (train) — ALL candles, for comparison only", ev_train, clus)
    _print_eval(f"IN-SAMPLE (train) — HIGH-CONVICTION only (conf >= {thr})", ev_train_hc, clus)

    # Volatility effect, reported separately from direction
    ev_vol = evaluate_states(test, prefix="absz", n_null=cfg.n_null, n_boot=cfg.n_boot)
    _print_eval("OUT-OF-SAMPLE (test) — VOLATILITY effect: |sigma-normalised fwd return|",
                ev_vol, "ex > 0 => the state is followed by LARGER moves than that coin's norm "
                        "(not a direction). The HMM sees GK/RealVol, so expect this to be significant.")

    print(f"\n{'─' * 64}\n  CONFIDENCE SWEEP, horizon={h}, target={cfg.target}  "
          f"(pick the threshold on TRAIN; test is confirmation only)\n{'─' * 64}")
    for name, d in (("train", train), ("test", test)):
        sw = confidence_sweep(d, h, prefix=cfg.target)
        print(f"\n  [{name}]")
        print("  (no rows)" if sw.empty else sw.round(3).to_string(index=False))

    piv = per_symbol_table(test, h, "State_HC", cfg.target)
    print(f"\n{'─' * 64}\n  Per-symbol excess ({unit}), test, HIGH-CONVICTION, horizon={h}\n"
          f"  (an edge worth trusting has the SAME sign pattern across most symbols)\n{'─' * 64}")
    print("  (no rows)" if piv.empty else piv.round(3).to_string())

    p = ev_test[ev_test["horizon"] == h]
    if not p.empty:
        best_adj = p["p_adj_BH"].min()
        print(f"\n{'─' * 64}\n  READ (PRIMARY horizon={h}, ALL candles, {unit}):\n"
              f"    spread best-worst = {p['spread'].iloc[0]:.3f}   "
              f"p_spread (circular-shift, selection bias included) = {p['p_spread'].iloc[0]:.4f}\n"
              f"    smallest BH-adjusted per-state p across the whole table = {best_adj:.4f}\n"
              f"  Proceed only if: p_spread is small at the PRIMARY horizon (other horizons are\n"
              f"  exploratory), the bootstrap CI of at least one state excludes 0, the pattern\n"
              f"  holds across symbols AND survives train -> test (same sign/order), and the\n"
              f"  volatility table above is not the only thing separating the states.\n{'─' * 64}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = _tag(cfg, feats)
    ev_test.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_test.csv"), index=False)
    ev_test_hc.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_test_hc.csv"), index=False)
    ev_train.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_train.csv"), index=False)
    ev_train_hc.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_train_hc.csv"), index=False)
    ev_vol.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_test_volatility.csv"), index=False)
    tcols = [f"fwd_{x}" for x in HORIZONS] + [f"fwdz_{x}" for x in HORIZONS]
    keep = ["Open_time", "Symbol", "Close", "State", "State_Raw", "State_Conf", "State_HC",
            "State_Viterbi"] + [f"P_State_{k}" for k in range(K)] + tcols
    lab[keep].assign(Split=np.where(lab["Open_time"] < cutoff, "train", "test")) \
        .to_csv(os.path.join(OUT_DIR, "universal_crypto_pooled_hmm_states_causal.csv"), index=False)
    print(f"\n  ✓ Saved eval tables + causal labelled dataset to {OUT_DIR}/")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    if cfg.walk_forward:
        print(f"\n{'=' * 64}\n  WALK-FORWARD  (step={cfg.wf_step_months}m, "
              f"{'expanding' if cfg.wf_train_months <= 0 else f'sliding {cfg.wf_train_months}m'}, "
              f"first test fold = static cutoff)\n{'=' * 64}")
        ref = model.means_[art["canon_to_raw"]]
        wf_raw, folds, arts = walk_forward_label(frames, cutoff, cfg, feats, ref)
        wf = add_signal(add_forward_returns(wf_raw), thr)
        ev_wf, ev_wf_hc = evaluate_states(wf, **ev_kw), evaluate_states(wf, state_col="State_HC", **ev_kw)

        print(f"\n  [Folds]")
        print(folds.drop(columns=["tags"]).to_string(index=False))
        _print_eval("WALK-FORWARD OOS — ALL candles", ev_wf, clus)
        _print_eval(f"WALK-FORWARD OOS — HIGH-CONVICTION (conf >= {thr})", ev_wf_hc, clus)

        cmp_rows = []
        for name, d, t in (("static / all", test, 0.0), (f"static / HC>={thr}", test, thr),
                           ("walk-fwd / all", wf, 0.0), (f"walk-fwd / HC>={thr}", wf, thr)):
            r = _summ(d, h, t, cfg.target)
            r["variant"] = name
            cmp_rows.append(r)
        print(f"\n{'─' * 64}\n  STATIC vs WALK-FORWARD — same OOS window, horizon={h}\n{'─' * 64}")
        print(pd.DataFrame(cmp_rows)[["variant", "coverage", "p_spread", "spread"]]
              .round(4).to_string(index=False))
        print(f"\n  [Walk-forward per-symbol excess ({unit}), HC, horizon={h}]")
        pw = per_symbol_table(wf, h, "State_HC", cfg.target)
        print("  (no rows)" if pw.empty else pw.round(3).to_string())

        folds.to_csv(os.path.join(OUT_DIR, f"hmm_wf_folds_{tag}.csv"), index=False)
        ev_wf.to_csv(os.path.join(OUT_DIR, "hmm_wf_eval.csv"), index=False)
        ev_wf_hc.to_csv(os.path.join(OUT_DIR, "hmm_wf_eval_hc.csv"), index=False)
        wf[keep[:3] + ["State", "State_Raw", "State_Conf", "State_HC", "Fold"]
           + [f"P_State_{k}" for k in range(K)] + tcols] \
            .to_csv(os.path.join(OUT_DIR, "universal_crypto_pooled_hmm_states_walkforward.csv"),
                    index=False)
        joblib.dump(arts, os.path.join(MODEL_DIR, f"pooled_hmm_{cfg.timeframe}_{tag}_walkforward.joblib"))
        print(f"\n  ✓ Saved WF folds / eval / labels to {OUT_DIR}/ and fold models to {MODEL_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="retrain and overwrite frozen model")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--primary-horizon", type=int, default=6, choices=HORIZONS)
    ap.add_argument("--timeframe", default=TIMEFRAME)
    ap.add_argument("--n-states", type=int, default=N_STATES)
    ap.add_argument("--n-init", type=int, default=N_INIT, help="random restarts per fit")
    ap.add_argument("--rescan-raw", action="store_true",
                    help="let update_master_data() rescan raw files (default: frozen masters)")
    # confidence
    ap.add_argument("--conf-threshold", type=float, default=0.75,
                    help="posterior needed for a high-conviction call (State_HC)")
    # walk-forward
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--wf-step-months", type=int, default=6)
    ap.add_argument("--wf-train-months", type=int, default=0,
                    help="0 = expanding window; N = sliding window of N months")
    # duration control
    ap.add_argument("--sticky", type=float, default=0.0,
                    help="pseudo-counts added to the transition diagonal (try 100-1000)")
    ap.add_argument("--min-dwell", type=int, default=1,
                    help="candles a new state must persist before being accepted (causal)")
    # features
    ap.add_argument("--feature-check", action="store_true",
                    help="contract test, cross-coin scaling, correlation/VIF/PCA (no fit)")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help="feature names, with or without the _Scaled suffix")
    ap.add_argument("--extra-features", action="store_true",
                    help=f"also use the opt-in registry features {OPTIONAL_FEATURES}")
    ap.add_argument("--legacy-features", action="store_true",
                    help=f"the old six-feature set {LEGACY_FEATURES}")
    # covariance
    ap.add_argument("--cov-type", choices=["diag", "full"], default=COV_TYPE)
    ap.add_argument("--min-covar", type=float, default=1e-3,
                    help="variance floor added/enforced by hmmlearn")
    # evaluation
    ap.add_argument("--target", choices=["fwdz", "fwd"], default=TARGET,
                    help="fwdz = sigma-normalised forward return (default); fwd = raw log return")
    ap.add_argument("--n-null", type=int, default=N_NULL, help="circular-shift null draws")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="block-bootstrap draws")
    a = ap.parse_args()

    run_pooled_hmm(Config(
        timeframe=a.timeframe, n_states=a.n_states, n_init=a.n_init,
        train_frac=a.train_frac, primary_h=a.primary_horizon, refit=a.refit,
        frozen_data=not a.rescan_raw, cov_type=a.cov_type, min_covar=a.min_covar,
        sticky=a.sticky, min_dwell=a.min_dwell, conf_threshold=a.conf_threshold,
        walk_forward=a.walk_forward, wf_step_months=a.wf_step_months,
        wf_train_months=a.wf_train_months, drop_features=tuple(a.drop_features),
        extra_features=a.extra_features, legacy_features=a.legacy_features,
        feature_check=a.feature_check, ablate=a.ablate,
        target=a.target, n_null=a.n_null, n_boot=a.n_boot,
    ))


# # Optional: feature diagnostics (train rows only, no model fit)
# python pooled_hmm_pipeline.py --feature-check
# python pooled_hmm_pipeline.py --ablate
#
# # Fit, freeze, and evaluate the baseline model
# python pooled_hmm_pipeline.py --refit
#
# # Re-evaluate the frozen model with a different confidence threshold (no refit)
# python pooled_hmm_pipeline.py --conf-threshold 0.8
#
# # Walk-forward comparison against the static model
# python pooled_hmm_pipeline.py --walk-forward
#
# # Optional variants (each saves under its own artifact name)
# python pooled_hmm_pipeline.py --cov-type full --min-covar 1e-3 --refit
# python pooled_hmm_pipeline.py --sticky 500 --min-dwell 2 --refit