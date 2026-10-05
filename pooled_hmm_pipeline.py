"""
pooled_hmm_pipeline.py
─────────────────────────────────────────────────────────────
Pooled GaussianHMM across symbols — STEP 1: do the states predict anything
out-of-sample?

CHANGES IN THIS REVISION
────────────────────────
1. POSTERIOR CONFIDENCE FILTERING
   Every candle still gets a causal filtered state (State) and its posterior
   (State_Conf), but a new column State_HC = State if State_Conf >= threshold
   else -1 ("no call"). Evaluation tables are produced both unfiltered and
   high-conviction, plus a threshold sweep (coverage / KW p / spread / hit
   rates) on train AND test so the threshold can be chosen on train and merely
   *confirmed* on test.  --conf-threshold (default 0.75).
   Bug fixed while doing this: the "excess return" baseline is now the
   per-symbol mean over ALL candles with a valid forward return, not over the
   filtered subset (a subset baseline silently forces excess returns to
   sum to ~0 and flatters the filter).

2. WALK-FORWARD RETRAINING   (--walk-forward)
   Refits the HMM every --wf-step-months (default 6) on an expanding window
   (or sliding, --wf-train-months N). The first out-of-sample fold starts at
   the same cutoff as the static model, so the WF vs static comparison covers
   exactly the same test period.  The HMM is fit on features only (never on
   returns), so no embargo is needed.  Each refit is label-aligned to the
   static model's state means (Hungarian matching) so "State 2" keeps the same
   meaning across folds instead of drifting with the sort order.
   Filtering for a fold starts WF_WARMUP candles before the fold.

3. DURATION CONTROL (NOT a full HSMM)
   hmmlearn has no explicit-duration model, and no maintained library covers
   multivariate-continuous-emission HSMMs, so a true HSMM would be a custom
   EM implementation — not done here. Two cheap, honest approximations:
     --sticky KAPPA   Dirichlet prior adding KAPPA pseudo-counts to the
                      transition-matrix diagonal during EM (longer dwell).
     --min-dwell M    causal hysteresis: a state change is only accepted
                      after the new argmax state persisted M consecutive
                      candles (costs M-1 candles of lag; uses no future data).

4. FEATURE DIAGNOSTICS
   --feature-check  correlation matrix, VIF, PCA spectrum (train data only).
   --ablate         drop-one-feature refits scored on an inner validation
                    slice carved from TRAIN, so the test set is not used for
                    feature selection.
   --drop-features  apply the result.  --extra-features adds optional
                    columns (OPTIONAL_FEATURES) if present in the masters,
                    e.g. a taker-buy imbalance built in preprocessing.py.

5. COVARIANCE OPTIONS
   --cov-type {diag,full}  and  --min-covar (variance floor).  Artifact names
   encode non-default settings (k4_full_..., _sticky..., _f<hash>) so
   variants coexist.  Default settings reproduce the previous artifact name.

Unchanged: no look-ahead (forward filtering), fit-once-then-freeze for the
static model, one global calendar train/test cutoff, canonical state ordering,
non-overlapping-subsample significance tests.

Run:
    python pooled_hmm_pipeline.py --refit
    python pooled_hmm_pipeline.py --conf-threshold 0.8
    python pooled_hmm_pipeline.py --walk-forward
    python pooled_hmm_pipeline.py --feature-check
    python pooled_hmm_pipeline.py --ablate
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
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

from data.data_manager import update_master_data, SYMBOLS

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "RSI_Scaled", "MACD_Scaled",
    "OBV_Scaled", "ATR_Scaled", "GK_Scaled", "MeanDev_Scaled",
]
# Used only with --extra-features; must already exist in the processed masters.
# Funding / liquidation data are NOT in Binance kline files, so they need a
# separate download; taker-buy imbalance (order-flow proxy) IS derivable from
# the kline columns Taker_buy_base_vol / Volume.
OPTIONAL_FEATURES = ["TakerImb_Scaled"]

CANON_FEATURE = "MeanDev_Scaled"        # states are ordered by this feature's mean
TIMEFRAME     = "4h"
CANDLE        = pd.Timedelta(hours=4)
N_STATES      = 4
N_INIT        = 5                       # random restarts; best train log-lik wins
HORIZONS      = (1, 3, 6, 12, 24)       # candles (4h -> 4h, 12h, 1d, 2d, 4d)
CONF_SWEEP    = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
WF_WARMUP     = 200                     # candles of filter warm-up before each WF fold
MODEL_DIR     = "models"
OUT_DIR       = "data/processed"


@dataclass
class Config:
    timeframe: str = TIMEFRAME
    n_states: int = N_STATES
    n_init: int = N_INIT
    train_frac: float = 0.70
    primary_h: int = 6
    refit: bool = False
    frozen_data: bool = True
    cov_type: str = "diag"
    min_covar: float = 1e-3
    sticky: float = 0.0
    min_dwell: int = 1
    conf_threshold: float = 0.75
    walk_forward: bool = False
    wf_step_months: int = 6
    wf_train_months: int = 0            # 0 = expanding window
    drop_features: tuple = ()
    extra_features: bool = False
    feature_check: bool = False
    ablate: bool = False

    def features(self) -> list:
        feats = list(FEATURE_COLS)
        if self.extra_features:
            feats += [f for f in OPTIONAL_FEATURES if f not in feats]
        feats = [f for f in feats if f not in self.drop_features]
        if len(feats) < 2:
            raise ValueError("Need at least 2 features after --drop-features.")
        return feats


def _tag(cfg: Config, feats: list) -> str:
    """Artifact tag. Default settings -> 'k<K>' (same name as before)."""
    t = f"k{cfg.n_states}"
    if cfg.cov_type != "diag":
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
    Research-time loader (reads the processed masters). frozen=True means no
    raw-file scan / master rewrite. NOT for live use.
    Each frame gets a `seg` id that increments at any gap > 1.5 candles, so
    the HMM never treats a data hole as consecutive candles.
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
                f"Master CSV for {sym}/{timeframe} is missing feature column(s) "
                f"{missing}. Either the master predates the rolling-"
                f"standardization / Garman-Klass change (back it up, then "
                f"rebuild with --rescan-raw), or it is an optional feature "
                f"(e.g. TakerImb_Scaled) that preprocessing.py doesn't "
                f"produce yet."
            )

        df = df.dropna(subset=feats).copy()
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
# Labelling (causal) + diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def label_frames(frames: dict, art: dict, min_dwell: int = 1,
                 with_viterbi: bool = True) -> pd.DataFrame:
    """
    Per symbol, per segment: causal filtered posteriors -> hard state in
    CANONICAL numbering.
      State_Raw  : argmax of the filtered posterior
      State      : State_Raw after optional causal min-dwell hysteresis
      State_Conf : filtered posterior of the reported State (== max posterior
                   when min_dwell == 1)
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


def evaluate_states(df: pd.DataFrame, horizons=HORIZONS, state_col: str = "State") -> pd.DataFrame:
    """
    Forward-return table by state. Rows with state_col < 0 (no-call) are
    excluded from the groups, but the excess-return BASELINE (per-symbol mean
    forward return) is always taken over ALL rows with a valid forward return,
    so filtering cannot shift the baseline.
    """
    rows = []
    for h in horizons:
        col = f"fwd_{h}"
        valid = df.dropna(subset=[col])
        if valid.empty:
            continue
        base = valid.groupby("Symbol")[col].mean()
        d = valid[valid[state_col] >= 0].copy()
        if d.empty:
            continue
        d["ex"] = d[col] - d["Symbol"].map(base)
        sub = d[d["t_idx"] % h == 0]                   # non-overlapping subsample
        groups = [g["ex"].values for _, g in sub.groupby(state_col) if len(g) > 5]
        try:
            kw_p = stats.kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
        except ValueError:
            kw_p = np.nan
        for s, g in d.groupby(state_col):
            gs = sub[sub[state_col] == s]["ex"]
            t = (gs.mean() / (gs.std(ddof=1) / np.sqrt(len(gs)))) if len(gs) > 5 else np.nan
            rows.append({
                "horizon": h, "state": int(s), "n": len(g), "n_nonoverlap": len(gs),
                "mean_bps": g[col].mean() * 1e4,
                "excess_bps": g["ex"].mean() * 1e4,
                "median_bps": g[col].median() * 1e4,
                "hit_rate": (g[col] > 0).mean(),
                "t_excess": t,
                "KW_p_all_states": kw_p,
            })
    return pd.DataFrame(rows)


def confidence_sweep(df: pd.DataFrame, h: int, thresholds=CONF_SWEEP) -> pd.DataFrame:
    """For each threshold: coverage and the state-separation stats at horizon h."""
    rows = []
    for thr in thresholds:
        d = df.assign(State_HC=np.where(df["State_Conf"] >= thr, df["State"], -1))
        ev = evaluate_states(d, (h,), "State_HC")
        if ev.empty:
            continue
        top, bot = ev.loc[ev["excess_bps"].idxmax()], ev.loc[ev["excess_bps"].idxmin()]
        rows.append({
            "conf_thr": thr,
            "coverage": float((d["State_HC"] >= 0).mean()),
            "KW_p": ev["KW_p_all_states"].iloc[0],
            "spread_bps": top["excess_bps"] - bot["excess_bps"],
            "best_state": int(top["state"]), "best_excess_bps": top["excess_bps"],
            "best_hit": top["hit_rate"], "best_t": top["t_excess"],
            "worst_state": int(bot["state"]), "worst_excess_bps": bot["excess_bps"],
            "worst_hit": bot["hit_rate"],
        })
    return pd.DataFrame(rows)


def per_symbol_table(df: pd.DataFrame, h: int, state_col: str = "State") -> pd.DataFrame:
    col = f"fwd_{h}"
    valid = df.dropna(subset=[col])
    base = valid.groupby("Symbol")[col].mean()
    d = valid[valid[state_col] >= 0].copy()
    if d.empty:
        return pd.DataFrame()
    d["ex"] = d[col] - d["Symbol"].map(base)
    piv = d.pivot_table(index="Symbol", columns=state_col, values="ex", aggfunc="mean") * 1e4
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
    """Correlation / VIF / PCA on TRAIN rows only (test never informs features)."""
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


def _summ(df, h, thr) -> dict:
    s = confidence_sweep(df, h, (thr,))
    if s.empty:
        return {"KW_p": np.nan, "spread_bps": np.nan, "coverage": np.nan}
    r = s.iloc[0]
    return {"KW_p": r["KW_p"], "spread_bps": r["spread_bps"], "coverage": r["coverage"]}


def run_ablation(frames: dict, cutoff: pd.Timestamp, feats: list, cfg: Config) -> None:
    """
    Drop-one-feature refits. Fit on the first 70% of TRAIN, score on the last
    30% of TRAIN (inner validation) — the test set is untouched. Log-likelihood
    isn't comparable across different feature counts, so the score is the
    downstream separation of forward returns at the primary horizon.
    """
    h = cfg.primary_h
    fr = {s: df[df["Open_time"] < cutoff].copy() for s, df in frames.items()}
    tr_ts = np.sort(np.concatenate([df["Open_time"].values for df in fr.values()]))
    val_cut = pd.Timestamp(tr_ts[int(len(tr_ts) * 0.7)])
    sub_cfg = replace(cfg, n_init=min(cfg.n_init, 3))
    print(f"\n{'─' * 64}\n  DROP-ONE ABLATION  fit < {val_cut}  |  validate {val_cut} → {cutoff}"
          f"\n  horizon={h}, conf_thr={cfg.conf_threshold}\n{'─' * 64}")

    variants = [("ALL", feats)] + [(f"-{f}", [x for x in feats if x != f]) for f in feats]
    rows = []
    for name, fs in variants:
        print(f"  fitting {name} …")
        art = fit_art(fr, val_cut, fs, sub_cfg, verbose=False)
        lab = add_signal(add_forward_returns(
            label_frames(fr, art, cfg.min_dwell, with_viterbi=False)), cfg.conf_threshold)
        val = lab[lab["Open_time"] >= val_cut]
        a, hc = _summ(val, h, 0.0), _summ(val, h, cfg.conf_threshold)
        rows.append({"variant": name,
                     "KW_p": a["KW_p"], "spread_bps": a["spread_bps"],
                     "avg_dwell": dwell_times(val, cfg.n_states).mean(),
                     "HC_coverage": hc["coverage"], "HC_KW_p": hc["KW_p"],
                     "HC_spread_bps": hc["spread_bps"]})
    print()
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n  Read: a feature whose removal leaves spread/KW p about the same (or better)\n"
          "  is a drop candidate. Differences within random-restart noise should not be\n"
          "  over-read. Apply with --drop-features X --refit and look at test ONCE.")


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_label(frames: dict, start: pd.Timestamp, cfg: Config, feats: list,
                       ref_means: np.ndarray | None):
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

_SHOW = ["horizon", "state", "n", "n_nonoverlap", "mean_bps", "excess_bps",
         "hit_rate", "t_excess", "KW_p_all_states"]


def _print_eval(title: str, ev: pd.DataFrame) -> None:
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}")
    print("  (no rows)" if ev.empty else ev[_SHOW].round(3).to_string(index=False))


def run_pooled_hmm(cfg: Config):
    feats = cfg.features()
    print(f"\n{'=' * 64}\n  POOLED HMM — STEP 1: OUT-OF-SAMPLE STATE CHECK\n{'=' * 64}")
    print(f"  features={feats}\n  K={cfg.n_states} cov={cfg.cov_type} "
          f"min_covar={cfg.min_covar:g} sticky={cfg.sticky:g} "
          f"min_dwell={cfg.min_dwell} conf_thr={cfg.conf_threshold}")
    frames = load_symbol_frames(SYMBOLS, cfg.timeframe, frozen=cfg.frozen_data, feats=feats)
    if not frames:
        print("❌ No data loaded.")
        return

    cutoff = global_cutoff(frames, cfg.train_frac)
    print(f"\n  Global train/test cutoff: {cutoff}  (train_frac={cfg.train_frac})")

    if cfg.feature_check:
        feature_redundancy_report(frames, cutoff, feats)
        return
    if cfg.ablate:
        run_ablation(frames, cutoff, feats, cfg)
        return

    art = fit_or_load(frames, cutoff, cfg, feats)
    cutoff = pd.Timestamp(art["cutoff"])                # trust the frozen cutoff
    model = art["model"]
    K = model.n_components
    h = cfg.primary_h
    thr = cfg.conf_threshold

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

    # The actual question: unfiltered vs high-conviction
    ev_test, ev_test_hc = evaluate_states(test), evaluate_states(test, state_col="State_HC")
    ev_train, ev_train_hc = evaluate_states(train), evaluate_states(train, state_col="State_HC")
    _print_eval("OUT-OF-SAMPLE (test) — ALL candles, causal state", ev_test)
    _print_eval(f"OUT-OF-SAMPLE (test) — HIGH-CONVICTION only (conf >= {thr})", ev_test_hc)
    _print_eval("IN-SAMPLE (train) — ALL candles, for comparison only", ev_train)
    _print_eval(f"IN-SAMPLE (train) — HIGH-CONVICTION only (conf >= {thr})", ev_train_hc)

    print(f"\n{'─' * 64}\n  CONFIDENCE SWEEP, horizon={h}  "
          f"(pick the threshold on TRAIN; test is confirmation only)\n{'─' * 64}")
    for name, d in (("train", train), ("test", test)):
        sw = confidence_sweep(d, h)
        print(f"\n  [{name}]")
        print("  (no rows)" if sw.empty else sw.round(3).to_string(index=False))

    piv = per_symbol_table(test, h, "State_HC")
    print(f"\n{'─' * 64}\n  Per-symbol excess return (bps), test, HIGH-CONVICTION, horizon={h}\n"
          f"  (an edge worth trusting has the SAME sign pattern across most symbols)\n{'─' * 64}")
    print("  (no rows)" if piv.empty else piv.round(1).to_string())

    p = ev_test[ev_test["horizon"] == h]
    if not p.empty:
        print(f"\n{'─' * 64}\n  READ (horizon={h}, ALL candles): "
              f"KW p={p['KW_p_all_states'].iloc[0]:.4f}  |  "
              f"excess spread best-worst = {p['excess_bps'].max() - p['excess_bps'].min():.1f} bps\n"
              f"  Proceed only if: spread is economically meaningful vs costs, |t| is large\n"
              f"  for at least one state on non-overlapping data, the pattern holds across\n"
              f"  symbols, AND it survives from train to test (same sign/order).\n{'─' * 64}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = _tag(cfg, feats)
    ev_test.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_test.csv"), index=False)
    ev_test_hc.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_test_hc.csv"), index=False)
    ev_train.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_train.csv"), index=False)
    ev_train_hc.to_csv(os.path.join(OUT_DIR, "hmm_oos_eval_train_hc.csv"), index=False)
    keep = ["Open_time", "Symbol", "Close", "State", "State_Raw", "State_Conf", "State_HC",
            "State_Viterbi"] + [f"P_State_{k}" for k in range(K)] + [f"fwd_{x}" for x in HORIZONS]
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
        ev_wf, ev_wf_hc = evaluate_states(wf), evaluate_states(wf, state_col="State_HC")

        print(f"\n  [Folds]")
        print(folds.drop(columns=["tags"]).to_string(index=False))
        _print_eval("WALK-FORWARD OOS — ALL candles", ev_wf)
        _print_eval(f"WALK-FORWARD OOS — HIGH-CONVICTION (conf >= {thr})", ev_wf_hc)

        cmp_rows = []
        for name, d, t in (("static / all", test, 0.0), (f"static / HC>={thr}", test, thr),
                           ("walk-fwd / all", wf, 0.0), (f"walk-fwd / HC>={thr}", wf, thr)):
            r = _summ(d, h, t)
            r["variant"] = name
            cmp_rows.append(r)
        print(f"\n{'─' * 64}\n  STATIC vs WALK-FORWARD — same OOS window, horizon={h}\n{'─' * 64}")
        print(pd.DataFrame(cmp_rows)[["variant", "coverage", "KW_p", "spread_bps"]]
              .round(4).to_string(index=False))
        print(f"\n  [Walk-forward per-symbol excess (bps), HC, horizon={h}]")
        pw = per_symbol_table(wf, h, "State_HC")
        print("  (no rows)" if pw.empty else pw.round(1).to_string())

        folds.to_csv(os.path.join(OUT_DIR, f"hmm_wf_folds_{tag}.csv"), index=False)
        ev_wf.to_csv(os.path.join(OUT_DIR, "hmm_wf_eval.csv"), index=False)
        ev_wf_hc.to_csv(os.path.join(OUT_DIR, "hmm_wf_eval_hc.csv"), index=False)
        wf[keep[:3] + ["State", "State_Raw", "State_Conf", "State_HC", "Fold"]
           + [f"P_State_{k}" for k in range(K)] + [f"fwd_{x}" for x in HORIZONS]] \
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
    # 1. confidence
    ap.add_argument("--conf-threshold", type=float, default=0.75,
                    help="posterior needed for a high-conviction call (State_HC)")
    # 2. walk-forward
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--wf-step-months", type=int, default=6)
    ap.add_argument("--wf-train-months", type=int, default=0,
                    help="0 = expanding window; N = sliding window of N months")
    # 3. duration control
    ap.add_argument("--sticky", type=float, default=0.0,
                    help="pseudo-counts added to the transition diagonal (try 100-1000)")
    ap.add_argument("--min-dwell", type=int, default=1,
                    help="candles a new state must persist before being accepted (causal)")
    # 4. features
    ap.add_argument("--feature-check", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--drop-features", nargs="*", default=[])
    ap.add_argument("--extra-features", action="store_true",
                    help=f"also use {OPTIONAL_FEATURES} (must exist in the masters)")
    # 5. covariance
    ap.add_argument("--cov-type", choices=["diag", "full"], default="diag")
    ap.add_argument("--min-covar", type=float, default=1e-3,
                    help="variance floor added/enforced by hmmlearn")
    a = ap.parse_args()

    run_pooled_hmm(Config(
        timeframe=a.timeframe, n_states=a.n_states, n_init=a.n_init,
        train_frac=a.train_frac, primary_h=a.primary_horizon, refit=a.refit,
        frozen_data=not a.rescan_raw, cov_type=a.cov_type, min_covar=a.min_covar,
        sticky=a.sticky, min_dwell=a.min_dwell, conf_threshold=a.conf_threshold,
        walk_forward=a.walk_forward, wf_step_months=a.wf_step_months,
        wf_train_months=a.wf_train_months, drop_features=tuple(a.drop_features),
        extra_features=a.extra_features, feature_check=a.feature_check, ablate=a.ablate,
    ))