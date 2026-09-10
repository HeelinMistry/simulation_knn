"""
main_gbt.py
─────────────────
Training entry point for the calibrated GBT meta-labeling agent —
an alternative pipeline to main_mcknn.py's k-NN vote-share approach.

Pipeline
──────────
  1. Build the same state vectors main_mcknn.py builds (4h aggregator +
     optional 15m/1h context via multi_timeframe_state.py) — reuses
     StateAggregator/data_manager/build_multi_timeframe_context exactly
     as-is, so the two pipelines are directly comparable on identical
     features.
  2. Label every tick with triple_barrier.build_meta_labels() — price-
     path-only labels, independent of any policy's exit timing.
  3. Run a MULTI-SEED CPCV STABILITY CHECK (new this revision — see
     below), then Combinatorial Purged CV (cpcv.py) at a representative
     seed: fit a fresh GBTAgent per CPCV path's train split, evaluate on
     its test split, collect the full distribution, and compute
     Probability of Backtest Overfitting (PBO).
  4. If the CPCV distribution passes a stability gate (directional
     consistency + downside floor + PBO ≤ 50%, mirroring main_mcknn.py's
     gate philosophy), fit the final deployable model on all data up to
     an embargo-safe cutoff, calibrate, and report reliability.

[... prior revision history unchanged — see repository history for the
multi-seed stability check, frozen-data, regularization rounds 1-5,
nested threshold selection/gate confirmation, bootstrap CI, capacity
round 4 + logistic baseline, and bagged-fits revisions ...]

CHANGES IN THIS REVISION — GATE HARDENING (diagnostic-driven)
───────────────────────────────────────────────────────────────
A full pipeline run CONFIRMED the nested gate (5/6 seeds passed, pooled
PASS) and then FAILED the true final holdout hard: avg/trade=-1.03%
over 36 trades, 90% bootstrap CI entirely negative
[-1.69%, -0.38%]. Root-cause analysis found:

  (a) The gate's own numbers were already marginal when it passed —
      pooled mean test avg/trade (+0.15%) was dwarfed by its own std
      (0.50%, a ~0.3 signal/noise ratio), and PBO (48.6%) was 1.4
      points from the >50% "fail" cutoff. evaluate_gate()'s three
      checks (directional consistency, downside floor, PBO) are
      tripwires for CLEARLY broken strategies, not for "statistically
      indistinguishable from a coin flip" ones — nothing in the gate
      penalized a low signal/noise ratio directly.

  (b) CPCV's C(8,2)=28 combinations dilute the most-recent group (7)
      across 7 different test-set combinations averaged together with
      27 others — nothing in the gate specifically asks "how does this
      generalize to the newest, most-recently-seen regime", which is
      exactly the question the final holdout (chronologically last 15%)
      answers, and exactly where the model failed hardest. Path 27 in
      the confirmation seed=7 run (test_groups=(6,7), the two most
      recent CPCV groups) already showed this: test_avg=-2.274% on 9
      trades — a visible red flag that got averaged away in the pooled
      144-path statistic.

  (c) run_logistic_baseline() (near-minimal capacity) found essentially
      the SAME weak signal as the tuned GBT (mean +0.243% vs the GBT's
      +0.1545%, both well within one std of each other) — per that
      function's own stated diagnostic logic, a baseline that ALSO
      finds the identical marginal edge is evidence for a label/regime
      explanation, not an addressable GBT-capacity problem. The
      pipeline computed this number but never actually acted on it.

Four changes address this directly, all implemented below:

  1. TIGHTER GATE STATISTICS (evaluate_gate()): PBO_MAX_ALLOWED lowered
     0.50 -> 0.40, and a new MIN_TEST_SHARPE floor (0.15) on
     mean_test_sharpe — a strategy whose signal isn't at least ~0.15
     Sharpe-equivalent above its own noise no longer passes just
     because it's nominally "positive".

  2. RECENT-REGIME CHECK (confirm_gate_nested()): for every confirmation
     seed, the CPCV path whose test_groups are the LAST N_TEST_GROUPS
     groups (the most-recently-seen regime — the closest CPCV analog to
     the real final holdout) is pulled out of that seed's already-fitted
     paths (no extra fitting cost — it's already in fitted_paths) and
     scored on its own. Confirmation now additionally requires
     RECENT_REGIME_MIN_PASS_FRAC of seeds' recent-regime path to clear
     RECENT_REGIME_AVG_PNL_FLOOR, separate from (and stricter than) the
     pooled/whole-CPCV floor — so a model that only works on older
     regimes can no longer sail through on pooled/diluted statistics.

  3. BASELINE-EDGE GATE (main()): the logistic-regression capacity-floor
     check is no longer purely informational. main() now computes
     edge = GBT pooled mean_test_avg_pnl - logistic pooled
     mean_test_avg_pnl and requires edge > MIN_EDGE_OVER_BASELINE
     before proceeding to final training. If the tuned GBT can't beat a
     near-linear model by a non-trivial margin, that's the (c) signal
     above made an actual gate condition instead of a printed aside.

  4. WIDER / CONFIGURABLE HOLDOUT (run_final_training(), CLI): holdout
     sizing (holdout_frac) and triple-barrier resolution speed
     (tp_mult/sl_mult/max_holding) are now CLI-overridable
     (--holdout-frac, --tp-mult, --sl-mult, --max-holding) instead of
     hardcoded, so the final holdout's trade count (36, at the edge of
     what a bootstrap CI can say anything precise about) can be grown
     without editing the file — narrower barriers / shorter max holding
     resolve trades faster, and a larger holdout_frac widens the
     evaluation window itself.

All four are additive to the existing nested selection/confirmation
machinery — selection/confirmation/pooled-gate logic, bagging, purged
early stopping, etc. are unchanged; these are new necessary conditions
layered on top.

Run
────
    python main_gbt.py                                              # gated pipeline, live data
    python main_gbt.py --frozen-data                                 # reproducible comparisons
    python main_gbt.py --frozen-data --selection-seeds 0 1 \\
                        --confirmation-seeds 2 3 4 5 6 7             # explicit nested seed pools
    python main_gbt.py --frozen-data --no-multi-timeframe            # 4h-only baseline
    python main_gbt.py --frozen-data --n-bagged-fits 5                # override bag size
    python main_gbt.py --force-final-training                        # override an unstable gate
    python main_gbt.py --holdout-frac 0.20 --max-holding 16 \\
                        --tp-mult 1.3 --sl-mult 0.7                   # more holdout trades, tighter CI
"""

import os
import json
import numpy as np
import pandas as pd

from agents.state_aggregator import StateAggregator
from data.data_manager import update_master_data, update_all_timeframes
from multi_timeframe_state import build_multi_timeframe_context
from walkforward import compute_required_lookback_ticks
from cpcv import generate_cpcv_paths, compute_pbo
from triple_barrier import build_meta_labels
from gbt_agent import GBTAgent
from calibration_report import evaluate_calibration, plot_reliability_diagram

# ── Config — mirrors main_mcknn.py exactly where it overlaps ────────────────
FEATURES   = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
              "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES      = (1, 6, 42, 90)
WARMUP_IDX = 128
ACTION_DIM = 4

# Kept True (default) deliberately: a --no-multi-timeframe A/B run
# showed dropping context makes out-of-sample results WORSE (mean
# test avg/trade +0.196% -> +0.046%, positive paths 31/54 -> 26/54)
# while barely moving train avg/trade at all — see GBT_HYPERPARAMS'
# round-4 comment above. The context is contributing real signal;
# --no-multi-timeframe remains available as a CLI override for further
# A/B comparisons, but it is no longer the recommended default.
ENABLE_MULTI_TIMEFRAME = True
CONTEXT_TIMEFRAMES = ("15m", "1h")
CONTEXT_PACES = (1, 4, 16)

# Triple-barrier config. Narrower than earlier revisions for more
# independent trades per fold/holdout window (see module docstring).
# CLI-overridable this revision (--tp-mult/--sl-mult/--max-holding) —
# see GATE HARDENING change 4: the deployed final holdout only produced
# 36 trades, at the edge of what a bootstrap CI can say anything
# precise about, so these are now easy to tighten further from the
# command line without editing the file.
TP_MULT      = 1.5         # was 2.0
SL_MULT      = 0.8         # was 1.0
MAX_HOLDING  = 20          # was 32 (note: unified_executor.py's live
                            # MAX_HOLD_TICKS is a separate, independent
                            # constant — this only affects label/training
                            # config, not live execution behaviour)
COMMISSION   = 0.00015     # matches unified_executor.py's COMMISSION

# CPCV config
N_GROUPS        = 8
N_TEST_GROUPS   = 2
MAX_PATHS       = 28
MIN_TRAIN_TICKS = 2000

# Base-model regularization (overfitting fix, round 4/5 — unchanged
# this revision; see module docstring history for the full rationale
# behind each tightening round).
GBT_HYPERPARAMS = dict(
    max_iter=50,                # was 150 -> 80 -> 50
    learning_rate=0.015,        # was 0.04 -> 0.02 -> 0.015
    max_depth=2,                 # was 3 -> 2 (unchanged since round 4)
    max_leaf_nodes=3,            # was 8 -> 4 -> 3
    min_samples_leaf=1600,       # was 650 -> 1000 -> 1600
    l2_regularization=28.0,      # was 14.0 -> 20.0 -> 28.0
    validation_fraction=0.15,    # fallback only — see gbt_agent.py note
    n_iter_no_change=15,         # fallback only — see gbt_agent.py note
    max_features=0.25,           # was 0.45 -> 0.35 -> 0.25
)

# ── Entry-conviction threshold sweep (CPCV-only — see module docstring) ─────
ENTRY_THRESHOLD_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70)
MIN_SWEEP_TRADES = 150
MIN_PATH_TEST_TRADES = 15

# ── Multi-seed CPCV stability check (legacy path — see run_stability_check) ──
DEFAULT_STABILITY_SEEDS = (0, 1, 2)
DEFAULT_MIN_PASS_FRAC   = 0.6   # >= 60% of seeds must independently pass

# ── Nested threshold selection / gate confirmation ───────────────────────────
DEFAULT_SELECTION_SEEDS    = (0, 1)
DEFAULT_CONFIRMATION_SEEDS = (2, 3, 4, 5, 6, 7)

# ── Bagged fits ────────────────────────────────────────────────────────────
DEFAULT_N_BAGGED_FITS = 5

# ── Final holdout deployment gate ────────────────────────────────────────────
HOLDOUT_AVG_TRADE_FLOOR = 0.0    # require a non-negative mean holdout trade
HOLDOUT_MIN_TRADES      = 10     # fewer trades than this -> gate can't be trusted either way
DEFAULT_HOLDOUT_FRAC    = 0.15   # CLI-overridable via --holdout-frac (GATE HARDENING change 4)

# Stability gate (mirrors main_mcknn.py's philosophy — directional
# consistency + downside floor — plus a PBO check CPCV newly enables).
#
# NORMALIZED, not raw-sum: VAL_AVG_TRADE_FLOOR is a floor on the WORST
# path's MEAN PER-TRADE test return, so a fold's severity is judged
# per-bet rather than being amplified/muted by however many trades that
# particular fold happened to generate.
VAL_AVG_TRADE_FLOOR = -0.03

# ── GATE HARDENING (this revision) ───────────────────────────────────────────
# See module docstring's "CHANGES IN THIS REVISION — GATE HARDENING"
# section for the full diagnosis these four constants/checks respond to.

# (1) Tighter gate statistics.
# PBO cutoff lowered from 0.50 -> 0.40: the run that triggered this
# revision passed at PBO=48.6%, only 1.4 points from the OLD >50% fail
# line — too close to trust. A Sharpe-like floor is new: a strategy
# whose mean_test_sharpe (computed per-path already, previously never
# gated on) doesn't clear this bar isn't distinguishable from noise
# even when its mean per-trade return happens to be nominally positive.
PBO_MAX_ALLOWED = 0.40          # was implicitly 0.50
MIN_TEST_SHARPE = 0.15          # new — NaN (too few paths) does not fail this check

# (2) Recent-regime check. The CPCV test-group combination equal to the
# LAST N_TEST_GROUPS groups (e.g. (6, 7) when N_GROUPS=8,
# N_TEST_GROUPS=2) is the closest CPCV analog to the real final
# holdout (both are "the most recently observed data"). Pulled out of
# each confirmation seed's already-fitted CPCV paths at zero extra
# fitting cost.
RECENT_REGIME_GROUPS         = tuple(range(N_GROUPS - N_TEST_GROUPS, N_GROUPS))
RECENT_REGIME_AVG_PNL_FLOOR  = -0.005   # stricter than VAL_AVG_TRADE_FLOOR (-0.03) —
                                         # this is specifically a recency-risk check
RECENT_REGIME_MIN_TEST_TRADES = 8       # below this, that seed's recent-regime
                                         # result is excluded (too few trades to trust)
RECENT_REGIME_MIN_PASS_FRAC  = DEFAULT_MIN_PASS_FRAC   # reuse the same 0.6 bar

# (3) Baseline-edge gate. The GBT's pooled confirmation mean test
# avg/trade must beat the logistic-regression capacity-floor baseline's
# pooled mean by at least this margin. The run that triggered this
# revision had the GBT at +0.1545% vs the baseline's +0.243% — i.e. the
# GBT did not even beat the near-linear baseline, previously reported
# only as an informational aside.
MIN_EDGE_OVER_BASELINE = 0.0005  # 0.05 percentage points of avg/trade

OUT_DIR = "outcomes/gbt"
os.makedirs(OUT_DIR, exist_ok=True)


def _json_default(obj):
    """
    Fallback encoder for json.dump(default=_json_default) calls in this
    module. numpy scalar types (np.bool_, np.int64, np.float32/64, ...)
    are not JSON-serializable even though some print/repr as if they
    were native Python types. Converting the ROOT CAUSE (evaluate_gate()'s
    numpy comparisons) to native bool/float at the source is the real
    fix; this is a defense-in-depth net so any other numpy scalar that
    slips into a dict here doesn't crash the run instead of just writing
    a slightly-off value.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ─────────────────────────────────────────────
# State + label construction
# ─────────────────────────────────────────────

def _load_master_and_indicators(frozen: bool = False):
    """
    Shared first step for every state/label-construction path below:
    load+merge the 4h master CSV and compute the raw indicator array.
    Factored out so `load_prices_for_labeling()` (label-only /
    stationarity analysis — see regime_stationarity.py) doesn't pay for
    anything beyond this, while `build_states()` (which needs the full
    state vectors) shares this exact loading logic so the two paths can
    never silently diverge on which rows/dtypes they see.
    """
    df = update_master_data("4h", frozen=frozen)
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    ind = df[FEATURES].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)
    return df, ind, prices


def load_prices_for_labeling(frozen: bool = False):
    """
    Lightweight path for label-only / stationarity analysis (see
    regime_stationarity.py and label_sweep.py's phase-1 screen).
    triple_barrier.build_meta_labels() only ever needs Close prices +
    ATR_Scaled — NOT the full state vector — so this skips
    StateAggregator entirely and, critically, skips
    update_all_timeframes()'s 15m/1h fetch, which is by far the most
    expensive part of build_states(). This lets a tp_mult/sl_mult/
    max_holding sweep or a rolling-window edge/stationarity check
    re-derive labels for many candidate configs without ever touching
    the state-construction machinery.

    Returns
    -------
    prices_aligned, atr_aligned, aligned_df — identical alignment/dtype
    to what build_states_and_labels() hands to
    triple_barrier.build_meta_labels(), i.e. row i here corresponds to
    the SAME tick as state row i from build_states().
    """
    df, ind, prices = _load_master_and_indicators(frozen=frozen)
    prices_aligned = prices[WARMUP_IDX + 1:]
    atr_idx = FEATURES.index("ATR_Scaled")
    atr_aligned = ind[WARMUP_IDX + 1:, atr_idx]
    open_times = df["Open_time"].values[WARMUP_IDX + 1:]
    aligned_df = pd.DataFrame({"Open_time": open_times})
    return prices_aligned, atr_aligned, aligned_df


def build_states(frozen: bool = False,
                  enable_multi_timeframe: bool = None,
                  context_paces: tuple = None):
    """
    Build ONLY the state matrix (+ its aligned prices/ATR/Open_time) —
    NO triple-barrier labeling. States depend solely on
    FEATURES/PACES/CONTEXT_* and the raw market data, never on
    tp_mult/sl_mult/max_holding, so factoring this out lets
    label_sweep.py build the (expensive — 15m/1h fetch + per-tick
    aggregator loop) state matrix ONCE and reuse it across every
    barrier-config candidate it screens, instead of rebuilding it from
    scratch per config the way looping over the old monolithic
    build_states_and_labels() would.

    Returns
    -------
    states, prices_aligned, atr_aligned, aligned_df
    """
    enable_multi_timeframe = (ENABLE_MULTI_TIMEFRAME if enable_multi_timeframe is None
                               else enable_multi_timeframe)
    context_paces = CONTEXT_PACES if context_paces is None else context_paces

    df, ind, prices = _load_master_and_indicators(frozen=frozen)
    n = len(df)

    agg = StateAggregator(PACES, num_indicators=len(FEATURES))
    agg.warm_up_all(ind, WARMUP_IDX)

    extra_context_arr = None
    if enable_multi_timeframe:
        try:
            timeframe_dfs = update_all_timeframes(CONTEXT_TIMEFRAMES, frozen=frozen)
            missing = [tf for tf in CONTEXT_TIMEFRAMES if tf not in timeframe_dfs]
            if missing:
                raise FileNotFoundError(f"missing timeframe(s): {missing}")
            timeframe_frames = {
                tf: tdf[["Open_time"] + FEATURES].dropna().reset_index(drop=True)
                for tf, tdf in timeframe_dfs.items()
            }
            extra_context_arr = build_multi_timeframe_context(
                df, timeframe_frames, FEATURES, context_paces=context_paces,
            )
            print(f"  Multi-timeframe context: {extra_context_arr.shape}")
        except FileNotFoundError as exc:
            print(f"  ⚠ Multi-timeframe context disabled — {exc}")
            extra_context_arr = None
    else:
        print("  Multi-timeframe context disabled (--no-multi-timeframe).")

    states = []
    agg.tick = 0
    for i in range(WARMUP_IDX + 1, n):
        agg.update(ind[i])
        market_vec = agg.get_state(portfolio_info=None)[:-2]
        if extra_context_arr is not None:
            state_vec = np.concatenate([market_vec, extra_context_arr[i]])
        else:
            state_vec = market_vec
        states.append(state_vec)
    states = np.array(states, dtype=np.float32)

    prices_aligned = prices[WARMUP_IDX + 1:]
    atr_idx = FEATURES.index("ATR_Scaled")
    atr_aligned = ind[WARMUP_IDX + 1:, atr_idx]
    open_times = df["Open_time"].values[WARMUP_IDX + 1:]
    aligned_df = pd.DataFrame({"Open_time": open_times})   # used only for
                                                            # its length by
                                                            # cpcv.py's group
                                                            # splitting

    return states, prices_aligned, atr_aligned, aligned_df


def labels_from_prices(prices_aligned: np.ndarray, atr_aligned: np.ndarray,
                       tp_mult: float = None, sl_mult: float = None,
                       max_holding: int = None) -> dict:
    """
    Thin wrapper around triple_barrier.build_meta_labels() that applies
    this module's TP_MULT/SL_MULT/MAX_HOLDING defaults exactly like the
    old monolithic build_states_and_labels() used to inline. Factored
    out so label_sweep.py can re-label the SAME prices_aligned/
    atr_aligned under many different barrier configs without ever
    re-deriving them from build_states()/_load_master_and_indicators().
    """
    tp_mult     = TP_MULT if tp_mult is None else tp_mult
    sl_mult     = SL_MULT if sl_mult is None else sl_mult
    max_holding = MAX_HOLDING if max_holding is None else max_holding
    return build_meta_labels(
        prices_aligned, atr_aligned,
        tp_mult=tp_mult, sl_mult=sl_mult,
        max_holding=max_holding, commission=COMMISSION,
    )


def build_states_and_labels(frozen: bool = False,
                             enable_multi_timeframe: bool = None,
                             context_paces: tuple = None,
                             tp_mult: float = None,
                             sl_mult: float = None,
                             max_holding: int = None):
    """
    Build (states, labels, prices, aligned_df) for the full dataset,
    using identical feature construction to main_mcknn.py/pre_training.py
    so results are directly comparable.

    UNCHANGED call shape/behaviour from prior revisions — every
    existing caller (main(), diagnostic scripts) keeps working with
    zero changes. Internally this is now just a thin composition of
    build_states() + labels_from_prices() (see above), so a caller that
    wants to sweep tp_mult/sl_mult/max_holding without paying for
    state-matrix reconstruction each time should call those two
    functions directly instead — see label_sweep.py.

    Parameters
    ----------
    frozen : forwarded to data_manager.update_master_data() /
             update_all_timeframes() — if True, uses the existing master
             CSV(s) verbatim instead of refetching/appending, so repeated
             calls all see the exact same dataset.
    enable_multi_timeframe, context_paces : override the module-level
             ENABLE_MULTI_TIMEFRAME / CONTEXT_PACES constants for this
             call only (used by --no-multi-timeframe).
    tp_mult, sl_mult, max_holding : override the module-level TP_MULT /
             SL_MULT / MAX_HOLDING triple-barrier constants for this call
             only (GATE HARDENING change 4, see
             --tp-mult/--sl-mult/--max-holding in main()). Narrower
             barriers / shorter max holding resolve trades faster,
             growing the final holdout's trade count without editing
             this file.
    """
    states, prices_aligned, atr_aligned, aligned_df = build_states(
        frozen=frozen, enable_multi_timeframe=enable_multi_timeframe,
        context_paces=context_paces,
    )
    labels = labels_from_prices(prices_aligned, atr_aligned,
                                tp_mult=tp_mult, sl_mult=sl_mult,
                                max_holding=max_holding)
    return states, labels, prices_aligned, aligned_df


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def simulate_pnl(states: np.ndarray, labels: dict, mask: np.ndarray,
                  agent: GBTAgent, prob_threshold: float = 0.5) -> tuple:
    """
    Score the CLASSIFIER's entry decisions against the triple-barrier
    label's own ground-truth realised return, enforcing ONE OPEN
    POSITION AT A TIME (matching unified_executor.py's single-slot
    `inventory` deque).

    Returns
    -------
    (total_pnl, n_trades, avg_pnl, trade_returns)
    """
    idx = np.flatnonzero(mask)
    total_pnl  = 0.0
    n_trades   = 0
    trade_returns: list = []
    next_free_tick = -1   # no trade open yet
    for row in idx:
        if row < next_free_tick:
            continue   # a previously opened trade is still "in the market"
        lsh = agent.actor._raw_probs(states[row])
        best = int(np.argmax(lsh))
        if best == 0 and lsh[0] > prob_threshold:
            r = float(labels["long_return"][row])
            total_pnl += r
            trade_returns.append(r)
            next_free_tick = int(labels["long_touch"][row]) + 1
            n_trades += 1
        elif best == 1 and lsh[1] > prob_threshold:
            r = float(labels["short_return"][row])
            total_pnl += r
            trade_returns.append(r)
            next_free_tick = int(labels["short_touch"][row]) + 1
            n_trades += 1
        # else: HOLD — stays flat, next_free_tick unchanged.

    avg_pnl = float(np.mean(trade_returns)) if trade_returns else 0.0
    return total_pnl, n_trades, avg_pnl, trade_returns


def _sharpe_like(trade_returns: list) -> float:
    """Same formula diagnostic_gbt.py's write_summary() uses for its
    real tick-by-tick backtest — kept identical so the two pipelines'
    notion of "risk-adjusted edge" is directly comparable. Returns NaN
    when there are too few trades or zero variance."""
    if len(trade_returns) < 5:
        return float("nan")
    arr = np.array(trade_returns, dtype=np.float64)
    std = arr.std()
    if std == 0:
        return float("nan")
    return float(arr.mean() / std * np.sqrt(len(arr)))


def bootstrap_ci(trade_returns: list, n_boot: int = 10_000, ci: float = 0.90,
                 random_state: int = 0) -> dict:
    """Percentile bootstrap CI on the mean per-trade return."""
    n = len(trade_returns)
    if n < 2:
        return {"mean": float(np.mean(trade_returns)) if n else float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "ci_level": ci,
                "n_trades": n, "note": "too few trades for a bootstrap interval"}

    arr = np.asarray(trade_returns, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    boot_means = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 - (1 - ci) / 2) * 100
    ci_lo, ci_hi = np.percentile(boot_means, [lo_pct, hi_pct])

    result = {"mean": float(arr.mean()), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
              "ci_level": ci, "n_trades": n}
    if n < 30:
        result["note"] = (f"n_trades={n} is small — bootstrap CI is itself "
                          f"unstable; treat as directional, not precise")
    return result


def run_cpcv(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
             random_state: int = 0, model_type: str = "hgb",
             hyperparams: dict = None, n_bagged_fits: int = None) -> tuple:
    """
    Returns (path_results, fitted_paths, valid).

    fitted_paths is kept (and, this revision, is the mechanism the new
    recent-regime check in confirm_gate_nested() uses — see GATE
    HARDENING change 2) so callers can re-score the SAME fitted models
    without refitting.
    """
    hp = GBT_HYPERPARAMS if hyperparams is None else hyperparams
    nb = DEFAULT_N_BAGGED_FITS if n_bagged_fits is None else n_bagged_fits

    lookback_ticks = compute_required_lookback_ticks()
    paths = generate_cpcv_paths(
        aligned_df, n_groups=N_GROUPS, n_test_groups=N_TEST_GROUPS,
        lookback_ticks=lookback_ticks, min_train_ticks=MIN_TRAIN_TICKS,
        max_paths=MAX_PATHS,
    )
    print(f"  Generated {len(paths)} CPCV paths "
          f"(groups={N_GROUPS}, test_groups={N_TEST_GROUPS}, "
          f"lookback_ticks={lookback_ticks}, random_state={random_state}, "
          f"model_type={model_type}, "
          f"n_bagged_fits={1 if model_type == 'logistic' else nb})")

    y = labels["best_action"]
    valid = labels["valid_mask"]

    path_results = []
    fitted_paths = []
    for path in paths:
        train_mask = path.train_mask & valid
        test_mask  = path.test_mask & valid
        if train_mask.sum() < MIN_TRAIN_TICKS or test_mask.sum() < 100:
            continue

        agent = GBTAgent(state_dim=states.shape[1], action_dim=ACTION_DIM,
                         model_type=model_type)
        agent.fit(states[train_mask], y[train_mask],
                  purge_ticks=lookback_ticks, random_state=random_state,
                  n_bagged_fits=nb, **hp)
        fitted_paths.append((path, agent))

        train_pnl, n_train_trades, train_avg_pnl, train_returns = simulate_pnl(
            states, labels, train_mask, agent)
        test_pnl, n_test_trades, test_avg_pnl, test_returns = simulate_pnl(
            states, labels, test_mask, agent)
        test_sharpe = _sharpe_like(test_returns)

        path_results.append({
            "path_id": path.path_id, "test_groups": list(path.test_groups),
            "train_pnl": train_pnl, "test_pnl": test_pnl,
            "train_avg_pnl": train_avg_pnl, "test_avg_pnl": test_avg_pnl,
            "test_sharpe": test_sharpe,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "n_train_trades": n_train_trades, "n_test_trades": n_test_trades,
        })
        sharpe_str = f"{test_sharpe:+.2f}" if not np.isnan(test_sharpe) else "n/a"
        print(f"  path {path.path_id:>3}  test_groups={path.test_groups}  "
              f"train_avg={train_avg_pnl:+.3%} ({n_train_trades} trades)  "
              f"test_avg={test_avg_pnl:+.3%} ({n_test_trades} trades)  "
              f"test_sharpe={sharpe_str}  "
              f"[raw: train={train_pnl:+.2%} test={test_pnl:+.2%}]  "
              f"[@0.50 baseline, seed={random_state}, model={model_type}]")

    return path_results, fitted_paths, valid


def sweep_entry_thresholds(states: np.ndarray, labels: dict, fitted_paths: list,
                           valid: np.ndarray, verbose: bool = True) -> dict:
    """Re-score every already-fitted CPCV (path, agent) at each candidate
    entry-probability threshold and pick the threshold that scores best
    OUT OF SAMPLE across CPCV test folds only."""
    if verbose:
        print(f"\n  Sweeping entry_threshold over {ENTRY_THRESHOLD_CANDIDATES} "
              f"across {len(fitted_paths)} fitted CPCV paths (test folds only)...")
    candidates = {}
    for t in ENTRY_THRESHOLD_CANDIDATES:
        avg_pnls = []
        trade_counts = []
        for path, agent in fitted_paths:
            test_mask = path.test_mask & valid
            _, n_test_trades, test_avg_pnl, _ = simulate_pnl(
                states, labels, test_mask, agent, prob_threshold=t)
            avg_pnls.append(test_avg_pnl)
            trade_counts.append(n_test_trades)

        avg_pnls = np.array(avg_pnls, dtype=np.float64)
        total_trades = int(sum(trade_counts))
        mean_avg = float(avg_pnls.mean()) if len(avg_pnls) else float("nan")
        std_avg  = float(avg_pnls.std()) if len(avg_pnls) else float("nan")
        eligible = total_trades >= MIN_SWEEP_TRADES
        score = (mean_avg / (std_avg + 1e-6)) if eligible else float("-inf")

        candidates[t] = {
            "mean_test_avg_pnl": mean_avg, "std_test_avg_pnl": std_avg,
            "total_test_trades": total_trades, "eligible": eligible,
            "score": score,
        }
        if verbose:
            elig_str = "" if eligible else "  [DISQUALIFIED: too few trades]"
            print(f"    threshold={t:.2f}  mean_test_avg={mean_avg:+.4%}  "
                  f"std={std_avg:.4%}  total_test_trades={total_trades}  "
                  f"score={score:+.3f}{elig_str}")

    best_t = max(candidates, key=lambda k: candidates[k]["score"])
    if candidates[best_t]["score"] == float("-inf"):
        if verbose:
            print(f"  ⚠ No threshold cleared MIN_SWEEP_TRADES={MIN_SWEEP_TRADES} — "
                  f"falling back to threshold=0.50.")
        best_t = 0.50
    if verbose:
        print(f"  ✓ Selected entry_threshold={best_t:.2f}  "
              f"(score={candidates[best_t]['score']:+.3f}, "
              f"total_test_trades={candidates[best_t]['total_test_trades']})")

    return {"chosen_threshold": best_t, "candidates": candidates}


def evaluate_paths_at_threshold(states: np.ndarray, labels: dict, fitted_paths: list,
                                valid: np.ndarray, threshold: float) -> list:
    """Re-score every fitted CPCV path at `threshold` (no refitting)."""
    results = []
    for path, agent in fitted_paths:
        train_mask = path.train_mask & valid
        test_mask  = path.test_mask & valid
        train_pnl, n_train_trades, train_avg_pnl, train_returns = simulate_pnl(
            states, labels, train_mask, agent, prob_threshold=threshold)
        test_pnl, n_test_trades, test_avg_pnl, test_returns = simulate_pnl(
            states, labels, test_mask, agent, prob_threshold=threshold)
        test_sharpe = _sharpe_like(test_returns)
        results.append({
            "path_id": path.path_id, "test_groups": list(path.test_groups),
            "train_pnl": train_pnl, "test_pnl": test_pnl,
            "train_avg_pnl": train_avg_pnl, "test_avg_pnl": test_avg_pnl,
            "test_sharpe": test_sharpe,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "n_train_trades": n_train_trades, "n_test_trades": n_test_trades,
        })
    return results


def _extract_recent_regime_result(states: np.ndarray, labels: dict,
                                  fitted_paths: list, valid: np.ndarray,
                                  threshold: float) -> dict:
    """
    GATE HARDENING change 2 (this revision).

    Pull the CPCV path whose test_groups equal RECENT_REGIME_GROUPS (the
    LAST N_TEST_GROUPS groups — the most-recently-observed regime, the
    closest CPCV analog to the real final holdout) out of an already-
    fitted seed's `fitted_paths`, and score it on its own at `threshold`.
    No extra model fitting — this path was already fit inside run_cpcv().

    Returns None if that exact test-group combination wasn't generated
    for this seed/dataset size (e.g. a very short dataset) — callers
    treat that as "no recent-regime signal available for this seed"
    rather than crashing.
    """
    for path, agent in fitted_paths:
        if tuple(path.test_groups) == RECENT_REGIME_GROUPS:
            test_mask = path.test_mask & valid
            _, n_test_trades, test_avg_pnl, test_returns = simulate_pnl(
                states, labels, test_mask, agent, prob_threshold=threshold)
            return {
                "test_groups": list(RECENT_REGIME_GROUPS),
                "n_test_trades": n_test_trades,
                "test_avg_pnl": test_avg_pnl,
                "test_sharpe": _sharpe_like(test_returns),
            }
    return None


def evaluate_gate(path_results: list) -> tuple:
    """
    Pure (no I/O) gate evaluation.

    GATE HARDENING (this revision): two of the three original checks
    (directional consistency, downside floor) are unchanged; the PBO
    check is now tighter (PBO_MAX_ALLOWED=0.40, was 0.50) and a NEW
    Sharpe-floor check (MIN_TEST_SHARPE=0.15) is added — see module
    docstring. A path set that is nominally "positive on average" but
    whose mean test_sharpe is below the floor (a low signal/noise
    ratio, indistinguishable from chance) now fails the gate even if
    it would have passed the old three-check version.

    Applies the reliability filter (MIN_PATH_TEST_TRADES) first, then
    checks directional consistency + downside floor + PBO + Sharpe.

    Returns
    -------
    (passed: bool, stats: dict)
    """
    reliable = [r for r in path_results if r["n_test_trades"] >= MIN_PATH_TEST_TRADES]
    excluded = [r for r in path_results if r["n_test_trades"] < MIN_PATH_TEST_TRADES]

    if not reliable:
        # Pre-existing bug fixed incidentally this revision: this branch
        # previously omitted "logit_lambda", which run_logistic_baseline()/
        # run_stability_check() then KeyError on when NO seed produces a
        # reliable path (e.g. an extremely conservative model). Included
        # here for the same reason mean_test_sharpe already was.
        return False, {
            "n_total": 0, "n_pos": 0, "pbo": float("nan"),
            "logit_lambda": float("nan"),
            "mean_test_avg_pnl": float("nan"), "std_test_avg_pnl": float("nan"),
            "min_test_avg_pnl": float("nan"), "mean_test_sharpe": float("nan"),
            "reliable_paths": [], "excluded_paths": excluded,
            "reason": "no CPCV path had >= MIN_PATH_TEST_TRADES test trades",
        }

    test_pnls     = np.array([r["test_pnl"] for r in reliable])
    test_avg_pnls = np.array([r["test_avg_pnl"] for r in reliable])
    sharpes       = np.array([r["test_sharpe"] for r in reliable])
    n_pos   = int((test_avg_pnls > 0).sum())
    n_total = len(reliable)

    pbo_input = [{"train_pnl": r["train_avg_pnl"], "test_pnl": r["test_avg_pnl"]}
                 for r in reliable]
    pbo_info = compute_pbo(pbo_input)

    mean_test_sharpe = float(np.nanmean(sharpes)) if len(sharpes) else float("nan")

    directionally_consistent = bool(n_pos >= (n_total / 2))
    downside_breach = bool(test_avg_pnls.min() < VAL_AVG_TRADE_FLOOR)
    pbo_high = bool((not np.isnan(pbo_info["pbo"])) and pbo_info["pbo"] > PBO_MAX_ALLOWED)
    # New this revision: a Sharpe that's genuinely unmeasurable (NaN —
    # e.g. too few paths had >=5 test trades to compute one) does NOT
    # fail this check on its own; it's a "can't tell" case, handled by
    # the other three checks instead. A COMPUTED Sharpe below the floor
    # DOES fail it.
    sharpe_low = bool((not np.isnan(mean_test_sharpe)) and mean_test_sharpe < MIN_TEST_SHARPE)
    passed = (directionally_consistent and (not downside_breach)
              and (not pbo_high) and (not sharpe_low))

    stats = {
        "n_total": n_total, "n_pos": n_pos,
        "mean_test_avg_pnl": float(test_avg_pnls.mean()),
        "std_test_avg_pnl": float(test_avg_pnls.std()),
        "min_test_avg_pnl": float(test_avg_pnls.min()),
        "mean_test_pnl_raw": float(test_pnls.mean()),
        "std_test_pnl_raw": float(test_pnls.std()),
        "min_test_pnl_raw": float(test_pnls.min()),
        "mean_test_sharpe": mean_test_sharpe,
        "pbo": pbo_info["pbo"], "logit_lambda": pbo_info["logit_lambda"],
        "directionally_consistent": directionally_consistent,
        "downside_breach": downside_breach, "pbo_high": pbo_high,
        "sharpe_low": sharpe_low,
        "reliable_paths": reliable, "excluded_paths": excluded,
    }
    return passed, stats


def gate_and_summarize(path_results: list, entry_threshold: float = 0.5,
                       threshold_sweep: dict = None, seed: int = None,
                       write: bool = True) -> bool:
    """NORMALIZED gate summary/report — decision logic lives in
    evaluate_gate(); this prints/saves it. Reliability filter
    (MIN_PATH_TEST_TRADES) applied inside evaluate_gate()."""
    if not path_results:
        print("  ⛔ No usable CPCV paths — aborting.")
        return False

    passed, stats = evaluate_gate(path_results)
    reliable = stats["reliable_paths"]
    excluded = stats["excluded_paths"]
    n_total, n_pos = stats["n_total"], stats["n_pos"]

    if n_total == 0:
        print(f"  ⛔ No CPCV path had >= {MIN_PATH_TEST_TRADES} test trades — "
              f"cannot compute a trustworthy gate verdict.")
        return False

    seed_str = f"  (seed={seed})" if seed is not None else ""
    print(f"\n{'='*62}\n  CPCV SUMMARY ({n_total} reliable paths)  "
          f"@ entry_threshold={entry_threshold:.2f}{seed_str}\n{'='*62}")
    if excluded:
        print(f"  Reliability filter  : {n_total} path(s) kept, {len(excluded)} "
              f"excluded (< {MIN_PATH_TEST_TRADES} test trades — too few to "
              f"trust their test_avg_pnl)")
    print(f"  Mean test avg/trade : {stats['mean_test_avg_pnl']:+.4%}")
    print(f"  Std  test avg/trade : {stats['std_test_avg_pnl']:.4%}")
    print(f"  Min  test avg/trade : {stats['min_test_avg_pnl']:+.4%}")
    if not np.isnan(stats["mean_test_sharpe"]):
        print(f"  Mean test Sharpe    : {stats['mean_test_sharpe']:+.3f}  "
              f"(floor: {MIN_TEST_SHARPE:+.2f})")
    print(f"  Positive paths      : {n_pos}/{n_total}")
    print(f"  PBO                 : {stats['pbo']:.1%}  "
          f"(fraction of in-sample-good paths that disappointed "
          f"out-of-sample — lower is better; > {PBO_MAX_ALLOWED:.0%} fails "
          f"this gate)")
    print(f"  logit_lambda        : {stats['logit_lambda']:+.3f}  "
          f"(more negative = more systematic overfitting)")
    print(f"  [context, raw sums] mean={stats['mean_test_pnl_raw']:+.2%}  "
          f"std={stats['std_test_pnl_raw']:.2%}  min={stats['min_test_pnl_raw']:+.2%}")

    if write:
        with open(os.path.join(OUT_DIR, "cpcv_summary.json"), "w") as f:
            json.dump({
                "n_paths": n_total,
                "min_path_test_trades": MIN_PATH_TEST_TRADES,
                "n_paths_excluded_low_trades": len(excluded),
                "excluded_path_ids": [r["path_id"] for r in excluded],
                "entry_threshold": entry_threshold,
                "seed": seed,
                "threshold_sweep": threshold_sweep,
                "mean_test_avg_pnl": stats["mean_test_avg_pnl"],
                "std_test_avg_pnl": stats["std_test_avg_pnl"],
                "min_test_avg_pnl": stats["min_test_avg_pnl"],
                "mean_test_sharpe": stats["mean_test_sharpe"],
                "mean_test_pnl_raw": stats["mean_test_pnl_raw"],
                "std_test_pnl_raw": stats["std_test_pnl_raw"],
                "min_test_pnl_raw": stats["min_test_pnl_raw"],
                "n_paths_positive": n_pos,
                "pbo": stats["pbo"], "logit_lambda": stats["logit_lambda"],
                "pbo_max_allowed": PBO_MAX_ALLOWED,
                "min_test_sharpe": MIN_TEST_SHARPE,
                "path_results": reliable,
            }, f, indent=2, default=_json_default)
        print(f"  ✓  CPCV summary saved → {OUT_DIR}/cpcv_summary.json")

    if not passed:
        print(f"\n  ⛔ UNSTABLE — refusing to train final deployable model.")
        if not stats["directionally_consistent"]:
            print(f"     ✗ only {n_pos}/{n_total} paths positive (per-trade avg)")
        if stats["downside_breach"]:
            print(f"     ✗ worst path avg/trade {stats['min_test_avg_pnl']:+.4%} "
                  f"< floor {VAL_AVG_TRADE_FLOOR:+.2%}")
        if stats["pbo_high"]:
            print(f"     ✗ PBO={stats['pbo']:.1%} > {PBO_MAX_ALLOWED:.0%} — in-sample "
                  f"performance is not predictive of out-of-sample performance")
        if stats.get("sharpe_low"):
            print(f"     ✗ mean test Sharpe {stats['mean_test_sharpe']:+.3f} "
                  f"< floor {MIN_TEST_SHARPE:+.2f} — signal is not "
                  f"distinguishable from noise")
        return False

    print(f"\n  ✓ CPCV gate passed.")
    return True


# ─────────────────────────────────────────────
# Logistic regression baseline (capacity floor check)
# ─────────────────────────────────────────────
#
# GATE HARDENING (this revision): this function's output is no longer
# purely informational — main() now uses its pooled mean_test_avg_pnl
# as the comparison point for the new baseline-edge gate (change 3).
# See run_logistic_baseline()'s printed verdict, which already flagged
# this exact situation ("Baseline ALSO ... near-minimal model
# capacity... GBT's instability [is a] label/regime explanation") but
# previously never blocked deployment on it.

def run_logistic_baseline(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                          seeds: tuple = None) -> dict:
    """
    Fit a plain logistic regression (via GBTAgent(model_type="logistic"))
    through the same CPCV path generation / purge-embargo / triple-
    barrier labels as the GBT pipeline, at the default entry_threshold
    (0.5). Pools results across `seeds` (defaults to
    DEFAULT_CONFIRMATION_SEEDS, so it's directly comparable to the
    GBT's confirmation-pool numbers) and reports the same PBO / mean
    test avg/trade / positive-path-fraction stats the GBT gate uses,
    plus per-seed pass/fail for direct comparison against the GBT's own
    per-seed confirmation result.

    Returns the pooled evaluate_gate() stats dict (with a "per_seed"
    key added) and also writes to outcomes/gbt/logistic_baseline.json.
    """
    seeds = DEFAULT_CONFIRMATION_SEEDS if seeds is None else seeds
    print(f"\n{'#'*62}\n  LOGISTIC REGRESSION BASELINE (capacity floor check)  "
          f"seeds={list(seeds)}\n{'#'*62}")

    per_seed = []
    pooled_results = []
    for seed in seeds:
        print(f"\n{'-'*62}\n  [baseline] seed {seed}\n{'-'*62}")
        path_results, _, _ = run_cpcv(
            states, labels, aligned_df, random_state=seed,
            model_type="logistic", hyperparams={},
        )
        pooled_results.extend(path_results)

        seed_passed, seed_stats = evaluate_gate(path_results)
        per_seed.append({
            "seed": seed, "passed": bool(seed_passed),
            "n_total": seed_stats["n_total"], "n_pos": seed_stats.get("n_pos", 0),
            "mean_test_avg_pnl": seed_stats["mean_test_avg_pnl"],
            "min_test_avg_pnl": seed_stats["min_test_avg_pnl"],
            "pbo": seed_stats["pbo"],
        })
        status = "PASS" if seed_passed else "FAIL"
        pbo_seed_str = f"{seed_stats['pbo']:.1%}" if not np.isnan(seed_stats["pbo"]) else "n/a"
        print(f"  [baseline] seed {seed}: {status}  "
              f"mean_test_avg={seed_stats['mean_test_avg_pnl']:+.4%}  "
              f"positive={seed_stats.get('n_pos', 0)}/{seed_stats['n_total']}  "
              f"PBO={pbo_seed_str}")

    passed, stats = evaluate_gate(pooled_results)
    n_pass = sum(r["passed"] for r in per_seed)
    frac_pass = n_pass / len(seeds) if seeds else float("nan")

    print(f"\n{'='*62}\n  LOGISTIC BASELINE VERDICT  ({stats['n_total']} reliable paths)"
          f"\n{'='*62}")
    print(f"  Per-seed stability  : {n_pass}/{len(seeds)} seeds passed "
          f"({frac_pass:.0%}) — compare directly against the GBT's own "
          f"per-seed confirmation result above.")
    print(f"  Mean test avg/trade : {stats['mean_test_avg_pnl']:+.4%}")
    print(f"  Std  test avg/trade : {stats['std_test_avg_pnl']:.4%}")
    print(f"  Min  test avg/trade : {stats['min_test_avg_pnl']:+.4%}")
    print(f"  Positive paths      : {stats.get('n_pos', 0)}/{stats['n_total']}")
    pbo_str = f"{stats['pbo']:.1%}" if not np.isnan(stats["pbo"]) else "n/a"
    print(f"  PBO                 : {pbo_str}")

    if not np.isnan(stats["pbo"]) and stats["pbo"] > PBO_MAX_ALLOWED:
        print(f"\n  → Baseline ALSO fails PBO<={PBO_MAX_ALLOWED:.0%} at near-minimal "
              f"model capacity. This supports a label/regime explanation for the "
              f"GBT's instability over a pure capacity/overfitting "
              f"explanation — further GBT regularization (or bagging) is "
              f"unlikely to be the fix by itself; consider regime-"
              f"conditional models, shorter re-fit windows, or a smaller/"
              f"more selective trading footprint.")
    elif np.isnan(stats["pbo"]):
        print(f"\n  → Not enough reliable logistic paths to compute a PBO "
              f"verdict — treat as inconclusive, not a pass.")
    else:
        print(f"\n  → Baseline PASSES PBO<={PBO_MAX_ALLOWED:.0%} at near-minimal "
              f"capacity. Whether this is good news for the GBT now ALSO "
              f"depends on whether the GBT beats this baseline by a real "
              f"margin — see the baseline-edge gate in main() (this "
              f"revision): a GBT that doesn't outperform this near-linear "
              f"model by at least {MIN_EDGE_OVER_BASELINE:.2%} avg/trade is "
              f"treated as having found the same marginal signal, not a "
              f"GBT-specific edge.")

    with open(os.path.join(OUT_DIR, "logistic_baseline.json"), "w") as f:
        json.dump({
            "seeds": list(seeds),
            "n_pass": n_pass, "frac_pass": frac_pass,
            "per_seed": per_seed,
            "mean_test_avg_pnl": stats["mean_test_avg_pnl"],
            "std_test_avg_pnl": stats["std_test_avg_pnl"],
            "min_test_avg_pnl": stats["min_test_avg_pnl"],
            "n_total": stats["n_total"], "n_pos": stats.get("n_pos", 0),
            "pbo": stats["pbo"], "logit_lambda": stats["logit_lambda"],
        }, f, indent=2, default=_json_default)
    print(f"\n  ✓  Logistic baseline saved → {OUT_DIR}/logistic_baseline.json")

    stats["per_seed"] = per_seed
    stats["n_pass"] = n_pass
    stats["frac_pass"] = frac_pass
    return stats


# ─────────────────────────────────────────────
# Multi-seed stability check (LEGACY — superseded by the nested
# selection/confirmation split below; kept only for backward
# compatibility with anything that still imports it directly)
# ─────────────────────────────────────────────

def run_stability_check(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                        seeds: tuple = DEFAULT_STABILITY_SEEDS,
                        min_pass_frac: float = DEFAULT_MIN_PASS_FRAC) -> tuple:
    """DEPRECATED as of the nested-selection revision — main() no longer
    calls this. Left in place only for backward compatibility with any
    external caller. See select_threshold_nested()/confirm_gate_nested()
    for the current (nested, non-circular) flow."""
    print(f"\n{'#'*62}\n  MULTI-SEED CPCV STABILITY CHECK  "
          f"(seeds={list(seeds)}, min_pass_frac={min_pass_frac:.0%})\n{'#'*62}")

    seed_results = []
    for seed in seeds:
        print(f"\n{'-'*62}\n  Seed {seed}\n{'-'*62}")
        path_results, fitted_paths, valid = run_cpcv(states, labels, aligned_df,
                                                       random_state=seed)
        sweep = sweep_entry_thresholds(states, labels, fitted_paths, valid,
                                       verbose=False)
        entry_threshold = sweep["chosen_threshold"]
        evaluated = evaluate_paths_at_threshold(states, labels, fitted_paths,
                                                valid, entry_threshold)
        passed, stats = evaluate_gate(evaluated)

        result = {
            "seed": seed, "passed": bool(passed), "entry_threshold": entry_threshold,
            "n_total": stats["n_total"], "n_pos": stats.get("n_pos", 0),
            "mean_test_avg_pnl": stats["mean_test_avg_pnl"],
            "min_test_avg_pnl": stats["min_test_avg_pnl"],
            "pbo": stats["pbo"],
        }
        seed_results.append(result)

        status = "PASS" if passed else "FAIL"
        pbo_str = f"{stats['pbo']:.1%}" if not np.isnan(stats["pbo"]) else "n/a"
        print(f"  Seed {seed}: {status}  entry_threshold={entry_threshold:.2f}  "
              f"mean_test_avg={stats['mean_test_avg_pnl']:+.4%}  "
              f"positive={stats.get('n_pos', 0)}/{stats['n_total']}  PBO={pbo_str}")

    n_pass = sum(r["passed"] for r in seed_results)
    frac_pass = n_pass / len(seeds)
    stable = frac_pass >= min_pass_frac

    print(f"\n{'='*62}\n  STABILITY VERDICT\n{'='*62}")
    print(f"  {n_pass}/{len(seeds)} seeds passed the CPCV gate "
          f"({frac_pass:.0%}, required >= {min_pass_frac:.0%})")
    print(f"  → {'STABLE' if stable else 'UNSTABLE'} across seeds.")

    with open(os.path.join(OUT_DIR, "stability_check.json"), "w") as f:
        json.dump({
            "seeds": list(seeds), "min_pass_frac": min_pass_frac,
            "n_pass": n_pass, "frac_pass": frac_pass, "stable": stable,
            "results": seed_results,
        }, f, indent=2, default=_json_default)
    print(f"  ✓  Stability check saved → {OUT_DIR}/stability_check.json")

    return stable, seed_results


def _pick_representative_seed(seed_results: list) -> int:
    """MEDIAN by mean_test_avg_pnl among passing seeds (or all seeds if
    none passed) — legacy, used only by run_stability_check()."""
    passed = [r for r in seed_results if r["passed"]]
    pool = passed if passed else seed_results
    pool_sorted = sorted(pool, key=lambda r: r["mean_test_avg_pnl"])
    median = pool_sorted[len(pool_sorted) // 2]
    return median["seed"]


# ─────────────────────────────────────────────
# Nested threshold selection + gate confirmation
# ─────────────────────────────────────────────

def select_threshold_nested(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                            selection_seeds: tuple = DEFAULT_SELECTION_SEEDS,
                            n_bagged_fits: int = None) -> dict:
    """Fit CPCV paths for every seed in `selection_seeds`, POOL all of
    their fitted (path, agent) pairs together, and run
    sweep_entry_thresholds() ONCE on the pooled set. Unchanged this
    revision."""
    print(f"\n{'#'*62}\n  THRESHOLD SELECTION  (selection_seeds={list(selection_seeds)})"
          f"\n{'#'*62}")
    pooled_fitted_paths = []
    pooled_valid = None
    for seed in selection_seeds:
        print(f"\n{'-'*62}\n  [selection] seed {seed}\n{'-'*62}")
        _, fitted_paths, valid = run_cpcv(states, labels, aligned_df, random_state=seed,
                                          n_bagged_fits=n_bagged_fits)
        pooled_fitted_paths.extend(fitted_paths)
        pooled_valid = valid   # valid_mask is seed-independent (comes from labels)

    print(f"\n  Pooled {len(pooled_fitted_paths)} fitted CPCV paths across "
          f"{len(selection_seeds)} selection seed(s) for threshold sweep.")
    sweep = sweep_entry_thresholds(states, labels, pooled_fitted_paths, pooled_valid)

    return {
        "chosen_threshold": sweep["chosen_threshold"],
        "candidates": sweep["candidates"],
        "selection_seeds": list(selection_seeds),
        "n_selection_paths": len(pooled_fitted_paths),
    }


def confirm_gate_nested(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                        entry_threshold: float,
                        confirmation_seeds: tuple = DEFAULT_CONFIRMATION_SEEDS,
                        min_pass_frac: float = DEFAULT_MIN_PASS_FRAC,
                        n_bagged_fits: int = None) -> dict:
    """
    Evaluate the ALREADY-CHOSEN `entry_threshold` (no further tuning)
    against CPCV paths built from `confirmation_seeds`.

    GATE HARDENING (this revision, change 2): in addition to the
    existing per-seed gate + pooled gate, this now ALSO extracts and
    scores the "recent regime" CPCV path (test_groups ==
    RECENT_REGIME_GROUPS — the last N_TEST_GROUPS groups, the closest
    CPCV analog to the real final holdout) from every confirmation
    seed's already-fitted paths, at zero extra fitting cost. A seed's
    recent-regime result "passes" if it has >= RECENT_REGIME_MIN_TEST_TRADES
    trades AND test_avg_pnl > RECENT_REGIME_AVG_PNL_FLOOR (stricter than
    the pooled/whole-CPCV floor — this is specifically a recency-risk
    check). Overall confirmation now additionally requires
    RECENT_REGIME_MIN_PASS_FRAC of seeds' recent-regime results to pass.

    This directly targets the diagnosed failure mode: the pooled/whole-
    CPCV statistics diluted a badly-negative recent-regime path (e.g.
    test_groups=(6,7): test_avg=-2.274% on 9 trades in the run that
    triggered this revision) across 27 other, mostly-fine paths. The
    recent-regime check can no longer be out-voted by older regimes.

    Returns
    -------
    dict with:
      passed               : bool, overall confirmation verdict (now
                              requires per-seed stability AND pooled gate
                              AND recent-regime stability)
      frac_pass             : fraction of confirmation seeds that passed
                               the whole-CPCV gate individually
      per_seed              : list of per-seed {seed, passed, stats}
      pooled_stats           : evaluate_gate() stats on ALL confirmation
                                paths pooled together
      pooled_path_results    : the pooled, threshold-evaluated
                                path_results (for gate_and_summarize())
      recent_regime_per_seed : list of per-seed recent-regime results
                                (or None where unavailable)
      recent_regime_frac_pass: fraction of seeds with an available
                                recent-regime result that passed it
      recent_regime_stable   : bool, recent_regime_frac_pass >=
                                RECENT_REGIME_MIN_PASS_FRAC
    """
    print(f"\n{'#'*62}\n  GATE CONFIRMATION  (confirmation_seeds={list(confirmation_seeds)}, "
          f"entry_threshold={entry_threshold:.2f})\n{'#'*62}")

    per_seed = []
    pooled_path_results = []
    recent_regime_per_seed = []
    for seed in confirmation_seeds:
        print(f"\n{'-'*62}\n  [confirmation] seed {seed}\n{'-'*62}")
        _, fitted_paths, valid = run_cpcv(states, labels, aligned_df, random_state=seed,
                                          n_bagged_fits=n_bagged_fits)
        evaluated = evaluate_paths_at_threshold(states, labels, fitted_paths,
                                                valid, entry_threshold)
        passed, stats = evaluate_gate(evaluated)
        pooled_path_results.extend(evaluated)

        per_seed.append({
            "seed": seed, "passed": bool(passed),
            "n_total": stats["n_total"], "n_pos": stats.get("n_pos", 0),
            "mean_test_avg_pnl": stats["mean_test_avg_pnl"],
            "min_test_avg_pnl": stats["min_test_avg_pnl"],
            "mean_test_sharpe": stats.get("mean_test_sharpe", float("nan")),
            "pbo": stats["pbo"],
        })
        status = "PASS" if passed else "FAIL"
        pbo_str = f"{stats['pbo']:.1%}" if not np.isnan(stats["pbo"]) else "n/a"
        print(f"  [confirmation] seed {seed}: {status}  "
              f"mean_test_avg={stats['mean_test_avg_pnl']:+.4%}  "
              f"positive={stats.get('n_pos', 0)}/{stats['n_total']}  PBO={pbo_str}")

        # ── Recent-regime check (this revision) — reuses fitted_paths,
        #    no additional model fitting. ─────────────────────────────
        recent = _extract_recent_regime_result(states, labels, fitted_paths,
                                               valid, entry_threshold)
        if recent is not None:
            recent["seed"] = seed
            recent["reliable"] = recent["n_test_trades"] >= RECENT_REGIME_MIN_TEST_TRADES
            recent["recent_regime_passed"] = bool(
                recent["reliable"] and recent["test_avg_pnl"] > RECENT_REGIME_AVG_PNL_FLOOR
            )
            rr_status = ("PASS" if recent["recent_regime_passed"] else
                         "FAIL" if recent["reliable"] else "SKIP (too few trades)")
            print(f"  [confirmation] seed {seed} RECENT REGIME "
                  f"{RECENT_REGIME_GROUPS}: {rr_status}  "
                  f"test_avg={recent['test_avg_pnl']:+.4%}  "
                  f"n_trades={recent['n_test_trades']}")
        else:
            print(f"  [confirmation] seed {seed} RECENT REGIME "
                  f"{RECENT_REGIME_GROUPS}: not available (path not "
                  f"generated for this dataset size)")
        recent_regime_per_seed.append(recent)

    n_pass = sum(r["passed"] for r in per_seed)
    frac_pass = n_pass / len(confirmation_seeds)
    per_seed_stable = frac_pass >= min_pass_frac

    pooled_passed, pooled_stats = evaluate_gate(pooled_path_results)

    # ── Recent-regime stability across seeds ───────────────────────────
    rr_reliable = [r for r in recent_regime_per_seed if r is not None and r["reliable"]]
    if rr_reliable:
        rr_n_pass = sum(1 for r in rr_reliable if r["recent_regime_passed"])
        recent_regime_frac_pass = rr_n_pass / len(rr_reliable)
    else:
        rr_n_pass = 0
        recent_regime_frac_pass = float("nan")
    recent_regime_stable = (not np.isnan(recent_regime_frac_pass)
                            and recent_regime_frac_pass >= RECENT_REGIME_MIN_PASS_FRAC)

    overall_passed = per_seed_stable and pooled_passed and recent_regime_stable

    print(f"\n{'='*62}\n  GATE CONFIRMATION VERDICT\n{'='*62}")
    print(f"  Per-seed: {n_pass}/{len(confirmation_seeds)} confirmation seeds passed "
          f"({frac_pass:.0%}, required >= {min_pass_frac:.0%})")
    pooled_pbo_str = (f"{pooled_stats['pbo']:.1%}" if not np.isnan(pooled_stats['pbo'])
                      else "n/a")
    print(f"  Pooled ({pooled_stats['n_total']} reliable paths across all confirmation "
          f"seeds): {'PASS' if pooled_passed else 'FAIL'}  "
          f"mean_test_avg={pooled_stats['mean_test_avg_pnl']:+.4%}  "
          f"positive={pooled_stats.get('n_pos', 0)}/{pooled_stats['n_total']}  "
          f"PBO={pooled_pbo_str}")
    if rr_reliable:
        print(f"  Recent regime {RECENT_REGIME_GROUPS} (this revision): "
              f"{rr_n_pass}/{len(rr_reliable)} seeds passed "
              f"({recent_regime_frac_pass:.0%}, required >= "
              f"{RECENT_REGIME_MIN_PASS_FRAC:.0%})  "
              f"— {'STABLE' if recent_regime_stable else 'UNSTABLE'}")
    else:
        print(f"  Recent regime {RECENT_REGIME_GROUPS}: no seed produced a "
              f"reliable result (>= {RECENT_REGIME_MIN_TEST_TRADES} trades) — "
              f"treated as UNSTABLE (cannot confirm recency risk is absent).")
    print(f"  → {'CONFIRMED' if overall_passed else 'NOT CONFIRMED'} "
          f"(requires per-seed stability AND a passing pooled gate AND "
          f"recent-regime stability)")

    with open(os.path.join(OUT_DIR, "nested_gate_confirmation.json"), "w") as f:
        json.dump({
            "entry_threshold": entry_threshold,
            "confirmation_seeds": list(confirmation_seeds),
            "min_pass_frac": min_pass_frac,
            "n_pass": n_pass, "frac_pass": frac_pass,
            "per_seed_stable": per_seed_stable,
            "pooled_passed": bool(pooled_passed),
            "recent_regime_groups": list(RECENT_REGIME_GROUPS),
            "recent_regime_avg_pnl_floor": RECENT_REGIME_AVG_PNL_FLOOR,
            "recent_regime_min_test_trades": RECENT_REGIME_MIN_TEST_TRADES,
            "recent_regime_per_seed": recent_regime_per_seed,
            "recent_regime_frac_pass": recent_regime_frac_pass,
            "recent_regime_stable": bool(recent_regime_stable),
            "overall_passed": bool(overall_passed),
            "per_seed": per_seed,
            "pooled_stats": {k: v for k, v in pooled_stats.items()
                             if k not in ("reliable_paths", "excluded_paths")},
        }, f, indent=2, default=_json_default)
    print(f"  ✓  Gate confirmation saved → {OUT_DIR}/nested_gate_confirmation.json")

    return {
        "passed": overall_passed, "frac_pass": frac_pass,
        "per_seed": per_seed, "pooled_stats": pooled_stats,
        "pooled_path_results": pooled_path_results,
        "recent_regime_per_seed": recent_regime_per_seed,
        "recent_regime_frac_pass": recent_regime_frac_pass,
        "recent_regime_stable": recent_regime_stable,
    }


def _pick_representative_confirmation_seed(per_seed: list) -> int:
    """Same MEDIAN-by-performance logic as _pick_representative_seed(),
    restricted to confirmation seeds only."""
    passed = [r for r in per_seed if r["passed"]]
    pool = passed if passed else per_seed
    pool_sorted = sorted(pool, key=lambda r: r["mean_test_avg_pnl"])
    median = pool_sorted[len(pool_sorted) // 2]
    return median["seed"]


# ─────────────────────────────────────────────
# Final deployable training
# ─────────────────────────────────────────────

def run_final_training(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                       entry_threshold: float = 0.5, random_state: int = 0,
                       n_bagged_fits: int = None,
                       holdout_frac: float = None):
    """Train the deployable model on all rows except the most recent
    embargo-safe holdout block, then report calibration on that holdout.

    holdout_frac : GATE HARDENING change 4 (this revision) — was
        hardcoded to 0.15; now a parameter (CLI-overridable via
        --holdout-frac) defaulting to DEFAULT_HOLDOUT_FRAC. The run
        that triggered this revision's changes produced only 36 holdout
        trades (a 90% bootstrap CI of [-1.69%, -0.38%] — informative,
        but not tight); widening the holdout window, or narrowing
        TP_MULT/SL_MULT/MAX_HOLDING via their own new CLI flags,
        directly grows this sample.

    n_bagged_fits : forwarded to GBTAgent.fit().

    entry_threshold: the value chosen by main()'s CPCV-only sweep.

    random_state : the representative seed chosen by
        _pick_representative_confirmation_seed().

    DEPLOYMENT GATE: CPCV/stability passing is necessary but not
    sufficient — this function additionally requires the trained
    model's HOLDOUT avg/trade to clear HOLDOUT_AVG_TRADE_FLOOR with at
    least HOLDOUT_MIN_TRADES before it's allowed to overwrite
    gbt_agent_best.joblib. If it doesn't clear the bar, the model is
    still saved (nothing is lost) but under a "_FAILED_HOLDOUT"
    filename, and any previously deployed gbt_agent_best.joblib is left
    untouched.
    """
    holdout_frac = DEFAULT_HOLDOUT_FRAC if holdout_frac is None else holdout_frac

    n = len(states)
    lookback_ticks = compute_required_lookback_ticks()
    holdout_start = int(n * (1 - holdout_frac))
    embargo_start = max(0, holdout_start - lookback_ticks)

    valid = labels["valid_mask"]
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:embargo_start] = True
    train_mask &= valid
    holdout_mask = np.zeros(n, dtype=bool)
    holdout_mask[holdout_start:] = True
    holdout_mask &= valid

    nb = DEFAULT_N_BAGGED_FITS if n_bagged_fits is None else n_bagged_fits
    print(f"\n  Final train: {train_mask.sum():,} rows  |  "
          f"Holdout: {holdout_mask.sum():,} rows  |  "
          f"embargo: {holdout_start - embargo_start} rows  |  "
          f"holdout_frac: {holdout_frac:.2f}  |  "
          f"entry_threshold: {entry_threshold:.2f}  |  random_state: {random_state}  |  "
          f"n_bagged_fits: {nb}")

    agent = GBTAgent(state_dim=states.shape[1], action_dim=ACTION_DIM,
                     entry_threshold=entry_threshold)
    agent.fit(states[train_mask], labels["best_action"][train_mask],
              purge_ticks=lookback_ticks, random_state=random_state,
              n_bagged_fits=nb, **GBT_HYPERPARAMS)

    # ── Calibration report on the genuinely held-out block ──────────
    probs = agent.actor.model.predict_proba(states[holdout_mask])
    slot = {0: 0, 1: 1, 3: 2}
    full_probs = np.zeros((probs.shape[0], 3), dtype=np.float32)
    for j, cls in enumerate(agent.actor.model.classes_):
        full_probs[:, slot[int(cls)]] = probs[:, j]

    y_holdout = labels["best_action"][holdout_mask]
    y_holdout_mapped = np.array([slot[int(c)] for c in y_holdout])

    report = evaluate_calibration(full_probs, y_holdout_mapped,
                                  class_names=["LONG", "SHORT", "HOLD"])
    print(f"\n  ── CALIBRATION (held-out) ──────────────────────────")
    print(f"  Macro ECE   : {report['macro_ece']:.4f}")
    print(f"  Macro Brier : {report['macro_brier']:.4f}")
    for name, m in report["per_class"].items():
        print(f"    {name:<6}: ECE={m['ece']:.4f}  Brier={m['brier']:.4f}  "
              f"n_pos={m['n_positive']}/{m['n_total']}")

    plot_reliability_diagram(full_probs, y_holdout_mapped,
                             ["LONG", "SHORT", "HOLD"],
                             os.path.join(OUT_DIR, "reliability_diagram.png"))

    with open(os.path.join(OUT_DIR, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"  ✓  Calibration report saved → {OUT_DIR}/calibration_report.json")

    test_pnl, n_trades, avg_pnl, trade_returns = simulate_pnl(
        states, labels, holdout_mask, agent, prob_threshold=entry_threshold)
    print(f"\n  Held-out simulated P/L (non-overlapping single-position, "
          f"see simulate_pnl docstring): {test_pnl:+.4%} raw sum  "
          f"({n_trades} trades, avg/trade={avg_pnl:+.4%})  "
          f"@ entry_threshold={entry_threshold:.2f}")

    ci = bootstrap_ci(trade_returns, ci=0.90, random_state=random_state)
    if not np.isnan(ci["ci_lo"]):
        straddles_zero = ci["ci_lo"] < 0 < ci["ci_hi"]
        print(f"  Bootstrap 90% CI on avg/trade: [{ci['ci_lo']:+.4%}, {ci['ci_hi']:+.4%}]"
              + ("  (interval straddles zero — cannot statistically "
                 "distinguish this result from no edge)" if straddles_zero else ""))
        if ci.get("note"):
            print(f"    ⚠ {ci['note']}")
    else:
        print(f"  Bootstrap CI unavailable — {ci.get('note', 'insufficient trades')}")

    deploy_ok = (n_trades >= HOLDOUT_MIN_TRADES) and (avg_pnl > HOLDOUT_AVG_TRADE_FLOOR)

    gate_status = {
        "entry_threshold": entry_threshold, "random_state": random_state,
        "holdout_frac": holdout_frac,
        "holdout_avg_pnl": avg_pnl, "holdout_total_pnl_raw": test_pnl,
        "holdout_n_trades": n_trades,
        "holdout_avg_trade_floor": HOLDOUT_AVG_TRADE_FLOOR,
        "holdout_min_trades": HOLDOUT_MIN_TRADES,
        "holdout_avg_pnl_bootstrap_ci": ci,
        "deploy_gate_passed": bool(deploy_ok),
    }
    with open(os.path.join(OUT_DIR, "deployment_gate.json"), "w") as f:
        json.dump(gate_status, f, indent=2, default=_json_default)

    if deploy_ok:
        save_path = os.path.join(OUT_DIR, "gbt_agent_best.joblib")
        agent.save(save_path)
        print(f"\n  ✓ HOLDOUT GATE PASSED (avg/trade={avg_pnl:+.4%} > "
              f"floor={HOLDOUT_AVG_TRADE_FLOOR:+.2%}, n_trades={n_trades}) "
              f"— deployed → {save_path}")
        if not np.isnan(ci["ci_lo"]) and ci["ci_lo"] < 0:
            print(f"     ⚠ Note: the 90% CI lower bound ({ci['ci_lo']:+.4%}) is "
                  f"still negative — this pass is on the point estimate only; "
                  f"treat as a provisional deploy, not a confirmed edge, until "
                  f"more holdout trades accumulate.")
    else:
        save_path = os.path.join(OUT_DIR, "gbt_agent_candidate_FAILED_HOLDOUT.joblib")
        agent.save(save_path)
        print(f"\n  ⛔ HOLDOUT GATE FAILED — avg/trade={avg_pnl:+.4%} "
              f"(floor={HOLDOUT_AVG_TRADE_FLOOR:+.2%}, n_trades={n_trades} "
              f"vs min={HOLDOUT_MIN_TRADES}).")
        if not np.isnan(ci["ci_lo"]) and ci["ci_hi"] > 0:
            print(f"     Note: the 90% CI [{ci['ci_lo']:+.4%}, {ci['ci_hi']:+.4%}] "
                  f"still includes positive values — this failure may reflect "
                  f"an unlucky small sample rather than a confirmed negative "
                  f"edge. More holdout trades (wider window / faster-resolving "
                  f"barriers) would narrow this before concluding either way.")
        print(f"     Saved as CANDIDATE ONLY → {save_path}. Any existing "
              f"gbt_agent_best.joblib was left untouched — the pipeline will "
              f"not silently deploy a model that lost money on its own holdout.")
        print(f"     Consider: --max-holding / --tp-mult / --sl-mult (more "
              f"holdout trades), --holdout-frac (wider window), lowering "
              f"HOLDOUT_MIN_TRADES only if you also raise --holdout-frac, or "
              f"gathering more data.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    # Declared at the top of main() (must precede any use of these names
    # in this function, including as argparse defaults below) since
    # they're reassigned from CLI overrides further down.
    global PBO_MAX_ALLOWED, MIN_TEST_SHARPE, MIN_EDGE_OVER_BASELINE

    parser = argparse.ArgumentParser(description="Calibrated GBT meta-labeling pipeline")
    parser.add_argument("--force-final-training", action="store_true",
                        help="Proceed to final training even if the nested gate "
                             "confirmation (per-seed + pooled + recent-regime) "
                             "and/or the baseline-edge check flags the result "
                             "as unstable/insufficiently better than the "
                             "logistic baseline.")
    parser.add_argument("--frozen-data", action="store_true",
                        help="Use the existing master CSV(s) as-is instead of "
                             "refetching/appending new raw data.")
    parser.add_argument("--selection-seeds", type=int, nargs="+",
                        default=list(DEFAULT_SELECTION_SEEDS),
                        help="Seeds used ONLY to choose entry_threshold "
                             "(default: %(default)s). Must be disjoint from "
                             "--confirmation-seeds.")
    parser.add_argument("--confirmation-seeds", type=int, nargs="+",
                        default=list(DEFAULT_CONFIRMATION_SEEDS),
                        help="Seeds used ONLY to confirm the gate at the "
                             "already-chosen entry_threshold, and to pick the "
                             "final deployed model's random_state (default: "
                             "%(default)s). Never used for threshold selection.")
    parser.add_argument("--min-pass-frac", type=float, default=DEFAULT_MIN_PASS_FRAC,
                        help="Fraction of confirmation seeds that must "
                             "independently pass the CPCV gate (and, this "
                             "revision, the recent-regime check) for the "
                             "pipeline to proceed to final training "
                             "(default: %(default)s).")
    parser.add_argument("--no-multi-timeframe", action="store_true",
                        help="Disable 15m/1h context (state_dim ~50 instead of "
                             "~122) for this run.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the logistic-regression capacity-floor "
                             "check. NOTE (this revision): skipping this also "
                             "skips the baseline-edge gate (change 3) — with "
                             "--skip-baseline, the pipeline can no longer "
                             "verify the GBT beats a near-linear model, so "
                             "that check is silently treated as passed. Only "
                             "skip for quick iteration, not for a run whose "
                             "verdict you intend to trust.")
    parser.add_argument("--n-bagged-fits", type=int, default=DEFAULT_N_BAGGED_FITS,
                        help="Number of independently-seeded base-estimator "
                             "fits to average per (fold, seed) (default: "
                             "%(default)s).")
    # ── GATE HARDENING (this revision) — new CLI overrides ──────────────
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC,
                        help="Fraction of the dataset (chronologically last) "
                             "held out for final deployment gating (default: "
                             "%(default)s). Widen this to get more holdout "
                             "trades / a tighter bootstrap CI at the cost of "
                             "less final-training data.")
    parser.add_argument("--tp-mult", type=float, default=TP_MULT,
                        help="Triple-barrier take-profit multiplier (default: "
                             "%(default)s). Lower = barriers resolve faster = "
                             "more independent trades per window.")
    parser.add_argument("--sl-mult", type=float, default=SL_MULT,
                        help="Triple-barrier stop-loss multiplier (default: "
                             "%(default)s).")
    parser.add_argument("--max-holding", type=int, default=MAX_HOLDING,
                        help="Triple-barrier vertical (time) barrier in ticks "
                             "(default: %(default)s). Shorter = faster "
                             "resolution = more trades per window.")
    parser.add_argument("--pbo-max-allowed", type=float, default=PBO_MAX_ALLOWED,
                        help="Gate fails if PBO exceeds this fraction "
                             "(default: %(default)s, was implicitly 0.50 "
                             "before this revision).")
    parser.add_argument("--min-test-sharpe", type=float, default=MIN_TEST_SHARPE,
                        help="Gate fails if mean_test_sharpe (when computable) "
                             "is below this floor (default: %(default)s, new "
                             "this revision).")
    parser.add_argument("--min-edge-over-baseline", type=float,
                        default=MIN_EDGE_OVER_BASELINE,
                        help="Minimum required margin (in avg/trade, "
                             "fractional) by which the GBT's pooled "
                             "confirmation mean must beat the logistic "
                             "baseline's pooled mean (default: %(default)s, "
                             "new this revision — see run_logistic_baseline()).")
    args = parser.parse_args()

    # Module-level constants that evaluate_gate()/confirm_gate_nested()
    # read directly are overridden here (rather than threaded as
    # parameters through every call) so existing callers/imports of
    # this module keep working unchanged when these flags aren't passed.
    PBO_MAX_ALLOWED = args.pbo_max_allowed
    MIN_TEST_SHARPE = args.min_test_sharpe
    MIN_EDGE_OVER_BASELINE = args.min_edge_over_baseline

    print("Building states + triple-barrier labels...")
    states, labels, prices, aligned_df = build_states_and_labels(
        frozen=args.frozen_data,
        enable_multi_timeframe=(False if args.no_multi_timeframe else None),
        tp_mult=args.tp_mult, sl_mult=args.sl_mult, max_holding=args.max_holding,
    )
    print(f"  {len(states):,} rows  |  state_dim={states.shape[1]}")
    n_long  = int((labels["best_action"] == 0).sum())
    n_short = int((labels["best_action"] == 1).sum())
    n_hold  = int((labels["best_action"] == 3).sum())
    print(f"  Label balance — LONG:{n_long:,}  SHORT:{n_short:,}  HOLD:{n_hold:,}")

    selection_seeds = tuple(args.selection_seeds)
    confirmation_seeds = tuple(args.confirmation_seeds)
    overlap = set(selection_seeds) & set(confirmation_seeds)
    if overlap:
        raise ValueError(
            f"--selection-seeds and --confirmation-seeds share seed(s) "
            f"{sorted(overlap)} — they must be disjoint. Pick non-overlapping "
            f"seed lists."
        )

    # ── STEP -1 — logistic regression capacity-floor check. This
    #    revision: its pooled mean is now ALSO used below for the
    #    baseline-edge gate (change 3), not just printed. ─────────────────
    baseline_stats = None
    if not args.skip_baseline:
        baseline_stats = run_logistic_baseline(states, labels, aligned_df,
                                               seeds=confirmation_seeds)
    else:
        print("\n  ⏭  Skipping logistic-regression baseline (--skip-baseline). "
              "The baseline-edge gate will be treated as passed.")

    # ── STEP 0 — nested threshold selection (selection seeds only) ────────
    selection = select_threshold_nested(states, labels, aligned_df,
                                        selection_seeds=selection_seeds,
                                        n_bagged_fits=args.n_bagged_fits)
    entry_threshold = selection["chosen_threshold"]

    # ── STEP 1 — gate confirmation (confirmation seeds only, disjoint
    #    from selection; threshold is fixed here, not re-tuned). This
    #    revision: now also includes the recent-regime check. ────────────
    confirmation = confirm_gate_nested(
        states, labels, aligned_df, entry_threshold=entry_threshold,
        confirmation_seeds=confirmation_seeds, min_pass_frac=args.min_pass_frac,
        n_bagged_fits=args.n_bagged_fits,
    )

    gate_and_summarize(
        confirmation["pooled_path_results"], entry_threshold=entry_threshold,
        threshold_sweep=selection, seed=None, write=True,
    )

    # ── STEP 1.5 — baseline-edge gate (this revision, GATE HARDENING
    #    change 3). The GBT's pooled confirmation mean must beat the
    #    logistic baseline's pooled mean by MIN_EDGE_OVER_BASELINE. ──────
    baseline_edge_ok = True
    edge = None
    if baseline_stats is not None:
        gbt_mean = confirmation["pooled_stats"]["mean_test_avg_pnl"]
        baseline_mean = baseline_stats["mean_test_avg_pnl"]
        edge = gbt_mean - baseline_mean
        baseline_edge_ok = edge > MIN_EDGE_OVER_BASELINE
        print(f"\n{'='*62}\n  BASELINE-EDGE GATE (this revision)\n{'='*62}")
        print(f"  GBT pooled mean test avg/trade      : {gbt_mean:+.4%}")
        print(f"  Logistic baseline pooled mean        : {baseline_mean:+.4%}")
        print(f"  Edge (GBT - baseline)                : {edge:+.4%}  "
              f"(required > {MIN_EDGE_OVER_BASELINE:+.4%})")
        if baseline_edge_ok:
            print(f"  → PASSED — GBT shows a real margin over a near-minimal-"
                  f"capacity model, supporting a GBT-specific edge rather "
                  f"than just the shared marginal signal a linear model "
                  f"also finds.")
        else:
            print(f"  ⛔ FAILED — the GBT does not meaningfully beat a "
                  f"near-linear baseline. Per run_logistic_baseline()'s own "
                  f"diagnostic logic, this is evidence the observed edge is "
                  f"a shared, marginal, possibly non-stationary label/regime "
                  f"signal rather than something the GBT's extra capacity is "
                  f"contributing — further GBT tuning is unlikely to close "
                  f"this gap by itself.")

    overall_gate_passed = confirmation["passed"] and baseline_edge_ok

    if not overall_gate_passed and not args.force_final_training:
        if not confirmation["passed"]:
            print("\n  Nested gate confirmation did not pass "
                  "(per-seed / pooled / recent-regime).")
        if not baseline_edge_ok:
            print("\n  Baseline-edge gate did not pass.")
        print("  Re-run with --force-final-training to override.")
        return

    chosen_seed = _pick_representative_confirmation_seed(confirmation["per_seed"])
    print(f"\n{'#'*62}\n  STEP 2 — FINAL DEPLOYABLE TRAINING + CALIBRATION  "
          f"(representative confirmation seed={chosen_seed})\n{'#'*62}")
    run_final_training(states, labels, aligned_df, entry_threshold=entry_threshold,
                       random_state=chosen_seed, n_bagged_fits=args.n_bagged_fits,
                       holdout_frac=args.holdout_frac)


if __name__ == "__main__":
    main()

# python main_gbt.py --min-path-test-trades 15
# python main_gbt.py --disable-multi-timeframe
# python main_gbt.py --frozen-data --force-final-training
# python main_gbt.py --frozen-data --holdout-frac 0.20 --max-holding 16 --tp-mult 1.3 --sl-mult 0.7