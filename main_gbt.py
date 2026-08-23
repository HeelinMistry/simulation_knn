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

CHANGES IN THIS REVISION — MULTI-SEED STABILITY CHECK
───────────────────────────────────────────────────────
Two back-to-back runs of the previous revision (same code, same
pipeline) produced materially different CPCV verdicts: one run showed
PBO=28.6% / 11 of 15 paths positive (gate PASSED), a later run on
ostensibly the same setup showed PBO=57.1% / 9 of 14 paths positive
(gate FAILED). Per-fold trade counts and even which paths survived the
reliability filter (`< 15 test trades`) shifted between runs too. A
single CPCV pass/fail is not trustworthy evidence when the result
itself is this unstable — an edge this close to the noise floor needs
to be confirmed across multiple independent fits, not accepted or
rejected on one draw.

`run_stability_check()` now runs the FULL CPCV → threshold-sweep → gate
pipeline once per random seed (see --seeds), and the pipeline only
proceeds to final training if at least `--min-pass-frac` of those seeds
independently pass the CPCV gate. Results for every seed are logged and
saved to outcomes/gbt/stability_check.json. The seed used for the
actual deployed CPCV summary / threshold choice / final model is picked
as the MEDIAN of mean_test_avg_pnl across the passing seeds (or across
all seeds if none passed and --force-final-training is used) — a
robust central estimate rather than an arbitrary "first seed" or a
cherry-picked best one.

CHANGES IN THIS REVISION — FROZEN DATA
───────────────────────────────────────
The instability described above could equally have been caused by the
underlying dataset itself changing between runs: update_master_data()
refetches/appends new candles every call, and CPCV's group boundaries
are ROW-COUNT based (cpcv._make_groups), so appending rows shifts which
ticks land in which fold even with nothing else different. `--frozen-
data` (see data_manager.py) skips the raw-file scan and loads the
existing master CSV(s) verbatim, so repeated runs — including every
seed inside run_stability_check() — are guaranteed to be evaluating the
exact same dataset. This is now the recommended way to run comparisons;
the pipeline still works without it (defaults to live refresh) for
normal/production runs.

CHANGES IN THIS REVISION — REGULARIZATION TIGHTENED FURTHER
───────────────────────────────────────────────────────────────
Both back-to-back runs above still showed train avg/trade of roughly
+2–5% (hundreds-of-percent raw sums) against test avg/trade under 1%,
even after the previous round's tightening — the model is still
finding fold-specific structure. min_samples_leaf raised further
(400→650), l2_regularization raised (10→14), max_features lowered
(0.45→0.35). This does not by itself fix the seed-to-seed instability
above (that's a data/variance problem, not purely a capacity problem),
but it is a prerequisite: no amount of multi-seed averaging will help
if individual fits are this overfit to their own training fold.

CHANGES IN THIS REVISION — NARROWER TRIPLE-BARRIER CONFIG
───────────────────────────────────────────────────────────
Several CPCV paths and the deployment holdout itself were being judged
on very few resolved trades (some paths under 15, the holdout at
entry_threshold=0.60 had only 2) — too few to trust the resulting
avg/trade estimate regardless of its sign. TP_MULT/SL_MULT narrowed
(2.0/1.0 → 1.5/0.8) and MAX_HOLDING shortened (32 → 20 ticks) so
barriers resolve faster and a "trade" (whether triple-barrier label
generation or simulate_pnl's non-overlapping position simulation)
occupies less of the timeline, allowing more independent trades to be
observed in the same holdout/fold window. This directly increases the
sample size behind every avg/trade and win-rate statistic reported
downstream, at the cost of somewhat smaller max profit-taking distance
per trade.

CHANGES IN THIS REVISION — MULTI-TIMEFRAME TOGGLE FOR COMPARISON
───────────────────────────────────────────────────────────────────
`--no-multi-timeframe` forces ENABLE_MULTI_TIMEFRAME=False for a given
run without editing the file, so a state_dim≈50 (4h-only) run can be
directly compared against the full ≈122-dim multi-timeframe run using
identical CPCV/stability machinery — useful for checking whether the
15m/1h context is actually earning its keep versus just adding
dimensions relative to the ~10-12k fit rows per CPCV fold.

CHANGES IN THIS REVISION — NESTED THRESHOLD SELECTION / GATE CONFIRMATION
───────────────────────────────────────────────────────────────────────────
The previous revision's run_stability_check() picked entry_threshold via
sweep_entry_thresholds() on a seed's own CPCV test folds, then scored
that SAME seed's gate at the chosen threshold — circular: the gate was
confirming a threshold selected to fit the data being used to judge it.
Symptomatically, the sweep's "winning" threshold (0.60) had the fewest
total test trades (522) and a std_test_avg_pnl more than 3x its mean —
the noisiest candidate of the five, selected anyway because "best
mean/std" over a shrinking sample is exactly what multiple-comparison
selection inflates.

Fixed by splitting seeds into two DISJOINT pools:
  - select_threshold_nested() fits CPCV on --selection-seeds only, pools
    every fitted path across those seeds, and sweeps entry_threshold
    ONCE on the pooled set.
  - confirm_gate_nested() then evaluates that fixed, already-chosen
    threshold against CPCV paths from --confirmation-seeds — seeds the
    threshold has never seen in any form — both per-seed (multi-seed
    stability, same idea as the old run_stability_check) and pooled
    across all confirmation seeds (the number reported as "the" CPCV
    summary). The deployed model's random_state is picked as the
    median-performing CONFIRMATION seed only.
run_stability_check()/--seeds are kept for backward compatibility but
are no longer called by main() — see select_threshold_nested()/
confirm_gate_nested() below.

CHANGES IN THIS REVISION — BOOTSTRAP CI ON THE HOLDOUT RESULT
───────────────────────────────────────────────────────────────
run_final_training()'s holdout avg/trade point estimate (e.g.
"-1.13% over 10 trades") was being read as a clean pass/fail fact, but
at n=10 the standard error is comparable in magnitude to the estimate
itself — CPCV path-level test_avg_pnl at similar sample sizes swings
from -0.81% to +1.13%. bootstrap_ci() now resamples the holdout's own
trade_returns to report a 90% CI alongside the point estimate, and both
the PASS and FAIL branches print a note when that interval straddles
(or on the FAIL side, still partly overlaps) zero, so a gate verdict on
a small holdout isn't mistaken for a statistically confirmed result.
The pass/fail RULE itself is unchanged (point estimate vs floor) — the
CI is reported for interpretation, not substituted into the gate logic.

CHANGES IN THIS REVISION — CAPACITY ROUND 4 + LOGISTIC BASELINE (A/B-DRIVEN)
───────────────────────────────────────────────────────────────────────────
A --no-multi-timeframe A/B run (state_dim ~122 -> ~50) was used to test
whether feature dimensionality was the dominant driver of the
train/test gap and PBO>50% instability seen in prior runs. It wasn't:
train_avg_pnl stayed essentially the same magnitude in the ~50-dim run
as the ~122-dim run, while out-of-sample performance got WORSE (mean
test avg/trade +0.196% -> +0.046%, positive paths 31/54 -> 26/54,
0/3 confirmation seeds passing vs 1/3 before). Two conclusions follow:
  1. ENABLE_MULTI_TIMEFRAME stays True — the 15m/1h context is
     contributing real signal, not just extra noise dimensions.
  2. GBT_HYPERPARAMS is tightened again (round 4), this time targeting
     tree capacity/iteration count DIRECTLY (shallower trees, larger
     min_samples_leaf, fewer/smaller boosting steps, higher L2) rather
     than via feature count, since dimensionality is now ruled out as
     the dominant lever.
Additionally, run_logistic_baseline() (new) fits a near-minimal-capacity
logistic regression through the exact same CPCV paths/purge/embargo/
labels as a floor check: if even a linear model can't clear PBO<=50%
on this state/label setup, that's evidence the instability is a label/
regime problem rather than something further GBT tuning can fix. Runs
automatically before threshold selection unless --skip-baseline is
passed; is purely informational and does not gate the pipeline.

CHANGES IN THIS REVISION — BAGGED FITS + WIDER CONFIRMATION POOL
───────────────────────────────────────────────────────────────────────
Diagnosis of the previous nested-gate run: confirmation seeds 2/3/4
disagreed on pass/fail (1/3 passed) even though generate_cpcv_paths()'s
train/test masks are ROW-COUNT based and therefore IDENTICAL across
those seeds — only the classifier's internal RNG (HGB's max_features
subsampling, early-stopping's validation carve-out) differed. That's a
stronger instability signal than "different folds disagreed" would be:
the verdict on the SAME evidence flips depending on fit randomness.

Two changes address this directly:
  1. gbt_agent.py's GBTPolicy.fit() now supports `n_bagged_fits`: it
     fits several independently-seeded members on the SAME fit/cal
     split and averages their calibrated probabilities
     (_BaggedGBTModel). This is threaded through run_cpcv(),
     select_threshold_nested(), confirm_gate_nested(), and
     run_final_training() here via DEFAULT_N_BAGGED_FITS / the new
     --n-bagged-fits flag. Logistic-baseline fits are unaffected
     (bagging is a no-op there — see gbt_agent.py).
  2. DEFAULT_CONFIRMATION_SEEDS widened from 3 to 6 seeds — a 3-seed
     pass-fraction only ever takes 3 possible values (0%, 33%, 67%,
     100%), too coarse to tell "bagging helped" from noise. 6 seeds
     gives a finer read on whether stability actually improved.

Additionally, run_logistic_baseline() now reports PER-SEED pass/fail
(previously pooled-only), so it can be compared apples-to-apples
against the GBT's own per-seed confirmation breakdown — needed to
actually tell whether the simpler model is more STABLE, not just
whether its pooled average looks fine.

Run
────
    python main_gbt.py                                              # gated pipeline, live data
    python main_gbt.py --frozen-data                                 # reproducible comparisons
    python main_gbt.py --frozen-data --selection-seeds 0 1 \\
                        --confirmation-seeds 2 3 4 5 6 7             # explicit nested seed pools
    python main_gbt.py --frozen-data --no-multi-timeframe            # 4h-only baseline
    python main_gbt.py --frozen-data --n-bagged-fits 5                # override bag size
    python main_gbt.py --force-final-training                        # override an unstable gate
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

# Triple-barrier config (narrower this revision — see module docstring:
# faster resolution -> more independent trades per fold/holdout window,
# so avg/trade estimates are backed by a larger sample).
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

# Base-model regularization (overfitting fix, round 4 — see module
# docstring). Rounds 1-3 progressively tightened min_samples_leaf/l2/
# max_features, aimed at reducing the state space's effective capacity
# relative to ~10-12k fit rows per CPCV fold. A --no-multi-timeframe
# A/B run (state_dim ~122 -> ~50) directly tested whether that
# dimensionality was the dominant driver: it wasn't. train_avg_pnl
# stayed at essentially the SAME magnitude (~1.7-3.5% per trade) in the
# ~50-dim run as in the ~122-dim run, while test performance actually
# got WORSE (mean_test_avg +0.196% -> +0.046%, positive paths 31/54 ->
# 26/54) — i.e. the 15m/1h context was contributing real signal, and
# cutting feature count did nothing to shrink the train/test gap that
# was supposedly caused by dimensionality. That falsifies "too many
# features" as the dominant cause.
#
# Round 4 therefore attacks CAPACITY/ITERATIONS DIRECTLY instead of via
# dimensionality: much shallower trees (depth 3->2, leaf nodes 8->4),
# a much larger min_samples_leaf (650->1000), fewer/smaller boosting
# steps (max_iter 150->80, learning_rate 0.04->0.02), and higher L2
# (14->20) — while KEEPING the full multi-timeframe state (see
# ENABLE_MULTI_TIMEFRAME below), since removing it measurably hurt
# out-of-sample performance in the A/B test above.
GBT_HYPERPARAMS = dict(
    max_iter=80,               # was 150 -> 80
    learning_rate=0.02,        # was 0.04 -> 0.02
    max_depth=2,                # was 3 -> 2
    max_leaf_nodes=4,           # was 8 -> 4
    min_samples_leaf=1000,      # was 650 -> 1000
    l2_regularization=20.0,     # was 14.0 -> 20.0
    validation_fraction=0.15,
    n_iter_no_change=15,
    max_features=0.35,          # unchanged — dimensionality already
                                 # ruled out as the dominant lever above
)

# ── Entry-conviction threshold sweep (CPCV-only — see module docstring) ─────
ENTRY_THRESHOLD_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70)
# A threshold whose aggregate CPCV test trade count falls below this is
# disqualified regardless of how good its per-trade stats look — a few
# great-looking trades on a tiny sample isn't trustworthy.
MIN_SWEEP_TRADES = 150
# Individual CPCV paths with fewer test trades than this are excluded
# from the gate's per-path statistics (their avg/trade is too noisy to
# trust even though the aggregate sweep above might still be fine).
MIN_PATH_TEST_TRADES = 15

# ── Multi-seed CPCV stability check (this revision) ──────────────────────────
DEFAULT_STABILITY_SEEDS = (0, 1, 2)
DEFAULT_MIN_PASS_FRAC   = 0.6   # >= 60% of seeds must independently pass

# ── Nested threshold selection / gate confirmation (this revision) ──────────
# PROBLEM THIS FIXES: the previous flow picked entry_threshold by sweeping
# over CPCV test-fold performance (sweep_entry_thresholds), then evaluated
# the CPCV "gate" on those SAME test folds at the chosen threshold. That's
# circular — the gate was confirming a threshold that was chosen to look
# good on exactly the data being used to judge it. Concretely, the
# threshold=0.60 candidate in the observed run had total_test_trades=522
# (smallest of the 5 candidates) and std_test_avg_pnl (2.09%) more than 3x
# its mean (0.60%) — it was the noisiest candidate, and the sweep picked it
# anyway because "best mean/std" on a shrinking, noisy sample is exactly
# the kind of statistic multiple-comparison selection inflates.
#
# FIX: split seeds into two DISJOINT pools.
#   - SELECTION_SEEDS build CPCV paths used ONLY to choose entry_threshold
#     (sweep_entry_thresholds, pooled across all selection seeds' paths).
#   - CONFIRMATION_SEEDS build a completely separate set of CPCV paths,
#     never used for threshold selection, and the gate (directional
#     consistency + downside floor + PBO) is evaluated ONLY on those, AT
#     the already-chosen threshold (no further tuning). This makes the
#     gate a genuine out-of-sample check on the threshold decision itself,
#     not just on that threshold's fit to a given path's train/test split.
# The final deployed model's random_state is picked as the median-by-
# performance CONFIRMATION seed (never a selection seed), so the reported
# CPCV numbers, the gate verdict, and the deployed model are all drawn
# from the confirmation pool alone.
DEFAULT_SELECTION_SEEDS    = (0, 1)
# Widened from (2, 3, 4) — a 3-seed pass-fraction only takes 4 possible
# values (0/3, 1/3, 2/3, 3/3), too coarse to distinguish "bagging fixed
# the instability" from noise. 6 seeds gives finer resolution on the
# per-seed stability check in confirm_gate_nested(). See module
# docstring's "BAGGED FITS + WIDER CONFIRMATION POOL" section.
DEFAULT_CONFIRMATION_SEEDS = (2, 3, 4, 5, 6, 7)

# ── Bagged fits (this revision) ──────────────────────────────────────────────
# Number of independently-seeded base-estimator fits GBTPolicy.fit()
# averages together per (fold, seed) call — see gbt_agent.py's
# "BAGGED FITS" docstring section for the full rationale. Threaded
# through run_cpcv()/select_threshold_nested()/confirm_gate_nested()/
# run_final_training() below and overridable via --n-bagged-fits.
# Ignored (forced to 1) for model_type="logistic" — see gbt_agent.py.
DEFAULT_N_BAGGED_FITS = 5

# ── Final holdout deployment gate ────────────────────────────────────────────
# Distinct from VAL_AVG_TRADE_FLOOR (a CPCV-path floor): this gates the
# actual deployable model's performance on ITS OWN held-out block, at
# the entry_threshold that will actually be used live.
HOLDOUT_AVG_TRADE_FLOOR = 0.0    # require a non-negative mean holdout trade
HOLDOUT_MIN_TRADES      = 10     # fewer trades than this -> gate can't be trusted either way

# Stability gate (mirrors main_mcknn.py's philosophy — directional
# consistency + downside floor — plus a PBO check CPCV newly enables).
#
# NORMALIZED, not raw-sum: VAL_AVG_TRADE_FLOOR is a floor on the WORST
# path's MEAN PER-TRADE test return, so a fold's severity is judged
# per-bet rather than being amplified/muted by however many trades that
# particular fold happened to generate.
VAL_AVG_TRADE_FLOOR = -0.03

OUT_DIR = "outcomes/gbt"
os.makedirs(OUT_DIR, exist_ok=True)


def _json_default(obj):
    """
    Fallback encoder for json.dump(default=_json_default) calls in this
    module. numpy scalar types (np.bool_, np.int64, np.float32/64, ...)
    are not JSON-serializable even though some print/repr as if they
    were native Python types (np.bool_'s class name is literally "bool",
    which is what made the earlier TypeError's message read
    "Object of type bool is not JSON serializable" and look confusing).
    Converting the ROOT CAUSE (evaluate_gate()'s numpy comparisons) to
    native bool/float at the source is the real fix; this is a defense-
    in-depth net so any other numpy scalar that slips into a dict here
    doesn't crash the run instead of just writing a slightly-off value.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ─────────────────────────────────────────────
# State + label construction
# ─────────────────────────────────────────────

def build_states_and_labels(frozen: bool = False,
                             enable_multi_timeframe: bool = None,
                             context_paces: tuple = None):
    """
    Build (states, labels, prices, aligned_df) for the full dataset,
    using identical feature construction to main_mcknn.py/pre_training.py
    so results are directly comparable.

    Parameters
    ----------
    frozen : forwarded to data_manager.update_master_data() /
             update_all_timeframes() — if True, uses the existing master
             CSV(s) verbatim instead of refetching/appending, so repeated
             calls (e.g. across run_stability_check()'s seeds) all see
             the exact same dataset. See data_manager.py and this
             module's docstring.
    enable_multi_timeframe, context_paces : override the module-level
             ENABLE_MULTI_TIMEFRAME / CONTEXT_PACES constants for this
             call only (used by --no-multi-timeframe), so a 4h-only
             baseline can be built without editing the file.
    """
    enable_multi_timeframe = (ENABLE_MULTI_TIMEFRAME if enable_multi_timeframe is None
                               else enable_multi_timeframe)
    context_paces = CONTEXT_PACES if context_paces is None else context_paces

    df = update_master_data("4h", frozen=frozen)
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)

    ind = df[FEATURES].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)
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
        # portfolio_info=None -> StateAggregator appends 2 zero dims;
        # strip them (this is entry-decision modeling, not live position
        # tracking — GBTAgent.select_action handles in-position CLOSE/
        # HOLD separately via _in_position_probs, without needing
        # portfolio features baked into the training state).
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

    labels = build_meta_labels(
        prices_aligned, atr_aligned,
        tp_mult=TP_MULT, sl_mult=SL_MULT,
        max_holding=MAX_HOLDING, commission=COMMISSION,
    )

    open_times = df["Open_time"].values[WARMUP_IDX + 1:]
    aligned_df = pd.DataFrame({"Open_time": open_times})   # used only for
                                                            # its length by
                                                            # cpcv.py's group
                                                            # splitting

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
    `inventory` deque) — at each tick in `mask` that isn't still
    "inside" a previously opened trade's holding window, take the
    higher-probability directional class if its calibrated probability
    exceeds `prob_threshold`, credit that tick's own long_return/
    short_return, and then skip every tick up to and including that
    trade's touch tick before considering another entry.

    Returns
    -------
    (total_pnl, n_trades, avg_pnl, trade_returns)
      total_pnl     : raw sum of realised trade returns (logging only —
                       NOT trade-count-normalized, do not use for gating).
      n_trades      : number of trades taken.
      avg_pnl       : mean per-trade realised return (trade-count-
                       independent — use this for cross-fold comparison).
      trade_returns : list of individual trade returns, for computing
                       a Sharpe-like stat (see run_cpcv()).
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
    """
    Same formula diagnostic_gbt.py's write_summary() already uses for
    its real tick-by-tick backtest (pnl_arr.mean() / pnl_arr.std() *
    sqrt(n)) — kept identical here so the two pipelines' notion of
    "risk-adjusted edge" is directly comparable. Returns NaN when there
    are too few trades or zero variance to compute a meaningful ratio.
    """
    if len(trade_returns) < 5:
        return float("nan")
    arr = np.array(trade_returns, dtype=np.float64)
    std = arr.std()
    if std == 0:
        return float("nan")
    return float(arr.mean() / std * np.sqrt(len(arr)))


def bootstrap_ci(trade_returns: list, n_boot: int = 10_000, ci: float = 0.90,
                 random_state: int = 0) -> dict:
    """
    Percentile bootstrap CI on the mean per-trade return.

    WHY THIS EXISTS: a point estimate like "avg/trade=-1.13% (n=10)" reads
    as a definitive result but isn't one — with few trades and typical
    per-trade volatility in this pipeline (CPCV path-level test_avg_pnl
    swings from -0.81% to +1.13% at similar sample sizes), the standard
    error can be comparable to or larger than the point estimate itself.
    Resampling trade_returns with replacement and taking the mean each
    time gives an empirical distribution of "what avg/trade would plausibly
    look like from this same underlying process", so the gate's pass/fail
    decision can be read alongside how much that decision could have swung
    on a slightly different sample.

    Returns
    -------
    dict with:
      mean       : point estimate (mean of trade_returns, matches simulate_pnl's avg_pnl)
      ci_lo/ci_hi: (1-ci)/2 and 1-(1-ci)/2 percentiles of the bootstrap
                   distribution of the mean (e.g. ci=0.90 -> 5th/95th pct)
      ci_level   : the requested CI level (for display)
      n_trades   : sample size actually used
      note       : set when n_trades is too small for a meaningful interval
    """
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

    random_state : forwarded to every fold's GBTAgent.fit() (which
        passes it through to sklearn's HistGradientBoostingClassifier).
        Exposed as a parameter (this revision) so run_stability_check()
        can re-run the entire CPCV pass under several independent seeds
        without touching anything else — see module docstring.

    model_type, hyperparams : forwarded to GBTAgent/GBTPolicy — lets a
        caller swap the base classifier (e.g. model_type="logistic" for
        run_logistic_baseline()'s capacity-floor check) without
        duplicating any of the CPCV path-generation/purge/simulate_pnl
        machinery below. Defaults ("hgb", None -> GBT_HYPERPARAMS)
        reproduce the exact prior behaviour.

    n_bagged_fits : forwarded to GBTAgent.fit()/GBTPolicy.fit() — number
        of independently-seeded members averaged per fold (see
        gbt_agent.py's "BAGGED FITS" docstring section). Defaults to
        DEFAULT_N_BAGGED_FITS when None. Ignored (forced to 1) for
        model_type="logistic", so run_logistic_baseline() doesn't pay
        the extra compute for a no-op.

    fitted_paths: list of (CPCVPath, GBTAgent) for every path that was
    actually fit — kept so sweep_entry_thresholds()/
    evaluate_paths_at_threshold() can re-score the SAME fitted models
    at different entry-probability thresholds without the cost (and
    the subtle risk of a different random_state/early-stopping path)
    of refitting per threshold.
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
        # purge_ticks: the calibration split GBTPolicy.fit() carves out
        # of THIS fold's train rows is chronological (last 20% by
        # default) — pass the same lookback_ticks used for the outer
        # CPCV purge/embargo so that internal fit/calibration split gets
        # an equivalent purge buffer instead of a raw random split that
        # could leak across overlapping rolling-window lookback.
        agent.fit(states[train_mask], y[train_mask],
                  purge_ticks=lookback_ticks, random_state=random_state,
                  n_bagged_fits=nb, **hp)
        fitted_paths.append((path, agent))

        # Baseline (threshold=0.5) results — used for human-readable
        # per-path logging here only. The actual gating decision in
        # main() uses evaluate_paths_at_threshold() at the SWEPT
        # threshold, computed after this loop returns.
        train_pnl, n_train_trades, train_avg_pnl, train_returns = simulate_pnl(
            states, labels, train_mask, agent)
        test_pnl, n_test_trades, test_avg_pnl, test_returns = simulate_pnl(
            states, labels, test_mask, agent)
        test_sharpe = _sharpe_like(test_returns)

        path_results.append({
            "path_id": path.path_id, "test_groups": list(path.test_groups),
            # Raw sums — logging/context only, NOT trade-count-normalized.
            "train_pnl": train_pnl, "test_pnl": test_pnl,
            # Normalized — baseline @ threshold=0.5, context only now.
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
    """
    Re-score every already-fitted CPCV (path, agent) at each candidate
    entry-probability threshold — cheap, no refitting — and pick the
    threshold that scores best OUT OF SAMPLE across CPCV test folds
    only. The holdout block is never touched here, so the chosen
    threshold can't leak holdout information the way hand-tuning it
    against the final backtest would.

    Score = mean(test_avg_pnl across paths) / (std(test_avg_pnl across
    paths) + eps) — a Sharpe-like ranking across INDEPENDENT CPCV
    paths, so a threshold with a good mean but wildly inconsistent
    per-path results scores worse than one that's modest but reliable.
    Thresholds whose aggregate test trade count falls below
    MIN_SWEEP_TRADES are disqualified (score = -inf) regardless of how
    good their stats look — too few trades to trust.
    """
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
    """Re-score every fitted CPCV path at `threshold` (no refitting),
    producing the same path_results shape run_cpcv() does — this is
    what actually gets gated/logged/saved to cpcv_summary.json, so the
    reported CPCV numbers match the threshold that will be deployed."""
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


def evaluate_gate(path_results: list) -> tuple:
    """
    Pure (no I/O) gate evaluation, factored out of gate_and_summarize()
    (this revision) so run_stability_check() can cheaply check pass/fail
    for every seed without printing/writing a full summary each time.

    Applies the reliability filter (MIN_PATH_TEST_TRADES) first, exactly
    as gate_and_summarize() does, then checks directional consistency +
    downside floor + PBO.

    Returns
    -------
    (passed: bool, stats: dict)
      stats contains everything gate_and_summarize() needs to print/save,
      plus the filtered path_results themselves under "reliable_paths".
    """
    reliable = [r for r in path_results if r["n_test_trades"] >= MIN_PATH_TEST_TRADES]
    excluded = [r for r in path_results if r["n_test_trades"] < MIN_PATH_TEST_TRADES]

    if not reliable:
        return False, {
            "n_total": 0, "n_pos": 0, "pbo": float("nan"),
            "mean_test_avg_pnl": float("nan"), "std_test_avg_pnl": float("nan"),
            "min_test_avg_pnl": float("nan"), "reliable_paths": [],
            "excluded_paths": excluded,
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

    directionally_consistent = bool(n_pos >= (n_total / 2))
    downside_breach = bool(test_avg_pnls.min() < VAL_AVG_TRADE_FLOOR)
    pbo_high = bool((not np.isnan(pbo_info["pbo"])) and pbo_info["pbo"] > 0.5)
    passed = directionally_consistent and (not downside_breach) and (not pbo_high)

    stats = {
        "n_total": n_total, "n_pos": n_pos,
        "mean_test_avg_pnl": float(test_avg_pnls.mean()),
        "std_test_avg_pnl": float(test_avg_pnls.std()),
        "min_test_avg_pnl": float(test_avg_pnls.min()),
        "mean_test_pnl_raw": float(test_pnls.mean()),
        "std_test_pnl_raw": float(test_pnls.std()),
        "min_test_pnl_raw": float(test_pnls.min()),
        "mean_test_sharpe": float(np.nanmean(sharpes)) if len(sharpes) else float("nan"),
        "pbo": pbo_info["pbo"], "logit_lambda": pbo_info["logit_lambda"],
        "directionally_consistent": directionally_consistent,
        "downside_breach": downside_breach, "pbo_high": pbo_high,
        "reliable_paths": reliable, "excluded_paths": excluded,
    }
    return passed, stats


def gate_and_summarize(path_results: list, entry_threshold: float = 0.5,
                       threshold_sweep: dict = None, seed: int = None,
                       write: bool = True) -> bool:
    """
    NORMALIZED gate: decisions are based on test_avg_pnl (mean per-trade
    test return) and test_sharpe, not the raw trade-count-scaled
    test_pnl sum. Raw sums are still reported for context.

    Additionally applies a per-path RELIABILITY FILTER: any CPCV path
    with fewer than MIN_PATH_TEST_TRADES test trades is excluded from
    the gate's statistics entirely — a handful of "lucky"/"unlucky"
    trades on a near-empty fold shouldn't be able to swing the
    directional-consistency or PBO verdict.

    path_results here is expected to already be evaluated AT
    entry_threshold (see evaluate_paths_at_threshold()), so the gate's
    accept/reject decision reflects the exact operating point that will
    be deployed.
    """
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
        print(f"  Mean test Sharpe    : {stats['mean_test_sharpe']:+.3f}")
    print(f"  Positive paths      : {n_pos}/{n_total}")
    print(f"  PBO                 : {stats['pbo']:.1%}  "
          f"(fraction of in-sample-good paths that disappointed "
          f"out-of-sample — lower is better; >50% means in-sample "
          f"selection is worse than a coin flip)")
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
                "mean_test_pnl_raw": stats["mean_test_pnl_raw"],
                "std_test_pnl_raw": stats["std_test_pnl_raw"],
                "min_test_pnl_raw": stats["min_test_pnl_raw"],
                "n_paths_positive": n_pos,
                "pbo": stats["pbo"], "logit_lambda": stats["logit_lambda"],
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
            print(f"     ✗ PBO={stats['pbo']:.1%} > 50% — in-sample "
                  f"performance is not predictive of out-of-sample performance")
        return False

    print(f"\n  ✓ CPCV gate passed.")
    return True


# ─────────────────────────────────────────────
# Logistic regression baseline (capacity floor check)
# ─────────────────────────────────────────────
#
# WHY THIS EXISTS: the --no-multi-timeframe A/B run ruled out feature
# dimensionality as the dominant driver of GBT instability — cutting
# state_dim from ~122 to ~50 didn't shrink train_avg_pnl at all, it
# just made test performance worse. That leaves two live hypotheses
# for the remaining PBO>50% instability:
#   (a) the GBT still has more capacity than this label/feature
#       relationship can support, even independent of dimension count
#       (addressed by GBT_HYPERPARAMS' round-4 tightening above), or
#   (b) the instability is inherent to the label/regime relationship
#       itself (crypto's non-stationarity across regimes), in which
#       case NO classifier — however regularized — will show PBO<=50%
#       on this exact CPCV split, and the fix has to be structural
#       (regime-conditional models, shorter re-fit windows, a smaller/
#       more selective trading footprint) rather than more tuning.
#
# A near-linear model (logistic regression, effectively minimal
# capacity) run through the EXACT SAME CPCV paths/purge/embargo/labels
# is a cheap way to tell these apart. If logistic ALSO fails PBO<=50%,
# that's evidence for (b) — the floor itself is broken, not the GBT's
# capacity. If logistic passes while the GBT still fails, that's
# evidence for (a) — the GBT genuinely has room to give back capacity.

def run_logistic_baseline(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                          seeds: tuple = None) -> dict:
    """
    Fit a plain logistic regression (via GBTAgent(model_type="logistic"))
    through the same CPCV path generation / purge-embargo / triple-
    barrier labels as the GBT pipeline, at the default entry_threshold
    (0.5 — no threshold sweep here; this is a floor check, not a
    deployment candidate). Pools results across `seeds` (defaults to
    DEFAULT_CONFIRMATION_SEEDS, so it's directly comparable to the
    GBT's confirmation-pool numbers) and reports the same PBO / mean
    test avg/trade / positive-path-fraction stats the GBT gate uses.

    PER-SEED REPORTING (this revision): previously this only reported
    the POOLED verdict, which can't distinguish "logistic is genuinely
    more stable per-seed" from "logistic's pooled average happens to
    look fine". confirm_gate_nested() already reports the GBT's own
    per-seed pass/fail on these same confirmation seeds — this function
    now computes the equivalent for logistic (evaluate_gate() on each
    seed's own path_results) so the two are directly comparable. If
    logistic is MORE stable per-seed at similar or better pooled
    performance, that's stronger evidence the GBT's instability is a
    fixable capacity/fit-variance problem (see gbt_agent.py's bagging)
    rather than a broken label/regime floor; if logistic is EQUALLY
    unstable per-seed, that points the other way (see docstring below).

    Returns the pooled evaluate_gate() stats dict (with a "per_seed"
    key added) and also writes the same information to
    outcomes/gbt/logistic_baseline.json.
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

    if not np.isnan(stats["pbo"]) and stats["pbo"] > 0.5:
        print(f"\n  → Baseline ALSO fails PBO<=50% at near-minimal model "
              f"capacity. This supports a label/regime explanation for the "
              f"GBT's instability over a pure capacity/overfitting "
              f"explanation — further GBT regularization (or bagging) is "
              f"unlikely to be the fix by itself; consider regime-"
              f"conditional models, shorter re-fit windows, or a smaller/"
              f"more selective trading footprint.")
    elif np.isnan(stats["pbo"]):
        print(f"\n  → Not enough reliable logistic paths to compute a PBO "
              f"verdict — treat as inconclusive, not a pass.")
    else:
        print(f"\n  → Baseline PASSES PBO<=50% at near-minimal capacity. "
              f"This supports treating the GBT's instability as an "
              f"addressable capacity/fit-variance problem rather than a "
              f"broken label/regime floor.")

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
    """
    DEPRECATED as of the nested-selection revision — main() no longer
    calls this. It picks entry_threshold via sweep_entry_thresholds()
    on the SAME CPCV paths that evaluate_gate() then scores per seed,
    so a seed's "pass" partly reflects that its own threshold was
    chosen to fit its own test folds — see select_threshold_nested() /
    confirm_gate_nested() below for the fix (disjoint seed pools for
    threshold selection vs. gate confirmation). Left in place only for
    backward compatibility with any external caller.

    Run the full CPCV -> threshold-sweep -> gate pipeline once per seed
    in `seeds`, entirely independently, and require that at least
    `min_pass_frac` of them pass the CPCV gate before the overall
    pipeline is allowed to proceed to final training.

    This exists because two back-to-back runs of the previous revision
    (same code, same nominal pipeline) produced flatly contradictory
    CPCV verdicts (PBO 28.6% pass vs PBO 57.1% fail) — a single run is
    not enough evidence either way when the result is this close to the
    noise floor. See module docstring.

    Returns
    -------
    (stable: bool, seed_results: list[dict])
      seed_results[i] has keys: seed, passed, entry_threshold, plus all
      of evaluate_gate()'s numeric stats (mean_test_avg_pnl, pbo, etc).
      Also written to outcomes/gbt/stability_check.json.
    """
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
    for r in seed_results:
        print(f"    seed={r['seed']}  {'PASS' if r['passed'] else 'FAIL'}  "
              f"mean_test_avg={r['mean_test_avg_pnl']:+.4%}  PBO={r['pbo']:.1%}"
              if not np.isnan(r["pbo"]) else
              f"    seed={r['seed']}  {'PASS' if r['passed'] else 'FAIL'}  "
              f"mean_test_avg={r['mean_test_avg_pnl']:+.4%}  PBO=n/a")
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
    """
    Choose the seed to actually deploy from among run_stability_check()'s
    results: the MEDIAN by mean_test_avg_pnl among the seeds that
    passed (a robust central estimate, not a cherry-picked best-case
    seed), or the median across ALL seeds if none passed (only reached
    via --force-final-training).
    """
    passed = [r for r in seed_results if r["passed"]]
    pool = passed if passed else seed_results
    pool_sorted = sorted(pool, key=lambda r: r["mean_test_avg_pnl"])
    median = pool_sorted[len(pool_sorted) // 2]
    return median["seed"]


# ─────────────────────────────────────────────
# Nested threshold selection + gate confirmation (this revision)
# ─────────────────────────────────────────────
#
# See DEFAULT_SELECTION_SEEDS / DEFAULT_CONFIRMATION_SEEDS docstring
# above for the full rationale. In one line: threshold selection and
# gate confirmation must draw on disjoint data, or the gate is just
# confirming a threshold that was chosen to fit it.

def select_threshold_nested(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                            selection_seeds: tuple = DEFAULT_SELECTION_SEEDS,
                            n_bagged_fits: int = None) -> dict:
    """
    Fit CPCV paths for every seed in `selection_seeds`, POOL all of
    their fitted (path, agent) pairs together, and run
    sweep_entry_thresholds() ONCE on the pooled set.

    Pooling (rather than sweeping per-seed and averaging the chosen
    thresholds) means the sweep's own eligibility filter
    (MIN_SWEEP_TRADES) and mean/std statistics are computed over the
    combined trade count across all selection seeds — a threshold that
    only looks good in one seed's idiosyncratic path split gets diluted
    by the others, rather than each seed independently overweighting
    whatever the noisiest-but-highest-scoring candidate happened to be
    for it.

    Returns
    -------
    dict: {"chosen_threshold", "candidates", "selection_seeds",
           "n_selection_paths"} — everything needed to log/save the
    selection step separately from the confirmation step below.
    """
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
    against CPCV paths built from `confirmation_seeds` — seeds that
    were never used in select_threshold_nested(). This is the genuine
    out-of-sample check on the threshold decision: every one of these
    paths' test folds is data the threshold has not seen in any form.

    Runs the gate (directional consistency + downside floor + PBO)
    independently per confirmation seed AND on the pooled confirmation
    set, and requires >= min_pass_frac of the per-seed gates to pass —
    same multi-seed stability logic as the old run_stability_check(),
    just applied after threshold selection instead of interleaved
    with it.

    Returns
    -------
    dict with:
      passed             : bool, overall confirmation verdict
      frac_pass          : fraction of confirmation seeds that passed individually
      per_seed           : list of per-seed {seed, passed, stats}
      pooled_stats        : evaluate_gate() stats on ALL confirmation
                             paths pooled together (the number reported
                             as "the" CPCV summary for this threshold)
      pooled_path_results : the pooled, threshold-evaluated path_results
                             (for gate_and_summarize()/json export)
    """
    print(f"\n{'#'*62}\n  GATE CONFIRMATION  (confirmation_seeds={list(confirmation_seeds)}, "
          f"entry_threshold={entry_threshold:.2f})\n{'#'*62}")

    per_seed = []
    pooled_path_results = []
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
            "pbo": stats["pbo"],
        })
        status = "PASS" if passed else "FAIL"
        pbo_str = f"{stats['pbo']:.1%}" if not np.isnan(stats["pbo"]) else "n/a"
        print(f"  [confirmation] seed {seed}: {status}  "
              f"mean_test_avg={stats['mean_test_avg_pnl']:+.4%}  "
              f"positive={stats.get('n_pos', 0)}/{stats['n_total']}  PBO={pbo_str}")

    n_pass = sum(r["passed"] for r in per_seed)
    frac_pass = n_pass / len(confirmation_seeds)
    per_seed_stable = frac_pass >= min_pass_frac

    # Pooled verdict: the same threshold's paths across ALL confirmation
    # seeds, evaluated together — this is the number that gets reported
    # as "the" CPCV summary (more paths -> more stable statistics than
    # any single confirmation seed alone).
    pooled_passed, pooled_stats = evaluate_gate(pooled_path_results)

    overall_passed = per_seed_stable and pooled_passed

    print(f"\n{'='*62}\n  GATE CONFIRMATION VERDICT\n{'='*62}")
    print(f"  Per-seed: {n_pass}/{len(confirmation_seeds)} confirmation seeds passed "
          f"({frac_pass:.0%}, required >= {min_pass_frac:.0%})")
    print(f"  Pooled ({pooled_stats['n_total']} reliable paths across all confirmation "
          f"seeds): {'PASS' if pooled_passed else 'FAIL'}  "
          f"mean_test_avg={pooled_stats['mean_test_avg_pnl']:+.4%}  "
          f"positive={pooled_stats.get('n_pos', 0)}/{pooled_stats['n_total']}  "
          f"PBO={pooled_stats['pbo']:.1%}" if not np.isnan(pooled_stats['pbo'])
          else f"  Pooled: {'PASS' if pooled_passed else 'FAIL'}  PBO=n/a")
    print(f"  → {'CONFIRMED' if overall_passed else 'NOT CONFIRMED'} "
          f"(requires both per-seed stability AND a passing pooled gate)")

    with open(os.path.join(OUT_DIR, "nested_gate_confirmation.json"), "w") as f:
        json.dump({
            "entry_threshold": entry_threshold,
            "confirmation_seeds": list(confirmation_seeds),
            "min_pass_frac": min_pass_frac,
            "n_pass": n_pass, "frac_pass": frac_pass,
            "per_seed_stable": per_seed_stable,
            "pooled_passed": bool(pooled_passed),
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
    }


def _pick_representative_confirmation_seed(per_seed: list) -> int:
    """
    Same MEDIAN-by-performance logic as _pick_representative_seed(),
    restricted to confirmation seeds only, so the deployed model's
    random_state is never a seed that was used for threshold selection.
    """
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
                       n_bagged_fits: int = None):
    """Train the deployable model on all rows except the most recent
    embargo-safe holdout block, then report calibration on that holdout.

    n_bagged_fits : forwarded to GBTAgent.fit() so the DEPLOYED model
        benefits from the same variance-averaging used during CPCV/
        gate confirmation (default: DEFAULT_N_BAGGED_FITS when None) —
        see gbt_agent.py's "BAGGED FITS" docstring section.

    entry_threshold: the value chosen by main()'s CPCV-only sweep
    (sweep_entry_thresholds()) — baked into the saved GBTAgent so live
    inference (get_action()) uses the same operating point this
    function evaluates on the holdout, and used here to score the
    holdout P/L that DEPLOYMENT GATING (below) decides on.

    random_state : the representative seed chosen by
        _pick_representative_seed() from run_stability_check() (this
        revision) — so the DEPLOYED model corresponds to a specific,
        reported, reproducible fit rather than an arbitrary default.

    DEPLOYMENT GATE: CPCV/stability passing is necessary but not
    sufficient — this function additionally requires the trained
    model's HOLDOUT avg/trade to clear HOLDOUT_AVG_TRADE_FLOOR with at
    least HOLDOUT_MIN_TRADES before it's allowed to overwrite
    gbt_agent_best.joblib. If it doesn't clear the bar, the model is
    still saved (nothing is lost) but under a "_FAILED_HOLDOUT"
    filename, and any previously deployed gbt_agent_best.joblib is left
    untouched.
    """
    n = len(states)
    lookback_ticks = compute_required_lookback_ticks()
    holdout_frac = 0.15
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

    # ── BOOTSTRAP CI on avg/trade ─────────────────────────────────────
    # A point estimate on n_trades this small can't be trusted at face
    # value — see bootstrap_ci()'s docstring. This resamples the SAME
    # trade_returns to show how much the point estimate could plausibly
    # swing, so a gate verdict can be read alongside its own uncertainty
    # rather than as a clean binary fact.
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

    # ── DEPLOYMENT GATE ──────────────────────────────────────────────
    # Gate decision itself is unchanged (point estimate vs floor,
    # n_trades vs min) — the CI is reported for interpretability, not
    # substituted into the pass/fail rule, so behaviour for existing
    # callers/checkpoints doesn't silently change. Read the CI alongside
    # the verdict below rather than treating the verdict as dispositive
    # when the interval straddles zero.
    deploy_ok = (n_trades >= HOLDOUT_MIN_TRADES) and (avg_pnl > HOLDOUT_AVG_TRADE_FLOOR)

    gate_status = {
        "entry_threshold": entry_threshold, "random_state": random_state,
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
        print(f"     Consider: shortening MAX_HOLDING further / narrowing "
              f"TP_MULT-SL_MULT more (more holdout trades), lowering "
              f"HOLDOUT_MIN_TRADES only if you also raise holdout_frac, or "
              f"gathering more data.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrated GBT meta-labeling pipeline")
    parser.add_argument("--force-final-training", action="store_true",
                        help="Proceed to final training even if the multi-seed "
                             "stability check and/or CPCV gate flags the result "
                             "as unstable.")
    parser.add_argument("--frozen-data", action="store_true",
                        help="Use the existing master CSV(s) as-is instead of "
                             "refetching/appending new raw data. Strongly "
                             "recommended whenever comparing two runs (e.g. "
                             "hyperparameter changes, the stability check's own "
                             "seeds) so the dataset itself isn't also changing.")
    parser.add_argument("--selection-seeds", type=int, nargs="+",
                        default=list(DEFAULT_SELECTION_SEEDS),
                        help="Seeds used ONLY to choose entry_threshold "
                             "(default: %(default)s). Must be disjoint from "
                             "--confirmation-seeds or the gate stops being a "
                             "genuine out-of-sample check — see module "
                             "docstring on nested selection/confirmation.")
    parser.add_argument("--confirmation-seeds", type=int, nargs="+",
                        default=list(DEFAULT_CONFIRMATION_SEEDS),
                        help="Seeds used ONLY to confirm the gate at the "
                             "already-chosen entry_threshold, and to pick the "
                             "final deployed model's random_state (default: "
                             "%(default)s). Never used for threshold selection.")
    parser.add_argument("--min-pass-frac", type=float, default=DEFAULT_MIN_PASS_FRAC,
                        help="Fraction of confirmation seeds that must "
                             "independently pass the CPCV gate for the "
                             "pipeline to proceed to final training "
                             "(default: %(default)s).")
    parser.add_argument("--no-multi-timeframe", action="store_true",
                        help="Disable 15m/1h context (state_dim ~50 instead of "
                             "~122) for this run, to compare against the full "
                             "multi-timeframe state using identical CPCV/"
                             "stability machinery. NOTE: a prior A/B run showed "
                             "this makes out-of-sample results worse, not "
                             "better — kept only for further comparison, not "
                             "recommended as the default.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the logistic-regression capacity-floor "
                             "check (run_logistic_baseline) that otherwise "
                             "runs automatically before threshold selection. "
                             "Skipping saves ~1 CPCV pass per confirmation "
                             "seed but loses the capacity-vs-label/regime "
                             "diagnostic signal — see run_logistic_baseline()'s "
                             "docstring.")
    parser.add_argument("--n-bagged-fits", type=int, default=DEFAULT_N_BAGGED_FITS,
                        help="Number of independently-seeded base-estimator "
                             "fits to average per (fold, seed) — see "
                             "gbt_agent.py's 'BAGGED FITS' docstring section "
                             "(default: %(default)s). Increases CPCV/"
                             "confirmation/final-training compute roughly "
                             "linearly; ignored (forced to 1) for the "
                             "logistic baseline, which is deterministic.")
    args = parser.parse_args()

    print("Building states + triple-barrier labels...")
    states, labels, prices, aligned_df = build_states_and_labels(
        frozen=args.frozen_data,
        enable_multi_timeframe=(False if args.no_multi_timeframe else None),
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
            f"{sorted(overlap)} — they must be disjoint, otherwise the gate "
            f"is partly confirming a threshold against data it was chosen "
            f"on. Pick non-overlapping seed lists."
        )

    # ── STEP -1 — logistic regression capacity-floor check (informational
    #    only; does not gate the pipeline) — see run_logistic_baseline()'s
    #    docstring for why this was added after the --no-multi-timeframe
    #    A/B run ruled out dimensionality as the dominant overfitting
    #    cause. Runs on the confirmation seeds so its PBO/mean-avg numbers
    #    are directly comparable to the GBT's own confirmation-pool
    #    numbers reported later in this run. ─────────────────────────────
    if not args.skip_baseline:
        run_logistic_baseline(states, labels, aligned_df, seeds=confirmation_seeds)
    else:
        print("\n  ⏭  Skipping logistic-regression baseline (--skip-baseline).")

    # ── STEP 0 — nested threshold selection (selection seeds only) ────────
    selection = select_threshold_nested(states, labels, aligned_df,
                                        selection_seeds=selection_seeds,
                                        n_bagged_fits=args.n_bagged_fits)
    entry_threshold = selection["chosen_threshold"]

    # ── STEP 1 — gate confirmation (confirmation seeds only, disjoint
    #    from selection; threshold is fixed here, not re-tuned) ───────────
    confirmation = confirm_gate_nested(
        states, labels, aligned_df, entry_threshold=entry_threshold,
        confirmation_seeds=confirmation_seeds, min_pass_frac=args.min_pass_frac,
        n_bagged_fits=args.n_bagged_fits,
    )

    # Save the pooled confirmation result in the same shape/location
    # gate_and_summarize() used to (cpcv_summary.json), so downstream
    # tooling (diagnostic_gbt.py etc.) that reads that file keeps working.
    gate_and_summarize(
        confirmation["pooled_path_results"], entry_threshold=entry_threshold,
        threshold_sweep=selection, seed=None, write=True,
    )

    if not confirmation["passed"] and not args.force_final_training:
        print("\n  Re-run with --force-final-training to override the "
              "nested selection/confirmation gate.")
        return

    chosen_seed = _pick_representative_confirmation_seed(confirmation["per_seed"])
    print(f"\n{'#'*62}\n  STEP 2 — FINAL DEPLOYABLE TRAINING + CALIBRATION  "
          f"(representative confirmation seed={chosen_seed})\n{'#'*62}")
    run_final_training(states, labels, aligned_df, entry_threshold=entry_threshold,
                       random_state=chosen_seed, n_bagged_fits=args.n_bagged_fits)


if __name__ == "__main__":
    main()

# python main_gbt.py --min-path-test-trades 15
# python main_gbt.py --disable-multi-timeframe
# python main_gbt.py --frozen-data --force-final-training
# python main_gbt.py --frozen-dat