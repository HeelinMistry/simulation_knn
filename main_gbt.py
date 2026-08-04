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

Run
────
    python main_gbt.py                                   # gated pipeline, live data
    python main_gbt.py --frozen-data                      # reproducible comparisons
    python main_gbt.py --frozen-data --seeds 0 1 2 3 4    # more thorough stability check
    python main_gbt.py --frozen-data --no-multi-timeframe # 4h-only baseline
    python main_gbt.py --force-final-training              # override an unstable gate
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

# Base-model regularization (overfitting fix, round 3 — see module
# docstring). Rounds 1-2 brought train P/L down from >1000% and then
# from the hundreds-of-percent range, but train avg/trade was still
# running ~2-5x test avg/trade even after round 2. Tightened further:
# higher min_samples_leaf/l2, lower max_features, aimed at making
# individual trees (and therefore the ensemble) less able to carve out
# training-set-specific decision boundaries in a state space that can
# be up to 122-dim relative to ~10-12k fit rows per CPCV fold.
GBT_HYPERPARAMS = dict(
    max_iter=150,
    learning_rate=0.04,
    max_depth=3,
    max_leaf_nodes=8,
    min_samples_leaf=650,      # was 400 -> 650
    l2_regularization=14.0,    # was 10.0 -> 14.0
    validation_fraction=0.15,
    n_iter_no_change=15,
    max_features=0.35,         # was 0.45 -> 0.35; sklearn>=1.2
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


def run_cpcv(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
             random_state: int = 0) -> tuple:
    """
    Returns (path_results, fitted_paths, valid).

    random_state : forwarded to every fold's GBTAgent.fit() (which
        passes it through to sklearn's HistGradientBoostingClassifier).
        Exposed as a parameter (this revision) so run_stability_check()
        can re-run the entire CPCV pass under several independent seeds
        without touching anything else — see module docstring.

    fitted_paths: list of (CPCVPath, GBTAgent) for every path that was
    actually fit — kept so sweep_entry_thresholds()/
    evaluate_paths_at_threshold() can re-score the SAME fitted models
    at different entry-probability thresholds without the cost (and
    the subtle risk of a different random_state/early-stopping path)
    of refitting per threshold.
    """
    lookback_ticks = compute_required_lookback_ticks()
    paths = generate_cpcv_paths(
        aligned_df, n_groups=N_GROUPS, n_test_groups=N_TEST_GROUPS,
        lookback_ticks=lookback_ticks, min_train_ticks=MIN_TRAIN_TICKS,
        max_paths=MAX_PATHS,
    )
    print(f"  Generated {len(paths)} CPCV paths "
          f"(groups={N_GROUPS}, test_groups={N_TEST_GROUPS}, "
          f"lookback_ticks={lookback_ticks}, random_state={random_state})")

    y = labels["best_action"]
    valid = labels["valid_mask"]

    path_results = []
    fitted_paths = []
    for path in paths:
        train_mask = path.train_mask & valid
        test_mask  = path.test_mask & valid
        if train_mask.sum() < MIN_TRAIN_TICKS or test_mask.sum() < 100:
            continue

        agent = GBTAgent(state_dim=states.shape[1], action_dim=ACTION_DIM)
        # purge_ticks: the calibration split GBTPolicy.fit() carves out
        # of THIS fold's train rows is chronological (last 20% by
        # default) — pass the same lookback_ticks used for the outer
        # CPCV purge/embargo so that internal fit/calibration split gets
        # an equivalent purge buffer instead of a raw random split that
        # could leak across overlapping rolling-window lookback.
        agent.fit(states[train_mask], y[train_mask],
                  purge_ticks=lookback_ticks, random_state=random_state,
                  **GBT_HYPERPARAMS)
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
              f"[@0.50 baseline, seed={random_state}]")

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

    directionally_consistent = n_pos >= (n_total / 2)
    downside_breach = test_avg_pnls.min() < VAL_AVG_TRADE_FLOOR
    pbo_high = (not np.isnan(pbo_info["pbo"])) and pbo_info["pbo"] > 0.5
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
            }, f, indent=2)
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
# Multi-seed stability check (this revision)
# ─────────────────────────────────────────────

def run_stability_check(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                        seeds: tuple = DEFAULT_STABILITY_SEEDS,
                        min_pass_frac: float = DEFAULT_MIN_PASS_FRAC) -> tuple:
    """
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
        }, f, indent=2)
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
# Final deployable training
# ─────────────────────────────────────────────

def run_final_training(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame,
                       entry_threshold: float = 0.5, random_state: int = 0):
    """Train the deployable model on all rows except the most recent
    embargo-safe holdout block, then report calibration on that holdout.

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

    print(f"\n  Final train: {train_mask.sum():,} rows  |  "
          f"Holdout: {holdout_mask.sum():,} rows  |  "
          f"embargo: {holdout_start - embargo_start} rows  |  "
          f"entry_threshold: {entry_threshold:.2f}  |  random_state: {random_state}")

    agent = GBTAgent(state_dim=states.shape[1], action_dim=ACTION_DIM,
                     entry_threshold=entry_threshold)
    agent.fit(states[train_mask], labels["best_action"][train_mask],
              purge_ticks=lookback_ticks, random_state=random_state,
              **GBT_HYPERPARAMS)

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
        json.dump(report, f, indent=2)
    print(f"  ✓  Calibration report saved → {OUT_DIR}/calibration_report.json")

    test_pnl, n_trades, avg_pnl, _returns = simulate_pnl(
        states, labels, holdout_mask, agent, prob_threshold=entry_threshold)
    print(f"\n  Held-out simulated P/L (non-overlapping single-position, "
          f"see simulate_pnl docstring): {test_pnl:+.4%} raw sum  "
          f"({n_trades} trades, avg/trade={avg_pnl:+.4%})  "
          f"@ entry_threshold={entry_threshold:.2f}")

    # ── DEPLOYMENT GATE ──────────────────────────────────────────────
    deploy_ok = (n_trades >= HOLDOUT_MIN_TRADES) and (avg_pnl > HOLDOUT_AVG_TRADE_FLOOR)

    gate_status = {
        "entry_threshold": entry_threshold, "random_state": random_state,
        "holdout_avg_pnl": avg_pnl, "holdout_total_pnl_raw": test_pnl,
        "holdout_n_trades": n_trades,
        "holdout_avg_trade_floor": HOLDOUT_AVG_TRADE_FLOOR,
        "holdout_min_trades": HOLDOUT_MIN_TRADES,
        "deploy_gate_passed": bool(deploy_ok),
    }
    with open(os.path.join(OUT_DIR, "deployment_gate.json"), "w") as f:
        json.dump(gate_status, f, indent=2)

    if deploy_ok:
        save_path = os.path.join(OUT_DIR, "gbt_agent_best.joblib")
        agent.save(save_path)
        print(f"\n  ✓ HOLDOUT GATE PASSED (avg/trade={avg_pnl:+.4%} > "
              f"floor={HOLDOUT_AVG_TRADE_FLOOR:+.2%}, n_trades={n_trades}) "
              f"— deployed → {save_path}")
    else:
        save_path = os.path.join(OUT_DIR, "gbt_agent_candidate_FAILED_HOLDOUT.joblib")
        agent.save(save_path)
        print(f"\n  ⛔ HOLDOUT GATE FAILED — avg/trade={avg_pnl:+.4%} "
              f"(floor={HOLDOUT_AVG_TRADE_FLOOR:+.2%}, n_trades={n_trades} "
              f"vs min={HOLDOUT_MIN_TRADES}).")
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
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(DEFAULT_STABILITY_SEEDS),
                        help="Random seeds for the multi-seed CPCV stability "
                             "check (default: %(default)s). More seeds = more "
                             "confidence but proportionally more compute (each "
                             "seed refits every CPCV path).")
    parser.add_argument("--min-pass-frac", type=float, default=DEFAULT_MIN_PASS_FRAC,
                        help="Fraction of seeds that must independently pass the "
                             "CPCV gate for the pipeline to proceed to final "
                             "training (default: %(default)s).")
    parser.add_argument("--no-multi-timeframe", action="store_true",
                        help="Disable 15m/1h context (state_dim ~50 instead of "
                             "~122) for this run, to compare against the full "
                             "multi-timeframe state using identical CPCV/"
                             "stability machinery.")
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

    stable, seed_results = run_stability_check(
        states, labels, aligned_df,
        seeds=tuple(args.seeds), min_pass_frac=args.min_pass_frac,
    )

    if not stable and not args.force_final_training:
        print("\n  Re-run with --force-final-training to override the "
              "multi-seed stability check.")
        return

    chosen_seed = _pick_representative_seed(seed_results)
    print(f"\n{'#'*62}\n  DEPLOYED CPCV SUMMARY — representative seed={chosen_seed}"
          f"\n{'#'*62}")

    path_results, fitted_paths, valid = run_cpcv(states, labels, aligned_df,
                                                  random_state=chosen_seed)
    sweep = sweep_entry_thresholds(states, labels, fitted_paths, valid)
    entry_threshold = sweep["chosen_threshold"]

    # Re-evaluate every CPCV path at the chosen threshold so the gate
    # decision reflects the exact operating point that will be deployed.
    path_results = evaluate_paths_at_threshold(states, labels, fitted_paths,
                                               valid, entry_threshold)
    ok = gate_and_summarize(path_results, entry_threshold=entry_threshold,
                            threshold_sweep=sweep, seed=chosen_seed)

    if not ok and not args.force_final_training:
        print("\n  Re-run with --force-final-training to override.")
        return

    print(f"\n{'#'*62}\n  STEP 2 — FINAL DEPLOYABLE TRAINING + CALIBRATION\n{'#'*62}")
    run_final_training(states, labels, aligned_df, entry_threshold=entry_threshold,
                       random_state=chosen_seed)


if __name__ == "__main__":
    main()

# python main_gbt.py --min-path-test-trades 15
# python main_gbt.py --disable-multi-timeframe   # comparison baseline