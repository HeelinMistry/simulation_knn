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
  3. Evaluate via Combinatorial Purged CV (cpcv.py): fit a fresh
     GBTAgent per CPCV path's train split, evaluate on its test split,
     collect the full distribution, and compute Probability of
     Backtest Overfitting (PBO).
  4. If the CPCV distribution passes a stability gate (directional
     consistency + downside floor + PBO ≤ 50%, mirroring main_mcknn.py's
     gate philosophy), fit the final deployable model on all data up to
     an embargo-safe cutoff, calibrate, and report reliability.

CHANGES IN THIS REVISION — NORMALIZED CPCV EVALUATION
───────────────────────────────────────────────────────
simulate_pnl() previously returned only a RAW SUM of per-trade returns
(total_pnl). That sum is trade-count-dependent: a fold that happens to
generate 600 trades will produce a far bigger number (in either
direction) than a fold generating 8 trades, purely from arithmetic,
with zero relationship to how GOOD the model's edge actually is. Two
concrete symptoms this caused:

  1. Train P/L in the hundreds-to-thousands of percent range was, in
     large part, hundreds of trades' worth of near-tautological
     "correct" predictions (the classifier scored against the exact
     labels it was fit on) additively summed with no compounding and
     no capital constraint — not proof of catastrophic overfitting by
     itself, though real overfitting is also present.
  2. VAL_PNL_FLOOR (a raw-sum floor) penalized high-trade-count folds
     far more harshly than low-trade-count folds of similar per-trade
     severity, making the gate's pass/fail decision largely a function
     of how many trades a fold happened to generate rather than the
     quality of the model's edge.

Fix: simulate_pnl() now also returns the list of individual trade
returns, from which we derive a MEAN PER-TRADE RETURN and a Sharpe-like
statistic (mean/std * sqrt(n), the same formula diagnostic_gbt.py's
write_summary() already uses for its real tick-by-tick backtest) — both
trade-count-independent. The CPCV gate (gate_and_summarize) and PBO
computation now operate on these normalized per-path statistics instead
of the raw sum. Raw totals are still logged for human-readable context,
but no longer drive the accept/reject decision.

GBT_HYPERPARAMS is also tightened further (lower max_depth/leaf_nodes,
higher min_samples_leaf/l2, lower max_features) — the normalized metric
fix corrects how we MEASURE the train/test gap, it doesn't shrink the
gap itself, and the gap is still large enough to warrant additional
regularization on top of the previous tightening pass (see
gbt_agent.py's GBTPolicy.fit() docstring for that history).

Run
────
    python main_gbt.py                        # gated pipeline
    python main_gbt.py --force-final-training  # override an unstable gate
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

# Triple-barrier config
TP_MULT      = 2.0
SL_MULT      = 1.0
MAX_HOLDING  = 32          # matches unified_executor.py's MAX_HOLD_TICKS
COMMISSION   = 0.00015     # matches unified_executor.py's COMMISSION

# CPCV config
N_GROUPS        = 8
N_TEST_GROUPS   = 2
MAX_PATHS       = 28
MIN_TRAIN_TICKS = 2000

# Base-model regularization (overfitting fix, round 2). Round 1 (see
# GBTPolicy.fit()'s docstring) brought train P/L down from >1000% but
# CPCV still showed train P/L in the hundreds-to-thousands of percent
# against near-zero/negative median test P/L — tightened further here:
# lower depth/leaf-node budget, higher min_samples_leaf and l2, and a
# stronger per-split feature subsample, all aimed at making individual
# trees (and therefore the ensemble) less able to carve out
# training-set-specific decision boundaries in a state space that can
# be up to 122-dim relative to ~10-12k fit rows per CPCV fold.
GBT_HYPERPARAMS = dict(
    max_iter=150,
    learning_rate=0.04,
    max_depth=3,               # was 4
    max_leaf_nodes=10,         # was 15
    min_samples_leaf=300,      # was 200
    l2_regularization=8.0,     # was 5.0
    validation_fraction=0.15,
    n_iter_no_change=15,
    max_features=0.5,          # was 0.7; sklearn>=1.2
)

# Stability gate (mirrors main_mcknn.py's philosophy — directional
# consistency + downside floor — plus a PBO check CPCV newly enables).
#
# NORMALIZED, not raw-sum: VAL_AVG_TRADE_FLOOR is a floor on the WORST
# path's MEAN PER-TRADE test return, so a fold's severity is judged
# per-bet rather than being amplified/muted by however many trades that
# particular fold happened to generate. -0.03 means: even in the worst
# CPCV path, the average trade shouldn't lose more than 3% net of
# commission — a materially different (and fairer) bar than the old
# "-40% of an unbounded, trade-count-scaled sum".
VAL_AVG_TRADE_FLOOR = -0.03

OUT_DIR = "outcomes/gbt"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# State + label construction
# ─────────────────────────────────────────────

def build_states_and_labels():
    """
    Build (states, labels, prices, aligned_df) for the full dataset,
    using identical feature construction to main_mcknn.py/pre_training.py
    so results are directly comparable.
    """
    df = update_master_data("4h")
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)

    ind = df[FEATURES].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)
    n = len(df)

    agg = StateAggregator(PACES, num_indicators=len(FEATURES))
    agg.warm_up_all(ind, WARMUP_IDX)

    extra_context_arr = None
    if ENABLE_MULTI_TIMEFRAME:
        try:
            timeframe_dfs = update_all_timeframes(CONTEXT_TIMEFRAMES)
            missing = [tf for tf in CONTEXT_TIMEFRAMES if tf not in timeframe_dfs]
            if missing:
                raise FileNotFoundError(f"missing timeframe(s): {missing}")
            timeframe_frames = {
                tf: tdf[["Open_time"] + FEATURES].dropna().reset_index(drop=True)
                for tf, tdf in timeframe_dfs.items()
            }
            extra_context_arr = build_multi_timeframe_context(
                df, timeframe_frames, FEATURES, context_paces=CONTEXT_PACES,
            )
            print(f"  Multi-timeframe context: {extra_context_arr.shape}")
        except FileNotFoundError as exc:
            print(f"  ⚠ Multi-timeframe context disabled — {exc}")
            extra_context_arr = None

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
                  agent: GBTAgent) -> tuple:
    """
    Score the CLASSIFIER's entry decisions against the triple-barrier
    label's own ground-truth realised return, enforcing ONE OPEN
    POSITION AT A TIME (matching unified_executor.py's single-slot
    `inventory` deque) — at each tick in `mask` that isn't still
    "inside" a previously opened trade's holding window, take the
    higher-probability directional class if its calibrated probability
    exceeds 0.5, credit that tick's own long_return/short_return, and
    then skip every tick up to and including that trade's touch tick
    before considering another entry.

    NORMALIZATION FIX (this revision): total_pnl (a raw sum of
    per-trade returns) is trade-count-dependent — a fold with 600
    trades produces a far larger number than a fold with 8 trades of
    similar per-trade quality, purely from arithmetic. That made the
    CPCV gate's pass/fail decision largely a function of how many
    trades a given fold happened to generate rather than genuine edge
    quality (see main_gbt.py module docstring). This function now also
    returns the individual trade_returns list and their mean
    (avg_pnl), which callers should use for any cross-fold comparison
    or gating decision; total_pnl/n_trades remain for human-readable
    logging only.

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
        if best == 0 and lsh[0] > 0.5:
            r = float(labels["long_return"][row])
            total_pnl += r
            trade_returns.append(r)
            next_free_tick = int(labels["long_touch"][row]) + 1
            n_trades += 1
        elif best == 1 and lsh[1] > 0.5:
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


def run_cpcv(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame) -> list:
    lookback_ticks = compute_required_lookback_ticks()
    paths = generate_cpcv_paths(
        aligned_df, n_groups=N_GROUPS, n_test_groups=N_TEST_GROUPS,
        lookback_ticks=lookback_ticks, min_train_ticks=MIN_TRAIN_TICKS,
        max_paths=MAX_PATHS,
    )
    print(f"  Generated {len(paths)} CPCV paths "
          f"(groups={N_GROUPS}, test_groups={N_TEST_GROUPS}, "
          f"lookback_ticks={lookback_ticks})")

    y = labels["best_action"]
    valid = labels["valid_mask"]

    path_results = []
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
                  purge_ticks=lookback_ticks, **GBT_HYPERPARAMS)

        train_pnl, n_train_trades, train_avg_pnl, train_returns = simulate_pnl(
            states, labels, train_mask, agent)
        test_pnl, n_test_trades, test_avg_pnl, test_returns = simulate_pnl(
            states, labels, test_mask, agent)
        test_sharpe = _sharpe_like(test_returns)

        path_results.append({
            "path_id": path.path_id, "test_groups": list(path.test_groups),
            # Raw sums — logging/context only, NOT trade-count-normalized.
            "train_pnl": train_pnl, "test_pnl": test_pnl,
            # Normalized — these drive gate_and_summarize()'s decision.
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
              f"[raw: train={train_pnl:+.2%} test={test_pnl:+.2%}]")

    return path_results


def gate_and_summarize(path_results: list) -> bool:
    """
    NORMALIZED gate (this revision): decisions are based on
    test_avg_pnl (mean per-trade test return) and test_sharpe, not the
    raw trade-count-scaled test_pnl sum — see module docstring for why.
    Raw sums are still reported for context.
    """
    if not path_results:
        print("  ⛔ No usable CPCV paths — aborting.")
        return False

    test_pnls      = np.array([r["test_pnl"] for r in path_results])       # raw, context only
    test_avg_pnls  = np.array([r["test_avg_pnl"] for r in path_results])   # normalized — gates on this
    sharpes        = np.array([r["test_sharpe"] for r in path_results])
    n_pos   = int((test_avg_pnls > 0).sum())
    n_total = len(path_results)

    # PBO fed on the normalized per-trade metric, so "in-sample-good"
    # vs "out-of-sample-good" ranking isn't itself confounded by
    # trade-count differences between paths (see compute_pbo()'s train/
    # test_pnl keys — reused here with normalized values).
    pbo_input = [{"train_pnl": r["train_avg_pnl"], "test_pnl": r["test_avg_pnl"]}
                 for r in path_results]
    pbo_info = compute_pbo(pbo_input)

    print(f"\n{'='*62}\n  CPCV SUMMARY ({n_total} paths)\n{'='*62}")
    print(f"  Mean test avg/trade : {test_avg_pnls.mean():+.4%}")
    print(f"  Std  test avg/trade : {test_avg_pnls.std():.4%}")
    print(f"  Min  test avg/trade : {test_avg_pnls.min():+.4%}")
    valid_sharpes = sharpes[~np.isnan(sharpes)]
    if len(valid_sharpes):
        print(f"  Mean test Sharpe    : {valid_sharpes.mean():+.3f}  "
              f"(n_paths_with_sharpe={len(valid_sharpes)}/{n_total})")
    print(f"  Positive paths      : {n_pos}/{n_total}")
    print(f"  PBO                 : {pbo_info['pbo']:.1%}  "
          f"(fraction of in-sample-good paths that disappointed "
          f"out-of-sample — lower is better; >50% means in-sample "
          f"selection is worse than a coin flip)")
    print(f"  logit_lambda        : {pbo_info['logit_lambda']:+.3f}  "
          f"(more negative = more systematic overfitting)")
    print(f"  [context, raw sums] mean={test_pnls.mean():+.2%}  "
          f"std={test_pnls.std():.2%}  min={test_pnls.min():+.2%}")

    with open(os.path.join(OUT_DIR, "cpcv_summary.json"), "w") as f:
        json.dump({
            "n_paths": n_total,
            "mean_test_avg_pnl": float(test_avg_pnls.mean()),
            "std_test_avg_pnl": float(test_avg_pnls.std()),
            "min_test_avg_pnl": float(test_avg_pnls.min()),
            "mean_test_pnl_raw": float(test_pnls.mean()),
            "std_test_pnl_raw": float(test_pnls.std()),
            "min_test_pnl_raw": float(test_pnls.min()),
            "n_paths_positive": n_pos, **pbo_info,
            "path_results": path_results,
        }, f, indent=2)
    print(f"  ✓  CPCV summary saved → {OUT_DIR}/cpcv_summary.json")

    directionally_consistent = n_pos >= (n_total / 2)
    downside_breach = test_avg_pnls.min() < VAL_AVG_TRADE_FLOOR
    pbo_high = (not np.isnan(pbo_info["pbo"])) and pbo_info["pbo"] > 0.5

    unstable = (not directionally_consistent) or downside_breach or pbo_high
    if unstable:
        print(f"\n  ⛔ UNSTABLE — refusing to train final deployable model.")
        if not directionally_consistent:
            print(f"     ✗ only {n_pos}/{n_total} paths positive (per-trade avg)")
        if downside_breach:
            print(f"     ✗ worst path avg/trade {test_avg_pnls.min():+.4%} "
                  f"< floor {VAL_AVG_TRADE_FLOOR:+.2%}")
        if pbo_high:
            print(f"     ✗ PBO={pbo_info['pbo']:.1%} > 50% — in-sample "
                  f"performance is not predictive of out-of-sample performance")
        return False

    print(f"\n  ✓ CPCV gate passed — proceeding to final training.")
    return True


# ─────────────────────────────────────────────
# Final deployable training
# ─────────────────────────────────────────────

def run_final_training(states: np.ndarray, labels: dict, aligned_df: pd.DataFrame):
    """Train the deployable model on all rows except the most recent
    embargo-safe holdout block, then report calibration on that holdout."""
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
          f"embargo: {holdout_start - embargo_start} rows")

    agent = GBTAgent(state_dim=states.shape[1], action_dim=ACTION_DIM)
    agent.fit(states[train_mask], labels["best_action"][train_mask],
              purge_ticks=lookback_ticks, **GBT_HYPERPARAMS)
    agent.save(os.path.join(OUT_DIR, "gbt_agent_best.joblib"))

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

    test_pnl, n_trades, avg_pnl, _returns = simulate_pnl(states, labels, holdout_mask, agent)
    print(f"\n  Held-out simulated P/L (non-overlapping single-position, "
          f"see simulate_pnl docstring): {test_pnl:+.4%} raw sum  "
          f"({n_trades} trades, avg/trade={avg_pnl:+.4%})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrated GBT meta-labeling pipeline")
    parser.add_argument("--force-final-training", action="store_true",
                        help="Proceed to final training even if the CPCV gate "
                             "flags the result as unstable.")
    args = parser.parse_args()

    print("Building states + triple-barrier labels...")
    states, labels, prices, aligned_df = build_states_and_labels()
    print(f"  {len(states):,} rows  |  state_dim={states.shape[1]}")
    n_long  = int((labels["best_action"] == 0).sum())
    n_short = int((labels["best_action"] == 1).sum())
    n_hold  = int((labels["best_action"] == 3).sum())
    print(f"  Label balance — LONG:{n_long:,}  SHORT:{n_short:,}  HOLD:{n_hold:,}")

    print(f"\n{'#'*62}\n  STEP 1 — COMBINATORIAL PURGED CROSS-VALIDATION\n{'#'*62}")
    path_results = run_cpcv(states, labels, aligned_df)
    ok = gate_and_summarize(path_results)

    if not ok and not args.force_final_training:
        print("\n  Re-run with --force-final-training to override.")
        return

    print(f"\n{'#'*62}\n  STEP 2 — FINAL DEPLOYABLE TRAINING + CALIBRATION\n{'#'*62}")
    run_final_training(states, labels, aligned_df)


if __name__ == "__main__":
    main()