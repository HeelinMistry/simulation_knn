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

# Base-model regularization (overfitting fix). Prior defaults (depth=6,
# l2=1.0, no leaf/feature cap, random calibration split) produced CPCV
# train P/L in the thousands of percent against a negative median test
# P/L — see GBTPolicy.fit()'s docstring for the full rationale. Exposed
# here (rather than left as GBTPolicy.fit()'s internal defaults) so it's
# visible and tunable in one place, and so run_cpcv()/run_final_training()
# always fit with the SAME config main_gbt.py reports on.
GBT_HYPERPARAMS = dict(
    max_iter=150,
    learning_rate=0.04,
    max_depth=4,
    max_leaf_nodes=15,
    min_samples_leaf=200,
    l2_regularization=5.0,
    validation_fraction=0.15,
    n_iter_no_change=15,
    max_features=0.7,   # per-split feature subsampling; sklearn>=1.2
)

# Stability gate (mirrors main_mcknn.py's philosophy — directional
# consistency + downside floor — plus a PBO check CPCV newly enables)
VAL_PNL_FLOOR = -0.40

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

    FIX: the prior version summed a return for every confident tick
    independently, with no concept of a position already being open.
    Because triple-barrier outcome windows span up to max_holding
    ticks and heavily overlap between adjacent ticks, any sustained
    stretch of conviction (e.g. a clean trend, which is exactly when a
    model is MOST confident) got counted as dozens of simultaneous
    "trades" stacked on top of each other — inflating P/L by roughly
    max_holding-fold and producing implausible five-digit percentage
    train returns. This does NOT re-run unified_executor.py's full
    position-management loop (commission-on-both-sides, stop-loss/
    max-hold forced exits, etc. are already baked into long_return/
    short_return by triple_barrier.py) — it only adds the missing
    "can't open two positions at once" constraint.

    Returns
    -------
    (total_pnl, n_trades) — n_trades lets callers sanity-check trade
    frequency (e.g. a suspiciously high count relative to len(mask)
    would flag a similar double-counting issue elsewhere).
    """
    idx = np.flatnonzero(mask)
    total_pnl  = 0.0
    n_trades   = 0
    next_free_tick = -1   # no trade open yet
    for row in idx:
        if row < next_free_tick:
            continue   # a previously opened trade is still "in the market"
        lsh = agent.actor._raw_probs(states[row])
        best = int(np.argmax(lsh))
        if best == 0 and lsh[0] > 0.5:
            total_pnl += float(labels["long_return"][row])
            next_free_tick = int(labels["long_touch"][row]) + 1
            n_trades += 1
        elif best == 1 and lsh[1] > 0.5:
            total_pnl += float(labels["short_return"][row])
            next_free_tick = int(labels["short_touch"][row]) + 1
            n_trades += 1
        # else: HOLD — stays flat, next_free_tick unchanged.
    return total_pnl, n_trades


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

        train_pnl, n_train_trades = simulate_pnl(states, labels, train_mask, agent)
        test_pnl,  n_test_trades  = simulate_pnl(states, labels, test_mask, agent)

        path_results.append({
            "path_id": path.path_id, "test_groups": list(path.test_groups),
            "train_pnl": train_pnl, "test_pnl": test_pnl,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "n_train_trades": n_train_trades, "n_test_trades": n_test_trades,
        })
        print(f"  path {path.path_id:>3}  test_groups={path.test_groups}  "
              f"train_pnl={train_pnl:+.4%} ({n_train_trades} trades)  "
              f"test_pnl={test_pnl:+.4%} ({n_test_trades} trades)")

    return path_results


def gate_and_summarize(path_results: list) -> bool:
    if not path_results:
        print("  ⛔ No usable CPCV paths — aborting.")
        return False

    test_pnls = np.array([r["test_pnl"] for r in path_results])
    n_pos = int((test_pnls > 0).sum())
    n_total = len(path_results)
    pbo_info = compute_pbo(path_results)

    print(f"\n{'='*62}\n  CPCV SUMMARY ({n_total} paths)\n{'='*62}")
    print(f"  Mean test P/L : {test_pnls.mean():+.4%}")
    print(f"  Std  test P/L : {test_pnls.std():.4%}")
    print(f"  Min  test P/L : {test_pnls.min():+.4%}")
    print(f"  Positive paths: {n_pos}/{n_total}")
    print(f"  PBO           : {pbo_info['pbo']:.1%}  "
          f"(fraction of in-sample-good paths that disappointed "
          f"out-of-sample — lower is better; >50% means in-sample "
          f"selection is worse than a coin flip)")
    print(f"  logit_lambda  : {pbo_info['logit_lambda']:+.3f}  "
          f"(more negative = more systematic overfitting)")

    with open(os.path.join(OUT_DIR, "cpcv_summary.json"), "w") as f:
        json.dump({
            "n_paths": n_total, "mean_test_pnl": float(test_pnls.mean()),
            "std_test_pnl": float(test_pnls.std()),
            "min_test_pnl": float(test_pnls.min()),
            "n_paths_positive": n_pos, **pbo_info,
            "path_results": path_results,
        }, f, indent=2)
    print(f"  ✓  CPCV summary saved → {OUT_DIR}/cpcv_summary.json")

    directionally_consistent = n_pos >= (n_total / 2)
    downside_breach = test_pnls.min() < VAL_PNL_FLOOR
    pbo_high = (not np.isnan(pbo_info["pbo"])) and pbo_info["pbo"] > 0.5

    unstable = (not directionally_consistent) or downside_breach or pbo_high
    if unstable:
        print(f"\n  ⛔ UNSTABLE — refusing to train final deployable model.")
        if not directionally_consistent:
            print(f"     ✗ only {n_pos}/{n_total} paths positive")
        if downside_breach:
            print(f"     ✗ worst path {test_pnls.min():+.4%} < floor {VAL_PNL_FLOOR:+.0%}")
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

    test_pnl, n_trades = simulate_pnl(states, labels, holdout_mask, agent)
    print(f"\n  Held-out simulated P/L (non-overlapping single-position, "
          f"see simulate_pnl docstring): {test_pnl:+.4%}  ({n_trades} trades)")


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