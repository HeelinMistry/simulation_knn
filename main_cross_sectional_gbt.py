"""
main_cross_sectional_gbt.py
─────────────────────────────
Training entry point for the CROSS-SECTIONAL variant of the GBT
meta-labeling pipeline — pools multiple symbols' states/labels into
one classifier, with purge/embargo/CPCV keyed on a SHARED calendar
timestamp axis (see cross_sectional_panel.py's module docstring for
the full "why a separate layer, not a change to master_processed.csv"
reasoning).

This module deliberately reuses main_gbt.py's GBTAgent, evaluate_gate,
gate_and_summarize, bootstrap_ci, and hyperparameter/gate constants
UNCHANGED — only the state/label ASSEMBLY (panel construction) and the
CPCV fold generation are new. Anything already validated about
GBTAgent's behavior (bagging, purged early stopping, calibration)
applies identically here.

SCOPE OF THIS FIRST VERSION
──────────────────────────────
Implements: panel construction, threshold selection (pooled sweep over
--selection-seeds), gate confirmation (--confirmation-seeds, reusing
main_gbt.evaluate_gate()/gate_and_summarize() verbatim), and final
holdout-gated deployment training.

NOT YET ported from main_gbt.py (straightforward to add later — they
only depend on `fitted_paths` + `evaluate_gate()`, both reused as-is
by this module): the logistic-regression baseline-edge gate, the
recent-regime check, and full per-seed nested-confirmation JSON
bookkeeping. To add them, follow main_gbt.py's
run_logistic_baseline()/confirm_gate_nested() pattern exactly, but
call run_panel_cpcv()/panel_evaluate_paths_at_threshold() from
cross_sectional_panel.py instead of gbt.run_cpcv()/
gbt.evaluate_paths_at_threshold() — nothing else differs.

Run
────
    python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT SOLUSDT ADAUSDT
    python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --frozen-data
    python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --force-final-training
"""

import argparse
import os

import numpy as np

import main_gbt as gbt
from cross_sectional_panel import (
    build_panel, run_panel_cpcv, panel_sweep_entry_thresholds,
    panel_evaluate_paths_at_threshold, run_panel_final_training,
)

DEFAULT_SYMBOLS = ["XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT",
                   "BNBUSDT", "LTCUSDT"]


def select_threshold(panel, selection_seeds, n_bagged_fits=None):
    """Panel analogue of main_gbt.select_threshold_nested(): fit CPCV
    across every selection seed, POOL the fitted (path, agent) pairs,
    sweep entry_threshold ONCE on the pooled set."""
    print(f"\n{'#'*62}\n  PANEL THRESHOLD SELECTION  "
          f"(selection_seeds={list(selection_seeds)})\n{'#'*62}")
    pooled_fitted_paths = []
    for seed in selection_seeds:
        print(f"\n{'-'*62}\n  [selection] seed {seed}\n{'-'*62}")
        _, fitted_paths = run_panel_cpcv(panel, random_state=seed,
                                         n_bagged_fits=n_bagged_fits)
        pooled_fitted_paths.extend(fitted_paths)

    print(f"\n  Pooled {len(pooled_fitted_paths)} fitted panel CPCV paths "
          f"across {len(selection_seeds)} selection seed(s) for threshold sweep.")
    sweep = panel_sweep_entry_thresholds(panel, pooled_fitted_paths)
    return sweep


def confirm_gate(panel, entry_threshold, confirmation_seeds, min_pass_frac,
                 n_bagged_fits=None):
    """Panel analogue of main_gbt.confirm_gate_nested() (without the
    recent-regime extension — see module docstring). Evaluates the
    ALREADY-CHOSEN entry_threshold (no further tuning) against CPCV
    paths built from confirmation_seeds, disjoint from selection_seeds."""
    print(f"\n{'#'*62}\n  PANEL GATE CONFIRMATION  "
          f"(confirmation_seeds={list(confirmation_seeds)}, "
          f"entry_threshold={entry_threshold:.2f})\n{'#'*62}")
    per_seed = []
    pooled_path_results = []
    for seed in confirmation_seeds:
        print(f"\n{'-'*62}\n  [confirmation] seed {seed}\n{'-'*62}")
        _, fitted_paths = run_panel_cpcv(panel, random_state=seed,
                                         n_bagged_fits=n_bagged_fits)
        evaluated = panel_evaluate_paths_at_threshold(panel, fitted_paths, entry_threshold)
        passed, stats = gbt.evaluate_gate(evaluated)
        pooled_path_results.extend(evaluated)

        per_seed.append({
            "seed": seed, "passed": bool(passed),
            "n_total": stats["n_total"], "n_pos": stats.get("n_pos", 0),
            "mean_test_avg_pnl": stats["mean_test_avg_pnl"],
            "mean_test_sharpe": stats.get("mean_test_sharpe", float("nan")),
            "pbo": stats["pbo"],
        })
        pbo_str = f"{stats['pbo']:.1%}" if not np.isnan(stats["pbo"]) else "n/a"
        print(f"  [confirmation] seed {seed}: {'PASS' if passed else 'FAIL'}  "
              f"mean_test_avg={stats['mean_test_avg_pnl']:+.4%}  "
              f"positive={stats.get('n_pos', 0)}/{stats['n_total']}  PBO={pbo_str}")

    n_pass = sum(r["passed"] for r in per_seed)
    frac_pass = n_pass / len(confirmation_seeds)
    per_seed_stable = frac_pass >= min_pass_frac

    pooled_passed, pooled_stats = gbt.evaluate_gate(pooled_path_results)
    overall_passed = per_seed_stable and pooled_passed

    print(f"\n{'='*62}\n  PANEL GATE CONFIRMATION VERDICT\n{'='*62}")
    print(f"  Per-seed: {n_pass}/{len(confirmation_seeds)} confirmation seeds "
          f"passed ({frac_pass:.0%}, required >= {min_pass_frac:.0%})")
    pooled_pbo_str = (f"{pooled_stats['pbo']:.1%}"
                      if not np.isnan(pooled_stats['pbo']) else "n/a")
    print(f"  Pooled ({pooled_stats['n_total']} reliable paths): "
          f"{'PASS' if pooled_passed else 'FAIL'}  "
          f"mean_test_avg={pooled_stats['mean_test_avg_pnl']:+.4%}  "
          f"positive={pooled_stats.get('n_pos', 0)}/{pooled_stats['n_total']}  "
          f"PBO={pooled_pbo_str}")
    print(f"  → {'CONFIRMED' if overall_passed else 'NOT CONFIRMED'} "
          f"(requires per-seed stability AND a passing pooled gate)")

    return {"passed": overall_passed, "frac_pass": frac_pass, "per_seed": per_seed,
            "pooled_stats": pooled_stats, "pooled_path_results": pooled_path_results}


def main():
    parser = argparse.ArgumentParser(
        description="Cross-sectional (multi-symbol) GBT meta-labeling pipeline")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help=f"Symbols to pool into one cross-sectional panel "
                             f"(default: {DEFAULT_SYMBOLS}). Every symbol needs "
                             f"data under data/raw/<symbol>/<timeframe>/ — see "
                             f"data_manager.py.")
    parser.add_argument("--frozen-data", action="store_true",
                        help="Use existing master CSV(s) as-is for every symbol "
                             "instead of refetching/appending new raw data.")
    parser.add_argument("--no-multi-timeframe", action="store_true",
                        help="Disable 15m/1h context for every symbol in this run.")
    parser.add_argument("--selection-seeds", type=int, nargs="+",
                        default=list(gbt.DEFAULT_SELECTION_SEEDS),
                        help="Seeds used ONLY to choose entry_threshold.")
    parser.add_argument("--confirmation-seeds", type=int, nargs="+",
                        default=list(gbt.DEFAULT_CONFIRMATION_SEEDS),
                        help="Seeds used ONLY to confirm the gate at the "
                             "already-chosen entry_threshold. Must be disjoint "
                             "from --selection-seeds.")
    parser.add_argument("--min-pass-frac", type=float, default=gbt.DEFAULT_MIN_PASS_FRAC)
    parser.add_argument("--n-bagged-fits", type=int, default=gbt.DEFAULT_N_BAGGED_FITS)
    parser.add_argument("--holdout-frac", type=float, default=gbt.DEFAULT_HOLDOUT_FRAC)
    parser.add_argument("--force-final-training", action="store_true",
                        help="Proceed to final training even if the panel gate "
                             "confirmation did not pass.")
    args = parser.parse_args()

    if len(args.symbols) < 3:
        print(f"  ⚠ Only {len(args.symbols)} symbol(s) requested — cross-sectional "
              f"pooling is most useful with several (more effective samples, "
              f"more behavioral diversity). Proceeding anyway.")

    overlap = set(args.selection_seeds) & set(args.confirmation_seeds)
    if overlap:
        raise ValueError(
            f"--selection-seeds and --confirmation-seeds share seed(s) "
            f"{sorted(overlap)} — they must be disjoint."
        )

    out_dir = os.path.join(gbt.OUT_DIR, "cross_sectional", "_".join(sorted(args.symbols)))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'='*62}\n  SYMBOLS: {args.symbols}\n  OUTPUT DIR: {out_dir}\n{'='*62}")

    panel = build_panel(
        args.symbols, frozen=args.frozen_data,
        enable_multi_timeframe=(False if args.no_multi_timeframe else None),
    )
    n_long = int((panel.best_action[panel.valid_mask] == 0).sum())
    n_short = int((panel.best_action[panel.valid_mask] == 1).sum())
    n_hold = int((panel.best_action[panel.valid_mask] == 3).sum())
    print(f"  Panel label balance — LONG:{n_long:,}  SHORT:{n_short:,}  HOLD:{n_hold:,}")

    # ── STEP 0 — threshold selection (selection seeds only) ─────────────
    selection = select_threshold(panel, tuple(args.selection_seeds),
                                 n_bagged_fits=args.n_bagged_fits)
    entry_threshold = selection["chosen_threshold"]

    # ── STEP 1 — gate confirmation (confirmation seeds only) ────────────
    confirmation = confirm_gate(panel, entry_threshold, tuple(args.confirmation_seeds),
                                args.min_pass_frac, n_bagged_fits=args.n_bagged_fits)

    gbt.gate_and_summarize(
        confirmation["pooled_path_results"], entry_threshold=entry_threshold,
        threshold_sweep=selection, seed=None, write=True, out_dir=out_dir,
    )

    if not confirmation["passed"] and not args.force_final_training:
        print("\n  Panel gate confirmation did not pass.")
        print("  Re-run with --force-final-training to override.")
        return

    passed_seeds = [r["seed"] for r in confirmation["per_seed"] if r["passed"]]
    pool = passed_seeds if passed_seeds else list(args.confirmation_seeds)
    chosen_seed = sorted(pool)[len(pool) // 2]   # median, same spirit as
                                                  # main_gbt.py's representative-
                                                  # seed selection

    print(f"\n{'#'*62}\n  STEP 2 — FINAL PANEL TRAINING + CALIBRATION  "
          f"(symbols={args.symbols}, representative seed={chosen_seed})\n{'#'*62}")
    run_panel_final_training(panel, entry_threshold=entry_threshold,
                             random_state=chosen_seed, n_bagged_fits=args.n_bagged_fits,
                             holdout_frac=args.holdout_frac, out_dir=out_dir)


if __name__ == "__main__":
    main()

# python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT SOLUSDT ADAUSDT
# python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --frozen-data
# python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --force-final-training