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

CHANGES IN THIS REVISION — GATE-HARDENING PARITY WITH main_gbt.py
────────────────────────────────────────────────────────────────────
The first 7-symbol run (XRPUSDT BTCUSDT ETHUSDT SOLUSDT ADAUSDT
BNBUSDT LTCUSDT, --frozen-data) failed confirm_gate() on PBO (54.1%
> 40% allowed) and mean test Sharpe (0.098 < 0.15 floor), passed
directional consistency with zero margin (74/148 = exactly 50.0%),
and its threshold sweep could only certify entry_threshold=0.5 (0.65/
0.7 disqualified for too few total test trades). Reviewing that run
surfaced three gaps that were about pipeline COVERAGE, not about
whether the strategy itself has an edge:

  1. confirm_gate() never ran the "recent-regime" check main_gbt.py's
     confirm_gate_nested() has (GATE HARDENING change 2) — there was
     no way to tell whether the panel's edge (what little of it there
     is) is concentrated in stale history versus holding up in the
     panel's own most-recently-observed shared calendar window.
  2. There was no logistic-regression capacity-floor baseline or
     baseline-edge gate (main_gbt.py's run_logistic_baseline() +
     MIN_EDGE_OVER_BASELINE) — so a "passing" GBT here could not be
     distinguished from a GBT that just rediscovered the same
     marginal signal a near-linear model would also find.
  3. The excluded-path pattern (specific test_groups combinations
     recurring as "too few test trades" across MULTIPLE confirmation
     seeds) was invisible in the printed output, even though it's a
     structural data-coverage signal (random_state never changes
     which rows exist in a path's masks — see
     cross_sectional_panel.summarize_recurring_exclusions()'s
     docstring), not per-seed noise.

This revision:
  - Ports the recent-regime check into confirm_gate() via
    cross_sectional_panel.panel_extract_recent_regime_result(), using
    the SAME gbt.RECENT_REGIME_* constants and pass/fail semantics as
    main_gbt.confirm_gate_nested().
  - Adds a STEP -1 logistic-baseline run (skippable via
    --skip-baseline, mirroring main_gbt.py) and a STEP 1.5 baseline-
    edge gate in main(), using the SAME gbt.MIN_EDGE_OVER_BASELINE
    constant/comparison main_gbt.py's main() uses.
  - Surfaces cross_sectional_panel.summarize_recurring_exclusions()'s
    output after confirm_gate(), and writes it to
    panel_recurring_exclusions.json when non-empty.
  - Adds --pbo-max-allowed / --min-test-sharpe / --min-edge-over-baseline
    CLI overrides (same names/semantics as main_gbt.py) so this script
    can be tuned without editing main_gbt.py's module constants by hand.

overall_gate_passed now requires confirm_gate()'s per-seed stability
AND pooled gate AND recent-regime stability (all inside confirm_gate()
itself) AND the baseline-edge gate — exactly mirroring main_gbt.py's
main()'s `overall_gate_passed = confirmation["passed"] and
baseline_edge_ok`.

SCOPE STILL NOT PORTED (straightforward to add later, following the
same pattern): full per-seed nested-confirmation JSON bookkeeping
beyond what's written here.

Run
────
    python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT SOLUSDT ADAUSDT
    python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --frozen-data
    python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --force-final-training
    python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --skip-baseline
"""

import argparse
import json
import os

import numpy as np

import main_gbt as gbt
from cross_sectional_panel import (
    build_panel, run_panel_cpcv, panel_sweep_entry_thresholds,
    panel_evaluate_paths_at_threshold, run_panel_final_training,
    panel_extract_recent_regime_result, run_panel_logistic_baseline,
    summarize_recurring_exclusions,
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
    """
    Panel analogue of main_gbt.confirm_gate_nested(). Evaluates the
    ALREADY-CHOSEN entry_threshold (no further tuning) against CPCV
    paths built from confirmation_seeds, disjoint from selection_seeds.

    THIS REVISION — GATE HARDENING PARITY (see module docstring): in
    addition to the existing per-seed gate + pooled gate, this now ALSO
    extracts and scores the "recent regime" panel CPCV path (test_groups
    == gbt.RECENT_REGIME_GROUPS — the last N_TEST_GROUPS shared
    calendar groups, the closest CPCV analog to the real final panel
    holdout) from every confirmation seed's already-fitted paths, at
    zero extra fitting cost, via
    cross_sectional_panel.panel_extract_recent_regime_result(). A
    seed's recent-regime result "passes" if it has
    >= gbt.RECENT_REGIME_MIN_TEST_TRADES trades AND test_avg_pnl >
    gbt.RECENT_REGIME_AVG_PNL_FLOOR. Overall confirmation additionally
    requires gbt.RECENT_REGIME_MIN_PASS_FRAC of seeds' recent-regime
    results to pass — identical semantics to main_gbt.confirm_gate_nested().

    Also collects each seed's excluded-path test_groups (paths with
    < gbt.MIN_PATH_TEST_TRADES test trades that seed, as evaluate_gate()
    already reports) into `recurring_exclusions` via
    cross_sectional_panel.summarize_recurring_exclusions(), so main()
    can report which calendar windows are chronically unmeasurable
    across MULTIPLE seeds (a structural coverage gap) rather than that
    pattern being invisible in the printed per-seed output.

    Returns
    -------
    dict with:
      passed                 : bool, overall verdict — per-seed
                                stability AND pooled gate AND
                                recent-regime stability, all required
      frac_pass               : fraction of confirmation seeds that
                                 passed the whole-CPCV gate individually
      per_seed                : list of per-seed {seed, passed, stats}
      pooled_stats             : evaluate_gate() stats on ALL
                                 confirmation paths pooled together
      pooled_path_results      : the pooled, threshold-evaluated
                                 path_results (for gate_and_summarize())
      recent_regime_per_seed   : list of per-seed recent-regime results
                                 (or None where unavailable)
      recent_regime_frac_pass  : fraction of seeds with an available
                                 recent-regime result that passed it
      recent_regime_stable     : bool, recent_regime_frac_pass >=
                                 gbt.RECENT_REGIME_MIN_PASS_FRAC
      recurring_exclusions     : list from summarize_recurring_exclusions()
    """
    print(f"\n{'#'*62}\n  PANEL GATE CONFIRMATION  "
          f"(confirmation_seeds={list(confirmation_seeds)}, "
          f"entry_threshold={entry_threshold:.2f})\n{'#'*62}")
    per_seed = []
    pooled_path_results = []
    recent_regime_per_seed = []
    per_seed_excluded = {}
    for seed in confirmation_seeds:
        print(f"\n{'-'*62}\n  [confirmation] seed {seed}\n{'-'*62}")
        _, fitted_paths = run_panel_cpcv(panel, random_state=seed,
                                         n_bagged_fits=n_bagged_fits)
        evaluated = panel_evaluate_paths_at_threshold(panel, fitted_paths, entry_threshold)
        passed, stats = gbt.evaluate_gate(evaluated)
        pooled_path_results.extend(evaluated)
        per_seed_excluded[seed] = stats.get("excluded_paths", [])

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

        # ── Recent-regime check (this revision) — reuses fitted_paths,
        #    no additional model fitting. ────────────────────────────────
        recent = panel_extract_recent_regime_result(panel, fitted_paths, entry_threshold)
        if recent is not None:
            recent["seed"] = seed
            recent["reliable"] = recent["n_test_trades"] >= gbt.RECENT_REGIME_MIN_TEST_TRADES
            recent["recent_regime_passed"] = bool(
                recent["reliable"] and recent["test_avg_pnl"] > gbt.RECENT_REGIME_AVG_PNL_FLOOR
            )
            rr_status = ("PASS" if recent["recent_regime_passed"] else
                         "FAIL" if recent["reliable"] else "SKIP (too few trades)")
            print(f"  [confirmation] seed {seed} RECENT REGIME "
                  f"{gbt.RECENT_REGIME_GROUPS}: {rr_status}  "
                  f"test_avg={recent['test_avg_pnl']:+.4%}  "
                  f"n_trades={recent['n_test_trades']}")
        else:
            print(f"  [confirmation] seed {seed} RECENT REGIME "
                  f"{gbt.RECENT_REGIME_GROUPS}: not available (path not "
                  f"generated for this panel/seed)")
        recent_regime_per_seed.append(recent)

    n_pass = sum(r["passed"] for r in per_seed)
    frac_pass = n_pass / len(confirmation_seeds)
    per_seed_stable = frac_pass >= min_pass_frac

    pooled_passed, pooled_stats = gbt.evaluate_gate(pooled_path_results)

    # ── Recent-regime stability across seeds (this revision) ───────────
    rr_reliable = [r for r in recent_regime_per_seed if r is not None and r["reliable"]]
    if rr_reliable:
        rr_n_pass = sum(1 for r in rr_reliable if r["recent_regime_passed"])
        recent_regime_frac_pass = rr_n_pass / len(rr_reliable)
    else:
        rr_n_pass = 0
        recent_regime_frac_pass = float("nan")
    recent_regime_stable = (not np.isnan(recent_regime_frac_pass)
                            and recent_regime_frac_pass >= gbt.RECENT_REGIME_MIN_PASS_FRAC)

    overall_passed = per_seed_stable and pooled_passed and recent_regime_stable

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
    if rr_reliable:
        print(f"  Recent regime {gbt.RECENT_REGIME_GROUPS} (this revision): "
              f"{rr_n_pass}/{len(rr_reliable)} seeds passed "
              f"({recent_regime_frac_pass:.0%}, required >= "
              f"{gbt.RECENT_REGIME_MIN_PASS_FRAC:.0%})  "
              f"— {'STABLE' if recent_regime_stable else 'UNSTABLE'}")
    else:
        print(f"  Recent regime {gbt.RECENT_REGIME_GROUPS}: no seed produced a "
              f"reliable result (>= {gbt.RECENT_REGIME_MIN_TEST_TRADES} trades) — "
              f"treated as UNSTABLE (cannot confirm recency risk is absent).")
    print(f"  → {'CONFIRMED' if overall_passed else 'NOT CONFIRMED'} "
          f"(requires per-seed stability AND a passing pooled gate AND "
          f"recent-regime stability)")

    # ── Recurring-exclusion diagnostic (this revision, no gating effect) ──
    recurring = summarize_recurring_exclusions(per_seed_excluded, len(confirmation_seeds))
    if recurring:
        print(f"\n  ⚠ {len(recurring)} test_groups combination(s) excluded "
              f"(< {gbt.MIN_PATH_TEST_TRADES} test trades) in at least half of "
              f"confirmation seeds — likely a structural data-coverage gap "
              f"(e.g. align_panel()'s strict inner-join thinning that calendar "
              f"window for every symbol at once), not per-seed noise:")
        for r in recurring:
            print(f"      test_groups={r['test_groups']}  excluded in "
                  f"{r['n_seeds_excluded']}/{r['n_seeds_total']} seeds "
                  f"({r['frac_seeds_excluded']:.0%})")

    return {"passed": overall_passed, "frac_pass": frac_pass, "per_seed": per_seed,
            "pooled_stats": pooled_stats, "pooled_path_results": pooled_path_results,
            "recent_regime_per_seed": recent_regime_per_seed,
            "recent_regime_frac_pass": recent_regime_frac_pass,
            "recent_regime_stable": recent_regime_stable,
            "recurring_exclusions": recurring}


def main():
    # Declared before any use of these names as argparse defaults below,
    # since they're reassigned from CLI overrides further down — mirrors
    # main_gbt.py's main()'s own `global` declaration for the same
    # module-level constants (evaluate_gate()/panel_logistic_baseline
    # read them directly off the gbt module).
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
                             "confirmation (per-seed + pooled + recent-regime, "
                             "this revision) and/or the baseline-edge check "
                             "flags the result as unstable/insufficiently "
                             "better than the logistic baseline.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the logistic-regression capacity-floor "
                             "check (this revision). NOTE: skipping this also "
                             "skips the baseline-edge gate — the pipeline can "
                             "no longer verify the panel GBT beats a near-"
                             "linear model, so that check is silently treated "
                             "as passed. Only skip for quick iteration.")
    # ── GATE HARDENING CLI overrides (this revision — same names/
    #    semantics as main_gbt.py's main(), applied to the same
    #    module-level constants since evaluate_gate() lives there). ──────
    parser.add_argument("--pbo-max-allowed", type=float, default=gbt.PBO_MAX_ALLOWED,
                        help="Gate fails if PBO exceeds this fraction "
                             "(default: %(default)s).")
    parser.add_argument("--min-test-sharpe", type=float, default=gbt.MIN_TEST_SHARPE,
                        help="Gate fails if mean_test_sharpe (when computable) "
                             "is below this floor (default: %(default)s).")
    parser.add_argument("--min-edge-over-baseline", type=float,
                        default=gbt.MIN_EDGE_OVER_BASELINE,
                        help="Minimum required margin (in avg/trade, "
                             "fractional) by which the panel GBT's pooled "
                             "confirmation mean must beat the panel logistic "
                             "baseline's pooled mean (default: %(default)s).")
    args = parser.parse_args()

    # Module-level constants evaluate_gate()/run_panel_logistic_baseline()
    # read directly off the gbt module — overridden here (rather than
    # threaded as parameters through every call) so existing
    # callers/imports keep working unchanged when these flags aren't
    # passed, exactly mirroring main_gbt.py's own main().
    gbt.PBO_MAX_ALLOWED = args.pbo_max_allowed
    gbt.MIN_TEST_SHARPE = args.min_test_sharpe
    gbt.MIN_EDGE_OVER_BASELINE = args.min_edge_over_baseline

    if len(args.symbols) < 3:
        print(f"  ⚠ Only {len(args.symbols)} symbol(s) requested — cross-sectional "
              f"pooling is most useful with several (more effective samples, "
              f"more behavioral diversity). Proceeding anyway.")

    selection_seeds = tuple(args.selection_seeds)
    confirmation_seeds = tuple(args.confirmation_seeds)
    overlap = set(selection_seeds) & set(confirmation_seeds)
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

    # ── STEP -1 — logistic regression capacity-floor check (this
    #    revision). Reuses confirmation_seeds so it's directly
    #    comparable to the panel GBT's own confirmation-pool numbers,
    #    exactly as main_gbt.py's main() does. ─────────────────────────
    baseline_stats = None
    if not args.skip_baseline:
        baseline_stats = run_panel_logistic_baseline(
            panel, seeds=confirmation_seeds, n_bagged_fits=args.n_bagged_fits,
            out_dir=out_dir,
        )
    else:
        print("\n  ⏭  Skipping panel logistic-regression baseline (--skip-baseline). "
              "The baseline-edge gate will be treated as passed.")

    # ── STEP 0 — threshold selection (selection seeds only) ─────────────
    selection = select_threshold(panel, selection_seeds, n_bagged_fits=args.n_bagged_fits)
    entry_threshold = selection["chosen_threshold"]

    # ── STEP 1 — gate confirmation (confirmation seeds only). This
    #    revision: now also includes the recent-regime check and the
    #    recurring-exclusion diagnostic — see confirm_gate() docstring. ──
    confirmation = confirm_gate(panel, entry_threshold, confirmation_seeds,
                                args.min_pass_frac, n_bagged_fits=args.n_bagged_fits)

    gbt.gate_and_summarize(
        confirmation["pooled_path_results"], entry_threshold=entry_threshold,
        threshold_sweep=selection, seed=None, write=True, out_dir=out_dir,
    )

    if confirmation.get("recurring_exclusions"):
        with open(os.path.join(out_dir, "panel_recurring_exclusions.json"), "w") as f:
            json.dump(confirmation["recurring_exclusions"], f, indent=2,
                      default=gbt._json_default)
        print(f"\n  ✓  Recurring-exclusion diagnostic saved → "
              f"{out_dir}/panel_recurring_exclusions.json")

    # ── STEP 1.5 — baseline-edge gate (this revision). The panel GBT's
    #    pooled confirmation mean must beat the panel logistic
    #    baseline's pooled mean by MIN_EDGE_OVER_BASELINE — identical
    #    comparison to main_gbt.py's main(). ───────────────────────────
    baseline_edge_ok = True
    edge = None
    if baseline_stats is not None:
        gbt_mean = confirmation["pooled_stats"]["mean_test_avg_pnl"]
        baseline_mean = baseline_stats["mean_test_avg_pnl"]
        edge = gbt_mean - baseline_mean
        baseline_edge_ok = edge > gbt.MIN_EDGE_OVER_BASELINE
        print(f"\n{'='*62}\n  PANEL BASELINE-EDGE GATE\n{'='*62}")
        print(f"  Panel GBT pooled mean test avg/trade : {gbt_mean:+.4%}")
        print(f"  Panel logistic baseline pooled mean   : {baseline_mean:+.4%}")
        print(f"  Edge (GBT - baseline)                 : {edge:+.4%}  "
              f"(required > {gbt.MIN_EDGE_OVER_BASELINE:+.4%})")
        if baseline_edge_ok:
            print(f"  → PASSED — panel GBT shows a real margin over a near-"
                  f"minimal-capacity model, supporting a GBT-specific edge "
                  f"rather than just the shared marginal signal a linear "
                  f"model also finds.")
        else:
            print(f"  ⛔ FAILED — the panel GBT does not meaningfully beat a "
                  f"near-linear baseline. Per run_panel_logistic_baseline()'s "
                  f"own diagnostic logic, this is evidence the observed edge "
                  f"is a shared, marginal, possibly non-stationary label/"
                  f"regime signal rather than something the GBT's extra "
                  f"capacity is contributing — further GBT tuning is unlikely "
                  f"to close this gap by itself.")

    overall_gate_passed = confirmation["passed"] and baseline_edge_ok

    if not overall_gate_passed and not args.force_final_training:
        if not confirmation["passed"]:
            print("\n  Panel gate confirmation did not pass "
                  "(per-seed / pooled / recent-regime).")
        if not baseline_edge_ok:
            print("\n  Panel baseline-edge gate did not pass.")
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
# python main_cross_sectional_gbt.py --symbols XRPUSDT BTCUSDT ETHUSDT --skip-baseline