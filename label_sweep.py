"""
label_sweep.py
─────────────────
Screens triple-barrier configs (tp_mult, sl_mult, max_holding) for one
specific, narrow purpose: finding a config whose RECENT-REGIME CPCV
check (test_groups == the last N_TEST_GROUPS groups — see main_gbt.py's
RECENT_REGIME_GROUPS) actually produces enough trades to be trustworthy,
instead of the "SKIP (too few trades)" / n_test_trades=0 outcome most
confirmation seeds hit under the current TP_MULT=1.5/SL_MULT=0.8/
MAX_HOLDING=20 defaults (see nested_gate_confirmation.json: 4 of 6
seeds got 0 recent-regime test trades; the two that did had 6 and 20).

This script does NOT try to make the gate pass. It tries to make the
recent-regime check MEASURABLE — a config that resolves more trades in
the same calendar window is a strictly more informative test than one
that mostly times out at the vertical (max_holding) barrier before
resolving. Whether the resulting numbers are good or bad is then judged
by main_gbt.py's own unmodified gate, not by anything in here.

Two phases
────────────
  PHASE 1 (cheap, label-only): build the state matrix's aligned prices/
      ATR ONCE via main_gbt.load_prices_for_labeling() (skips the
      15m/1h fetch and the per-tick StateAggregator loop entirely —
      labels don't need either), then for every candidate
      (tp_mult, sl_mult, max_holding) combo, relabel with
      main_gbt.labels_from_prices() and count how many candidate
      favorable-label "signals" (best_action != HOLD, among valid rows)
      fall inside the recent-regime row range. This is a fast upper-
      bound screen — actual trade counts under simulate_pnl()'s
      single-open-position constraint will be <= these candidate counts
      — used only to shortlist configs worth spending real model-fitting
      compute on.

  PHASE 2 (real CPCV, top-K candidates only): build the actual state
      matrix ONCE via main_gbt.build_states() (states don't depend on
      the barrier config, so this is not repeated per candidate), then
      for each of the top-K phase-1 candidates, relabel and run
      main_gbt.run_cpcv() at a small number of seeds with a reduced
      n_bagged_fits (for speed — this is a screen, not the final
      confirmation run). Reports each candidate's recent-regime trade
      count/reliability per seed plus the pooled evaluate_gate() stats,
      exactly as main_gbt.py's own confirm_gate_nested() would compute
      them.

This script deliberately reuses main_gbt.py's own functions
(build_states, labels_from_prices, run_cpcv, evaluate_gate,
_extract_recent_regime_result) rather than reimplementing any of that
logic, so a promising config found here is guaranteed to behave
identically when re-run through the real pipeline
(`python main_gbt.py --tp-mult ... --sl-mult ... --max-holding ...`)
for the actual nested selection/confirmation/final-training gate.

Usage
──────
    # quick default sweep, phase 1 only (seconds, no model fitting)
    python label_sweep.py --frozen-data --phase2-top-k 0

    # full screen: phase 1 + phase 2 CPCV check on the top 5 candidates
    python label_sweep.py --frozen-data

    # custom grid
    python label_sweep.py --frozen-data \\
        --tp-mults 1.0 1.3 1.5 1.8 2.2 \\
        --sl-mults 0.6 0.8 1.0 \\
        --max-holdings 10 14 20 28

    # once you've picked a winner from phase 2, confirm it for real:
    python main_gbt.py --frozen-data --tp-mult 1.0 --sl-mult 0.6 --max-holding 12
"""

import argparse
import itertools
import json
import os

import numpy as np

import main_gbt as gbt

DEFAULT_TP_MULTS      = (1.0, 1.3, 1.5, 1.8, 2.2)
DEFAULT_SL_MULTS      = (0.6, 0.8, 1.0)
DEFAULT_MAX_HOLDINGS  = (10, 14, 20, 28)

DEFAULT_PHASE2_TOP_K       = 5
DEFAULT_PHASE2_SEEDS       = (2, 3, 4)   # subset of DEFAULT_CONFIRMATION_SEEDS
DEFAULT_PHASE2_BAGGED_FITS = 2           # reduced from DEFAULT_N_BAGGED_FITS=5 for speed

OUT_PHASE1_PATH = os.path.join(gbt.OUT_DIR, "label_sweep_phase1.json")
OUT_PHASE2_PATH = os.path.join(gbt.OUT_DIR, "label_sweep_phase2.json")


def _compute_group_boundaries(n: int, n_groups: int) -> list:
    """Identical formula to cpcv.py's private _make_groups() — kept as
    a local, dependency-free copy so this script only relies on
    main_gbt.py's public surface."""
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(edges[g], edges[g + 1]) for g in range(n_groups)]


# ─────────────────────────────────────────────
# Phase 1 — cheap, label-only screen
# ─────────────────────────────────────────────

def run_phase1(prices_aligned: np.ndarray, atr_aligned: np.ndarray,
               tp_mults, sl_mults, max_holdings) -> list:
    n = len(prices_aligned)
    groups = _compute_group_boundaries(n, gbt.N_GROUPS)
    recent_lo = groups[gbt.N_GROUPS - gbt.N_TEST_GROUPS][0]
    recent_hi = groups[-1][1]   # exclusive

    print(f"\n{'='*62}\n  PHASE 1 — label-only screen  (n={n:,} rows, "
          f"recent-regime rows [{recent_lo}:{recent_hi}] = "
          f"{recent_hi - recent_lo:,} rows)\n{'='*62}")

    results = []
    combos = list(itertools.product(tp_mults, sl_mults, max_holdings))
    for tp, sl, mh in combos:
        labels = gbt.labels_from_prices(prices_aligned, atr_aligned,
                                        tp_mult=tp, sl_mult=sl, max_holding=mh)
        valid = labels["valid_mask"]
        best_action = labels["best_action"]

        recent_mask = np.zeros(n, dtype=bool)
        recent_mask[recent_lo:recent_hi] = True
        recent_valid = recent_mask & valid
        overall_valid = valid

        n_recent_candidates  = int((best_action[recent_valid] != 3).sum())
        n_overall_candidates = int((best_action[overall_valid] != 3).sum())
        recent_signal_density  = (n_recent_candidates / max(1, recent_valid.sum()))
        overall_signal_density = (n_overall_candidates / max(1, overall_valid.sum()))

        results.append({
            "tp_mult": tp, "sl_mult": sl, "max_holding": mh,
            "n_recent_candidates": n_recent_candidates,
            "n_overall_candidates": n_overall_candidates,
            "recent_signal_density": recent_signal_density,
            "overall_signal_density": overall_signal_density,
        })

    results.sort(key=lambda r: r["n_recent_candidates"], reverse=True)

    print(f"\n  {'tp':>5} {'sl':>5} {'max_hold':>9}  "
          f"{'recent_cand':>12} {'recent_dens':>12}  "
          f"{'overall_cand':>13} {'overall_dens':>13}")
    for r in results:
        print(f"  {r['tp_mult']:>5.2f} {r['sl_mult']:>5.2f} {r['max_holding']:>9d}  "
              f"{r['n_recent_candidates']:>12,} {r['recent_signal_density']:>11.2%}  "
              f"{r['n_overall_candidates']:>13,} {r['overall_signal_density']:>12.2%}")

    os.makedirs(gbt.OUT_DIR, exist_ok=True)
    with open(OUT_PHASE1_PATH, "w") as f:
        json.dump({
            "n_rows": n, "recent_lo": recent_lo, "recent_hi": recent_hi,
            "results": results,
        }, f, indent=2, default=gbt._json_default)
    print(f"\n  ✓  Phase 1 results saved → {OUT_PHASE1_PATH}")

    return results


# ─────────────────────────────────────────────
# Phase 2 — real CPCV screen on top-K candidates
# ─────────────────────────────────────────────

def run_phase2(states: np.ndarray, prices_aligned: np.ndarray, atr_aligned: np.ndarray,
              aligned_df, candidates: list, seeds: tuple, n_bagged_fits: int) -> list:
    print(f"\n{'='*62}\n  PHASE 2 — real CPCV screen on top {len(candidates)} "
          f"candidate(s)  (seeds={list(seeds)}, n_bagged_fits={n_bagged_fits})"
          f"\n{'='*62}")
    print(f"  NOTE: this is a fast SCREEN, not the final confirmation run —\n"
          f"  it uses fewer seeds and less bagging than main_gbt.py's real\n"
          f"  confirm_gate_nested() (which uses "
          f"{list(gbt.DEFAULT_CONFIRMATION_SEEDS)} seeds and "
          f"n_bagged_fits={gbt.DEFAULT_N_BAGGED_FITS}). Re-run the winning\n"
          f"  config through `python main_gbt.py --tp-mult ... --sl-mult ...\n"
          f"  --max-holding ...` for the real gate verdict before trusting it.")

    phase2_results = []
    for cand in candidates:
        tp, sl, mh = cand["tp_mult"], cand["sl_mult"], cand["max_holding"]
        print(f"\n{'-'*62}\n  Candidate: tp_mult={tp}  sl_mult={sl}  "
              f"max_holding={mh}\n{'-'*62}")
        labels = gbt.labels_from_prices(prices_aligned, atr_aligned,
                                        tp_mult=tp, sl_mult=sl, max_holding=mh)

        per_seed = []
        pooled_path_results = []
        recent_per_seed = []
        for seed in seeds:
            path_results, fitted_paths, valid = gbt.run_cpcv(
                states, labels, aligned_df, random_state=seed,
                n_bagged_fits=n_bagged_fits,
            )
            pooled_path_results.extend(path_results)
            passed, stats = gbt.evaluate_gate(path_results)
            per_seed.append({
                "seed": seed, "passed": bool(passed),
                "n_total": stats["n_total"], "n_pos": stats.get("n_pos", 0),
                "mean_test_avg_pnl": stats["mean_test_avg_pnl"],
                "mean_test_sharpe": stats.get("mean_test_sharpe", float("nan")),
                "pbo": stats["pbo"],
            })

            recent = gbt._extract_recent_regime_result(
                states, labels, fitted_paths, valid, threshold=0.5,
            )
            if recent is not None:
                recent["seed"] = seed
                recent["reliable"] = recent["n_test_trades"] >= gbt.RECENT_REGIME_MIN_TEST_TRADES
                recent["recent_regime_passed"] = bool(
                    recent["reliable"]
                    and recent["test_avg_pnl"] > gbt.RECENT_REGIME_AVG_PNL_FLOOR
                )
            recent_per_seed.append(recent)

            rstr = ("n/a" if recent is None else
                    f"n_trades={recent['n_test_trades']}  "
                    f"test_avg={recent['test_avg_pnl']:+.4%}  "
                    f"reliable={recent['reliable']}")
            print(f"    seed {seed}: gate={'PASS' if passed else 'FAIL'}  "
                  f"mean_test_avg={stats['mean_test_avg_pnl']:+.4%}  "
                  f"recent_regime[{rstr}]")

        pooled_passed, pooled_stats = gbt.evaluate_gate(pooled_path_results)
        rr_reliable = [r for r in recent_per_seed if r is not None and r["reliable"]]
        rr_frac_pass = (
            sum(1 for r in rr_reliable if r["recent_regime_passed"]) / len(rr_reliable)
            if rr_reliable else float("nan")
        )

        result = {
            "tp_mult": tp, "sl_mult": sl, "max_holding": mh,
            "seeds": list(seeds), "n_bagged_fits": n_bagged_fits,
            "per_seed": per_seed,
            "pooled_mean_test_avg_pnl": pooled_stats["mean_test_avg_pnl"],
            "pooled_mean_test_sharpe": pooled_stats.get("mean_test_sharpe", float("nan")),
            "pooled_pbo": pooled_stats["pbo"],
            "pooled_passed": bool(pooled_passed),
            "recent_regime_per_seed": recent_per_seed,
            "recent_regime_n_reliable_seeds": len(rr_reliable),
            "recent_regime_frac_pass": rr_frac_pass,
        }
        phase2_results.append(result)

        print(f"  → pooled: {'PASS' if pooled_passed else 'FAIL'}  "
              f"mean_test_avg={pooled_stats['mean_test_avg_pnl']:+.4%}  "
              f"PBO={pooled_stats['pbo']:.1%}  |  recent-regime reliable "
              f"seeds: {len(rr_reliable)}/{len(seeds)}  "
              f"({'n/a' if np.isnan(rr_frac_pass) else f'{rr_frac_pass:.0%} passed'})")

    # Rank: prefer configs with more reliable recent-regime reads first,
    # then by how many of those reliable reads actually passed the
    # recency-risk floor, then by pooled mean test avg/trade.
    phase2_results.sort(key=lambda r: (
        r["recent_regime_n_reliable_seeds"],
        0 if np.isnan(r["recent_regime_frac_pass"]) else r["recent_regime_frac_pass"],
        r["pooled_mean_test_avg_pnl"],
    ), reverse=True)

    print(f"\n{'='*62}\n  PHASE 2 RANKING (best first)\n{'='*62}")
    for r in phase2_results:
        rr_str = ("n/a" if np.isnan(r["recent_regime_frac_pass"])
                  else f"{r['recent_regime_frac_pass']:.0%}")
        print(f"  tp={r['tp_mult']:.2f} sl={r['sl_mult']:.2f} "
              f"max_holding={r['max_holding']:>3d}  |  "
              f"recent-regime reliable={r['recent_regime_n_reliable_seeds']}/"
              f"{len(r['seeds'])} (pass={rr_str})  |  "
              f"pooled_gate={'PASS' if r['pooled_passed'] else 'FAIL'}  "
              f"pooled_avg={r['pooled_mean_test_avg_pnl']:+.4%}")

    with open(OUT_PHASE2_PATH, "w") as f:
        json.dump(phase2_results, f, indent=2, default=gbt._json_default)
    print(f"\n  ✓  Phase 2 results saved → {OUT_PHASE2_PATH}")

    return phase2_results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Screen triple-barrier configs for a trustworthy "
                     "recent-regime CPCV read.")
    parser.add_argument("--frozen-data", action="store_true",
                        help="Use the existing master CSV(s) as-is instead "
                             "of refetching/appending new raw data.")
    parser.add_argument("--no-multi-timeframe", action="store_true",
                        help="Disable 15m/1h context for phase 2's state "
                             "matrix (only relevant if --phase2-top-k > 0).")
    parser.add_argument("--tp-mults", type=float, nargs="+",
                        default=list(DEFAULT_TP_MULTS))
    parser.add_argument("--sl-mults", type=float, nargs="+",
                        default=list(DEFAULT_SL_MULTS))
    parser.add_argument("--max-holdings", type=int, nargs="+",
                        default=list(DEFAULT_MAX_HOLDINGS))
    parser.add_argument("--phase2-top-k", type=int, default=DEFAULT_PHASE2_TOP_K,
                        help="Number of top phase-1 candidates to run through "
                             "real CPCV in phase 2. Set to 0 to skip phase 2 "
                             "entirely (label-only screen, seconds not minutes).")
    parser.add_argument("--phase2-seeds", type=int, nargs="+",
                        default=list(DEFAULT_PHASE2_SEEDS),
                        help="Seeds used for the phase-2 CPCV screen. Kept "
                             "separate from main_gbt.py's real "
                             "DEFAULT_CONFIRMATION_SEEDS so this script's "
                             "screen never masquerades as the real gate run.")
    parser.add_argument("--phase2-bagged-fits", type=int,
                        default=DEFAULT_PHASE2_BAGGED_FITS,
                        help="n_bagged_fits for phase 2 (reduced from "
                             "main_gbt.py's default for speed).")
    args = parser.parse_args()

    print("Loading prices/ATR for phase 1 (no 15m/1h fetch, no state "
          "aggregation needed)...")
    prices_aligned, atr_aligned, aligned_df = gbt.load_prices_for_labeling(
        frozen=args.frozen_data,
    )

    phase1_results = run_phase1(prices_aligned, atr_aligned,
                                args.tp_mults, args.sl_mults, args.max_holdings)

    if args.phase2_top_k <= 0:
        print("\n  --phase2-top-k=0 — skipping phase 2. Pick a promising "
              "config from the phase-1 table above and either re-run with "
              "--phase2-top-k > 0 to screen it with real CPCV, or go "
              "straight to `python main_gbt.py --tp-mult ... --sl-mult ... "
              "--max-holding ...` for the real gate.")
        return

    top_candidates = phase1_results[:args.phase2_top_k]

    print(f"\nBuilding state matrix ONCE for phase 2 "
          f"(multi_timeframe={'no' if args.no_multi_timeframe else 'yes'})...")
    states, states_prices, states_atr, states_aligned_df = gbt.build_states(
        frozen=args.frozen_data,
        enable_multi_timeframe=(False if args.no_multi_timeframe else None),
    )
    print(f"  state_dim={states.shape[1]}  n_rows={len(states):,}")

    run_phase2(states, states_prices, states_atr, states_aligned_df,
              top_candidates, tuple(args.phase2_seeds), args.phase2_bagged_fits)


if __name__ == "__main__":
    main()

# python label_sweep.py --frozen-data --phase2-top-k 0
# python label_sweep.py --frozen-data
# python label_sweep.py --frozen-data --tp-mults 1.0 1.3 1.5 --sl-mults 0.6 0.8 --max-holdings 10 14 20