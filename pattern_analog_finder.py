"""
pattern_analog_finder.py
─────────────────────────
Live analog retrieval: given the most recent (or any) tick's state vector,
find its k nearest historical neighbors by raw state-space distance, and
report what ACTUALLY happened after each analog — dates, distances, and
their triple-barrier outcomes (triple_barrier.py) — rather than a single
vote-share number.

This is deliberately NOT MCKNNAgent/MCKNNMemory: that bank returns
inverse-distance VOTE SHARES over actions, collapsed into one number.
This tool surfaces the individual analogs themselves, so you can see
*why* "today" looks like a favorable (or unfavorable) setup — which
past dates it resembles and what happened after each one — not just a
probability.

State representation
──────────────────────
Identical to main_gbt.py's build_states_and_labels(): StateAggregator's
4h multi-pace market vector, optionally fused with 15m/1h context via
multi_timeframe_state.py. Reusing this exactly (rather than a bespoke
feature set) means analogs found here are directly comparable to what
the GBT/MC-kNN pipelines already train and evaluate against.

Outcome definition
─────────────────────
triple_barrier.build_meta_labels() — price-path-only, policy-independent
"would LONG/SHORT have been net-profitable from this tick" labels. This
is what "favorable outcome" means throughout this codebase; reusing it
here (rather than inventing a new definition) keeps this tool consistent
with how main_gbt.py's classifier is trained and gated.

Leakage discipline — TWO independent guards, both required
────────────────────────────────────────────────────────────
  1. TEMPORAL EXCLUSION (lookback_ticks, from walkforward.py): candidates
     within `lookback_ticks` of the query tick are excluded, because
     their own state vectors were built from rolling-window history
     (up to 200 ticks) that overlaps the query's — they are not
     genuinely independent observations, just near-duplicates of "now".

  2. RESOLVED-OUTCOME ONLY: a candidate is only usable if its own
     triple-barrier outcome (entry tick .. touch tick) is fully in the
     past relative to the query tick. Concretely: valid_mask is True
     AND max(long_touch, short_touch) < query_idx. This guarantees
     every analog shown is a genuinely KNOWN historical outcome, not
     something whose "future" is still ahead of the query point.

Distance metric
──────────────────
Raw Euclidean distance in the (un-weighted) state space — the same
convention MC-kNN already uses elsewhere in this codebase. Note:
pre_training.py's diagnostic found raw Euclidean distance in the full
~122-dim multi-timeframe space gets noisier (NN/overall density ratio
degrades) than in the 4h-only ~50-dim space — if analogs look too
scattered/low-signal, try --no-multi-timeframe first, or come back to
add block-weighting between the 4h and context blocks.

Usage
──────
    python pattern_analog_finder.py                    # most recent tick, live data
    python pattern_analog_finder.py --tick -30          # query 30 ticks before the end
    python pattern_analog_finder.py --k 15 --frozen-data
    python pattern_analog_finder.py --no-multi-timeframe
"""

import argparse
import numpy as np
import pandas as pd

from agents.state_aggregator import StateAggregator
from data.data_manager import update_master_data, update_all_timeframes
from multi_timeframe_state import build_multi_timeframe_context
from walkforward import compute_required_lookback_ticks
from triple_barrier import build_meta_labels

# ── Config — mirrors main_gbt.py exactly, so analogs found here are
#    directly comparable to what that pipeline trains/evaluates on. ────────
FEATURES = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
            "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES = (1, 6, 42, 90)
WARMUP_IDX = 128

ENABLE_MULTI_TIMEFRAME = True
CONTEXT_TIMEFRAMES = ("15m", "1h")
CONTEXT_PACES = (1, 4, 16)

TP_MULT, SL_MULT, MAX_HOLDING = 1.5, 0.8, 20
COMMISSION = 0.00015


# ─────────────────────────────────────────────────────────────────────────
# State + label construction (mirrors main_gbt.build_states_and_labels())
# ─────────────────────────────────────────────────────────────────────────

def build_states(frozen: bool = False, enable_multi_timeframe: bool = True):
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
                df, timeframe_frames, FEATURES, context_paces=CONTEXT_PACES,
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
    open_times_aligned = df["Open_time"].values[WARMUP_IDX + 1:]
    atr_idx = FEATURES.index("ATR_Scaled")
    atr_aligned = ind[WARMUP_IDX + 1:, atr_idx]

    labels = build_meta_labels(
        prices_aligned, atr_aligned, tp_mult=TP_MULT, sl_mult=SL_MULT,
        max_holding=MAX_HOLDING, commission=COMMISSION,
    )

    return states, labels, prices_aligned, open_times_aligned


# ─────────────────────────────────────────────────────────────────────────
# Analog retrieval
# ─────────────────────────────────────────────────────────────────────────

def find_analogs(states: np.ndarray, labels: dict, query_idx: int,
                  k: int = 10, lookback_ticks: int = None) -> list:
    """
    Return the k nearest neighbors (raw Euclidean state-space distance)
    to states[query_idx], subject to the two leakage guards described in
    the module docstring. Each returned analog carries its OWN resolved
    triple-barrier outcome — not a re-derived/aggregated vote.
    """
    if lookback_ticks is None:
        lookback_ticks = compute_required_lookback_ticks()

    n = len(states)
    query_vec = states[query_idx]

    candidate_mask = np.ones(n, dtype=bool)
    lo = max(0, query_idx - lookback_ticks)
    hi = min(n, query_idx + lookback_ticks + 1)
    candidate_mask[lo:hi] = False  # guard 1: temporal exclusion

    resolved = labels["valid_mask"] & (
        np.maximum(labels["long_touch"], labels["short_touch"]) < query_idx
    )
    candidate_mask &= resolved       # guard 2: resolved-outcome only

    candidate_idx = np.flatnonzero(candidate_mask)
    if len(candidate_idx) == 0:
        return []

    dists = np.linalg.norm(states[candidate_idx] - query_vec[None, :], axis=1)
    order = np.argsort(dists)[:k]
    top_idx = candidate_idx[order]
    top_dist = dists[order]

    analogs = []
    for idx, dist in zip(top_idx, top_dist):
        long_fav = labels["long_label"][idx] == 1
        short_fav = labels["short_label"][idx] == 1
        if long_fav and short_fav:
            side = "LONG" if labels["long_return"][idx] >= labels["short_return"][idx] else "SHORT"
        elif long_fav:
            side = "LONG"
        elif short_fav:
            side = "SHORT"
        else:
            side = "NEITHER"

        if side == "LONG":
            ret, touch = labels["long_return"][idx], labels["long_touch"][idx]
        elif side == "SHORT":
            ret, touch = labels["short_return"][idx], labels["short_touch"][idx]
        else:
            # Neither direction resolved favorably — report whichever had
            # the smaller loss, for context (still real, still resolved).
            if labels["long_return"][idx] >= labels["short_return"][idx]:
                ret, touch = labels["long_return"][idx], labels["long_touch"][idx]
            else:
                ret, touch = labels["short_return"][idx], labels["short_touch"][idx]

        analogs.append({
            "tick": int(idx),
            "distance": float(dist),
            "favorable_side": side,
            "realised_return": float(ret),
            "duration_ticks": int(touch - idx),
        })
    return analogs


def summarize(analogs: list, open_times: np.ndarray, query_idx: int):
    query_date = pd.to_datetime(open_times[query_idx])
    if not analogs:
        print(f"\n  No valid historical analogs found for query tick {query_idx} "
              f"({query_date}). Try a smaller --k, more history, or check the "
              f"lookback_ticks exclusion isn't eating the whole dataset.")
        return

    print(f"\n{'='*72}\n  {len(analogs)} nearest historical analogs to "
          f"{query_date}  (tick {query_idx})\n{'='*72}")

    n_fav = sum(1 for a in analogs if a["favorable_side"] != "NEITHER")
    returns = np.array([a["realised_return"] for a in analogs])
    print(f"  Favorable-outcome rate among analogs : {n_fav}/{len(analogs)} "
          f"({n_fav/len(analogs):.0%})")
    print(f"  Mean realised return                 : {returns.mean():+.3%}")
    print(f"  Median realised return                : {np.median(returns):+.3%}")

    print(f"\n  {'Date':<20}{'Dist':>8}  {'Side':<8}{'Return':>10}  {'Hold(ticks)':>12}")
    print(f"  {'-'*20}{'-'*8}  {'-'*8}{'-'*10}  {'-'*12}")
    for a in sorted(analogs, key=lambda x: x["distance"]):
        d = pd.to_datetime(open_times[a["tick"]])
        print(f"  {str(d):<20}{a['distance']:>8.3f}  {a['favorable_side']:<8}"
              f"{a['realised_return']:>+9.3%}  {a['duration_ticks']:>12d}")
    print()


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live historical pattern-analog finder")
    parser.add_argument("--k", type=int, default=10,
                        help="number of nearest historical analogs to return")
    parser.add_argument("--tick", type=int, default=-1,
                        help="query tick, indexed from the END of the state "
                             "array (-1 = most recent tick, -30 = 30 ticks ago)")
    parser.add_argument("--frozen-data", action="store_true",
                        help="use the existing master CSV(s) as-is instead of "
                             "refetching/appending new raw data")
    parser.add_argument("--no-multi-timeframe", action="store_true",
                        help="use 4h-only state (~50-dim) instead of the full "
                             "multi-timeframe state (~122-dim)")
    args = parser.parse_args()

    print("Building states + triple-barrier ground truth...")
    states, labels, prices, open_times = build_states(
        frozen=args.frozen_data,
        enable_multi_timeframe=not args.no_multi_timeframe,
    )
    print(f"  {len(states):,} ticks  |  state_dim={states.shape[1]}")

    query_idx = args.tick if args.tick >= 0 else len(states) + args.tick
    query_idx = int(np.clip(query_idx, 0, len(states) - 1))

    analogs = find_analogs(states, labels, query_idx, k=args.k)
    summarize(analogs, open_times, query_idx)


if __name__ == "__main__":
    main()

# python pattern_analog_finder.py --frozen-data