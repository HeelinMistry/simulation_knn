"""
pre_training.py
─────────────────
Near-duplicate state-density diagnostic — run BEFORE training to sanity
check whether the state space carries enough information for the
Monte-Carlo k-NN memory bank to distinguish genuinely different market
moments.

CHANGES IN THIS REVISION — ALIGNMENT FIX
─────────────────────────────────────────────
The prior revision only built the 4h-only StateAggregator state (50-dim)
and ran the near-duplicate diagnostic on that. But main_mcknn.py now
trains on a richer state when ENABLE_MULTI_TIMEFRAME=True — the primary
4h market vector PLUS aligned 15m/1h context (122-dim total). Running
this diagnostic against the 50-dim space alone checks the WRONG state
space: if multi-timeframe context improved duplicate density (the whole
point of adding it — see multi_timeframe_state.py's docstring), this
diagnostic would never show it, since it never builds that context at
all.

Fix: this file now mirrors main_mcknn.py's ENABLE_MULTI_TIMEFRAME /
CONTEXT_TIMEFRAMES / CONTEXT_PACES config exactly and, when enabled,
concatenates the same multi-timeframe context onto each tick's state
before running the duplicate-density check — so "is the duplicate
problem fixed" is answered against the actual state space the model
trains on, not a stale proxy.

If raw 15m/1h data isn't available, falls back to the 4h-only check
with a loud warning, same fallback pattern as main_mcknn.py uses.
"""

import sys; sys.path.insert(0, '.')
import os

import numpy as np
import pandas as pd

from data.data_manager import update_master_data, update_all_timeframes
from agents.state_aggregator import StateAggregator
from multi_timeframe_state import build_multi_timeframe_context

# ── Config — MUST match main_mcknn.py exactly ────────────────────────────────
FEATURES = ['RSI_Scaled', 'MACD_Scaled', 'BB_Scaled',
            'OBV_Scaled', 'ATR_Scaled', 'MeanDev_Scaled']
PACES = (1, 6, 42, 90)
WARMUP_IDX = 128

ENABLE_MULTI_TIMEFRAME = True
CONTEXT_TIMEFRAMES = ("15m", "1h")
CONTEXT_PACES = (1, 4, 16)

MAX_TICKS_TO_SAMPLE = 5000   # cap for tractability, same as the prior revision


# ─────────────────────────────────────────────────────────────────────────────
# Build the actual state space (4h-only, or 4h + multi-timeframe context)
# ─────────────────────────────────────────────────────────────────────────────

def build_state_space(df: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """
    Returns (states, used_context). states is (N, state_dim) float32,
    built EXACTLY like main_mcknn.py's run_epoch/UnifiedExecutor.get_state
    does: [4h market vector][extra_context if enabled][2 zero portfolio
    dims, since this is a structural diagnostic with no real position
    open — matches StateAggregator.get_state()'s default zero-padding
    for portfolio_info=None].
    """
    ind = df[FEATURES].values.astype(np.float32)
    n_total = min(len(df), WARMUP_IDX + MAX_TICKS_TO_SAMPLE)

    agg = StateAggregator(PACES, num_indicators=len(FEATURES))
    agg.warm_up_all(ind, WARMUP_IDX)

    extra_context_arr = None
    used_context = False
    if ENABLE_MULTI_TIMEFRAME:
        try:
            timeframe_dfs = update_all_timeframes(CONTEXT_TIMEFRAMES)
            missing = [tf for tf in CONTEXT_TIMEFRAMES if tf not in timeframe_dfs]
            if missing:
                raise FileNotFoundError(f"timeframe(s) {missing} unavailable")
            timeframe_frames = {
                tf: tdf[["Open_time"] + FEATURES].dropna().reset_index(drop=True)
                for tf, tdf in timeframe_dfs.items()
            }
            print(f"Building multi-timeframe context ({', '.join(timeframe_frames.keys())}) "
                  f"to match the ACTUAL training state space...")
            extra_context_arr = build_multi_timeframe_context(
                df, timeframe_frames, FEATURES, context_paces=CONTEXT_PACES,
            )
            used_context = True
        except FileNotFoundError as exc:
            print(f"⚠ Multi-timeframe context unavailable ({exc}) — falling back "
                  f"to 4h-only state for this diagnostic. NOTE: if "
                  f"main_mcknn.py's ENABLE_MULTI_TIMEFRAME=True and raw 15m/1h "
                  f"data IS available there, this diagnostic will be checking "
                  f"a smaller state space than what actually gets trained on.")

    states = []
    agg.tick = 0
    for i in range(WARMUP_IDX + 1, n_total):
        agg.update(ind[i])
        # portfolio_info=None -> StateAggregator appends 2 zero dims,
        # matching UnifiedExecutor.get_state()'s layout exactly when no
        # position is open (the common case at most ticks).
        market_vec = agg.get_state(portfolio_info=None)[:-2]
        portfolio_vec = np.zeros(2, dtype=np.float32)
        if used_context:
            ctx_row = extra_context_arr[i]
            state_vec = np.concatenate([market_vec, ctx_row, portfolio_vec])
        else:
            state_vec = np.concatenate([market_vec, portfolio_vec])
        states.append(state_vec)

    return np.array(states, dtype=np.float32), used_context


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

df = update_master_data()
df = df[['Open_time', 'Close'] + FEATURES].dropna().reset_index(drop=True)

states, used_context = build_state_space(df)
print(f"\nn_states {len(states)}  dim {states.shape[1]}  "
      f"(multi_timeframe_context={'YES' if used_context else 'NO — 4h-only'})")

# ── Nearest-neighbor distance distribution ───────────────────────────────────
# BUG FIX (carried over from prior revision): `sample = states[np.random.choice(...)]`
# copies VALUES into a new array with no memory of which original row each
# came from, so `np.fill_diagonal(d[:, :500], np.inf)` was zeroing out
# whatever happened to be the first 500 *columns* of the distance matrix —
# not necessarily (or even usually) each sampled row's own self-distance.
# That meant a query state's distance to itself (always exactly 0.0) was
# often still included as a "neighbor" distance, which is almost certainly
# why `median nearest-neighbor dist: 0.0` was reported even before
# accounting for any genuine near-duplicate states in the data.
#
# Fix: sample INDICES (not values) so each sampled row's position within
# the full `states` array is known, then exclude that exact self-index
# when computing nearest-neighbor distances for that row.
n_sample = min(500, len(states))
sample_idx = np.random.choice(len(states), n_sample, replace=False)
sample = states[sample_idx]

from scipy.spatial.distance import cdist
d = cdist(sample, states)
for row, orig_idx in enumerate(sample_idx):
    d[row, orig_idx] = np.inf   # exclude true self-distance, not a guessed column

nn_dist = np.sort(d, axis=1)[:, :10]  # 10 nearest per query (self now excluded)
print('median nearest-neighbor dist:', np.median(nn_dist[:, 0]))
print('median 10th-NN dist:', np.median(nn_dist[:, 9]))
print('overall pairwise dist median:', np.median(d[np.isfinite(d)]))
ratio = np.median(nn_dist[:, 0]) / np.median(d[np.isfinite(d)])
print('ratio (NN/overall) -- want this << 1:', ratio)

# ── Genuine near-duplicate diagnostic (separate from the self-exclusion fix) ──
# Even with self-distance correctly excluded, report how many *other* rows
# sit suspiciously close to each sampled row — this is the actual signal
# the integrity check needs: are there real near-duplicate states (e.g.
# from flat/low-vol stretches, or slow paces barely moving tick-to-tick)
# independent of the self-pairing bug above.
near_dup_threshold = 0.05  # tune relative to overall pairwise scale above
frac_with_near_dup = (nn_dist[:, 0] < near_dup_threshold).mean()
print(f'fraction of sampled states with a near-duplicate '
      f'(dist < {near_dup_threshold}): {frac_with_near_dup:.1%}')
if frac_with_near_dup > 0.05:
    print('  ⚠  Meaningful near-duplicate density detected — this is the '
          'mechanism behind train/val leakage in main_mcknn.py; confirm '
          'MIN_TICK_GAP in main_mcknn.py is large enough relative to the '
          'typical gap between duplicate ticks before retraining.')
    if not used_context:
        print('  ℹ  This check ran on the 4h-only state space. If '
              'main_mcknn.py has ENABLE_MULTI_TIMEFRAME=True with raw '
              '15m/1h data available, re-run this diagnostic with that data '
              'present — multi-timeframe context is specifically designed '
              'to reduce this density, and the 4h-only number above may '
              'overstate the problem the model actually faces.')
else:
    print('  ✓  Near-duplicate density is low.'
          + ('' if used_context else
             ' (4h-only check — if multi-timeframe context is enabled in '
             'main_mcknn.py, this is a conservative/stale number; the real '
             'training state space may be even better separated.)'))