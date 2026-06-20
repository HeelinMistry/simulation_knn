import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.data_manager import update_master_data
from agents.state_aggregator import StateAggregator

FEATURES = ['RSI_Scaled','MACD_Scaled','BB_Scaled','OBV_Scaled','ATR_Scaled','MeanDev_Scaled']
PACES = (1, 6, 42, 90)
WARMUP_IDX = 128

df = update_master_data()
df = df[['Open_time','Close'] + FEATURES].dropna().reset_index(drop=True)
ind = df[FEATURES].values.astype(np.float32)

agg = StateAggregator(PACES, num_indicators=len(FEATURES))
agg.warm_up_all(ind, WARMUP_IDX)
states = []
for i in range(WARMUP_IDX+1, min(len(df), WARMUP_IDX+5000)):
    agg.tick = 0 if i == WARMUP_IDX+1 else agg.tick
    agg.update(ind[i])
    states.append(agg.get_state())
states = np.array(states)
print('n_states', len(states), 'dim', states.shape[1])

# ── Nearest-neighbor distance distribution ───────────────────────────────────
# BUG FIX vs prior revision: `sample = states[np.random.choice(...)]` copies
# VALUES into a new array with no memory of which original row each came
# from, so `np.fill_diagonal(d[:, :500], np.inf)` was zeroing out whatever
# happened to be the first 500 *columns* of the distance matrix — not
# necessarily (or even usually) each sampled row's own self-distance. That
# meant a query state's distance to itself (always exactly 0.0) was often
# still included as a "neighbor" distance, which is almost certainly why
# `median nearest-neighbor dist: 0.0` was reported even before accounting
# for any genuine near-duplicate states in the data.
#
# Fix: sample INDICES (not values) so each sampled row's position within
# the full `states` array is known, then exclude that exact self-index
# when computing nearest-neighbor distances for that row.
sample_idx = np.random.choice(len(states), 500, replace=False)
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