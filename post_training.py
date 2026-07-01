"""
post_training.py
───────────────────
Quick post-training memory bank inspection.

CHANGES IN THIS REVISION
───────────────────────────
No functional/crash risk existed here before — MCKNNMemory.load()
already restores state_dim/eps_dist/max_weight_ratio/min_tick_gap from
the saved .npz regardless of how it was produced. This revision just
surfaces those fields (previously invisible here) for the same
transparency reasons live_mcknn.py / diagnostic_mcknn.py now report
them: knowing whether the loaded bank actually used multi-timeframe
context, what its temporal-exclusion/weight-cap config was, and how
many distinct training episodes/ticks it spans is useful context when
interpreting bank size / return stats / action balance below.
"""

import sys; sys.path.insert(0, '.')
import numpy as np
from agents.mc_knn_agent import MCKNNAgent
from agents.mc_knn_memory import MCKNNMemory

CHECKPOINT_PATH = "outcomes/mc_knn_agent_best.npz"

mem = MCKNNMemory.load(CHECKPOINT_PATH)

print("=" * 62)
print(f"  POST-TRAINING MEMORY BANK SUMMARY")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print("=" * 62)

# 4h-only baseline dim, mirroring main_mcknn.py's STATE_DIM_WITHOUT_CONTEXT,
# used only to infer whether this bank used multi-timeframe context.
FEATURES = ['RSI_Scaled', 'MACD_Scaled', 'BB_Scaled',
            'OBV_Scaled', 'ATR_Scaled', 'MeanDev_Scaled']
PACES = (1, 6, 42, 90)
STATE_DIM_4H_ONLY = (len(FEATURES) * 2 * len(PACES)) + 2
uses_context = mem.state_dim > STATE_DIM_4H_ONLY

print(f"\n── BANK CONFIG ─────────────────────────────────────────────")
print(f"  state_dim         : {mem.state_dim}  "
      f"(multi_timeframe_context={'YES' if uses_context else 'NO — 4h-only'})")
print(f"  action_dim        : {mem.action_dim}")
print(f"  k (neighbors)     : {mem.k}")
print(f"  max_size          : {mem.max_size:,}")
print(f"  signal_threshold  : {mem.signal_threshold}")
print(f"  eps_dist          : {getattr(mem, 'eps_dist', 'n/a (pre-integrity-check checkpoint)')}")
print(f"  max_weight_ratio  : {getattr(mem, 'max_weight_ratio', 'n/a (pre-integrity-check checkpoint)')}")
print(f"  min_tick_gap      : {getattr(mem, 'min_tick_gap', 'n/a (pre-integrity-check checkpoint)')}")

print(f"\n── BANK CONTENTS ────────────────────────────────────────────")
print(f"  bank size         : {len(mem):,}")
print(f"  n_commits         : {mem.n_commits}")
print(f"  n_prunes          : {mem.n_prunes}")

n = len(mem)
if n > 0:
    episode_ids = mem.episode_id[:n]
    ticks       = mem.ticks[:n]
    distinct_episodes = np.unique(episode_ids)
    print(f"  distinct episode_id(s): {len(distinct_episodes)}  "
          f"{distinct_episodes.tolist() if len(distinct_episodes) <= 10 else '(>10, omitted)'}")
    print(f"  tick range        : [{ticks.min()}, {ticks.max()}]")

    print(f"\n── RETURN STATS ─────────────────────────────────────────────")
    print(f"  mean return       : {mem.returns[:n].mean():.5f}")
    print(f"  std return        : {mem.returns[:n].std():.5f}")
    print(f"  min / max return  : {mem.returns[:n].min():.5f} / {mem.returns[:n].max():.5f}")

    print(f"\n── ACTION BALANCE ───────────────────────────────────────────")
    action_names = ["LONG", "SHORT", "CLOSE", "HOLD"]
    counts = np.bincount(mem.actions[:n], minlength=4)
    for name, c in zip(action_names, counts):
        print(f"  {name:<6}: {c:>8,}  ({c/n:.1%})")
else:
    print("  ⚠  Bank is empty — nothing further to report.")

print("\n" + "=" * 62)