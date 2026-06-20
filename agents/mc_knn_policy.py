"""
agents/mc_knn_policy.py
─────────────────────────
Monte Carlo k-NN "actor" — drop-in replacement for models/actor.py's
Actor class, with the same act()/get_action() call signature so
unified_executor.py needs minimal edits.

CHANGES IN THIS REVISION
───────────────────────────
get_action()/act() now accept optional query_tick / query_episode_id /
min_tick_gap kwargs and simply forward them to MCKNNMemory.query() —
see mc_knn_memory.py and episode_buffer.py docstrings for why this
exists (temporal exclusion to stop a query from voting on a
near-duplicate of itself, which was the dominant cause of the
train/val performance gap found in the integrity check). All three
default to None, so existing call sites that don't pass them are
unaffected (no temporal exclusion applied — identical to prior
behaviour).

What "probs" means here
  These are NOT a calibrated softmax probability distribution — there
  are no logits to softmax. They are vote-share weights from the k
  nearest neighbors (see MCKNNMemory.query docstring), rescaled to sum
  to 1 purely so that:
    (a) live.py's probability bars render without modification,
    (b) diagnostic.py's entropy/confidence sections still have a
        well-defined input,
  but they should be read as "how lopsided was the neighbor vote",
  not "P(action | state)" in any Bayesian sense.

deterministic vs sampled
  "non-deterministic" mode here means: with probability `epsilon`
  (passed in from the caller, same epsilon-greedy knob main_mcknn.py
  uses) pick a uniformly random valid action instead of the kNN-vote
  winner. This keeps unified_executor.py's epsilon-greedy exploration
  branch structurally unchanged.
"""

import numpy as np

from agents.mc_knn_memory import MCKNNMemory


ACTION_NAMES = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}


class MCKNNPolicy:
    def __init__(self, memory: MCKNNMemory, action_dim: int = 4):
        self.memory = memory
        self.action_dim = action_dim

    # ── Same shape as Actor.get_action, plus optional temporal-exclusion args ──

    def get_action(self, state: np.ndarray, action_mask=None, k: int = None,
                    query_tick: int = None, query_episode_id: int = None,
                    min_tick_gap: int = None):
        action, probs, info = self.memory.query(
            state, k=k, action_mask=action_mask,
            query_tick=query_tick, query_episode_id=query_episode_id,
            min_tick_gap=min_tick_gap,
        )
        log_probs = np.log(probs + 1e-8)
        return action, probs, log_probs, info

    # ── Same shape as Actor.act ──────────────────────────────────────────────

    def act(self, state_np: np.ndarray, deterministic: bool = True,
            device: str = "cpu", action_mask=None, k: int = None,
            query_tick: int = None, query_episode_id: int = None,
            min_tick_gap: int = None):
        """
        `deterministic` and `device` are accepted (and ignored beyond
        deterministic's role below) purely so call sites written against
        Actor.act's signature don't need changes. There is no device —
        everything here is numpy.
        """
        action, probs, _log_probs, _info = self.get_action(
            state_np, action_mask=action_mask, k=k,
            query_tick=query_tick, query_episode_id=query_episode_id,
            min_tick_gap=min_tick_gap,
        )
        return action, probs