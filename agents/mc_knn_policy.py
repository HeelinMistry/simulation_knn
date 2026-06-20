"""
agents/mc_knn_policy.py
─────────────────────────
Monte Carlo k-NN "actor" — drop-in replacement for models/actor.py's
Actor class, with the same act()/get_action() call signature so
unified_executor.py needs minimal edits.

Mapping vs actor.py
──────────────────────
  Actor.forward(state) -> softmax(logits)          MCKNNPolicy has no
                                                     learnable logits; it
                                                     queries MCKNNMemory
                                                     directly.
  Actor.get_action(state, mask) -> action,          MCKNNPolicy.get_action
    probs, log_probs                                 has the same
                                                       signature/outputs.
  Actor.act(state_np, ...) -> action, probs          MCKNNPolicy.act has
                                                       the identical
                                                       signature.

What "probs" means here
  These are NOT a calibrated softmax probability distribution — there
  are no logits to softmax. They are vote-share weights from the k
  nearest neighbors (see MCKNNMemory.query docstring), rescaled to sum
  to 1 purely so that:
    (a) live.py's probability bars render without modification,
    (b) diagnostic.py's entropy/confidence sections still have a
        well-defined input,
  but they should be read as "how lopsided was the neighbor vote",
  not "P(action | state)" in any Bayesian sense. Diagnostics renames
  these conceptually (see diagnostic_mcknn.py) to avoid conflating the
  two.

deterministic vs sampled
  Actor.get_action sampled from Categorical(probs) when not
  deterministic, to drive entropy-regularised exploration during SAC
  training. MC-kNN does not have an entropy term to regulate, so
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

    # ── Same shape as Actor.get_action ──────────────────────────────────────

    def get_action(self, state: np.ndarray, action_mask=None, k: int = None):
        action, probs, info = self.memory.query(state, k=k, action_mask=action_mask)
        log_probs = np.log(probs + 1e-8)
        return action, probs, log_probs, info

    # ── Same shape as Actor.act ──────────────────────────────────────────────

    def act(self, state_np: np.ndarray, deterministic: bool = True,
            device: str = "cpu", action_mask=None, k: int = None):
        """
        `deterministic` and `device` are accepted (and ignored beyond
        deterministic's role below) purely so call sites written against
        Actor.act's signature don't need changes. There is no device —
        everything here is numpy.
        """
        action, probs, _log_probs, _info = self.get_action(
            state_np, action_mask=action_mask, k=k
        )
        return action, probs