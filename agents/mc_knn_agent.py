"""
agents/mc_knn_agent.py
────────────────────────
Top-level agent object — drop-in structural replacement for
agents/sac_agent.py's SACAgent.

CHANGES IN THIS REVISION
───────────────────────────
__init__ now accepts and forwards eps_dist / max_weight_ratio /
min_tick_gap to MCKNNMemory (see mc_knn_memory.py docstring for what
each one fixes). select_action() accepts optional query_tick /
query_episode_id / min_tick_gap and forwards them to the policy/memory
so the caller (unified_executor.py / main_mcknn.py) can opt in to
temporal exclusion during training without changing the public
select_action signature's required arguments.
"""

import os
import numpy as np

from agents.mc_knn_memory import MCKNNMemory
from agents.mc_knn_policy import MCKNNPolicy


class MCKNNAgent:
    def __init__(
        self,
        state_dim:  int = 50,
        action_dim: int = 4,
        k:          int = 25,
        max_size:   int = 200_000,
        gamma:      float = 0.97,
        signal_threshold: float = 0.0005,
        eps_dist: float = 1e-3,
        max_weight_ratio: float = 50.0,
        min_tick_gap: int = 100,
        device: str = None,   # accepted, unused — keeps call sites unchanged
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.device      = "cpu"   # no GPU work happens here; numpy only

        self.memory = MCKNNMemory(
            state_dim=state_dim, action_dim=action_dim,
            k=k, max_size=max_size, signal_threshold=signal_threshold,
            eps_dist=eps_dist, max_weight_ratio=max_weight_ratio,
            min_tick_gap=min_tick_gap,
        )
        self.actor = MCKNNPolicy(self.memory, action_dim=action_dim)

        # ── Diagnostics bookkeeping (kNN analogues of SACAgent's fields) ────
        self.n_episodes_committed = 0
        self.last_vote_margin     = 0.0
        self.last_bank_size       = 0
        self.last_n_prunes        = 0

        print(f"[MCKNNAgent] state_dim={state_dim} action_dim={action_dim} "
              f"k={k} max_size={max_size:,} gamma={gamma}  "
              f"max_weight_ratio={max_weight_ratio} min_tick_gap={min_tick_gap}  "
              f"(no GPU/optimiser — memory bank only)")

    # ── critic shim: explicit failure instead of silent wrong behaviour ─────

    def critic(self, *args, **kwargs):
        raise NotImplementedError(
            "MCKNNAgent has no Q-network. Code that called agent.critic(state) "
            "for SAC's Q1/Q2 values should instead call "
            "agent.memory.query(state) and use info['neighbor_returns'] / "
            "info['vote_margin'] — see diagnostic_mcknn.py Section 2/9 for "
            "the kNN-analogue replacements."
        )

    # ── Public API — matches SACAgent.select_action, plus optional temporal args ──

    def select_action(self, state: np.ndarray, deterministic: bool = False,
                      action_mask=None, query_tick: int = None,
                      query_episode_id: int = None, min_tick_gap: int = None):
        return self.actor.act(
            state, deterministic=deterministic, device=self.device,
            action_mask=action_mask, query_tick=query_tick,
            query_episode_id=query_episode_id, min_tick_gap=min_tick_gap,
        )

    # ── Update — called once per finished episode, not once per N ticks ─────

    def update(self, episode_buffer):
        """
        Backfill the episode's MC returns and commit to the memory bank.
        Call this once at the end of each training episode/epoch pass —
        the kNN analogue of calling SACAgent.update() every UPDATE_EVERY
        ticks, except batched per-episode because MC returns require the
        full episode to exist first (see episode_buffer.py docstring).
        """
        if len(episode_buffer) == 0:
            return
        episode_buffer.end_episode_and_commit(self.memory)
        self.n_episodes_committed += 1
        self.last_bank_size = len(self.memory)
        self.last_n_prunes  = self.memory.n_prunes

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "outcomes/mc_knn_agent.npz", episode_buffer=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if episode_buffer is not None and len(episode_buffer) > 0:
            episode_buffer.end_episode_and_commit(self.memory)
        self.memory.save(path)
        print(f"[MCKNNAgent] ✅ Saved → {path}  "
              f"(bank_size={len(self.memory):,}  episodes={self.n_episodes_committed})")

    def load(self, path: str = "outcomes/mc_knn_agent.npz", episode_buffer=None):
        npz_path = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(npz_path):
            print(f"[MCKNNAgent] ⚠️  No checkpoint at {npz_path} — starting fresh (empty bank).")
            return
        self.memory = MCKNNMemory.load(npz_path)
        self.actor  = MCKNNPolicy(self.memory, action_dim=self.action_dim)
        self.last_bank_size = len(self.memory)
        self.last_n_prunes  = self.memory.n_prunes
        print(f"[MCKNNAgent] ✅ Loaded ← {npz_path}  (bank_size={len(self.memory):,})")