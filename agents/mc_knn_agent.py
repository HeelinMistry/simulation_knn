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
        block_weights: np.ndarray = None,
        dim_scale_floor: float = 1e-3,
        device: str = None,   # accepted, unused — keeps call sites unchanged
        data_signature: str = None,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.max_size   = max_size   # stored so load() can enforce the cap
        self.device     = "cpu"

        # ── Integrity fix: fold/data-layout signature ────────────────────
        # This is the signature of the CURRENT run's (dataset snapshot,
        # purge/embargo fold boundaries, state config) — see
        # main_mcknn.py's compute_data_signature(). Stored here (not just
        # on self.memory) so load() can compare "what this run expects"
        # against "what the checkpoint on disk was actually built under"
        # even after self.memory gets replaced wholesale by
        # MCKNNMemory.load(). Callers that don't care about resume-safety
        # (live inference, diagnostics — anything that doesn't keep
        # training the loaded bank) can simply omit this and no check is
        # performed, preserving old behaviour.
        self.data_signature = data_signature

        self.memory = MCKNNMemory(
            state_dim=state_dim, action_dim=action_dim,
            k=k, max_size=max_size, signal_threshold=signal_threshold,
            eps_dist=eps_dist, max_weight_ratio=max_weight_ratio,
            min_tick_gap=min_tick_gap,
            block_weights=block_weights, dim_scale_floor=dim_scale_floor,
            data_signature=data_signature,
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
              f"weighted_distance={'custom block_weights' if block_weights is not None else 'uniform (std-normalized only)'}  "
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
        loaded_memory = MCKNNMemory.load(npz_path)

        # ── Integrity fix: refuse to resume onto an incompatible bank. ──────
        # A checkpoint's stored `tick` values are only meaningful relative
        # to the exact (dataset snapshot, purge/embargo fold boundaries,
        # state config) it was committed under (see
        # MCKNNMemory.data_signature docstring). If this run's expected
        # signature (self.data_signature, set at construction time) is
        # non-empty and DIFFERS from what's stored on disk, resuming would
        # silently mix rows whose ticks refer to different historical
        # slices under the same small integer episode_id space — this is
        # the confirmed root cause of a previously-reported "best val"
        # P/L not reproducing under diagnostic_mcknn.py's replay (bank
        # held episode_ids [42, 53] spanning two incompatible fold
        # layouts). We reset to a fresh bank rather than trusting stale,
        # possibly-misaligned data. Legacy checkpoints (no stored
        # data_signature, i.e. "") and callers that didn't opt into the
        # check (self.data_signature is None, e.g. live/diagnostic
        # loading) are unaffected — this only fires when BOTH sides
        # declare a signature and they disagree.
        stored_sig = getattr(loaded_memory, "data_signature", "") or ""
        if self.data_signature and stored_sig and stored_sig != self.data_signature:
            print(f"[MCKNNAgent] ⛔ REFUSING to resume onto incompatible bank at "
                  f"{npz_path}:")
            print(f"             stored data_signature  = {stored_sig}")
            print(f"             this run's data_signature = {self.data_signature}")
            print(f"             (dataset snapshot and/or purge/embargo fold "
                  f"boundaries differ from what produced this checkpoint — "
                  f"the stored `tick` values are not comparable to this "
                  f"run's ticks.) Starting from a FRESH empty bank instead "
                  f"of silently contaminating training with misaligned "
                  f"history. If you intended to keep training the same "
                  f"model on the same data, check whether data/raw/ "
                  f"changed between runs.")
            self.memory = MCKNNMemory(
                state_dim=self.state_dim, action_dim=self.action_dim,
                k=loaded_memory.k, max_size=self.max_size,
                signal_threshold=loaded_memory.signal_threshold,
                eps_dist=loaded_memory.eps_dist,
                max_weight_ratio=loaded_memory.max_weight_ratio,
                min_tick_gap=loaded_memory.min_tick_gap,
                block_weights=loaded_memory.block_weights,
                dim_scale_floor=loaded_memory.dim_scale_floor,
                data_signature=self.data_signature,
            )
            self.actor = MCKNNPolicy(self.memory, action_dim=self.action_dim)
            self.last_bank_size = 0
            self.last_n_prunes  = 0
            return

        self.memory = loaded_memory
        # ── Enforce the constructor's max_size, not the file's. ─────────────
        # MCKNNMemory.load() restores max_size from the .npz file (e.g. 200,000
        # from a prior run). If the caller set a smaller BANK_MAX_SIZE in the
        # MCKNNAgent constructor (e.g. 25,000), that setting is silently lost
        # unless we re-apply it here. We always trust the live config over the
        # checkpoint config for capacity limits, since the whole point of
        # changing BANK_MAX_SIZE is to take effect on the next run.
        # Pruning immediately if the loaded bank exceeds the new cap means the
        # model resumes training with the right bank size rather than carrying
        # 150K+ stale entries that override the cap for the entire run.
        if self.memory.max_size != self.max_size:
            self.memory.max_size = self.max_size
            if len(self.memory) > self.max_size:
                print(f"[MCKNNAgent] ↷ Loaded bank ({len(self.memory):,}) exceeds "
                      f"max_size={self.max_size:,} — pruning to cap.")
                self.memory.prune(target_size=self.max_size)
        self.actor  = MCKNNPolicy(self.memory, action_dim=self.action_dim)
        self.last_bank_size = len(self.memory)
        self.last_n_prunes  = self.memory.n_prunes
        print(f"[MCKNNAgent] ✅ Loaded ← {npz_path}  (bank_size={len(self.memory):,})")