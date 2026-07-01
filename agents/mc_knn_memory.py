"""
agents/mc_knn_memory.py
────────────────────────
Monte Carlo k-NN memory bank — the non-parametric analogue of DualCritic
in critic.py.

CHANGES IN THIS REVISION (fixes integrity-check findings)
───────────────────────────────────────────────────────────
1. TEMPORAL EXCLUSION (the main fix)
   pre_training.py's diagnostic showed median nearest-neighbor distance
   of 0.0 across the indicator-derived state stream — i.e. many states
   are near-duplicates of other states close in time (flat/low-vol
   stretches, slow-pace (42/90) windows barely moving tick-to-tick).
   Combined with multi-epoch training over the same fixed historical
   sequence, this meant a query at tick T could retrieve a neighbor
   that is effectively "itself" from a few ticks away / a previous
   epoch's pass over the same period — leaking the future outcome of
   (almost) the same moment back into its own vote. This explains the
   ~20-30x gap between train P/L (+2,348%) and val P/L (+79-135%).

   Fix: every stored row now carries a `tick` (and `episode_id`).
   query() accepts an optional `query_tick` / `query_episode_id` and
   `min_tick_gap`; any stored row from the same episode whose tick is
   within `min_tick_gap` of the query tick is excluded from voting.
   This is opt-in (defaults preserve old behaviour when tick info is
   not supplied) but commit_episode/query call sites have been updated
   to always supply it — see episode_buffer.py and diagnostic_mcknn.py.

2. CAPPED KERNEL WEIGHT (secondary fix)
   The old inverse-distance kernel `1/(dist+EPS_DIST)` with
   EPS_DIST=1e-3 let a single near-duplicate neighbor (dist≈0) produce
   a weight ~1000x larger than a "normal" 10th-nearest neighbor
   (median dist ≈ 3.4 in the pre-training diagnostic), i.e. roughly a
   3,000:1 weight ratio for one vote vs the rest combined. Even with
   temporal exclusion in place as the primary defence, this is capped
   independently so a single neighbor — temporally distant or not —
   can never dominate the vote by more than MAX_WEIGHT_RATIO.

3. Distance metric, vote aggregation, pruning: unchanged.
"""

import numpy as np


EPS = 1e-8


class MCKNNMemory:
    """
    Growable bank of (state, action, mc_return, tick, episode_id) rows
    with k-NN query, temporal-exclusion, capped-weight voting, and
    periodic stratified downsampling.

    Unlike DualCritic this holds no learnable parameters — "training"
    means appending more (state, action, return) triples and occasionally
    pruning, never gradient descent.
    """

    def __init__(
        self,
        state_dim: int = 50,
        action_dim: int = 4,
        k: int = 25,
        max_size: int = 200_000,
        signal_threshold: float = 0.0005,
        eps_dist: float = 1e-3,
        max_weight_ratio: float = 50.0,
        min_tick_gap: int = 0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.max_size = max_size
        self.signal_threshold = signal_threshold

        # ── Kernel-weight cap config ────────────────────────────────────
        # EPS_DIST is still a floor (anti-div-by-zero), but the dominant
        # protection against any single neighbor swamping the vote is now
        # max_weight_ratio: after computing raw inverse-distance kernel
        # weights, we clip every weight to at most
        # max_weight_ratio * median(weight) for that query, so a
        # dist≈0 duplicate can outvote a "normal" neighbor by at most
        # max_weight_ratio:1, not ~3,000:1.
        self.eps_dist = eps_dist
        self.max_weight_ratio = max_weight_ratio

        # ── Temporal-exclusion default (can be overridden per query) ────
        self.min_tick_gap = min_tick_gap

        # Pre-allocate growable arrays; track a logical size separately
        # from physical capacity to avoid per-append reallocation.
        self._cap = 4096
        self._size = 0
        self.states     = np.zeros((self._cap, state_dim), dtype=np.float32)
        self.actions    = np.zeros(self._cap, dtype=np.int64)
        self.returns    = np.zeros(self._cap, dtype=np.float32)
        self.ticks      = np.zeros(self._cap, dtype=np.int64)
        self.episode_id = np.zeros(self._cap, dtype=np.int64)

        # Diagnostics
        self.n_commits = 0
        self.n_prunes  = 0
        self._next_episode_id = 0

    # ── Capacity management ────────────────────────────────────────────────

    def _grow(self, min_extra: int):
        if self._size + min_extra <= self._cap:
            return
        new_cap = max(self._cap * 2, self._size + min_extra)
        new_states     = np.zeros((new_cap, self.state_dim), dtype=np.float32)
        new_actions    = np.zeros(new_cap, dtype=np.int64)
        new_returns    = np.zeros(new_cap, dtype=np.float32)
        new_ticks      = np.zeros(new_cap, dtype=np.int64)
        new_episode_id = np.zeros(new_cap, dtype=np.int64)
        new_states[:self._size]     = self.states[:self._size]
        new_actions[:self._size]    = self.actions[:self._size]
        new_returns[:self._size]    = self.returns[:self._size]
        new_ticks[:self._size]      = self.ticks[:self._size]
        new_episode_id[:self._size] = self.episode_id[:self._size]
        self.states, self.actions, self.returns = new_states, new_actions, new_returns
        self.ticks, self.episode_id = new_ticks, new_episode_id
        self._cap = new_cap

    def __len__(self) -> int:
        return self._size

    # ── Commit: bulk-insert a full episode's worth of (s, a, G, tick) ───────

    def commit_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        mc_returns: np.ndarray,
        ticks: np.ndarray = None,
        episode_id: int = None,
    ):
        """
        Append an entire episode's backfilled (state, action, G_t, tick)
        rows in one call.

        ticks       : per-row tick index within the episode. If None,
                      falls back to 0..n-1 (positional) — temporal
                      exclusion still works in this fallback, it just
                      assumes rows were committed in tick order, which
                      is always true since EpisodeBuffer.add() appends
                      sequentially.
        episode_id  : integer episode tag. If None, an auto-incrementing
                      id is assigned so two different commit_episode()
                      calls are never treated as the same episode
                      (important: cross-episode temporal exclusion is
                      NOT intended — only "this happened a few ticks
                      ago in the SAME pass" should be excluded; states
                      from a different historical period that happen to
                      be numerically similar are legitimate signal).
        """
        n = len(states)
        if n == 0:
            return
        if episode_id is None:
            episode_id = self._next_episode_id
        self._next_episode_id = max(self._next_episode_id, episode_id + 1)

        if ticks is None:
            ticks = np.arange(n, dtype=np.int64)

        self._grow(n)
        s = slice(self._size, self._size + n)
        self.states[s]     = states.astype(np.float32)
        self.actions[s]    = actions.astype(np.int64)
        self.returns[s]    = mc_returns.astype(np.float32)
        self.ticks[s]      = np.asarray(ticks, dtype=np.int64)
        self.episode_id[s] = episode_id
        self._size += n
        self.n_commits += 1

        if self._size > self.max_size:
            self.prune(target_size=self.max_size)

    # ── Stratified pruning (downsample while keeping signal/noise ratio) ────

    def prune(self, target_size: int = None):
        """
        Keep a stratified sample of the bank: signal rows (|G| above
        signal_threshold) are preferentially retained over noise rows
        (|G| below threshold), mirroring the SIGNAL_THRESHOLD split used
        in replay_buffer.py's sample(). This keeps the bank bounded
        without an explicit circular-buffer maxlen, per the periodic
        stratified downsampling design choice for this refactor.
        """
        if target_size is None:
            target_size = self.max_size
        if self._size <= target_size:
            return

        abs_ret = np.abs(self.returns[:self._size])
        signal_mask = abs_ret > self.signal_threshold
        signal_idx = np.flatnonzero(signal_mask)
        noise_idx  = np.flatnonzero(~signal_mask)

        n_signal_keep = min(len(signal_idx), target_size // 2)
        n_noise_keep  = target_size - n_signal_keep

        rng = np.random.default_rng()
        keep_signal = rng.choice(signal_idx, size=n_signal_keep, replace=False) \
            if n_signal_keep > 0 else np.array([], dtype=np.int64)
        pool_noise = noise_idx if len(noise_idx) >= n_noise_keep else np.arange(self._size)
        n_noise_keep = min(n_noise_keep, len(pool_noise))
        keep_noise = rng.choice(pool_noise, size=n_noise_keep, replace=False) \
            if n_noise_keep > 0 else np.array([], dtype=np.int64)

        keep_idx = np.concatenate([keep_signal, keep_noise])
        keep_idx.sort()

        self.states[:len(keep_idx)]     = self.states[keep_idx]
        self.actions[:len(keep_idx)]    = self.actions[keep_idx]
        self.returns[:len(keep_idx)]    = self.returns[keep_idx]
        self.ticks[:len(keep_idx)]      = self.ticks[keep_idx]
        self.episode_id[:len(keep_idx)] = self.episode_id[keep_idx]
        self._size = len(keep_idx)
        self.n_prunes += 1

    # ── k-NN query ────────────────────────────────────────────────────────

    def query(self, state: np.ndarray, k: int = None, action_mask=None,
              query_tick: int = None, query_episode_id: int = None,
              min_tick_gap: int = None):
        """
        Find the k nearest neighbors (plain Euclidean, unweighted) to
        `state`, EXCLUDING any stored row from the same episode whose
        tick lies within `min_tick_gap` of `query_tick` (temporal
        exclusion — see module docstring), then return a capped-weight
        majority-vote distribution over actions plus diagnostic info.

        Parameters
        ----------
        query_tick, query_episode_id, min_tick_gap
            Optional. If query_tick/query_episode_id are None, no
            temporal exclusion is applied (back-compatible with old
            callers). min_tick_gap defaults to self.min_tick_gap when
            not supplied.

        Returns
        -------
        action      : int, the winning action (highest vote weight)
        vote_probs  : np.ndarray (action_dim,), vote-share per action
                      (sums to 1; NOT a calibrated softmax).
        info        : dict with neighbor_returns, neighbor_actions,
                      neighbor_dists, vote_margin, vote_raw,
                      n_excluded_temporal (diagnostic: how many bank
                      rows were excluded by the temporal filter — useful
                      for spotting whether the filter is actually doing
                      anything on a given dataset).
        """
        if k is None:
            k = self.k
        if min_tick_gap is None:
            min_tick_gap = self.min_tick_gap

        if self._size == 0:
            probs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return 3, probs, {
                "neighbor_returns": np.array([]),
                "neighbor_actions": np.array([]),
                "neighbor_dists":   np.array([]),
                "vote_margin": 0.0,
                "vote_raw": np.zeros(self.action_dim),
                "n_excluded_temporal": 0,
            }

        # ── Build the candidate pool, applying temporal exclusion ────────
        valid = np.ones(self._size, dtype=bool)
        n_excluded = 0
        if (query_tick is not None and query_episode_id is not None
                and min_tick_gap > 0):
            same_ep = self.episode_id[:self._size] == query_episode_id
            too_close = np.abs(self.ticks[:self._size] - query_tick) < min_tick_gap
            excluded = same_ep & too_close
            n_excluded = int(excluded.sum())
            valid &= ~excluded

        cand_idx_all = np.flatnonzero(valid)
        if len(cand_idx_all) == 0:
            # Degenerate case: everything excluded (e.g. tiny bank +
            # large min_tick_gap) — fall back to using the full bank
            # rather than returning a meaningless cold-start HOLD.
            cand_idx_all = np.arange(self._size)
            n_excluded = 0

        k_eff = min(k, len(cand_idx_all))
        cand_states = self.states[cand_idx_all]
        diffs = cand_states - state[None, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))

        nn_local = np.argpartition(dists, k_eff - 1)[:k_eff]
        nn_local = nn_local[np.argsort(dists[nn_local])]
        nn_idx   = cand_idx_all[nn_local]

        nn_dists   = dists[nn_local]
        nn_actions = self.actions[nn_idx]
        nn_returns = self.returns[nn_idx]

        # ── Inverse-distance kernel with a capped weight ratio ───────────
        # Raw kernel weight (floor still present for numerical safety),
        # then clipped so no single neighbor's |weight| can exceed
        # max_weight_ratio times the median |weight| among this query's
        # neighbors. This bounds the influence of any one neighbor —
        # whether it's a near-duplicate from temporal leakage we failed
        # to exclude, or a legitimate but unusually close historical
        # match — to a sane multiple of a "typical" neighbor's say.
        kernel_w = 1.0 / (nn_dists + self.eps_dist)
        med_w = np.median(kernel_w) + EPS
        cap = self.max_weight_ratio * med_w
        kernel_w = np.minimum(kernel_w, cap)

        vote_weight = kernel_w * nn_returns   # signed: losing trades vote AGAINST that action

        vote = np.zeros(self.action_dim, dtype=np.float64)
        for a in range(self.action_dim):
            mask = nn_actions == a
            if mask.any():
                vote[a] = vote_weight[mask].sum()

        if action_mask is not None:
            invalid = ~np.asarray(action_mask, dtype=bool)
            vote[invalid] = -np.inf

        finite_vote = np.where(np.isfinite(vote), vote, 0.0)
        shifted = finite_vote - finite_vote.min() + EPS
        if action_mask is not None:
            shifted[invalid] = 0.0
        total = shifted.sum()
        vote_probs = (shifted / total) if total > 0 else np.full(
            self.action_dim, 1.0 / self.action_dim, dtype=np.float32
        )

        action = int(np.argmax(vote))
        sorted_vote = np.sort(vote[np.isfinite(vote)])[::-1] if np.isfinite(vote).any() else np.array([0.0])
        vote_margin = float(sorted_vote[0] - sorted_vote[1]) if len(sorted_vote) > 1 else float(sorted_vote[0])

        info = {
            "neighbor_returns": nn_returns,
            "neighbor_actions": nn_actions,
            "neighbor_dists":   nn_dists,
            "vote_margin":      vote_margin,
            "vote_raw":         vote,
            "n_excluded_temporal": n_excluded,
        }
        return action, vote_probs.astype(np.float32), info

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        np.savez_compressed(
            path,
            states=self.states[:self._size],
            actions=self.actions[:self._size],
            returns=self.returns[:self._size],
            ticks=self.ticks[:self._size],
            episode_id=self.episode_id[:self._size],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            k=self.k,
            max_size=self.max_size,
            signal_threshold=self.signal_threshold,
            eps_dist=self.eps_dist,
            max_weight_ratio=self.max_weight_ratio,
            min_tick_gap=self.min_tick_gap,
            n_commits=self.n_commits,
            n_prunes=self.n_prunes,
            next_episode_id=self._next_episode_id,
        )

    @classmethod
    def load(cls, path: str) -> "MCKNNMemory":
        data = np.load(path, allow_pickle=False)
        mem = cls(
            state_dim=int(data["state_dim"]),
            action_dim=int(data["action_dim"]),
            k=int(data["k"]),
            max_size=int(data["max_size"]),
            signal_threshold=float(data["signal_threshold"]),
            eps_dist=float(data["eps_dist"]) if "eps_dist" in data else 1e-3,
            max_weight_ratio=float(data["max_weight_ratio"]) if "max_weight_ratio" in data else 50.0,
            min_tick_gap=int(data["min_tick_gap"]) if "min_tick_gap" in data else 0,
        )
        n = len(data["states"])
        mem._grow(n)
        mem.states[:n]  = data["states"]
        mem.actions[:n] = data["actions"]
        mem.returns[:n] = data["returns"]
        if "ticks" in data:
            mem.ticks[:n] = data["ticks"]
        else:
            # Loading an old checkpoint saved before this revision —
            # we have no real tick info, so temporal exclusion simply
            # won't fire for this legacy data (min_tick_gap default 0
            # at query time keeps behaviour identical until retrained).
            mem.ticks[:n] = 0
        if "episode_id" in data:
            mem.episode_id[:n] = data["episode_id"]
        else:
            mem.episode_id[:n] = -1  # never matches any live query_episode_id
        mem._size = n
        mem.n_commits = int(data["n_commits"])
        mem.n_prunes  = int(data["n_prunes"])
        mem._next_episode_id = int(data["next_episode_id"]) if "next_episode_id" in data else 0
        return mem