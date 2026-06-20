"""
agents/mc_knn_memory.py
────────────────────────
Monte Carlo k-NN memory bank — the non-parametric analogue of DualCritic
in critic.py.

Mapping vs the SAC critic
──────────────────────────
  SAC critic.py            MC-kNN equivalent (this file)
  ─────────────────────────────────────────────────────────────────────
  QNetwork(state)->Q(s,a)  MCKNNMemory.query(state) -> per-action vote
  DualCritic (Q1,Q2 min)   no bootstrapping; "Q" is replaced by stored
                            ground-truth discounted returns from rollouts
  forward() one pass       k-NN search over the memory bank (brute-force
                            Euclidean — see distance metric note below)
  min_q() pessimism        no min-of-two; pessimism instead comes from
                            averaging neighbor returns rather than trusting
                            a single nearest neighbor

Why state-only storage, action as a column (not concatenated)?
  Same reasoning as the SAC critic: discrete action spaces let us store
  one state vector and tag it with whichever action was actually taken
  at that tick, plus the realised MC return for that (state, action)
  pair. A query returns a vote distribution over all 4 actions in one
  k-NN search, mirroring the critic's one-forward-pass-per-state design.

Distance metric
  Plain Euclidean distance on the raw 92-d state vector, unweighted.
  Brute-force (no KD-tree/ball-tree) — simplest and correct; the memory
  bank is bounded by periodic stratified pruning (see prune()) rather
  than an indexing structure, so brute-force kNN remains tractable.

Vote aggregation
  For each of the k nearest neighbors, weight = inverse-distance kernel
  × sign-and-magnitude of that neighbor's stored MC return. Weights are
  summed per action; the action with the highest summed weight wins.
  This is the weighted-majority-vote design (vs argmax-of-mean-return,
  vs full softmax) chosen for this refactor.
"""

import numpy as np


EPS = 1e-8


class MCKNNMemory:
    """
    Growable bank of (state, action, mc_return) triples with k-NN query
    and periodic stratified downsampling.

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
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.max_size = max_size
        self.signal_threshold = signal_threshold

        # Pre-allocate growable arrays; track a logical size separately
        # from physical capacity to avoid per-append reallocation.
        self._cap = 4096
        self._size = 0
        self.states  = np.zeros((self._cap, state_dim), dtype=np.float32)
        self.actions = np.zeros(self._cap, dtype=np.int64)
        self.returns = np.zeros(self._cap, dtype=np.float32)

        # Diagnostics
        self.n_commits = 0
        self.n_prunes  = 0

    # ── Capacity management ────────────────────────────────────────────────

    def _grow(self, min_extra: int):
        if self._size + min_extra <= self._cap:
            return
        new_cap = max(self._cap * 2, self._size + min_extra)
        new_states  = np.zeros((new_cap, self.state_dim), dtype=np.float32)
        new_actions = np.zeros(new_cap, dtype=np.int64)
        new_returns = np.zeros(new_cap, dtype=np.float32)
        new_states[:self._size]  = self.states[:self._size]
        new_actions[:self._size] = self.actions[:self._size]
        new_returns[:self._size] = self.returns[:self._size]
        self.states, self.actions, self.returns = new_states, new_actions, new_returns
        self._cap = new_cap

    def __len__(self) -> int:
        return self._size

    # ── Commit: bulk-insert a full episode's worth of (s, a, G) ─────────────

    def commit_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        mc_returns: np.ndarray,
    ):
        """
        Append an entire episode's backfilled (state, action, G_t) triples
        in one call. This is the kNN analogue of replay_buffer.push() but
        operates on whole episodes since full-episode MC returns require
        the episode to have already finished and been backfilled (see
        episode_buffer.py).
        """
        n = len(states)
        if n == 0:
            return
        self._grow(n)
        s = slice(self._size, self._size + n)
        self.states[s]  = states.astype(np.float32)
        self.actions[s] = actions.astype(np.int64)
        self.returns[s] = mc_returns.astype(np.float32)
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

        # Target composition: up to half signal, remainder noise — same
        # 50/50 signal/noise spirit as ReplayBuffer.sample()'s n_signal /
        # n_noise split, but applied to what we KEEP rather than what we
        # sample for a batch.
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

        self.states[:len(keep_idx)]  = self.states[keep_idx]
        self.actions[:len(keep_idx)] = self.actions[keep_idx]
        self.returns[:len(keep_idx)] = self.returns[keep_idx]
        self._size = len(keep_idx)
        self.n_prunes += 1

    # ── k-NN query ────────────────────────────────────────────────────────

    def query(self, state: np.ndarray, k: int = None, action_mask=None):
        """
        Find the k nearest neighbors (plain Euclidean, unweighted) to
        `state` and return a weighted-majority-vote distribution over
        actions plus diagnostic info.

        Returns
        -------
        action      : int, the winning action (highest vote weight)
        vote_probs  : np.ndarray (action_dim,), vote-share per action
                      (sums to 1; NOT a calibrated softmax — see note in
                      mc_knn_policy.py). Populates the same `probs` slot
                      that Actor.act() used to fill, so live.py's
                      probability bars keep working unmodified.
        info        : dict with neighbor_returns, neighbor_actions,
                      neighbor_dists, vote_margin (winner minus runner-up
                      weight, used by diagnostics as the kNN analogue of
                      critic disagreement)
        """
        if k is None:
            k = self.k
        if self._size == 0:
            # No data yet — caller should fall back to a safe default
            # (HOLD), exactly as the cold-start case for SAC's actor
            # before any training would also be near-uniform.
            probs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return 3, probs, {
                "neighbor_returns": np.array([]),
                "neighbor_actions": np.array([]),
                "neighbor_dists":   np.array([]),
                "vote_margin": 0.0,
            }

        k = min(k, self._size)
        diffs = self.states[:self._size] - state[None, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))

        nn_idx = np.argpartition(dists, k - 1)[:k]
        # sort the k by distance (argpartition doesn't guarantee order)
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]

        nn_dists   = dists[nn_idx]
        nn_actions = self.actions[nn_idx]
        nn_returns = self.returns[nn_idx]

        # Inverse-distance kernel weight; EPS_DIST is a floor (not just an
        # anti-div-by-zero epsilon) so a near-duplicate neighbor doesn't
        # produce an exploding weight that swamps every other neighbor's
        # vote. Without this floor, vote_margin can blow up to ~1/EPS
        # whenever a live state nearly matches a stored one (e.g. a flat
        # market repeating indicator values) — this was caught by an
        # end-to-end smoke test producing vote_margin ~6e8 on synthetic
        # data with repeated near-identical states.
        EPS_DIST = 1e-3
        kernel_w = 1.0 / (nn_dists + EPS_DIST)
        vote_weight = kernel_w * nn_returns   # signed: losing trades vote AGAINST that action

        vote = np.zeros(self.action_dim, dtype=np.float64)
        for a in range(self.action_dim):
            mask = nn_actions == a
            if mask.any():
                vote[a] = vote_weight[mask].sum()

        if action_mask is not None:
            # action_mask is a bool tensor/array, True = valid.
            invalid = ~np.asarray(action_mask, dtype=bool)
            vote[invalid] = -np.inf

        # Convert raw signed vote weights into a non-negative vote-share
        # distribution for display purposes (mirrors probs shape from
        # Actor.act()) without claiming probabilistic calibration.
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
        }
        return action, vote_probs.astype(np.float32), info

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        np.savez_compressed(
            path,
            states=self.states[:self._size],
            actions=self.actions[:self._size],
            returns=self.returns[:self._size],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            k=self.k,
            max_size=self.max_size,
            signal_threshold=self.signal_threshold,
            n_commits=self.n_commits,
            n_prunes=self.n_prunes,
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
        )
        n = len(data["states"])
        mem._grow(n)
        mem.states[:n]  = data["states"]
        mem.actions[:n] = data["actions"]
        mem.returns[:n] = data["returns"]
        mem._size = n
        mem.n_commits = int(data["n_commits"])
        mem.n_prunes  = int(data["n_prunes"])
        return mem