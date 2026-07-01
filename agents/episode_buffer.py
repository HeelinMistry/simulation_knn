"""
agents/episode_buffer.py
─────────────────────────
Per-episode transition collector + Monte Carlo return backfill.

CHANGES IN THIS REVISION
───────────────────────────
add() now optionally takes a `tick` (the absolute tick index in the
underlying DataFrame, e.g. main_mcknn.py's loop variable `i`) and the
buffer is tagged with an `episode_id`. Both are threaded through to
MCKNNMemory.commit_episode() so the memory bank can apply temporal
exclusion at query time (see mc_knn_memory.py docstring) — i.e. a
query made while replaying this same historical pass cannot retrieve
a neighbor that is really "itself" from a handful of ticks earlier,
which was the dominant cause of the train/val performance gap found
in the integrity check (near-zero nearest-neighbor distances in
pre_training.py's diagnostic + repeated multi-epoch passes over the
same fixed historical sequence).

If `tick` is never supplied to add(), behaviour is unchanged from the
previous revision (positional 0..n-1 ticks, single fallback episode_id) —
fully backward compatible.

Why this still can't be incremental like the old replay buffer
  G_t depends on ALL future rewards in the episode, so no (state, action)
  pair has a known return until the episode (or at minimum, the rest of
  the episode) has actually happened. This is the defining cost of true
  MC vs TD: full-episode buffering is mandatory, not a choice we could
  optimise away while keeping "true MC return" as the target.
"""

import numpy as np


class EpisodeBuffer:
    """
    Accumulates one episode's (state, action, reward, tick) rows in
    plain Python lists (fast append, fine for episode lengths up to
    ~50k ticks which comfortably covers a multi-year 4h-candle pass),
    then backfills discounted MC returns at the end.
    """

    _next_episode_id = 0  # class-level counter shared across instances

    def __init__(self, gamma: float = 0.97, episode_id: int = None):
        self.gamma = gamma
        self.states  = []
        self.actions = []
        self.rewards = []
        self.ticks   = []

        if episode_id is None:
            episode_id = EpisodeBuffer._next_episode_id
            EpisodeBuffer._next_episode_id += 1
        self.episode_id = episode_id

    def add(self, state: np.ndarray, action: int, reward: float, tick: int = None):
        self.states.append(state.astype(np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        # Fallback: positional tick if caller didn't supply one — keeps
        # old call sites working, and temporal exclusion still functions
        # correctly under this fallback since rows are always appended
        # in tick order.
        self.ticks.append(int(tick) if tick is not None else len(self.ticks))

    def __len__(self) -> int:
        return len(self.states)

    def reset(self, new_episode_id: bool = True):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.ticks.clear()
        if new_episode_id:
            self.episode_id = EpisodeBuffer._next_episode_id
            EpisodeBuffer._next_episode_id += 1

    def compute_mc_returns(self) -> np.ndarray:
        """
        Backfill G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        for every t, working backwards from the end of the episode.
        This is the textbook full-episode Monte Carlo return — no
        bootstrapped value estimate is used anywhere in this computation,
        unlike SAC's TD target in sac_agent.py's update().
        """
        n = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = self.rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    def end_episode_and_commit(self, memory):
        """
        Backfill MC returns and commit the whole episode to an
        MCKNNMemory in a single batched call, including this episode's
        tick indices and episode_id so the memory's query() can apply
        temporal exclusion against rows committed from this same pass.
        """
        if len(self) == 0:
            return
        states  = np.array(self.states,  dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        ticks   = np.array(self.ticks,   dtype=np.int64)
        mc_returns = self.compute_mc_returns()
        memory.commit_episode(
            states, actions, mc_returns,
            ticks=ticks, episode_id=self.episode_id,
        )
        # IMPORTANT: keep the SAME episode_id across repeated commits from
        # a long-lived buffer (e.g. main_mcknn.py's single `episode_buffer`
        # reused every epoch for the train split). If a new id were assigned
        # here, epoch N+1's query at tick i would have a different
        # episode_id than epoch N's committed row at the same tick i, and
        # temporal exclusion would silently fail to catch that cross-epoch
        # near-duplicate — exactly the leak the integrity check flagged.
        # Callers that genuinely want a fresh, unrelated episode (e.g. a
        # one-off diagnostic episode) should call
        # buffer.reset(new_episode_id=True) explicitly instead of relying
        # on this method.
        self.reset(new_episode_id=False)