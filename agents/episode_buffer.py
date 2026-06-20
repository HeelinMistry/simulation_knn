"""
agents/episode_buffer.py
─────────────────────────
Per-episode transition collector + Monte Carlo return backfill.

Mapping vs replay_buffer.py
─────────────────────────────
  ReplayBuffer.push(s,a,r,s',done) called every tick, sampled randomly
  later for TD bootstrapping.

  EpisodeBuffer.add(s,a,r) called every tick within ONE episode; nothing
  is usable until end_episode() is called, at which point discounted
  returns G_t = sum_{k=t}^{T} gamma^(k-t) * r_k are computed for every
  t in the episode (true full-episode Monte Carlo return, per the design
  choice for this refactor) and the whole episode is committed to an
  MCKNNMemory in one batched call.

Why this can't be incremental like the old replay buffer
  G_t depends on ALL future rewards in the episode, so no (state, action)
  pair has a known return until the episode (or at minimum, the rest of
  the episode) has actually happened. This is the defining cost of true
  MC vs TD: full-episode buffering is mandatory, not a choice we could
  optimise away while keeping "true MC return" as the target.
"""

import numpy as np


class EpisodeBuffer:
    """
    Accumulates one episode's (state, action, reward) triples in plain
    Python lists (fast append, fine for episode lengths up to ~50k ticks
    which comfortably covers a multi-year 4h-candle pass), then backfills
    discounted MC returns at the end.
    """

    def __init__(self, gamma: float = 0.97):
        self.gamma = gamma
        self.states  = []
        self.actions = []
        self.rewards = []

    def add(self, state: np.ndarray, action: int, reward: float):
        self.states.append(state.astype(np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))

    def __len__(self) -> int:
        return len(self.states)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

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
        MCKNNMemory in a single batched call (mirrors how
        ReplayBuffer.push() was always a single-transition write, except
        here the "transition" is the entire episode at once).
        """
        if len(self) == 0:
            return
        states  = np.array(self.states,  dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        mc_returns = self.compute_mc_returns()
        memory.commit_episode(states, actions, mc_returns)
        self.reset()