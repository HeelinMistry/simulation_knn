"""
agents/unified_executor.py  (SAC refactor, MC-kNN temporal-exclusion patch)
─────────────────────────────────────────────────────────────────────────────
Thin execution layer that sits between the agent and the environment.

CHANGES IN THIS REVISION
───────────────────────────
step() accepts an optional `episode_id` kwarg. When supplied (training
calls main_mcknn.py's run_epoch will pass it; eval/live calls leave it
None), it is forwarded — along with the current `tick` — to
agent.select_action() as query_tick/query_episode_id, enabling
MCKNNMemory's temporal exclusion. For SACAgent this kwarg is simply
unused (SACAgent.select_action doesn't accept it... see note below).

Backward-compat note: SACAgent.select_action() does NOT have
query_tick/query_episode_id/min_tick_gap parameters. To keep this
executor usable for BOTH agent types without an isinstance check, the
call below only passes the extra kwargs when episode_id is not None
AND the agent advertises support for them (duck-typed via hasattr on
the bound method's __code__ co_varnames — see _agent_supports_temporal_args).
If the agent doesn't support them, they're silently dropped, so nothing
changes for SAC.

Everything else (position/PnL tracking, commission model, get_status())
is unchanged from the prior revision.
"""

import inspect
import torch
import numpy as np
import collections
from agents.state_aggregator import StateAggregator

COMMISSION = 0.00015  # Matches training — do not change without retraining
MAX_HOLD_TICKS = 32


def _agent_supports_temporal_args(agent) -> bool:
    try:
        params = inspect.signature(agent.select_action).parameters
        return "query_tick" in params and "query_episode_id" in params
    except (TypeError, ValueError):
        return False


class UnifiedExecutor:
    """
    Wraps an agent (SACAgent or MCKNNAgent) for step-by-step interaction
    with market data.

    Parameters
    ----------
    name        : identifier used for logging and checkpoint naming
    agent       : agent instance (already loaded/initialised)
    paces       : pace tuple forwarded to StateAggregator (must match training)
    deterministic: use argmax policy (live) vs sampled policy (training)
    """

    def __init__(
        self,
        name:          str,
        agent,
        paces: tuple = (1, 4, 16, 64),
        deterministic: bool  = False,
        num_indicators: int = 6,
    ):
        self.name          = name
        self.agent         = agent
        self.aggregator    = StateAggregator(paces, num_indicators=num_indicators)
        self.deterministic = deterministic
        self._agent_supports_temporal = _agent_supports_temporal_args(agent)

        # Position state
        self.inventory     = collections.deque(maxlen=1)
        self.current_side  = None   # "LONG" | "SHORT" | None

        # P/L tracking
        self.total_reward  = 0.0
        self.tick          = 0

        # For environment.py / diagnostics compatibility
        self.last_probs    = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._entry_tick: int = 0

    # ── State construction ────────────────────────────────────────────────────

    def portfolio_info(self, current_price: float) -> dict:
        if self.current_side is not None and self.inventory:
            entry = self.inventory[0]
            if self.current_side == "LONG":
                side_val = 1.0
                u_pnl    = (current_price - entry) / entry
            else:
                side_val = -1.0
                u_pnl    = (entry - current_price) / entry
            u_pnl = np.clip(u_pnl, -0.05, 0.05)
        else:
            side_val, u_pnl = 0.0, 0.0

        return {"position": side_val, "unrealized_pnl": u_pnl}

    def get_state(self, indicators: np.ndarray, price: float) -> np.ndarray:
        """Build and return the current state vector."""
        self.aggregator.update(indicators)
        portfolio_info = self.portfolio_info(price)
        return self.aggregator.get_state(portfolio_info)

    # ── Core step ────────────────────────────────────────────────────────────

    def _get_action_mask(self) -> torch.Tensor:
        if self.current_side is None:
            return torch.tensor([True, True, False, True])  # flat: no CLOSE
        else:
            return torch.tensor([False, False, True, True])  # in-pos: CLOSE or HOLD

    def _select_action(self, state, deterministic, action_mask, episode_id):
        """
        Calls agent.select_action(), passing query_tick/query_episode_id
        only if the agent actually supports them (MCKNNAgent does;
        SACAgent does not) and only if the caller supplied an episode_id
        (i.e. opted in to temporal exclusion — typically training calls
        only, not live/eval).
        """
        if self._agent_supports_temporal and episode_id is not None:
            return self.agent.select_action(
                state, deterministic=deterministic, action_mask=action_mask,
                query_tick=self.tick, query_episode_id=episode_id,
            )
        return self.agent.select_action(
            state, deterministic=deterministic, action_mask=action_mask,
        )

    def step(self, indicators, price, tick, epsilon=0.0, episode_id=None):
        self.tick = tick
        state = self.get_state(indicators, price)

        ATR_IDX = 4  # ATR_Scaled in the indicators array
        if self.current_side is None and abs(indicators[ATR_IDX]) > 2.0:
            self.last_probs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return 3, self.last_probs, 0.0, self.get_state(indicators, price)

        if self.current_side is not None:
            u_pnl = self.portfolio_info(price)["unrealized_pnl"]
            hold_duration = tick - self._entry_tick
            if u_pnl <= -0.015 or hold_duration >= MAX_HOLD_TICKS:
                action = 2
                probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                self.last_probs = probs
                reward = self._execute(action, price)
                self.total_reward += reward
                return action, probs, reward, state

        mask = self._get_action_mask()

        if not self.deterministic and epsilon > 0 and np.random.random() < epsilon:
            valid_actions = mask.nonzero().squeeze(-1).tolist()
            action = np.random.choice(valid_actions)
            _, probs = self._select_action(state, False, None, episode_id)
        else:
            action, probs = self._select_action(state, self.deterministic, mask, episode_id)

        self.last_probs = probs
        reward = self._execute(action, price)
        self.total_reward += reward
        return action, probs, reward, state

    # ── Execution logic ───────────────────────────────────────────────────────

    def _execute(self, action: int, price: float) -> float:
        reward = 0.0
        if action == 0:  # LONG
            if self.current_side == 'SHORT':
                reward = self._close(price)
            if self.current_side is None:
                self.inventory.append(price * (1 + COMMISSION))
                self.current_side = 'LONG'
                self._entry_tick = self.tick
        elif action == 1:  # SHORT
            if self.current_side == 'LONG':
                reward = self._close(price)
            if self.current_side is None:
                self.inventory.append(price * (1 - COMMISSION))
                self.current_side = 'SHORT'
                self._entry_tick = self.tick
        elif action == 2:
            if self.current_side is not None:
                reward = self._close(price)
        return reward

    def _close(self, price: float) -> float:
        """Close current position with exit commission. Returns net P/L."""
        if not self.inventory:
            return 0.0
        entry = self.inventory.popleft()
        if self.current_side == "LONG":
            pnl = (price * (1 - COMMISSION) - entry) / entry
        else:
            pnl = (entry - price * (1 + COMMISSION)) / entry
        self.current_side = None
        return float(pnl)

    # ── Status / compatibility ────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "position": self.current_side or "FLAT",
            "pnl":      self.total_reward,
            "pnl_str":  f"{self.total_reward:.4%}",
            "entry":    self.inventory[0] if self.inventory else None,
        }

    def reset_position(self):
        """Force-close any open position without recording P/L (e.g. end of epoch)."""
        self.inventory.clear()
        self.current_side = None