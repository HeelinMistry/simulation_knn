"""
agents/unified_executor.py  (SAC refactor, MC-kNN temporal-exclusion +
                              multi-timeframe context patch)
─────────────────────────────────────────────────────────────────────────────
Thin execution layer that sits between the agent and the environment.

CHANGES IN THIS REVISION
───────────────────────────
1. (carried over) step()/select_action() temporal-exclusion plumbing
   for MCKNNMemory — unchanged from the prior revision, see below.

2. NEW: step() accepts an optional `extra_context` vector — a
   precomputed, leakage-safe multi-timeframe context block (e.g. from
   multi_timeframe_state.build_multi_timeframe_context()) for THIS
   exact tick. When supplied, get_state() concatenates it into the
   state vector AFTER the primary aggregator's market vector and
   BEFORE the 2 portfolio features, so:
     - diagnostic_mcknn.py's `ep["states_arr"][:, feat_idx]` for
       feat_idx in range(len(FEATURES)) still correctly indexes the
       first pace's raw-indicator block (unaffected — that block's
       position hasn't moved),
     - `s_t[-1]` (unrealized PnL) and `s_t[-2]` (position side) are
       still the last two state dimensions regardless of how much
       extra context is inserted in between.
   This is purely an executor-level concatenation; all the actual
   multi-timeframe alignment/leakage-prevention work happens upstream
   in multi_timeframe_state.py — this file just plugs the result in.
   If `extra_context` is never passed, behaviour is byte-for-byte
   identical to the previous revision (4h-only state).

Backward-compat note: SACAgent.select_action() does NOT have
query_tick/query_episode_id/min_tick_gap parameters. To keep this
executor usable for BOTH agent types without an isinstance check, the
call below only passes the extra kwargs when episode_id is not None
AND the agent advertises support for them (duck-typed via
inspect.signature — see _agent_supports_temporal_args). If the agent
doesn't support them, they're silently dropped, so nothing changes
for SAC.

Everything else (position/PnL tracking, commission model, get_status())
is unchanged from the prior revision.
"""

import inspect
import numpy as np
import collections
from agents.state_aggregator import StateAggregator

COMMISSION = 0.00015  # Matches training — do not change without retraining
MAX_HOLD_TICKS = 32

# ── numpy-only note ──────────────────────────────────────────────────────────
# This executor has ZERO torch dependency end-to-end: action masks are
# plain numpy bool arrays, consumed by MCKNNMemory.query() via
# np.asarray(action_mask, dtype=bool). (A torch dependency briefly crept
# back into _get_action_mask() in an earlier edit pass — reverted here.)


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

    def get_state(self, indicators: np.ndarray, price: float,
                  extra_context: np.ndarray = None) -> np.ndarray:
        """
        Build and return the current state vector.

        Layout: [primary 4h aggregator market vector] + [extra_context,
        if provided] + [portfolio: position, unrealized_pnl].
        Portfolio features are always last regardless of extra_context,
        so any code indexing `state[-2:]` or `state[-1]` for portfolio
        info (e.g. diagnostic_mcknn.py's unrealized_pnl extraction)
        keeps working unmodified whether or not multi-timeframe context
        is in use.
        """
        self.aggregator.update(indicators)
        portfolio_info = self.portfolio_info(price)
        # StateAggregator.get_state() always appends a 2-dim portfolio
        # block (zeros if portfolio_info=None) — strip it here since we
        # build the real portfolio_vec ourselves below and need control
        # over where it sits relative to extra_context.
        market_vec = self.aggregator.get_state(portfolio_info=None)[:-2]
        portfolio_vec = np.array([
            portfolio_info["position"], portfolio_info["unrealized_pnl"],
        ], dtype=np.float32)

        if extra_context is not None:
            extra_context = np.asarray(extra_context, dtype=np.float32)
            return np.concatenate([market_vec, extra_context, portfolio_vec])
        return np.concatenate([market_vec, portfolio_vec])

    # ── Core step ────────────────────────────────────────────────────────────

    def _get_action_mask(self) -> np.ndarray:
        if self.current_side is None:
            return np.array([True, True, False, True])  # flat: no CLOSE
        else:
            return np.array([False, False, True, True])  # in-pos: CLOSE or HOLD

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

    def step(self, indicators, price, tick, epsilon=0.0, episode_id=None,
             extra_context: np.ndarray = None):
        self.tick = tick
        state = self.get_state(indicators, price, extra_context=extra_context)

        # ATR_Scaled is a rolling z-score that preprocessing.py clips to
        # [-3, 3] and then DIVIDES BY 3 (same convention as MACD_Scaled/
        # OBV_Scaled/MeanDev_Scaled), so its stored range is strictly
        # [-1, 1]. The old threshold of 2.0 was written against the
        # pre-division z-score and could never be exceeded post-rescale,
        # making this panic-exit permanently unreachable. ATR_THRESHOLD
        # is expressed in the SAME post-rescale units the indicators
        # array actually carries: 0.67 ≈ a z-score of 2.0 pre-division
        # (2.0 / 3), preserving the original "~2 std devs of ATR" intent.
        ATR_IDX = 4  # ATR_Scaled in the indicators array
        ATR_THRESHOLD = 0.67
        if self.current_side is None and abs(indicators[ATR_IDX]) > ATR_THRESHOLD:
            self.last_probs = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            # Reuse the state already computed above instead of calling
            # get_state() a second time. get_state() is NOT a pure read —
            # it drives self.aggregator.update(indicators), which
            # increments aggregator.tick and appends this tick's
            # indicators into each due pace's history. Calling it twice
            # per panic-exit tick was double-advancing aggregator.tick
            # (permanently phase-shifting which real ticks land on each
            # pace's `tick % pace == 0` sampling boundary for the rest of
            # the episode) and inserting a duplicate row into the
            # rolling window MultiPaceAgent.get_state() uses for its
            # mean/std/slope computation — silently biasing those
            # features for up to `max_history` ticks afterward.
            return 3, self.last_probs, 0.0, state

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
            valid_actions = np.flatnonzero(mask).tolist()
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