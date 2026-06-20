"""
main_mcknn.py
─────────────
Headless training loop for the Monte Carlo k-NN trading agent.

Structural diff vs main_sac.py
────────────────────────────────
  KEPT IDENTICAL: candle timeframe, epoch structure (one epoch = one full
    pass through train_df), regime-balanced val split (2022 bear + 2025
    mixed), directional-collapse / bear-floor saving criteria, SHORT
    collapse hard-stop, early-stopping patience logic, epoch print format,
    one-step-lag reward bookkeeping pattern.

  CHANGED: ReplayBuffer -> EpisodeBuffer (per-episode, not per-step).
  CHANGED: agent.update(replay_buffer, batch_size) called every
    UPDATE_EVERY ticks  →  agent.update(episode_buffer) called ONCE per
    epoch, after the full pass, because true full-episode MC returns
    can't be computed mid-episode (see episode_buffer.py docstring).
  CHANGED: UPDATES_PER_STEP, BATCH_SIZE, LR, TAU, lr_actor/lr_critic,
    target_entropy — all removed; nothing to tune since there's no
    gradient descent. Replaced by K (neighbors) and BUFFER_CAPACITY
    (memory bank max_size) as the new tunable knobs.
  CHANGED: epsilon-greedy schedule kept (EPSILON_START/END/DECAY) since
    unified_executor.py's epsilon branch is unchanged and still useful
    for diversifying which (state, action) pairs get stored in the bank
    during training, even though there's no entropy-driven exploration
    pressure anymore.
  REMOVED: agent.last_alpha / last_entropy / last_critic_loss /
    last_actor_loss prints — replaced with agent.last_bank_size /
    n_episodes_committed / last_n_prunes, the kNN-analogue diagnostics.

Year-boundary safety, reward shaping (compute_shaped_reward) — identical
to main_sac.py, copied verbatim, since both are algorithm-agnostic.
"""

import time

import numpy as np
import pandas as pd

from agents.episode_buffer import EpisodeBuffer
from agents.mc_knn_agent import MCKNNAgent
from agents.unified_executor import UnifiedExecutor
from data.data_manager import update_master_data

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MASTER_CSV       = "data/processed/XRPUSDT_master_processed.csv"
CHECKPOINT_PATH  = "outcomes/mc_knn_agent.npz"
BEST_PATH        = "outcomes/mc_knn_agent_best.npz"

FEATURES   = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
              "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES      = (1, 6, 42, 90)
STATE_DIM  = (len(FEATURES) * 2 * len(PACES)) + 2
ACTION_DIM = 4

# ── Regime-balanced split — must match diagnostic_mcknn.py exactly ───────────
VAL_YEARS   = [2022, 2025]   # bear + recent out-of-sample
BEAR_YEAR   = 2022

# Training hyperparameters
NUM_EPOCHS       = 100
WARMUP_IDX       = 128          # aggregator warm-up rows (= max_pace × max_history)

PATIENCE         = 1            # epochs without improvement before stopping
WARMUP_EPOCHS    = 2
MIN_IMPROVE      = 0.005        # val PnL must improve by 0.5 pp to reset patience

# ── MC-kNN specific knobs (replace BUFFER_CAPACITY/BATCH_SIZE/LR/GAMMA/TAU
#    tuning surface from SAC; gamma is kept since MC returns still discount) ──
BANK_MAX_SIZE    = 200_000      # memory bank cap before stratified pruning kicks in
K_NEIGHBORS      = 25
GAMMA            = 0.97
SIGNAL_THRESHOLD = 0.0005       # matches replay_buffer.py's SIGNAL_THRESHOLD

# Logging
LOG_EVERY_TICKS  = 500
SAVE_EVERY_EPOCH = 5

# ── Reward shaping ────────────────────────────────────────────────────────────
# REWARD_SCALE is kept even though there's no critic to keep in a 2-10
# range — MC returns are stored and queried in the same units as pushed,
# so scaling still affects vote-weight magnitudes during k-NN aggregation.
# Kept identical to main_sac.py so the two algorithms see the same reward
# signal and any performance gap is attributable to the algorithm, not
# the reward design.
REWARD_SCALE    = 100.0
MICRO_HOLD_COST = 0.000005   # 0.5 bp/tick while in position (variance signal)
EPSILON_START   = 0.10
EPSILON_END     = 0.01


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compute_shaped_reward(realised_pnl: float,
                          is_holding: bool,
                          in_position: bool) -> float:
    """Identical to main_sac.py's compute_shaped_reward — copied verbatim."""
    shaped = realised_pnl

    if realised_pnl > 0.001:
        shaped += realised_pnl * 0.1

    if in_position and is_holding:
        shaped -= MICRO_HOLD_COST

    return shaped


def run_epoch(executor: UnifiedExecutor, df: pd.DataFrame,
              episode_buffer: EpisodeBuffer, agent: MCKNNAgent,
              train: bool = True, epsilon: float = 0.0) -> dict:
    """
    Single deterministic or stochastic pass through df.

    train=True  : push scaled rewards to the episode buffer, then commit
                  the whole episode's backfilled MC returns to the memory
                  bank via agent.update() AFTER this function returns
                  (caller's responsibility — see main(), mirrors how
                  main_sac.py called agent.update() mid-loop but here
                  it must happen post-loop).
    train=False : evaluation only — no buffer writes, no bank commits.

    Returns the same metrics dict shape as main_sac.py's run_epoch:
        realised_pnl, n_trades, action_counts, update_count
    update_count is always 0 here (no per-tick updates exist); kept in
    the dict so the print/report code downstream needs no changes.
    """
    indicators_arr = df[FEATURES].values.astype(np.float32)
    prices_arr     = df["Close"].values.astype(np.float32)
    n              = len(df)

    # ── Reset executor state ──────────────────────────────────────────────────
    executor.total_reward  = 0.0
    executor.inventory.clear()
    executor.current_side  = None
    executor._entry_tick   = 0

    # Warm up aggregator
    executor.aggregator.tick = 0
    executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)

    total_realised = 0.0
    n_trades       = 0
    action_counts  = [0, 0, 0, 0]

    # One-step lag: identical pattern to main_sac.py — we push
    # (prev_state, prev_action, prev_reward) once we know it, so the
    # reward correctly belongs to the action that earned it.
    prev_state  = executor.aggregator.get_state(
        executor.portfolio_info(prices_arr[WARMUP_IDX])
    )
    prev_action = None
    prev_reward = 0.0   # unscaled; scaled at push time

    for i in range(WARMUP_IDX + 1, n):
        indicators = indicators_arr[i]
        price      = prices_arr[i]

        action, probs, realised_pnl, s_t = executor.step(
            indicators, price, tick=i, epsilon=epsilon
        )

        action_counts[action] += 1

        if realised_pnl != 0.0:
            total_realised += realised_pnl
            n_trades       += 1

        reward = compute_shaped_reward(
            realised_pnl,
            is_holding=(executor.current_side is not None and realised_pnl == 0.0),
            in_position=(executor.current_side is not None),
        )

        if train and prev_action is not None:
            episode_buffer.add(
                prev_state, prev_action,
                prev_reward * REWARD_SCALE,   # scaled, same convention as main_sac.py
            )

        prev_state  = s_t
        prev_action = action
        prev_reward = reward   # carry unscaled; scaled at next push

        # ── Console heartbeat (training only) ────────────────────────────────
        if train and (i % LOG_EVERY_TICKS == 0):
            total_ticks = max(i - WARMUP_IDX, 1)
            pct = ", ".join(f"{c/total_ticks:.0%}" for c in action_counts)
            print(
                f"  tick {i:>6}  |  realised P/L: {total_realised:+.4%}"
                f"  | trades: {n_trades}"
                f"  | actions [L/S/C/H]: {pct}"
                f"  | bank={len(agent.memory):,}"
            )

    return {
        "realised_pnl":  total_realised,
        "n_trades":      n_trades,
        "action_counts": action_counts,
        "update_count":  0,   # no per-tick updates for MC-kNN
    }


def _combine_metrics(m1: dict, m2: dict) -> dict:
    """Identical to main_sac.py — copied verbatim."""
    return {
        "realised_pnl":  m1["realised_pnl"]  + m2["realised_pnl"],
        "n_trades":      m1["n_trades"]       + m2["n_trades"],
        "action_counts": [a + b for a, b in
                          zip(m1["action_counts"], m2["action_counts"])],
        "update_count":  m1["update_count"]   + m2["update_count"],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    df = update_master_data()

    # ── Load and split data ───────────────────────────────────────────────────
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    df["year"] = pd.to_datetime(df["Open_time"]).dt.year

    val_mask   = df["year"].isin(VAL_YEARS)
    train_mask = ~val_mask

    train_df   = df[train_mask].reset_index(drop=True)
    val_2022_df = df[df["year"] == 2022].reset_index(drop=True)
    val_2025_df = df[df["year"] == 2025].reset_index(drop=True)
    bear_df     = val_2022_df   # 2022 is both val-bear and the bear monitor

    print(f"Train: {len(train_df):,} rows  |  "
          f"Val 2022: {len(val_2022_df):,}  |  Val 2025: {len(val_2025_df):,}")
    print(f"Train years: {sorted(df[train_mask]['year'].unique().tolist())}")
    print(f"Val years:   {VAL_YEARS}")

    # ── Initialise agent and episode buffer ───────────────────────────────────
    agent = MCKNNAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        k=K_NEIGHBORS, max_size=BANK_MAX_SIZE, gamma=GAMMA,
        signal_threshold=SIGNAL_THRESHOLD,
    )
    agent.load(CHECKPOINT_PATH)

    episode_buffer = EpisodeBuffer(gamma=GAMMA)

    train_executor = UnifiedExecutor(
        "Train", agent, paces=PACES,
        deterministic=False, num_indicators=len(FEATURES)
    )

    best_val_pnl = -np.inf
    no_improve   = 0
    EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / 20)
    epsilon       = EPSILON_START

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ── Training pass: collect the episode, THEN commit MC returns ────────
        train_metrics = run_epoch(
            train_executor, train_df, episode_buffer, agent,
            train=True, epsilon=epsilon,
        )
        # Full-episode MC backfill + commit to the memory bank — this is the
        # one place the control flow structurally differs from main_sac.py,
        # which called agent.update() mid-epoch every UPDATE_EVERY ticks.
        agent.update(episode_buffer)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # ── Val: two separate episodes to avoid year-boundary phantom trades ──
        val_exec_2022 = UnifiedExecutor(
            "Val2022", agent, paces=PACES,
            deterministic=True, num_indicators=len(FEATURES)
        )
        val_exec_2025 = UnifiedExecutor(
            "Val2025", agent, paces=PACES,
            deterministic=True, num_indicators=len(FEATURES)
        )
        # Val passes use a throwaway episode_buffer since train=False means
        # nothing gets added to it anyway — kept for run_epoch's signature.
        _val_buf = EpisodeBuffer(gamma=GAMMA)
        m_2022 = run_epoch(val_exec_2022, val_2022_df, _val_buf,
                           agent, train=False)
        m_2025 = run_epoch(val_exec_2025, val_2025_df, _val_buf,
                           agent, train=False)
        val_metrics = _combine_metrics(m_2022, m_2025)

        # ── Bear health monitor (same as val 2022, re-use result) ─────────────
        b_pnl       = m_2022["realised_pnl"]
        b_short_pct = (m_2022["action_counts"][1] /
                       max(sum(m_2022["action_counts"]), 1))

        # ── Epoch reporting ───────────────────────────────────────────────────
        elapsed = time.time() - t0
        t_pnl   = train_metrics["realised_pnl"]
        v_pnl   = val_metrics["realised_pnl"]
        t_ac    = train_metrics["action_counts"]
        v_ac    = val_metrics["action_counts"]

        def pct_str(counts):
            total = max(sum(counts), 1)
            return "/".join(f"{c/total:.0%}" for c in counts)

        print(f"\n  ── Epoch {epoch} ─────────────────────────────────────────")
        print(f"  Train P/L : {t_pnl:+.4%}  |  trades={train_metrics['n_trades']}")
        print(f"  Val 2022  : {m_2022['realised_pnl']:+.4%}"
              f"  |  trades={m_2022['n_trades']}")
        print(f"  Val 2025  : {m_2025['realised_pnl']:+.4%}"
              f"  |  trades={m_2025['n_trades']}")
        print(f"  Val total : {v_pnl:+.4%}  |  trades={val_metrics['n_trades']}")
        print(f"  Bear P/L  : {b_pnl:+.4%}  |  SHORT={b_short_pct:.0%}")
        print(f"  Train actions [L/S/C/H]: {pct_str(t_ac)}")
        print(f"  Val   actions [L/S/C/H]: {pct_str(v_ac)}")
        print(f"  MC-kNN bank_size={agent.last_bank_size:,}"
              f"  episodes_committed={agent.n_episodes_committed}"
              f"  prunes={agent.last_n_prunes}"
              f"  epsilon={epsilon:.4f}")
        print(f"  Time  : {elapsed:.1f}s")

        # ── SHORT collapse hard-stop (identical to main_sac.py) ───────────────
        short_pct_train = t_ac[1] / max(sum(t_ac), 1)
        if epoch > WARMUP_EPOCHS and short_pct_train < 0.03:
            print(f"  ⛔ SHORT collapsed to {short_pct_train:.1%} — stopping")
            break
        if epoch > WARMUP_EPOCHS and short_pct_train < 0.05:
            print(f"  ⚠ SHORT at {short_pct_train:.1%} in training — watch")

        # ── Directional saving criterion (identical to main_sac.py) ───────────
        short_pct_val = v_ac[1] / max(sum(v_ac), 1)
        long_pct_val  = v_ac[0] / max(sum(v_ac), 1)

        bear_pnl_floor = -0.10

        is_directional = short_pct_val >= 0.03 and long_pct_val >= 0.03

        if is_directional and v_pnl > best_val_pnl + MIN_IMPROVE and b_pnl > bear_pnl_floor:
            best_val_pnl = v_pnl
            no_improve = 0
            agent.save(BEST_PATH)
            print(f"  ⭐ New best val P/L: {best_val_pnl:+.4%}  bear={b_pnl:+.4%}"
                  f"  (L={long_pct_val:.0%} S={short_pct_val:.0%})")
        elif not is_directional:
            print(f"  ↷ Skipped save — directional collapse"
                  f" (L={long_pct_val:.0%} S={short_pct_val:.0%})")
            no_improve += 1
        elif b_pnl <= bear_pnl_floor:
            print(f"  ↷ Skipped save — bear floor breached (bear={b_pnl:+.4%})")
            no_improve += 1
        else:
            no_improve += 1

        # ── Early stopping (identical to main_sac.py) ──────────────────────────
        if epoch >= WARMUP_EPOCHS and no_improve >= PATIENCE:
            print(f"  ⚠ No val improvement for {PATIENCE} epochs — early stop")
            break

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % SAVE_EVERY_EPOCH == 0:
            agent.save(CHECKPOINT_PATH)

    print(f"\n✅ Training complete.  Best val P/L: {best_val_pnl:+.4%}")
    agent.save(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()

# ── Useful one-liners ─────────────────────────────────────────────────────────
# copy outcomes\mc_knn_agent_best.npz outcomes\mc_knn_agent.npz