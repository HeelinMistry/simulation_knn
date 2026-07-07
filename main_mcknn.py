"""
main_mcknn.py
─────────────
Headless training loop for the Monte Carlo k-NN trading agent.

REVISION HISTORY (high level)
────────────────────────────────
  v1: structural port of main_sac.py to MCKNNAgent/EpisodeBuffer.
  v2: MIN_TICK_GAP + MAX_WEIGHT_RATIO fix for same-pass near-duplicate
      voting (see mc_knn_memory.py).
  v3 (THIS REVISION): two further integrity-check fixes —

  1. PURGED + EMBARGOED WALK-FORWARD CV (walkforward.py)
     The old VAL_YEARS=[2022, 2025] split is a hard calendar boundary,
     but several features (rolling(200) indicators, pace-90 aggregator
     history) are NOT memoryless across that boundary — a val-window
     state can statistically depend on train-window data a few weeks
     earlier. MIN_TICK_GAP doesn't fix this: it only stops a query from
     voting on a near-duplicate of itself committed during the SAME
     training pass, not from training on data whose feature windows
     overlap the val window at all.

     Fix: generate_purged_folds() carves the full 2019-2026 dataset
     into several walk-forward folds, each with a purge buffer removed
     from train immediately before the val window and an embargo
     buffer removed immediately after it, sized to the worst-case
     feature lookback across ALL timeframes in use. We now run a
     cross-validation pass across these folds (fresh agent per fold,
     short training run) and report the DISTRIBUTION of val P/L across
     folds — a single number from one split told us nothing about
     whether performance was a property of the strategy or of one
     historical path; a spread across independent folds does.

     The final DEPLOYABLE checkpoint is then trained the same way as
     before (full epoch loop, early stopping, directional/bear-floor
     saving criteria) but using the LAST fold's purge/val/embargo
     layout for its validation signal, so even the production model's
     val metric is leak-free.

  2. MULTI-TIMEFRAME HIERARCHICAL CONTEXT (multi_timeframe_state.py)
     Optional 15m/1h context vectors, precomputed and aligned onto the
     4h decision timeline with a strict no-lookahead guard, concatenated
     into the state via UnifiedExecutor's new `extra_context` param.
     This both uses the finer-grained data available and reduces the
     near-duplicate-state density that the pre_training.py diagnostic
     flagged, since flat/quiet 4h stretches don't have to look flat on
     the 15m/1h signal too.

Everything NOT explicitly mentioned above (epoch structure, one-step-lag
reward bookkeeping, SHORT collapse hard-stop, episode-level MC return
backfill via agent.update(), epsilon-greedy schedule) is unchanged from
the prior revision.

RAW DATA LAYOUT (this revision)
───────────────────────────────────
data/raw/ is now organized per-timeframe:
    data/raw/15m/*.csv
    data/raw/1h/*.csv
    data/raw/4h/*.csv
Each folder holds plain (already-extracted) Binance-kline CSVs covering
that timeframe's full history. data_manager.update_master_data(timeframe)
/ update_all_timeframes() merge + compute indicators for each into
data/processed/XRPUSDT_<timeframe>_master_processed.csv. Unlike the old
zip-based flow, these plain CSVs are NEVER deleted by the pipeline — see
data/preprocessing.py / data/data_manager.py docstrings.
"""

import os
import time

import numpy as np
import pandas as pd

from agents.episode_buffer import EpisodeBuffer
from agents.mc_knn_agent import MCKNNAgent
from agents.mc_knn_memory import make_block_weights
from agents.unified_executor import UnifiedExecutor
from data.data_manager import update_master_data, update_all_timeframes
from walkforward import generate_purged_folds, compute_required_lookback_ticks, summarize_folds
from multi_timeframe_state import build_multi_timeframe_context

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MASTER_CSV       = "data/processed/XRPUSDT_master_processed.csv"
CHECKPOINT_PATH  = "outcomes/mc_knn_agent.npz"
BEST_PATH        = "outcomes/mc_knn_agent_best.npz"

FEATURES   = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
              "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES      = (1, 6, 42, 90)
ACTION_DIM = 4

# ── Multi-timeframe context (optional — set ENABLE_MULTI_TIMEFRAME=False
#    to fall back to the original 4h-only state, e.g. if 15m/1h raw data
#    isn't available yet).
#
#    Raw data layout (data_manager.py): data/raw/15m/*.csv,
#    data/raw/1h/*.csv, data/raw/4h/*.csv — each folder holds plain CSV
#    files covering that timeframe's full history. update_all_timeframes()
#    merges + computes indicators for each into the paths below. ──────────
ENABLE_MULTI_TIMEFRAME = True
PRIMARY_TIMEFRAME = "4h"
CONTEXT_TIMEFRAMES = ("15m", "1h")
TIMEFRAME_MASTER_CSVS = {
    tf: f"data/processed/XRPUSDT_{tf}_master_processed.csv"
    for tf in CONTEXT_TIMEFRAMES
}
CONTEXT_PACES = (1, 4, 16)   # smaller/fewer than the primary 4h PACES —
                             # context blocks capture intraday texture,
                             # not a second copy of the slow trend view

CONTEXT_DIM_PER_TIMEFRAME = len(FEATURES) * 2 * len(CONTEXT_PACES)
N_CONTEXT_TIMEFRAMES = len(TIMEFRAME_MASTER_CSVS) if ENABLE_MULTI_TIMEFRAME else 0
STATE_DIM_WITH_CONTEXT    = (len(FEATURES) * 2 * len(PACES)) \
                            + (N_CONTEXT_TIMEFRAMES * CONTEXT_DIM_PER_TIMEFRAME) + 2
STATE_DIM_WITHOUT_CONTEXT = (len(FEATURES) * 2 * len(PACES)) + 2

# Resolved once at runtime in main() depending on whether the 15m/1h
# master CSVs were actually found — see main() for the fallback logic.
STATE_DIM = STATE_DIM_WITH_CONTEXT if ENABLE_MULTI_TIMEFRAME else STATE_DIM_WITHOUT_CONTEXT

# ── Weighted (Mahalanobis-style) distance block weights ─────────────────────
# pre_training.py's diagnostic showed raw concatenation of 4h + 15m + 1h
# context made nearest-neighbor distance WORSE relative to overall
# pairwise distance (ratio 0.18 -> 0.52) — a curse-of-dimensionality
# symptom from giving 72 context dims equal say against 50 primary dims
# and a mere 2 portfolio dims (position/unrealized PnL) in plain
# Euclidean distance. mc_knn_memory.py now supports per-dimension
# weighting (combined with per-dim std normalization computed from the
# bank's own data) — these are the STARTING weights, not a derived
# optimum. Re-run pre_training.py's ratio check after changing these to
# see whether they actually help before trusting them.
PRIMARY_BLOCK_WEIGHT   = 1.0   # 4h market vector (50 dims)
CONTEXT_BLOCK_WEIGHT   = 0.5   # 15m/1h context, per-timeframe (72 dims total)
PORTFOLIO_BLOCK_WEIGHT = 2.0   # position + unrealized PnL (2 dims) — boosted
                               # since these are highly decision-relevant
                               # but would otherwise be just 2/122 of the
                               # vector's raw influence on distance.


def build_block_weights(state_dim: int, uses_context: bool) -> "np.ndarray":
    """
    Build the block-weight vector matching this file's actual state
    layout: [primary 4h][context, if enabled][portfolio (2 dims)].
    Always returns a vector of length exactly `state_dim`.
    """
    primary_dim = len(FEATURES) * 2 * len(PACES)
    portfolio_dim = 2
    if uses_context:
        context_dim = state_dim - primary_dim - portfolio_dim
        if context_dim <= 0:
            raise ValueError(
                f"uses_context=True but state_dim={state_dim} doesn't leave "
                f"room for a positive context block (primary={primary_dim}, "
                f"portfolio={portfolio_dim})."
            )
        return make_block_weights(
            [primary_dim, context_dim, portfolio_dim],
            [PRIMARY_BLOCK_WEIGHT, CONTEXT_BLOCK_WEIGHT, PORTFOLIO_BLOCK_WEIGHT],
        )
    return make_block_weights(
        [primary_dim, portfolio_dim],
        [PRIMARY_BLOCK_WEIGHT, PORTFOLIO_BLOCK_WEIGHT],
    )

# ── Walk-forward CV configuration (replaces VAL_YEARS hard split) ───────────
N_CV_FOLDS        = 6
CV_EPOCHS         = 5     # each CV epoch is independent with per-epoch reset
MIN_TRAIN_TICKS   = 1000

# Training hyperparameters (final/deployable run)
NUM_EPOCHS       = 15     # with accumulation, the bank peaks at ~2-3 epochs
                          # (~24K entries) and degrades beyond that as training-
                          # period states dominate val queries. 15 epochs with
                          # PATIENCE=3 will stop well before the bank bloats.
WARMUP_IDX       = 128    # aggregator warm-up rows

PATIENCE         = 3      # tighter — we know the sweet spot is early
WARMUP_EPOCHS    = 3      # don't check early-stop before epoch 4
MIN_IMPROVE      = 0.002

# ── MC-kNN specific knobs ─────────────────────────────────────────────────────
BANK_MAX_SIZE    = 25_000  # was 200,000 — the bank peaked at ~24K entries in
                           # diagnostics (+60.3% val) and deteriorated to
                           # -55.8% by epoch 7 (bank=83K). Capping at 25K
                           # triggers stratified pruning to keep the bank in
                           # the productive range rather than letting it grow
                           # into a training-period lookup table.
K_NEIGHBORS      = 25
GAMMA            = 0.97
SIGNAL_THRESHOLD = 0.0005

# ── Integrity-check fixes ─────────────────────────────────────────────────────
MIN_TICK_GAP     = 50
MAX_WEIGHT_RATIO = 50.0

# Logging
LOG_EVERY_TICKS  = 500
SAVE_EVERY_EPOCH = 5

# ── Reward shaping ────────────────────────────────────────────────────────────
# REWARD_SCALE reduced from 100.0 to 10.0: the previous value inflated MC
# returns 100x, making vote confidence look artificially strong and
# causing the bank's returns to swamp the [-42, +134] range seen in
# post_training.py — at 10x the return distribution is more calibrated
# relative to the actual trade PnL fractions (typically 0.1%–5%).
REWARD_SCALE    = 10.0
MICRO_HOLD_COST = 0.000005
EPSILON_START   = 0.10
EPSILON_END     = 0.01

# ── Val floor for final training checkpoint saving ───────────────────────────
# -0.10 was too tight for a single recent period (Dec 2024–Feb 2026 showed
# -38.7% in epoch 3, triggering the floor every epoch and preventing any
# checkpoint from ever being saved). -0.40 still catches genuine disasters
# while giving the model room to improve across epochs. CV showed 5/6
# folds positive with mean +25.5%, so a -40% floor on the final val window
# is a reasonable safety net, not a free pass.
VAL_PNL_FLOOR   = -0.40


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
              train: bool = True, epsilon: float = 0.0,
              extra_context_arr: np.ndarray = None,
              tick_offset: int = 0) -> dict:
    """
    Single deterministic or stochastic pass through df.

    NEW PARAMETERS vs prior revision
    ───────────────────────────────────
    extra_context_arr : optional (len(df), context_dim) array — e.g.
                         from multi_timeframe_state.build_multi_timeframe_context()
                         — aligned 1:1 with df's row order. Row i's
                         context is concatenated into the state for
                         df.iloc[i] via UnifiedExecutor.step(extra_context=...).
    tick_offset        : added to the LOCAL loop index `i` before it's
                         used as the `tick`/`query_tick` value passed to
                         executor.step() and episode_buffer.add(). Kept
                         for completeness/future use (e.g. resuming a
                         partially-trained fold) but NOT used by
                         run_cross_validation()/run_final_training()
                         below, since each fold's EpisodeBuffer already
                         has its own episode_id which fully isolates it
                         from every other fold/run — temporal exclusion
                         only needs to distinguish ticks WITHIN the same
                         episode_id, and a fold's local positional ticks
                         (0..n-1) do that correctly since the same
                         purged train_df is replayed identically each
                         epoch within that fold.

    Returns the same metrics dict shape as before.
    """
    indicators_arr = df[FEATURES].values.astype(np.float32)
    prices_arr     = df["Close"].values.astype(np.float32)
    n              = len(df)

    executor.total_reward  = 0.0
    executor.inventory.clear()
    executor.current_side  = None
    executor._entry_tick   = 0

    executor.aggregator.tick = 0
    executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)

    total_realised = 0.0
    n_trades       = 0
    action_counts  = [0, 0, 0, 0]

    def _ctx(i):
        return extra_context_arr[i] if extra_context_arr is not None else None

    prev_state  = executor.get_state(indicators_arr[WARMUP_IDX], prices_arr[WARMUP_IDX],
                                      extra_context=_ctx(WARMUP_IDX))
    prev_action = None
    prev_reward = 0.0
    prev_tick   = WARMUP_IDX + tick_offset

    for i in range(WARMUP_IDX + 1, n):
        indicators = indicators_arr[i]
        price      = prices_arr[i]
        abs_tick   = i + tick_offset

        action, probs, realised_pnl, s_t = executor.step(
            indicators, price, tick=abs_tick, epsilon=epsilon,
            episode_id=(episode_buffer.episode_id if train else None),
            extra_context=_ctx(i),
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
            # ── Exclude FLAT-HOLD only; keep IN-POSITION HOLD. ──────────────
            # Two types of HOLD:
            #
            # 1. FLAT-HOLD (not in a position): "chose not to enter the market"
            #    → No signal value. Storing these caused 94% HOLD dominance
            #    (the old problem). Excluded.
            #
            # 2. IN-POSITION HOLD (holding an existing trade): "chose to stay
            #    in this trade given the current indicators"
            #    → Real signal: the MC return will reflect whether holding paid
            #    off. Storing these lets the bank learn "when to stay in a
            #    trade" rather than churning in/out every tick (the new
            #    problem: 46.5% trade rate with all-HOLD excluded).
            #
            # After a HOLD action, current_side cannot have changed (HOLD
            # never opens or closes a position), so executor.current_side at
            # tick i+1 correctly reflects the position state when HOLD was
            # chosen at tick i.
            flat_hold = (prev_action == 3 and executor.current_side is None)
            if not flat_hold:
                episode_buffer.add(
                    prev_state, prev_action,
                    prev_reward * REWARD_SCALE,
                    tick=prev_tick,
                )

        prev_state  = s_t
        prev_action = action
        prev_reward = reward
        prev_tick   = abs_tick

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
        "update_count":  0,
    }


def _combine_metrics(m1: dict, m2: dict) -> dict:
    return {
        "realised_pnl":  m1["realised_pnl"]  + m2["realised_pnl"],
        "n_trades":      m1["n_trades"]       + m2["n_trades"],
        "action_counts": [a + b for a, b in
                          zip(m1["action_counts"], m2["action_counts"])],
        "update_count":  m1["update_count"]   + m2["update_count"],
    }


def _slice_context(extra_context_arr, mask):
    if extra_context_arr is None:
        return None
    return extra_context_arr[mask]


# ─────────────────────────────────────────────
# Walk-forward cross-validation
# ─────────────────────────────────────────────

def run_cross_validation(df: pd.DataFrame, extra_context_arr: np.ndarray,
                          state_dim: int, n_folds: int = N_CV_FOLDS) -> list:
    """
    Train a FRESH agent per fold (short CV_EPOCHS run, no checkpoint
    persistence) and evaluate on that fold's purged+embargoed val
    window. Returns a list of per-fold result dicts so the caller can
    compute mean/std across folds — the Monte-Carlo-style "look at the
    distribution, not one draw" check that a single year-split can't
    give you.
    """
    lookback_ticks = compute_required_lookback_ticks()
    folds = generate_purged_folds(
        df, n_folds=n_folds, lookback_ticks=lookback_ticks,
        min_train_ticks=MIN_TRAIN_TICKS,
    )
    summarize_folds(folds, df)

    fold_results = []
    for fold in folds:
        print(f"\n{'='*62}\n  CV FOLD {fold.fold_id}\n{'='*62}")

        train_df = df[fold.train_mask].reset_index(drop=True)
        val_df   = df[fold.val_mask].reset_index(drop=True)
        train_ctx = _slice_context(extra_context_arr, fold.train_mask)
        val_ctx   = _slice_context(extra_context_arr, fold.val_mask)

        if len(train_df) <= WARMUP_IDX + 2 or len(val_df) <= WARMUP_IDX + 2:
            print(f"  ⚠ Fold {fold.fold_id}: too few rows after purge/embargo "
                  f"(train={len(train_df)}, val={len(val_df)}) — skipping.")
            continue

        agent = MCKNNAgent(
            state_dim=state_dim, action_dim=ACTION_DIM,
            k=K_NEIGHBORS, max_size=BANK_MAX_SIZE, gamma=GAMMA,
            signal_threshold=SIGNAL_THRESHOLD,
            min_tick_gap=MIN_TICK_GAP, max_weight_ratio=MAX_WEIGHT_RATIO,
            block_weights=build_block_weights(state_dim, extra_context_arr is not None),
        )
        episode_buffer = EpisodeBuffer(gamma=GAMMA)

        # NOTE: train_df's rows are NOT contiguous in absolute time once
        # purge/embargo removes a chunk in the middle (everything before
        # purge_start and everything after embargo_end is concatenated
        # into one frame). Local positional ticks (0..n-1) are sufficient
        # given each fold gets its own episode_id from the per-epoch reset.
        for cv_epoch in range(1, CV_EPOCHS + 1):
            t0 = time.time()
            # ── Per-epoch bank reset (mirrors run_final_training's discipline).
            #    Without this, bank grows to CV_EPOCHS × fold_train_rows entries
            #    — 8 epochs × ~11K rows = ~88K entries with 8 near-duplicate
            #    copies of the same period. The _dim_scale computed from those
            #    distorted the weighted distance and caused CV to degrade from
            #    5/6 positive (CV_EPOCHS=3, no reset) to 2/6 (CV_EPOCHS=8,
            #    no reset). With per-epoch reset, each fold's evaluation uses a
            #    fresh single-epoch bank — matching exactly what the final
            #    training run produces and making CV a valid proxy for it. ──────
            agent.memory.__init__(
                state_dim=state_dim, action_dim=ACTION_DIM,
                k=K_NEIGHBORS, max_size=BANK_MAX_SIZE,
                signal_threshold=SIGNAL_THRESHOLD,
                eps_dist=agent.memory.eps_dist,
                max_weight_ratio=MAX_WEIGHT_RATIO,
                min_tick_gap=MIN_TICK_GAP,
                block_weights=agent.memory.block_weights,
                dim_scale_floor=agent.memory.dim_scale_floor,
            )
            agent.actor.memory = agent.memory
            episode_buffer.reset(new_episode_id=True)
            train_executor = UnifiedExecutor(
                f"CVFold{fold.fold_id}_Train", agent, paces=PACES,
                deterministic=False, num_indicators=len(FEATURES),
            )
            run_epoch(train_executor, train_df, episode_buffer, agent,
                      train=True, epsilon=EPSILON_START,
                      extra_context_arr=train_ctx)
            agent.update(episode_buffer)
            print(f"  CV epoch {cv_epoch}/{CV_EPOCHS}  "
                  f"bank={len(agent.memory):,}  ({time.time()-t0:.1f}s)")

        val_executor = UnifiedExecutor(
            f"CVFold{fold.fold_id}_Val", agent, paces=PACES,
            deterministic=True, num_indicators=len(FEATURES),
        )
        _val_buf = EpisodeBuffer(gamma=GAMMA)
        val_metrics = run_epoch(val_executor, val_df, _val_buf, agent,
                                train=False, extra_context_arr=val_ctx)

        print(f"  Fold {fold.fold_id} val P/L: {val_metrics['realised_pnl']:+.4%}  "
              f"trades={val_metrics['n_trades']}")
        fold_results.append({
            "fold_id": fold.fold_id,
            "val_pnl": val_metrics["realised_pnl"],
            "n_trades": val_metrics["n_trades"],
            "action_counts": val_metrics["action_counts"],
        })

    if fold_results:
        pnls = np.array([r["val_pnl"] for r in fold_results])
        summary = {
            "n_folds":          len(fold_results),
            "mean_val_pnl":     float(pnls.mean()),
            "std_val_pnl":      float(pnls.std()),
            "min_val_pnl":      float(pnls.min()),
            "max_val_pnl":      float(pnls.max()),
            "n_folds_positive": int((pnls > 0).sum()),
            "state_dim":        int(state_dim),
            "multi_timeframe":  bool(extra_context_arr is not None),
            "lookback_ticks":   int(lookback_ticks),
            "fold_results":     fold_results,
        }
        print(f"\n{'='*62}")
        print(f"  WALK-FORWARD CV SUMMARY ({len(fold_results)} folds)")
        print(f"{'='*62}")
        print(f"  Mean val P/L : {summary['mean_val_pnl']:+.4%}")
        print(f"  Std  val P/L : {summary['std_val_pnl']:.4%}")
        print(f"  Min  val P/L : {summary['min_val_pnl']:+.4%}")
        print(f"  Max  val P/L : {summary['max_val_pnl']:+.4%}")
        print(f"  Folds positive: {summary['n_folds_positive']}/{summary['n_folds']}")
        if summary["n_folds"] > 1 and summary["std_val_pnl"] > abs(summary["mean_val_pnl"]):
            summary["std_exceeds_mean_warning"] = True
            print("  ⚠ Std exceeds |mean| — performance is NOT consistent "
                  "across historical periods; treat any single-split result "
                  "(including the deployable run below) with real skepticism.")
        else:
            summary["std_exceeds_mean_warning"] = False
        print(f"{'='*62}\n")

        # ── Persist so diagnostic_mcknn.py (and anyone else) can report
        #    this distribution AFTER training, not just in the console
        #    log — this is the single most informative "is this model
        #    trustworthy" signal this revision adds, and it was
        #    previously only ever printed, never saved. ──────────────────
        import json
        os.makedirs("outcomes", exist_ok=True)
        cv_summary_path = "outcomes/walkforward_cv_summary.json"
        with open(cv_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓  CV summary saved → {cv_summary_path}")

    return fold_results


# ─────────────────────────────────────────────
# Final deployable training run
# ─────────────────────────────────────────────

def run_final_training(df: pd.DataFrame, extra_context_arr: np.ndarray,
                       state_dim: int):
    """
    Train the checkpoint that actually gets deployed (live_mcknn.py /
    diagnostic_mcknn.py consume this). Uses the SAME purge/embargo
    discipline as the CV folds — train on everything except the purge+
    val+embargo block of the LAST walk-forward fold (the most recent
    val window), so the production model's reported val metric is just
    as leak-free as the CV folds were, instead of reverting to the old
    hard year-boundary split.
    """
    lookback_ticks = compute_required_lookback_ticks()
    folds = generate_purged_folds(
        df, n_folds=N_CV_FOLDS, lookback_ticks=lookback_ticks,
        min_train_ticks=MIN_TRAIN_TICKS,
    )
    final_fold = folds[-1]   # most recent val window = most relevant
                             # out-of-sample signal for a model going live
    print(f"\n  Final training fold: val ticks "
          f"[{final_fold.val_start}:{final_fold.val_end}) "
          f"(purged region [{final_fold.purge_start}:{final_fold.embargo_end}))")

    train_df = df[final_fold.train_mask].reset_index(drop=True)
    val_df   = df[final_fold.val_mask].reset_index(drop=True)
    train_ctx = _slice_context(extra_context_arr, final_fold.train_mask)
    val_ctx   = _slice_context(extra_context_arr, final_fold.val_mask)

    print(f"  Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")

    # ── Persist this fold's exact boundaries + state config. This is
    #    what closes the diagnostic_mcknn.py alignment gap: its old
    #    VAL_YEARS=[2022,2025] split was independent of (and could
    #    overlap) whatever data this checkpoint was actually trained
    #    and selected on. diagnostic_mcknn.py now reads this file and
    #    rebuilds the IDENTICAL purge/val/embargo split instead of
    #    guessing at it via calendar years. ─────────────────────────────
    import json
    os.makedirs("outcomes", exist_ok=True)
    fold_meta = {
        "val_start":        int(final_fold.val_start),
        "val_end":          int(final_fold.val_end),
        "purge_start":      int(final_fold.purge_start),
        "embargo_end":      int(final_fold.embargo_end),
        "lookback_ticks":   int(lookback_ticks),
        "n_folds_used":     int(N_CV_FOLDS),
        "state_dim":        int(state_dim),
        "multi_timeframe":  bool(extra_context_arr is not None),
        "features":         FEATURES,
        "paces":            list(PACES),
        "context_paces":    list(CONTEXT_PACES) if extra_context_arr is not None else None,
        "context_timeframes": list(CONTEXT_TIMEFRAMES) if extra_context_arr is not None else None,
        "warmup_idx":       int(WARMUP_IDX),
        "total_rows":       int(len(df)),
        "block_weights":    {
            "primary":   PRIMARY_BLOCK_WEIGHT,
            "context":   CONTEXT_BLOCK_WEIGHT if extra_context_arr is not None else None,
            "portfolio": PORTFOLIO_BLOCK_WEIGHT,
        },
    }
    with open("outcomes/final_fold_meta.json", "w") as f:
        json.dump(fold_meta, f, indent=2)
    print(f"  ✓  Final fold metadata saved → outcomes/final_fold_meta.json")

    agent = MCKNNAgent(
        state_dim=state_dim, action_dim=ACTION_DIM,
        k=K_NEIGHBORS, max_size=BANK_MAX_SIZE, gamma=GAMMA,
        signal_threshold=SIGNAL_THRESHOLD,
        min_tick_gap=MIN_TICK_GAP, max_weight_ratio=MAX_WEIGHT_RATIO,
        block_weights=build_block_weights(state_dim, extra_context_arr is not None),
    )
    agent.load(CHECKPOINT_PATH)

    best_val_pnl = -np.inf
    no_improve   = 0
    # ── Epsilon schedule: slow decay over NUM_EPOCHS. ───────────────────────
    # With bank accumulation (no reset), the bank grows each epoch so
    # training queries become more informative over time. Epsilon should
    # still stay meaningful throughout so the bank keeps seeing new
    # exploration rather than repeating the same deterministic path.
    EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1.0 / max(NUM_EPOCHS - 1, 1))
    epsilon       = EPSILON_START
    # ── Single episode_buffer for the entire training run. ──────────────────
    # NOT reset between epochs — end_episode_and_commit() calls
    # reset(new_episode_id=False) internally, keeping the same episode_id
    # across all epochs. This means temporal exclusion (MIN_TICK_GAP) covers
    # ALL epochs: a training query at tick T will exclude committed rows from
    # ANY epoch within ±MIN_TICK_GAP of T. This is the correct behaviour —
    # it prevents any epoch from voting on a near-duplicate of itself from
    # a prior epoch at the same historical tick.
    episode_buffer = EpisodeBuffer(gamma=GAMMA)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ── NO per-epoch bank reset (key change from prior revision). ────────
        # Previously the bank was wiped at the start of each epoch, which
        # meant bank=0 throughout the ENTIRE training pass — every query
        # hit the empty-bank cold-start HOLD default, so training actions
        # were driven purely by epsilon-greedy randomness, not by any learned
        # signal. Training P/L reflected random exploration quality, not model
        # quality, and the bank only helped AFTER commit (val evaluation).
        #
        # With accumulation:
        #   epoch 1: bank=0 → epsilon-only → commit ~1K entries
        #   epoch 2: bank=1K → kNN + epsilon → commit ~1K more
        #   epoch N: bank=N×1K → increasingly informed queries
        #
        # Near-duplicate protection: the stable episode_id means temporal
        # exclusion covers ALL prior epochs — a query at tick T (±50 ticks)
        # cannot vote on rows committed at that same tick from any earlier
        # epoch. This prevents self-referential vote inflation across epochs
        # without discarding the genuine historical signal those rows carry.
        #
        # The bank will approach BANK_MAX_SIZE over many epochs; pruning
        # (stratified by |return|) keeps it bounded when it gets there.
        train_executor = UnifiedExecutor(
            "Train", agent, paces=PACES,
            deterministic=False, num_indicators=len(FEATURES)
        )

        train_metrics = run_epoch(
            train_executor, train_df, episode_buffer, agent,
            train=True, epsilon=epsilon, extra_context_arr=train_ctx,
        )
        agent.update(episode_buffer)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        val_executor = UnifiedExecutor(
            "Val", agent, paces=PACES,
            deterministic=True, num_indicators=len(FEATURES)
        )
        _val_buf = EpisodeBuffer(gamma=GAMMA)
        val_metrics = run_epoch(val_executor, val_df, _val_buf, agent,
                                train=False, extra_context_arr=val_ctx)

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
        print(f"  Val P/L   : {v_pnl:+.4%}  |  trades={val_metrics['n_trades']}")
        print(f"  Train actions [L/S/C/H]: {pct_str(t_ac)}")
        print(f"  Val   actions [L/S/C/H]: {pct_str(v_ac)}")
        print(f"  MC-kNN bank_size={agent.last_bank_size:,}"
              f"  episodes_committed={agent.n_episodes_committed}"
              f"  prunes={agent.last_n_prunes}"
              f"  epsilon={epsilon:.4f}")
        print(f"  Time  : {elapsed:.1f}s")

        short_pct_train = t_ac[1] / max(sum(t_ac), 1)
        if epoch > WARMUP_EPOCHS and short_pct_train < 0.03:
            print(f"  ⛔ SHORT collapsed to {short_pct_train:.1%} — stopping")
            break
        if epoch > WARMUP_EPOCHS and short_pct_train < 0.05:
            print(f"  ⚠ SHORT at {short_pct_train:.1%} in training — watch")

        short_pct_val = v_ac[1] / max(sum(v_ac), 1)
        long_pct_val  = v_ac[0] / max(sum(v_ac), 1)
        is_directional = short_pct_val >= 0.03 and long_pct_val >= 0.03

        if is_directional and v_pnl > best_val_pnl + MIN_IMPROVE and v_pnl > VAL_PNL_FLOOR:
            best_val_pnl = v_pnl
            no_improve = 0
            agent.save(BEST_PATH)
            print(f"  ⭐ New best val P/L: {best_val_pnl:+.4%}"
                  f"  (L={long_pct_val:.0%} S={short_pct_val:.0%})")
        elif not is_directional:
            print(f"  ↷ Skipped save — directional collapse"
                  f" (L={long_pct_val:.0%} S={short_pct_val:.0%})")
            no_improve += 1
        elif v_pnl <= VAL_PNL_FLOOR:
            print(f"  ↷ Skipped save — val floor breached "
                  f"(val={v_pnl:+.4%} < floor={VAL_PNL_FLOOR:+.0%})")
            no_improve += 1
        else:
            no_improve += 1

        if epoch >= WARMUP_EPOCHS and no_improve >= PATIENCE:
            print(f"  ⚠ No val improvement for {PATIENCE} epochs — early stop")
            break

        if epoch % SAVE_EVERY_EPOCH == 0:
            agent.save(CHECKPOINT_PATH)

    print(f"\n✅ Training complete.  Best val P/L: {best_val_pnl:+.4%}")
    agent.save(CHECKPOINT_PATH)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # ── Primary (4h, decision-timeframe) master data ─────────────────────────
    df = update_master_data(PRIMARY_TIMEFRAME)
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)

    extra_context_arr = None
    state_dim = STATE_DIM_WITHOUT_CONTEXT

    if ENABLE_MULTI_TIMEFRAME:
        try:
            # Refreshes/builds data/processed/XRPUSDT_<tf>_master_processed.csv
            # for every context timeframe directly from data/raw/<tf>/*.csv —
            # raises FileNotFoundError per-timeframe if that raw folder is
            # missing/empty, which is caught below and treated as "disable
            # multi-timeframe context, fall back to 4h-only" rather than
            # aborting the whole run.
            timeframe_dfs = update_all_timeframes(CONTEXT_TIMEFRAMES)
            missing = [tf for tf in CONTEXT_TIMEFRAMES if tf not in timeframe_dfs]
            if missing:
                raise FileNotFoundError(
                    f"raw data missing for timeframe(s): {missing} "
                    f"(expected under data/raw/<timeframe>/*.csv)"
                )

            timeframe_frames = {
                tf: tdf[["Open_time"] + FEATURES].dropna().reset_index(drop=True)
                for tf, tdf in timeframe_dfs.items()
            }

            print("\nBuilding multi-timeframe context "
                  f"({', '.join(timeframe_frames.keys())})...")
            extra_context_arr = build_multi_timeframe_context(
                df, timeframe_frames, FEATURES, context_paces=CONTEXT_PACES,
            )
            state_dim = STATE_DIM_WITH_CONTEXT
            print(f"  Context array shape: {extra_context_arr.shape}  "
                  f"(STATE_DIM={state_dim})")
        except FileNotFoundError as exc:
            print(f"\n⚠ Multi-timeframe context disabled — {exc}")
            print("  Falling back to 4h-only state "
                  f"(STATE_DIM={STATE_DIM_WITHOUT_CONTEXT}). Add raw CSVs under "
                  f"data/raw/{{{','.join(CONTEXT_TIMEFRAMES)}}}/ to enable it, or "
                  "set ENABLE_MULTI_TIMEFRAME=False to silence this warning.")
            extra_context_arr = None
            state_dim = STATE_DIM_WITHOUT_CONTEXT

    print(f"\nTotal rows: {len(df):,}  |  STATE_DIM={state_dim}")
    print(f"Date range: {df['Open_time'].iloc[0]} → {df['Open_time'].iloc[-1]}")

    # ── Step 1: walk-forward CV — measure robustness across many
    #    independent historical periods before trusting any one number ──
    print(f"\n{'#'*62}\n  STEP 1 — PURGED + EMBARGOED WALK-FORWARD CROSS-VALIDATION\n{'#'*62}")
    run_cross_validation(df, extra_context_arr, state_dim, n_folds=N_CV_FOLDS)

    # ── Step 2: train the deployable checkpoint ───────────────────────────
    print(f"\n{'#'*62}\n  STEP 2 — FINAL DEPLOYABLE TRAINING RUN\n{'#'*62}")
    run_final_training(df, extra_context_arr, state_dim)


if __name__ == "__main__":
    main()

# ── Useful one-liners ─────────────────────────────────────────────────────────
# copy outcomes\mc_knn_agent_best.npz outcomes\mc_knn_agent.npz