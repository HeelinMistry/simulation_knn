"""
diagnostic_gbt.py
──────────────────
Post-training diagnostic suite for the calibrated GBT meta-labeling agent
(gbt_agent.py / main_gbt.py) — the GBT analogue of diagnostic_mcknn.py.

TWO REAL BUGS THIS FILE WORKS AROUND (see GBTExecutor below)
──────────────────────────────────────────────────────────────
1. STATE-SHAPE MISMATCH. main_gbt.py's build_states_and_labels() trains
   on states of shape (market_dim + context_dim) — it never appends the
   2 portfolio features (position, unrealized_pnl) that
   UnifiedExecutor.get_state() always appends for every other agent in
   this codebase. Feeding UnifiedExecutor's full state straight into
   GBTAgent.select_action() therefore raises a scikit-learn
   n_features mismatch (or, if state_dim happens to be "fixed" to hide
   the error, silently misinterprets position/unrealized_pnl as market
   features). GBTExecutor strips the trailing 2 dims before querying
   the agent, while get_state() still returns the FULL state — so
   feature-indexing conventions used elsewhere in this project
   (ep["states_arr"][:, :len(FEATURES)] for raw indicators, state[-1]
   for unrealized_pnl) keep working exactly like every other
   diagnostic here.

2. CURRENT_SIDE NOT THREADED THROUGH. UnifiedExecutor._select_action()
   calls agent.select_action(state, deterministic, action_mask) with no
   knowledge of GBT-specific kwargs. GBTAgent.select_action() accepts an
   explicit current_side to correctly resolve the in-position
   CLOSE-vs-HOLD decision; without it, GBTPolicy._in_position_probs()
   falls back to guessing the open side from whichever of LONG/SHORT
   currently has higher probability (see gbt_agent.py's GBTAgent
   docstring), which can disagree with the side actually held.
   GBTExecutor threads self.current_side through so in-position
   decisions use ground truth.

CHANGES IN THIS REVISION — MULTI-SYMBOL SUPPORT
───────────────────────────────────────────────
Mirrors main_gbt.py's multi-symbol revision:

  - New `--symbol` CLI flag (default main_gbt.SYMBOL i.e. "XRPUSDT",
    choices = data_manager's SYMBOLS list via main_gbt.AVAILABLE_SYMBOLS).
  - load_data() now takes a `symbol` parameter and forwards it to
    data_manager.update_master_data()/update_all_timeframes(), so
    diagnostics always run against the SAME symbol's data the
    checkpoint being diagnosed was trained on.
  - Outputs are read from / written to a per-symbol subfolder,
    matching main_gbt.py's symbol_out_dir(): checkpoint default,
    cpcv_summary.json / calibration_report.json lookups, and this
    file's own `diagnostics/` output directory (OUT_DIR) are all
    computed from `--symbol` inside main(), instead of the fixed,
    symbol-less paths this file used before. OUT_DIR is a module-level
    variable (not a constant) reassigned once at the top of main() —
    every plotting/summary helper below reads it as a global at call
    time, so this reassignment is picked up without needing to thread
    an out_dir parameter through every function.

Run
────
    python diagnostic_gbt.py --split val
    python diagnostic_gbt.py --symbol BTCUSDT --split val
    python diagnostic_gbt.py --split both --checkpoint outcomes/gbt/XRPUSDT/gbt_agent_best.joblib
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from agents.unified_executor import UnifiedExecutor
from gbt_agent               import GBTAgent
from data.data_manager       import update_master_data, update_all_timeframes
from multi_timeframe_state   import build_multi_timeframe_context
from walkforward             import compute_required_lookback_ticks
from calibration_report      import evaluate_calibration, plot_reliability_diagram
from triple_barrier          import build_meta_labels, atr_to_frac

# main_gbt.py is imported (not duplicated) as the single source of truth
# for training config, so this file can never silently drift out of sync
# with what the checkpoint was actually trained on. This also gives us
# main_gbt's multi-symbol helpers (SYMBOL, AVAILABLE_SYMBOLS,
# symbol_out_dir) for free — see module docstring.
import main_gbt as gbt_cfg

FEATURES               = gbt_cfg.FEATURES
PACES                  = gbt_cfg.PACES
WARMUP_IDX             = gbt_cfg.WARMUP_IDX
ENABLE_MULTI_TIMEFRAME = gbt_cfg.ENABLE_MULTI_TIMEFRAME
CONTEXT_TIMEFRAMES     = gbt_cfg.CONTEXT_TIMEFRAMES
CONTEXT_PACES          = gbt_cfg.CONTEXT_PACES
TP_MULT, SL_MULT       = gbt_cfg.TP_MULT, gbt_cfg.SL_MULT
MAX_HOLDING            = gbt_cfg.MAX_HOLDING
COMMISSION             = gbt_cfg.COMMISSION
DEFAULT_SYMBOL         = gbt_cfg.SYMBOL
AVAILABLE_SYMBOLS      = gbt_cfg.AVAILABLE_SYMBOLS

ACTION_DIM    = 4
ACTION_NAMES  = ["LONG", "SHORT", "CLOSE", "HOLD"]
ACTION_COLORS = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]
ATR_IDX       = FEATURES.index("ATR_Scaled")

# NOTE (multi-symbol revision): these three are now just FALLBACK
# defaults for the flat/symbol-less layout — main() recomputes all of
# them (and reassigns OUT_DIR) from `--symbol` before running anything,
# via gbt_cfg.symbol_out_dir(). Any caller that imports this module and
# uses these constants directly (without going through main()) still
# gets the pre-multi-symbol XRPUSDT-flat behaviour unchanged.
CHECKPOINT_DEFAULT = os.path.join(gbt_cfg.OUT_DIR, "gbt_agent_best.joblib")
CPCV_SUMMARY_PATH  = os.path.join(gbt_cfg.OUT_DIR, "cpcv_summary.json")
CALIB_JSON_PATH    = os.path.join(gbt_cfg.OUT_DIR, "calibration_report.json")
OUT_DIR             = os.path.join(gbt_cfg.OUT_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "axes.facecolor": "#1a1a2e", "figure.facecolor": "#0f0f1a",
    "axes.edgecolor": "#444466", "axes.labelcolor": "#ccccee",
    "xtick.color": "#aaaacc", "ytick.color": "#aaaacc",
    "text.color": "#ddddff", "grid.color": "#2a2a4a",
    "grid.linestyle": "--", "grid.alpha": 0.5,
}


# ─────────────────────────────────────────────────────────────────────────
# GBTExecutor — the two fixes described in the module docstring
# ─────────────────────────────────────────────────────────────────────────

class GBTExecutor(UnifiedExecutor):
    def _select_action(self, state, deterministic, action_mask, episode_id):
        gbt_state = state[:-2]   # strip portfolio dims — see module docstring, bug 1
        return self.agent.select_action(
            gbt_state, deterministic=deterministic, action_mask=action_mask,
            current_side=self.current_side,   # bug 2
        )


def _to_gbt_state(state: np.ndarray) -> np.ndarray:
    """Same strip, for the callers below that go around GBTExecutor
    entirely and invoke agent.actor.get_action() directly."""
    return state[:-2]


# ─────────────────────────────────────────────────────────────────────────
# Data loading — mirrors main_gbt.build_states_and_labels()'s data path,
# but keeps indicators_arr / extra_context_arr separately (rather than
# only the final concatenated state matrix that function returns), since
# GBTExecutor needs to rebuild states tick-by-tick through a live
# aggregator to reproduce real position/commission dynamics — not from a
# precomputed batch.
# ─────────────────────────────────────────────────────────────────────────

def load_data(symbol: str = DEFAULT_SYMBOL, frozen: bool = False):
    """
    symbol : which coin's master CSV(s) to load (default
        DEFAULT_SYMBOL/"XRPUSDT"). Forwarded to
        data_manager.update_master_data()/update_all_timeframes() so
        the primary (4h) and context (15m/1h) timeframes are always
        the SAME symbol as the checkpoint being diagnosed.
    frozen : forwarded to data_manager — if True, uses the existing
        master CSV(s) as-is instead of refetching/appending new raw
        data (mirrors main_gbt.py's --frozen-data).
    """
    df = update_master_data("4h", symbol=symbol, frozen=frozen)
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    indicators_arr = df[FEATURES].values.astype(np.float32)
    prices_arr     = df["Close"].values.astype(np.float32)
    n = len(df)

    extra_context_arr = None
    if ENABLE_MULTI_TIMEFRAME:
        try:
            timeframe_dfs = update_all_timeframes(CONTEXT_TIMEFRAMES, symbol=symbol,
                                                  frozen=frozen)
            missing = [tf for tf in CONTEXT_TIMEFRAMES if tf not in timeframe_dfs]
            if missing:
                raise FileNotFoundError(f"missing timeframe(s): {missing}")
            timeframe_frames = {
                tf: tdf[["Open_time"] + FEATURES].dropna().reset_index(drop=True)
                for tf, tdf in timeframe_dfs.items()
            }
            extra_context_arr = build_multi_timeframe_context(
                df, timeframe_frames, FEATURES, context_paces=CONTEXT_PACES,
            )
            print(f"  Multi-timeframe context: {extra_context_arr.shape}")
        except FileNotFoundError as exc:
            print(f"  ⚠ Multi-timeframe context unavailable — {exc}")
            extra_context_arr = None

    return df, indicators_arr, prices_arr, extra_context_arr, n


def make_holdout_range(n: int, holdout_frac: float = 0.15) -> dict:
    """
    Reproduces main_gbt.py's run_final_training() split EXACTLY, so
    "val" here means the genuinely held-out block the deployed
    checkpoint was calibrated/reported against — not an independently
    invented split that might silently overlap train.

    Indices below are in STATE-ARRAY space (0 == first row after
    WARMUP_IDX+1, matching build_states_and_labels()'s indexing); the
    caller converts to raw df tick indices via `WARMUP_IDX + 1 + j`.
    """
    states_n = n - WARMUP_IDX - 1
    lookback_ticks = compute_required_lookback_ticks()
    holdout_start = int(states_n * (1 - holdout_frac))
    embargo_start = max(0, holdout_start - lookback_ticks)
    return {
        "states_n": states_n,
        "train_range":   (0, embargo_start),
        "holdout_range": (holdout_start, states_n),
        "lookback_ticks": lookback_ticks,
    }


def _state_range_to_ticks(lo: int, hi: int) -> tuple:
    return WARMUP_IDX + 1 + lo, WARMUP_IDX + 1 + hi


# ─────────────────────────────────────────────────────────────────────────
# Tick-by-tick backtest pass
# ─────────────────────────────────────────────────────────────────────────

def collect_episode(agent: GBTAgent, indicators_arr, prices_arr,
                     extra_context_arr, open_times, tick_start, tick_end,
                     label: str) -> dict:
    """Real sequential backtest over df ticks [tick_start, tick_end)
    using GBTExecutor — same shape/semantics as diagnostic_mcknn.py's
    collect_episode(), adapted for GBT's calibrated probabilities."""
    executor = GBTExecutor(
        name=label, agent=agent, paces=PACES,
        deterministic=True, num_indicators=len(FEATURES),
    )
    executor.aggregator.tick = 0
    executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)
    # Fast-forward the aggregator through any ticks between WARMUP_IDX and
    # tick_start so each pace's `tick % pace == 0` sampling phase is
    # correct for splits that don't start right after warm-up (e.g. the
    # holdout window, which begins ~85% of the way through the series).
    for i in range(WARMUP_IDX + 1, tick_start):
        executor.aggregator.update(indicators_arr[i])

    def _ctx(i):
        return extra_context_arr[i] if extra_context_arr is not None else None

    ticks, prices, actions, entropies = [], [], [], []
    prob_long, prob_short, prob_close, prob_hold = [], [], [], []
    raw_features, unrealized_pnl, states_arr = [], [], []
    trades = []
    open_tick = open_price = open_side = None

    for i in range(tick_start, tick_end):
        ind, price = indicators_arr[i], prices_arr[i]
        action, probs, realised_pnl, s_t = executor.step(
            ind, price, tick=i, extra_context=_ctx(i),
        )
        ent = -np.sum(probs * np.log2(probs + 1e-9))

        ticks.append(i); prices.append(price); actions.append(action)
        entropies.append(ent)
        prob_long.append(probs[0]); prob_short.append(probs[1])
        prob_close.append(probs[2]); prob_hold.append(probs[3])
        raw_features.append(ind.copy())
        unrealized_pnl.append(float(s_t[-1]))
        states_arr.append(s_t.copy())

        if realised_pnl != 0.0:
            if open_tick is not None:
                local = open_tick - tick_start
                trades.append({
                    "entry_tick": open_tick, "exit_tick": i,
                    "duration": i - open_tick, "side": open_side,
                    "pnl": realised_pnl,
                    "entry_price": open_price, "exit_price": price,
                    "entry_conviction": float(
                        prob_long[local] if open_side == "LONG" else prob_short[local]
                    ),
                    "exit_conviction": float(probs[action]),
                    "win": realised_pnl > 0,
                })
            open_tick = open_price = open_side = None

        if action in (0, 1) and executor.current_side is not None and open_tick is None:
            open_tick, open_price, open_side = i, price, executor.current_side

    if not ticks:
        return _empty_episode(label)

    ticks      = np.array(ticks, dtype=np.int32)
    prices     = np.array(prices, dtype=np.float32)
    actions    = np.array(actions, dtype=np.int32)
    entropies  = np.array(entropies, dtype=np.float32)
    prob_long  = np.array(prob_long, dtype=np.float32)
    prob_short = np.array(prob_short, dtype=np.float32)
    prob_close = np.array(prob_close, dtype=np.float32)
    prob_hold  = np.array(prob_hold, dtype=np.float32)
    raw_features   = np.array(raw_features, dtype=np.float32)
    unrealized_pnl = np.array(unrealized_pnl, dtype=np.float32)
    states_arr     = np.array(states_arr, dtype=np.float32)

    probs_all = np.stack([prob_long, prob_short, prob_close, prob_hold], axis=1)
    max_prob  = probs_all.max(axis=1)

    pnl_curve = np.zeros(len(ticks))
    for t in trades:
        pnl_curve[t["exit_tick"] - tick_start:] += t["pnl"]

    timestamps = None
    if open_times is not None:
        try:
            ts = pd.to_datetime(open_times[tick_start:tick_end], errors="coerce")
            timestamps = list(ts)
        except Exception:
            timestamps = None

    return {
        "label": label, "ticks": ticks, "prices": prices, "actions": actions,
        "entropies": entropies, "probs_all": probs_all, "max_prob": max_prob,
        "raw_features": raw_features, "unrealized_pnl": unrealized_pnl,
        "states_arr": states_arr, "trades": trades, "pnl_curve": pnl_curve,
        "timestamps": timestamps,
    }


def _empty_episode(label: str) -> dict:
    empty = np.array([], dtype=np.float32)
    return {
        "label": label, "ticks": np.array([], dtype=np.int32),
        "prices": empty, "actions": np.array([], dtype=np.int32),
        "entropies": empty, "probs_all": np.zeros((0, 4), dtype=np.float32),
        "max_prob": empty, "raw_features": np.zeros((0, 6), dtype=np.float32),
        "unrealized_pnl": empty, "states_arr": np.zeros((0, 2), dtype=np.float32),
        "trades": [], "pnl_curve": empty, "timestamps": None,
    }


# ─────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────

def make_fig(rows, cols, title, figsize=None):
    with plt.rc_context(STYLE):
        fs = figsize or (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=fs, squeeze=False)
        fig.suptitle(title, color="#ffffff", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor(STYLE["figure.facecolor"])
    return fig, axes


def savefig(fig, name):
    # OUT_DIR is a module-level variable (not a constant) reassigned by
    # main() once --symbol is parsed — see module docstring. Reading it
    # here (rather than capturing it at import time) is what makes that
    # reassignment take effect for every plot this run produces.
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
    plt.close(fig)
    print(f"  ✓  {path}")


# ── Section 1 — Conviction, Entropy & Calibration ──────────────────────────

def plot_confidence_calibration(ep: dict, calib_report: dict = None):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(1, 3, f"[{ep['label']}] Section 1 — Conviction, Entropy & Calibration")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        ax.hist(ep["max_prob"], bins=50, color="#7f5af0", edgecolor="none", alpha=0.85)
        ax.axvline(0.5, color="#ff6b6b", lw=1.5, linestyle="--", label="50% conviction")
        ax.axvline(np.mean(ep["max_prob"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['max_prob']):.2f}")
        ax.set_title("Max Calibrated Probability (\"Conviction\")")
        ax.set_xlabel("max P(a|s)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8); ax.grid(True)

        ax = axes[0][1]
        ax.hist(ep["entropies"], bins=50, color="#2cb67d", edgecolor="none", alpha=0.85)
        ax.axvline(np.mean(ep["entropies"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['entropies']):.2f}b")
        ax.set_title("Prediction Entropy Distribution (bits)")
        ax.set_xlabel("H[P(·|s)] bits"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8); ax.grid(True)

        ax = axes[0][2]
        if calib_report is not None:
            names = list(calib_report["per_class"].keys())
            eces  = [calib_report["per_class"][n]["ece"] for n in names]
            ax.bar(names, eces, color=ACTION_COLORS[:len(names)], alpha=0.85)
            ax.axhline(0.05, color="#ff6b6b", lw=1.2, linestyle="--", label="0.05 (good)")
            ax.set_title(f"ECE by class (macro={calib_report['macro_ece']:.3f})")
            ax.set_ylabel("Expected Calibration Error")
            ax.legend(fontsize=8); ax.grid(True, axis="y")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No calibration labels\navailable for this window",
                    ha="center", va="center", fontsize=9, color="gray")

    savefig(fig, f"{ep['label']}_01_confidence_calibration.png")


# ── Section 2 — Action Distribution & Sequencing ────────────────────────────

def plot_action_distribution(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(1, 3, f"[{ep['label']}] Section 2 — Action Distribution & Sequencing")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        counts = [(ep["actions"] == i).sum() for i in range(4)]
        ax.pie(counts, labels=ACTION_NAMES, colors=ACTION_COLORS, autopct="%1.1f%%",
               startangle=90, textprops={"color": "#ddddff", "fontsize": 9})
        ax.set_title("Action Distribution")

        ax = axes[0][1]
        window = max(1, min(200, len(ep["actions"]) // 10))
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            mask = (ep["actions"] == j).astype(float)
            rolling = pd.Series(mask).rolling(window).mean().values
            ax.plot(ep["ticks"], rolling, color=col, lw=1.0, alpha=0.8, label=name)
        ax.set_title(f"Action Frequency Over Time (rolling {window})")
        ax.set_xlabel("Tick"); ax.set_ylabel("Fraction of ticks")
        ax.legend(fontsize=7); ax.grid(True)

        ax = axes[0][2]
        N = min(1000, len(ep["ticks"]))
        sl_t, sl_p, sl_a = ep["ticks"][-N:], ep["prices"][-N:], ep["actions"][-N:]
        ax.plot(sl_t, sl_p, color="#aaaacc", lw=0.8, alpha=0.7)
        for ai, col, marker in zip([0, 1, 2], ["#2ecc71", "#e74c3c", "#f39c12"], ["^", "v", "x"]):
            mask = sl_a == ai
            ax.scatter(sl_t[mask], sl_p[mask], color=col, s=15, marker=marker,
                       zorder=3, label=ACTION_NAMES[ai], alpha=0.8)
        ax.set_title(f"Signals on Price (last {N} ticks)")
        ax.set_xlabel("Tick"); ax.set_ylabel("Price")
        ax.legend(fontsize=7); ax.grid(True)

    savefig(fig, f"{ep['label']}_02_action_distribution.png")


# ── Section 3 — Trade Outcomes ──────────────────────────────────────────────

def plot_trade_outcomes(ep: dict):
    trades = ep["trades"]
    if not trades:
        print(f"  ⚠  [{ep['label']}] No trades completed — skipping Section 3.")
        return None
    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 3 — Trade Outcome Analysis")
    pnls       = np.array([t["pnl"] for t in trades])
    wins       = pnls > 0
    durations  = np.array([t["duration"] for t in trades])
    conviction = np.array([t["entry_conviction"] for t in trades])

    with plt.rc_context(STYLE):
        ax = axes[0][0]
        ax.hist(pnls[wins],  bins=30, color="#2ecc71", alpha=0.8, label="Win",  edgecolor="none")
        ax.hist(pnls[~wins], bins=30, color="#e74c3c", alpha=0.8, label="Loss", edgecolor="none")
        ax.axvline(0, color="white", lw=1, linestyle="--")
        ax.set_title(f"Trade PnL (WR={wins.mean():.1%}, n={len(trades)})")
        ax.set_xlabel("PnL (fraction)"); ax.set_ylabel("Count")
        ax.legend(fontsize=7); ax.grid(True)

        ax = axes[0][1]
        cum = np.cumsum(pnls)
        ax.plot(cum, color="#7f5af0", lw=1.5)
        ax.fill_between(range(len(cum)), 0, cum, where=cum >= 0, color="#2ecc71", alpha=0.2)
        ax.fill_between(range(len(cum)), 0, cum, where=cum < 0, color="#e74c3c", alpha=0.2)
        ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
        ax.set_title(f"Cumulative PnL (total={cum[-1]:.4%})")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL"); ax.grid(True)

        ax = axes[0][2]
        ax.hist(durations, bins=40, color="#f39c12", edgecolor="none", alpha=0.8)
        ax.axvline(np.median(durations), color="#ffd700", lw=1.5,
                   label=f"median={np.median(durations):.0f} ticks")
        ax.set_title("Holding Duration (1 tick = 4h)")
        ax.set_xlabel("Duration (ticks)"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.grid(True)

        ax = axes[1][0]
        ax.scatter(conviction[wins],  pnls[wins],  s=8, color="#2ecc71", alpha=0.5, label="Win")
        ax.scatter(conviction[~wins], pnls[~wins], s=8, color="#e74c3c", alpha=0.5, label="Loss")
        ax.axhline(0, color="white", lw=0.8, linestyle="--")
        for q_lo, q_hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.)]:
            mask = (conviction >= q_lo) & (conviction < q_hi)
            if mask.sum() > 0:
                ax.text((q_lo + q_hi) / 2, pnls.min() * 0.9, f"{wins[mask].mean():.0%}",
                        ha="center", fontsize=7, color="#ffd700")
        ax.set_title("Entry Conviction vs PnL (quintile WR in yellow)")
        ax.set_xlabel("Entry conviction (calibrated P)"); ax.set_ylabel("PnL")
        ax.legend(fontsize=7); ax.grid(True)

        ax = axes[1][1]
        ax.scatter(durations[wins],  pnls[wins],  s=8, color="#2ecc71", alpha=0.5)
        ax.scatter(durations[~wins], pnls[~wins], s=8, color="#e74c3c", alpha=0.5)
        ax.axhline(0, color="white", lw=0.8, linestyle="--")
        ax.set_title("Holding Duration vs PnL")
        ax.set_xlabel("Duration (ticks)"); ax.set_ylabel("PnL")
        if len(durations) > 1:
            ax.set_xlim(0, np.percentile(durations, 95))
        ax.grid(True)

        ax = axes[1][2]
        for side, col, lbl in [("LONG", "#2ecc71", "Long"), ("SHORT", "#e74c3c", "Short")]:
            sp = np.array([t["pnl"] for t in trades if t["side"] == side])
            if len(sp):
                ax.hist(sp, bins=25, color=col, alpha=0.7, edgecolor="none",
                        label=f"{lbl}: WR={(sp>0).mean():.1%} n={len(sp)}")
        ax.axvline(0, color="white", lw=0.8, linestyle="--")
        ax.set_title("PnL by Trade Side")
        ax.set_xlabel("PnL"); ax.set_ylabel("Count")
        ax.legend(fontsize=7); ax.grid(True)

    savefig(fig, f"{ep['label']}_03_trade_outcomes.png")
    return pnls


# ── Section 4 — Feature → Action Sensitivity ────────────────────────────────

def plot_feature_sensitivity(agent: GBTAgent, ep: dict):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 4 — Feature → Action Sensitivity (p10→p90)")
    with plt.rc_context(STYLE):
        for feat_idx, feat_name in enumerate(FEATURES):
            ax = axes[feat_idx // 3][feat_idx % 3]
            feat_vals = ep["states_arr"][:, feat_idx]
            p10, p90 = np.percentile(feat_vals, 10), np.percentile(feat_vals, 90)

            n_probe = min(300, len(ep["states_arr"]))
            probe_idx = np.random.choice(len(ep["states_arr"]), n_probe, replace=False)

            deltas = []
            for idx in probe_idx:
                s_lo = ep["states_arr"][idx].copy()
                s_hi = ep["states_arr"][idx].copy()
                s_lo[feat_idx] = p10
                s_hi[feat_idx] = p90
                _, probs_lo, _, _ = agent.actor.get_action(_to_gbt_state(s_lo))
                _, probs_hi, _, _ = agent.actor.get_action(_to_gbt_state(s_hi))
                deltas.append(probs_hi - probs_lo)
            deltas = np.array(deltas)
            means, stds = deltas.mean(axis=0), deltas.std(axis=0)

            ax.bar(range(4), means, color=ACTION_COLORS, alpha=0.85, width=0.6)
            ax.errorbar(range(4), means, yerr=stds, fmt="none", color="white", capsize=4, lw=1.5)
            ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
            ax.set_xticks(range(4)); ax.set_xticklabels(ACTION_NAMES, fontsize=8)
            ax.set_title(f"{feat_name}\n(p10={p10:.2f} → p90={p90:.2f})")
            ax.set_ylabel("Δ P(a|s)"); ax.grid(True, axis="y")

    savefig(fig, f"{ep['label']}_04_feature_sensitivity.png")


# ── Section 5 — Regime Analysis ─────────────────────────────────────────────

def plot_regime_analysis(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    mean_dev = ep["raw_features"][:, FEATURES.index("MeanDev_Scaled")]
    atr      = ep["raw_features"][:, ATR_IDX]

    regimes = {
        "Trending Up":   (mean_dev >  0.1) & (atr > 0),
        "Trending Down": (mean_dev < -0.1) & (atr > 0),
        "Ranging":       (mean_dev >= -0.1) & (mean_dev <= 0.1),
        "High Vol":      atr > np.percentile(atr, 75),
        "Low Vol":       atr < np.percentile(atr, 25),
    }
    regime_colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#f39c12", "#3498db"]

    fig, axes = make_fig(1, 3, f"[{ep['label']}] Section 5 — Regime Analysis")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        x, width = np.arange(4), 0.15
        for k, (rname, mask) in enumerate(regimes.items()):
            if mask.sum() < 10:
                continue
            counts = np.array([(ep["actions"][mask] == i).mean() for i in range(4)])
            ax.bar(x + k * width, counts, width=width, color=regime_colors[k], alpha=0.8, label=rname)
        ax.set_xticks(x + 2 * width); ax.set_xticklabels(ACTION_NAMES, fontsize=8)
        ax.set_title("Action Distribution by Regime")
        ax.set_ylabel("Fraction of ticks")
        ax.legend(fontsize=6); ax.grid(True, axis="y")

        ax = axes[0][1]
        conv_data, conv_labels = [], []
        for rname, mask in regimes.items():
            if mask.sum() > 10:
                conv_data.append(ep["max_prob"][mask])
                conv_labels.append(f"{rname}\n(n={mask.sum():,})")
        if conv_data:
            bp = ax.boxplot(conv_data, patch_artist=True, medianprops={"color": "white", "lw": 2})
            for patch, col in zip(bp["boxes"], regime_colors):
                patch.set_facecolor(col); patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(conv_labels) + 1)); ax.set_xticklabels(conv_labels, fontsize=6)
        ax.set_title("Conviction by Regime"); ax.set_ylabel("max P(a|s)")
        ax.grid(True, axis="y")

        ax = axes[0][2]
        if ep["trades"]:
            wr_vals, wr_names = [], []
            for rname, mask in regimes.items():
                regime_ticks = set(ep["ticks"][mask].tolist())
                rt = [t for t in ep["trades"] if t["entry_tick"] in regime_ticks]
                if len(rt) >= 3:
                    wr_vals.append(np.mean([t["win"] for t in rt]))
                    wr_names.append(f"{rname}\n(n={len(rt)})")
            if wr_vals:
                ax.bar(range(len(wr_vals)), wr_vals,
                       color=[regime_colors[i] for i in range(len(wr_vals))], alpha=0.8)
                ax.axhline(0.5, color="white", lw=1, linestyle="--")
                ax.set_xticks(range(len(wr_names))); ax.set_xticklabels(wr_names, fontsize=6)
                ax.set_title("Trade Win Rate by Entry Regime")
                ax.set_ylabel("Win rate"); ax.set_ylim(0, 1); ax.grid(True, axis="y")

    savefig(fig, f"{ep['label']}_05_regime_analysis.png")


# ── Section 6 — Timing Analysis ─────────────────────────────────────────────

def plot_timing_analysis(ep: dict):
    if ep["timestamps"] is None or len(ep["ticks"]) == 0:
        print(f"  ⚠  [{ep['label']}] No timestamps — skipping Section 6.")
        return
    ts    = ep["timestamps"]
    hours = np.array([t.hour for t in ts])

    fig, axes = make_fig(1, 2, f"[{ep['label']}] Section 6 — Timing Analysis")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        means = [ep["max_prob"][hours == h].mean() if (hours == h).sum() > 0 else 0 for h in range(24)]
        ax.bar(range(24), means, color="#7f5af0", alpha=0.8)
        ax.axhline(ep["max_prob"].mean(), color="#ffd700", lw=1.5, linestyle="--",
                   label=f"overall={ep['max_prob'].mean():.2f}")
        ax.set_title("Mean Conviction by Hour (UTC)")
        ax.set_xlabel("Hour"); ax.set_ylabel("Mean max P(a|s)")
        ax.set_xticks(range(24)); ax.legend(fontsize=7); ax.grid(True, axis="y")

        ax = axes[0][1]
        if ep["trades"]:
            hour_of_entry = {}
            for t in ep["trades"]:
                idx = t["entry_tick"] - ep["ticks"][0]
                if 0 <= idx < len(ts):
                    hour_of_entry.setdefault(ts[idx].hour, []).append(t["win"])
            if hour_of_entry:
                hrs = sorted(hour_of_entry.keys())
                wr  = [np.mean(hour_of_entry[h]) for h in hrs]
                ax.bar(hrs, wr, color=["#2ecc71" if w >= 0.5 else "#e74c3c" for w in wr], alpha=0.8)
                ax.axhline(0.5, color="white", lw=1, linestyle="--")
                ax.set_title("Trade Win Rate by Hour of Day")
                ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Win rate")
                ax.set_ylim(0, 1.1); ax.grid(True, axis="y")

    savefig(fig, f"{ep['label']}_06_timing_analysis.png")


# ── Section 7 — Action-State Consistency ────────────────────────────────────

def plot_action_consistency(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 7 — Action vs Market Context Consistency")
    with plt.rc_context(STYLE):
        action_masks  = {"LONG": ep["actions"] == 0, "SHORT": ep["actions"] == 1, "HOLD": ep["actions"] == 3}
        action_colors = {"LONG": ACTION_COLORS[0], "SHORT": ACTION_COLORS[1], "HOLD": ACTION_COLORS[3]}

        for fi, fname in enumerate(FEATURES):
            ax = axes[fi // 3][fi % 3]
            plot_data, plot_labels, plot_pos, plot_cols = [], [], [], []
            pos = 0
            for aname in ["LONG", "SHORT", "HOLD"]:
                mask = action_masks[aname]
                if mask.sum() > 0:
                    plot_data.append(ep["raw_features"][mask, fi])
                    plot_labels.append(f"{aname}\n(n={mask.sum():,})")
                    plot_pos.append(pos); plot_cols.append(action_colors[aname])
                    pos += 1
            if plot_data:
                parts = ax.violinplot(plot_data, positions=plot_pos, showmedians=True, showextrema=False)
                for pc, col in zip(parts["bodies"], plot_cols):
                    pc.set_facecolor(col); pc.set_alpha(0.6)
                parts["cmedians"].set_color("white")
                ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--", alpha=0.7)
                ax.set_xticks(plot_pos); ax.set_xticklabels(plot_labels, fontsize=7)
                ax.set_title(fname); ax.set_ylabel("Scaled value"); ax.grid(True, axis="y")

    savefig(fig, f"{ep['label']}_07_action_consistency.png")


# ─────────────────────────────────────────────────────────────────────────
# Calibration against ground-truth triple-barrier labels
# ─────────────────────────────────────────────────────────────────────────

def compute_calibration(ep: dict, full_labels: dict, tick_offset: int) -> dict:
    """
    Checks whether the model's calibrated P(LONG)/P(SHORT)/P(HOLD) at
    FLAT-position ticks actually resolves at that frequency, using the
    same triple-barrier ground truth main_gbt.py trains against. Only
    flat ticks are scored (prob_close is forced to 0 there by the
    action mask, giving a cheap flat/in-position flag) — CLOSE is not a
    meta-label target (see triple_barrier.build_meta_labels docstring),
    so scoring it here would compare apples to oranges.
    """
    if len(ep["ticks"]) == 0:
        return None
    flat_mask = np.isclose(ep["probs_all"][:, 2], 0.0)
    if flat_mask.sum() < 30:
        return None

    idx = ep["ticks"][flat_mask] - 0  # ticks are already full-df indices
    valid = full_labels["valid_mask"][idx]
    if valid.sum() < 30:
        return None

    slot = {0: 0, 1: 1, 3: 2}
    y_true = np.array([slot[int(a)] for a in full_labels["best_action"][idx][valid]])
    pred_probs = np.stack([
        ep["probs_all"][flat_mask, 0][valid],
        ep["probs_all"][flat_mask, 1][valid],
        ep["probs_all"][flat_mask, 3][valid],
    ], axis=1)

    report = evaluate_calibration(pred_probs, y_true, class_names=["LONG", "SHORT", "HOLD"])
    plot_reliability_diagram(pred_probs, y_true, ["LONG", "SHORT", "HOLD"],
                             os.path.join(OUT_DIR, f"{ep['label']}_reliability_diagram.png"))
    return report


# ─────────────────────────────────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────────────────────────────────

def write_summary(ep: dict, agent: GBTAgent, pnls, calib_report: dict = None):
    trades = ep["trades"]
    n = len(ep["ticks"])
    label = ep["label"]

    lines = ["=" * 62, f"  GBT DIAGNOSTIC SUMMARY — {label.upper()}",
             f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"  state_dim (trained): {agent.state_dim}",
             f"  entry_threshold: {getattr(agent.actor, 'entry_threshold', 0.5):.2f}",
             f"  classes_: {agent.actor.classes_.tolist() if agent.actor.classes_ is not None else 'n/a'}",
             "=" * 62]

    if n == 0:
        lines.append("\n  ⚠  No ticks collected — split was too short.")
        text = "\n".join(lines); print(text); return

    lines.append("\n── POLICY HEALTH ──────────────────────────────────────────")
    lines.append(f"  Mean conviction (max P):     {ep['max_prob'].mean():.3f}")
    lines.append(f"  Conviction > 0.50:           {(ep['max_prob'] > 0.50).mean():.1%}")
    lines.append(f"  Conviction > 0.70:           {(ep['max_prob'] > 0.70).mean():.1%}")
    lines.append(f"  Mean entropy:                {ep['entropies'].mean():.3f} bits")

    if calib_report is not None:
        lines.append("\n── CALIBRATION vs TRIPLE-BARRIER GROUND TRUTH ─────────────")
        lines.append(f"  Macro ECE   : {calib_report['macro_ece']:.4f}  (lower is better; <0.05 is good)")
        lines.append(f"  Macro Brier : {calib_report['macro_brier']:.4f}")
        for name, m in calib_report["per_class"].items():
            lines.append(f"    {name:<6}: ECE={m['ece']:.4f}  Brier={m['brier']:.4f}  "
                         f"n_pos={m['n_positive']}/{m['n_total']}")
    else:
        lines.append("\n  ⚠  Not enough flat-position ticks with valid forward labels "
                     "to compute a calibration check for this window.")

    lines.append("\n── ACTION DISTRIBUTION ────────────────────────────────────")
    for j, aname in enumerate(ACTION_NAMES):
        count = (ep["actions"] == j).sum()
        lines.append(f"  {aname:<6}: {count:>6,}  ({count/n:.1%})")

    if trades:
        pnl_arr = np.array([t["pnl"] for t in trades])
        wins = pnl_arr > 0
        dur_arr = np.array([t["duration"] for t in trades])
        longs  = [t for t in trades if t["side"] == "LONG"]
        shorts = [t for t in trades if t["side"] == "SHORT"]
        cum = np.cumsum(pnl_arr)
        drawdown = cum - np.maximum.accumulate(cum)

        lines.append("\n── TRADE OUTCOMES ─────────────────────────────────────────")
        lines.append(f"  Total trades:                {len(trades):,}")
        lines.append(f"  Win rate:                    {wins.mean():.1%}")
        lines.append(f"  Mean PnL per trade:          {pnl_arr.mean():.4%}")
        lines.append(f"  Median PnL per trade:        {np.median(pnl_arr):.4%}")
        lines.append(f"  Total cumulative PnL:        {pnl_arr.sum():.4%}")
        lines.append(f"  Max drawdown:                {drawdown.min():.4%}")
        if pnl_arr.std() > 0:
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(len(trades))
            lines.append(f"  Sharpe (simplified):         {sharpe:.3f}")
        lines.append(f"  Median hold duration:        {np.median(dur_arr):.0f} ticks "
                     f"({np.median(dur_arr) * 4:.0f} hrs)")
        if longs:
            lp = np.array([t["pnl"] for t in longs])
            lines.append(f"  LONG  win rate:              {(lp>0).mean():.1%} (n={len(longs)})")
        if shorts:
            sp = np.array([t["pnl"] for t in shorts])
            lines.append(f"  SHORT win rate:              {(sp>0).mean():.1%} (n={len(shorts)})")

        conv = np.array([t["entry_conviction"] for t in trades])
        hi_conv, lo_conv = conv >= np.percentile(conv, 66), conv < np.percentile(conv, 33)
        lines.append("\n── CONVICTION EDGE ────────────────────────────────────────")
        lines.append(f"  Top-33% conviction WR:       {wins[hi_conv].mean():.1%}")
        if lo_conv.sum() > 0:
            lines.append(f"  Bot-33% conviction WR:       {wins[lo_conv].mean():.1%}")
            edge = wins[hi_conv].mean() - wins[lo_conv].mean()
            lines.append(f"  Conviction edge (WR delta):  {edge:+.1%}")
            lines.append(f"  → {'Conviction is predictive ✓' if edge > 0.03 else 'Conviction not yet predictive'}")

        if wins.mean() > 0.80:
            lines.append("\n⚠ WARNING: Win rate >80% strongly suggests regime overfitting.")

    lines.append("\n" + "=" * 62)
    text = "\n".join(lines)
    summary_path = os.path.join(OUT_DIR, f"diagnostic_summary_{label}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n  ✓  {summary_path}")


def run_full_suite(agent, ep, full_labels, calib_report=None):
    if len(ep["ticks"]) == 0:
        print(f"  ⚠  [{ep['label']}] Empty episode — skipping all plots.")
        return None, None
    if calib_report is None and full_labels is not None:
        calib_report = compute_calibration(ep, full_labels, tick_offset=0)
    plot_confidence_calibration(ep, calib_report)
    plot_action_distribution(ep)
    pnls = plot_trade_outcomes(ep)
    plot_feature_sensitivity(agent, ep)
    plot_regime_analysis(ep)
    plot_timing_analysis(ep)
    plot_action_consistency(ep)
    return pnls, calib_report


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    global OUT_DIR

    parser = argparse.ArgumentParser(description="GBT Diagnostic Suite")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, choices=AVAILABLE_SYMBOLS,
                        help=f"Trading pair whose data/checkpoint to diagnose "
                             f"(default: %(default)s). Must match the symbol "
                             f"the checkpoint under --checkpoint was actually "
                             f"trained on — see main_gbt.py's --symbol.")
    parser.add_argument("--split", default="val", choices=["train", "val", "both"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to the .joblib checkpoint to diagnose. "
                             "Defaults to outcomes/gbt/<symbol>/gbt_agent_best.joblib "
                             "for whichever --symbol was given.")
    parser.add_argument("--frozen-data", action="store_true",
                        help="Use the existing master CSV(s) as-is instead of "
                             "refetching/appending new raw data (mirrors "
                             "main_gbt.py's --frozen-data).")
    args = parser.parse_args()

    symbol = args.symbol

    # MULTI-SYMBOL SUPPORT: every symbol's checkpoint/CPCV-summary/
    # diagnostics live under their own subfolder — reuse main_gbt's
    # symbol_out_dir() so the layout is guaranteed identical to what
    # main_gbt.py actually wrote. OUT_DIR (this file's own diagnostics/
    # output folder) is reassigned here — every plot/summary helper
    # above reads the OUT_DIR global at call time, so this takes effect
    # for the rest of the run without threading an out_dir parameter
    # through each of them.
    symbol_dir = gbt_cfg.symbol_out_dir(symbol)
    OUT_DIR = os.path.join(symbol_dir, "diagnostics")
    os.makedirs(OUT_DIR, exist_ok=True)

    checkpoint = args.checkpoint or os.path.join(symbol_dir, "gbt_agent_best.joblib")
    cpcv_summary_path = os.path.join(symbol_dir, "cpcv_summary.json")

    sep = "=" * 62
    print(f"\n{sep}\n  GBT DIAGNOSTIC SUITE\n  Symbol: {symbol}\n"
          f"  Checkpoint: {checkpoint}\n  Split: {args.split}\n"
          f"  Output: {OUT_DIR}\n{sep}\n")

    print("Loading data...")
    df, indicators_arr, prices_arr, extra_context_arr, n = load_data(
        symbol=symbol, frozen=args.frozen_data,
    )

    print("Loading GBT checkpoint...")
    agent = GBTAgent(state_dim=1, action_dim=ACTION_DIM)  # placeholder, load() overwrites
    agent.load(checkpoint)

    market_dim  = len(FEATURES) * 2 * len(PACES)
    context_dim = extra_context_arr.shape[1] if extra_context_arr is not None else 0
    expected_state_dim = market_dim + context_dim
    if agent.state_dim != expected_state_dim:
        raise RuntimeError(
            f"Checkpoint state_dim={agent.state_dim} doesn't match this codebase's "
            f"rebuilt state_dim={expected_state_dim} (market={market_dim}, "
            f"context={context_dim}) for symbol='{symbol}'. Either "
            f"ENABLE_MULTI_TIMEFRAME/CONTEXT_PACES/CONTEXT_TIMEFRAMES in "
            f"main_gbt.py changed since this checkpoint was trained, the raw "
            f"data under data/raw/{symbol}/<timeframe>/ is different, or "
            f"--checkpoint/--symbol point at a checkpoint trained for a "
            f"DIFFERENT symbol. Refusing to run diagnostics on a mismatched "
            f"state — see gbt_agent.py's GBTAgent for what state_dim is "
            f"actually used for."
        )
    print(f"  ✓  state_dim confirmed: {agent.state_dim}  "
          f"(multi_timeframe={'yes' if context_dim else 'no'})")

    ranges = make_holdout_range(n)
    print(f"  Train range (state idx): {ranges['train_range']}  |  "
          f"Holdout range: {ranges['holdout_range']}  |  "
          f"lookback_ticks={ranges['lookback_ticks']}")

    # Ground-truth labels for calibration checking, computed once over the
    # full price series (cheap; O(n * MAX_HOLDING)).
    atr_full = indicators_arr[:, ATR_IDX]
    full_labels = build_meta_labels(prices_arr, atr_full, tp_mult=TP_MULT,
                                     sl_mult=SL_MULT, max_holding=MAX_HOLDING,
                                     commission=COMMISSION)

    if os.path.exists(cpcv_summary_path):
        with open(cpcv_summary_path) as f:
            cpcv = json.load(f)
        print(f"\n{sep}\n  CPCV SUMMARY (from training — {cpcv_summary_path})\n{sep}")
        print(f"  Paths          : {cpcv['n_paths']}  ({cpcv['n_paths_positive']} positive)")
        print(f"  Mean test P/L  : {cpcv['mean_test_pnl']:+.4%}")
        print(f"  Std  test P/L  : {cpcv['std_test_pnl']:.4%}")
        print(f"  Min  test P/L  : {cpcv['min_test_pnl']:+.4%}")
        print(f"  PBO            : {cpcv.get('pbo', float('nan')):.1%}")
        print(f"{sep}\n")
    else:
        print(f"\n  ⚠  No {cpcv_summary_path} found — run "
              f"`python main_gbt.py --symbol {symbol}`'s full pipeline to "
              f"generate CPCV robustness metrics.\n")

    if args.split in ("val", "both"):
        print("\n" + "-" * 62 + "\n  Full diagnostic: VAL (held-out)\n" + "-" * 62)
        t_start, t_end = _state_range_to_ticks(*ranges["holdout_range"])
        ep = collect_episode(agent, indicators_arr, prices_arr, extra_context_arr,
                             df["Open_time"].values, t_start, t_end, "val")
        print(f"  Ticks: {len(ep['ticks']):,}  |  Trades: {len(ep['trades']):,}")
        pnls, calib_report = run_full_suite(agent, ep, full_labels)
        write_summary(ep, agent, pnls, calib_report)

    if args.split in ("train", "both"):
        print("\n" + "-" * 62 + "\n  Full diagnostic: TRAIN\n" + "-" * 62)
        t_start, t_end = _state_range_to_ticks(*ranges["train_range"])
        ep = collect_episode(agent, indicators_arr, prices_arr, extra_context_arr,
                             df["Open_time"].values, t_start, t_end, "train")
        print(f"  Ticks: {len(ep['ticks']):,}  |  Trades: {len(ep['trades']):,}")
        pnls, calib_report = run_full_suite(agent, ep, full_labels)
        write_summary(ep, agent, pnls, calib_report)

    print("\n" + sep + f"\n  Diagnostic complete. All outputs in: {OUT_DIR}/\n" + sep + "\n")


if __name__ == "__main__":
    main()

# python diagnostic_gbt.py - -split val
# python diagnostic_gbt.py - -symbol BTCUSDT - -split val
# python diagnostic_gbt.py - -split both - -checkpoint
# outcomes / gbt / XRPUSDT / gbt_agent_best.joblib