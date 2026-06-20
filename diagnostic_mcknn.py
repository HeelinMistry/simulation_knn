"""
diagnostic_mcknn.py
─────────────────────
Full post-training diagnostic suite for the Monte Carlo k-NN trading agent.

Run from project root:
    python diagnostic_mcknn.py [--split val|train|both] [--checkpoint outcomes/mc_knn_agent_best.npz]

Mapping vs diagnostic.py
───────────────────────────
  Sections 1, 3, 4, 5, 6, 7, 8, 10 — STRUCTURALLY UNCHANGED. Same 10-
    section layout, same plot titles/axes, same file-naming convention.
    Section 5 (Feature->Action Sensitivity) and Section 6 (state-
    conditional probability maps) still work as-is because they only
    depend on agent.actor(...)-equivalent outputs (here: vote-shares
    from MCKNNPolicy), which MCKNNAgent.actor provides with the same
    call signature.

  Section 2  "Q-Value Health"            -> "Neighbor-Return Distribution
                                              Health" — replaces Q1/Q2
                                              scatter and min-Q histograms
                                              with the distribution of
                                              k-NN neighbor MC returns and
                                              policy-vs-best-neighbor-
                                              return match rate.

  Section 9  "Critic Disagreement vs      -> "Neighbor Agreement (Vote
              Outcome"                       Margin) vs Outcome" —
                                              replaces |Q1-Q2| with
                                              vote_margin (winning action's
                                              vote weight minus runner-up),
                                              which plays the same
                                              "how confident/contested was
                                              this decision" diagnostic role.

  Wherever the original called `agent.critic(s_tensor)` to get Q1/Q2,
  this version calls `agent.memory.query(state)` to get vote info,
  collected per-tick during the same single deterministic pass.

FIX 1-5 notes from diagnostic.py are preserved as-is (regime-balanced
split, bear/bull reporting, guarded random-baseline execution, per-label
summary files, corrected annualised Sharpe) since none of those fixes
were specific to SAC.
"""

import argparse
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

from agents.mc_knn_agent     import MCKNNAgent
from agents.unified_executor import UnifiedExecutor
from data.data_manager       import update_master_data

# ── Configuration — must match main_mcknn.py exactly ──────────────────────────
BEST_PATH    = "outcomes/mc_knn_agent_best.npz"
FEATURES     = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
                "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES        = (1, 6, 42, 90)
STATE_DIM    = (len(FEATURES) * 2 * len(PACES)) + 2
ACTION_DIM   = 4
ACTION_NAMES  = ["LONG", "SHORT", "CLOSE", "HOLD"]
ACTION_COLORS = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]
WARMUP_IDX   = 128        # must match main_mcknn.py
GAMMA        = 0.97
K_NEIGHBORS  = 25
OUT_DIR      = "outcomes/diagnostics_mcknn"

# ── Regime-balanced split — MUST match main_mcknn.py exactly ─────────────────
VAL_YEARS   = [2022, 2025]

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data collection pass
# ─────────────────────────────────────────────────────────────────────────────

def collect_episode(agent: MCKNNAgent, df: pd.DataFrame, label: str) -> dict:
    """
    Full deterministic pass through df.
    Returns a dict of parallel arrays (one entry per tick from WARMUP_IDX+1).

    Field-by-field mapping vs diagnostic.py's collect_episode:
      q1_chosen, q2_chosen, q1_all, q2_all, q_disagreement
        -> neighbor_return_mean, neighbor_return_std, vote_raw_all,
           vote_margin
      (max_prob, entropies, probs_all, actions, trades, pnl_curve,
       raw_features, unrealized_pnl, states_arr, timestamps — all kept
       with identical names/shapes since they don't depend on critic
       internals)
    """
    executor = UnifiedExecutor(
        name=label, agent=agent, paces=PACES,
        deterministic=True, num_indicators=len(FEATURES)
    )

    indicators_arr = df[FEATURES].values.astype(np.float32)
    prices_arr     = df["Close"].values.astype(np.float32)
    n              = len(df)

    if n <= WARMUP_IDX + 1:
        print(f"  ⚠  {label}: only {n} rows — too few to evaluate (need >{WARMUP_IDX+1}).")
        return _empty_episode(label)

    executor.aggregator.tick = 0
    executor.aggregator.warm_up_all(indicators_arr, WARMUP_IDX)

    ticks, prices, actions, entropies               = [], [], [], []
    prob_long, prob_short, prob_close, prob_hold    = [], [], [], []
    # kNN analogues of q1_chosen/q2_chosen/q1_all/q2_all/q_disagreement:
    neighbor_return_mean, neighbor_return_std        = [], []
    vote_raw_all, vote_margin_arr                    = [], []
    raw_features, unrealized_pnl, states_arr        = [], [], []

    trades     = []
    open_tick  = None
    open_price = None
    open_side  = None

    for i in range(WARMUP_IDX + 1, n):
        ind   = indicators_arr[i]
        price = prices_arr[i]

        action, probs, realised_pnl, s_t = executor.step(ind, price, tick=i)

        # kNN analogue of the critic forward pass: query the memory bank
        # at this exact state to get neighbor return stats + vote info.
        _q_action, _q_probs, q_info = agent.memory.query(s_t, k=K_NEIGHBORS)
        nbr_returns = q_info["neighbor_returns"]
        vote_raw    = q_info["vote_raw"]
        margin      = q_info["vote_margin"]

        ent = -np.sum(probs * np.log2(probs + 1e-9))

        ticks.append(i)
        prices.append(price)
        actions.append(action)
        entropies.append(ent)
        prob_long.append(probs[0])
        prob_short.append(probs[1])
        prob_close.append(probs[2])
        prob_hold.append(probs[3])

        neighbor_return_mean.append(float(nbr_returns.mean()) if len(nbr_returns) else 0.0)
        neighbor_return_std.append(float(nbr_returns.std()) if len(nbr_returns) else 0.0)
        vote_raw_all.append(np.where(np.isfinite(vote_raw), vote_raw, 0.0).copy())
        vote_margin_arr.append(margin)

        raw_features.append(ind.copy())
        unrealized_pnl.append(float(s_t[-1]))
        states_arr.append(s_t.copy())

        if realised_pnl != 0.0:
            if open_tick is not None:
                duration         = i - open_tick
                local_idx        = open_tick - WARMUP_IDX - 1
                entry_conviction = float(
                    prob_long[local_idx] if open_side == "LONG"
                    else prob_short[local_idx]
                ) if local_idx >= 0 else 0.0
                trades.append({
                    "entry_tick":       open_tick,
                    "exit_tick":        i,
                    "duration":         duration,
                    "side":             open_side,
                    "pnl":              realised_pnl,
                    "entry_price":      open_price,
                    "exit_price":       price,
                    "entry_conviction": entry_conviction,
                    "exit_conviction":  float(probs[action]),
                    "vote_margin_entry": vote_margin_arr[local_idx] if local_idx >= 0 else 0.0,
                    "win":              realised_pnl > 0,
                    "rsi_at_entry":     float(raw_features[local_idx][0]) if local_idx >= 0 else 0.0,
                    "macd_at_entry":    float(raw_features[local_idx][1]) if local_idx >= 0 else 0.0,
                })
            open_tick = open_price = open_side = None

        if action in (0, 1) and executor.current_side is not None and open_tick is None:
            open_tick  = i
            open_price = price
            open_side  = executor.current_side

    ticks          = np.array(ticks,          dtype=np.int32)
    prices         = np.array(prices,         dtype=np.float32)
    actions        = np.array(actions,        dtype=np.int32)
    entropies      = np.array(entropies,      dtype=np.float32)
    prob_long      = np.array(prob_long,      dtype=np.float32)
    prob_short     = np.array(prob_short,     dtype=np.float32)
    prob_close     = np.array(prob_close,     dtype=np.float32)
    prob_hold      = np.array(prob_hold,      dtype=np.float32)
    neighbor_return_mean = np.array(neighbor_return_mean, dtype=np.float32)
    neighbor_return_std  = np.array(neighbor_return_std,  dtype=np.float32)
    vote_raw_all   = np.array(vote_raw_all,   dtype=np.float32)  # (n, action_dim)
    vote_margin_arr = np.array(vote_margin_arr, dtype=np.float32)
    raw_features   = np.array(raw_features,   dtype=np.float32)
    unrealized_pnl = np.array(unrealized_pnl, dtype=np.float32)
    states_arr     = np.array(states_arr,     dtype=np.float32)

    probs_all = np.stack([prob_long, prob_short, prob_close, prob_hold], axis=1)
    max_prob  = probs_all.max(axis=1)

    pnl_curve = np.zeros(len(ticks))
    for t in trades:
        pnl_curve[t["exit_tick"] - WARMUP_IDX - 1:] += t["pnl"]

    timestamps = None
    if "Open_time" in df.columns:
        try:
            ts = pd.to_datetime(df["Open_time"].iloc[WARMUP_IDX + 1:], errors="coerce")
            timestamps = ts.tolist()
        except Exception:
            timestamps = None

    return {
        "label":          label,
        "ticks":          ticks,
        "prices":         prices,
        "actions":        actions,
        "entropies":      entropies,
        "probs_all":      probs_all,
        "max_prob":       max_prob,
        # kNN analogues of q1_chosen/q2_chosen/q1_all/q2_all/q_disagreement:
        "neighbor_return_mean": neighbor_return_mean,
        "neighbor_return_std":  neighbor_return_std,
        "vote_raw_all":          vote_raw_all,
        "vote_margin":           vote_margin_arr,
        "raw_features":   raw_features,
        "unrealized_pnl": unrealized_pnl,
        "states_arr":     states_arr,
        "trades":         trades,
        "pnl_curve":      pnl_curve,
        "timestamps":     timestamps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data split helper — IDENTICAL to diagnostic.py
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(df: pd.DataFrame) -> dict:
    """Identical to diagnostic.py's make_splits — copied verbatim."""
    df = df.copy()
    df["_year"] = pd.to_datetime(df["Open_time"], errors="coerce").dt.year

    val_mask   = df["_year"].isin(VAL_YEARS)
    train_mask = ~val_mask

    train_df    = df[train_mask].drop(columns=["_year"]).reset_index(drop=True)
    val_2022_df = df[df["_year"] == 2022].drop(columns=["_year"]).reset_index(drop=True)
    val_2025_df = df[df["_year"] == 2025].drop(columns=["_year"]).reset_index(drop=True)

    bear_df = df[df["_year"] == 2022].drop(columns=["_year"]).reset_index(drop=True)
    bull_df = df[df["_year"].isin([2020, 2021, 2024])].drop(
                 columns=["_year"]).reset_index(drop=True)

    train_years = sorted(df[train_mask]["_year"].dropna().unique().tolist())
    val_years   = sorted(df[val_mask]["_year"].dropna().unique().tolist())

    print(f"  Train: {len(train_df):,} rows  |  years: {train_years}")
    print(f"  Val 2022: {len(val_2022_df):,} rows  |  "
          f"Val 2025: {len(val_2025_df):,} rows  |  years: {val_years}")
    print(f"  Bear holdout (2022):          {len(bear_df):,} rows")
    print(f"  Bull holdout (2020+2021+2024): {len(bull_df):,} rows\n")

    return {
        "train":      train_df,
        "val_2022":   val_2022_df,
        "val_2025":   val_2025_df,
        "bear_2022":  bear_df,
        "bull_trend": bull_df,
    }


def _empty_episode(label: str) -> dict:
    """Safe empty episode dict — kNN field names instead of q1/q2 fields."""
    empty = np.array([], dtype=np.float32)
    return {
        "label": label, "ticks": np.array([], dtype=np.int32),
        "prices": empty, "actions": np.array([], dtype=np.int32),
        "entropies": empty,
        "probs_all": np.zeros((0, 4), dtype=np.float32),
        "max_prob": empty,
        "neighbor_return_mean": empty, "neighbor_return_std": empty,
        "vote_raw_all": np.zeros((0, 4), dtype=np.float32),
        "vote_margin": empty,
        "raw_features": np.zeros((0, 6), dtype=np.float32),
        "unrealized_pnl": empty,
        "states_arr": np.zeros((0, STATE_DIM), dtype=np.float32),
        "trades": [], "pnl_curve": empty, "timestamps": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers — IDENTICAL to diagnostic.py
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "axes.facecolor":   "#1a1a2e",
    "figure.facecolor": "#0f0f1a",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccee",
    "xtick.color":      "#aaaacc",
    "ytick.color":      "#aaaacc",
    "text.color":       "#ddddff",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
}


def make_fig(rows, cols, title, figsize=None):
    with plt.rc_context(STYLE):
        fs = figsize or (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=fs, squeeze=False)
        fig.suptitle(title, color="#ffffff", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor(STYLE["figure.facecolor"])
    return fig, axes


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=STYLE["figure.facecolor"])
    plt.close(fig)
    print(f"  ✓  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Policy Confidence & Entropy  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_entropy(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 1 — Policy Confidence & Entropy")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        ax.hist(ep["max_prob"], bins=50, color="#7f5af0", edgecolor="none", alpha=0.85)
        ax.axvline(0.5, color="#ff6b6b", lw=1.5, linestyle="--", label="50% conviction")
        ax.axvline(np.mean(ep["max_prob"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['max_prob']):.2f}")
        ax.set_title("Max Vote-Share (\"Conviction\") Distribution")
        ax.set_xlabel("max vote-share(a|s)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[0][1]
        ax.hist(ep["entropies"], bins=50, color="#2cb67d", edgecolor="none", alpha=0.85)
        target_bits = 0.75 * np.log2(4)
        ax.axvline(target_bits, color="#ff6b6b", lw=1.5, linestyle="--",
                   label=f"target={target_bits:.2f}b")
        ax.axvline(np.mean(ep["entropies"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['entropies']):.2f}b")
        ax.set_title("Vote-Share Entropy Distribution (bits)")
        ax.set_xlabel("H[vote-share(·|s)] bits")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[0][2]
        window = max(1, min(200, len(ep["max_prob"]) // 10))
        rolling = pd.Series(ep["max_prob"]).rolling(window).mean().values
        ax.plot(ep["ticks"], rolling, color="#7f5af0", lw=1.0)
        ax.axhline(0.5, color="#ff6b6b", lw=1, linestyle="--")
        ax.set_title(f"Conviction Over Time (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Mean max vote-share")
        ax.grid(True)

        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            if j < 3:
                ax = axes[1][j]
                ax.hist(ep["probs_all"][:, j], bins=40, color=col,
                        edgecolor="none", alpha=0.75, label=name)
                ax.set_title(f"vote-share({name}|s) distribution")
                ax.set_xlabel("Vote-share")
                ax.set_ylabel("Frequency")
                ax.grid(True)

        ax = axes[1][2]
        ax.hist(ep["probs_all"][:, 3], bins=40, color=ACTION_COLORS[3],
                edgecolor="none", alpha=0.55, label="HOLD")
        ax.set_title("vote-share(HOLD|s) distribution")
        ax.set_xlabel("Vote-share")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

    savefig(fig, f"{ep['label']}_01_confidence_entropy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Neighbor-Return Distribution Health
# (replaces diagnostic.py's "Q-Value Health" — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

def plot_neighbor_return_health(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 2 — Neighbor-Return Distribution Health")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        ax.scatter(ep["neighbor_return_mean"], ep["neighbor_return_std"],
                   s=2, alpha=0.2, color="#7f5af0")
        ax.set_title("Neighbor Return Mean vs Std (per tick)")
        ax.set_xlabel("mean(neighbor MC returns)")
        ax.set_ylabel("std(neighbor MC returns)")
        ax.grid(True)

        ax = axes[0][1]
        ax.hist(ep["vote_margin"], bins=50, color="#f25f4c", edgecolor="none", alpha=0.8)
        ax.axvline(np.mean(ep["vote_margin"]), color="#ffd700", lw=1.5,
                   label=f"mean={np.mean(ep['vote_margin']):.4f}")
        ax.set_title("Vote Margin per Tick (winner − runner-up weight)")
        ax.set_xlabel("vote_margin")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[0][2]
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            ax.hist(ep["vote_raw_all"][:, j], bins=40, color=col, alpha=0.55,
                    label=name, edgecolor="none")
        ax.set_title("Raw Vote Weight Distribution per Action")
        ax.set_xlabel("vote weight")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[1][0]
        best_vote_action = ep["vote_raw_all"].argmax(axis=1)
        match_rate = (best_vote_action == ep["actions"]).mean()
        confusion = np.zeros((4, 4), dtype=np.int32)
        for pa, qa in zip(ep["actions"], best_vote_action):
            confusion[pa, qa] += 1
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(ACTION_NAMES, fontsize=7)
        ax.set_yticklabels(ACTION_NAMES, fontsize=7)
        ax.set_xlabel("Best-Vote Action")
        ax.set_ylabel("Policy Action")
        ax.set_title(f"Policy vs Best-Vote  (match={match_rate:.1%})")
        for r in range(4):
            for c in range(4):
                ax.text(c, r, str(confusion[r, c]), ha="center",
                        va="center", fontsize=7, color="white")

        ax = axes[1][1]
        window = max(1, min(200, len(ep["neighbor_return_mean"]) // 10))
        rolling = pd.Series(ep["neighbor_return_mean"]).rolling(window).mean().values
        ax.plot(ep["ticks"], rolling, color="#2cb67d", lw=1.0)
        ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
        ax.set_title(f"Mean Neighbor Return of Chosen-Tick Query (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("mean neighbor MC return")
        ax.grid(True)

        ax = axes[1][2]
        if ep["trades"]:
            entry_margin, outcomes = [], []
            for t in ep["trades"]:
                idx = t["entry_tick"] - WARMUP_IDX - 1
                if 0 <= idx < len(ep["vote_margin"]):
                    entry_margin.append(float(ep["vote_margin"][idx]))
                    outcomes.append(t["pnl"])
            if entry_margin:
                em, pl = np.array(entry_margin), np.array(outcomes)
                wins = pl > 0
                ax.hist(em[wins],  bins=20, color="#2ecc71", alpha=0.7,
                        label="Win",  edgecolor="none")
                ax.hist(em[~wins], bins=20, color="#e74c3c", alpha=0.7,
                        label="Loss", edgecolor="none")
                ax.set_title("Entry Vote Margin: Wins vs Losses")
                ax.set_xlabel("vote_margin at entry")
                ax.set_ylabel("Trade count")
                ax.legend(fontsize=8)
                ax.grid(True)

    savefig(fig, f"{ep['label']}_02_neighbor_return_health.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Action Distribution & Sequencing  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_distribution(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 3 — Action Distribution & Sequencing")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        counts = [(ep["actions"] == i).sum() for i in range(4)]
        ax.pie(counts, labels=ACTION_NAMES, colors=ACTION_COLORS,
               autopct="%1.1f%%", startangle=90,
               textprops={"color": "#ddddff", "fontsize": 9})
        ax.set_title("Action Distribution")

        ax = axes[0][1]
        window = max(1, min(200, len(ep["actions"]) // 10))
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            mask = (ep["actions"] == j).astype(float)
            rolling = pd.Series(mask).rolling(window).mean().values
            ax.plot(ep["ticks"], rolling, color=col, lw=1.0, alpha=0.8, label=name)
        ax.set_title(f"Action Frequency Over Time (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Fraction of ticks")
        ax.legend(fontsize=7)
        ax.grid(True)

        ax = axes[0][2]
        trans = np.zeros((4, 4), dtype=np.float32)
        for curr, nxt in zip(ep["actions"][:-1], ep["actions"][1:]):
            trans[curr, nxt] += 1
        row_sums = trans.sum(axis=1, keepdims=True) + 1e-9
        trans_pct = trans / row_sums
        im = ax.imshow(trans_pct, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([f"→{n}" for n in ACTION_NAMES], fontsize=7)
        ax.set_yticklabels(ACTION_NAMES, fontsize=7)
        ax.set_title("Action Transition Matrix")
        for r in range(4):
            for c in range(4):
                ax.text(c, r, f"{trans_pct[r,c]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="black" if trans_pct[r, c] > 0.5 else "white")
        plt.colorbar(im, ax=ax)

        ax = axes[1][0]
        runs = []
        cur_run = 1
        for i in range(1, len(ep["actions"])):
            if ep["actions"][i] == 3 and ep["actions"][i-1] == 3:
                cur_run += 1
            else:
                if ep["actions"][i-1] == 3:
                    runs.append(cur_run)
                cur_run = 1
        if runs:
            ax.hist(runs, bins=min(50, max(runs)), color="#95a5a6",
                    edgecolor="none", alpha=0.8)
            ax.set_title(f"Consecutive HOLD Runs  (median={np.median(runs):.0f})")
            ax.set_xlabel("Run length (ticks)")
            ax.set_ylabel("Count")
            ax.set_xlim(0, np.percentile(runs, 95) if len(runs) > 1 else 10)
            ax.grid(True)

        ax = axes[1][1]
        N = min(1000, len(ep["ticks"]))
        sl_t = ep["ticks"][-N:]
        sl_p = ep["prices"][-N:]
        sl_a = ep["actions"][-N:]
        ax.plot(sl_t, sl_p, color="#aaaacc", lw=0.8, alpha=0.7)
        for ai, col, marker in zip([0, 1, 2], ["#2ecc71", "#e74c3c", "#f39c12"],
                                   ["^", "v", "x"]):
            mask = sl_a == ai
            ax.scatter(sl_t[mask], sl_p[mask], color=col, s=15,
                       marker=marker, zorder=3, label=ACTION_NAMES[ai], alpha=0.8)
        ax.set_title(f"Signals on Price (last {N} ticks)")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Price")
        ax.legend(fontsize=7)
        ax.grid(True)

        ax = axes[1][2]
        for j, (name, col) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            mask = ep["actions"] == j
            if mask.sum() > 0:
                ax.hist(ep["max_prob"][mask], bins=30, color=col, alpha=0.6,
                        label=f"{name} (n={mask.sum()})", edgecolor="none")
        ax.set_title("Conviction (max vote-share) by Action Type")
        ax.set_xlabel("max vote-share(a|s)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)
        ax.grid(True)

    savefig(fig, f"{ep['label']}_03_action_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Trade Outcome Analysis  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_trade_outcomes(ep: dict):
    trades = ep["trades"]
    if not trades:
        print(f"  ⚠  [{ep['label']}] No trades completed — skipping Section 4.")
        return None

    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 4 — Trade Outcome Analysis")
    pnls       = np.array([t["pnl"] for t in trades])
    wins       = pnls > 0
    durations  = np.array([t["duration"] for t in trades])
    conviction = np.array([t["entry_conviction"] for t in trades])

    with plt.rc_context(STYLE):
        ax = axes[0][0]
        ax.hist(pnls[wins],  bins=30, color="#2ecc71", alpha=0.8,
                label="Win",  edgecolor="none")
        ax.hist(pnls[~wins], bins=30, color="#e74c3c", alpha=0.8,
                label="Loss", edgecolor="none")
        ax.axvline(0, color="white", lw=1, linestyle="--")
        ax.axvline(pnls.mean(), color="#ffd700", lw=1.5,
                   label=f"mean={pnls.mean():.4%}")
        ax.set_title(f"Trade PnL  (WR={wins.mean():.1%}, n={len(trades)})")
        ax.set_xlabel("PnL (fraction)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True)

        ax = axes[0][1]
        cum = np.cumsum(pnls)
        ax.plot(cum, color="#7f5af0", lw=1.5)
        ax.fill_between(range(len(cum)), 0, cum,
                        where=cum >= 0, color="#2ecc71", alpha=0.2)
        ax.fill_between(range(len(cum)), 0, cum,
                        where=cum < 0, color="#e74c3c", alpha=0.2)
        ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
        ax.set_title(f"Cumulative PnL  (total={cum[-1]:.4%})")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True)

        ax = axes[0][2]
        ax.hist(durations, bins=40, color="#f39c12", edgecolor="none", alpha=0.8)
        ax.axvline(np.median(durations), color="#ffd700", lw=1.5,
                   label=f"median={np.median(durations):.0f} ticks")
        ax.set_title("Holding Duration (1 tick = 4 h)")
        ax.set_xlabel("Duration (ticks)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[1][0]
        ax.scatter(conviction[wins],  pnls[wins],  s=8, color="#2ecc71",
                   alpha=0.5, label="Win")
        ax.scatter(conviction[~wins], pnls[~wins], s=8, color="#e74c3c",
                   alpha=0.5, label="Loss")
        ax.axhline(0, color="white", lw=0.8, linestyle="--")
        for q_lo, q_hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.)]:
            mask = (conviction >= q_lo) & (conviction < q_hi)
            if mask.sum() > 0:
                ax.text((q_lo + q_hi) / 2, pnls.min() * 0.9,
                        f"{wins[mask].mean():.0%}", ha="center",
                        fontsize=7, color="#ffd700")
        ax.set_title("Conviction vs PnL (quintile WR in yellow)")
        ax.set_xlabel("Entry conviction")
        ax.set_ylabel("PnL")
        ax.legend(fontsize=7)
        ax.grid(True)

        ax = axes[1][1]
        ax.scatter(durations[wins],  pnls[wins],  s=8, color="#2ecc71", alpha=0.5)
        ax.scatter(durations[~wins], pnls[~wins], s=8, color="#e74c3c", alpha=0.5)
        ax.axhline(0, color="white", lw=0.8, linestyle="--")
        ax.set_title("Holding Duration vs PnL")
        ax.set_xlabel("Duration (ticks)")
        ax.set_ylabel("PnL")
        if len(durations) > 1:
            ax.set_xlim(0, np.percentile(durations, 95))
        ax.grid(True)

        ax = axes[1][2]
        for side, col, lbl in [("LONG", "#2ecc71", "Long"),
                                ("SHORT", "#e74c3c", "Short")]:
            sp = np.array([t["pnl"] for t in trades if t["side"] == side])
            if len(sp):
                wr = (sp > 0).mean()
                ax.hist(sp, bins=25, color=col, alpha=0.7, edgecolor="none",
                        label=f"{lbl}: WR={wr:.1%} n={len(sp)}")
        ax.axvline(0, color="white", lw=0.8, linestyle="--")
        ax.set_title("PnL by Trade Side")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True)

    savefig(fig, f"{ep['label']}_04_trade_outcomes.png")
    return pnls


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Feature → Action Sensitivity  (IDENTICAL structure;
# uses agent.actor(...) which MCKNNPolicy provides via __call__-free
# direct invocation — see note below)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_sensitivity(agent: MCKNNAgent, ep: dict):
    """
    diagnostic.py called agent.actor(states_tensor) directly because
    nn.Module supports batched forward passes. MCKNNPolicy has no
    batched forward (kNN queries are inherently per-point), so this
    version loops per-state through agent.actor.get_action(...) instead.
    This is slower than the SAC version for large episodes but produces
    the same Δ-probability-by-feature-percentile plot.
    """
    if len(ep["ticks"]) == 0:
        return

    fig, axes = make_fig(2, 3,
        f"[{ep['label']}] Section 5 — Feature → Action Sensitivity (p10→p90)")
    with plt.rc_context(STYLE):
        for feat_idx, feat_name in enumerate(FEATURES):
            ax = axes[feat_idx // 3][feat_idx % 3]
            feat_vals = ep["states_arr"][:, feat_idx]
            p10 = np.percentile(feat_vals, 10)
            p90 = np.percentile(feat_vals, 90)

            # Subsample for tractability — kNN query cost is O(bank_size)
            # per call, so a full per-tick double-pass over a multi-year
            # episode is wasteful; a few hundred probe states is enough
            # to estimate the mean Δ-probability per feature.
            n_probe = min(300, len(ep["states_arr"]))
            probe_idx = np.random.choice(len(ep["states_arr"]), n_probe, replace=False)

            deltas = []
            for idx in probe_idx:
                s_lo = ep["states_arr"][idx].copy()
                s_hi = ep["states_arr"][idx].copy()
                s_lo[feat_idx] = p10
                s_hi[feat_idx] = p90
                _, probs_lo, _, _ = agent.actor.get_action(s_lo)
                _, probs_hi, _, _ = agent.actor.get_action(s_hi)
                deltas.append(probs_hi - probs_lo)
            deltas = np.array(deltas)
            means = deltas.mean(axis=0)
            stds  = deltas.std(axis=0)

            ax.bar(range(4), means, color=ACTION_COLORS, alpha=0.85, width=0.6)
            ax.errorbar(range(4), means, yerr=stds, fmt="none",
                        color="white", capsize=4, lw=1.5)
            ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--")
            ax.set_xticks(range(4))
            ax.set_xticklabels(ACTION_NAMES, fontsize=8)
            ax.set_title(f"{feat_name}\n(p10={p10:.2f} → p90={p90:.2f})")
            ax.set_ylabel("Δ vote-share")
            ax.grid(True, axis="y")

    savefig(fig, f"{ep['label']}_05_feature_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — State-Conditional Probability Maps  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_state_probability_maps(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    rsi   = ep["raw_features"][:, 0]
    macd  = ep["raw_features"][:, 1]
    probs = ep["probs_all"]
    bins  = 10
    rsi_edges  = np.linspace(rsi.min(),  rsi.max(),  bins + 1)
    macd_edges = np.linspace(macd.min(), macd.max(), bins + 1)

    fig, axes = make_fig(2, 2,
        f"[{ep['label']}] Section 6 — Policy Surface: RSI × MACD Grid",
        figsize=(12, 10))
    with plt.rc_context(STYLE):
        for j, (name, _) in enumerate(zip(ACTION_NAMES, ACTION_COLORS)):
            ax = axes[j // 2][j % 2]
            grid = np.full((bins, bins), np.nan)
            for ri in range(bins):
                for mi in range(bins):
                    mask = (
                        (rsi  >= rsi_edges[ri])  & (rsi  < rsi_edges[ri + 1]) &
                        (macd >= macd_edges[mi]) & (macd < macd_edges[mi + 1])
                    )
                    if mask.sum() > 5:
                        grid[ri, mi] = probs[mask, j].mean()
            valid = grid[~np.isnan(grid)]
            vmax  = min(0.9, np.nanpercentile(valid, 95)) if len(valid) else 0.5
            im = ax.imshow(grid, origin="lower", aspect="auto",
                           cmap="plasma", vmin=0, vmax=vmax)
            ax.set_title(f"Mean vote-share({name}|s)  — RSI × MACD grid")
            ax.set_xlabel("MACD_Scaled bins")
            ax.set_ylabel("RSI_Scaled bins")
            rsi_c  = (rsi_edges[:-1]  + rsi_edges[1:])  / 2
            macd_c = (macd_edges[:-1] + macd_edges[1:]) / 2
            ax.set_xticks([0, bins//2, bins-1])
            ax.set_xticklabels([f"{macd_c[0]:.1f}", f"{macd_c[bins//2]:.1f}",
                                 f"{macd_c[-1]:.1f}"], fontsize=7)
            ax.set_yticks([0, bins//2, bins-1])
            ax.set_yticklabels([f"{rsi_c[0]:.1f}", f"{rsi_c[bins//2]:.1f}",
                                 f"{rsi_c[-1]:.1f}"], fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    savefig(fig, f"{ep['label']}_06_state_probability_maps.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Regime Analysis  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_analysis(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    mean_dev = ep["raw_features"][:, 5]
    atr      = ep["raw_features"][:, 4]

    regimes = {
        "Trending Up":   (mean_dev >  0.1) & (atr > 0),
        "Trending Down": (mean_dev < -0.1) & (atr > 0),
        "Ranging":       (mean_dev >= -0.1) & (mean_dev <= 0.1),
        "High Vol":      atr > np.percentile(atr, 75),
        "Low Vol":       atr < np.percentile(atr, 25),
    }
    regime_colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#f39c12", "#3498db"]

    fig, axes = make_fig(2, 3, f"[{ep['label']}] Section 7 — Regime Analysis")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        x = np.arange(4)
        width = 0.15
        for k, (rname, mask) in enumerate(regimes.items()):
            if mask.sum() < 10:
                continue
            counts = np.array([(ep["actions"][mask] == i).mean() for i in range(4)])
            ax.bar(x + k * width, counts, width=width,
                   color=regime_colors[k], alpha=0.8, label=rname)
        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(ACTION_NAMES, fontsize=8)
        ax.set_title("Action Distribution by Regime")
        ax.set_ylabel("Fraction of ticks")
        ax.legend(fontsize=6)
        ax.grid(True, axis="y")

        ax = axes[0][1]
        conv_data, conv_labels = [], []
        for rname, mask in regimes.items():
            if mask.sum() > 10:
                conv_data.append(ep["max_prob"][mask])
                conv_labels.append(f"{rname}\n(n={mask.sum():,})")
        if conv_data:
            bp = ax.boxplot(conv_data, patch_artist=True,
                            medianprops={"color": "white", "lw": 2})
            for patch, col in zip(bp["boxes"], regime_colors):
                patch.set_facecolor(col); patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(conv_labels) + 1))
            ax.set_xticklabels(conv_labels, fontsize=6)
        ax.set_title("Conviction by Regime")
        ax.set_ylabel("max vote-share(a|s)")
        ax.grid(True, axis="y")

        ax = axes[0][2]
        ent_data = [ep["entropies"][mask] for _, mask in regimes.items()
                    if mask.sum() > 10]
        if ent_data:
            bp2 = ax.boxplot(ent_data, patch_artist=True,
                             medianprops={"color": "white", "lw": 2})
            for patch, col in zip(bp2["boxes"], regime_colors):
                patch.set_facecolor(col); patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(conv_labels) + 1))
            ax.set_xticklabels(conv_labels, fontsize=6)
        ax.set_title("Vote-Share Entropy by Regime")
        ax.set_ylabel("H[vote-share] bits")
        ax.grid(True, axis="y")

        ax = axes[1][0]
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
                       color=[regime_colors[i] for i in range(len(wr_vals))],
                       alpha=0.8)
                ax.axhline(0.5, color="white", lw=1, linestyle="--")
                ax.set_xticks(range(len(wr_names)))
                ax.set_xticklabels(wr_names, fontsize=6)
                ax.set_title("Trade Win Rate by Entry Regime")
                ax.set_ylabel("Win rate")
                ax.set_ylim(0, 1)
                ax.grid(True, axis="y")

        ax = axes[1][1]
        for j, (name, col) in enumerate(zip(ACTION_NAMES[:2], ACTION_COLORS[:2])):
            mask = ep["actions"] == j
            ax.scatter(mean_dev[mask], atr[mask], s=4, color=col,
                       alpha=0.3, label=name)
        ax.set_title("MeanDev vs ATR at LONG/SHORT Ticks")
        ax.set_xlabel("MeanDev_Scaled (trend)")
        ax.set_ylabel("ATR_Scaled (volatility)")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[1][2]
        hold_mask    = ep["actions"] == 3
        nonhold_mask = ep["actions"] != 3
        if nonhold_mask.sum() > 0:
            ax.hist(ep["max_prob"][nonhold_mask], bins=40, color="#7f5af0",
                    alpha=0.7, label="Active", edgecolor="none", density=True)
        if hold_mask.sum() > 0:
            ax.hist(ep["max_prob"][hold_mask], bins=40, color="#95a5a6",
                    alpha=0.7, label="HOLD", edgecolor="none", density=True)
        ax.set_title("Conviction: HOLD vs Active")
        ax.set_xlabel("max vote-share(a|s)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True)

    savefig(fig, f"{ep['label']}_07_regime_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Timing Analysis  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_timing_analysis(ep: dict):
    if ep["timestamps"] is None or len(ep["ticks"]) == 0:
        print(f"  ⚠  [{ep['label']}] No timestamps — skipping Section 8.")
        return

    ts       = ep["timestamps"]
    hours    = np.array([t.hour      for t in ts])
    weekdays = np.array([t.dayofweek for t in ts])

    fig, axes = make_fig(2, 2, f"[{ep['label']}] Section 8 — Timing Analysis")
    with plt.rc_context(STYLE):
        ax = axes[0][0]
        means = [ep["max_prob"][hours == h].mean() if (hours == h).sum() > 0 else 0
                 for h in range(24)]
        ax.bar(range(24), means, color="#7f5af0", alpha=0.8)
        ax.axhline(ep["max_prob"].mean(), color="#ffd700", lw=1.5,
                   linestyle="--", label=f"overall={ep['max_prob'].mean():.2f}")
        ax.set_title("Mean Conviction by Hour (UTC)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean max vote-share")
        ax.set_xticks(range(24))
        ax.legend(fontsize=7)
        ax.grid(True, axis="y")

        ax = axes[0][1]
        if ep["trades"]:
            hour_of_entry: dict = {}
            for t in ep["trades"]:
                idx = t["entry_tick"] - WARMUP_IDX - 1
                if 0 <= idx < len(ts):
                    h = ts[idx].hour
                    hour_of_entry.setdefault(h, []).append(t["win"])
            if hour_of_entry:
                hrs_list = sorted(hour_of_entry.keys())
                wr_list  = [np.mean(hour_of_entry[h]) for h in hrs_list]
                cnt_list = [len(hour_of_entry[h]) for h in hrs_list]
                ax.bar(hrs_list, wr_list,
                       color=["#2ecc71" if w >= 0.5 else "#e74c3c"
                              for w in wr_list], alpha=0.8)
                ax.axhline(0.5, color="white", lw=1, linestyle="--")
                for h, wr, cnt in zip(hrs_list, wr_list, cnt_list):
                    ax.text(h, wr + 0.02, str(cnt), ha="center",
                            fontsize=6, color="#aaaacc")
                ax.set_title("Trade Win Rate by Hour of Day")
                ax.set_xlabel("Hour (UTC)")
                ax.set_ylabel("Win rate")
                ax.set_ylim(0, 1.1)
                ax.grid(True, axis="y")

        ax = axes[1][0]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        means_d = [ep["max_prob"][weekdays == d].mean()
                   if (weekdays == d).sum() > 0 else 0 for d in range(7)]
        ax.bar(range(7), means_d, color="#2cb67d", alpha=0.8)
        ax.axhline(ep["max_prob"].mean(), color="#ffd700", lw=1.5, linestyle="--")
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names)
        ax.set_title("Mean Conviction by Day of Week")
        ax.set_ylabel("Mean max vote-share")
        ax.grid(True, axis="y")

        ax = axes[1][1]
        long_h  = [(ep["actions"][hours == h] == 0).sum() for h in range(24)]
        short_h = [(ep["actions"][hours == h] == 1).sum() for h in range(24)]
        x = np.arange(24)
        ax.bar(x - 0.2, long_h,  0.4, color="#2ecc71", alpha=0.8, label="LONG")
        ax.bar(x + 0.2, short_h, 0.4, color="#e74c3c", alpha=0.8, label="SHORT")
        ax.set_title("LONG vs SHORT by Hour")
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel("Count")
        ax.set_xticks(range(24))
        ax.legend(fontsize=8)
        ax.grid(True, axis="y")

    savefig(fig, f"{ep['label']}_08_timing_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Neighbor Agreement (Vote Margin) vs Outcome
# (replaces diagnostic.py's "Critic Disagreement vs Outcome" — see
# module docstring)
# ─────────────────────────────────────────────────────────────────────────────

def plot_vote_margin_vs_outcome(ep: dict):
    if not ep["trades"] or len(ep["ticks"]) == 0:
        print(f"  ⚠  [{ep['label']}] No trades — skipping Section 9.")
        return

    fig, axes = make_fig(2, 2,
        f"[{ep['label']}] Section 9 — Neighbor Agreement (Vote Margin) vs Outcome")
    with plt.rc_context(STYLE):
        entry_margin, pnls = [], []
        for t in ep["trades"]:
            idx = t["entry_tick"] - WARMUP_IDX - 1
            if 0 <= idx < len(ep["vote_margin"]):
                entry_margin.append(ep["vote_margin"][idx])
                pnls.append(t["pnl"])

        if entry_margin:
            em   = np.array(entry_margin)
            pl   = np.array(pnls)
            wins = pl > 0

            ax = axes[0][0]
            ax.scatter(em[wins],  pl[wins],  s=10, color="#2ecc71", alpha=0.5, label="Win")
            ax.scatter(em[~wins], pl[~wins], s=10, color="#e74c3c", alpha=0.5, label="Loss")
            ax.axhline(0, color="white", lw=0.8, linestyle="--")
            corr = np.corrcoef(em, pl)[0, 1] if len(em) > 1 else 0.0
            ax.set_title(f"Vote Margin at Entry vs PnL  (r={corr:.3f})")
            ax.set_xlabel("vote_margin at entry")
            ax.set_ylabel("PnL")
            ax.legend(fontsize=8)
            ax.grid(True)

            ax = axes[0][1]
            q_wr, q_labels = [], []
            for q in range(5):
                lo = np.percentile(em, q * 20)
                hi = np.percentile(em, (q + 1) * 20)
                mask = (em >= lo) & (em <= hi)
                if mask.sum() > 0:
                    q_wr.append(wins[mask].mean())
                    q_labels.append(f"Q{q+1}\n({lo:.3f}-{hi:.3f})")
            ax.bar(range(len(q_wr)), q_wr,
                   color=["#2ecc71" if w >= 0.5 else "#e74c3c" for w in q_wr],
                   alpha=0.8)
            ax.axhline(0.5, color="white", lw=1, linestyle="--")
            ax.set_xticks(range(len(q_labels)))
            ax.set_xticklabels(q_labels, fontsize=7)
            ax.set_title("Win Rate by Vote-Margin Quintile")
            ax.set_ylabel("Win rate")
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y")

        ax = axes[1][0]
        window = max(1, min(200, len(ep["vote_margin"]) // 10))
        rolling = pd.Series(ep["vote_margin"]).rolling(window).mean().values
        ax.plot(ep["ticks"], rolling, color="#f39c12", lw=1.0)
        ax.set_title(f"Vote Margin Over Time (rolling {window})")
        ax.set_xlabel("Tick")
        ax.set_ylabel("vote_margin")
        ax.grid(True)

        ax = axes[1][1]
        ax.scatter(ep["vote_margin"], ep["max_prob"], s=2, alpha=0.15,
                   color="#7f5af0")
        corr = np.corrcoef(ep["vote_margin"], ep["max_prob"])[0, 1] \
            if len(ep["vote_margin"]) > 1 else 0.0
        ax.set_title(f"Vote Margin vs Conviction  (r={corr:.3f})")
        ax.set_xlabel("vote_margin")
        ax.set_ylabel("max vote-share(a|s)")
        ax.grid(True)

    savefig(fig, f"{ep['label']}_09_vote_margin_vs_outcome.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Action-State Consistency  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_consistency(ep: dict):
    if len(ep["ticks"]) == 0:
        return
    fig, axes = make_fig(2, 3,
        f"[{ep['label']}] Section 10 — Action vs Market Context Consistency")
    with plt.rc_context(STYLE):
        action_masks  = {"LONG": ep["actions"] == 0,
                         "SHORT": ep["actions"] == 1,
                         "HOLD": ep["actions"] == 3}
        action_colors = {"LONG": ACTION_COLORS[0],
                         "SHORT": ACTION_COLORS[1],
                         "HOLD": ACTION_COLORS[3]}

        for fi, fname in enumerate(FEATURES):
            ax = axes[fi // 3][fi % 3]
            plot_data, plot_labels, plot_pos, plot_cols = [], [], [], []
            pos = 0
            for aname in ["LONG", "SHORT", "HOLD"]:
                mask = action_masks[aname]
                if mask.sum() > 0:
                    plot_data.append(ep["raw_features"][mask, fi])
                    plot_labels.append(f"{aname}\n(n={mask.sum():,})")
                    plot_pos.append(pos)
                    plot_cols.append(action_colors[aname])
                    pos += 1
            if plot_data:
                parts = ax.violinplot(plot_data, positions=plot_pos,
                                      showmedians=True, showextrema=False)
                for pc, col in zip(parts["bodies"], plot_cols):
                    pc.set_facecolor(col); pc.set_alpha(0.6)
                parts["cmedians"].set_color("white")
                ax.axhline(0, color="#aaaacc", lw=0.8, linestyle="--", alpha=0.7)
                ax.set_xticks(plot_pos)
                ax.set_xticklabels(plot_labels, fontsize=7)
                ax.set_title(fname)
                ax.set_ylabel("Scaled value")
                ax.grid(True, axis="y")
            else:
                ax.set_title(f"{fname}\n(No data)")
                ax.text(0.5, 0.5, "No action data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=10)

    savefig(fig, f"{ep['label']}_10_action_consistency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Text summary  (same per-label-file structure as diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(ep: dict, agent: MCKNNAgent, pnls):
    trades = ep["trades"]
    n      = len(ep["ticks"])
    label  = ep["label"]

    lines = []
    lines.append("=" * 62)
    lines.append(f"  MC-kNN DIAGNOSTIC SUMMARY — {label.upper()}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Memory bank size: {len(agent.memory):,}")
    lines.append(f"  Episodes committed: {agent.n_episodes_committed}")
    lines.append(f"  k (neighbors): {agent.memory.k}")
    lines.append("=" * 62)

    if n == 0:
        lines.append("\n  ⚠  No ticks collected — episode was too short.")
        text = "\n".join(lines)
        print(text)
        return

    lines.append("\n── POLICY HEALTH ──────────────────────────────────────────")
    lines.append(f"  Mean conviction (max vote-share): {ep['max_prob'].mean():.3f}")
    lines.append(f"  Conviction > 0.50:           {(ep['max_prob'] > 0.50).mean():.1%}")
    lines.append(f"  Conviction > 0.70:           {(ep['max_prob'] > 0.70).mean():.1%}")
    lines.append(f"  Mean entropy:                {ep['entropies'].mean():.3f} bits")
    best_vote_action = ep["vote_raw_all"].argmax(axis=1)
    lines.append(f"  Policy match to best-vote:   "
                 f"{(best_vote_action == ep['actions']).mean():.1%}")

    lines.append("\n── ACTION DISTRIBUTION ────────────────────────────────────")
    for j, aname in enumerate(ACTION_NAMES):
        count = (ep["actions"] == j).sum()
        lines.append(f"  {aname:<6}: {count:>6,}  ({count/n:.1%})")

    lines.append("\n── NEIGHBOR / VOTE HEALTH ─────────────────────────────────")
    lines.append(f"  Mean vote margin:            {ep['vote_margin'].mean():.5f}")
    lines.append(f"  Mean neighbor return:        {ep['neighbor_return_mean'].mean():.5f}")
    lines.append(f"  Mean neighbor return std:    {ep['neighbor_return_std'].mean():.5f}")

    if trades:
        pnl_arr = np.array([t["pnl"] for t in trades])
        wins    = pnl_arr > 0
        dur_arr = np.array([t["duration"] for t in trades])
        longs   = [t for t in trades if t["side"] == "LONG"]
        shorts  = [t for t in trades if t["side"] == "SHORT"]
        cum     = np.cumsum(pnl_arr)
        roll_max = np.maximum.accumulate(cum)
        drawdown = cum - roll_max

        lines.append("\n── TRADE OUTCOMES ─────────────────────────────────────────")
        lines.append(f"  Total trades:                {len(trades):,}")
        lines.append(f"  Win rate:                    {wins.mean():.1%}")
        lines.append(f"  Mean PnL per trade:          {pnl_arr.mean():.4%}")
        lines.append(f"  Median PnL per trade:        {np.median(pnl_arr):.4%}")
        lines.append(f"  Std PnL per trade:           {pnl_arr.std():.4%}")
        lines.append(f"  Total cumulative PnL:        {pnl_arr.sum():.4%}")
        lines.append(f"  Best trade:                  {pnl_arr.max():.4%}")
        lines.append(f"  Worst trade:                 {pnl_arr.min():.4%}")
        lines.append(f"  Max drawdown:                {drawdown.min():.4%}")
        if pnl_arr.std() > 0:
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(len(trades))
            lines.append(f"  Sharpe (simplified):         {sharpe:.3f}")
        lines.append(f"  Median hold duration:        {np.median(dur_arr):.0f} ticks "
                     f"({np.median(dur_arr) * 4:.0f} hrs)")
        if longs:
            lp = np.array([t["pnl"] for t in longs])
            lines.append(f"  LONG  win rate:              {(lp>0).mean():.1%} "
                         f"(n={len(longs)}, mean={lp.mean():.4%})")
        if shorts:
            sp = np.array([t["pnl"] for t in shorts])
            lines.append(f"  SHORT win rate:              {(sp>0).mean():.1%} "
                         f"(n={len(shorts)}, mean={sp.mean():.4%})")

        conv    = np.array([t["entry_conviction"] for t in trades])
        hi_conv = conv >= np.percentile(conv, 66)
        lo_conv = conv <  np.percentile(conv, 33)
        lines.append(f"\n── CONVICTION EDGE ────────────────────────────────────────")
        lines.append(f"  Top-33% conviction WR:       {wins[hi_conv].mean():.1%}  "
                     f"(mean PnL {pnl_arr[hi_conv].mean():.4%})")
        if lo_conv.sum() > 0:
            lines.append(f"  Bot-33% conviction WR:       {wins[lo_conv].mean():.1%}  "
                         f"(mean PnL {pnl_arr[lo_conv].mean():.4%})")
            edge = wins[hi_conv].mean() - wins[lo_conv].mean()
            lines.append(f"  Conviction edge (WR delta):  {edge:+.1%}")
            lines.append(f"  → {'Conviction is predictive ✓' if edge > 0.03 else 'Conviction not yet predictive'}")

        if pnl_arr.std() > 0:
            trades_per_day = len(trades) / max(1, len(ep["ticks"]) / 6)  # 6 ticks/day on 4h
            ann_sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(252 * trades_per_day)
            lines.append(f"  Annualized Sharpe (realistic): {ann_sharpe:.2f}")
            if ann_sharpe > 5:
                lines.append("  ⚠ Sharpe >5 — check for regime bias or data leakage.")

        if wins.mean() > 0.80:
            lines.append("\n⚠ WARNING: Win rate >80% strongly suggests regime overfitting.")

    lines.append("\n" + "=" * 62)
    text = "\n".join(lines)

    summary_path = os.path.join(OUT_DIR, f"diagnostic_summary_{label}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n  ✓  {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Full plot suite for a single episode
# ─────────────────────────────────────────────────────────────────────────────

def run_full_suite(agent: MCKNNAgent, ep: dict):
    """Run all 10 sections for one episode — same call order as diagnostic.py."""
    if len(ep["ticks"]) == 0:
        print(f"  ⚠  [{ep['label']}] Empty episode — skipping all plots.")
        return None

    plot_confidence_entropy(ep)
    plot_neighbor_return_health(ep)
    plot_action_distribution(ep)
    pnls = plot_trade_outcomes(ep)
    plot_feature_sensitivity(agent, ep)
    plot_state_probability_maps(ep)
    plot_regime_analysis(ep)
    plot_timing_analysis(ep)
    plot_vote_margin_vs_outcome(ep)
    plot_action_consistency(ep)
    return pnls


# ─────────────────────────────────────────────────────────────────────────────
# Random baseline  (IDENTICAL to diagnostic.py)
# ─────────────────────────────────────────────────────────────────────────────

def random_policy_baseline(df: pd.DataFrame, n_trials: int = 1000,
                            n_trades: int = 500) -> None:
    """Identical to diagnostic.py's random_policy_baseline — copied verbatim."""
    prices     = df["Close"].values
    COMMISSION = 0.00015

    pnls = []
    for _ in range(n_trials):
        pnl = 0.0
        for _ in range(n_trades):
            entry_idx = np.random.randint(0, max(1, len(prices) - 33))
            hold      = np.random.randint(1, 33)
            side      = np.random.choice([-1, 1])
            entry     = prices[entry_idx] * (1 + side * COMMISSION)
            exit_     = prices[min(entry_idx + hold, len(prices) - 1)] \
                        * (1 - side * COMMISSION)
            pnl += side * (exit_ - entry) / entry
        pnls.append(pnl)

    pnls = np.array(pnls)
    print(f"Random policy ({n_trades} trades × {n_trials} trials): "
          f"mean={pnls.mean():.4%}  std={pnls.std():.4%}  "
          f"win_rate={(pnls > 0).mean():.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  (same CLI surface as diagnostic.py, --checkpoint defaults to .npz)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MC-kNN Diagnostic Suite")
    parser.add_argument("--split",      default="val",
                        choices=["train", "val", "both"],
                        help="Which regime-split set(s) to run full diagnostics on")
    parser.add_argument("--checkpoint", default=BEST_PATH,
                        help="Path to .npz memory-bank checkpoint")
    parser.add_argument("--no-regime",  action="store_true",
                        help="Skip bear/bull regime episodes (faster)")
    args = parser.parse_args()

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  MC-kNN DIAGNOSTIC SUITE")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split       : {args.split}")
    print(f"  Output dir : {OUT_DIR}")
    print(f"{sep}\n")

    # ── Load agent ────────────────────────────────────────────────────────────
    agent = MCKNNAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        k=K_NEIGHBORS, max_size=200_000, gamma=GAMMA,
    )
    agent.load(args.checkpoint)

    # ── Load data and build splits ────────────────────────────────────────────
    print("Loading data...")
    df = update_master_data()
    df = df[["Open_time", "Close"] + FEATURES].dropna().reset_index(drop=True)
    splits = make_splits(df)

    # ── Random baseline on each val year separately ───────────────────────────
    print("Random policy baseline (val 2022 — bear market):")
    random_policy_baseline(splits["val_2022"])
    print("Random policy baseline (val 2025 — recent mixed):")
    random_policy_baseline(splits["val_2025"])
    print()

    # ── Bear / bull regime episodes ───────────────────────────────────────────
    dash62 = "-" * 62
    eq62   = "=" * 62
    if not args.no_regime:
        for regime_label in ["bear_2022", "bull_trend"]:
            print("\n" + dash62)
            print(f"  Regime episode: {regime_label.upper()}")
            print(dash62)
            ep = collect_episode(agent, splits[regime_label], regime_label)
            print(f"  Ticks: {len(ep['ticks']):,}  |  Trades: {len(ep['trades']):,}")
            pnls = plot_trade_outcomes(ep)
            write_summary(ep, agent, pnls)

    # ── Val: each year as a separate episode — no year-boundary phantom trades ─
    if args.split in ("val", "both"):
        ep_2022 = None
        ep_2025 = None
        for val_label, val_key in [("val_2022", "val_2022"),
                                    ("val_2025", "val_2025")]:
            print("\n" + dash62)
            print(f"  Full diagnostic: {val_label.upper()}")
            print(dash62)
            ep = collect_episode(agent, splits[val_key], val_label)
            print(f"  Ticks: {len(ep['ticks']):,}  |  Trades: {len(ep['trades']):,}")
            print("\n  Generating plots...")
            pnls = run_full_suite(agent, ep)
            print()
            write_summary(ep, agent, pnls)
            if val_key == "val_2022":
                ep_2022 = ep
            else:
                ep_2025 = ep

        # ── Combined val summary (matches training loop v_pnl) ────────────────
        if ep_2022 is not None and ep_2025 is not None:
            pnl_2022 = sum(t["pnl"] for t in ep_2022["trades"])
            pnl_2025 = sum(t["pnl"] for t in ep_2025["trades"])
            n_2022   = len(ep_2022["trades"])
            n_2025   = len(ep_2025["trades"])
            print("\n  ── Val combined (matches training loop) " + "-" * 20)
            print(f"  Val 2022 P/L : {pnl_2022:+.4%}  |  trades={n_2022}")
            print(f"  Val 2025 P/L : {pnl_2025:+.4%}  |  trades={n_2025}")
            print(f"  Val total    : {pnl_2022 + pnl_2025:+.4%}"
                  f"  |  trades={n_2022 + n_2025}")

    # ── Train full diagnostic ─────────────────────────────────────────────────
    if args.split in ("train", "both"):
        print("\n" + dash62)
        print("  Full diagnostic: TRAIN")
        print(dash62)
        ep = collect_episode(agent, splits["train"], "train")
        print(f"  Ticks: {len(ep['ticks']):,}  |  Trades: {len(ep['trades']):,}")
        print("\n  Generating plots...")
        pnls = run_full_suite(agent, ep)
        print()
        write_summary(ep, agent, pnls)

    print("\n" + eq62)
    print(f"  Diagnostic complete.  All outputs in: {OUT_DIR}/")
    print(eq62 + "\n")


if __name__ == "__main__":
    main()