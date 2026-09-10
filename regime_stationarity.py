"""
regime_stationarity.py
─────────────────────────
Checks whether the triple-barrier LABEL-LEVEL edge itself — independent
of any model (GBT, logistic, or otherwise) — is stationary over time,
or whether it has decayed/reversed in the most recent regime.

Why this exists
──────────────────
The confirmation run showed: (a) a linear model finds an edge
comparable to (or better than) the tuned GBT (baseline-edge gate
failed), and (b) the recent-regime CPCV check (test_groups == the last
N_TEST_GROUPS groups) is mostly unmeasurable — 4 of 6 confirmation
seeds produced 0 test trades there, and the two that resolved trades
disagreed in sign (seed 5: +0.08%/trade over 20 trades; seed 7:
-1.36%/trade over 6 trades). That combination — a model-agnostic
result plus a starved/inconsistent recent window — raises the
question this script answers directly: does the underlying, model-free
"oracle" edge (what you'd get by perfectly trading every favorable
triple-barrier label) look stationary across calendar time, or is the
recent window structurally different (less signal, worse realised
edge, or both)?

This is diagnostic only — it never touches a classifier, never re-fits
anything, and never feeds into any gate. It answers "is there
something to find at all, and has it held up over time", which is a
prerequisite question the modeling pipeline can't answer on its own
(a model can only be as stationary as the labels it's fit to).

What it computes
──────────────────
For every tick, the "oracle return" is:
  - long_return[i]  if best_action[i] == 0 (LONG was the favorable side)
  - short_return[i] if best_action[i] == 1 (SHORT was the favorable side)
  - NaN             if best_action[i] == 3 (HOLD — no favorable side)
i.e. the realised net return of the SINGLE best trade the labeling
scheme identified at that tick, ignoring the single-open-position
constraint main_gbt.simulate_pnl() enforces (this is intentionally an
upper-bound/ceiling view — "if you could take every signal", not "what
one continuously-in-market strategy would realise" — because the
question here is about the SIGNAL's stationarity, not about capital
allocation).

These are aggregated into calendar windows (default: monthly) and
plotted alongside the CPCV group boundaries (mapped from row-index
space into calendar time), with the recent-regime window
(RECENT_REGIME_GROUPS) shaded, so you can see directly whether that
window sits in a visibly different part of the time series or is just
an ordinary-looking segment that happened to have few trades.

Usage
──────
    python regime_stationarity.py --frozen-data
    python regime_stationarity.py --frozen-data --window Q          # quarterly bins
    python regime_stationarity.py --frozen-data --tp-mult 1.0 --sl-mult 0.6 --max-holding 12
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main_gbt as gbt

STYLE = {
    "axes.facecolor": "#1a1a2e", "figure.facecolor": "#0f0f1a",
    "axes.edgecolor": "#444466", "axes.labelcolor": "#ccccee",
    "xtick.color": "#aaaacc", "ytick.color": "#aaaacc",
    "text.color": "#ddddff", "grid.color": "#2a2a4a",
    "grid.linestyle": "--", "grid.alpha": 0.5,
}

OUT_PATH = os.path.join(gbt.OUT_DIR, "label_edge_stationarity.png")
OUT_SUMMARY_PATH = os.path.join(gbt.OUT_DIR, "label_edge_stationarity_summary.txt")


def _resolve_resample_rule(requested: str = None) -> str:
    """
    Pandas renamed the month-end resample alias from 'M' to 'ME' in
    2.2 (removing 'M' entirely in later versions) — pick whichever this
    installed pandas actually accepts rather than hardcoding one and
    breaking on the other. If the user passed an explicit --window,
    it's used as-is (not remapped) since they may be targeting an alias
    this heuristic doesn't need to touch (e.g. 'W', 'Q', 'QE').
    """
    if requested is not None:
        return requested
    for candidate in ("ME", "M"):
        try:
            pd.tseries.frequencies.to_offset(candidate)
            return candidate
        except ValueError:
            continue
    return "ME"  # fall through; pandas will raise its own clear error


def _compute_group_boundaries(n: int, n_groups: int) -> list:
    """Same formula as cpcv.py's private _make_groups() — a local,
    dependency-free copy so this script only relies on main_gbt.py's
    public surface (see label_sweep.py's identical helper)."""
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(edges[g], edges[g + 1]) for g in range(n_groups)]


def compute_oracle_returns(labels: dict, n: int) -> np.ndarray:
    """
    Per-tick oracle return (see module docstring). NaN where invalid or
    HOLD.
    """
    best_action = labels["best_action"]
    long_return = labels["long_return"]
    short_return = labels["short_return"]
    valid = labels["valid_mask"]

    oracle = np.full(n, np.nan, dtype=np.float64)
    oracle[best_action == 0] = long_return[best_action == 0]
    oracle[best_action == 1] = short_return[best_action == 1]
    oracle[~valid] = np.nan
    return oracle


def main():
    parser = argparse.ArgumentParser(
        description="Check whether the triple-barrier label-level edge "
                     "is stationary over calendar time.")
    parser.add_argument("--frozen-data", action="store_true",
                        help="Use the existing master CSV(s) as-is instead "
                             "of refetching/appending new raw data.")
    parser.add_argument("--tp-mult", type=float, default=None,
                        help=f"Default: main_gbt.TP_MULT ({gbt.TP_MULT})")
    parser.add_argument("--sl-mult", type=float, default=None,
                        help=f"Default: main_gbt.SL_MULT ({gbt.SL_MULT})")
    parser.add_argument("--max-holding", type=int, default=None,
                        help=f"Default: main_gbt.MAX_HOLDING ({gbt.MAX_HOLDING})")
    parser.add_argument("--window", type=str, default=None,
                        help="Pandas resample rule for calendar bins "
                             "(default: month-end, auto-picking whichever "
                             "alias this pandas version accepts — 'ME' on "
                             "pandas>=2.2, 'M' on older pandas; try 'Q'/'QE' "
                             "for quarterly or 'W' for weekly on shorter "
                             "histories).")
    args = parser.parse_args()

    tp = gbt.TP_MULT if args.tp_mult is None else args.tp_mult
    sl = gbt.SL_MULT if args.sl_mult is None else args.sl_mult
    mh = gbt.MAX_HOLDING if args.max_holding is None else args.max_holding

    print("Loading prices/ATR (no 15m/1h fetch, no state aggregation needed)...")
    prices_aligned, atr_aligned, aligned_df = gbt.load_prices_for_labeling(
        frozen=args.frozen_data,
    )
    n = len(prices_aligned)
    print(f"  {n:,} rows")

    print(f"Building triple-barrier labels (tp_mult={tp}, sl_mult={sl}, "
          f"max_holding={mh})...")
    labels = gbt.labels_from_prices(prices_aligned, atr_aligned,
                                    tp_mult=tp, sl_mult=sl, max_holding=mh)
    valid = labels["valid_mask"]
    best_action = labels["best_action"]

    oracle = compute_oracle_returns(labels, n)
    is_signal = (best_action != 3) & valid

    ts = pd.to_datetime(aligned_df["Open_time"], errors="coerce")

    # ── Recent-regime window, mapped from row-index space to calendar time ──
    groups = _compute_group_boundaries(n, gbt.N_GROUPS)
    recent_lo = groups[gbt.N_GROUPS - gbt.N_TEST_GROUPS][0]
    recent_hi = groups[-1][1] - 1
    recent_start_ts = ts.iloc[recent_lo]
    recent_end_ts = ts.iloc[recent_hi]

    recent_mask = np.zeros(n, dtype=bool)
    recent_mask[recent_lo:recent_hi + 1] = True

    # ── Rolling calendar-window aggregation ──────────────────────────────
    df_r = pd.DataFrame({"ts": ts, "oracle": oracle, "is_signal": is_signal})
    df_r = df_r.set_index("ts")
    resample_rule = _resolve_resample_rule(args.window)
    binned = df_r.resample(resample_rule).agg(
        mean_oracle=("oracle", "mean"),
        n_signals=("is_signal", "sum"),
        n_ticks=("oracle", "size"),
    )
    binned["signal_density"] = binned["n_signals"] / binned["n_ticks"].replace(0, np.nan)

    # ── Plot ──────────────────────────────────────────────────────────────
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        fig.suptitle(
            f"Label-level edge stationarity  (tp_mult={tp}, sl_mult={sl}, "
            f"max_holding={mh})", color="#ffffff", fontsize=13, fontweight="bold",
        )

        ax = axes[0]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c"
                 for v in binned["mean_oracle"].fillna(0)]
        bar_width = (binned.index[1] - binned.index[0]).days * 0.8 if len(binned) > 1 else 20
        ax.bar(binned.index, binned["mean_oracle"] * 100, width=bar_width,
               color=colors, alpha=0.85)
        ax.axhline(0, color="white", lw=0.8, linestyle="--")
        ax.axvspan(recent_start_ts, recent_end_ts, color="#ffd700", alpha=0.15,
                  label=f"recent-regime groups {gbt.RECENT_REGIME_GROUPS}")
        ax.set_ylabel("Mean oracle return per\nfavorable-labeled entry (%)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(binned.index, binned["signal_density"] * 100, color="#7f5af0",
                 marker="o", ms=3, lw=1.2)
        ax2.axvspan(recent_start_ts, recent_end_ts, color="#ffd700", alpha=0.15)
        ax2.set_ylabel("Signal density\n(% ticks favorable-labeled)")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(gbt.OUT_DIR, exist_ok=True)
        fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight",
                   facecolor=STYLE["figure.facecolor"])
        plt.close(fig)
    print(f"\n  ✓  {OUT_PATH}")

    # ── Recent vs. rest — plain-language verdict ────────────────────────
    recent_oracle = oracle[recent_mask]
    recent_oracle = recent_oracle[~np.isnan(recent_oracle)]
    other_oracle = oracle[~recent_mask]
    other_oracle = other_oracle[~np.isnan(other_oracle)]

    recent_density = float(is_signal[recent_mask].mean()) if recent_mask.sum() else float("nan")
    other_density = float(is_signal[~recent_mask].mean()) if (~recent_mask).sum() else float("nan")

    lines = ["=" * 62, "  LABEL-LEVEL EDGE STATIONARITY SUMMARY",
             f"  Config: tp_mult={tp}  sl_mult={sl}  max_holding={mh}",
             f"  Recent-regime window: {recent_start_ts.date()} → "
             f"{recent_end_ts.date()}  (row range [{recent_lo}:{recent_hi+1}], "
             f"groups {gbt.RECENT_REGIME_GROUPS})",
             "=" * 62]

    lines.append(f"\n  Recent-regime oracle trades : {len(recent_oracle):,}")
    if len(recent_oracle):
        lines.append(f"  Recent-regime mean return   : {recent_oracle.mean():+.4%}")
        lines.append(f"  Recent-regime win rate      : {(recent_oracle > 0).mean():.1%}")
    lines.append(f"  Recent-regime signal density: {recent_density:.2%} of ticks")

    lines.append(f"\n  Rest-of-history oracle trades : {len(other_oracle):,}")
    if len(other_oracle):
        lines.append(f"  Rest-of-history mean return   : {other_oracle.mean():+.4%}")
        lines.append(f"  Rest-of-history win rate      : {(other_oracle > 0).mean():.1%}")
    lines.append(f"  Rest-of-history signal density: {other_density:.2%} of ticks")

    lines.append("\n  ── Interpretation ─────────────────────────────────────")
    if len(recent_oracle) < 30:
        lines.append(
            f"  ⚠ Only {len(recent_oracle)} oracle trades in the recent-regime "
            f"window even at the LABEL level (no model, no single-position "
            f"constraint) — this config's barriers are too slow/wide to "
            f"resolve enough trades in this window to say anything reliable "
            f"about it. This matches the CPCV gate's own 'SKIP (too few "
            f"trades)' result. Try tighter tp_mult/sl_mult or a shorter "
            f"max_holding (see label_sweep.py) before concluding anything "
            f"about whether this regime has an edge."
        )
    else:
        density_ratio = (recent_density / other_density) if other_density else float("nan")
        mean_gap = (recent_oracle.mean() - other_oracle.mean()) if len(other_oracle) else float("nan")
        lines.append(f"  Signal density ratio (recent / rest): {density_ratio:.2f}x")
        lines.append(f"  Mean-return gap (recent - rest)      : {mean_gap:+.4%}")
        if recent_oracle.mean() <= 0 and (len(other_oracle) == 0 or other_oracle.mean() > 0):
            lines.append(
                "  → The recent-regime window's label-level edge is "
                "non-positive while the rest of history is positive. This "
                "is consistent with a genuine regime shift, not just a "
                "modeling failure — no classifier can be expected to "
                "extract a positive edge from labels that are themselves "
                "non-positive in this window."
            )
        elif density_ratio < 0.5:
            lines.append(
                "  → Signal density has dropped sharply in the recent "
                "window (fewer than half as many favorable setups per "
                "tick as the rest of history). Even with a similar "
                "per-trade edge, this window will generically produce too "
                "few trades for any confirmation check to trust — treat "
                "the recent-regime CPCV result as inconclusive rather than "
                "a genuine failure, and prioritize widening the holdout / "
                "loosening barriers over re-tuning the classifier."
            )
        else:
            lines.append(
                "  → Recent-regime signal density and mean return are in "
                "the same ballpark as the rest of history — the label-"
                "level edge does not show an obvious stationarity break "
                "here. If the CPCV/model-level result for this window is "
                "still weak or negative, that points more toward a "
                "modeling/threshold issue than a labels/regime issue."
            )

    text = "\n".join(lines)
    with open(OUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"\n  ✓  {OUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

# python regime_stationarity.py --frozen-data
# python regime_stationarity.py --frozen-data --window Q
# python regime_stationarity.py --frozen-data --tp-mult 1.0 --sl-mult 0.6 --max-holding 12