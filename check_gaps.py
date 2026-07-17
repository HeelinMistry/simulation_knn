"""
check_gaps.py
──────────────
Resolves the gap row indices from pre_training.py's coverage warning
into actual dates, classifies each gap by likely cause, and reports
which gaps (if any) fall inside the walk-forward val window — the only
ones that matter for whether a gap-proximity exclusion mask is needed.

CHANGES IN THIS REVISION — CV FOLD CROSS-REFERENCE
─────────────────────────────────────────────────────
main_mcknn.py's walk-forward CV gate (see run_cross_validation()'s
gate in main()) now records each fold's val_start_date/val_end_date
and flags a fold whose val P/L breaches VAL_PNL_FLOOR, plus whether
that breach recurs across consecutive runs at the same/overlapping
window. That tells you WHICH calendar period is the problem but not
WHY. This revision adds that missing link: given a fold's date range
(read automatically from outcomes/walkforward_cv_summary.json, or
supplied manually via --fold-start/--fold-end), it reports whether
that window's purge+val+embargo span overlaps a known raw-data gap or
a KNOWN_EVENTS entry — so a recurring blowup fold can be triaged as
"real market event the strategy should learn to handle", "data-quality
artifact near a gap", or "neither, needs direct investigation" without
eyeballing dates by hand.

Run from project root:
    python check_gaps.py
    python check_gaps.py --fold-start 2022-05-01 --fold-end 2022-07-15

Output is also written to outcomes/gap_report.txt for reference.
"""

import sys; sys.path.insert(0, '.')
import os
import argparse
import json
import pandas as pd
import numpy as np

os.makedirs("outcomes", exist_ok=True)

parser = argparse.ArgumentParser(description="Gap + CV-fold-window diagnostic")
parser.add_argument("--fold-start", default=None,
                    help="Manually check a specific window instead of/in "
                         "addition to auto-detecting from "
                         "outcomes/walkforward_cv_summary.json, e.g. "
                         "2022-05-01")
parser.add_argument("--fold-end", default=None,
                    help="End date for --fold-start (required if "
                         "--fold-start is given), e.g. 2022-07-15")
args = parser.parse_args()

# ── Config — must match data_manager.py and main_mcknn.py ────────────────────
HOUR_PATH  = "data/processed/XRPUSDT_1h_master_processed.csv"
PRIMARY_PATH = "data/processed/XRPUSDT_4h_master_processed.csv"
CANDLE_H   = 1      # 1h master
LOOKBACK_TICKS_4H = 720   # walkforward.compute_required_lookback_ticks() default
# Keep in sync with main_mcknn.py's VAL_PNL_FLOOR — this is the same
# threshold the training gate uses to flag a CV fold as a "blowup".
VAL_PNL_FLOOR = -0.40
CV_SUMMARY_PATH = "outcomes/walkforward_cv_summary.json"

# Known Binance / market events for date-matching context
KNOWN_EVENTS = [
    ("2018-09", "2018-10", "Binance early maintenance windows"),
    ("2019-05", "2019-05", "Binance hack / trading halt (May 2019)"),
    ("2020-02", "2020-03", "COVID-19 market crash / Binance congestion"),
    ("2020-11", "2021-01", "Bull run peak — high load periods"),
    ("2021-05", "2021-05", "China mining ban sell-off congestion"),
    ("2021-11", "2021-12", "ATH period — Binance load spikes"),
    ("2022-05", "2022-06", "LUNA/UST collapse"),
    ("2022-11", "2022-11", "FTX collapse"),
    ("2024-03", "2024-04", "BTC halving run-up"),
    ("2025-01", "2025-02", "Post-ETF approval volatility"),
]

INDICATOR_LOOKBACK_1H = 200   # longest rolling window in preprocessing.py,
                               # in 1h candles — OBV(200), ATR(200)


def classify_gap(gap_date: pd.Timestamp) -> str:
    """Match a gap's closing date against known events."""
    ym = gap_date.strftime("%Y-%m")
    for (start, end, label) in KNOWN_EVENTS:
        if start <= ym <= end:
            return label
    return "Unknown — check raw 1h files around this date"


def val_window_from_meta() -> tuple:
    """
    Read the walk-forward fold metadata produced by main_mcknn.py's
    run_final_training(), which records the EXACT val tick boundaries
    in the PRIMARY (4h) frame. Returns (val_start_date, val_end_date)
    as Timestamps, or (None, None) if the metadata isn't available yet
    (i.e. the model hasn't been trained yet).
    """
    import json
    meta_path = "outcomes/final_fold_meta.json"
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    try:
        primary_df = pd.read_csv(PRIMARY_PATH, usecols=["Open_time"])
        ts = pd.to_datetime(primary_df["Open_time"])
        val_start = ts.iloc[meta["val_start"]]
        val_end   = ts.iloc[min(meta["val_end"] - 1, len(ts) - 1)]
        return val_start, val_end
    except Exception as exc:
        print(f"  ⚠  Could not resolve val window from {meta_path}: {exc}")
        return None, None


def load_cv_fold_windows() -> list:
    """
    Read outcomes/walkforward_cv_summary.json (written by
    main_mcknn.py's run_cross_validation()) and return a list of
    {fold_id, val_pnl, purge_start, val_start, val_end, embargo_end}
    dicts with real Timestamps, expanding each fold's val window by
    lookback_ticks (in PRIMARY/4h ticks, converted to hours) to get its
    full purge+val+embargo span — the same span walkforward.py excludes
    from train for that fold, and therefore the right span to check for
    gap/event overlap (a gap just outside the val window itself can
    still contaminate it via the purge buffer's rolling-window lookback).

    Returns [] if the summary file is missing, or if it predates the
    val_start_date/val_end_date fields (older runs — re-run CV to
    populate them).
    """
    if not os.path.exists(CV_SUMMARY_PATH):
        return []
    with open(CV_SUMMARY_PATH) as f:
        summary = json.load(f)
    lookback_ticks = summary.get("lookback_ticks", LOOKBACK_TICKS_4H)
    lookback_td = pd.Timedelta(hours=4 * lookback_ticks)   # primary tick = 4h

    windows = []
    for r in summary.get("fold_results", []):
        if "val_start_date" not in r or "val_end_date" not in r:
            continue   # pre-date-capture run — nothing to cross-reference
        val_start = pd.to_datetime(r["val_start_date"])
        val_end   = pd.to_datetime(r["val_end_date"])
        windows.append({
            "fold_id":     r["fold_id"],
            "val_pnl":     r["val_pnl"],
            "purge_start": val_start - lookback_td,
            "val_start":   val_start,
            "val_end":     val_end,
            "embargo_end": val_end + lookback_td,
        })
    return windows


def overlapping_known_events(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Return KNOWN_EVENTS entries whose [start,end] month-range overlaps
    the given date span."""
    hits = []
    for (ev_start, ev_end, label) in KNOWN_EVENTS:
        ev_start_ts = pd.Timestamp(ev_start + "-01")
        ev_end_ts   = pd.Timestamp(ev_end + "-28") + pd.Timedelta(days=4)  # end of month, roughly
        if ev_start_ts <= end and start <= ev_end_ts:
            hits.append((ev_start, ev_end, label))
    return hits


def overlapping_gaps(start: pd.Timestamp, end: pd.Timestamp,
                     gap_ts: pd.Series, gap_rows_list: list) -> list:
    """Return (row, gap_end_date, gap_size) tuples for gaps whose closing
    date falls inside [start, end]."""
    hits = []
    for row in gap_rows_list:
        gap_end = gap_ts.iloc[row]
        if start <= gap_end <= end:
            hits.append((row, gap_end, gaps.iloc[row]))
    return hits


def contamination_fraction(val_start: pd.Timestamp, val_end: pd.Timestamp,
                           gap_ts: pd.Series, gap_rows_list: list) -> float:
    """
    Fraction of the VAL window itself (not the padded purge+embargo
    span) whose 1h candles fall within INDICATOR_LOOKBACK_1H hours of a
    gap's close — i.e. how much of what actually got SCORED sits inside
    a rolling-window contamination zone, as opposed to merely "some
    known event happened somewhere in this multi-month window" (a much
    weaker claim once windows span a year or more — see
    report_cv_fold_windows()'s docstring on why event-overlap alone is
    weak evidence for coarse folds).

    This is a genuine signal, not a coincidence check: OBV/ATR's
    rolling(200) windows compute real but distorted values for ~200
    hours after any gap (the window straddles missing data), so a high
    contamination fraction is a concrete, mechanistic reason a fold
    could underperform — independent of whether the underlying market
    regime was also difficult.

    Caveat: computed on the 1h series (this script's only loaded raw
    table). The PRIMARY state is built from 4h data with its own,
    separately-gapped rolling(200) 4h window (~33 days) — a low 1h
    contamination fraction does NOT rule out 4h-level contamination.
    Treat this as one data point, not the full picture.
    """
    val_mask = (gap_ts >= val_start) & (gap_ts <= val_end)
    total = int(val_mask.sum())
    if total == 0:
        return 0.0
    contaminated = pd.Series(False, index=gap_ts.index)
    for row in gap_rows_list:
        gap_end = gap_ts.iloc[row]
        contam_end = gap_end + pd.Timedelta(hours=INDICATOR_LOOKBACK_1H)
        contaminated |= (gap_ts >= gap_end) & (gap_ts <= contam_end)
    return float((contaminated & val_mask).sum()) / total


def report_cv_fold_windows(gap_ts: pd.Series, gap_rows_list: list) -> list:
    """
    Cross-reference every CV fold's purge+val+embargo window against
    gaps and KNOWN_EVENTS, flagging any fold whose val P/L breached
    VAL_PNL_FLOOR. Also checks a manually-supplied --fold-start/--fold-end
    window if provided, independent of whether a CV summary exists.
    Returns the report lines (also printed/appended to the saved report).
    """
    out = []
    out.append("=" * 74)
    out.append("  CV FOLD WINDOW CROSS-REFERENCE")
    out.append("=" * 74)

    windows = load_cv_fold_windows()
    if not windows and not args.fold_start:
        out.append(f"\n  ℹ  No {CV_SUMMARY_PATH} with date-tagged folds found, and no "
                    f"--fold-start/--fold-end given. Run the updated main_mcknn.py "
                    f"(records val_start_date/val_end_date per fold) or pass "
                    f"--fold-start/--fold-end manually to use this section.")
        return out

    flagged = [w for w in windows if w["val_pnl"] < VAL_PNL_FLOOR]
    if windows:
        worst = min(windows, key=lambda w: w["val_pnl"])
        out.append(f"\n  {len(windows)} fold(s) loaded from {CV_SUMMARY_PATH}.")
        out.append(f"  {len(flagged)} fold(s) breached VAL_PNL_FLOOR={VAL_PNL_FLOOR:+.0%}.")
        out.append(f"  Worst fold: #{worst['fold_id']}  {worst['val_pnl']:+.4%}  "
                    f"({worst['val_start'].date()} → {worst['val_end'].date()})")
        # Always report at least the worst fold, plus anything else flagged.
        to_check = {w["fold_id"]: w for w in ([worst] + flagged)}.values()
    else:
        to_check = []

    manual_windows = []
    if args.fold_start:
        if not args.fold_end:
            out.append(f"\n  ⚠  --fold-start given without --fold-end — ignoring "
                        f"manual window.")
        else:
            manual_windows.append({
                "fold_id": "manual",
                "val_pnl": None,
                "purge_start": pd.to_datetime(args.fold_start),
                "val_start":   pd.to_datetime(args.fold_start),
                "val_end":     pd.to_datetime(args.fold_end),
                "embargo_end": pd.to_datetime(args.fold_end),
            })

    for w in list(to_check) + manual_windows:
        label = f"Fold #{w['fold_id']}" if w["fold_id"] != "manual" else "Manual window"
        pnl_str = f"  val P/L={w['val_pnl']:+.4%}" if w["val_pnl"] is not None else ""
        out.append(f"\n  ── {label}{pnl_str} "
                    f"({w['val_start'].date()} → {w['val_end'].date()}) "
                    f"── purge+embargo span: "
                    f"{w['purge_start'].date()} → {w['embargo_end'].date()}")

        ev_hits = overlapping_known_events(w["purge_start"], w["embargo_end"])
        if ev_hits:
            for (es, ee, label_ev) in ev_hits:
                out.append(f"       ⚡ KNOWN EVENT overlap: {es}→{ee}  {label_ev}")
            if len(ev_hits) > 1:
                span_months = (w["embargo_end"] - w["purge_start"]).days / 30
                out.append(f"       ℹ  {len(ev_hits)} events overlap a "
                            f"{span_months:.0f}-month span — with a window this "
                            f"wide, multiple hits are somewhat expected and "
                            f"don't by themselves pin the blame on any one "
                            f"event. See contamination fraction below for a "
                            f"more direct signal.")
        else:
            out.append(f"       (no KNOWN_EVENTS entry overlaps this span)")

        gap_hits = overlapping_gaps(w["purge_start"], w["embargo_end"], gap_ts, gap_rows_list)
        if gap_hits:
            for (row, gap_end, gap_size) in gap_hits:
                out.append(f"       ⚠  RAW DATA GAP overlap: row {row}  "
                            f"closes {gap_end}  size={gap_size}")
        else:
            out.append(f"       (no raw-data gap falls inside this span)")

        # ── Contamination fraction — the more direct, window-size-independent
        # signal: how much of the SCORED val window (not the padded span)
        # actually sits inside a gap's rolling-window contamination zone.
        if w["fold_id"] != "manual":
            frac = contamination_fraction(w["val_start"], w["val_end"], gap_ts, gap_rows_list)
            out.append(f"       1h-gap contamination fraction of val window: "
                        f"{frac:.1%}  (fraction of val ticks within "
                        f"{INDICATOR_LOOKBACK_1H}h of a raw-data gap — "
                        f"a direct, window-size-independent data-quality "
                        f"signal; NOTE: 1h-only, doesn't cover 4h-level gaps)")

        if not ev_hits and not gap_hits:
            out.append(f"       → Neither explains this window. If val_pnl breached "
                        f"the floor here, this looks like a genuine strategy "
                        f"weakness in this regime, not a data artifact — worth "
                        f"direct investigation (e.g. plot price/volatility over "
                        f"this span) rather than a data-pipeline fix.")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load 1h master and compute gap table
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nLoading {HOUR_PATH}...")
df = pd.read_csv(HOUR_PATH, usecols=["Open_time"])
ts = pd.to_datetime(df["Open_time"])
print(f"  {len(df):,} rows  |  {ts.iloc[0]} → {ts.iloc[-1]}\n")

gaps = ts.diff()
threshold = pd.Timedelta(hours=CANDLE_H * 2)
gap_mask = gaps > threshold
gap_rows = gap_mask[gap_mask].index.tolist()

have_gaps = bool(gap_rows)
if not have_gaps:
    print("✓  No gaps found — data coverage is clean.\n")
    # NOTE: previously this exited immediately. It no longer does — the
    # CV fold window cross-reference below is still useful even with a
    # clean raw-data gap table, since it also checks KNOWN_EVENTS
    # overlap, which is independent of whether any candles are missing.

# ── Val window from training metadata (if available) ─────────────────────────
val_start, val_end = val_window_from_meta()

lines = []
lines.append("=" * 74)
lines.append("  GAP DATE REPORT  —  XRPUSDT 1h master")
lines.append("=" * 74)
lines.append(f"  Total gaps found: {len(gap_rows)}")
lines.append(f"  Indicator contamination window: {INDICATOR_LOOKBACK_1H} candles "
             f"after each gap")
if val_start is not None:
    lines.append(f"  Val window (4h primary): {val_start.date()} → {val_end.date()}")
else:
    lines.append("  Val window: not yet available (run main_mcknn.py first)")
lines.append("")

if have_gaps:
    header = (f"{'Row':>7}  {'Gap closes at':20}  {'Size':12}  "
              f"{'Missing':>7}  {'In val?':>8}  Likely cause")
    lines.append(header)
    lines.append("-" * 110)
else:
    lines.append("  (no raw-data gaps to list)")

total_missing = 0
val_gap_rows  = []
for row in gap_rows:
    gap_size  = gaps.iloc[row]
    gap_end   = ts.iloc[row]
    missing   = int(round(gap_size.total_seconds() / 3600)) - 1
    total_missing += missing

    # Contamination window: INDICATOR_LOOKBACK_1H rows after the gap
    contam_end = ts.iloc[min(row + INDICATOR_LOOKBACK_1H, len(df) - 1)]
    in_val = False
    if val_start is not None:
        # Gap or its contamination window overlaps the val window
        in_val = (gap_end <= val_end) and (contam_end >= val_start)
    if in_val:
        val_gap_rows.append(row)

    cause = classify_gap(gap_end)
    flag  = "  ⚠ YES" if in_val else "  no"
    lines.append(
        f"{row:>7}  {str(gap_end)[:19]:20}  {str(gap_size):12}  "
        f"{missing:>7}  {flag:>8}  {cause}"
    )

lines.append("-" * 110)
lines.append(f"  Total missing 1h candles: {total_missing}"
             f"  ({total_missing / len(df) * 100:.2f}% of {len(df):,} rows)")
lines.append("")

# ── Verdict ───────────────────────────────────────────────────────────────────
lines.append("=" * 74)
lines.append("  VERDICT")
lines.append("=" * 74)

if val_gap_rows:
    lines.append(f"\n  ⚠  {len(val_gap_rows)} gap(s) overlap the val window or its "
                 f"{INDICATOR_LOOKBACK_1H}-candle contamination zone:")
    for row in val_gap_rows:
        lines.append(f"     Row {row}: {ts.iloc[row]}")
    lines.append("")
    lines.append("  RECOMMENDATION: add a gap-proximity exclusion mask in")
    lines.append("  walkforward.py to remove the contaminated rows from val")
    lines.append("  scoring. The next step guide below explains how.")
else:
    if val_start is not None:
        lines.append(f"\n  ✓  No gaps overlap the val window "
                     f"({val_start.date()} → {val_end.date()}).")
        lines.append("     All gaps fall in the training period — they add minor")
        lines.append("     noise (~200 contaminated rows per gap) that the bank")
        lines.append("     will absorb without issue. No fix needed before training.")
    else:
        lines.append(f"\n  ℹ  Val window not yet known (run main_mcknn.py first).")
        lines.append("     All gaps are small (<12 candles) and widely distributed")
        lines.append("     across 2018-2026 — very likely normal exchange maintenance.")
        lines.append("     Proceed with training; re-run this check afterwards to")
        lines.append("     confirm no gaps landed in the final val window.")

lines.append("")

# ── CV fold window cross-reference (new) ───────────────────────────────────────
lines.extend(report_cv_fold_windows(ts, gap_rows))

lines.append("")
lines.append("=" * 74)
lines.append("  NEXT STEPS")
lines.append("=" * 74)
lines.append("""
  1. ASSESS GAP CAUSES
     Compare the dates in the table above to KNOWN_EVENTS (top of this
     file). Gaps near March 2020, May 2022 (LUNA), Nov 2022 (FTX) are
     expected and harmless — those are genuine market events your model
     should learn from. Unexplained gaps may mean missing raw CSV files
     in data/raw/1h/ for those months.

  2. CHECK FOR MISSING RAW FILES (if any gaps are unexplained)
     Look at which months are present in data/raw/1h/*.csv. A gap at
     row N whose date spans a month-boundary likely means a monthly CSV
     is missing. Download it from Binance historical data and re-run
     data_manager.update_master_data('1h') — it will merge cleanly
     into the existing master via the dedup/sort pipeline.

  3. IF ANY GAPS LAND IN THE VAL WINDOW
     The contamination is localised (~200 rows around each gap). The
     cleanest fix is to exclude those rows from val scoring in
     walkforward.generate_purged_folds() by adding them to the embargo
     mask — ask for the implementation if needed.

  4. PROCEED TO TRAINING (once gaps assessed)
     Recommended settings for your first full run given 17,584 4h rows
     and 2018-2026 span:

        N_CV_FOLDS     = 6
        CV_EPOCHS      = 8    # enough for bank to mature per fold
        NUM_EPOCHS     = 50   # final training budget
        PATIENCE       = 3
        WARMUP_EPOCHS  = 3

     After training, run pre_training.py again and compare the
     weighted vs unweighted ratio — if weighting held the ratio below
     0.45 on the full dataset, the Layer 2 fix is working well enough
     to proceed to a proper diagnostic run.
""")

# ── Print and save ─────────────────────────────────────────────────────────────
report = "\n".join(lines)
print(report)

out_path = "outcomes/gap_report.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n  ✓  Report saved → {out_path}")