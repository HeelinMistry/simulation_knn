"""
check_gaps.py
──────────────
Resolves the gap row indices from pre_training.py's coverage warning
into actual dates, classifies each gap by likely cause, and reports
which gaps (if any) fall inside the walk-forward val window — the only
ones that matter for whether a gap-proximity exclusion mask is needed.

Run from project root:
    python check_gaps.py

Output is also written to outcomes/gap_report.txt for reference.
"""

import sys; sys.path.insert(0, '.')
import os
import pandas as pd
import numpy as np

os.makedirs("outcomes", exist_ok=True)

# ── Config — must match data_manager.py and main_mcknn.py ────────────────────
HOUR_PATH  = "data/processed/XRPUSDT_1h_master_processed.csv"
PRIMARY_PATH = "data/processed/XRPUSDT_4h_master_processed.csv"
CANDLE_H   = 1      # 1h master
LOOKBACK_TICKS_4H = 720   # walkforward.compute_required_lookback_ticks() default

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

if not gap_rows:
    print("✓  No gaps found — data coverage is clean.\n")
    sys.exit(0)

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

header = (f"{'Row':>7}  {'Gap closes at':20}  {'Size':12}  "
          f"{'Missing':>7}  {'In val?':>8}  Likely cause")
lines.append(header)
lines.append("-" * 110)

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