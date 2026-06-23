"""
walkforward.py
─────────────────
Purged + embargoed walk-forward fold generator.

Why this exists
──────────────────
The previous validation scheme (VAL_YEARS = [2022, 2025] in main_mcknn.py)
treats calendar years as hard train/val boundaries, but several features
in this pipeline are NOT memoryless at the boundary:

  - preprocessing.py's OBV_Scaled / ATR_Scaled use rolling(200) windows
    (≈ 33 days on 4h candles).
  - state_aggregator.py's StateAggregator runs MultiPaceAgent at paces
    up to 90 with an 8-tick history window each (≈ 15 days on 4h candles
    for the pace=90 agent: 8 * 90 = 720 ticks of *raw* lookback, though
    only 8 *sampled* points are kept — the effective calendar span
    touched is what matters here, not just the sample count).

  A state computed for a tick on, say, Jan 3 2022 statistically depends
  on price/volume data from late December 2021. If Dec 2021 is in TRAIN
  and Jan 2022 is in VAL, the split is not actually independent — this
  is a real leak that the previous MIN_TICK_GAP fix in mc_knn_memory.py
  does NOT address, because that gap only excludes a query from voting
  on rows committed *during the same training pass* (same episode_id),
  not from training on data that overlaps a validation window's lookback
  at all.

What this module does
──────────────────────
Given a tick-indexed DataFrame (already feature-engineered, one row per
decision point — e.g. the 4h master frame) and a `lookback_ticks` value
sized to the LARGEST lookback among all features/timeframes actually
feeding the state vector, this generates a sequence of walk-forward
folds where:

  - the val window is a contiguous block of `val_span_ticks`,
  - the train set is everything else MINUS a `lookback_ticks`-wide
    buffer immediately before the val window (PURGE — stops training
    rows whose own rolling windows overlap the val window from being
    used to predict into it) and immediately after the val window
    (EMBARGO — stops val-window information leaking forward into
    training rows that resume right after).

Sizing lookback_ticks correctly across timeframes
───────────────────────────────────────────────────
If you are fusing 15m/1h/4h timeframes into one state (see
multi_timeframe_state.py), `lookback_ticks` MUST be expressed in units
of the PRIMARY (4h) tick index, but sized to the longest lookback in
ABSOLUTE TIME across all timeframes, not the longest lookback in
ticks of any single timeframe. A rolling(200) window on 15m bars is
only 50 hours of real time (≈ 12.5 ticks of 4h data), while the same
rolling(200) window on 4h bars is ~33 days (≈ 200 ticks of 4h data).
Always compute the worst case in real time, then convert to 4h-tick
units. See `compute_required_lookback_ticks()` below for a helper that
does this conversion explicitly rather than asking the caller to do
unit arithmetic by hand.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One purged+embargoed walk-forward fold, expressed as boolean
    masks over the ORIGINAL (un-split) primary-timeframe DataFrame so
    callers can slice train_df/val_df with `df[fold.train_mask]` /
    `df[fold.val_mask]` without re-deriving indices."""
    fold_id:    int
    train_mask: np.ndarray   # bool, len == len(df)
    val_mask:   np.ndarray   # bool, len == len(df)
    val_start:  int          # row index (inclusive) of val window
    val_end:    int          # row index (exclusive) of val window
    purge_start:  int        # row index where the purge buffer begins
    embargo_end:  int        # row index where the embargo buffer ends


def compute_required_lookback_ticks(
    primary_tick_minutes: int = 240,                 # 4h
    timeframe_lookback_minutes: dict = None,
) -> int:
    """
    Convert the worst-case lookback across all timeframes feeding the
    state vector into a tick count expressed in the PRIMARY timeframe's
    units, so it can be used directly as `lookback_ticks` in
    generate_purged_folds().

    Parameters
    ----------
    primary_tick_minutes
        Minutes per tick of the decision-making timeframe (4h = 240).
    timeframe_lookback_minutes
        Dict mapping a human label -> the longest lookback that
        timeframe's features use, IN MINUTES OF REAL TIME (not ticks).
        Defaults to this project's actual feature windows:
          - "4h_rolling200":   200 * 240          = 33.3 days
          - "4h_pace90_hist8": 90 * 8 * 240        = 120 days  (worst
            case raw calendar span an 8-point pace-90 history could
            reference; in practice MultiPaceAgent samples sparsely, but
            for a SAFE purge buffer we use the full span it COULD touch)
          - "1h_rolling200":   200 * 60            = 8.3 days
          - "15m_rolling200":  200 * 15            = 50 hours

    Returns
    -------
    int — lookback expressed in primary-timeframe ticks, rounded up.
    """
    if timeframe_lookback_minutes is None:
        timeframe_lookback_minutes = {
            "4h_rolling200":   200 * 240,
            "4h_pace90_hist8": 90 * 8 * 240,
            "1h_rolling200":   200 * 60,
            "15m_rolling200":  200 * 15,
        }
    worst_minutes = max(timeframe_lookback_minutes.values())
    return int(np.ceil(worst_minutes / primary_tick_minutes))


def generate_purged_folds(
    df: pd.DataFrame,
    n_folds: int = 6,
    val_span_ticks: int = None,
    lookback_ticks: int = None,
    min_train_ticks: int = 1000,
) -> list:
    """
    Generate `n_folds` walk-forward folds over `df`, each with a purge
    buffer before and an embargo buffer after the val window.

    Layout for fold i (val windows slide forward through the dataset,
    non-overlapping):

        [ ... TRAIN ... ][ purge ][ VAL window ][ embargo ][ ... TRAIN ... ]
        |<------------------ everything else in df is potential TRAIN ----->|

    Rows inside [purge_start, val_start) and [val_end, embargo_end) are
    excluded from BOTH train and val for this fold — they're the
    "no-man's-land" that could leak in either direction.

    Parameters
    ----------
    df              : primary-timeframe feature-engineered DataFrame,
                       already sorted by time, default RangeIndex.
    n_folds         : number of walk-forward folds.
    val_span_ticks  : ticks per val window. If None, computed as
                       len(df) // (n_folds + 1) so folds roughly tile
                       the dataset with room for one trailing train-only
                       segment.
    lookback_ticks  : purge/embargo buffer width. If None, falls back
                       to compute_required_lookback_ticks()'s default
                       (sized to this project's actual feature windows).
    min_train_ticks : skip a fold (with a warning) if purging would
                       leave fewer than this many usable train rows —
                       protects early folds in a long history where the
                       train set might otherwise be too thin.

    Returns
    -------
    list[Fold]
    """
    n = len(df)
    if lookback_ticks is None:
        lookback_ticks = compute_required_lookback_ticks()
    if val_span_ticks is None:
        val_span_ticks = max(1, n // (n_folds + 1))

    folds = []
    # Slide the val window forward, leaving room for at least one
    # lookback-sized purge buffer before it (so fold 0 still has some
    # genuine train history) and one embargo buffer after it.
    earliest_val_start = lookback_ticks
    latest_val_start    = n - val_span_ticks - lookback_ticks
    if latest_val_start <= earliest_val_start:
        raise ValueError(
            f"Dataset too short ({n} rows) for n_folds={n_folds}, "
            f"val_span_ticks={val_span_ticks}, lookback_ticks={lookback_ticks}. "
            f"Reduce n_folds/val_span_ticks or increase data."
        )

    val_starts = np.linspace(earliest_val_start, latest_val_start, n_folds).astype(int)

    for fold_id, val_start in enumerate(val_starts):
        val_end      = val_start + val_span_ticks
        purge_start  = max(0, val_start - lookback_ticks)
        embargo_end  = min(n, val_end + lookback_ticks)

        val_mask   = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True

        train_mask = np.ones(n, dtype=bool)
        train_mask[purge_start:embargo_end] = False   # purge + val + embargo all excluded from train

        n_train = int(train_mask.sum())
        if n_train < min_train_ticks:
            print(f"  ⚠ Fold {fold_id}: only {n_train} train rows after "
                  f"purge/embargo (< min_train_ticks={min_train_ticks}) — skipping.")
            continue

        folds.append(Fold(
            fold_id=fold_id,
            train_mask=train_mask,
            val_mask=val_mask,
            val_start=int(val_start),
            val_end=int(val_end),
            purge_start=int(purge_start),
            embargo_end=int(embargo_end),
        ))

    return folds


def summarize_folds(folds: list, df: pd.DataFrame, time_col: str = "Open_time") -> None:
    """Pretty-print fold date ranges and row counts for sanity-checking
    that purge/embargo buffers look right before committing to a long
    training run."""
    print(f"\n  Generated {len(folds)} purged+embargoed walk-forward fold(s):")
    for f in folds:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        val_lo, val_hi = ts.iloc[f.val_start], ts.iloc[f.val_end - 1]
        n_train = int(f.train_mask.sum())
        n_val   = int(f.val_mask.sum())
        n_purged = (f.embargo_end - f.purge_start) - n_val
        print(f"    Fold {f.fold_id}: val {val_lo.date()} → {val_hi.date()}  "
              f"|  train rows={n_train:,}  val rows={n_val:,}  "
              f"purged/embargoed (excluded from both)={n_purged:,}")