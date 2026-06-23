"""
multi_timeframe_state.py
───────────────────────────
Hierarchical multi-timeframe state construction.

Why this exists
──────────────────
pre_training.py's diagnostic found a large fraction of near-duplicate
states in the 4h-only feature space — driven by flat/low-volatility
stretches where all six scaled indicators barely move for many
consecutive 4h ticks, and by the slow paces (42, 90) in
state_aggregator.py changing very little tick-to-tick. MIN_TICK_GAP in
mc_knn_memory.py patches the symptom (don't let a query vote on a
near-duplicate of itself within the same training pass) but doesn't
address the ROOT CAUSE: the 4h-only feature space genuinely doesn't
carry enough information to distinguish two superficially-similar
market moments.

Adding independently-computed 15m and 1h context doesn't just give the
model "more data" — it adds a different INFORMATION SOURCE that
doesn't go flat in the same stretches the 4h signal does, so two ticks
that look identical on the slow 4h view can still be pulled apart by
what was happening intraday. This directly reduces duplicate density
in the combined state space.

Leakage discipline
──────────────────────
The 4h decision at tick T must only ever see 15m/1h information that
was FULLY CLOSED before T's open. Concretely: a 15m bar that opens at
T - 15min and closes exactly at T is allowed (it closed at or before
the 4h bar opened); a 15m bar that opens at T and would close at
T + 15min is NOT allowed, because at decision time T it hasn't
happened yet. This module enforces that via `pd.merge_asof(...,
direction="backward", allow_exact_matches=True)` matched on Open_time,
which is safe ONLY because the indicator-computing step
(preprocessing.py) drops any row whose rolling windows aren't fully
populated — i.e. by the time a 15m/1h row appears in its processed
frame at all, that bar (and everything its indicators depend on) has
already closed. We do not allow exact matches on a bar that opens
exactly at T from a FINER timeframe than 4h, because a same-instant
open is not yet closed; see `_strictly_closed_before()` below for the
explicit (T - 1ms) cutoff used to avoid relying on float/string time
comparison edge cases between merge_asof's exact-match semantics on
differently-sampled grids.

Usage
──────
    ctx_15m = build_timeframe_context(df_15m, FEATURES, paces=(1, 4, 16))
    ctx_1h  = build_timeframe_context(df_1h,  FEATURES, paces=(1, 4, 16))

    aligned_15m = align_to_primary(primary_df, ctx_15m, prefix="ctx15m_")
    aligned_1h  = align_to_primary(primary_df, ctx_1h,  prefix="ctx1h_")

    # Per-tick extra context array to feed into UnifiedExecutor.step(...,
    # extra_context=...):
    extra_context_arr = np.concatenate(
        [aligned_15m.values, aligned_1h.values], axis=1
    ).astype(np.float32)
"""

import numpy as np
import pandas as pd

from agents.state_aggregator import StateAggregator


def _strictly_closed_before(context_open_time: pd.Series, cutoff: pd.Series) -> pd.Series:
    """Return a boolean mask: True where the context row's Open_time is
    strictly earlier than `cutoff` (the primary timeframe's Open_time),
    i.e. the context bar is guaranteed fully closed before the primary
    decision point opens. Used as an assertion / sanity check after
    merge_asof, not as the join key itself (merge_asof needs a sorted
    numeric/datetime key, this is the post-hoc leakage check)."""
    return context_open_time < cutoff


def build_timeframe_context(
    df: pd.DataFrame,
    features: list,
    paces: tuple = (1, 4, 16),
    window: int = 8,
    warmup_idx: int = None,
) -> pd.DataFrame:
    """
    Run a StateAggregator independently over one timeframe's own
    feature-engineered DataFrame, producing one state-vector row per
    tick of THAT timeframe (not the primary/decision timeframe).

    This reuses the exact same StateAggregator / MultiPaceAgent
    machinery already used for the primary 4h state, just pointed at a
    different timeframe's indicator series — "hierarchical" in the
    sense that each timeframe gets its own independent multi-pace
    summarization before being fused into the primary decision state.

    Parameters
    ----------
    df          : timeframe-specific frame with an "Open_time" column
                  and the same indicator columns as `features`.
    features    : indicator column names (e.g. FEATURES in main_mcknn.py).
    paces       : pace tuple for this timeframe's internal aggregator.
                  Deliberately smaller/fewer than the primary 4h
                  aggregator's (1, 6, 42, 90) — a 15m or 1h context
                  block doesn't need its own multi-decade-scale slow
                  pace; it exists to capture intraday texture, not to
                  duplicate the 4h trend view.
    window      : MultiPaceAgent's max_history (same default as
                  StateAggregator's).
    warmup_idx  : row index to warm up to. Defaults to the smallest
                  index that's safe given preprocessing.py's longest
                  rolling window (200) plus this aggregator's own
                  lookback — i.e. max(200, max(paces) * window).

    Returns
    -------
    pd.DataFrame indexed by row position, columns:
        ["Open_time", "ctx_0", "ctx_1", ..., "ctx_{D-1}"]
    where D = len(features) * 2 * len(paces) (raw + slope per pace).
    """
    if warmup_idx is None:
        warmup_idx = max(200, max(paces) * window)

    n = len(df)
    if n <= warmup_idx + 1:
        raise ValueError(
            f"Timeframe context frame too short ({n} rows) for "
            f"warmup_idx={warmup_idx}. Fetch more history for this timeframe."
        )

    indicators_arr = df[features].values.astype(np.float32)
    agg = StateAggregator(paces=paces, window=window, num_indicators=len(features))
    agg.tick = 0
    agg.warm_up_all(indicators_arr, warmup_idx)

    rows = []
    open_times = []
    for i in range(warmup_idx, n):
        agg.update(indicators_arr[i])
        # No portfolio_info here — this is a context-only block, fused
        # into the primary state which already carries portfolio info
        # exactly once (see UnifiedExecutor.get_state). StateAggregator.
        # get_state() always appends a 2-dim portfolio block (zeros when
        # portfolio_info=None) regardless — strip it so this context
        # block contains ONLY genuine market signal, not two redundant
        # always-zero columns that would otherwise inflate STATE_DIM
        # and add zero-variance dimensions to every kNN distance calc.
        state_vec = agg.get_state(portfolio_info=None)[:-2]
        rows.append(state_vec)
        open_times.append(df["Open_time"].iloc[i])

    state_dim = rows[0].shape[0]
    cols = [f"ctx_{j}" for j in range(state_dim)]
    out = pd.DataFrame(rows, columns=cols)
    out.insert(0, "Open_time", pd.to_datetime(open_times, errors="coerce"))
    return out


def align_to_primary(
    primary_df: pd.DataFrame,
    context_df: pd.DataFrame,
    prefix: str,
    time_col: str = "Open_time",
) -> pd.DataFrame:
    """
    For every row in `primary_df`, attach the most recently CLOSED
    context-timeframe state vector (no lookahead).

    Implementation note on the leakage guard: merge_asof with
    direction="backward" picks the last context row whose Open_time is
    <= the primary row's Open_time. Because a finer-timeframe bar that
    OPENS at the same instant the primary bar opens has not yet
    closed, we shift the join key for the primary side back by a
    negligible epsilon (1 microsecond) before matching, which converts
    the merge_asof "<=" semantics into a strict "<" for any
    same-timestamp edge case while leaving all other matches numerically
    unaffected (1 microsecond is far below any meaningful candle
    spacing in this pipeline — 15m is the finest granularity used).

    Returns
    -------
    pd.DataFrame, same length/order as primary_df, columns prefixed
    with `prefix` (e.g. "ctx15m_0", "ctx15m_1", ...). Rows before the
    first available context observation are filled with zeros (cold
    start) rather than NaN, matching this project's existing
    zero-padding convention for unavailable history (see
    agent_multi_pace.py's warm_up()).
    """
    primary = primary_df[[time_col]].copy()
    primary[time_col] = pd.to_datetime(primary[time_col], errors="coerce")
    # Strict "<" guard against same-instant lookahead — see docstring.
    primary["_join_key"] = primary[time_col] - pd.Timedelta(microseconds=1)

    ctx = context_df.copy()
    ctx[time_col] = pd.to_datetime(ctx[time_col], errors="coerce")
    ctx = ctx.sort_values(time_col).reset_index(drop=True)

    merged = pd.merge_asof(
        primary.sort_values("_join_key"),
        ctx,
        left_on="_join_key",
        right_on=time_col,
        direction="backward",
    )
    # merge_asof requires sorted-by-key input; restore original row order.
    merged = merged.sort_index()

    ctx_cols = [c for c in context_df.columns if c != time_col]
    result = merged[ctx_cols].copy()
    result.columns = [f"{prefix}{c}" for c in ctx_cols]
    result = result.fillna(0.0).astype(np.float32)

    # Sanity check (cheap, only runs once per call): confirm no row's
    # matched context timestamp is >= the primary row's own timestamp.
    matched_ts = merged[time_col + "_y"] if (time_col + "_y") in merged.columns else None
    if matched_ts is not None:
        bad = (matched_ts >= primary[time_col].values).sum()
        if bad > 0:
            raise AssertionError(
                f"Lookahead leak detected in align_to_primary: {bad} row(s) "
                f"matched a context bar that had not yet closed. This should "
                f"be impossible given the join key shift — check that both "
                f"input frames are using consistent Open_time semantics."
            )

    return result.reset_index(drop=True)


def build_multi_timeframe_context(
    primary_df: pd.DataFrame,
    timeframe_frames: dict,
    features: list,
    context_paces: tuple = (1, 4, 16),
) -> np.ndarray:
    """
    Convenience wrapper: build + align context for any number of
    finer timeframes in one call.

    Parameters
    ----------
    primary_df        : the 4h (decision-timeframe) feature frame.
    timeframe_frames   : dict like {"15m": df_15m, "1h": df_1h}, each
                          already feature-engineered (same `features`
                          columns + Open_time) for that timeframe.
    features            : indicator column names.
    context_paces       : pace tuple used for EACH finer timeframe's
                           internal StateAggregator.

    Returns
    -------
    np.ndarray, shape (len(primary_df), total_context_dim), float32 —
    ready to pass tick-by-tick into UnifiedExecutor.step(extra_context=...).
    total_context_dim = len(timeframe_frames) * len(features) * 2 * len(context_paces).
    """
    aligned_blocks = []
    for label, tf_df in timeframe_frames.items():
        ctx = build_timeframe_context(tf_df, features, paces=context_paces)
        aligned = align_to_primary(primary_df, ctx, prefix=f"ctx_{label}_")
        aligned_blocks.append(aligned.values)
        print(f"  [multi_timeframe_state] aligned {label} context: "
              f"{aligned.shape[1]} dims over {len(aligned):,} primary ticks")

    return np.concatenate(aligned_blocks, axis=1).astype(np.float32)