"""
cross_sectional_panel.py
─────────────────────────
Layer 3 (panel alignment) + Layer 4 (panel-aware purge/embargo/CPCV) for
running the GBT meta-labeling pipeline across MULTIPLE symbols at once
("cross-sectional"), instead of one symbol in isolation.

WHY THIS IS A SEPARATE MODULE, NOT A CHANGE TO master_processed.csv
──────────────────────────────────────────────────────────────────
The per-symbol master CSVs (data_manager.py), per-symbol indicator
computation (preprocessing.py), and per-symbol state construction
(main_gbt.build_states) all assume a SINGLE CONTINUOUS calendar time
series. Naively concatenating multiple symbols' rows into one file
would corrupt:
  - preprocessing.py's rolling(200)/rolling(14) windows across the
    symbol boundary (a window straddling two symbols' data is
    meaningless),
  - state_aggregator.py's tick-modulo pace sampling across the
    boundary (MultiPaceAgent's history would blend two unrelated
    price series),
  - triple_barrier.py's forward-looking barrier scan (prices[i+h]
    would read into the NEXT symbol's price action near a boundary —
    a real lookahead/cross-contamination leak, not just noise),
  - walkforward.py/cpcv.py's row-index-based purge/embargo, which
    assumes one row index <-> one calendar time mapping throughout.

This module instead:
  1. Builds each symbol's state matrix and triple-barrier labels
     completely independently, via main_gbt.py's EXISTING, already
     leakage-safe build_states_and_labels() — nothing about
     single-symbol correctness changes (build_symbol_data()).
  2. Aligns the symbols onto a SHARED calendar timestamp grid
     (align_panel()), remapping each symbol's triple-barrier "touch"
     indices onto the new panel row space rather than silently
     invalidating or corrupting them.
  3. Generates purge/embargo/CPCV folds keyed on the SHARED timestamp
     index (generate_panel_cpcv_paths()), so a fold's train/test split
     is the SAME set of calendar dates for every symbol simultaneously
     — this is what actually prevents a leak where e.g. BTC's train
     rows sit purge-adjacent (in calendar time) to ETH's test rows,
     something per-symbol-independent purging could never catch.
  4. Evaluates P&L per symbol independently within a given mask
     (panel_simulate_pnl()) — cross-sectional trading genuinely allows
     one open position PER SYMBOL concurrently (mirroring one
     UnifiedExecutor per symbol live), so this reuses
     main_gbt.simulate_pnl()'s single-open-position walk UNCHANGED,
     once per symbol, then aggregates.

ROW-ORDER INVARIANT (read before modifying align_panel)
──────────────────────────────────────────────────────────
Panel.states/best_action/long_touch/etc. are stored as SYMBOL-MAJOR
CONTIGUOUS BLOCKS (all of symbol 0's rows, then all of symbol 1's
rows, ...), each internally in chronological order. panel_simulate_pnl
does not rely on this (it slices per symbol explicitly), but if you
add new code that calls main_gbt.simulate_pnl() directly on the full
panel arrays, it will only produce correct results because of this
block structure — do not reorder Panel's arrays without updating every
touch index accordingly.

CHANGES IN THIS REVISION — GATE-HARDENING PARITY WITH main_gbt.py
────────────────────────────────────────────────────────────────────
The first 7-symbol cross-sectional run (main_cross_sectional_gbt.py)
failed its own confirm_gate() on PBO (54.1% > 40% allowed) and mean
test Sharpe (0.098 < 0.15 floor), and even the checks it did pass
(directional consistency 74/148 = exactly 50.0%) did so with no
margin at all. Reviewing that run also surfaced two real coverage
gaps that had nothing to do with whether the strategy itself has an
edge:

  1. This module never checked whether the panel's edge is
     concentrated in stale history (main_gbt.py's "recent-regime"
     check, confirm_gate_nested()'s GATE HARDENING change 2, was never
     ported here — see this module's own prior docstring note).
  2. This module never ran the logistic-regression capacity-floor
     baseline or the resulting baseline-edge gate (main_gbt.py's
     run_logistic_baseline() + MIN_EDGE_OVER_BASELINE check), so there
     was no way to tell whether a GBT-specific edge exists at all
     versus the GBT merely rediscovering the same marginal, shared
     signal a near-linear model would also find.
  3. The excluded-path pattern in that run's cpcv_summary.json (path
     ids 7, 8, 10, 17, 18, 22, 27 recurring as "too few test trades"
     across MULTIPLE seeds) was invisible in the printed output —
     random_state only ever reaches model fitting, never which rows
     exist in a path's train/test mask, so a test_groups combination
     excluded in most/all seeds is a structural data-coverage gap
     (e.g. align_panel()'s strict inner-join leaving too few
     overlapping, barrier-resolved rows for that calendar window
     across every symbol at once), not per-seed noise, and is worth
     surfacing distinctly from ordinary per-seed variance.

This revision adds panel_extract_recent_regime_result() (item 1),
run_panel_logistic_baseline() (item 2), and
summarize_recurring_exclusions() (item 3) — all reusing existing
main_gbt.py constants/functions so the panel pipeline can never
silently drift out of sync with the single-symbol pipeline's own
definitions of these checks. main_cross_sectional_gbt.py wires all
three into confirm_gate()/main().

Usage
──────
    from cross_sectional_panel import build_panel, run_panel_cpcv

    panel = build_panel(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
                        frozen=True)
    path_results, fitted_paths = run_panel_cpcv(panel, random_state=0)
"""

import json
import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

import main_gbt as gbt
from walkforward import compute_required_lookback_ticks


# ─────────────────────────────────────────────
# Layer 3 — per-symbol data + panel alignment
# ─────────────────────────────────────────────

@dataclass
class SymbolData:
    """One symbol's independently-built state matrix + triple-barrier
    labels, PRE-alignment. states/labels/prices/open_time all share
    the same row indexing (main_gbt.build_states_and_labels()'s own
    row space) — touch_ticks in `labels` are indices into THIS array,
    not yet remapped to any shared panel space."""
    symbol: str
    states: np.ndarray            # (n, state_dim)
    prices: np.ndarray            # (n,)
    open_time: pd.Series          # (n,) datetime64, default RangeIndex
    labels: dict                  # triple_barrier.build_meta_labels() output


@dataclass
class Panel:
    """Multiple symbols' state/label data aligned onto ONE shared
    calendar timestamp grid, stored long-format (one row per
    (timestamp, symbol) pair that survived alignment) as
    SYMBOL-MAJOR CONTIGUOUS BLOCKS — see module docstring's
    "ROW-ORDER INVARIANT"."""
    symbols: list                 # list[str], length = n_symbols
    timestamps: pd.DatetimeIndex  # (T,) sorted, common to every symbol
    state_dim: int
    states: np.ndarray            # (n_rows, state_dim)
    symbol_ids: np.ndarray        # (n_rows,) int32, index into .symbols
    timestamp_ids: np.ndarray     # (n_rows,) int64, index into .timestamps
    best_action: np.ndarray       # (n_rows,)
    long_return: np.ndarray       # (n_rows,)
    short_return: np.ndarray      # (n_rows,)
    long_touch: np.ndarray        # (n_rows,) PANEL row index, or -1
    short_touch: np.ndarray       # (n_rows,) PANEL row index, or -1
    valid_mask: np.ndarray        # (n_rows,) bool

    def symbol_row_mask(self, symbol: str) -> np.ndarray:
        sid = self.symbols.index(symbol)
        return self.symbol_ids == sid

    def as_labels_dict(self) -> dict:
        """Shape main_gbt.simulate_pnl() expects — a dict with the
        five keys it reads, all in PANEL row-index space."""
        return {
            "best_action": self.best_action, "long_return": self.long_return,
            "short_return": self.short_return, "long_touch": self.long_touch,
            "short_touch": self.short_touch,
        }


def build_symbol_data(symbol: str, frozen: bool = False,
                      enable_multi_timeframe: bool = None,
                      tp_mult: float = None, sl_mult: float = None,
                      max_holding: int = None) -> SymbolData:
    """Thin wrapper around main_gbt.build_states_and_labels() — the
    single-symbol pipeline is reused completely unmodified; this just
    packages its output for align_panel() below."""
    states, labels, prices, aligned_df = gbt.build_states_and_labels(
        frozen=frozen, enable_multi_timeframe=enable_multi_timeframe,
        tp_mult=tp_mult, sl_mult=sl_mult, max_holding=max_holding,
        symbol=symbol,
    )
    open_time = pd.to_datetime(aligned_df["Open_time"], errors="coerce")
    return SymbolData(symbol=symbol, states=states, prices=prices,
                      open_time=open_time, labels=labels)


def align_panel(symbol_data_list: list) -> Panel:
    """
    Intersect all symbols onto a shared timestamp grid (a timestamp
    only survives if EVERY symbol has a row for it — a strict inner
    join, not a fuzzy/backward-looking one like
    multi_timeframe_state.align_to_primary(), since same-timeframe
    symbols should share exact candle-close times and any deviation is
    a genuine gap worth being conservative about), then remap each
    symbol's triple-barrier touch indices from "position in that
    symbol's OWN pre-alignment array" to "row index in the new
    long-format panel", so simulate_pnl()'s single-open-position walk
    keeps working correctly post-alignment.

    A touch that resolves at a timestamp that didn't survive the
    intersection (can happen right at the calendar edges of the
    overlap) is marked invalid (valid_mask=False) rather than guessed
    at — losing a handful of boundary rows is far safer than silently
    feeding simulate_pnl() a touch index that means something
    different than it thinks.

    NOTE (this revision): this strict inner join is also the leading
    suspect for the recurring path exclusions surfaced by
    summarize_recurring_exclusions() — a calendar window where even
    ONE symbol has a data gap (holiday-thin volume, an exchange outage,
    a listing that started later than the others) drops that window
    for every symbol at once, shrinking the row count available to
    CPCV's train/test masks for that window regardless of random_state.
    Fixing that (if warranted) would mean loosening the join itself —
    a real design decision with leakage implications, not something
    this revision changes; see summarize_recurring_exclusions()'s
    docstring for how to use its output to decide whether to.
    """
    common = None
    for sd in symbol_data_list:
        s = set(sd.open_time.tolist())
        common = s if common is None else (common & s)
    if not common:
        raise ValueError(
            "No overlapping timestamps across the requested symbols — check "
            "that every symbol's raw data actually covers an overlapping "
            "calendar range (see data_manager._log_coverage() output)."
        )

    common_sorted = sorted(common)
    timestamps = pd.DatetimeIndex(common_sorted)
    T = len(timestamps)
    ts_to_pos = {t: i for i, t in enumerate(common_sorted)}

    print(f"\n  [cross_sectional_panel] common timestamp grid: {T:,} rows "
          f"({timestamps[0]} → {timestamps[-1]})  across "
          f"{len(symbol_data_list)} symbols")

    symbols = [sd.symbol for sd in symbol_data_list]
    blocks = {k: [] for k in (
        "states", "symbol_ids", "timestamp_ids", "best_action",
        "long_return", "short_return", "long_touch", "short_touch",
        "valid_mask",
    )}

    row_offset = 0
    for sid, sd in enumerate(symbol_data_list):
        keep_mask = sd.open_time.isin(common_sorted).values
        kept_orig_idx = np.flatnonzero(keep_mask)
        n_kept = len(kept_orig_idx)
        dropped = len(sd.open_time) - n_kept
        print(f"    {sd.symbol}: {len(sd.open_time):,} rows -> {n_kept:,} kept "
              f"({dropped:,} dropped, not present in every symbol's series)")
        if n_kept == 0:
            raise ValueError(f"{sd.symbol} has zero rows in the common "
                             f"timestamp grid — its calendar range doesn't "
                             f"overlap the other symbols at all.")

        kept_ts = sd.open_time.iloc[kept_orig_idx]
        timestamp_ids = kept_ts.map(ts_to_pos).values.astype(np.int64)

        # Map "original row index in sd's OWN pre-alignment arrays" ->
        # "new row index in the panel" — needed because touch_ticks
        # are indices into sd's own original (pre-alignment) arrays.
        remap_arr = np.full(len(sd.open_time), -1, dtype=np.int64)
        remap_arr[kept_orig_idx] = row_offset + np.arange(n_kept)

        def _remap_touch(orig_touch: np.ndarray) -> np.ndarray:
            out = np.full(len(orig_touch), -1, dtype=np.int64)
            ok = orig_touch >= 0
            out[ok] = remap_arr[orig_touch[ok]]
            return out

        blocks["states"].append(sd.states[kept_orig_idx])
        blocks["symbol_ids"].append(np.full(n_kept, sid, dtype=np.int32))
        blocks["timestamp_ids"].append(timestamp_ids)
        blocks["best_action"].append(sd.labels["best_action"][kept_orig_idx])
        blocks["long_return"].append(sd.labels["long_return"][kept_orig_idx])
        blocks["short_return"].append(sd.labels["short_return"][kept_orig_idx])
        blocks["valid_mask"].append(sd.labels["valid_mask"][kept_orig_idx])
        blocks["long_touch"].append(_remap_touch(sd.labels["long_touch"][kept_orig_idx]))
        blocks["short_touch"].append(_remap_touch(sd.labels["short_touch"][kept_orig_idx]))

        row_offset += n_kept

    states = np.concatenate(blocks["states"], axis=0).astype(np.float32)
    symbol_ids = np.concatenate(blocks["symbol_ids"])
    timestamp_ids = np.concatenate(blocks["timestamp_ids"])
    best_action = np.concatenate(blocks["best_action"])
    long_return = np.concatenate(blocks["long_return"])
    short_return = np.concatenate(blocks["short_return"])
    long_touch = np.concatenate(blocks["long_touch"])
    short_touch = np.concatenate(blocks["short_touch"])
    valid_mask = np.concatenate(blocks["valid_mask"])

    # A barrier touch that resolved at a timestamp outside the common
    # grid (edge_loss) can no longer be walked by simulate_pnl()'s
    # single-open-position logic — invalidate those rows explicitly
    # rather than let a -1 index alias numpy's "last row" semantics.
    edge_loss = (((best_action == 0) & (long_touch < 0)) |
                 ((best_action == 1) & (short_touch < 0)))
    if edge_loss.any():
        valid_mask = valid_mask & ~edge_loss
        print(f"  [cross_sectional_panel] {int(edge_loss.sum()):,} row(s) had "
              f"a barrier touch outside the common grid — marked invalid.")

    return Panel(
        symbols=symbols, timestamps=timestamps, state_dim=int(states.shape[1]),
        states=states, symbol_ids=symbol_ids, timestamp_ids=timestamp_ids,
        best_action=best_action, long_return=long_return, short_return=short_return,
        long_touch=long_touch, short_touch=short_touch, valid_mask=valid_mask,
    )


def build_panel(symbols: list, frozen: bool = False,
                enable_multi_timeframe: bool = None,
                tp_mult: float = None, sl_mult: float = None,
                max_holding: int = None) -> Panel:
    """Convenience wrapper: build every symbol's data independently,
    then align. Raises if symbols end up with mismatched state_dim
    (usually means one symbol silently fell back to 4h-only because
    its 15m/1h raw data was missing — see build_states()'s
    FileNotFoundError handling in main_gbt.py — while others got full
    multi-timeframe context; a GBTAgent needs one consistent feature
    space across every symbol, so this is caught here rather than
    surfacing as a confusing shape-mismatch deep inside sklearn)."""
    print(f"\n{'='*62}\n  BUILDING CROSS-SECTIONAL PANEL — {len(symbols)} symbols\n"
          f"  {symbols}\n{'='*62}")
    symbol_data = []
    state_dims = {}
    for sym in symbols:
        print(f"\n{'-'*62}\n  [{sym}] building states + triple-barrier labels\n{'-'*62}")
        sd = build_symbol_data(sym, frozen=frozen,
                               enable_multi_timeframe=enable_multi_timeframe,
                               tp_mult=tp_mult, sl_mult=sl_mult, max_holding=max_holding)
        state_dims[sym] = sd.states.shape[1]
        print(f"  [{sym}] {len(sd.states):,} rows  state_dim={sd.states.shape[1]}  "
              f"range {sd.open_time.iloc[0]} → {sd.open_time.iloc[-1]}")
        symbol_data.append(sd)

    if len(set(state_dims.values())) > 1:
        raise ValueError(
            f"Symbols produced DIFFERENT state_dim values: {state_dims}. This "
            f"usually means multi-timeframe context is available for some "
            f"symbols but not others (a symbol missing data/raw/<symbol>/"
            f"15m|1h/ silently falls back to 4h-only — see build_states()'s "
            f"FileNotFoundError handling). Fix the underlying raw data before "
            f"building a panel; every symbol needs the same feature space."
        )

    panel = align_panel(symbol_data)
    print(f"\n  ✓  Panel built: {len(panel.states):,} total rows across "
          f"{len(panel.symbols)} symbols, {len(panel.timestamps):,} shared "
          f"timestamps, state_dim={panel.state_dim}")
    return panel


# ─────────────────────────────────────────────
# Layer 4 — panel-aware purge/embargo/CPCV
# ─────────────────────────────────────────────

@dataclass
class PanelCPCVPath:
    path_id: int
    test_groups: tuple
    train_mask: np.ndarray    # over PANEL ROWS (n_rows,)
    test_mask: np.ndarray     # over PANEL ROWS (n_rows,)


def _compute_group_boundaries(n: int, n_groups: int) -> list:
    """Same formula as cpcv.py's private _make_groups() — a local,
    dependency-free copy, matching the convention already used by
    label_sweep.py / regime_stationarity.py for the same reason."""
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(edges[g], edges[g + 1]) for g in range(n_groups)]


def generate_panel_cpcv_paths(panel: Panel, n_groups: int = None,
                              n_test_groups: int = None,
                              lookback_ticks: int = None,
                              min_train_rows: int = None,
                              max_paths: int = None) -> list:
    """
    Same combinatorial-purged-CV construction as cpcv.py, applied to
    the PANEL's shared timestamp axis (length T) instead of a single
    symbol's row index. Every symbol shares the EXACT SAME train/test
    calendar split for every path — see module docstring for why this
    is the piece that actually prevents cross-symbol leakage, which
    per-symbol-independent purging could never catch (a leak where
    BTC's train row sits purge-adjacent, in calendar time, to ETH's
    test row).

    `lookback_ticks` here is a count of SHARED CALENDAR TICKS (each
    tick = one shared timestamp, i.e. one primary-timeframe candle),
    identical units to walkforward.compute_required_lookback_ticks().

    NOTE: this function's `n_groups`/`n_test_groups` default to
    gbt.N_GROUPS/gbt.N_TEST_GROUPS — the SAME constants
    gbt.RECENT_REGIME_GROUPS is derived from. That's what lets
    panel_extract_recent_regime_result() below identify "the panel's
    own most-recently-observed regime" path by the exact same
    test_groups tuple main_gbt.py uses for a single symbol, without
    needing a separate panel-specific constant.
    """
    n_groups = gbt.N_GROUPS if n_groups is None else n_groups
    n_test_groups = gbt.N_TEST_GROUPS if n_test_groups is None else n_test_groups
    max_paths = gbt.MAX_PATHS if max_paths is None else max_paths
    min_train_rows = gbt.MIN_TRAIN_TICKS if min_train_rows is None else min_train_rows

    T = len(panel.timestamps)
    if lookback_ticks is None:
        lookback_ticks = compute_required_lookback_ticks()

    groups = _compute_group_boundaries(T, n_groups)
    all_combos = list(combinations(range(n_groups), n_test_groups))
    if len(all_combos) > max_paths:
        print(f"  ⚠ C({n_groups},{n_test_groups})={len(all_combos)} exceeds "
              f"max_paths={max_paths} — using the first {max_paths}.")
        all_combos = all_combos[:max_paths]

    paths = []
    for path_id, combo in enumerate(all_combos):
        test_ts_mask = np.zeros(T, dtype=bool)
        for g in combo:
            lo, hi = groups[g]
            test_ts_mask[lo:hi] = True

        exclude_ts_mask = test_ts_mask.copy()
        diffs = np.diff(test_ts_mask.astype(int))
        starts = np.flatnonzero(diffs == 1) + 1
        ends = np.flatnonzero(diffs == -1) + 1
        if test_ts_mask[0]:
            starts = np.r_[0, starts]
        if test_ts_mask[-1]:
            ends = np.r_[ends, T]
        for s, e in zip(starts, ends):
            lo = max(0, s - lookback_ticks)
            hi = min(T, e + lookback_ticks)
            exclude_ts_mask[lo:hi] = True

        train_ts_mask = ~exclude_ts_mask

        # Broadcast timestamp-level masks to PANEL ROWS via each row's
        # timestamp_id — this is what makes every symbol share the
        # same calendar train/test split.
        train_mask = train_ts_mask[panel.timestamp_ids] & panel.valid_mask
        test_mask = test_ts_mask[panel.timestamp_ids] & panel.valid_mask

        if train_mask.sum() < min_train_rows or test_mask.sum() < 100:
            continue

        paths.append(PanelCPCVPath(path_id=path_id, test_groups=combo,
                                   train_mask=train_mask, test_mask=test_mask))
    return paths


# ─────────────────────────────────────────────
# Evaluation — per-symbol P&L, panel-level fitting
# ─────────────────────────────────────────────

def panel_simulate_pnl(panel: Panel, mask: np.ndarray, agent,
                       prob_threshold: float = 0.5) -> dict:
    """
    Aggregate main_gbt.simulate_pnl() across every symbol
    INDEPENDENTLY — not pooled into one single-open-position walk.
    Cross-sectional trading genuinely allows one open position PER
    SYMBOL at once (mirroring one UnifiedExecutor per symbol live), so
    each symbol's P&L is walked independently via the UNCHANGED
    main_gbt.simulate_pnl(), then summed/concatenated here.

    `mask` must be expressed in PANEL row-index space (e.g. a
    PanelCPCVPath's train_mask/test_mask) — sliced down to each
    symbol's own rows internally via Panel.symbol_row_mask().
    """
    labels = panel.as_labels_dict()
    total_pnl = 0.0
    n_trades = 0
    all_returns: list = []
    for sym in panel.symbols:
        sym_mask = mask & panel.symbol_row_mask(sym)
        if not sym_mask.any():
            continue
        pnl, trades, _avg, returns = gbt.simulate_pnl(
            panel.states, labels, sym_mask, agent, prob_threshold=prob_threshold)
        total_pnl += pnl
        n_trades += trades
        all_returns.extend(returns)

    avg_pnl = float(np.mean(all_returns)) if all_returns else 0.0
    return {"total_pnl": total_pnl, "n_trades": n_trades,
            "avg_pnl": avg_pnl, "trade_returns": all_returns}


def panel_evaluate_paths_at_threshold(panel: Panel, fitted_paths: list,
                                      threshold: float) -> list:
    """Re-score every fitted panel CPCV path at `threshold` (no
    refitting) — panel analogue of main_gbt.evaluate_paths_at_threshold()."""
    results = []
    for path, agent in fitted_paths:
        train_res = panel_simulate_pnl(panel, path.train_mask, agent, prob_threshold=threshold)
        test_res = panel_simulate_pnl(panel, path.test_mask, agent, prob_threshold=threshold)
        test_sharpe = gbt._sharpe_like(test_res["trade_returns"])
        results.append({
            "path_id": path.path_id, "test_groups": list(path.test_groups),
            "train_pnl": train_res["total_pnl"], "test_pnl": test_res["total_pnl"],
            "train_avg_pnl": train_res["avg_pnl"], "test_avg_pnl": test_res["avg_pnl"],
            "test_sharpe": test_sharpe,
            "n_train": int(path.train_mask.sum()), "n_test": int(path.test_mask.sum()),
            "n_train_trades": train_res["n_trades"], "n_test_trades": test_res["n_trades"],
        })
    return results


def panel_extract_recent_regime_result(panel: Panel, fitted_paths: list,
                                       threshold: float) -> dict:
    """
    GATE-HARDENING PARITY (this revision) — panel analogue of
    main_gbt._extract_recent_regime_result().

    Pulls the panel CPCV path whose test_groups equal
    gbt.RECENT_REGIME_GROUPS (the last N_TEST_GROUPS groups on the
    SHARED calendar-group axis — see generate_panel_cpcv_paths()'s
    docstring on why the same constant applies unmodified here) out of
    an already-fitted seed's `fitted_paths`, and scores it at
    `threshold` via panel_simulate_pnl() — no additional model
    fitting, mirroring how main_gbt.py reuses an already-fitted CPCV
    path for this check at zero extra cost.

    This is the piece that answers a question confirm_gate()'s core
    checks (directional consistency / downside floor / PBO / Sharpe,
    all pooled across every calendar window) cannot: is the panel's
    apparent edge concentrated in the MOST RECENT shared regime, or
    does it hold up there specifically — the calendar window closest,
    among CPCV's paths, to the real final holdout.

    Returns None if that exact test-group combination wasn't generated
    for this panel/seed (e.g. a short shared timestamp grid, or the
    combo happened to be pruned by generate_panel_cpcv_paths()'s
    min_train_rows/min-test-rows checks) — callers treat that as "no
    recent-regime signal available for this seed", exactly as
    main_gbt.py's confirm_gate_nested() does.
    """
    for path, agent in fitted_paths:
        if tuple(path.test_groups) == gbt.RECENT_REGIME_GROUPS:
            res = panel_simulate_pnl(panel, path.test_mask, agent, prob_threshold=threshold)
            return {
                "test_groups": list(gbt.RECENT_REGIME_GROUPS),
                "n_test_trades": res["n_trades"],
                "test_avg_pnl": res["avg_pnl"],
                "test_sharpe": gbt._sharpe_like(res["trade_returns"]),
            }
    return None


def summarize_recurring_exclusions(per_seed_excluded: dict, n_seeds: int,
                                   min_recur_frac: float = 0.5) -> list:
    """
    GATE-HARDENING PARITY (this revision) — diagnostic only, no gating
    effect.

    Given {seed: [excluded path_result, ...]} — the "excluded_paths"
    list evaluate_gate() already produces per seed (test_groups combos
    with < gbt.MIN_PATH_TEST_TRADES test trades that seed) — tallies
    which test_groups combinations recur as excluded across MULTIPLE
    seeds.

    Why this matters: random_state only ever reaches model-fitting
    randomness (max_features per-split subsampling, bagging bootstrap
    resampling — see gbt_agent.py). It NEVER changes which rows exist
    in a given path's train/test mask — that's determined purely by
    generate_panel_cpcv_paths()'s row counts, which are seed-
    independent. So a test_groups combo excluded for exactly ONE seed
    is unremarkable (that seed's fitted model happened to trade
    little in that window), but a combo excluded in
    `min_recur_frac` or more of ALL seeds means every seed's model —
    regardless of its own randomness — found too few resolvable
    trades in that calendar window. That points at a structural data-
    coverage gap (most likely align_panel()'s strict inner-join
    dropping rows where even one symbol has a gap — see align_panel()'s
    docstring), not per-seed noise, and is worth surfacing distinctly
    so a low pooled trade count isn't misread as "the strategy rarely
    trades" when it may really mean "the panel's calendar coverage is
    thin in specific windows".

    Returns a list of {test_groups, n_seeds_excluded, frac_seeds_excluded}
    dicts for combos at or above `min_recur_frac`, sorted by recurrence
    (most-recurring first).
    """
    tally = {}
    for seed, excluded in per_seed_excluded.items():
        seen_this_seed = set()
        for r in excluded:
            key = tuple(r["test_groups"])
            if key in seen_this_seed:
                continue
            seen_this_seed.add(key)
            tally[key] = tally.get(key, 0) + 1

    recurring = []
    for key, count in tally.items():
        frac = count / n_seeds if n_seeds else float("nan")
        if not np.isnan(frac) and frac >= min_recur_frac:
            recurring.append({
                "test_groups": list(key), "n_seeds_excluded": count,
                "n_seeds_total": n_seeds, "frac_seeds_excluded": frac,
            })
    recurring.sort(key=lambda r: r["frac_seeds_excluded"], reverse=True)
    return recurring


def panel_sweep_entry_thresholds(panel: Panel, fitted_paths: list,
                                 verbose: bool = True) -> dict:
    """Panel analogue of main_gbt.sweep_entry_thresholds() — re-scores
    every already-fitted panel CPCV (path, agent) at each candidate
    entry_threshold and picks the best OUT-OF-SAMPLE (test folds only)."""
    if verbose:
        print(f"\n  Sweeping entry_threshold over {gbt.ENTRY_THRESHOLD_CANDIDATES} "
              f"across {len(fitted_paths)} fitted panel CPCV paths (test folds only)...")
    candidates = {}
    for t in gbt.ENTRY_THRESHOLD_CANDIDATES:
        avg_pnls, trade_counts = [], []
        for path, agent in fitted_paths:
            res = panel_simulate_pnl(panel, path.test_mask, agent, prob_threshold=t)
            avg_pnls.append(res["avg_pnl"])
            trade_counts.append(res["n_trades"])

        avg_pnls = np.array(avg_pnls, dtype=np.float64)
        total_trades = int(sum(trade_counts))
        mean_avg = float(avg_pnls.mean()) if len(avg_pnls) else float("nan")
        std_avg = float(avg_pnls.std()) if len(avg_pnls) else float("nan")
        eligible = total_trades >= gbt.MIN_SWEEP_TRADES
        score = (mean_avg / (std_avg + 1e-6)) if eligible else float("-inf")

        candidates[t] = {"mean_test_avg_pnl": mean_avg, "std_test_avg_pnl": std_avg,
                         "total_test_trades": total_trades, "eligible": eligible,
                         "score": score}
        if verbose:
            elig_str = "" if eligible else "  [DISQUALIFIED: too few trades]"
            print(f"    threshold={t:.2f}  mean_test_avg={mean_avg:+.4%}  "
                  f"std={std_avg:.4%}  total_test_trades={total_trades}  "
                  f"score={score:+.3f}{elig_str}")

    best_t = max(candidates, key=lambda k: candidates[k]["score"])
    n_disqualified = sum(1 for c in candidates.values() if not c["eligible"])
    if candidates[best_t]["score"] == float("-inf"):
        if verbose:
            print(f"  ⚠ No threshold cleared MIN_SWEEP_TRADES={gbt.MIN_SWEEP_TRADES} "
                  f"— falling back to threshold=0.50.")
        best_t = 0.50
    elif verbose and n_disqualified > 0:
        # Surfaces the "degenerate threshold sweep" issue directly: if
        # the higher-conviction candidates can't even be TESTED for
        # lack of trades, entry_threshold selection is effectively
        # constrained to whichever low bars still clear MIN_SWEEP_TRADES
        # — the model isn't producing enough high-conviction
        # predictions to exercise its own conviction-gating mechanism.
        print(f"  ⚠ {n_disqualified}/{len(candidates)} candidate threshold(s) "
              f"disqualified (< {gbt.MIN_SWEEP_TRADES} total test trades) — "
              f"threshold selection is constrained to the surviving, looser "
              f"candidates only; this pipeline cannot currently verify "
              f"whether a stricter entry_threshold would help.")
    if verbose:
        print(f"  ✓ Selected entry_threshold={best_t:.2f}  "
              f"(score={candidates[best_t]['score']:+.3f}, "
              f"total_test_trades={candidates[best_t]['total_test_trades']})")

    return {"chosen_threshold": best_t, "candidates": candidates}


def run_panel_cpcv(panel: Panel, random_state: int = 0, model_type: str = "hgb",
                   hyperparams: dict = None, n_bagged_fits: int = None,
                   n_groups: int = None, n_test_groups: int = None,
                   max_paths: int = None, min_train_rows: int = None) -> tuple:
    """
    Fit a fresh GBTAgent per panel CPCV path's TRAIN split (pooled
    across every symbol), evaluate on its TEST split. Structurally
    mirrors main_gbt.run_cpcv() — returns (path_results, fitted_paths)
    in the same shape, so downstream gate/threshold-sweep code that
    already works on main_gbt.run_cpcv()'s output needs only the
    panel-aware evaluation helpers above (panel_simulate_pnl etc.)
    instead of main_gbt.simulate_pnl() directly.
    """
    hp = gbt.GBT_HYPERPARAMS if hyperparams is None else hyperparams
    nb = gbt.DEFAULT_N_BAGGED_FITS if n_bagged_fits is None else n_bagged_fits

    lookback_ticks = compute_required_lookback_ticks()
    paths = generate_panel_cpcv_paths(
        panel, n_groups=n_groups, n_test_groups=n_test_groups,
        lookback_ticks=lookback_ticks, min_train_rows=min_train_rows,
        max_paths=max_paths,
    )
    print(f"  Generated {len(paths)} panel CPCV paths  "
          f"(lookback_ticks={lookback_ticks} shared calendar ticks, "
          f"random_state={random_state}, model_type={model_type}, "
          f"symbols={panel.symbols})")

    path_results = []
    fitted_paths = []
    for path in paths:
        train_idx = np.flatnonzero(path.train_mask)
        test_idx = np.flatnonzero(path.test_mask)

        # Panel.states is stored SYMBOL-MAJOR (see module docstring's
        # row-order invariant) — resort the TRAIN slice into
        # chronological (timestamp, symbol) order before fitting.
        # GBTPolicy.fit()'s internal fit/calibration split and
        # _select_n_iterations()'s purged-validation slice both assume
        # X/y arrive in chronological order (see gbt_agent.py's fit()
        # docstring) — without this resort, "the last calibration_frac
        # of rows" would mean "the tail of the last symbol's block",
        # not "the most recent calendar rows across the whole panel".
        train_order = np.lexsort((panel.symbol_ids[train_idx], panel.timestamp_ids[train_idx]))
        train_idx = train_idx[train_order]

        X_train, y_train = panel.states[train_idx], panel.best_action[train_idx]
        if (len(np.unique(y_train)) < 2 or len(train_idx) < 200 or len(test_idx) < 100):
            continue

        # GBTPolicy.fit()'s purge_ticks parameter is a ROW count;
        # lookback_ticks above is in CALENDAR ticks (shared
        # timestamps). Convert using the actual rows-per-tick density
        # in THIS path's train slice (rather than assuming exactly
        # len(panel.symbols), since a few symbol/timestamp cells can
        # be missing from valid_mask near edges) so the purge buffer
        # inside fit() covers the same calendar span the panel-level
        # CPCV purge/embargo already enforces.
        n_unique_train_ts = len(np.unique(panel.timestamp_ids[train_idx]))
        rows_per_tick = max(1, round(len(train_idx) / max(1, n_unique_train_ts)))
        purge_rows = lookback_ticks * rows_per_tick

        agent = gbt.GBTAgent(state_dim=panel.state_dim, action_dim=gbt.ACTION_DIM,
                             model_type=model_type)
        agent.fit(X_train, y_train, purge_ticks=purge_rows, random_state=random_state,
                  n_bagged_fits=nb, **hp)
        fitted_paths.append((path, agent))

        train_res = panel_simulate_pnl(panel, path.train_mask, agent)
        test_res = panel_simulate_pnl(panel, path.test_mask, agent)
        test_sharpe = gbt._sharpe_like(test_res["trade_returns"])

        path_results.append({
            "path_id": path.path_id, "test_groups": list(path.test_groups),
            "train_pnl": train_res["total_pnl"], "test_pnl": test_res["total_pnl"],
            "train_avg_pnl": train_res["avg_pnl"], "test_avg_pnl": test_res["avg_pnl"],
            "test_sharpe": test_sharpe,
            "n_train": int(path.train_mask.sum()), "n_test": int(path.test_mask.sum()),
            "n_train_trades": train_res["n_trades"], "n_test_trades": test_res["n_trades"],
        })
        sharpe_str = f"{test_sharpe:+.2f}" if not np.isnan(test_sharpe) else "n/a"
        print(f"  path {path.path_id:>3}  test_groups={path.test_groups}  "
              f"train_avg={train_res['avg_pnl']:+.3%} ({train_res['n_trades']} trades)  "
              f"test_avg={test_res['avg_pnl']:+.3%} ({test_res['n_trades']} trades)  "
              f"test_sharpe={sharpe_str}  [panel: {len(panel.symbols)} symbols, "
              f"seed={random_state}]")

    return path_results, fitted_paths


# ─────────────────────────────────────────────
# Logistic-regression capacity-floor baseline (GATE-HARDENING PARITY)
# ─────────────────────────────────────────────

def run_panel_logistic_baseline(panel: Panel, seeds: tuple = None,
                                n_bagged_fits: int = None,
                                n_groups: int = None, n_test_groups: int = None,
                                max_paths: int = None, min_train_rows: int = None,
                                out_dir: str = None) -> dict:
    """
    GATE-HARDENING PARITY (this revision) — panel analogue of
    main_gbt.run_logistic_baseline().

    Fits GBTAgent(model_type="logistic") — a near-minimal-capacity
    baseline, see gbt_agent.py's module docstring — through the
    IDENTICAL panel CPCV path generation / purge-embargo / triple-
    barrier labels the panel GBT pipeline uses, at the default
    entry_threshold (0.5, panel_simulate_pnl()'s own default), pooled
    across `seeds` (defaults to gbt.DEFAULT_CONFIRMATION_SEEDS so it's
    directly comparable to the panel GBT's own confirmation-pool
    numbers).

    Existing without this, main_cross_sectional_gbt.py's confirm_gate()
    could report the GBT "passing" its own stability checks with no
    way to tell whether that stability reflects a GBT-specific edge or
    just the same marginal signal a near-linear model finds — exactly
    the ambiguity main_gbt.py's MIN_EDGE_OVER_BASELINE gate exists to
    resolve for the single-symbol pipeline.

    Returns the pooled gbt.evaluate_gate() stats dict (with "per_seed",
    "n_pass", "frac_pass" keys added) and writes
    panel_logistic_baseline.json to `out_dir`, mirroring
    main_gbt.run_logistic_baseline()'s output shape so
    main_cross_sectional_gbt.py's baseline-edge gate can consume it
    identically to the single-symbol pipeline.
    """
    out_dir = out_dir or os.path.join(gbt.OUT_DIR, "cross_sectional")
    os.makedirs(out_dir, exist_ok=True)
    seeds = gbt.DEFAULT_CONFIRMATION_SEEDS if seeds is None else seeds

    print(f"\n{'#'*62}\n  PANEL LOGISTIC REGRESSION BASELINE (capacity floor check)  "
          f"seeds={list(seeds)}\n{'#'*62}")

    per_seed = []
    pooled_results = []
    for seed in seeds:
        print(f"\n{'-'*62}\n  [panel baseline] seed {seed}\n{'-'*62}")
        path_results, _ = run_panel_cpcv(
            panel, random_state=seed, model_type="logistic", hyperparams={},
            n_bagged_fits=n_bagged_fits, n_groups=n_groups, n_test_groups=n_test_groups,
            max_paths=max_paths, min_train_rows=min_train_rows,
        )
        pooled_results.extend(path_results)

        seed_passed, seed_stats = gbt.evaluate_gate(path_results)
        per_seed.append({
            "seed": seed, "passed": bool(seed_passed),
            "n_total": seed_stats["n_total"], "n_pos": seed_stats.get("n_pos", 0),
            "mean_test_avg_pnl": seed_stats["mean_test_avg_pnl"],
            "min_test_avg_pnl": seed_stats["min_test_avg_pnl"],
            "pbo": seed_stats["pbo"],
        })
        status = "PASS" if seed_passed else "FAIL"
        pbo_seed_str = f"{seed_stats['pbo']:.1%}" if not np.isnan(seed_stats["pbo"]) else "n/a"
        print(f"  [panel baseline] seed {seed}: {status}  "
              f"mean_test_avg={seed_stats['mean_test_avg_pnl']:+.4%}  "
              f"positive={seed_stats.get('n_pos', 0)}/{seed_stats['n_total']}  "
              f"PBO={pbo_seed_str}")

    passed, stats = gbt.evaluate_gate(pooled_results)
    n_pass = sum(r["passed"] for r in per_seed)
    frac_pass = n_pass / len(seeds) if seeds else float("nan")

    print(f"\n{'='*62}\n  PANEL LOGISTIC BASELINE VERDICT  "
          f"({stats['n_total']} reliable paths)\n{'='*62}")
    print(f"  Per-seed stability  : {n_pass}/{len(seeds)} seeds passed ({frac_pass:.0%}) "
          f"— compare directly against the panel GBT's own per-seed confirmation "
          f"result.")
    print(f"  Mean test avg/trade : {stats['mean_test_avg_pnl']:+.4%}")
    print(f"  Std  test avg/trade : {stats['std_test_avg_pnl']:.4%}")
    print(f"  Min  test avg/trade : {stats['min_test_avg_pnl']:+.4%}")
    print(f"  Positive paths      : {stats.get('n_pos', 0)}/{stats['n_total']}")
    pbo_str = f"{stats['pbo']:.1%}" if not np.isnan(stats["pbo"]) else "n/a"
    print(f"  PBO                 : {pbo_str}")

    if not np.isnan(stats["pbo"]) and stats["pbo"] > gbt.PBO_MAX_ALLOWED:
        print(f"\n  → Baseline ALSO fails PBO<={gbt.PBO_MAX_ALLOWED:.0%} at near-minimal "
              f"model capacity, pooled across the whole panel. This supports a "
              f"label/regime explanation for the panel GBT's instability over a "
              f"pure capacity/overfitting explanation.")
    elif np.isnan(stats["pbo"]):
        print(f"\n  → Not enough reliable logistic paths to compute a PBO verdict "
              f"— treat as inconclusive, not a pass.")
    else:
        print(f"\n  → Baseline PASSES PBO<={gbt.PBO_MAX_ALLOWED:.0%} at near-minimal "
              f"capacity. Whether this is good news for the panel GBT now ALSO "
              f"depends on the baseline-edge gate in main_cross_sectional_gbt.py's "
              f"main(): a GBT that doesn't outperform this near-linear model by "
              f"at least {gbt.MIN_EDGE_OVER_BASELINE:.2%} avg/trade is treated as "
              f"having found the same marginal signal, not a GBT-specific edge.")

    with open(os.path.join(out_dir, "panel_logistic_baseline.json"), "w") as f:
        json.dump({
            "seeds": list(seeds),
            "n_pass": n_pass, "frac_pass": frac_pass,
            "per_seed": per_seed,
            "mean_test_avg_pnl": stats["mean_test_avg_pnl"],
            "std_test_avg_pnl": stats["std_test_avg_pnl"],
            "min_test_avg_pnl": stats["min_test_avg_pnl"],
            "n_total": stats["n_total"], "n_pos": stats.get("n_pos", 0),
            "pbo": stats["pbo"], "logit_lambda": stats["logit_lambda"],
        }, f, indent=2, default=gbt._json_default)
    print(f"\n  ✓  Panel logistic baseline saved → {out_dir}/panel_logistic_baseline.json")

    stats["per_seed"] = per_seed
    stats["n_pass"] = n_pass
    stats["frac_pass"] = frac_pass
    return stats


# ─────────────────────────────────────────────
# Final deployable training
# ─────────────────────────────────────────────

def run_panel_final_training(panel: Panel, entry_threshold: float = 0.5,
                             random_state: int = 0, n_bagged_fits: int = None,
                             holdout_frac: float = None, out_dir: str = None):
    """
    Panel analogue of main_gbt.run_final_training(). Holdout is the
    chronologically LAST holdout_frac of the SHARED timestamp grid,
    applied to every symbol simultaneously (not per symbol) — matches
    generate_panel_cpcv_paths()'s calendar-level purge/embargo logic.
    """
    out_dir = out_dir or os.path.join(gbt.OUT_DIR, "cross_sectional")
    os.makedirs(out_dir, exist_ok=True)
    holdout_frac = gbt.DEFAULT_HOLDOUT_FRAC if holdout_frac is None else holdout_frac

    T = len(panel.timestamps)
    lookback_ticks = compute_required_lookback_ticks()
    holdout_start = int(T * (1 - holdout_frac))
    embargo_start = max(0, holdout_start - lookback_ticks)

    train_ts_mask = np.zeros(T, dtype=bool)
    train_ts_mask[:embargo_start] = True
    holdout_ts_mask = np.zeros(T, dtype=bool)
    holdout_ts_mask[holdout_start:] = True

    train_mask = train_ts_mask[panel.timestamp_ids] & panel.valid_mask
    holdout_mask = holdout_ts_mask[panel.timestamp_ids] & panel.valid_mask

    train_idx = np.flatnonzero(train_mask)
    order = np.lexsort((panel.symbol_ids[train_idx], panel.timestamp_ids[train_idx]))
    train_idx = train_idx[order]

    n_unique_train_ts = len(np.unique(panel.timestamp_ids[train_idx]))
    rows_per_tick = max(1, round(len(train_idx) / max(1, n_unique_train_ts)))
    purge_rows = lookback_ticks * rows_per_tick

    nb = gbt.DEFAULT_N_BAGGED_FITS if n_bagged_fits is None else n_bagged_fits
    print(f"\n  Panel final train: {len(train_idx):,} rows across "
          f"{len(panel.symbols)} symbols  |  Holdout: {int(holdout_mask.sum()):,} rows  |  "
          f"embargo: {holdout_start - embargo_start} shared ticks  |  "
          f"entry_threshold={entry_threshold:.2f}  random_state={random_state}")

    agent = gbt.GBTAgent(state_dim=panel.state_dim, action_dim=gbt.ACTION_DIM,
                         entry_threshold=entry_threshold)
    agent.fit(panel.states[train_idx], panel.best_action[train_idx],
              purge_ticks=purge_rows, random_state=random_state,
              n_bagged_fits=nb, **gbt.GBT_HYPERPARAMS)

    holdout_res = panel_simulate_pnl(panel, holdout_mask, agent, prob_threshold=entry_threshold)
    print(f"\n  Held-out panel P/L: {holdout_res['total_pnl']:+.4%} raw sum  "
          f"({holdout_res['n_trades']} trades across {len(panel.symbols)} symbols, "
          f"avg/trade={holdout_res['avg_pnl']:+.4%})  @ entry_threshold={entry_threshold:.2f}")

    ci = gbt.bootstrap_ci(holdout_res["trade_returns"], ci=0.90, random_state=random_state)
    if not np.isnan(ci["ci_lo"]):
        straddles = ci["ci_lo"] < 0 < ci["ci_hi"]
        print(f"  Bootstrap 90% CI on avg/trade: [{ci['ci_lo']:+.4%}, {ci['ci_hi']:+.4%}]"
              + ("  (straddles zero — cannot statistically distinguish from "
                 "no edge)" if straddles else ""))
        if ci.get("note"):
            print(f"    ⚠ {ci['note']}")
    else:
        print(f"  Bootstrap CI unavailable — {ci.get('note', 'insufficient trades')}")

    deploy_ok = (holdout_res["n_trades"] >= gbt.HOLDOUT_MIN_TRADES
                 and holdout_res["avg_pnl"] > gbt.HOLDOUT_AVG_TRADE_FLOOR)

    gate_status = {
        "symbols": panel.symbols, "entry_threshold": entry_threshold,
        "random_state": random_state, "holdout_frac": holdout_frac,
        "holdout_avg_pnl": holdout_res["avg_pnl"],
        "holdout_total_pnl_raw": holdout_res["total_pnl"],
        "holdout_n_trades": holdout_res["n_trades"],
        "holdout_avg_trade_floor": gbt.HOLDOUT_AVG_TRADE_FLOOR,
        "holdout_min_trades": gbt.HOLDOUT_MIN_TRADES,
        "holdout_avg_pnl_bootstrap_ci": ci,
        "deploy_gate_passed": bool(deploy_ok),
    }
    with open(os.path.join(out_dir, "panel_deployment_gate.json"), "w") as f:
        json.dump(gate_status, f, indent=2, default=gbt._json_default)

    save_name = ("panel_gbt_agent_best.joblib" if deploy_ok
                else "panel_gbt_agent_candidate_FAILED_HOLDOUT.joblib")
    save_path = os.path.join(out_dir, save_name)
    agent.save(save_path)

    if deploy_ok:
        print(f"\n  ✓ HOLDOUT GATE PASSED (avg/trade={holdout_res['avg_pnl']:+.4%} > "
              f"floor={gbt.HOLDOUT_AVG_TRADE_FLOOR:+.2%}, n_trades={holdout_res['n_trades']}) "
              f"— deployed → {save_path}")
    else:
        print(f"\n  ⛔ HOLDOUT GATE FAILED — avg/trade={holdout_res['avg_pnl']:+.4%} "
              f"(floor={gbt.HOLDOUT_AVG_TRADE_FLOOR:+.2%}, "
              f"n_trades={holdout_res['n_trades']} vs min={gbt.HOLDOUT_MIN_TRADES}). "
              f"Saved as CANDIDATE ONLY → {save_path}.")

    return agent, gate_status