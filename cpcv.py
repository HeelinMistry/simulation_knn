"""
cpcv.py
─────────────────
Combinatorial Purged Cross-Validation (Lopez de Prado, ch.12).

Why this supplements the linear 6-fold walk-forward CV in walkforward.py
────────────────────────────────────────────────────────────────────────
main_mcknn.py's walk-forward CV slides ONE val window forward through
the dataset per fold — 6 folds means 6 (train, val) pairs, each using a
different, non-overlapping slice of history as "val" exactly once. That
gives 6 independent P/L draws, which is already better than a single
split, but it's still a small sample, and a strategy that's genuinely
fine except for one bad regime (like fold 5, Dec-2024→Feb-2026, at
-96.8%) shows up as "1 catastrophic fold out of 6", which is hard to
distinguish statistically from "1 unlucky fold out of 6 for a fine
strategy". You need many more independent train/test combinations of
the SAME underlying data to see whether that result recurs across
different path constructions or was an artifact of exactly where the
walk-forward boundary fell.

CPCV addresses this by splitting the dataset into N contiguous groups,
then generating every C(N, k) combination of k groups as the test set
(the rest as train), purging/embargoing around each test group boundary
exactly as walkforward.py already does. With N=8, k=2 you get 28
independent (train, test) paths instead of 6 — each test set is a
different COMBINATION of 2 groups, not necessarily contiguous in time,
so you get a genuine sampling distribution of out-of-sample performance.
This also lets you estimate the Probability of Backtest Overfitting
(PBO, see compute_pbo() below) — a direct measure of whether a reported
"best" result is likely noise, which a simple std/mean check can't give.

This module reuses walkforward.py's compute_required_lookback_ticks()
so purge/embargo sizing stays identical between the two CV schemes.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from walkforward import compute_required_lookback_ticks


@dataclass
class CPCVPath:
    """One CPCV train/test combination."""
    path_id:     int
    test_groups: tuple      # which group indices form the test set
    train_mask:  np.ndarray
    test_mask:   np.ndarray


def _make_groups(n: int, n_groups: int) -> list:
    """Split [0, n) into n_groups contiguous, near-equal-size index ranges."""
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [(edges[g], edges[g + 1]) for g in range(n_groups)]


def generate_cpcv_paths(
    df: pd.DataFrame,
    n_groups: int = 8,
    n_test_groups: int = 2,
    lookback_ticks: int = None,
    min_train_ticks: int = 1000,
    max_paths: int = 40,
) -> list:
    """
    Generate CPCV train/test paths.

    Parameters
    ----------
    df              : primary-timeframe feature-engineered DataFrame
                       (only len(df) is actually used — this parameter
                       exists so callers can pass the same frame they
                       use elsewhere for readability).
    n_groups        : number of contiguous groups to split the dataset
                       into (N in Lopez de Prado's notation).
    n_test_groups   : how many groups form each test combination (k).
                       C(n_groups, n_test_groups) total combinations —
                       e.g. C(8,2)=28.
    lookback_ticks  : purge/embargo width; defaults to
                       walkforward.compute_required_lookback_ticks(),
                       identical sizing to the linear walk-forward CV.
    min_train_ticks : skip a path (with a warning) if purging leaves
                       too few train rows.
    max_paths       : hard cap on generated paths (C(n,k) grows fast —
                       e.g. C(10,3)=120). Combinations are taken in the
                       order itertools.combinations produces them, a
                       fixed, reproducible subset, not a random one.

    Returns
    -------
    list[CPCVPath]
    """
    n = len(df)
    if lookback_ticks is None:
        lookback_ticks = compute_required_lookback_ticks()

    groups = _make_groups(n, n_groups)
    all_combos = list(combinations(range(n_groups), n_test_groups))
    if len(all_combos) > max_paths:
        print(f"  ⚠ C({n_groups},{n_test_groups})={len(all_combos)} exceeds "
              f"max_paths={max_paths} — using the first {max_paths} "
              f"combinations (fixed, reproducible subset).")
        all_combos = all_combos[:max_paths]

    paths = []
    for path_id, combo in enumerate(all_combos):
        test_mask = np.zeros(n, dtype=bool)
        for g in combo:
            lo, hi = groups[g]
            test_mask[lo:hi] = True

        # Purge/embargo: for every contiguous test block, remove a
        # lookback_ticks buffer from train on both sides of that
        # block's boundary — same rationale as walkforward.py's single-
        # window purge/embargo, applied per contiguous run since CPCV's
        # test set can be made of several disjoint groups.
        exclude_mask = test_mask.copy()
        diffs = np.diff(test_mask.astype(int))
        starts = np.flatnonzero(diffs == 1) + 1
        ends   = np.flatnonzero(diffs == -1) + 1
        if test_mask[0]:
            starts = np.r_[0, starts]
        if test_mask[-1]:
            ends = np.r_[ends, n]
        for s, e in zip(starts, ends):
            lo = max(0, s - lookback_ticks)
            hi = min(n, e + lookback_ticks)
            exclude_mask[lo:hi] = True

        train_mask = ~exclude_mask
        n_train = int(train_mask.sum())
        if n_train < min_train_ticks:
            print(f"  ⚠ CPCV path {path_id} (test groups {combo}): only "
                  f"{n_train} train rows after purge/embargo — skipping.")
            continue

        paths.append(CPCVPath(
            path_id=path_id, test_groups=combo,
            train_mask=train_mask, test_mask=test_mask,
        ))

    return paths


def compute_pbo(path_results: list) -> dict:
    """
    Probability of Backtest Overfitting, following the CPCV logit
    procedure (Bailey, Borwein, Lopez de Prado & Zhu 2016), simplified
    to this project's single-strategy-configuration case (no grid of
    competing configurations — every path here evaluates the SAME
    fitting procedure, so this measures "how often does the train-set
    ranking fail to predict the test-set ranking", the relevant
    question when you have one strategy, not a search over many).

    Parameters
    ----------
    path_results : list of dicts, each with at least
                    {"path_id", "train_pnl", "test_pnl"} from running
                    the SAME model-fitting procedure on each CPCV
                    path's train split and evaluating on its test split.

    Returns
    -------
    dict with:
      pbo             : fraction of "in-sample-good" paths (train_pnl
                         above median) whose test_pnl fell at or below
                         the test-set median — i.e. how often looking
                         good in-sample failed to predict looking good
                         out-of-sample. >50% means in-sample selection
                         is worse than a coin flip.
      logit_lambda    : average logit of the out-of-sample percentile
                         rank — large negative values indicate
                         systematic overfitting.
      n_paths         : number of paths used.
    """
    if len(path_results) < 4:
        return {"pbo": float("nan"), "logit_lambda": float("nan"),
                "n_paths": len(path_results),
                "note": "too few CPCV paths for a meaningful PBO estimate"}

    train_pnls = np.array([r["train_pnl"] for r in path_results])
    test_pnls  = np.array([r["test_pnl"]  for r in path_results])

    train_median = np.median(train_pnls)
    test_median  = np.median(test_pnls)

    above_train_median = train_pnls > train_median
    below_test_median  = test_pnls <= test_median
    if above_train_median.sum() > 0:
        pbo = float((above_train_median & below_test_median).sum() /
                     above_train_median.sum())
    else:
        pbo = float("nan")

    ranks = pd.Series(test_pnls).rank(pct=True).values
    ranks = np.clip(ranks, 1e-3, 1 - 1e-3)
    logit_lambda = float(np.mean(np.log(ranks / (1 - ranks))))

    return {"pbo": pbo, "logit_lambda": logit_lambda,
            "n_paths": len(path_results)}