"""
triple_barrier.py
─────────────────────────
Label generation via the triple-barrier method (Lopez de Prado,
"Advances in Financial Machine Learning", ch.3), used to replace the
MC-kNN pipeline's policy-entangled Monte Carlo return labels
(episode_buffer.py's compute_mc_returns()) with a label that depends
ONLY on the forward price path and a fixed rule — not on what the
trading policy happened to do afterward (exit timing, epsilon-greedy
noise, HOLD duration, etc).

Why this matters
──────────────────
episode_buffer.py's G_t = r_t + gamma*r_{t+1} + ... is backfilled from
the ACTUAL trajectory the policy took. Two runs with different
epsilon-greedy draws, or a policy that changed its exit rule, produce
DIFFERENT labels for the same market state. That's not "was this a
good moment to go long" — it's "was this a good moment to go long,
GIVEN everything this specific policy instance did afterward". The
triple-barrier label below is computed once, from price data alone,
and is stable across any number of retrains/policy changes.

Barrier definition (per candidate entry tick i, given a side ∈ {+1,-1}):
  - upper (take-profit) barrier:  entry_price * (1 + side * tp_mult * vol_i)
  - lower (stop-loss)   barrier:  entry_price * (1 - side * sl_mult * vol_i)
  - vertical (time)     barrier:  tick i + max_holding
  vol_i is a per-tick volatility estimate (this project's ATR_Scaled,
  rescaled back to a fractional-return unit — see `atr_to_frac()`)
  so barrier width adapts to the regime instead of being one fixed
  percentage for both 2019's chop and a high-volatility 2024 spike.

Label:
   1  if the take-profit barrier is touched before the stop-loss AND
      before the vertical barrier ("favorable", net of commission).
   0  otherwise (stop-loss hit first, or vertical barrier reached
      without the trade being net-profitable).
Realised return at the touch point is also returned so labels can be
sanity-checked and so meta-labeling can rank LONG vs SHORT when both
resolve favorably from the same tick.
"""

import numpy as np


def atr_to_frac(atr_scaled: np.ndarray, atr_pct_std_est: float = 0.02) -> np.ndarray:
    """
    preprocessing.py's ATR_Scaled is a [-1, 1]-clipped z-score of
    ATR-as-%-of-price (see preprocess_indicators_data: atr_pct rolling
    mean/std, clipped to ±3 then divided by 3). It is NOT itself a
    fractional volatility — it's a standardized deviation from that
    rolling mean. For barrier sizing we want an actual fractional
    magnitude, so we re-expand: |ATR_Scaled| * 3 std devs back onto a
    rough ATR-%-of-price scale, using atr_pct_std_est as a stand-in
    for the rolling std this project's preprocessing already computes
    but doesn't expose per-row. This is intentionally an approximation
    — good enough to make barriers volatility-ADAPTIVE (wider in high
    vol, narrower in low vol) rather than perfectly calibrated; if you
    want exact ATR-in-%-terms, thread atr_pct itself through from
    preprocessing.py instead of reconstructing it from the scaled column.
    """
    # Undo the /3 scaling, sign is irrelevant (volatility is unsigned),
    # floor so barriers never collapse to zero width in dead-flat regimes.
    return np.maximum(np.abs(atr_scaled) * 3.0 * atr_pct_std_est, 0.003)


def triple_barrier_labels(
    prices: np.ndarray,
    vol: np.ndarray,
    side: int,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 32,
    commission: float = 0.00015,
) -> tuple:
    """
    Compute triple-barrier labels for every candidate entry tick, for a
    SINGLE fixed trade direction (`side` = +1 for LONG, -1 for SHORT).

    For each start tick i we scan forward at most max_holding ticks and
    stop at the first barrier touch. For max_holding=32 and n~17K rows
    this is comfortably fast in pure Python/numpy without needing
    numba/joblib parallelism.

    Commission is charged on both entry and exit (matching
    unified_executor.py's COMMISSION convention) so a label of 1 means
    "this trade would have been net profitable after costs", not just
    "price moved the right way".

    Parameters
    ----------
    prices       : (n,) float array of Close prices.
    vol          : (n,) float array of per-tick fractional volatility
                   (see atr_to_frac()) used to size both barriers.
    side         : +1 (evaluate LONG entries) or -1 (evaluate SHORT).
    tp_mult      : take-profit barrier width, in units of `vol`.
    sl_mult      : stop-loss barrier width, in units of `vol`.
    max_holding  : vertical (time) barrier, in ticks.
    commission   : per-side commission fraction, matching
                   unified_executor.py's COMMISSION.

    Returns
    -------
    labels       : (n,) int8 array. 1 = TP hit first (favorable, net of
                   commission), 0 = otherwise. Last max_holding rows are
                   marked -1 (undefined — insufficient forward data).
    touch_ticks  : (n,) int array, index (into `prices`) where the
                   episode resolved. -1 where undefined.
    net_returns  : (n,) float array, realised net return at the touch
                   point (signed so positive = profitable regardless of
                   side). NaN where undefined.
    """
    n = len(prices)
    labels      = np.full(n, -1, dtype=np.int8)
    touch_ticks = np.full(n, -1, dtype=np.int64)
    net_returns = np.full(n, np.nan, dtype=np.float64)

    entry_adj = (1 + side * commission)   # worse fill on entry
    exit_adj  = (1 - side * commission)   # matches unified_executor.py's
                                           # _execute/_close commission model

    last_valid_start = n - max_holding - 1
    for i in range(max(0, last_valid_start + 1)):
        entry_price = prices[i] * entry_adj
        v = vol[i]
        if side == 1:
            upper = entry_price * (1 + tp_mult * v)
            lower = entry_price * (1 - sl_mult * v)
        else:
            upper = entry_price * (1 - tp_mult * v)   # "upper" = favorable side for shorts
            lower = entry_price * (1 + sl_mult * v)

        touched = False
        for h in range(1, max_holding + 1):
            p = prices[i + h] * exit_adj
            if side == 1:
                if p >= upper:
                    labels[i] = 1
                    touch_ticks[i] = i + h
                    net_returns[i] = (p - entry_price) / entry_price
                    touched = True
                    break
                if p <= lower:
                    labels[i] = 0
                    touch_ticks[i] = i + h
                    net_returns[i] = (p - entry_price) / entry_price
                    touched = True
                    break
            else:
                if p <= upper:
                    labels[i] = 1
                    touch_ticks[i] = i + h
                    net_returns[i] = (entry_price - p) / entry_price
                    touched = True
                    break
                if p >= lower:
                    labels[i] = 0
                    touch_ticks[i] = i + h
                    net_returns[i] = (entry_price - p) / entry_price
                    touched = True
                    break

        if not touched:
            # Vertical barrier: resolve by sign of net return at
            # max_holding — matches unified_executor.py's MAX_HOLD_TICKS
            # forced-close convention (see step()'s hold_duration >=
            # MAX_HOLD_TICKS branch).
            p = prices[i + max_holding] * exit_adj
            ret = (p - entry_price) / entry_price if side == 1 \
                else (entry_price - p) / entry_price
            labels[i] = 1 if ret > 0 else 0
            touch_ticks[i] = i + max_holding
            net_returns[i] = ret

    return labels, touch_ticks, net_returns


def build_meta_labels(
    prices: np.ndarray,
    atr_scaled: np.ndarray,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 32,
    commission: float = 0.00015,
) -> dict:
    """
    Convenience wrapper: compute LONG and SHORT triple-barrier labels
    for every tick in one call, plus a combined 3-class target used as
    gbt_agent.py's training target.

    Combined target (`best_action`):
        0 = LONG favorable  (long_label==1 and, if both directions
            resolve favorably, long_return >= short_return)
        1 = SHORT favorable (short_label==1 and better than long)
        3 = HOLD            (neither direction resolves favorably)
    Action id 2 (CLOSE) is never a meta-label target — CLOSE is an
    executor-level position-management action, not an entry decision;
    see gbt_agent.py's GBTPolicy._in_position_probs() for how
    CLOSE/HOLD-while-in-position are handled at inference time instead.

    Returns a dict of parallel (n,) arrays: long_label, long_return,
    long_touch, short_label, short_return, short_touch, best_action,
    valid_mask (False for the last max_holding rows, where labels are
    undefined). long_touch/short_touch are the tick index (into the
    SAME 0-based array all these outputs share) where that direction's
    barrier resolved — required by main_gbt.py's simulate_pnl() to
    enforce non-overlapping single-position trade simulation instead
    of double-counting return over a barrier's still-open holding
    window (each label's outcome window can span up to max_holding
    ticks and adjacent ticks' windows overlap heavily — summing every
    tick's return independently, as if capital is unconstrained and
    trades don't overlap, silently inflates P/L by roughly
    max_holding-fold during any stretch of sustained conviction).
    """
    vol = atr_to_frac(atr_scaled)
    long_label,  long_touch,  long_return  = triple_barrier_labels(
        prices, vol, side=1, tp_mult=tp_mult, sl_mult=sl_mult,
        max_holding=max_holding, commission=commission,
    )
    short_label, short_touch, short_return = triple_barrier_labels(
        prices, vol, side=-1, tp_mult=tp_mult, sl_mult=sl_mult,
        max_holding=max_holding, commission=commission,
    )

    n = len(prices)
    best_action = np.full(n, 3, dtype=np.int8)  # default HOLD
    both_favorable = (long_label == 1) & (short_label == 1)
    long_only  = (long_label == 1) & (short_label != 1)
    short_only = (short_label == 1) & (long_label != 1)

    best_action[long_only]  = 0
    best_action[short_only] = 1
    # When both directions would have been favorable from the same tick
    # (can happen near max_holding in choppy regimes), prefer whichever
    # had the larger realised net return rather than an arbitrary
    # tie-break.
    prefer_long  = both_favorable & (long_return >= short_return)
    prefer_short = both_favorable & (long_return <  short_return)
    best_action[prefer_long]  = 0
    best_action[prefer_short] = 1

    valid_mask = np.ones(n, dtype=bool)
    valid_mask[n - max_holding:] = False

    return {
        "long_label": long_label, "long_return": long_return,
        "long_touch": long_touch,
        "short_label": short_label, "short_return": short_return,
        "short_touch": short_touch,
        "best_action": best_action, "valid_mask": valid_mask,
    }