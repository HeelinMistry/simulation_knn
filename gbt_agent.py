"""
gbt_agent.py
──────────────────
Calibrated gradient-boosted-tree agent — a batch-trained, probability-
calibrated replacement for MCKNNAgent, using the triple-barrier labels
from triple_barrier.py instead of policy-entangled Monte Carlo returns.

Why this instead of the k-NN vote-share
──────────────────────────────────────────
1. mc_knn_memory.py's "probs" are inverse-distance vote SHARES, never
   checked against how often those states actually resolved favorably
   (mc_knn_policy.py's own docstring: "NOT a calibrated softmax
   probability... should be read as how lopsided the neighbor vote
   was, not P(action|state)"). This file's probabilities ARE checked —
   see calibration_report.py — and are produced by isotonic regression
   fit against held-out outcome frequencies, so "70% confidence" is
   meant to (and is verified to, on the calibration split) mean "wins
   ~70% of the time".
2. Gradient-boosted trees handle the curse-of-dimensionality problem
   pre_training.py's diagnostic found in raw 122-dim Euclidean distance
   (NN/overall ratio got WORSE with added context, 0.18->0.52) far
   better than k-NN — tree splits use one feature at a time and are
   naturally robust to irrelevant/noisy dimensions, no block_weights/
   dim_scale hand-tuning required.
3. Labels come from triple_barrier.py, which depends only on the
   forward price path — not on this-or-any policy's exit timing, so
   retraining the classifier or changing epsilon schedules can never
   change what a given historical state's label should have been.

Interface parity with MCKNNAgent
────────────────────────────────────
select_action(state, deterministic, action_mask, ...) mirrors
MCKNNAgent.select_action's call shape. Deliberately does NOT declare
query_tick/query_episode_id parameters, so unified_executor.py's
_agent_supports_temporal_args() duck-typing correctly detects this
agent has no temporal-exclusion concept (GBT is never queried against
its own training rows the way a k-NN bank is) and skips those kwargs
automatically — no changes needed to unified_executor.py.

Batch- vs online-trained
────────────────────────────
Unlike MCKNNAgent (which grows its bank tick-by-tick via
agent.update(episode_buffer) every epoch), GBTAgent.fit(X, y) is called
ONCE on a full (state, best_action) dataset built upstream (see
main_gbt.py) — there is no per-episode incremental update. update()
is kept as a no-op for interface parity with call sites that might
still invoke it out of habit.
"""

import os
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import joblib

ACTION_NAMES = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}


def _supports_prefit() -> bool:
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    return (major, minor) < (1, 6)   # cv="prefit" deprecated/removed in 1.6+


class _ManualIsotoneCalibrator:
    """
    Fallback calibrator for sklearn>=1.6, where CalibratedClassifierCV's
    cv='prefit' path was removed. Fits one isotonic regressor per class
    (one-vs-rest) directly against the pre-fitted base estimator's raw
    probabilities on the calibration split — equivalent in spirit to
    what cv='prefit' used to do internally.
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.classes_ = base_estimator.classes_
        self._calibrators = {}

    def fit(self, X_cal, y_cal):
        raw = self.base_estimator.predict_proba(X_cal)
        for j, cls in enumerate(self.classes_):
            y_bin = (y_cal == cls).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw[:, j], y_bin)
            self._calibrators[cls] = iso
        return self

    def predict_proba(self, X):
        raw = self.base_estimator.predict_proba(X)
        out = np.zeros_like(raw)
        for j, cls in enumerate(self.classes_):
            out[:, j] = self._calibrators[cls].predict(raw[:, j])
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums


class GBTPolicy:
    """
    Drop-in replacement for MCKNNPolicy's act()/get_action() surface,
    backed by a calibrated HistGradientBoostingClassifier trained on
    3-class triple-barrier targets (LONG-favorable / SHORT-favorable /
    HOLD — see triple_barrier.build_meta_labels).

    CLOSE (action 2) is never predicted directly by this classifier —
    it's an executor-level position-management decision
    (unified_executor.py's step() already forces CLOSE on stop-loss/
    max-hold breach independently of the policy). When action_mask
    marks CLOSE as the only non-HOLD legal action (currently in a
    position), the CALLER (GBTAgent.select_action) routes to
    _in_position_probs() instead of the flat-position 4-class mapping
    below — see that method's docstring for why this needs the
    currently-open side, which the policy itself doesn't track.
    """

    def __init__(self, model=None, action_dim: int = 4):
        self.model = model
        self.action_dim = action_dim
        self.classes_ = None  # set on fit(); subset of {0,1,3}

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            calibration_frac: float = 0.2,
            random_state: int = 0,
            **hgb_kwargs):
        """
        Fit the base classifier on a (1 - calibration_frac) split, then
        calibrate on the held-out calibration_frac split via isotonic
        regression. The split is done manually (rather than via
        CalibratedClassifierCV's built-in cv folds) so the calibration
        data is VISIBLY separate from the fitting data, and so this
        works identically across sklearn versions where cv='prefit'
        was removed (see _supports_prefit()).

        IMPORTANT: X/y here should already be the TRAIN-fold data from
        an outer CPCV/walk-forward split (see main_gbt.py) — this
        train/calibration split is a further internal split of that,
        purely for calibration, not a substitute for proper purged
        out-of-sample evaluation.
        """
        strat = y if len(np.unique(y)) > 1 else None
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=calibration_frac, random_state=random_state,
            stratify=strat,
        )

        base = HistGradientBoostingClassifier(
            max_iter=hgb_kwargs.pop("max_iter", 300),
            learning_rate=hgb_kwargs.pop("learning_rate", 0.05),
            max_depth=hgb_kwargs.pop("max_depth", 6),
            l2_regularization=hgb_kwargs.pop("l2_regularization", 1.0),
            early_stopping=True,
            random_state=random_state,
            **hgb_kwargs,
        )
        base.fit(X_fit, y_fit)

        if _supports_prefit():
            self.model = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        else:
            self.model = _ManualIsotoneCalibrator(base)
        self.model.fit(X_cal, y_cal)

        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def _raw_probs(self, state: np.ndarray) -> np.ndarray:
        """Return a length-3 (LONG, SHORT, HOLD) calibrated probability
        vector for a single state, with 0 for any class absent from
        training data (e.g. SHORT never favorable in the training
        split)."""
        p = self.model.predict_proba(state.reshape(1, -1))[0]
        full = np.zeros(3, dtype=np.float32)   # slot order: LONG=0, SHORT=1, HOLD=2
        slot = {0: 0, 1: 1, 3: 2}
        for cls, prob in zip(self.model.classes_, p):
            full[slot[int(cls)]] = prob
        return full  # [P(LONG), P(SHORT), P(HOLD)]

    def _in_position_probs(self, long_short_hold: np.ndarray, current_side: str) -> np.ndarray:
        """
        Convert the 3-class (LONG, SHORT, HOLD) output into a 4-class
        (LONG, SHORT, CLOSE, HOLD) vote when the executor is currently
        in a position (action_mask only allows CLOSE/HOLD). We treat
        "the model no longer favors the currently-open side" as the
        CLOSE signal: P(stay) = P(currently-open side) + half of
        P(HOLD)'s mass (a state the model is broadly unsure about
        shouldn't get force-closed just because P(open_side) alone is
        modest), P(CLOSE) = 1 - P(stay). This reuses the SAME
        calibrated probability the model would use to decide whether
        to newly open that side, rather than a separate uncalibrated
        heuristic.
        """
        p_long, p_short, p_hold = long_short_hold
        p_open_side = p_long if current_side == "LONG" else p_short
        p_stay = float(np.clip(p_open_side + 0.5 * p_hold, 0.0, 1.0))
        probs = np.zeros(4, dtype=np.float32)
        probs[3] = p_stay          # HOLD
        probs[2] = 1.0 - p_stay    # CLOSE
        return probs

    def get_action(self, state: np.ndarray, action_mask=None, **_ignored):
        """
        Same call shape as MCKNNPolicy.get_action for a FLAT-position
        decision (LONG/SHORT/HOLD). Extra kwargs (k, query_tick,
        query_episode_id, min_tick_gap) are accepted and ignored for
        interface parity — see module docstring on why
        unified_executor.py never actually passes them to this class.

        NOTE: this method does not know which side is open when
        action_mask indicates an in-position state (mask doesn't
        encode direction) — GBTAgent.select_action is the layer with
        access to current_side and is what UnifiedExecutor actually
        calls; use that for correct in-position CLOSE/HOLD behaviour.
        This method falls back to a flat-position interpretation of
        the 3-class output when called standalone.
        """
        lsh = self._raw_probs(state)  # [LONG, SHORT, HOLD]
        probs4 = np.zeros(4, dtype=np.float32)
        probs4[0], probs4[1], probs4[3] = lsh[0], lsh[1], lsh[2]

        if action_mask is not None:
            invalid = ~np.asarray(action_mask, dtype=bool)
            probs4[invalid] = 0.0
            total = probs4.sum()
            probs4 = probs4 / total if total > 0 else np.full(4, 0.25, dtype=np.float32)

        action = int(np.argmax(probs4))
        log_probs = np.log(probs4 + 1e-8)
        info = {"raw_long_short_hold": lsh}
        return action, probs4, log_probs, info

    def act(self, state_np: np.ndarray, deterministic: bool = True,
            device: str = "cpu", action_mask=None, **_ignored):
        action, probs, _log_probs, _info = self.get_action(state_np, action_mask=action_mask)
        if not deterministic:
            valid = probs > 0
            if valid.any():
                p = probs / probs.sum()
                action = int(np.random.choice(4, p=p))
        return action, probs


class GBTAgent:
    """
    MCKNNAgent-interface-compatible wrapper around GBTPolicy, so
    UnifiedExecutor works with minimal/no changes. select_action()
    additionally accepts current_side explicitly (since — unlike
    MCKNNMemory — this agent has no bank to infer position context
    from) to correctly resolve the in-position CLOSE-vs-HOLD decision.

    If the caller doesn't pass current_side (e.g. UnifiedExecutor,
    which doesn't know about this GBT-specific kwarg), the in-position
    branch defaults to treating the open side as whichever of
    LONG/SHORT currently has higher probability — a reasonable
    approximation but exact behaviour requires wiring current_side
    through explicitly if you extend UnifiedExecutor for this agent.
    """

    def __init__(self, state_dim: int = 122, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = GBTPolicy(action_dim=action_dim)
        self.device = "cpu"

    # Deliberately NOT declaring query_tick/query_episode_id — see
    # GBTPolicy's module docstring on why this keeps
    # unified_executor.py's duck-typed temporal-arg detection OFF.
    def select_action(self, state: np.ndarray, deterministic: bool = True,
                      action_mask=None, current_side: str = None):
        lsh = self.actor._raw_probs(state)
        if action_mask is not None and not action_mask[0]:
            # In-position branch (LONG/SHORT disallowed => mask==[F,F,T,T])
            side = current_side
            if side is None:
                side = "LONG" if lsh[0] >= lsh[1] else "SHORT"
            probs = self.actor._in_position_probs(lsh, side)
            action = int(np.argmax(probs))
            if not deterministic:
                p = probs / probs.sum()
                action = int(np.random.choice(4, p=p))
            return action, probs
        return self.actor.act(state, deterministic=deterministic, action_mask=action_mask)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.actor.fit(X, y, **kwargs)
        print(f"[GBTAgent] ✅ Fit complete on {len(X):,} rows  "
              f"(classes={self.actor.classes_.tolist()})")

    def update(self, episode_buffer=None):
        # No-op — kept for interface parity with MCKNNAgent.update(),
        # which callers loop over once per epoch. GBTAgent is trained
        # once via fit(), not incrementally.
        pass

    def save(self, path: str = "outcomes/gbt_agent.joblib"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "model": self.actor.model, "classes_": self.actor.classes_,
            "state_dim": self.state_dim, "action_dim": self.action_dim,
        }, path)
        print(f"[GBTAgent] ✅ Saved → {path}")

    def load(self, path: str = "outcomes/gbt_agent.joblib"):
        if not os.path.exists(path):
            print(f"[GBTAgent] ⚠️  No checkpoint at {path} — model is untrained.")
            return
        data = joblib.load(path)
        self.actor.model = data["model"]
        self.actor.classes_ = data["classes_"]
        self.state_dim = data["state_dim"]
        self.action_dim = data["action_dim"]
        print(f"[GBTAgent] ✅ Loaded ← {path}")