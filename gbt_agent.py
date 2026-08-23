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

model_type — pluggable base classifier (this revision)
────────────────────────────────────────────────────────
GBTPolicy/GBTAgent now accept a `model_type` ("hgb", the default, or
"logistic") that swaps ONLY the base estimator construction in
GBTPolicy.fit() — the chronological fit/calibration split, purge_ticks
handling, isotonic calibration wrapper, and every downstream inference
method (_raw_probs, _in_position_probs, get_action, act,
GBTAgent.select_action) are shared verbatim between both model types.

This exists for main_gbt.py's run_logistic_baseline(): after a
--no-multi-timeframe A/B run ruled out feature dimensionality as the
dominant driver of the GBT's CPCV instability, the next question is
whether the instability is a GBT-capacity problem (fixable by more
regularization) or a label/regime problem (not fixable by swapping
classifiers at all). Running a near-minimal-capacity logistic
regression through the IDENTICAL CPCV paths/purge/embargo/labels the
GBT uses gives a cheap floor check for that question — see
main_gbt.py's run_logistic_baseline() docstring for the full
reasoning. "logistic" is not intended as a deployment candidate, only
a diagnostic.

BAGGED FITS (this revision) — fixing seed-to-seed fit variance
────────────────────────────────────────────────────────────────
main_gbt.py's nested gate confirmation showed confirmation seeds 2/3/4
disagreeing on pass/fail (1/3 passed) even though — critically —
generate_cpcv_paths()'s train/test masks are derived purely from row
counts and are therefore IDENTICAL across those three seeds.
random_state only ever reaches HistGradientBoostingClassifier's own
internal randomness (max_features per-split feature subsampling, the
early-stopping validation carve-out). Getting PASS/FAIL/FAIL on the
exact same evidence, purely from re-rolling the classifier's RNG, is a
stronger overfitting signal than "different folds gave different
results" would be — the model's conclusion about whether it has an
edge is not settled by the data alone.

GBTPolicy.fit() now accepts `n_bagged_fits`: instead of one base
estimator per (fold, seed), it fits `n_bagged_fits` independently-
seeded base estimators on the SAME chronological fit/calibration
split, calibrates each independently, and averages their predict_proba
output at inference time (see _BaggedGBTModel below). This is the
standard bagging remedy for exactly this kind of fit-variance, and
because max_features<1.0 already makes each member see a different
random feature subset per tree split — much like the base learners in
a random forest — the members are usefully decorrelated rather than
near-duplicates of one another, so averaging them should actually
reduce variance rather than just adding compute.

Logistic-regression members are deterministic given the same
(X_fit, y_fit) (lbfgs has no data-dependent randomness here), so
bagging is a no-op for model_type="logistic" and fit() forces
n_bagged_fits=1 for that branch regardless of what's requested — no
point spending compute re-fitting an identical model.
"""

import os
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import joblib

ACTION_NAMES = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}
VALID_MODEL_TYPES = ("hgb", "logistic")


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


class _BaggedGBTModel:
    """
    Averages predict_proba() across several independently-seeded,
    independently-calibrated members fit on the SAME chronological
    fit/calibration split — see GBTPolicy.fit()'s `n_bagged_fits` and
    the module docstring's "BAGGED FITS" section for why this exists
    (seed-to-seed CPCV pass/fail flips on IDENTICAL train/test folds,
    driven purely by the base estimator's internal RNG).

    Deliberately not a scikit-learn estimator with its own fit() —
    every member is already fully fit/calibrated by the time this
    wraps them. This is a thin inference-time averaging shim so
    GBTPolicy._raw_probs() and every downstream caller (get_action,
    act, GBTAgent.select_action) don't need to know whether bagging is
    in effect; with n_bagged_fits=1 this reduces to exactly one
    member's predict_proba(), i.e. byte-for-byte the old behaviour.
    """
    def __init__(self, members: list):
        if not members:
            raise ValueError("_BaggedGBTModel requires at least one fitted member.")
        self.members = members
        self.classes_ = members[0].classes_

    def predict_proba(self, X):
        probs = np.stack([m.predict_proba(X) for m in self.members], axis=0)
        return probs.mean(axis=0)


def _fit_hgb_base(X_fit, y_fit, random_state: int, hgb_kwargs: dict):
    """
    Construct + fit one HistGradientBoostingClassifier member. Factored
    out of GBTPolicy.fit() so the bagging loop there can call this once
    per member with a fresh copy of hgb_kwargs (avoids one member's
    max_features-unsupported fallback silently affecting a different
    member's kwargs dict).
    """
    hgb_kwargs.setdefault("max_iter", 150)
    hgb_kwargs.setdefault("learning_rate", 0.04)
    hgb_kwargs.setdefault("max_depth", 4)
    hgb_kwargs.setdefault("max_leaf_nodes", 15)
    hgb_kwargs.setdefault("min_samples_leaf", 200)
    hgb_kwargs.setdefault("l2_regularization", 5.0)
    hgb_kwargs.setdefault("validation_fraction", 0.15)
    hgb_kwargs.setdefault("n_iter_no_change", 15)
    # max_features (per-split feature subsampling) needs sklearn>=1.2;
    # degrade gracefully on older installs rather than hard-failing.
    hgb_kwargs.setdefault("max_features", 0.7)
    try:
        base = HistGradientBoostingClassifier(
            early_stopping=True, random_state=random_state, **hgb_kwargs,
        )
        base.fit(X_fit, y_fit)
    except TypeError:
        hgb_kwargs.pop("max_features", None)
        base = HistGradientBoostingClassifier(
            early_stopping=True, random_state=random_state, **hgb_kwargs,
        )
        base.fit(X_fit, y_fit)
    return base


def _calibrate_member(base, X_fit, y_fit, X_cal, y_cal):
    """Wrap one fitted base estimator in an isotonic calibrator, using
    the same prefit/manual-fallback logic GBTPolicy.fit() always used —
    factored out so it's called once per bagged member."""
    if _supports_prefit():
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    else:
        cal = _ManualIsotoneCalibrator(base)

    if len(np.unique(y_cal)) < 2:
        # Calibration split is degenerate (e.g. very short fold) —
        # fall back to calibrating on the fit split rather than
        # crashing; isotonic regression still runs, just without a
        # genuinely held-out calibration slice for this fold.
        cal.fit(X_fit, y_fit)
    else:
        cal.fit(X_cal, y_cal)
    return cal


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

    ENTRY-CONVICTION THRESHOLD (entry_threshold)
    ─────────────────────────────────────────────
    Previously get_action() always took argmax(LONG, SHORT, HOLD) —
    meaning a directional class could be selected with as little as
    ~34% probability if it merely edged out the other two. CPCV showed
    this produced hundreds of low-conviction trades per fold with
    near-zero aggregate edge (mean test avg/trade ≈ +0.01%, std 70x the
    mean). entry_threshold requires a directional class's OWN
    probability to exceed this bar before it's eligible to be chosen;
    otherwise that mass is folded into HOLD (see get_action()) so the
    policy stays flat instead of taking a low-conviction bet. Defaults
    to 0.5 (weakly stricter than pure argmax, since a directional class
    must clear 50% on its own now rather than just outscore HOLD/the
    other side). The deployed value is chosen empirically by
    main_gbt.py's CPCV threshold sweep, never hand-picked or fit on the
    holdout — see sweep_entry_thresholds() there.
    """

    def __init__(self, model=None, action_dim: int = 4, entry_threshold: float = 0.5,
                model_type: str = "hgb"):
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type={model_type!r} not in {VALID_MODEL_TYPES}")
        self.model = model
        self.action_dim = action_dim
        self.classes_ = None  # set on fit(); subset of {0,1,3}
        self.entry_threshold = entry_threshold
        self.model_type = model_type  # "hgb" (default, deployable) or
                                       # "logistic" (capacity-floor
                                       # diagnostic — see module docstring)

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            calibration_frac: float = 0.2,
            random_state: int = 0,
            purge_ticks: int = 0,
            min_fit_rows: int = 200,
            n_bagged_fits: int = 1,
            bag_seed_stride: int = 1000,
            **hgb_kwargs):
        """
        Fit the base classifier on a chronological "fit" prefix, then
        calibrate on a chronological "calibration" suffix via isotonic
        regression, with an optional purge gap removed from between them.

        BAGGING (n_bagged_fits, this revision): instead of a single
        (base, calibrator) pair, fits `n_bagged_fits` independently-
        seeded members on the SAME X_fit/y_fit and X_cal/y_cal splits
        (random_state, random_state + bag_seed_stride, + 2*stride, ...)
        and wraps them in _BaggedGBTModel, which averages predict_proba
        across members at inference time. See the module docstring's
        "BAGGED FITS" section for why: CPCV confirmation seeds were
        shown to flip pass/fail on IDENTICAL train/test folds purely
        from the base estimator's internal RNG, and averaging several
        such fits is the standard remedy. n_bagged_fits=1 (the default)
        is byte-for-byte the old single-fit behaviour. Forced to 1 for
        model_type="logistic" regardless of the requested value, since
        LogisticRegression(lbfgs) here is deterministic given the data
        and bagging it would just re-fit an identical model for no
        benefit.

        CHANGE (overfitting fix): the fit/calibration split used to be a
        RANDOM train_test_split (optionally stratified). X/y arrive here
        in original tick order — train_mask/np.flatnonzero indexing in
        main_gbt.py's run_cpcv()/run_final_training() never reorders rows
        — so a random split let calibration rows land tick-adjacent to
        fit rows. Because several features here have rolling lookbacks up
        to 200 ticks (see preprocessing.py) and the state can include
        15m/1h context (multi_timeframe_state.py), tick-adjacent rows
        share substantial overlapping information: the "held-out"
        calibration split wasn't actually held out. This is precisely the
        leak walkforward.py's purge/embargo logic exists to prevent
        everywhere else in this codebase, so the fit/calibration split
        now follows the same discipline: calibration is the LAST
        `calibration_frac` of rows (chronologically), and a `purge_ticks`
        buffer (pass walkforward.compute_required_lookback_ticks(), sized
        to whatever CPCV/walk-forward fold this X/y came from) is dropped
        from the fit set immediately before it.

        Also tightened base-model regularization (max_depth, max_leaf_nodes,
        min_samples_leaf, l2_regularization, max_features where supported)
        — the prior defaults (depth=6, l2=1.0, no leaf/feature cap) were
        producing >1000% train P/L against negative median test P/L in
        CPCV (see main_gbt.py's gate output), a classic small-n/high-dim
        (state_dim up to 122) overfitting signature for tree ensembles.
        (`hgb_kwargs` is ignored entirely when self.model_type=="logistic"
        — see the model_type branch below and the module docstring.)

        IMPORTANT: X/y here should already be the TRAIN-fold data from
        an outer CPCV/walk-forward split (see main_gbt.py) — this
        fit/calibration split is a further internal split of that,
        purely for calibration, not a substitute for proper purged
        out-of-sample evaluation.
        """
        n = len(X)
        cal_size = max(1, int(n * calibration_frac))
        cal_start = n - cal_size
        fit_end = max(0, cal_start - purge_ticks)
        if fit_end < min_fit_rows:
            # purge_ticks would eat too much of a short fold — fall back
            # to an unpurged (but still chronological, non-random) split
            # rather than starving the fit set.
            fit_end = cal_start

        X_fit, y_fit = X[:fit_end], y[:fit_end]
        X_cal, y_cal = X[cal_start:], y[cal_start:]

        if len(np.unique(y_fit)) < 2:
            raise ValueError(
                f"Fit split (rows 0:{fit_end}) contains only "
                f"{len(np.unique(y_fit))} class(es) — cannot train a "
                f"classifier. Check class balance / fold size."
            )

        effective_n_bags = 1 if self.model_type == "logistic" else max(1, n_bagged_fits)

        members = []
        for b in range(effective_n_bags):
            member_seed = random_state + b * bag_seed_stride
            if self.model_type == "logistic":
                # Near-minimal-capacity baseline — see module docstring on
                # why this exists (capacity-vs-label/regime floor check).
                # class_weight="balanced" compensates for HOLD dominating
                # the label distribution (see triple_barrier's label
                # balance) without needing a separate sample-weighting
                # scheme; multi_class handling and solver are left at
                # sklearn defaults (lbfgs, one-vs-rest for 3 classes) since
                # this model is a diagnostic floor, not a tuned deployment
                # candidate. hgb_kwargs (HGB-specific) are intentionally
                # ignored here.
                base = LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=member_seed,
                )
                base.fit(X_fit, y_fit)
            else:
                base = _fit_hgb_base(X_fit, y_fit, member_seed, dict(hgb_kwargs))

            members.append(_calibrate_member(base, X_fit, y_fit, X_cal, y_cal))

        self.model = _BaggedGBTModel(members)
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
        # Entry-conviction gate: a directional class only stays "live"
        # if its OWN calibrated probability clears entry_threshold.
        # Suppressed mass folds into HOLD (rather than vanishing) so
        # probs4 remains a valid distribution and argmax naturally
        # resolves to HOLD instead of a weak directional plurality —
        # see class docstring.
        long_ok  = lsh[0] > self.entry_threshold
        short_ok = lsh[1] > self.entry_threshold
        probs4[0] = lsh[0] if long_ok else 0.0
        probs4[1] = lsh[1] if short_ok else 0.0
        probs4[3] = lsh[2] + (0.0 if long_ok else lsh[0]) + (0.0 if short_ok else lsh[1])

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

    def __init__(self, state_dim: int = 122, action_dim: int = 4, entry_threshold: float = 0.5,
                model_type: str = "hgb"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = model_type
        self.actor = GBTPolicy(action_dim=action_dim, entry_threshold=entry_threshold,
                               model_type=model_type)
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

    def fit(self, X: np.ndarray, y: np.ndarray, purge_ticks: int = 0,
            n_bagged_fits: int = 1, **kwargs):
        self.actor.fit(X, y, purge_ticks=purge_ticks, n_bagged_fits=n_bagged_fits, **kwargs)
        effective_bags = 1 if self.model_type == "logistic" else max(1, n_bagged_fits)
        print(f"[GBTAgent] ✅ Fit complete on {len(X):,} rows  "
              f"(model_type={self.model_type}, classes={self.actor.classes_.tolist()}, "
              f"purge_ticks={purge_ticks}, n_bagged_fits={effective_bags})")

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
            "entry_threshold": self.actor.entry_threshold,
            "model_type": self.model_type,
        }, path)
        print(f"[GBTAgent] ✅ Saved → {path}  "
              f"(model_type={self.model_type}, entry_threshold={self.actor.entry_threshold:.2f})")

    def load(self, path: str = "outcomes/gbt_agent.joblib"):
        if not os.path.exists(path):
            print(f"[GBTAgent] ⚠️  No checkpoint at {path} — model is untrained.")
            return
        data = joblib.load(path)
        self.actor.model = data["model"]
        self.actor.classes_ = data["classes_"]
        self.state_dim = data["state_dim"]
        self.action_dim = data["action_dim"]
        # Backward-compat: checkpoints saved before this revision won't
        # have entry_threshold/model_type — fall back to the old
        # implicit behaviour (0.5 threshold, "hgb" model) rather than
        # erroring.
        self.actor.entry_threshold = data.get("entry_threshold", 0.5)
        self.model_type = data.get("model_type", "hgb")
        self.actor.model_type = self.model_type
        print(f"[GBTAgent] ✅ Loaded ← {path}  "
              f"(model_type={self.model_type}, entry_threshold={self.actor.entry_threshold:.2f})")