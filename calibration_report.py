"""
calibration_report.py
─────────────────────────
Reliability diagrams + calibration metrics (Brier score, Expected
Calibration Error) for ANY probability-producing policy (GBTPolicy,
MCKNNPolicy, etc). Answers the question the vote-share approach never
checked: does "80% conviction" actually resolve favorably ~80% of the
time?

Usage
──────
    from calibration_report import evaluate_calibration, plot_reliability_diagram
    report = evaluate_calibration(pred_probs, true_class, class_names=["LONG","SHORT","HOLD"])
    print(report["macro_ece"], report["macro_brier"])
    plot_reliability_diagram(pred_probs, true_class, class_names, out_path)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    Standard ECE for a one-vs-rest probability column: bins predictions
    by predicted probability, compares mean predicted probability to
    observed frequency within each bin, weights by bin population.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return float("nan")
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for a single one-vs-rest probability column
    (lower is better, 0 = perfect)."""
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def evaluate_calibration(pred_probs: np.ndarray, true_class: np.ndarray,
                          class_names: list, n_bins: int = 10) -> dict:
    """
    pred_probs  : (n, n_classes) calibrated probability matrix.
    true_class  : (n,) integer class labels, values in [0, n_classes)
                  indexing into class_names (i.e. already remapped to
                  dense 0..n_classes-1, not raw action ids — see
                  main_gbt.py's `slot` mapping).
    Returns per-class ECE/Brier plus a macro-average.
    """
    per_class = {}
    for j, name in enumerate(class_names):
        p = pred_probs[:, j]
        y = (true_class == j).astype(float)
        per_class[name] = {
            "ece":   expected_calibration_error(p, y, n_bins=n_bins),
            "brier": brier_score(p, y),
            "n_positive": int(y.sum()), "n_total": len(y),
        }
    macro_ece   = float(np.mean([v["ece"]   for v in per_class.values()]))
    macro_brier = float(np.mean([v["brier"] for v in per_class.values()]))
    return {"per_class": per_class, "macro_ece": macro_ece, "macro_brier": macro_brier}


def plot_reliability_diagram(pred_probs: np.ndarray, true_class: np.ndarray,
                              class_names: list, out_path: str, n_bins: int = 10):
    n_classes = pred_probs.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5))
    if n_classes == 1:
        axes = [axes]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for j, (name, ax) in enumerate(zip(class_names, axes)):
        p = pred_probs[:, j]
        y = (true_class == j).astype(float)
        obs_freq, counts = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
            if mask.sum() == 0:
                obs_freq.append(np.nan); counts.append(0)
            else:
                obs_freq.append(y[mask].mean())
                counts.append(int(mask.sum()))

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
        ax.plot(bin_centers, obs_freq, "o-", color="#7f5af0", label="observed")
        ax.set_title(f"{name}\nECE={expected_calibration_error(p, y, n_bins):.3f}  "
                     f"Brier={brier_score(p, y):.3f}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        for x, c in zip(bin_centers, counts):
            if c > 0:
                ax.annotate(str(c), (x, 0.02), fontsize=6, ha="center", color="gray")

    fig.suptitle("Reliability Diagram (bin labels = sample count)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")