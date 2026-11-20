"""Phase 4 fairness check: HBCU vs. other institutions at the final model's
fixed test-set cutoff (0.717), with bootstrap confidence intervals since the
HBCU group is small (128 of 4046 test rows -- see NOTES.md "Phase 4").
"""

import numpy as np


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def rates_from_counts(tp, fp, tn, fn):
    """recall (TPR), false positive rate, precision -- NaN when the
    denominator is zero (e.g. a bootstrap resample with no actual positives),
    rather than raising or silently returning 0."""
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    return recall, fpr, precision


def group_summary(y_true, y_pred) -> dict:
    """Raw counts and point-estimate rates for one group -- reported
    alongside the bootstrap CIs below since counts this small (the HBCU
    group has very few distressed rows) can make a single point-estimate
    rate misleading on its own."""
    y_true = np.asarray(y_true)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    recall, fpr, precision = rates_from_counts(tp, fp, tn, fn)
    return {
        "n": len(y_true),
        "n_distressed": int((y_true == 1).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "fpr": fpr, "precision": precision,
    }


def _percentile_ci(values, lower_pct=2.5, upper_pct=97.5):
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(valid, [lower_pct, upper_pct]))


def bootstrap_group_metrics(y_true, y_pred, n_boot=2000, random_state=0) -> dict:
    """Resample (with replacement) from this group's own rows `n_boot` times
    and recompute recall/FPR/precision each time, to get a 95% CI around each
    rate -- the right way to show uncertainty for a small group (the point
    estimate alone can't convey how much a handful of rows could shift it).
    A resample with zero actual positives (or zero predicted positives) makes
    that resample's recall (or precision) undefined; those draws are dropped
    from that metric's CI rather than treated as 0, and the count of valid
    draws is reported so a CI built from very few draws is visible as such.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    recalls, fprs, precisions = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        tp, fp, tn, fn = confusion_counts(y_true[idx], y_pred[idx])
        r, f, p = rates_from_counts(tp, fp, tn, fn)
        recalls.append(r)
        fprs.append(f)
        precisions.append(p)

    recalls, fprs, precisions = np.array(recalls), np.array(fprs), np.array(precisions)
    return {
        "n_boot": n_boot,
        "recall_ci": _percentile_ci(recalls),
        "fpr_ci": _percentile_ci(fprs),
        "precision_ci": _percentile_ci(precisions),
        "n_valid_recall": int((~np.isnan(recalls)).sum()),
        "n_valid_fpr": int((~np.isnan(fprs)).sum()),
        "n_valid_precision": int((~np.isnan(precisions)).sum()),
    }
