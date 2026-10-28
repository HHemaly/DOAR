"""Statistics used throughout the journal analyses.

Design rules (pre-registered in experiments/journal/PREREGISTRATION.md):
* Uncertainty about a score on a fixed evaluation set is estimated by a
  CLUSTER bootstrap: the resampling unit is the duplicate/source GROUP (all
  images that are copies of one drawing stay together), never the training
  seed. With `groups=None` every image is its own group (plain image-level
  bootstrap); that is used only where no grouping exists. Seed variation is
  reported separately as model variance.
* Estimand: the metric (e.g. macro-F1) on the evaluation set as sampled;
  the interval describes sampling variability of that set at the group level.
* Model-vs-model comparisons on the same images are PAIRED (the same group
  resamples for both models). Exact McNemar treats images as independent, so
  with clustered images it is reported only as a descriptive count table.
* Proportions use Wilson intervals. Families of tests use Holm correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt

import numpy as np
from scipy import stats as _st
from sklearn.metrics import cohen_kappa_score, f1_score

CLASSES = ("Angry", "Fear", "Happy", "Sad")


def macro_f1(y_true, y_pred, labels=CLASSES) -> float:
    """Macro F1 over a FIXED label set; a class never predicted scores 0."""
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n: int
    method: str

    def as_dict(self) -> dict:
        return {"estimate": self.estimate, "ci_low": self.low, "ci_high": self.high,
                "n": self.n, "method": self.method}


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _codes(labels, classes=CLASSES) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup.get(str(v), -1) for v in np.asarray(labels).tolist()], dtype=np.int64)


def draw_resamples(n: int, groups=None, n_boot: int = 10_000, seed: int = 20261020):
    """Bootstrap draws as (rows, flat): resample b contains images flat[rows == b].

    groups=None -> image-level bootstrap (n images with replacement).
    groups given -> cluster bootstrap: G groups drawn with replacement, every
    member of a drawn group enters the resample (resample sizes vary)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        return np.repeat(np.arange(n_boot), n), idx.ravel()
    groups = np.asarray([str(g) for g in groups])
    if len(groups) != n:
        raise ValueError("groups must have one entry per image")
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sizes = np.bincount(inv, minlength=len(uniq))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))      # groups per resample
    lens = sizes[picks]                                                  # images per picked group
    rows = np.repeat(np.repeat(np.arange(n_boot), len(uniq)), lens.ravel())
    # offsets within each picked group
    tot = lens.ravel()
    within = np.arange(tot.sum()) - np.repeat(np.cumsum(tot) - tot, tot)
    flat = order[np.repeat(starts[picks].ravel(), tot) + within]
    return rows, flat


def macro_f1_draws(y_true, y_pred, rows: np.ndarray, flat: np.ndarray, n_boot: int,
                   classes=CLASSES) -> np.ndarray:
    """Macro-F1 of every resample, from confusion counts in one vectorised pass.
    Identical to `macro_f1` (sklearn, fixed labels, zero_division=0) per resample
    (tested). Labels outside `classes` never count as a true positive."""
    k = len(classes)
    yf = _codes(y_true, classes)[flat]
    pf = _codes(y_pred, classes)[flat]

    def counts(mask, cls):
        return np.bincount(rows[mask] * k + cls[mask], minlength=n_boot * k).reshape(n_boot, k).astype(float)
    tp = counts((yf == pf) & (yf >= 0), yf)
    denom = counts(yf >= 0, yf) + counts(pf >= 0, pf)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(axis=1)


def macro_f1_resamples(y_true, y_pred, idx: np.ndarray, classes=CLASSES) -> np.ndarray:
    """Macro-F1 for every row of an (n_boot x n) index matrix (image-level draws)."""
    b, n = idx.shape
    return macro_f1_draws(y_true, y_pred, np.repeat(np.arange(b), n), idx.ravel(), b, classes)


def _resample(metric, y_true, y_pred, draws, n_boot):
    rows, flat = draws
    if metric is macro_f1:
        return macro_f1_draws(y_true, y_pred, rows, flat, n_boot)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.array([metric(y_true[flat[rows == b]], y_pred[flat[rows == b]]) for b in range(n_boot)])


def _unit(groups) -> str:
    return "image" if groups is None else "duplicate/source group"


def bootstrap_metric(y_true, y_pred, metric=macro_f1, n_boot: int = 10_000,
                     alpha: float = 0.05, seed: int = 20261020, groups=None) -> Interval:
    """Percentile bootstrap CI of `metric`; resampling unit = group (or image)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    samples = _resample(metric, y_true, y_pred, draw_resamples(n, groups, n_boot, seed), n_boot)
    lo, hi = _percentile_ci(samples, alpha)
    n_units = n if groups is None else len(set(map(str, groups)))
    return Interval(metric(y_true, y_pred), lo, hi, n,
                    f"percentile bootstrap, {n_boot} resamples of {n_units} {_unit(groups)}s")


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric=macro_f1, n_boot: int = 10_000,
                          alpha: float = 0.05, seed: int = 20261020, groups=None) -> dict:
    """metric(a) - metric(b) on the SAME images; both models see the same resamples.

    Returns the estimate, percentile CI and a two-sided bootstrap p-value
    (2 x the smaller tail mass of the resampled differences around 0).
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    d = draw_resamples(n, groups, n_boot, seed)
    diffs = _resample(metric, y_true, pred_a, d, n_boot) - _resample(metric, y_true, pred_b, d, n_boot)
    lo, hi = _percentile_ci(diffs, alpha)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"estimate": metric(y_true, pred_a) - metric(y_true, pred_b), "ci_low": lo,
            "ci_high": hi, "p_bootstrap": float(min(1.0, p)), "n": n,
            "resampling_unit": _unit(groups),
            "method": f"paired percentile bootstrap, {n_boot} resamples by {_unit(groups)}"}


def unpaired_bootstrap_diff(y_true_a, pred_a, y_true_b, pred_b, metric=macro_f1,
                            n_boot: int = 10_000, alpha: float = 0.05, seed: int = 20261020,
                            groups_a=None, groups_b=None) -> dict:
    """metric(subset A) - metric(subset B) for two DISJOINT image subsets,
    each resampled within itself (by group when groups are given)."""
    ya, pa, yb, pb = map(np.asarray, (y_true_a, pred_a, y_true_b, pred_b))
    da = draw_resamples(len(ya), groups_a, n_boot, seed)
    db = draw_resamples(len(yb), groups_b, n_boot, seed + 1)
    diffs = _resample(metric, ya, pa, da, n_boot) - _resample(metric, yb, pb, db, n_boot)
    lo, hi = _percentile_ci(diffs, alpha)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"estimate": metric(ya, pa) - metric(yb, pb), "ci_low": lo, "ci_high": hi,
            "p_bootstrap": float(min(1.0, p)), "n_a": len(ya), "n_b": len(yb),
            "resampling_unit": _unit(groups_a),
            "method": f"two-sample percentile bootstrap, {n_boot} resamples by {_unit(groups_a)}"}


def wilson(k: int, n: int, alpha: float = 0.05) -> Interval:
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "wilson")
    z = _st.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n, "wilson")


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact two-sided McNemar test on per-image correctness of two models.

    Assumes independent images. When images are clustered (duplicate groups),
    report the discordant counts descriptively and use the group-paired
    bootstrap p-value for inference."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("correctness vectors must align image by image")
    n01 = int(np.sum(a & ~b))   # A right, B wrong
    n10 = int(np.sum(~a & b))   # A wrong, B right
    m = n01 + n10
    if m == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "both_right": int(np.sum(a & b)),
            "both_wrong": int(np.sum(~a & ~b)), "p_exact": float(p), "n": int(a.size)}


def kendall_tau(order_a: list[str], order_b: list[str]) -> dict:
    """Kendall's tau-b between two rankings of the same items (best first)."""
    if sorted(order_a) != sorted(order_b):
        raise ValueError("rankings must contain the same items")
    ra = {m: i for i, m in enumerate(order_a)}
    rb = {m: i for i, m in enumerate(order_b)}
    items = sorted(order_a)
    res = _st.kendalltau([ra[m] for m in items], [rb[m] for m in items])
    return {"tau": float(res.statistic), "p": float(res.pvalue), "n_items": len(items)}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjustment; returns adjusted p and reject flag per key."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        adj = min(1.0, (m - i) * pvalues[k])
        running = max(running, adj)
        out[k] = {"p": pvalues[k], "p_holm": running, "reject": running <= alpha}
    return out


def icc_2_1(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater
    (Shrout & Fleiss 1979; Koo & Li 2016). `ratings` is n_targets x k_raters."""
    x = np.asarray(ratings, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom != 0 else float("nan")


def cohen_kappa(rater_a, rater_b) -> float:
    return float(cohen_kappa_score(rater_a, rater_b))
