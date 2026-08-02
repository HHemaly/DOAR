"""
psych_review.py — psychologist-review master table + inter-rater agreement.

Two responsibilities:
  1. Maintain a master review CSV (one row per reviewed item per reviewer).
  2. Compute agreement metrics ONLY from real submitted reviews:
       - raw agreement %
       - per-rule / per-class agreement
       - Cohen's kappa (exactly two raters)
       - Fleiss' kappa (three or more raters)

It never fabricates reviews. If no reviews exist, it says so and returns empty
metrics. Real reviews are appended (e.g. parsed from submitted forms) via
`append_review_rows`.
"""

from __future__ import annotations
import os
import csv
from collections import defaultdict

MASTER_FIELDS = [
    "case_id", "image_filename", "true_class", "predicted_class",
    "prediction_confidence", "item_type", "item_id", "reviewer_id",
    "rating", "comment", "timestamp",
]

RATINGS = ["Agree", "Partially agree", "Disagree", "Uncertain", "N/A"]


def init_master(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MASTER_FIELDS).writeheader()
    return path


def append_review_rows(path: str, rows: list[dict]) -> int:
    """Append reviewer rows to the master CSV. Returns count appended."""
    init_master(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        for r in rows:
            w.writerow(r)
    return len(rows)


def load_master(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def _cohen_kappa(pairs: list[tuple]) -> float | None:
    """Cohen's kappa for two raters. pairs = [(rating_a, rating_b), ...]."""
    if not pairs:
        return None
    cats = sorted({r for p in pairs for r in p})
    idx = {c: i for i, c in enumerate(cats)}
    n = len(pairs)
    k = len(cats)
    mat = [[0] * k for _ in range(k)]
    for a, b in pairs:
        mat[idx[a]][idx[b]] += 1
    po = sum(mat[i][i] for i in range(k)) / n
    row = [sum(mat[i]) / n for i in range(k)]
    col = [sum(mat[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    if pe == 1:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def _fleiss_kappa(item_ratings: list[dict]) -> float | None:
    """
    Fleiss' kappa for >=3 raters.
    item_ratings: list of {category: count} per item; each item must have the
    SAME total number of ratings n.
    """
    if not item_ratings:
        return None
    cats = sorted({c for d in item_ratings for c in d})
    N = len(item_ratings)
    n = sum(item_ratings[0].get(c, 0) for c in cats)
    if n < 2:
        return None
    # p_j: proportion of all assignments to category j
    total = N * n
    p = {c: sum(d.get(c, 0) for d in item_ratings) / total for c in cats}
    # P_i: extent of agreement for item i
    Pi = []
    for d in item_ratings:
        s = sum(d.get(c, 0) ** 2 for c in cats)
        Pi.append((s - n) / (n * (n - 1)))
    P_bar = sum(Pi) / N
    Pe = sum(v ** 2 for v in p.values())
    if Pe == 1:
        return 1.0
    return round((P_bar - Pe) / (1 - Pe), 4)


def _binary_agree(rating: str) -> int:
    """Collapse to agree(1)/not-agree(0) for raw-agreement %; ignore N/A upstream."""
    return 1 if rating == "Agree" else 0


def compute_agreement(master_csv: str) -> dict:
    """
    Compute agreement metrics from the master review CSV.

    Returns a dict; if there are no reviews, `has_reviews=False` and metrics
    are empty (nothing is invented).
    """
    rows = [r for r in load_master(master_csv) if r.get("rating") and r["rating"] != "N/A"]
    if not rows:
        return {"has_reviews": False,
                "note": "No psychologist reviews submitted yet. Agreement not computed."}

    reviewers = sorted({r["reviewer_id"] for r in rows if r.get("reviewer_id")})
    n_reviewers = len(reviewers)

    # Group ratings by (item_type,item_id,case_id) -> {reviewer: rating}
    items = defaultdict(dict)
    for r in rows:
        key = (r["case_id"], r["item_type"], r["item_id"])
        items[key][r["reviewer_id"]] = r["rating"]

    # Raw agreement %: share of items where all present reviewers gave same rating
    full_agree = sum(1 for d in items.values() if len(set(d.values())) == 1)
    raw_pct = round(100 * full_agree / len(items), 2) if items else 0.0

    # Per-item-type agreement
    by_type = defaultdict(lambda: [0, 0])
    for (cid, itype, iid), d in items.items():
        by_type[itype][1] += 1
        if len(set(d.values())) == 1:
            by_type[itype][0] += 1
    per_type = {t: round(100 * a / n, 2) for t, (a, n) in by_type.items() if n}

    # Per-class agreement (by true_class)
    class_lookup = {(r["case_id"], r["item_type"], r["item_id"]): r.get("true_class", "")
                    for r in rows}
    by_class = defaultdict(lambda: [0, 0])
    for key, d in items.items():
        cls = class_lookup.get(key, "")
        by_class[cls][1] += 1
        if len(set(d.values())) == 1:
            by_class[cls][0] += 1
    per_class = {c: round(100 * a / n, 2) for c, (a, n) in by_class.items() if n}

    result = {
        "has_reviews": True,
        "n_reviewers": n_reviewers,
        "n_items_reviewed": len(items),
        "raw_agreement_pct": raw_pct,
        "per_item_type_agreement_pct": per_type,
        "per_class_agreement_pct": per_class,
        "reviewers": reviewers,
    }

    # Cohen's kappa (exactly 2 raters, items both rated)
    if n_reviewers == 2:
        a, b = reviewers
        pairs = [(d[a], d[b]) for d in items.values() if a in d and b in d]
        result["cohen_kappa"] = _cohen_kappa(pairs)
        result["kappa_type"] = "cohen"
        result["kappa_n_items"] = len(pairs)
    # Fleiss' kappa (>=3 raters, items rated by ALL raters)
    elif n_reviewers >= 3:
        item_counts = []
        for d in items.values():
            if len(d) == n_reviewers:  # fully-rated items only
                c = defaultdict(int)
                for rat in d.values():
                    c[rat] += 1
                item_counts.append(dict(c))
        result["fleiss_kappa"] = _fleiss_kappa(item_counts)
        result["kappa_type"] = "fleiss"
        result["kappa_n_items"] = len(item_counts)
    else:
        result["kappa_note"] = "Kappa needs >=2 reviewers rating shared items."

    # Disagreement examples
    disagree = [{"case_id": k[0], "item_type": k[1], "item_id": k[2],
                 "ratings": d}
                for k, d in items.items() if len(set(d.values())) > 1][:20]
    result["disagreement_examples"] = disagree
    return result
