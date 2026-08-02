"""
review.py — structured psychologist review + inter-rater agreement (Item 16).

Item-level ratings for: segmentation, objective_features, emotion_prediction,
rules, parent_wording, concerns, overall_usefulness. Ratings: Agree, Partially
agree, Disagree, Uncertain, N/A. Reviews are appended (never overwritten) to a
review-master CSV with reviewer_id, case_id, item, rating, comment, timestamp.

Agreement (raw %, Cohen's kappa for 2 raters, Fleiss' kappa for >=3) is computed
ONLY from real submitted reviews. Synthetic or incomplete reviews are excluded,
and when no real reviews exist the result is explicitly "unavailable" — never
fabricated.

Ported and adapted from the legacy reports.psych_review implementation.
"""

from __future__ import annotations
import csv
import os
from collections import defaultdict
from pathlib import Path

REVIEW_ITEMS = [
    "segmentation", "objective_features", "emotion_prediction", "rules",
    "parent_wording", "concerns", "overall_usefulness",
]
RATINGS = ["Agree", "Partially agree", "Disagree", "Uncertain", "N/A"]
MASTER_FIELDS = ["case_id", "item", "reviewer_id", "rating", "comment",
                 "timestamp", "is_synthetic"]


def init_master(path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MASTER_FIELDS).writeheader()
    return str(p)


def append_reviews(path, rows: list[dict]) -> int:
    """Append reviewer rows (append-only history). Unknown items/ratings rejected."""
    init_master(path)
    valid = []
    for r in rows:
        if r.get("item") not in REVIEW_ITEMS:
            raise ValueError(f"Unknown review item: {r.get('item')}")
        if r.get("rating") not in RATINGS:
            raise ValueError(f"Unknown rating: {r.get('rating')}")
        valid.append(r)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        w.writerows(valid)
    return len(valid)


def load_master(path) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def form_to_rows(case_id: str, reviewer_id: str, ratings: dict, comments: dict,
                 timestamp: str, is_synthetic: bool = False) -> list[dict]:
    """Convert a structured review form to master rows (UI-independent, C2).

    `ratings`/`comments` are {item: value} over REVIEW_ITEMS. Requires a
    reviewer_id and a stable case_id. Skips items left unrated. Rows are validated
    by append_reviews before being written."""
    if not (reviewer_id and str(reviewer_id).strip()):
        raise ValueError("reviewer_id is required")
    if not (case_id and str(case_id).strip()):
        raise ValueError("case_id is required")
    rows = []
    for item in REVIEW_ITEMS:
        rating = ratings.get(item)
        if rating in (None, ""):
            continue
        rows.append({
            "case_id": str(case_id), "item": item, "reviewer_id": str(reviewer_id).strip(),
            "rating": rating, "comment": comments.get(item, ""),
            "timestamp": timestamp, "is_synthetic": "true" if is_synthetic else "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Kappa (ported from legacy)
# ---------------------------------------------------------------------------

def cohen_kappa(pairs: list[tuple]) -> float | None:
    if not pairs:
        return None
    cats = sorted({r for p in pairs for r in p})
    idx = {c: i for i, c in enumerate(cats)}
    n, k = len(pairs), len(cats)
    mat = [[0] * k for _ in range(k)]
    for a, b in pairs:
        mat[idx[a]][idx[b]] += 1
    po = sum(mat[i][i] for i in range(k)) / n
    row = [sum(mat[i]) / n for i in range(k)]
    col = [sum(mat[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


def fleiss_kappa(item_ratings: list[dict]) -> float | None:
    if not item_ratings:
        return None
    cats = sorted({c for d in item_ratings for c in d})
    N = len(item_ratings)
    n = sum(item_ratings[0].get(c, 0) for c in cats)
    if n < 2:
        return None
    total = N * n
    p = {c: sum(d.get(c, 0) for d in item_ratings) / total for c in cats}
    Pi = [(sum(d.get(c, 0) ** 2 for c in cats) - n) / (n * (n - 1)) for d in item_ratings]
    P_bar = sum(Pi) / N
    Pe = sum(v ** 2 for v in p.values())
    return 1.0 if Pe == 1 else round((P_bar - Pe) / (1 - Pe), 4)


def compute_agreement(master_csv, exclude_synthetic: bool = True) -> dict:
    """Agreement from REAL reviews only. Excludes synthetic/incomplete; returns
    has_reviews=False (never fabricates) when none exist."""
    rows = load_master(master_csv)
    if exclude_synthetic:
        rows = [r for r in rows if str(r.get("is_synthetic", "")).lower() not in ("true", "1")]
    rows = [r for r in rows if r.get("rating") and r["rating"] != "N/A"
            and r.get("reviewer_id")]
    if not rows:
        return {"has_reviews": False,
                "note": "No real psychologist reviews collected yet; agreement unavailable."}

    reviewers = sorted({r["reviewer_id"] for r in rows})
    items = defaultdict(dict)
    for r in rows:
        items[(r["case_id"], r["item"])][r["reviewer_id"]] = r["rating"]

    full_agree = sum(1 for d in items.values() if len(set(d.values())) == 1)
    raw_pct = round(100 * full_agree / len(items), 2) if items else 0.0

    by_item = defaultdict(lambda: [0, 0])
    for (cid, item), d in items.items():
        by_item[item][1] += 1
        if len(set(d.values())) == 1:
            by_item[item][0] += 1
    per_item = {it: round(100 * a / n, 2) for it, (a, n) in by_item.items() if n}

    result = {
        "has_reviews": True,
        "n_reviewers": len(reviewers),
        "n_items_reviewed": len(items),
        "raw_agreement_pct": raw_pct,
        "per_item_agreement_pct": per_item,
        "reviewers": reviewers,
    }
    if len(reviewers) == 2:
        a, b = reviewers
        pairs = [(d[a], d[b]) for d in items.values() if a in d and b in d]
        result["cohen_kappa"] = cohen_kappa(pairs)
        result["kappa_type"] = "cohen"
        result["kappa_n_items"] = len(pairs)
    elif len(reviewers) >= 3:
        counts = []
        for d in items.values():
            if len(d) == len(reviewers):
                c = defaultdict(int)
                for rat in d.values():
                    c[rat] += 1
                counts.append(dict(c))
        result["fleiss_kappa"] = fleiss_kappa(counts)
        result["kappa_type"] = "fleiss"
        result["kappa_n_items"] = len(counts)
    else:
        result["kappa_note"] = "Kappa needs >=2 reviewers rating shared items."
    disagree = [{"case_id": k[0], "item": k[1], "ratings": d}
                for k, d in items.items() if len(set(d.values())) > 1][:20]
    result["disagreement_examples"] = disagree
    return result
