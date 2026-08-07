"""Phase 2C.1 Stage E -- annotation quality tooling.

All statistics are computed from whatever is actually in the store passed
in -- nothing here fabricates a second annotator, a reviewer, or an
agreement score. Where a statistic is not meaningful (e.g. Cohen's kappa
with fewer than MIN_PAIRS_FOR_KAPPA matched judgments, or with only one
distinct annotator_id in the store), the function says so explicitly in the
returned dict rather than returning a misleading number.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..phase2b.ontology import CLASS_NAMES
from .schema import AnnotationRecord, OBJECT_STATUSES

# Below this many matched (pilot_id, class_name) judgments from the same two
# annotators, a kappa point estimate is not reported -- consistent with this
# project's existing discipline (docs/PHASE2B_EVALUATION_PROTOCOL.md's
# MIN_POSITIVE_SUPPORT convention: state raw counts and a clear "not
# reported" flag rather than a number nobody should trust).
MIN_PAIRS_FOR_KAPPA = 5


def class_support(store: dict[str, AnnotationRecord]) -> dict[str, dict[str, int]]:
    """Per-class counts of present/absent/uncertain/not_assessable, across
    every row currently in the store (all annotators combined -- callers
    who want per-annotator support should filter the store first)."""
    out = {c: {s: 0 for s in sorted(OBJECT_STATUSES)} for c in CLASS_NAMES}
    for r in store.values():
        out.setdefault(r.class_name, {s: 0 for s in sorted(OBJECT_STATUSES)})
        out[r.class_name][r.status] += 1
    return out


def completion_rate(store: dict[str, AnnotationRecord], all_pilot_ids: list[str],
                     annotator_id: str) -> dict:
    """Fraction of `all_pilot_ids` that have a full 10-class row set from
    `annotator_id`. An image with 3 of 10 classes saved counts as
    incomplete, not partially complete -- matches the app's own
    'mark complete' semantics (schema.validate_image_complete)."""
    from .schema import validate_image_complete

    rows = [r for r in store.values() if r.annotator_id == annotator_id]
    complete = 0
    incomplete_pilot_ids = []
    for pid in all_pilot_ids:
        missing = validate_image_complete(rows, pid, annotator_id)
        if not missing:
            complete += 1
        else:
            incomplete_pilot_ids.append(pid)
    total = len(all_pilot_ids)
    return {
        "annotator_id": annotator_id,
        "total_images": total,
        "complete_images": complete,
        "incomplete_images": total - complete,
        "completion_rate": (complete / total) if total else None,
        "incomplete_pilot_ids": sorted(incomplete_pilot_ids),
    }


def review_coverage(store: dict[str, AnnotationRecord]) -> dict:
    n_total = len(store)
    n_reviewed = sum(1 for r in store.values() if r.review_status == "reviewed")
    n_in_review = sum(1 for r in store.values() if r.review_status == "in_review")
    return {
        "total_rows": n_total,
        "reviewed": n_reviewed,
        "in_review": n_in_review,
        "unreviewed": n_total - n_reviewed - n_in_review,
        "review_coverage_rate": (n_reviewed / n_total) if n_total else None,
    }


def _distinct_annotators(store: dict[str, AnnotationRecord]) -> list[str]:
    return sorted({r.annotator_id for r in store.values()})


def _best_annotator_pair(store: dict[str, AnnotationRecord]) -> tuple[str, str] | None:
    """Picks the two annotator_ids with the most overlapping
    (pilot_id, class_name) judgments -- the pair agreement statistics
    should actually be computed over. Returns None if fewer than 2
    annotators exist in the store at all."""
    annotators = _distinct_annotators(store)
    if len(annotators) < 2:
        return None
    by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in store.values():
        by_key[(r.pilot_id, r.class_name)].add(r.annotator_id)
    pair_overlap: Counter = Counter()
    for annos in by_key.values():
        if len(annos) >= 2:
            for a in annos:
                for b in annos:
                    if a < b:
                        pair_overlap[(a, b)] += 1
    if not pair_overlap:
        # Two annotators exist but never judged the same (pilot_id, class_name).
        return (annotators[0], annotators[1])
    return pair_overlap.most_common(1)[0][0]


def _matched_status_pairs(store: dict[str, AnnotationRecord], annotator_a: str,
                           annotator_b: str, class_name: str | None = None) -> list[tuple[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for r in store.values():
        if class_name is not None and r.class_name != class_name:
            continue
        if r.annotator_id in (annotator_a, annotator_b):
            by_key[(r.pilot_id, r.class_name)][r.annotator_id] = r.status
    pairs = []
    for statuses in by_key.values():
        if annotator_a in statuses and annotator_b in statuses:
            pairs.append((statuses[annotator_a], statuses[annotator_b]))
    return pairs


def percent_agreement(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    return agree / len(pairs)


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Standard two-rater Cohen's kappa. Returns None if there are no pairs
    or if expected agreement is 1.0 (undefined, e.g. every judgment is the
    same single category -- kappa's denominator would be zero)."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((count_a[label] / n) * (count_b[label] / n) for label in labels)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def count_disagreement(pairs_with_counts: list[tuple[int, int]]) -> dict:
    """Among (present, present) judgment pairs from two annotators, how
    often the recorded instance_count differs, and by how much."""
    if not pairs_with_counts:
        return {"n_compared": 0, "n_disagree": 0, "disagreement_rate": None, "mean_abs_diff": None}
    diffs = [abs(a - b) for a, b in pairs_with_counts]
    n_disagree = sum(1 for d in diffs if d != 0)
    return {
        "n_compared": len(pairs_with_counts),
        "n_disagree": n_disagree,
        "disagreement_rate": n_disagree / len(pairs_with_counts),
        "mean_abs_diff": sum(diffs) / len(diffs),
    }


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Standard IoU for normalized (x, y, w, h) axis-aligned boxes."""
    ax0, ay0, aw, ah = box_a
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = box_b
    bx1, by1 = bx0 + bw, by0 + bh

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def unresolved_disagreements(store: dict[str, AnnotationRecord]) -> list[AnnotationRecord]:
    return sorted(
        (r for r in store.values() if r.adjudication_status == "disagreement_unresolved"),
        key=lambda r: (r.pilot_id, r.class_name),
    )


def compute_agreement_report(store: dict[str, AnnotationRecord]) -> dict:
    """The top-level Stage E report. Honest about single-annotator state:
    returns `sufficient_annotators: False` and a plain-language note rather
    than inventing a kappa or agreement rate from one rater's own data
    compared against itself."""
    annotators = _distinct_annotators(store)
    if len(annotators) < 2:
        return {
            "distinct_annotators": annotators,
            "sufficient_annotators": False,
            "note": (
                "Only one human annotator (or zero) has recorded annotations in this "
                "store. Inter-annotator agreement, percent agreement, and Cohen's kappa "
                "are not reported -- there is nothing to compare against. This is not a "
                "missing feature; it is the honest state of a single-annotator pilot."
            ),
        }

    pair = _best_annotator_pair(store)
    annotator_a, annotator_b = pair
    per_class = {}
    for cls in CLASS_NAMES:
        matched = _matched_status_pairs(store, annotator_a, annotator_b, class_name=cls)
        kappa = cohens_kappa(matched) if len(matched) >= MIN_PAIRS_FOR_KAPPA else None
        per_class[cls] = {
            "n_matched": len(matched),
            "percent_agreement": percent_agreement(matched),
            "cohens_kappa": kappa,
            "kappa_reported": len(matched) >= MIN_PAIRS_FOR_KAPPA,
        }

    overall_matched = _matched_status_pairs(store, annotator_a, annotator_b)
    overall_kappa = cohens_kappa(overall_matched) if len(overall_matched) >= MIN_PAIRS_FOR_KAPPA else None

    count_pairs = []
    by_key: dict[tuple[str, str], dict[str, tuple[str, int]]] = defaultdict(dict)
    for r in store.values():
        if r.annotator_id in (annotator_a, annotator_b):
            by_key[(r.pilot_id, r.class_name)][r.annotator_id] = (r.status, r.instance_count)
    for entry in by_key.values():
        if annotator_a in entry and annotator_b in entry:
            (status_a, count_a), (status_b, count_b) = entry[annotator_a], entry[annotator_b]
            if status_a == "present" and status_b == "present":
                count_pairs.append((count_a, count_b))

    return {
        "distinct_annotators": annotators,
        "sufficient_annotators": True,
        "compared_pair": [annotator_a, annotator_b],
        "n_matched_overall": len(overall_matched),
        "percent_agreement_overall": percent_agreement(overall_matched),
        "disagreement_rate_overall": (
            1 - percent_agreement(overall_matched) if overall_matched else None
        ),
        "cohens_kappa_overall": overall_kappa,
        "kappa_reported_overall": len(overall_matched) >= MIN_PAIRS_FOR_KAPPA,
        "min_pairs_for_kappa": MIN_PAIRS_FOR_KAPPA,
        "per_class": per_class,
        "count_disagreement": count_disagreement(count_pairs),
        "unresolved_disagreement_count": len(unresolved_disagreements(store)),
    }
