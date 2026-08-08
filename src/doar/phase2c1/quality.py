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

from ..phase2b.evaluation import MIN_POSITIVE_SUPPORT
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
    """ALL distinct annotator_ids, regardless of type -- includes any
    migrated Phase 2B legacy_provisional_human rows. Used only for the
    top-level inventory (`distinct_annotators` in the report); never used
    directly to select a pair for human-human Cohen's kappa -- see
    `_distinct_human_annotators`."""
    return sorted({r.annotator_id for r in store.values()})


def _distinct_human_annotators(store: dict[str, AnnotationRecord]) -> list[str]:
    """Distinct annotator_ids whose rows are annotator_type == 'human' --
    i.e. a genuine Phase 2C.1 annotation event, never a migrated Phase 2B
    legacy_provisional_human row. This is the only candidate pool
    human-human Cohen's kappa / percent agreement may be computed over:
    a legacy_provisional_human row is a real human judgment, but it is
    single-annotator, unreviewed, and never confirmed independent ground
    truth (docs/PHASE2B_ANNOTATION_PROTOCOL.md) -- counting it as a second
    genuine rater would fabricate an inter-rater reliability claim this
    pilot's own design explicitly disclaims."""
    return sorted({r.annotator_id for r in store.values() if r.annotator_type == "human"})


def _best_annotator_pair(store: dict[str, AnnotationRecord],
                          candidate_ids: list[str] | None = None) -> tuple[str, str] | None:
    """Picks the two annotator_ids (restricted to `candidate_ids` if given)
    with the most overlapping (pilot_id, class_name) judgments -- the pair
    agreement statistics should actually be computed over. Returns None if
    fewer than 2 candidate annotators exist."""
    annotators = sorted(set(candidate_ids)) if candidate_ids is not None else _distinct_annotators(store)
    if len(annotators) < 2:
        return None
    candidates = set(annotators)
    by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in store.values():
        if r.annotator_id in candidates:
            by_key[(r.pilot_id, r.class_name)].add(r.annotator_id)
    pair_overlap: Counter = Counter()
    for annos in by_key.values():
        if len(annos) >= 2:
            for a in annos:
                for b in annos:
                    if a < b:
                        pair_overlap[(a, b)] += 1
    if not pair_overlap:
        # Two candidate annotators exist but never judged the same (pilot_id, class_name).
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
    """The top-level Stage E report for GENUINE human-human inter-rater
    agreement. Restricted to annotator_type == 'human' rows only -- any
    migrated Phase 2B legacy_provisional_human row is excluded from the
    candidate pool, so it can never be silently counted as a second real
    annotator here (see _distinct_human_annotators). Honest about
    single-annotator state: returns `sufficient_annotators: False` and a
    plain-language note rather than inventing a kappa or agreement rate.
    For a comparison against the legacy provisional labels specifically,
    see `compute_provisional_reference_comparison` -- a separate,
    differently-labeled statistic, never conflated with this one."""
    annotators = _distinct_annotators(store)
    human_annotators = _distinct_human_annotators(store)
    if len(human_annotators) < 2:
        return {
            "distinct_annotators": annotators,
            "human_annotators": human_annotators,
            "sufficient_annotators": False,
            "note": (
                "Fewer than 2 genuine human annotators (annotator_type=='human') have "
                "recorded annotations in this store. Inter-annotator agreement, percent "
                "agreement, and Cohen's kappa are not reported -- there is nothing to "
                "compare against. Migrated Phase 2B legacy_provisional_human rows, if "
                "present, are never counted as a second annotator for this statistic -- "
                "see compute_provisional_reference_comparison for a separate, "
                "explicitly-labeled comparison against them. This is not a missing "
                "feature; it is the honest state of a single-annotator pilot."
            ),
        }

    pair = _best_annotator_pair(store, candidate_ids=human_annotators)
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
        "human_annotators": human_annotators,
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


def _distinct_provisional_annotators(store: dict[str, AnnotationRecord]) -> list[str]:
    return sorted({r.annotator_id for r in store.values() if r.annotator_type == "legacy_provisional_human"})


def compute_provisional_reference_comparison(store: dict[str, AnnotationRecord]) -> dict:
    """A SEPARATE statistic from compute_agreement_report: compares each
    genuine human annotator's judgments against the migrated Phase 2B
    legacy_provisional_human labels, where both exist for the same
    (pilot_id, class_name). This is explicitly a reference comparison, NOT
    an inter-rater reliability statistic in the formal sense -- the
    provisional side is a single, unreviewed annotator's judgment, never
    confirmed independent ground truth (docs/PHASE2B_ANNOTATION_PROTOCOL.md).
    Still useful as a sanity/anchoring check, so it is reported here,
    clearly labeled, rather than omitted or silently folded into the
    human-human report."""
    human_annotators = _distinct_human_annotators(store)
    provisional_annotators = _distinct_provisional_annotators(store)
    if not human_annotators or not provisional_annotators:
        return {
            "human_annotators": human_annotators,
            "provisional_annotators": provisional_annotators,
            "available": False,
            "note": (
                "Requires at least one genuine human annotator (annotator_type=='human') "
                "and at least one legacy_provisional_human annotator in the store."
            ),
        }

    per_pair = {}
    for human_id in human_annotators:
        for provisional_id in provisional_annotators:
            matched = _matched_status_pairs(store, human_id, provisional_id)
            per_pair[f"{human_id}_vs_{provisional_id}"] = {
                "human_annotator_id": human_id,
                "provisional_annotator_id": provisional_id,
                "n_matched": len(matched),
                "percent_agreement": percent_agreement(matched),
            }

    return {
        "human_annotators": human_annotators,
        "provisional_annotators": provisional_annotators,
        "available": True,
        "methodology": (
            "Percent agreement only, computed per (human, provisional) annotator pair. "
            "Cohen's kappa is deliberately NOT reported here -- kappa is a measure of "
            "inter-RATER reliability between two raters assumed to be independently "
            "credible; the provisional side is a single, unreviewed Phase 2B annotator, "
            "not validated ground truth, so a kappa value here would misleadingly imply "
            "a formal reliability claim this comparison cannot support."
        ),
        "per_pair": per_pair,
    }


def compute_human_vs_provisional_disagreements(store: dict[str, AnnotationRecord]) -> dict:
    """Row-level companion to compute_provisional_reference_comparison:
    the full disagreement table (one row per disagreeing (pilot_id,
    class_name, human_annotator, provisional_annotator) judgment), plus
    the specific present<->absent transition counts and an
    uncertain/not_assessable-involvement count. Explicitly labeled
    'human vs legacy provisional' throughout -- never inter-rater
    reliability, never Cohen's kappa (see compute_provisional_reference_comparison
    for why)."""
    human_annotators = _distinct_human_annotators(store)
    provisional_annotators = _distinct_provisional_annotators(store)

    by_key: dict[tuple[str, str], dict[str, AnnotationRecord]] = defaultdict(dict)
    for r in store.values():
        if r.annotator_id in human_annotators or r.annotator_id in provisional_annotators:
            by_key[(r.pilot_id, r.class_name)][r.annotator_id] = r

    confusion: Counter = Counter()
    disagreement_rows = []
    n_compared = 0
    n_agree = 0
    per_class_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for (pilot_id, class_name), by_annotator in by_key.items():
        for human_id in human_annotators:
            for provisional_id in provisional_annotators:
                h = by_annotator.get(human_id)
                p = by_annotator.get(provisional_id)
                if h is None or p is None:
                    continue
                n_compared += 1
                confusion[(p.status, h.status)] += 1
                per_class_pairs[class_name].append((p.status, h.status))
                if h.status == p.status:
                    n_agree += 1
                else:
                    disagreement_rows.append({
                        "pilot_id": pilot_id,
                        "class_name": class_name,
                        "human_annotator_id": human_id,
                        "provisional_annotator_id": provisional_id,
                        "human_status": h.status,
                        "provisional_status": p.status,
                        "human_instance_count": h.instance_count,
                        "provisional_instance_count": p.instance_count,
                    })

    present_to_absent = sum(
        1 for r in disagreement_rows
        if r["provisional_status"] == "present" and r["human_status"] == "absent"
    )
    absent_to_present = sum(
        1 for r in disagreement_rows
        if r["provisional_status"] == "absent" and r["human_status"] == "present"
    )
    uncertain_involved = sum(
        1 for r in disagreement_rows
        if "uncertain" in (r["human_status"], r["provisional_status"])
        or "not_assessable" in (r["human_status"], r["provisional_status"])
    )

    per_class_agreement = {
        cls: percent_agreement(pairs) for cls, pairs in per_class_pairs.items()
    }
    per_class_n_compared = {cls: len(pairs) for cls, pairs in per_class_pairs.items()}

    return {
        "human_annotators": human_annotators,
        "provisional_annotators": provisional_annotators,
        "n_compared": n_compared,
        "n_agree": n_agree,
        "n_disagree": len(disagreement_rows),
        "overall_percent_agreement": (n_agree / n_compared) if n_compared else None,
        "per_class_percent_agreement": per_class_agreement,
        "per_class_n_compared": per_class_n_compared,
        "confusion_counts": {f"provisional={p}_human={h}": c for (p, h), c in confusion.items()},
        "provisional_present_to_human_absent": present_to_absent,
        "provisional_absent_to_human_present": absent_to_present,
        "disagreements_involving_uncertain_or_not_assessable": uncertain_involved,
        "disagreement_rows": sorted(disagreement_rows, key=lambda r: (r["pilot_id"], r["class_name"])),
        "label": "human vs legacy provisional comparison -- NOT inter-rater reliability",
    }


def build_integrity_report(store: dict[str, AnnotationRecord], *,
                            expected_human_pilot_ids: list[str] | None = None,
                            expected_provisional_pilot_ids: list[str] | None = None,
                            expected_total_rows: int | None = None,
                            expected_human_rows: int | None = None,
                            expected_provisional_rows: int | None = None) -> dict:
    """Structural integrity report for Stage F/post-annotation review.
    Everything here re-derives from the already-loaded (and therefore
    already schema-validated -- AnnotationRecord.__post_init__ would have
    raised on any invalid status/count/bbox combination) store; this
    function adds the checks that validation alone can't catch: exact
    row/pilot counts, per-image class completeness, duplicate keys, and
    unknown annotator types.

    The expected_* counts are optional and unspecified by default -- this
    function is deliberately reusable for any store size, not hardcoded to
    one round's numbers (e.g. this round's 1000/800/200). Callers who know
    what to expect (e.g. a specific annotation round) should pass them
    explicitly to get the corresponding check."""
    from .schema import ANNOTATOR_TYPES, validate_image_complete

    human_rows = [r for r in store.values() if r.annotator_type == "human"]
    provisional_rows = [r for r in store.values() if r.annotator_type == "legacy_provisional_human"]
    other_type_rows = [r for r in store.values()
                        if r.annotator_type not in ("human", "legacy_provisional_human")]

    human_pilot_ids = sorted({r.pilot_id for r in human_rows})
    provisional_pilot_ids = sorted({r.pilot_id for r in provisional_rows})

    incomplete_human = {}
    for pid in human_pilot_ids:
        for annotator_id in {r.annotator_id for r in human_rows if r.pilot_id == pid}:
            missing = validate_image_complete(human_rows, pid, annotator_id)
            if missing:
                incomplete_human[f"{pid}__{annotator_id}"] = missing
    incomplete_provisional = {}
    for pid in provisional_pilot_ids:
        for annotator_id in {r.annotator_id for r in provisional_rows if r.pilot_id == pid}:
            missing = validate_image_complete(provisional_rows, pid, annotator_id)
            if missing:
                incomplete_provisional[f"{pid}__{annotator_id}"] = missing

    annotation_ids = [r.annotation_id for r in store.values()]
    duplicate_ids = sorted({aid for aid in annotation_ids if annotation_ids.count(aid) > 1})

    unknown_types = sorted({r.annotator_type for r in store.values()} - set(ANNOTATOR_TYPES))
    blank_image_id = [r.annotation_id for r in store.values() if not r.image_id.strip()]
    blank_group = [r.annotation_id for r in store.values() if not r.source_image_group.strip()]

    checks = {
        "no_other_annotator_types": len(other_type_rows) == 0,
        "no_duplicate_annotation_ids": len(duplicate_ids) == 0,
        "no_unknown_annotator_types": len(unknown_types) == 0,
        "no_blank_image_id": len(blank_image_id) == 0,
        "no_blank_source_image_group": len(blank_group) == 0,
        "every_human_image_has_all_10_classes": len(incomplete_human) == 0,
        "every_provisional_image_has_all_10_classes": len(incomplete_provisional) == 0,
    }
    if expected_total_rows is not None:
        checks["total_rows_matches_expected"] = len(store) == expected_total_rows
    if expected_human_rows is not None:
        checks["human_rows_matches_expected"] = len(human_rows) == expected_human_rows
    if expected_provisional_rows is not None:
        checks["provisional_rows_matches_expected"] = len(provisional_rows) == expected_provisional_rows
    if expected_human_pilot_ids is not None:
        checks["human_pilot_ids_match_expected"] = human_pilot_ids == sorted(expected_human_pilot_ids)
    if expected_provisional_pilot_ids is not None:
        checks["provisional_pilot_ids_match_expected"] = (
            provisional_pilot_ids == sorted(expected_provisional_pilot_ids)
        )

    return {
        "total_rows": len(store),
        "n_human_rows": len(human_rows),
        "n_provisional_rows": len(provisional_rows),
        "n_other_type_rows": len(other_type_rows),
        "n_unique_human_pilot_ids": len(human_pilot_ids),
        "n_unique_provisional_pilot_ids": len(provisional_pilot_ids),
        "human_pilot_ids": human_pilot_ids,
        "provisional_pilot_ids": provisional_pilot_ids,
        "duplicate_annotation_ids": duplicate_ids,
        "unknown_annotator_types": unknown_types,
        "incomplete_human_images": incomplete_human,
        "incomplete_provisional_images": incomplete_provisional,
        "blank_image_id_rows": blank_image_id,
        "blank_source_image_group_rows": blank_group,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "note": (
            "Status/instance_count/bbox validity is enforced structurally by "
            "AnnotationRecord.__post_init__ at load time (store.load_store), not "
            "re-checked here -- an invalid combination would have raised before "
            "this function could even be called, so 'the store loaded successfully' "
            "already proves that dimension."
        ),
    }


# Readiness tiers, applied in this priority order. Defined independently of
# any detector/model output -- these thresholds are fixed here, in code,
# before being applied to this round's real counts, not chosen afterward to
# fit a result. MIN_POSITIVE_SUPPORT (5) reuses Phase 2B's own existing,
# documented convention (docs/PHASE2B_ANNOTATION_PROTOCOL.md); the other two
# are new, Phase 2C.1-specific thresholds, stated explicitly rather than
# implied.
EVALUATION_SUPPORTED_MIN_POSITIVE = 20  # enough for a reasonably tight 95% CI at n=80 images
HIGH_UNCERTAINTY_RATE = 0.15  # >15% of images uncertain/not_assessable for this class


def classify_detector_readiness(support: dict[str, dict[str, int]], n_images: int) -> dict[str, dict]:
    """Classifies each class into exactly one readiness tier from real
    human annotation counts only (never from any detector/model output):

    - 'high_uncertainty': annotator couldn't confidently judge this class
      in > HIGH_UNCERTAINTY_RATE of images -- any support number for it is
      suspect regardless of count.
    - 'insufficient_positive_support': n_present < MIN_POSITIVE_SUPPORT (5).
    - 'evaluation_supported': n_present >= EVALUATION_SUPPORTED_MIN_POSITIVE (20).
    - 'exploratory_only': everything else (5 <= n_present < 20).

    Checked in that order, so a class that is both low-support AND
    high-uncertainty is reported as high_uncertainty (the more fundamental
    problem -- more annotation wouldn't help until the class definition/
    ambiguity itself is addressed).
    """
    out = {}
    for cls, counts in support.items():
        n_present = counts.get("present", 0)
        n_uncertain_total = counts.get("uncertain", 0) + counts.get("not_assessable", 0)
        uncertainty_rate = (n_uncertain_total / n_images) if n_images else 0.0

        if uncertainty_rate > HIGH_UNCERTAINTY_RATE:
            tier = "high_uncertainty"
            rationale = (
                f"{n_uncertain_total}/{n_images} images ({uncertainty_rate:.1%}) could not be "
                f"confidently judged for this class -- above the {HIGH_UNCERTAINTY_RATE:.0%} bar."
            )
        elif n_present < MIN_POSITIVE_SUPPORT:
            tier = "insufficient_positive_support"
            rationale = f"Only {n_present} positive example(s), below the minimum of {MIN_POSITIVE_SUPPORT}."
        elif n_present >= EVALUATION_SUPPORTED_MIN_POSITIVE:
            tier = "evaluation_supported"
            rationale = (
                f"{n_present} positive examples -- enough for a descriptive precision/recall/"
                f"ranking-separation evaluation with a reasonably informative 95% CI at n={n_images}."
            )
        else:
            tier = "exploratory_only"
            rationale = (
                f"{n_present} positive examples -- above the minimum-support floor but below "
                f"the {EVALUATION_SUPPORTED_MIN_POSITIVE}-positive bar for a confident evaluation; "
                "any metric computed for this class should be treated as directional, not final."
            )
        out[cls] = {
            "readiness_tier": tier,
            "rationale": rationale,
            "n_present": n_present,
            "n_uncertain_total": n_uncertain_total,
            "uncertainty_rate": uncertainty_rate,
        }
    return out
