"""Phase 2C.7 Stage 1: summarizes a real human-reviewed export from
Phase 2C.5/2C.6's part-annotation store (any `target_name`, not just
'eye' -- kept general so this module works unchanged if you later export
a different target). Read-only: never writes back to the export file.
"""
from __future__ import annotations

from collections import Counter

from ..phase2c5.schema import PartAnnotationRecord


def summarize_target_export(store: dict[str, PartAnnotationRecord], target_name: str) -> dict:
    rows = [r for r in store.values() if r.target_name == target_name]
    unique_pilots = {r.pilot_id for r in rows}
    status_counts = Counter(r.status for r in rows)
    annotator_ids = Counter(r.annotator_id for r in rows)
    annotator_types = Counter(r.annotator_type for r in rows)

    bbox_source_counts = Counter()
    n_instances_total = 0
    for r in rows:
        for inst in r.instances:
            bbox_source_counts[inst.bbox_source] += 1
            n_instances_total += 1

    instance_count_dist = Counter(len(r.instances) for r in rows if r.status == "present")
    n_one = instance_count_dist.get(1, 0)
    n_two = instance_count_dist.get(2, 0)
    n_more_than_two = sum(v for k, v in instance_count_dist.items() if k > 2)

    attr_value_dist: dict[str, Counter] = {}
    for r in rows:
        for k, v in r.attributes.items():
            attr_value_dist.setdefault(k, Counter())[v] += 1

    return {
        "target_name": target_name,
        "n_rows": len(rows),
        "n_unique_pilot_ids": len(unique_pilots),
        "status_counts": dict(status_counts),
        "annotator_ids": dict(annotator_ids),
        "annotator_types": dict(annotator_types),
        "n_instances_total": n_instances_total,
        "bbox_source_counts": dict(bbox_source_counts),
        "instance_count_distribution": dict(sorted(instance_count_dist.items())),
        "n_images_with_one_instance": n_one,
        "n_images_with_two_instances": n_two,
        "n_images_with_more_than_two_instances": n_more_than_two,
        "attribute_value_distributions": {k: dict(v) for k, v in attr_value_dist.items()},
        "reviewed_pilot_ids_sorted": sorted(unique_pilots),
    }


def completion_against_manifest(reviewed_pilot_ids: set[str], manifest_pilot_ids: set[str]) -> dict:
    missing = sorted(manifest_pilot_ids - reviewed_pilot_ids)
    unexpected = sorted(reviewed_pilot_ids - manifest_pilot_ids)
    return {
        "n_manifest": len(manifest_pilot_ids),
        "n_reviewed": len(reviewed_pilot_ids),
        "n_missing": len(missing),
        "missing_pilot_ids": missing,
        "n_unexpected_outside_manifest": len(unexpected),
        "unexpected_pilot_ids": unexpected,
        "complete": len(missing) == 0,
    }
