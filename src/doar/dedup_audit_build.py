"""Phase G0.2, Step 2-3 -- builds MASTER_RULE_DEDUP_AUDIT.csv: a per-
SOURCE-ENTRY audit row explaining exactly why it was (or was not) merged
into a canonical observable, and why.

Reads `master_registry_v3_build.build_master_registry_v3()`'s ALREADY
hand-verified canonical_observable_id assignments (never re-derives a
different merge decision here -- this module is a REPORTING/explanation
layer over G0.1's real merge decisions, not a second, possibly-
inconsistent dedup pass). Classification is based on WHAT IS MEASURED
(the canonical_observable_id grouping), never on shared psychological
interpretation -- per the explicit instruction that deduplication must
never be interpretation-based.

review_status values: MERGED, KEPT_DISTINCT, SOURCE_SPECIFIC,
FRAMEWORK_ONLY, PROCESS_ONLY, LONGITUDINAL_ONLY, ETHICS_REVIEW.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .master_registry_v3_build import build_master_registry_v3

ROOT = Path(__file__).resolve().parents[2]
MASTER_DEDUP_AUDIT_PATH = ROOT / "MASTER_RULE_DEDUP_AUDIT.csv"

FIELDNAMES = [
    "source_entry_id", "source_observable", "source_family", "source_reference",
    "current_canonical_observable_id", "proposed_canonical_observable_id",
    "same_measurement_as_other_sources", "merged_with_source_ids", "merge_reason",
    "kept_distinct_reason", "evidence_family", "review_status",
]


def _review_status(entry: dict[str, Any], canonical_group_ids: list[str],
                    family_source_families: set[str]) -> str:
    """Priority order matters: a framework/ethics/process/longitudinal
    classification is more informative than a bare MERGED/KEPT_DISTINCT/
    SOURCE_SPECIFIC label and is reported first even for an entry that
    also happens to share a canonical observable with others."""
    if entry["source_family"] == "framework_source":
        return "FRAMEWORK_ONLY"
    if entry["maturity_status"] == "ETHICS_REVIEW_LIKELY_REJECT":
        return "ETHICS_REVIEW"
    if entry["requires_longitudinal_data"]:
        return "LONGITUDINAL_ONLY"
    if entry["requires_process_data"]:
        return "PROCESS_ONLY"
    if len(canonical_group_ids) > 1:
        return "MERGED"
    # No verified merge for this entry. Was there even a plausible
    # cross-literature candidate (another source family sharing this
    # evidence_family)? If yes, staying separate was a real, meaningful
    # decision (KEPT_DISTINCT); if no other source family ever touches
    # this evidence_family, there was nothing to consider merging with
    # in the first place (SOURCE_SPECIFIC).
    if len(family_source_families) > 1:
        return "KEPT_DISTINCT"
    return "SOURCE_SPECIFIC"


def _merge_reason(entry: dict[str, Any], other_ids: list[str]) -> str:
    return (f"Same measurable concept ('{entry['canonical_observable']}') as "
            f"{', '.join(other_ids)} -- merged under evidence family '{entry['evidence_family']}' "
            "because both describe the identical observable, not merely a similar interpretation.")


def _kept_distinct_reason(entry: dict[str, Any], family_source_families: set[str]) -> str:
    other_families = sorted(family_source_families - {entry["source_family"]})
    return (f"Shares evidence family '{entry['evidence_family']}' with entries from "
            f"{', '.join(other_families)}, but names a distinct, object/context-specific "
            f"measurable concept ('{entry['canonical_observable']}') -- not merged without "
            "verified measurement equivalence (see Step 2's own worked counter-examples).")


def _source_specific_reason(entry: dict[str, Any]) -> str:
    return (f"No entry from a different source family shares evidence family "
            f"'{entry['evidence_family']}' -- '{entry['canonical_observable']}' is inherently "
            f"specific to {entry['source_family']}, so no merge candidate exists.")


def _framework_reason() -> str:
    return "Governs HOW DOAR phrases/contextualizes findings (question generation, alternative explanations, ethics) -- not a measurable observable at all, so canonical-observable/evidence-family dedup does not apply."


def _ethics_reason() -> str:
    return "Historical gender-norm-dependent coding marker -- flagged ETHICS_REVIEW / likely reject, never operationalized; kept as a distinct historical source entry, not merged toward activation."


def _process_reason(entry: dict[str, Any]) -> str:
    return f"requires_process_data=True -- '{entry['canonical_observable']}' cannot be assessed from a single static image regardless of any evidence-family overlap."


def _longitudinal_reason(entry: dict[str, Any]) -> str:
    return f"requires_longitudinal_data=True -- '{entry['canonical_observable']}' requires multiple drawings/sessions, not assessable from one static image."


def build_dedup_audit() -> list[dict[str, Any]]:
    doc = build_master_registry_v3()
    entries = doc["entries"]
    canonical_observables = doc["canonical_observables"]

    # Which source_family values ever contribute to a given evidence_family
    # -- used to distinguish KEPT_DISTINCT (a real cross-literature merge
    # candidate existed) from SOURCE_SPECIFIC (no candidate ever existed).
    family_source_families: dict[str, set[str]] = {}
    for e in entries:
        if e["evidence_family"]:
            family_source_families.setdefault(e["evidence_family"], set()).add(e["source_family"])

    rows = []
    for e in entries:
        cid = e["canonical_observable_id"]
        group_ids = sorted(canonical_observables[cid]["source_entry_ids"]) if cid in canonical_observables else [e["id"]]
        other_ids = [i for i in group_ids if i != e["id"]]
        fam_families = family_source_families.get(e["evidence_family"], set())

        status = _review_status(e, group_ids, fam_families)
        merge_reason = kept_distinct_reason = ""
        if status == "MERGED":
            merge_reason = _merge_reason(e, other_ids)
        elif status == "KEPT_DISTINCT":
            kept_distinct_reason = _kept_distinct_reason(e, fam_families)
        elif status == "SOURCE_SPECIFIC":
            kept_distinct_reason = _source_specific_reason(e)
        elif status == "FRAMEWORK_ONLY":
            kept_distinct_reason = _framework_reason()
        elif status == "ETHICS_REVIEW":
            kept_distinct_reason = _ethics_reason()
        elif status == "PROCESS_ONLY":
            kept_distinct_reason = _process_reason(e)
        elif status == "LONGITUDINAL_ONLY":
            kept_distinct_reason = _longitudinal_reason(e)

        rows.append({
            "source_entry_id": e["id"], "source_observable": e["canonical_observable"],
            "source_family": e["source_family"], "source_reference": e["source_reference"],
            "current_canonical_observable_id": cid, "proposed_canonical_observable_id": cid,
            "same_measurement_as_other_sources": "yes" if len(other_ids) > 0 else "no",
            "merged_with_source_ids": "; ".join(other_ids),
            "merge_reason": merge_reason, "kept_distinct_reason": kept_distinct_reason,
            "evidence_family": e["evidence_family"] or "", "review_status": status,
        })
    return rows


def write_dedup_audit() -> Path:
    rows = build_dedup_audit()
    with open(MASTER_DEDUP_AUDIT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return MASTER_DEDUP_AUDIT_PATH


if __name__ == "__main__":
    path = write_dedup_audit()
    print(f"wrote {path}")
