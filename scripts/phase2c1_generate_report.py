#!/usr/bin/env python
"""Phase 2C.1 post-annotation quality/detector-readiness report generator.

Reads the PRIVATE, gitignored annotation store and pilot mapping
(outputs/phase2c1/) and writes only non-private, derived summary tables to
artifacts/phase2c1/ -- no source paths, no emotion labels, no drawings.
Safe to commit the artifacts this script writes; never commit its inputs.

Usage (from repo root, with the venv active):
    python scripts/phase2c1_generate_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.ontology import CLASS_NAMES  # noqa: E402
from doar.phase2c1 import quality as quality_mod  # noqa: E402
from doar.phase2c1 import store as store_mod  # noqa: E402
from doar.phase2c1 import workspace as workspace_mod  # noqa: E402

EXPECTED_N_HUMAN_IMAGES = 80
EXPECTED_N_PROVISIONAL_IMAGES = 20


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / "outputs/phase2c1/annotation_store.csv")
    parser.add_argument("--mapping", type=Path, default=ROOT / "outputs/phase2c1/private_pilot_mapping.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase2c1")
    args = parser.parse_args()

    store = store_mod.load_store(args.store)
    human = {k: r for k, r in store.items() if r.annotator_type == "human"}

    expected_human_ids = [f"p2b_{i:04d}" for i in range(EXPECTED_N_HUMAN_IMAGES)]
    expected_provisional_ids = [f"p2b_{i:04d}" for i in range(EXPECTED_N_PROVISIONAL_IMAGES)]
    integrity = quality_mod.build_integrity_report(
        store,
        expected_human_pilot_ids=expected_human_ids,
        expected_provisional_pilot_ids=expected_provisional_ids,
        expected_total_rows=(EXPECTED_N_HUMAN_IMAGES + EXPECTED_N_PROVISIONAL_IMAGES) * len(CLASS_NAMES),
        expected_human_rows=EXPECTED_N_HUMAN_IMAGES * len(CLASS_NAMES),
        expected_provisional_rows=EXPECTED_N_PROVISIONAL_IMAGES * len(CLASS_NAMES),
    )

    n_images = len({r.pilot_id for r in human.values()})
    support = quality_mod.class_support(human)
    readiness = quality_mod.classify_detector_readiness(support, n_images=n_images)
    disagreements = quality_mod.compute_human_vs_provisional_disagreements(store)

    split_counts: dict[str, int] = {}
    locked_test_ids: list[str] = []
    if args.mapping.exists():
        mapping_rows = workspace_mod.load_pilot_mapping(args.mapping)
        for r in mapping_rows:
            split_counts[r["original_split"]] = split_counts.get(r["original_split"], 0) + 1
        locked_test_ids = sorted(workspace_mod.pilot_ids_from_locked_test_split(args.mapping))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. annotation_integrity_report.json
    integrity_out = dict(integrity)
    integrity_out["locked_test_split_distribution"] = split_counts
    integrity_out["n_locked_test_pilot_ids"] = len(locked_test_ids)
    (args.output_dir / "annotation_integrity_report.json").write_text(
        json.dumps(integrity_out, indent=2), encoding="utf-8")

    # 2. human_class_frequency.csv
    with (args.output_dir / "human_class_frequency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "present", "absent", "uncertain", "not_assessable",
                    "n_images", "prevalence", "assessable_count", "uncertainty_rate"])
        for cls in CLASS_NAMES:
            s = support[cls]
            assessable = s["present"] + s["absent"]
            prevalence = s["present"] / n_images if n_images else None
            uncertainty_rate = (s["uncertain"] + s["not_assessable"]) / n_images if n_images else None
            w.writerow([cls, s["present"], s["absent"], s["uncertain"], s["not_assessable"],
                        n_images, f"{prevalence:.4f}" if prevalence is not None else "",
                        assessable, f"{uncertainty_rate:.4f}" if uncertainty_rate is not None else ""])

    # 3. annotation_status_distribution.csv (overall, human rows only; plus
    #    the classes-marked-present-per-image distribution as extra rows)
    with (args.output_dir / "annotation_status_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "key", "count", "percentage_of_human_rows"])
        total_human_rows = len(human)
        status_totals: dict[str, int] = {}
        for s in support.values():
            for status, n in s.items():
                status_totals[status] = status_totals.get(status, 0) + n
        for status, n in sorted(status_totals.items()):
            pct = n / total_human_rows if total_human_rows else 0
            w.writerow(["status_total", status, n, f"{pct:.4f}"])

        pilot_ids = sorted({r.pilot_id for r in human.values()})
        present_counts: dict[int, int] = {}
        for pid in pilot_ids:
            n_present = sum(1 for r in human.values() if r.pilot_id == pid and r.status == "present")
            present_counts[n_present] = present_counts.get(n_present, 0) + 1
        for k in sorted(present_counts):
            pct = present_counts[k] / len(pilot_ids) if pilot_ids else 0
            w.writerow(["classes_present_per_image", k, present_counts[k], f"{pct:.4f}"])

    # 4. human_vs_provisional_comparison.csv (per-class + overall)
    with (args.output_dir / "human_vs_provisional_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "n_compared", "percent_agreement"])
        for cls in CLASS_NAMES:
            n_compared_cls = disagreements["per_class_n_compared"].get(cls, 0)
            pct = disagreements["per_class_percent_agreement"].get(cls)
            w.writerow([cls, n_compared_cls, f"{pct:.4f}" if pct is not None else ""])
        w.writerow(["OVERALL", disagreements["n_compared"],
                    f"{disagreements['overall_percent_agreement']:.4f}"
                    if disagreements["overall_percent_agreement"] is not None else ""])

    # 5. human_vs_provisional_disagreements.csv
    with (args.output_dir / "human_vs_provisional_disagreements.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["pilot_id", "class_name", "human_status", "provisional_status",
                   "human_instance_count", "provisional_instance_count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in disagreements["disagreement_rows"]:
            w.writerow({k: row[k] for k in fields})

    # 6. detector_readiness_by_class.csv
    with (args.output_dir / "detector_readiness_by_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "readiness_tier", "n_present", "uncertainty_rate", "rationale"])
        for cls in CLASS_NAMES:
            r = readiness[cls]
            w.writerow([cls, r["readiness_tier"], r["n_present"], f"{r['uncertainty_rate']:.4f}", r["rationale"]])

    summary = {
        "integrity_all_checks_passed": integrity["all_checks_passed"],
        "n_human_images": n_images,
        "n_disagreements_vs_provisional": disagreements["n_disagree"],
        "overall_percent_agreement_vs_provisional": disagreements["overall_percent_agreement"],
        "readiness_tiers": {cls: readiness[cls]["readiness_tier"] for cls in CLASS_NAMES},
        "locked_test_split_distribution": split_counts,
        "output_dir": str(args.output_dir),
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
