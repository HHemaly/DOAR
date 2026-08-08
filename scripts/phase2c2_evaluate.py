#!/usr/bin/env python
"""Phase 2C.2: evaluate Phase 2B's existing, unchanged CLIP zero-shot and
classical-CV baselines against genuine Phase 2C.1 human annotations, for
all 80 pilot images, split into three cohorts (full / dev-eligible /
locked-test). Writes only non-private, derived tables to
artifacts/phase2c2/ -- no source paths, no emotion labels, no drawings.

Reuses the model/threshold/prompt/preprocessing exactly as recorded in
artifacts/phase2b/model_manifest.json -- this script tunes nothing.
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
from doar.phase2c1 import store as store_mod  # noqa: E402
from doar.phase2c1 import workspace as workspace_mod  # noqa: E402
from doar.phase2c2 import evaluation as eval_mod  # noqa: E402


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / "outputs/phase2c1/annotation_store.csv")
    parser.add_argument("--mapping", type=Path, default=ROOT / "outputs/phase2c1/private_pilot_mapping.csv")
    parser.add_argument("--raw-predictions", type=Path,
                         default=ROOT / "artifacts/phase2c2/raw_predictions_80.csv")
    parser.add_argument("--phase2b-per-class-metrics", type=Path,
                         default=ROOT / "artifacts/phase2b/per_class_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase2c2")
    args = parser.parse_args()

    store = store_mod.load_store(args.store)
    mapping_rows = workspace_mod.load_pilot_mapping(args.mapping)
    raw_predictions = {r["pilot_id"]: r for r in csv.DictReader(args.raw_predictions.open(encoding="utf-8"))}

    results = eval_mod.evaluate_all(store, raw_predictions, mapping_rows)

    old_rows = []
    if args.phase2b_per_class_metrics.exists():
        old_rows = list(csv.DictReader(args.phase2b_per_class_metrics.open(encoding="utf-8")))
    comparison = eval_mod.compare_to_phase2b_20(results, old_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. class_metrics_by_cohort.csv -- everything except the FP/FN lists.
    metrics_fields = ["class_name", "baseline", "cohort", "n_images_in_cohort", "n_present",
                       "n_absent", "n_uncertain", "n_not_assessable", "n_usable_for_metrics",
                       "sufficient_support", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
                       "specificity", "ranking_separation"]
    with (args.output_dir / "class_metrics_by_cohort.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics_fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in metrics_fields})

    # 2. fp_fn_examples.csv -- one row per (class, baseline, cohort, error_type, pilot_id).
    with (args.output_dir / "fp_fn_examples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "baseline", "cohort", "error_type", "pilot_id"])
        for r in results:
            for pid in r["false_positive_pilot_ids"]:
                w.writerow([r["class_name"], r["baseline"], r["cohort"], "false_positive", pid])
            for pid in r["false_negative_pilot_ids"]:
                w.writerow([r["class_name"], r["baseline"], r["cohort"], "false_negative", pid])

    # 3. comparison_to_phase2b_20.csv
    comparison_fields = ["class_name", "old_n_present_20", "new_n_present_80", "old_precision",
                          "new_precision", "old_recall", "new_recall", "old_ranking_separation",
                          "new_ranking_separation", "old_sufficient_support", "new_sufficient_support"]
    with (args.output_dir / "comparison_to_phase2b_20.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=comparison_fields)
        w.writeheader()
        for row in comparison:
            w.writerow(row)

    # 4. model_manifest.json -- explicit record that config is unchanged from Phase 2B.
    phase2b_manifest_path = ROOT / "artifacts/phase2b/model_manifest.json"
    phase2b_manifest = json.loads(phase2b_manifest_path.read_text(encoding="utf-8")) \
        if phase2b_manifest_path.exists() else {}
    manifest_out = dict(phase2b_manifest)
    manifest_out["phase"] = "2C.2_descriptive_baseline_evaluation"
    manifest_out["reused_unchanged_from_phase2b"] = True
    manifest_out["pilot_sample"] = dict(phase2b_manifest.get("pilot_sample", {}))
    manifest_out["pilot_sample"]["n_images"] = 80
    manifest_out["pilot_sample"]["annotator"] = "genuine_phase2c1_human_annotation"
    manifest_out["note"] = (
        "No threshold, prompt, model, or preprocessing was changed for this evaluation. "
        "Ground truth is genuine Phase 2C.1 human annotation only, never legacy Phase 2B "
        "provisional labels. No threshold tuning, calibration, or training performed."
    )
    (args.output_dir / "model_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

    summary = {
        "n_result_rows": len(results),
        "n_classes": len(CLASS_NAMES),
        "n_comparison_rows": len(comparison),
        "output_dir": str(args.output_dir),
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
