#!/usr/bin/env python
"""Phase 2C.4: evaluate every real detector's raw predictions (written by
scripts/phase2c4_run_detector.py) against genuine Phase 2C.1 human
annotations, across all three cohorts, for every model that has a raw
predictions file present. Writes only non-private, derived tables to
artifacts/phase2c4/ -- no source paths, no emotion labels, no drawings.
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
from doar.phase2c4 import evaluation as eval_mod  # noqa: E402

MODELS = ("owlv2", "grounding_dino", "florence2", "yolo_world")


def load_predictions(path: Path) -> dict[str, dict[str, bool]]:
    """Returns {class_name: {pilot_id: detected}} from a
    scripts/phase2c4_run_detector.py output file."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = {cls: {} for cls in CLASS_NAMES}
    for row in rows:
        pid = row["pilot_id"]
        for cls in CLASS_NAMES:
            out[cls][pid] = row[f"{cls}__detected"].strip().lower() == "true"
    return out


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = {cls: {} for cls in CLASS_NAMES}
    for row in rows:
        pid = row["pilot_id"]
        for cls in CLASS_NAMES:
            out[cls][pid] = float(row[f"{cls}__score"])
    return out


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / "outputs/phase2c1/annotation_store.csv")
    parser.add_argument("--mapping", type=Path, default=ROOT / "outputs/phase2c1/private_pilot_mapping.csv")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "outputs/phase2c4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase2c4")
    args = parser.parse_args()

    store = store_mod.load_store(args.store)
    mapping_rows = workspace_mod.load_pilot_mapping(args.mapping)

    all_results = []
    models_evaluated = []
    for model in MODELS:
        pred_path = args.predictions_dir / f"raw_predictions_{model}_private.csv"
        if not pred_path.exists():
            continue
        predicted_positive_by_class = load_predictions(pred_path)
        similarity_by_class = load_scores(pred_path)
        # Florence-2's score is a fixed 1.0 sentinel (no real confidence
        # exposed) -- ranking-separation is not meaningful for it, so its
        # similarity dict is omitted rather than fed a fake constant signal.
        if model == "florence2":
            similarity_by_class = None
        results = eval_mod.evaluate_model(model, store, mapping_rows,
                                           predicted_positive_by_class, similarity_by_class)
        all_results.extend(results)
        models_evaluated.append(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "class_name", "cohort", "n_images_in_cohort", "n_present", "n_absent",
              "n_uncertain", "n_not_assessable", "n_usable_for_metrics", "sufficient_support",
              "tp", "fp", "fn", "tn", "precision", "recall", "f1", "specificity",
              "balanced_accuracy", "ranking_separation"]
    with (args.output_dir / "phase2c4_model_class_cohort_metrics.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in fields})

    with (args.output_dir / "phase2c4_fp_fn_examples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "class_name", "cohort", "error_type", "pilot_id"])
        for r in all_results:
            for pid in r["false_positive_pilot_ids"]:
                w.writerow([r["model"], r["class_name"], r["cohort"], "false_positive", pid])
            for pid in r["false_negative_pilot_ids"]:
                w.writerow([r["model"], r["class_name"], r["cohort"], "false_negative", pid])

    summary = {
        "models_evaluated": models_evaluated,
        "n_result_rows": len(all_results),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "phase2c4_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
