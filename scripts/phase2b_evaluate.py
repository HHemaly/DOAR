#!/usr/bin/env python
"""Phase 2B: compute honest per-class metrics, threshold sensitivity, and
error analysis from real predictions vs. the real annotation manifest.
Writes artifacts/phase2b/{per_class_metrics,threshold_sensitivity,
error_analysis,baseline_metrics}.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.evaluation import (  # noqa: E402
    compute_class_metrics, ranking_separation, threshold_sensitivity, write_csv,
)
from doar.phase2b.ontology import CLASS_NAMES  # noqa: E402

THRESHOLDS = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "artifacts" / "phase2b" / "raw_predictions.csv")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "artifacts" / "phase2b" / "annotation_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase2b")
    args = parser.parse_args()

    preds = {r["pilot_id"]: r for r in csv.DictReader(args.predictions.open(encoding="utf-8"))}
    gt_rows = list(csv.DictReader(args.annotations.open(encoding="utf-8")))

    per_class_metrics, threshold_rows_all, error_rows = [], [], []
    for cls in CLASS_NAMES:
        ground_truth = {r["pilot_id"]: r["status"] for r in gt_rows if r["class"] == cls}
        similarity = {pid: float(r[f"{cls}__similarity"]) for pid, r in preds.items()}
        predicted_positive = {pid: r[f"{cls}__status"] == "detected" for pid, r in preds.items()}

        m = compute_class_metrics(cls, ground_truth, predicted_positive)
        sep = ranking_separation(ground_truth, similarity)
        per_class_metrics.append({
            "class": cls, "n_present": m.n_present, "n_absent": m.n_absent,
            "n_uncertain": m.n_uncertain, "n_not_assessable": m.n_not_assessable,
            "n_usable_for_metrics": m.n_usable, "sufficient_support": m.sufficient_support,
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
            "ranking_separation_vs_random_0.5": sep,
        })

        if m.n_present > 0 and m.n_absent > 0:
            for row in threshold_sensitivity(ground_truth, similarity, THRESHOLDS):
                row["class"] = cls
                threshold_rows_all.append(row)

        for pid, truth in ground_truth.items():
            if truth not in ("present", "absent"):
                continue
            pred_status = preds[pid][f"{cls}__status"]
            pred_positive = pred_status == "detected"
            if pred_positive != (truth == "present"):
                error_rows.append({
                    "class": cls, "pilot_id": pid, "ground_truth": truth,
                    "predicted_status": pred_status, "similarity": similarity[pid],
                    "error_type": "false_positive" if pred_positive else "false_negative",
                })

    write_csv(args.output_dir / "per_class_metrics.csv", per_class_metrics)
    write_csv(args.output_dir / "threshold_sensitivity.csv", threshold_rows_all)
    write_csv(args.output_dir / "error_analysis.csv", error_rows)
    print(f"wrote metrics for {len(per_class_metrics)} classes, "
          f"{len(threshold_rows_all)} threshold rows, {len(error_rows)} error rows")


if __name__ == "__main__":
    main()
