#!/usr/bin/env python
"""Phase 2C.4A: validation/correction pass over Phase 2C.4's already-generated
predictions ONLY. No detector is rerun, nothing is downloaded, nothing is
annotated, no rule is activated. Reads the already-committed
artifacts/phase2c4/raw_predictions_{model}.csv and
artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv (commit 39e65b8,
frozen, never overwritten) plus the private local store/mapping (never
committed) for ground truth, and writes only new, non-private, derived
tables to artifacts/phase2c4a/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.ontology import CLASS_NAMES  # noqa: E402
from doar.phase2c1 import store as store_mod  # noqa: E402
from doar.phase2c1 import workspace as workspace_mod  # noqa: E402
from doar.phase2c2.cohorts import DEV_ELIGIBLE, FULL, LOCKED_TEST, split_pilot_ids_by_cohort  # noqa: E402
from doar.phase2c2.ground_truth import genuine_human_ground_truth  # noqa: E402
from doar.phase2c4 import calibration as cal_mod  # noqa: E402
from doar.phase2c4 import macro_metrics as macro_mod  # noqa: E402
from doar.phase2c4 import model_status as status_mod  # noqa: E402
from doar.phase2c4 import uncertainty as unc_mod  # noqa: E402

MODELS = ("owlv2", "grounding_dino", "florence2", "yolo_world")
CALIBRATION_MODELS = ("owlv2", "grounding_dino")
ORIGINAL_THRESHOLD = {"owlv2": cal_mod.OWLV2_ORIGINAL_THRESHOLD,
                       "grounding_dino": cal_mod.GROUNDING_DINO_ORIGINAL_THRESHOLD}


def load_detected(path: Path) -> dict[str, dict[str, bool]]:
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


def load_class_cohort_metrics(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> dict:
    metrics_path = ROOT / "artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv"
    store_path = ROOT / "outputs/phase2c1/annotation_store.csv"
    mapping_path = ROOT / "outputs/phase2c1/private_pilot_mapping.csv"
    out_dir = ROOT / "artifacts/phase2c4a"

    all_metric_rows = load_class_cohort_metrics(metrics_path)
    store = store_mod.load_store(store_path)
    mapping_rows = workspace_mod.load_pilot_mapping(mapping_path)
    cohorts = split_pilot_ids_by_cohort(mapping_rows)

    # ---- Section 1+2: dev-only model selection with fixed 10-class macro,
    # documented zero-division policy ------------------------------------
    dev_summaries = macro_mod.macro_summaries_by_model(all_metric_rows, DEV_ELIGIBLE)
    locked_summaries = macro_mod.macro_summaries_by_model(all_metric_rows, LOCKED_TEST)
    full_summaries_for_reference = macro_mod.macro_summaries_by_model(all_metric_rows, FULL)

    def summary_rows(summaries):
        return [{
            "model": s.model, "cohort": s.cohort,
            "n_classes_total": s.n_classes_total,
            "macro_f1": s.macro_f1,
            "n_classes_included_f1": s.n_classes_included_f1,
            "n_classes_excluded_f1": s.n_classes_excluded_f1,
            "excluded_f1_classes": ";".join(s.excluded_f1_classes),
            "macro_balanced_accuracy": s.macro_balanced_accuracy,
            "n_classes_included_bal_acc": s.n_classes_included_bal_acc,
            "n_classes_excluded_bal_acc": s.n_classes_excluded_bal_acc,
            "excluded_bal_acc_classes": ";".join(s.excluded_bal_acc_classes),
            "low_support_classes": ";".join(s.low_support_classes),
        } for s in summaries]

    write_csv(out_dir / "dev_only_model_selection_summary.csv", summary_rows(dev_summaries))
    write_csv(out_dir / "locked_test_descriptive_summary.csv", summary_rows(locked_summaries))
    write_csv(out_dir / "full_80_summary_for_reference_only.csv", summary_rows(full_summaries_for_reference))

    # ---- Section 4: bootstrap CIs on the dev cohort only -----------------
    ci_rows = []
    for model in MODELS:
        detected_path = ROOT / f"artifacts/phase2c4/raw_predictions_{model}.csv"
        predicted_by_class = load_detected(detected_path)
        gt_by_class = {cls: genuine_human_ground_truth(store, cls) for cls in CLASS_NAMES}
        ci = unc_mod.bootstrap_macro_ci(cohorts[DEV_ELIGIBLE], gt_by_class, predicted_by_class)
        ci_rows.append({"model": model, "cohort": DEV_ELIGIBLE, **ci})
    write_csv(out_dir / "dev_cohort_bootstrap_ci.csv", ci_rows)

    # ---- Section 5: calibration (owlv2, grounding_dino only) -------------
    # Belt-and-suspenders: the dev cohort itself already excludes the
    # locked-test split, but this asserts it explicitly before any
    # threshold is chosen, matching Phase 2C.1's own future-readiness guard.
    workspace_mod.assert_pilot_ids_exclude_locked_test(cohorts[DEV_ELIGIBLE], mapping_path)

    frozen_by_model = {}
    all_sweep_rows = []
    for model in CALIBRATION_MODELS:
        raw_path = ROOT / f"artifacts/phase2c4/raw_predictions_{model}.csv"
        scores_by_class = load_scores(raw_path)
        dev_set = set(cohorts[DEV_ELIGIBLE])
        gt_dev = {cls: {pid: s for pid, s in genuine_human_ground_truth(store, cls).items() if pid in dev_set}
                  for cls in CLASS_NAMES}
        scores_dev = {cls: {pid: v for pid, v in scores_by_class[cls].items() if pid in dev_set}
                      for cls in CLASS_NAMES}
        for cls in CLASS_NAMES:
            all_sweep_rows.extend(
                {"model": model, **row}
                for row in cal_mod.sweep_thresholds(cls, gt_dev[cls], scores_dev[cls],
                                                      cal_mod.CANDIDATE_THRESHOLDS[model]))
        frozen = cal_mod.freeze_operating_points(
            model, ORIGINAL_THRESHOLD[model], gt_dev, scores_dev,
            cal_mod.CANDIDATE_THRESHOLDS[model], min_positive_support=5)
        frozen_by_model[model] = frozen

    write_csv(out_dir / "calibration_dev_threshold_sweep.csv", all_sweep_rows)

    frozen_rows = [{"model": f.model, "class_name": f.class_name,
                     "original_threshold": f.original_threshold, "decision": f.decision,
                     "frozen_threshold": f.frozen_threshold, "dev_precision": f.dev_precision,
                     "dev_recall": f.dev_recall, "dev_balanced_accuracy": f.dev_balanced_accuracy,
                     "dev_n_present": f.dev_n_present, "low_support": f.low_support,
                     "reason": f.reason}
                    for model_frozen in frozen_by_model.values() for f in model_frozen]
    write_csv(out_dir / "calibration_frozen_operating_points.csv", frozen_rows)

    # ---- locked-test cohort: descriptive only, applied AFTER freezing ----
    locked_rows_all = []
    for model in CALIBRATION_MODELS:
        raw_path = ROOT / f"artifacts/phase2c4/raw_predictions_{model}.csv"
        scores_by_class = load_scores(raw_path)
        locked_set = set(cohorts[LOCKED_TEST])
        gt_locked = {cls: {pid: s for pid, s in genuine_human_ground_truth(store, cls).items() if pid in locked_set}
                     for cls in CLASS_NAMES}
        scores_locked = {cls: {pid: v for pid, v in scores_by_class[cls].items() if pid in locked_set}
                          for cls in CLASS_NAMES}
        locked_rows_all.extend(cal_mod.apply_frozen_thresholds(frozen_by_model[model], gt_locked, scores_locked))
    write_csv(out_dir / "calibration_locked_test_descriptive.csv", locked_rows_all)

    # ---- Section 3: Florence-2 / model configuration status --------------
    write_csv(out_dir / "model_configuration_status.csv", status_mod.to_rows())

    summary = {
        "dev_ranking": [(s.model, s.macro_balanced_accuracy, s.macro_f1) for s in dev_summaries],
        "output_dir": str(out_dir),
    }
    (out_dir / "phase2c4a_correction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
