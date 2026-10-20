"""C3 — evaluate every foreground method and the 49 E2 drawing features against
the human reference masks (CPU).

  python experiments/journal/scripts/c3_evaluate_references.py \
      --batch <Drive>/DOAR_JOURNAL_2026/C2_reference_batch \
      --primary HE --secondary AS \
      --out <Drive>/DOAR_JOURNAL_2026/C3_reference_evaluation

Outputs
  per_image_method.csv     one row per (task, method): region + boundary metrics
  features_long.csv        one row per (task, mask source, feature): value
  inter_rater.csv/json     agreement between primary and secondary annotator
  gate_inputs.csv          per task: V2/A0 states, assessability, V2 error
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from PIL import Image  # noqa: E402

from doar_journal import foreground as fg  # noqa: E402
from doar_journal import measurement, provenance, seg_metrics  # noqa: E402
from doar_journal.stats import cohen_kappa  # noqa: E402


def load_records(batch: Path, annotator: str) -> dict[str, dict]:
    recs = {}
    for p in sorted((batch / "records").glob(f"*__{annotator}.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        recs[r["task_id"]] = r
    return recs


def load_mask(batch: Path, task: str, annotator: str) -> np.ndarray:
    return np.asarray(Image.open(batch / "masks" / f"{task}__{annotator}.png").convert("L")) > 127


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boundary-tol", type=int, default=2)
    a = ap.parse_args(argv)
    batch, out = Path(a.batch), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(batch / "key_PRIVATE.csv", newline="", encoding="utf-8") as f:
        key = {r["task_id"]: r for r in csv.DictReader(f)}
    prim = load_records(batch, a.primary)
    if not prim:
        raise SystemExit(f"no records for annotator {a.primary}")

    method_rows, feat_rows, gate_rows = [], [], []
    for task, rec in sorted(prim.items()):
        img_path = batch / "images" / f"{task}.png"
        img = fg.load_rgb(img_path)
        k = key[task]
        results = {name: fn(img) for name, fn in fg.METHODS.items()}
        v2meta = results["v2"].meta
        base = {"task_id": task, "class": k["class"], "assessability": rec["assessability"],
                "v2_regime": v2meta["support_regime"], "v2_confidence": v2meta["support_confidence"],
                "v2_policy": v2meta["measurement_policy"], "a0_state": k["a0_state"]}
        assessable = rec["assessability"] == "ASSESSABLE"
        ref = load_mask(batch, task, a.primary) if assessable else None
        if assessable:
            ref_feats = measurement.numeric_features(measurement.features_with_mask(img_path, ref))
            for f_name, v in ref_feats.items():
                feat_rows.append({"task_id": task, "source": "reference", "feature": f_name, "value": v})
        for name, res in results.items():
            row = {**base, "method": name, "pred_fraction": float(res.mask.mean())}
            if assessable:
                row.update(seg_metrics.all_metrics(res.mask, ref, a.boundary_tol))
                feats = measurement.numeric_features(measurement.features_with_mask(img_path, res.mask))
                for f_name, v in feats.items():
                    feat_rows.append({"task_id": task, "source": name, "feature": f_name, "value": v})
            method_rows.append(row)
        v2row = next(r for r in method_rows if r["task_id"] == task and r["method"] == "v2")
        gate_rows.append({**base, "v2_f1": v2row.get("f1"), "v2_iou": v2row.get("iou"),
                          "v2_boundary_f1": v2row.get("boundary_f1"),
                          "a0_f1": next(r for r in method_rows if r["task_id"] == task and r["method"] == "a0").get("f1")})
        print(task, rec["assessability"], v2meta["measurement_policy"], v2row.get("f1"))

    write_csv(out / "per_image_method.csv", method_rows)
    write_csv(out / "features_long.csv", feat_rows)
    write_csv(out / "gate_inputs.csv", gate_rows)

    outputs = [out / "per_image_method.csv", out / "gate_inputs.csv"]
    if a.secondary:
        sec = load_records(batch, a.secondary)
        both = sorted(set(sec) & set(prim))
        ir = []
        for t in both:
            r = {"task_id": t, "assess_primary": prim[t]["assessability"], "assess_secondary": sec[t]["assessability"]}
            if prim[t]["assessability"] == sec[t]["assessability"] == "ASSESSABLE":
                r.update(seg_metrics.all_metrics(load_mask(batch, t, a.secondary), load_mask(batch, t, a.primary),
                                                 a.boundary_tol))
            ir.append(r)
        write_csv(out / "inter_rater.csv", ir)
        if ir:
            kappa = cohen_kappa([r["assess_primary"] for r in ir], [r["assess_secondary"] for r in ir])
            f1s = [r["f1"] for r in ir if "f1" in r]
            provenance.write_json(out / "inter_rater.json", {
                "n_double": len(ir), "assessability_kappa": kappa,
                "assessability_agreement": float(np.mean([r["assess_primary"] == r["assess_secondary"] for r in ir])),
                "mask_f1_median": float(np.median(f1s)) if f1s else None,
                "mask_f1_mean": float(np.mean(f1s)) if f1s else None, "n_mask_pairs": len(f1s)})
            outputs.append(out / "inter_rater.json")
    meta = provenance.run_metadata("C3_reference_evaluation", vars(a),
                                   inputs={"key": batch / "key_PRIVATE.csv"})
    provenance.write_json(out / "run_metadata.json", meta)
    provenance.mark_done(out, meta, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
