"""
per_image.py — generate the full per-image output folder from an AnalysisRecord.

Layout produced (matches the thesis brief):
    examples/<case_id>/
        original.<ext>
        annotated.png            (if detections with bbox exist)
        crops/crop_XXX.png       (per-detection crops)
        gradcam.png              (if a checkpoint is supplied)
        analysis.json            (the full AnalysisRecord)
        technical_report.html
        parent_report_en.html
        parent_report_ar.html
        psychologist_review.html

Also a top-level batch runner that builds many of these and updates the
psychologist master CSV + thesis collation.
"""

from __future__ import annotations
import os
import json
import shutil
from pathlib import Path

from src.reports import html_reports, annotate
from src.reports.psych_review import init_master, MASTER_FIELDS


def generate_case(record: dict, out_root: str,
                  checkpoint: str = None) -> dict:
    """
    Build one example folder from an AnalysisRecord. Returns updated artifacts.
    `checkpoint` (optional) enables Grad-CAM.
    """
    cid = record["image"]["id"]
    case_dir = os.path.join(out_root, cid)
    crops_dir = os.path.join(case_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    art = record.setdefault("artifacts", {})

    # Copy original
    src_img = record["image"].get("path")
    if src_img and os.path.exists(src_img):
        ext = Path(src_img).suffix or ".png"
        dst = os.path.join(case_dir, f"original{ext}")
        shutil.copy2(src_img, dst)
        art["original"] = dst
    else:
        art["original"] = src_img

    dets = record.get("detections", [])

    # Save per-detection crops
    if dets and art.get("original") and os.path.exists(art["original"]):
        crop_paths = annotate.save_crops(art["original"], dets, crops_dir)
        art["crops"] = crop_paths

        # Annotated image (verified detections only)
        ann_path = os.path.join(case_dir, "annotated.png")
        annotate.annotate_image(art["original"], dets, ann_path, only_verified=True)
        art["annotated"] = ann_path

    # Grad-CAM (optional)
    if checkpoint and art.get("original") and os.path.exists(art["original"]):
        try:
            from src.models.gradcam import generate_gradcam
            gc = generate_gradcam(art["original"], checkpoint,
                                  os.path.join(case_dir, "gradcam.png"))
            if gc:
                art["gradcam"] = gc
        except Exception as e:
            record.setdefault("_warnings", []).append(f"gradcam: {e}")

    # HTML reports (relative image paths resolve inside case_dir)
    tech = os.path.join(case_dir, "technical_report.html")
    _write(tech, html_reports.render_technical(record, base_dir=case_dir))
    art["technical_html"] = tech

    p_en = os.path.join(case_dir, "parent_report_en.html")
    _write(p_en, html_reports.render_parent(record, lang="en"))
    art["parent_en_html"] = p_en

    p_ar = os.path.join(case_dir, "parent_report_ar.html")
    _write(p_ar, html_reports.render_parent(record, lang="ar"))
    art["parent_ar_html"] = p_ar

    psy = os.path.join(case_dir, "psychologist_review.html")
    _write(psy, html_reports.render_psychologist(record, base_dir=case_dir))
    art["psychologist_html"] = psy

    # analysis.json (full record, with resolved artifact paths)
    ajson = os.path.join(case_dir, "analysis.json")
    with open(ajson, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    art["analysis_json"] = ajson

    return art


def generate_batch(records: list[dict], out_root: str,
                   checkpoint: str = None) -> dict:
    """
    Build example folders for many records and seed the psychologist master CSV
    (empty of ratings — reviews are filled in later by real psychologists).
    """
    examples_dir = os.path.join(out_root, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    master_csv = os.path.join(out_root, "psychologist_review", "review_master.csv")
    init_master(master_csv)

    generated = []
    for rec in records:
        try:
            generate_case(rec, examples_dir, checkpoint=checkpoint)
            generated.append(rec["image"]["id"])
        except Exception as e:
            print(f"[reports] case {rec.get('image',{}).get('id')} failed: {e}")

    # Seed master CSV with rows to be reviewed (rating blank until a human fills it)
    _seed_master_rows(records, master_csv)

    print(f"[reports] generated {len(generated)} example folders in {examples_dir}")
    return {"examples_dir": examples_dir, "master_csv": master_csv,
            "generated": generated}


def _seed_master_rows(records, master_csv):
    """Write one blank-rating row per reviewable item (model/detection/rule/overall)."""
    import csv
    rows = []
    for rec in records:
        cid = rec["image"]["id"]
        gt = rec["ground_truth"].get("label", "")
        mp = rec["model_prediction"]
        pred = mp.get("predicted_class", "")
        conf = mp.get("confidence", "")
        base = {"case_id": cid, "image_filename": rec["image"].get("filename", ""),
                "true_class": gt, "predicted_class": pred,
                "prediction_confidence": conf, "reviewer_id": "",
                "rating": "", "comment": "", "timestamp": ""}
        rows.append({**base, "item_type": "model_prediction", "item_id": "0"})
        for i, d in enumerate(rec.get("detections", [])):
            rows.append({**base, "item_type": "detection", "item_id": str(i)})
        for i, r in enumerate(rec.get("rules", [])):
            rows.append({**base, "item_type": "rule", "item_id": r.get("rule_id", str(i))})
        rows.append({**base, "item_type": "overall", "item_id": "0"})

    # Append template rows (rating empty) so reviewers see every item to score
    with open(master_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        w.writerows(rows)


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
