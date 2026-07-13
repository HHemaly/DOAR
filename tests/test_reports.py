"""
tests/test_reports.py — tests for the Phase-7 reporting/review/annotation stack.

No dataset, no torch, no network required. Uses the clearly-labelled synthetic
AnalysisRecord to exercise every report type + agreement maths.
"""

from __future__ import annotations
import os
import sys
import csv
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _synthetic():
    from src.reports.synthetic_example import make_synthetic_image, build_synthetic_record
    d = tempfile.mkdtemp()
    img = make_synthetic_image(os.path.join(d, "syn.png"))
    return build_synthetic_record(img), d


def test_schema_build_from_doc():
    from src.reports.schema import build_analysis_record, empty_model_prediction
    doc = {"source_image": "/x/a.png", "metadata": {"label_from_dataset": "Sad"},
           "image_quality": {"width_px": 100, "height_px": 100, "quality_score": 0.5},
           "psychological_rule_activations": [], "psychological_rule_activations_v2": []}
    rec = build_analysis_record(doc, {"parent_answer": "hi", "disclaimer": "not diagnostic"},
                                {"final_answer_status": "PASS"})
    assert rec["schema_version"] == "1.0"
    assert rec["ground_truth"]["label"] == "Sad"
    assert rec["model_prediction"]["available"] is False   # not run yet


def test_synthetic_record_is_flagged():
    rec, _ = _synthetic()
    assert rec["is_synthetic"] is True


def test_technical_report_renders_and_banners_synthetic():
    from src.reports import html_reports
    rec, _ = _synthetic()
    htmlout = html_reports.render_technical(rec)
    assert "<html" in htmlout.lower()
    assert "SYNTHETIC" in htmlout                       # banner present
    assert "not a probability" in htmlout               # CLIP caveat present
    assert "ALSCHULER_RED_WARM" in htmlout


def test_parent_reports_en_and_ar():
    from src.reports import html_reports
    rec, _ = _synthetic()
    en = html_reports.render_parent(rec, "en")
    ar = html_reports.render_parent(rec, "ar")
    assert "not diagnostic" in en.lower()
    assert "class=\"rtl\"" in ar or "class='rtl'" in ar  # RTL for Arabic
    assert "شكرا" in ar or "شكراً" in ar                 # Arabic content present


def test_psychologist_form_has_rating_options():
    from src.reports import html_reports
    rec, _ = _synthetic()
    form = html_reports.render_psychologist(rec)
    for opt in ("Agree", "Partially agree", "Disagree", "Uncertain", "N/A"):
        assert opt in form
    assert "review_overall" in form


def test_bbox_crop_and_annotation():
    from src.reports import annotate
    rec, d = _synthetic()
    crops = annotate.save_crops(rec["image"]["path"], rec["detections"],
                                os.path.join(d, "crops"))
    assert len(crops) == 2                               # both detections have bbox
    ann = annotate.annotate_image(rec["image"]["path"], rec["detections"],
                                  os.path.join(d, "annotated.png"), only_verified=True)
    assert os.path.exists(ann)


def test_per_image_folder_generation():
    from src.reports.per_image import generate_case
    rec, d = _synthetic()
    art = generate_case(rec, os.path.join(d, "examples"))
    for key in ("technical_html", "parent_en_html", "parent_ar_html",
                "psychologist_html", "analysis_json"):
        assert art[key] and os.path.exists(art[key])
    assert art["annotated"] and os.path.exists(art["annotated"])
    assert len(art["crops"]) == 2


def test_cohen_kappa_perfect_and_partial():
    from src.reports.psych_review import _cohen_kappa
    perfect = [("Agree", "Agree"), ("Disagree", "Disagree"), ("Agree", "Agree")]
    assert _cohen_kappa(perfect) == 1.0
    mixed = [("Agree", "Agree"), ("Agree", "Disagree"),
             ("Disagree", "Disagree"), ("Disagree", "Agree")]
    k = _cohen_kappa(mixed)
    assert -1.0 <= k <= 1.0


def test_fleiss_kappa_runs():
    from src.reports.psych_review import _fleiss_kappa
    # 3 raters, 2 items, perfect agreement -> kappa 1.0
    items = [{"Agree": 3}, {"Disagree": 3}]
    assert _fleiss_kappa(items) == 1.0


def test_agreement_no_reviews_is_honest():
    from src.reports.psych_review import init_master, compute_agreement
    d = tempfile.mkdtemp()
    mc = os.path.join(d, "review_master.csv")
    init_master(mc)
    res = compute_agreement(mc)
    assert res["has_reviews"] is False                   # never fabricates


def test_agreement_with_two_reviewers():
    from src.reports.psych_review import init_master, append_review_rows, compute_agreement
    d = tempfile.mkdtemp()
    mc = os.path.join(d, "review_master.csv")
    init_master(mc)
    rows = []
    for item in ("rule|R1", "rule|R2", "overall|0"):
        it, iid = item.split("|")
        rows.append({"case_id": "c1", "item_type": it, "item_id": iid,
                     "true_class": "Happy", "reviewer_id": "revA", "rating": "Agree"})
        # revB disagrees on one item
        rating = "Disagree" if iid == "R2" else "Agree"
        rows.append({"case_id": "c1", "item_type": it, "item_id": iid,
                     "true_class": "Happy", "reviewer_id": "revB", "rating": rating})
    append_review_rows(mc, rows)
    res = compute_agreement(mc)
    assert res["has_reviews"] is True
    assert res["n_reviewers"] == 2
    assert res["kappa_type"] == "cohen"
    assert "raw_agreement_pct" in res


def test_batch_generation_and_master_seed():
    from src.reports.per_image import generate_batch
    from src.reports.psych_review import load_master
    rec, d = _synthetic()
    out = generate_batch([rec], os.path.join(d, "out"))
    assert os.path.isdir(out["examples_dir"])
    assert len(out["generated"]) == 1
    master = load_master(out["master_csv"])
    # one row per reviewable item (model + 2 detections + 1 rule + overall = 5)
    assert len(master) == 5


def test_model_comparison_plot():
    from src.models.compare import _plot_comparison, DEFAULT_MODELS
    assert DEFAULT_MODELS == ["baseline", "mobilenet", "resnet18"]
    d = tempfile.mkdtemp()
    res = [{"model": "baseline", "best_val_acc": 0.42},
           {"model": "mobilenet", "best_val_acc": 0.61},
           {"model": "resnet18", "best_val_acc": 0.58}]
    _plot_comparison(d, res, "mobilenet")
    assert os.path.exists(os.path.join(d, "model_comparison.png"))
    assert os.path.exists(os.path.join(d, "model_comparison.svg"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} report tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
