"""Tests for the journal Stage C modules (foreground, metrics, risk-coverage, features)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doar_journal import foreground, measurement, risk_coverage, seg_metrics  # noqa: E402
from doar_journal import _frozen_e2_features as fe  # noqa: E402
from doar_journal import _frozen_r1v1 as r1  # noqa: E402
from doar_journal import _frozen_v2 as v2  # noqa: E402


def synthetic_drawing(h=240, w=320, seed=0):
    """Off-white paper with a dark stick figure and a red filled disc."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (236, 232, 222), np.uint8)
    img = np.clip(img.astype(int) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    ref = np.zeros((h, w), np.uint8)
    for canvas, col in ((img, (40, 40, 40)), (ref, 1)):
        cv2.circle(canvas, (160, 60), 25, col if canvas is img else 1, 3)
        cv2.line(canvas, (160, 85), (160, 170), col if canvas is img else 1, 3)
        cv2.line(canvas, (120, 120), (200, 120), col if canvas is img else 1, 3)
        cv2.line(canvas, (160, 170), (130, 220), col if canvas is img else 1, 3)
        cv2.line(canvas, (160, 170), (190, 220), col if canvas is img else 1, 3)
    cv2.circle(img, (260, 60), 20, (200, 30, 30), -1)
    cv2.circle(ref, (260, 60), 20, 1, -1)
    return img, ref.astype(bool)


@pytest.fixture(scope="module")
def drawing(tmp_path_factory):
    from PIL import Image
    img, ref = synthetic_drawing()
    p = tmp_path_factory.mktemp("img") / "draw.png"
    Image.fromarray(img).save(p)
    return p, img, ref


def test_all_methods_run_and_find_marks(drawing):
    _, img, ref = drawing
    for name, fn in foreground.METHODS.items():
        r = fn(img)
        assert r.mask.shape == ref.shape and r.mask.dtype == bool, name
        m = seg_metrics.all_metrics(r.mask, ref)
        assert m["recall"] > 0.5, (name, m)
    meta = foreground.v2(img).meta
    assert meta["measurement_policy"] in ("MASK_USABLE", "MASK_REVIEW_REQUIRED", "MASK_UNRELIABLE")


def test_a0_definitions_agree_across_archived_cells(drawing):
    _, img, _ = drawing
    m20 = fe.build_foreground_mask(img)[0] > 0
    m22 = r1.build_a0_mask(img)["mask"] > 0
    m3 = v2.build_a0_mask(img)["mask"] > 0
    assert np.array_equal(m20, m3) and np.array_equal(m22, m3)


def test_injectable_extractor_reproduces_archive(drawing):
    p, img, _ = drawing
    original = measurement.archived_features(p)
    a0_mask = fe.build_foreground_mask(img)[0]
    injected = measurement.features_with_mask(p, a0_mask)
    fo, fi = measurement.numeric_features(original), measurement.numeric_features(injected)
    assert fo.keys() == fi.keys()
    for k in fo:
        assert (np.isnan(fo[k]) and np.isnan(fi[k])) or fo[k] == pytest.approx(fi[k]), k
    assert len(fo) >= 40  # the archive reports 51 numeric columns incl. two mask-metadata ones


def test_mask_dependence_detects_mask_free_features(drawing):
    p, img, ref = drawing
    dep = measurement.mask_dependence(p, ref, np.zeros_like(ref))
    assert dep["foreground_fraction"] is True
    assert dep["canvas_width_px"] is False


def test_seg_metrics_edge_cases():
    z = np.zeros((10, 10), bool)
    assert seg_metrics.region_metrics(z, z)["both_empty"]
    a = z.copy(); a[2:5, 2:5] = True
    m = seg_metrics.all_metrics(a, a)
    assert m["iou"] == 1.0 and m["boundary_f1"] == 1.0
    assert seg_metrics.region_metrics(a, z)["f1"] == 0.0


def test_risk_coverage_and_auroc():
    scores = [2, 2, 1, 0]
    errors = [0.0, 0.1, 0.5, 0.9]
    rc = risk_coverage.risk_coverage(scores, errors)
    assert rc["aurc"] < rc["aurc_random"] and rc["aurc_oracle"] <= rc["aurc"]
    assert risk_coverage.auroc_bad(scores, errors, 0.3) == 1.0
    tied = risk_coverage.risk_coverage([1, 1, 1, 1], errors)
    assert tied["aurc"] == pytest.approx(tied["aurc_random"])


def test_c3_c4_pipeline_smoke(tmp_path):
    """End-to-end run of C3 and C4 on a synthetic batch (test data only)."""
    import csv
    import json
    import subprocess
    from PIL import Image
    batch = tmp_path / "batch"
    (batch / "images").mkdir(parents=True)
    (batch / "masks").mkdir()
    (batch / "records").mkdir()
    key = []
    for i in range(6):
        img, ref = synthetic_drawing(seed=i)
        t = f"R{i + 1:03d}"
        Image.fromarray(img).save(batch / "images" / f"{t}.png")
        assess = "ASSESSABLE" if i < 5 else "NOT_ASSESSABLE_FULL_CANVAS"
        for who in ("AA", "BB"):
            Image.fromarray((ref * 255).astype(np.uint8)).save(batch / "masks" / f"{t}__{who}.png")
            (batch / "records" / f"{t}__{who}.json").write_text(json.dumps({"task_id": t, "assessability": assess}))
        key.append({"task_id": t, "image_id": f"x{i}", "class": "Happy", "a0_state": "OK"})
    with open(batch / "key_PRIVATE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    root = Path(__file__).resolve().parents[1]
    scripts = root / "experiments/journal/scripts"
    subprocess.run([sys.executable, scripts / "c3_evaluate_references.py", "--batch", batch, "--primary", "AA",
                    "--secondary", "BB", "--out", tmp_path / "c3"], check=True, capture_output=True)
    ir = json.loads((tmp_path / "c3" / "inter_rater.json").read_text())
    assert ir["mask_f1_median"] == 1.0 and ir["assessability_agreement"] == 1.0
    r = subprocess.run([sys.executable, scripts / "c4_measurement_analysis.py", "--c3", tmp_path / "c3",
                        "--out", tmp_path / "c4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "c4" / "F_risk_coverage.pdf").exists()


def test_annotation_proposal_and_strokes():
    from doar_journal import annotation as ann
    img, ref = synthetic_drawing()
    prop = ann.proposal(img, [(10, 10), (300, 230)], threshold=25)
    m = seg_metrics.region_metrics(prop, ref)
    assert m["recall"] > 0.8 and m["precision"] > 0.8
    rgba = np.zeros((120, 160, 4), np.uint8)
    rgba[0:5, 0:5] = (0, 255, 0, 255)      # add stroke
    rgba[50:55, 50:55] = (255, 0, 0, 255)  # erase stroke
    add, erase = ann.strokes_from_canvas(rgba, (240, 320))
    assert add[:10, :10].all() and erase[100:110, 100:110].all()
    final = ann.combine(prop, add, erase)
    assert final[:10, :10].all() and not final[100:110, 100:110].any()
    assert ann.working_size(2000, 1000) == (1024, 512)
