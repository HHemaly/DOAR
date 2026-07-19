"""
tests/test_pipeline.py — critical-component tests + full-pipeline smoke test.

Run with:  python -m pytest tests/ -v      (or)   python tests/test_pipeline.py

These tests do NOT require the dataset, torch, or a network. They use a
synthetic drawing created on the fly, so they run anywhere (CI, laptop, Colab).
"""

from __future__ import annotations
import os
import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_synthetic_drawing(path: str):
    """Create a simple synthetic child-like drawing (sun + figure + ground)."""
    import numpy as np
    import cv2
    img = np.full((400, 400, 3), 255, np.uint8)          # white paper
    cv2.circle(img, (320, 80), 40, (0, 200, 255), -1)    # yellow sun
    cv2.line(img, (200, 180), (200, 300), (0, 0, 0), 4)  # body
    cv2.circle(img, (200, 150), 30, (0, 0, 0), 3)        # head
    cv2.line(img, (200, 220), (160, 260), (0, 0, 0), 3)  # arm
    cv2.line(img, (200, 220), (240, 260), (0, 0, 0), 3)  # arm
    cv2.line(img, (0, 340), (400, 340), (0, 150, 0), 4)  # ground
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dataset_discovery():
    from src.data.inspect_dataset import discover_images, discover_classes
    with tempfile.TemporaryDirectory() as d:
        for cls in ("Happy", "Sad"):
            os.makedirs(os.path.join(d, cls))
            _make_synthetic_drawing(os.path.join(d, cls, "a.png"))
        recs = discover_images(d)
        assert len(recs) == 2
        assert discover_classes(recs) == ["Happy", "Sad"]


def test_image_loading_and_features():
    from pipeline import _extract_features
    with tempfile.TemporaryDirectory() as d:
        p = _make_synthetic_drawing(os.path.join(d, "draw.png"))
        doc = _extract_features(p)
        assert doc["image_quality"]["width_px"] == 400
        assert "empty_space_ratio" in doc["composition_features"]
        assert 0.0 <= doc["composition_features"]["empty_space_ratio"] <= 1.0


def test_claim_building():
    from pipeline import _extract_features
    from claim_builder import build_all_claims
    with tempfile.TemporaryDirectory() as d:
        p = _make_synthetic_drawing(os.path.join(d, "draw.png"))
        doc = _extract_features(p)
        claims = build_all_claims(doc)
        assert isinstance(claims, list)
        assert all("claim_id" in c and "claim_type" in c for c in claims)


def test_numeric_validation():
    from numeric_validator import find_ungrounded_numbers
    doc = {"composition_features": {"empty_space_ratio": 0.58}}
    # A false "99% empty" claim must be flagged as a mismatch
    violations = find_ungrounded_numbers("The drawing is 99% empty.", doc)
    assert any(v.get("verdict") == "mismatch" for v in violations)


def test_safety_language_blocking():
    from safety_policy import check_for_diagnostic_language
    bad = "This drawing proves that the child has depression."
    violations = check_for_diagnostic_language(bad)
    assert any(v["severity"] == "BLOCK" for v in violations)
    # "not diagnostic" must NOT be blocked
    good = "This indicator is not diagnostic on its own."
    assert not any(v["severity"] == "BLOCK"
                   for v in check_for_diagnostic_language(good))


def test_missing_optional_dependency_graceful():
    # visual_claim_validator must degrade to 'uncertain' without a CLIP model
    from visual_claim_validator import validate_visual_claim
    claim = {"claim_type": "visual_object", "evidence": {"label": "sun"},
             "confidence": 0.5}
    out = validate_visual_claim(claim, "nonexistent.png",
                                clip_model=None, clip_preprocess=None, clip_tokenize=None)
    assert out["validator_status"] == "uncertain"


def test_leak_safe_split():
    from src.data.split import make_split
    import csv
    with tempfile.TemporaryDirectory() as d:
        # Build a fake dataset_summary.csv with duplicate md5 across two files
        summary = os.path.join(d, "summary.csv")
        rows = []
        for cls in ("Happy", "Sad"):
            for i in range(10):
                rows.append({"path": f"/x/{cls}/{i}.png", "class": cls,
                             "filename": f"{i}.png", "md5": f"{cls}{i}",
                             "phash": f"{i:016x}"})
        with open(summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "class", "filename", "md5", "phash"])
            w.writeheader(); w.writerows(rows)
        meta = make_split(summary, d, seed=1)
        assert meta["leakage_ok"] is True
        assert set(meta["split_totals"]) <= {"train", "val", "test"}


def test_full_pipeline_smoke():
    """The headline smoke test: one image passes through the whole v2 pipeline."""
    from pipeline import run_full_pipeline_v2
    with tempfile.TemporaryDirectory() as d:
        p = _make_synthetic_drawing(os.path.join(d, "smoke.png"))
        result = run_full_pipeline_v2(p, parent_question="What is in the drawing?",
                                      run_ocr=False, run_arabic=False,
                                      session_dir=d)
        assert result["pipeline_version"] == "v2"
        assert result["final_judgment"]["final_answer_status"] in (
            "PASS", "REWRITE_REQUIRED", "BLOCK")
        assert "parent_answer" in result["parent_facing_output"]
        # Disclaimer must always be present
        assert "not diagnostic" in (
            result["parent_facing_output"].get("disclaimer", "") +
            result["parent_facing_output"].get("safety_note", "")
        ).lower()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow running without pytest
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
