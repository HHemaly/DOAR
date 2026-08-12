#!/usr/bin/env python
"""DOAR Visual Verifier development check -- NOT a benchmark. Runs the
real, independent, label-blind `GeminiVisualVerifier` against every
observer candidate already saved from the prior DOAR Gemini Observer
development run (outputs/prototype_cases/gemini_observer_dev_check_
1786540127/*_raw.json) for the five permanently DEVELOPMENT-ONLY
drawings (h38, p2b_0001..0004) -- never rerun/tuned, per this phase's
own instruction ("Do NOT rerun/tune the observer").

For each observer candidate, this builds ONE `VisualEntity` directly
(entity_id encodes a stable "observation ID", e.g. "h38_c00") -- a
dev-script-only simplification, NOT the app's real `merge_observer_
candidates_into_entities` path, because that function deliberately
MERGES same-labeled candidates (e.g. this run's 5 separate "butterfly"
observations on h38) into one enriched entity, which would silently
drop the very per-instance candidates this check needs to verify
individually. The entity's OWN field conventions (model_validation_
status="UNKNOWN", source="visual_observer", evidence_status=
"experimental_evidence_technical_view_only", ...) are copied verbatim
from that same merge function, so verification behaves identically to
how it would on a real merged entity.

Persists, per drawing, under this script's own output folder:
  - <observation_id>_crop.png       -- the exact candidate crop
  - <observation_id>_context.png    -- the padded context crop (if different)
  - <drawing_id>_annotated.png      -- the full drawing with every
                                        candidate's bbox + observation ID
                                        drawn on it, for visual inspection
                                        without reading JSON
  - <drawing_id>_verification.json  -- one row per candidate: drawing ID,
                                        observer candidate/alternatives/
                                        bbox, verifier independent label/
                                        alternatives, status, both models,
                                        provenance, runtime
  - summary.json                    -- counts across all five drawings
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_entity import VisualEntity  # noqa: E402
from doar.visual_observer import GeminiVisualVerifier, build_verifier_crops  # noqa: E402

OBSERVER_RESULTS_DIR = ROOT / "outputs" / "prototype_cases" / "gemini_observer_dev_check_1786540127"
OUT_DIR = ROOT / "outputs" / "prototype_cases" / f"gemini_verifier_dev_check_{int(time.time())}"

# Same five permanently DEVELOPMENT-ONLY images/raw-result files as the
# prior observer phase -- never the future held-out benchmark set.
DRAWINGS = [
    ("h38", ROOT / "outputs" / "prototype_cases" / "h38_1786305027" / "h38.jpg"),
    ("p2b_0001", ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0001.jpg"),
    ("p2b_0002", ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0002.jpg"),
    ("p2b_0003", ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0003.jpeg"),
    ("p2b_0004", ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0004.jpg"),
]


def _candidate_to_entity(candidate: dict, observation_id: str) -> VisualEntity:
    """Same field conventions `visual_entity.merge_observer_candidates_
    into_entities` uses for a NEW observer-only entity -- copied
    verbatim, not redesigned. See module docstring for why this builds
    one entity per candidate instead of reusing that function directly."""
    bbox = tuple(candidate["bbox"]) if candidate.get("bbox") else None
    confidence = candidate.get("confidence") if candidate.get("confidence") is not None else 0.0
    label = candidate["label"]
    alt_labels = tuple(candidate.get("alternative_labels") or ())
    return VisualEntity(
        entity_id=f"ve_{observation_id}", entity_type=candidate.get("entity_type", "unknown"),
        canonical_label=label,
        candidate_labels=((label, confidence),),
        aliases_en=alt_labels, aliases_ar=(),
        broader_categories=(), possible_subtypes=(), visual_similarities=(),
        bbox=bbox, crop_ref=None, dominant_colors=None, relative_size=None, page_position=None,
        shape_features=None, line_features=None,
        detector=f"visual_observer:{candidate.get('source_note', '')}", checkpoint="", prompt="",
        confidence=confidence, model_validation_status="UNKNOWN", case_verification_status="unreviewed",
        evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
        related_rule_ids=(), source="visual_observer", query=None,
        timestamp="2026-08-13T00:00:00+00:00",
    )


def _save_annotated_drawing(image_path: Path, rows: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    colors = {"verified": (0, 170, 0), "uncertain": (220, 150, 0), "rejected": (200, 0, 0),
              "unreviewed": (120, 120, 120)}
    for row in rows:
        bbox = row["observer_candidate"]["bbox"]
        if not bbox:
            continue
        x, y, bw, bh = bbox
        left, top, right, bottom = x * w, y * h, (x + bw) * w, (y + bh) * h
        color = colors.get(row["verification_status"], (0, 0, 0))
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left + 2, max(0, top - 14)), row["observation_id"], fill=color)
    img.save(out_path)


def _verify_drawing(drawing_id: str, image_path: Path) -> dict:
    raw_path = OBSERVER_RESULTS_DIR / f"{drawing_id}_raw.json"
    observer_result = json.loads(raw_path.read_text(encoding="utf-8"))
    candidates = observer_result["candidates"]
    observer_model = observer_result["model"]

    verifier = GeminiVisualVerifier()
    rows = []
    for i, candidate in enumerate(candidates):
        observation_id = f"{drawing_id}_c{i:02d}"
        entity = _candidate_to_entity(candidate, observation_id)

        crop, context_crop = build_verifier_crops(str(image_path), entity.bbox)
        if crop is not None:
            crop.save(OUT_DIR / f"{observation_id}_crop.png")
        if context_crop is not None and context_crop is not crop:
            context_crop.save(OUT_DIR / f"{observation_id}_context.png")

        t0 = time.monotonic()
        result = verifier.verify(str(image_path), entity)
        runtime = time.monotonic() - t0
        time.sleep(3.0)  # stay well under the free-tier per-minute rate limit across ~23 calls

        rows.append({
            "drawing_id": drawing_id,
            "observation_id": observation_id,
            "observer_candidate": {
                "label": candidate["label"], "alternative_labels": candidate.get("alternative_labels", []),
                "bbox": candidate.get("bbox"), "confidence": candidate.get("confidence"),
                "entity_type": candidate.get("entity_type"),
            },
            "verifier_independent_label": result.independent_label,
            "verifier_independent_alternative_labels": list(result.independent_alternative_labels),
            "verifier_confidence": result.confidence,
            "verification_status": result.status,
            "verifier_notes": result.notes,
            "observer_model": observer_model,
            "verifier_model": verifier.model,
            "verifier_source_note": result.source_note,
            "runtime_seconds": round(runtime, 2),
        })
        print(f"      [{observation_id}] observer={candidate['label']!r} -> "
              f"verifier={result.independent_label!r} -> {result.status} ({runtime:.2f}s)", flush=True)

    _save_annotated_drawing(image_path, rows, OUT_DIR / f"{drawing_id}_annotated.png")
    (OUT_DIR / f"{drawing_id}_verification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"drawing_id": drawing_id, "candidate_count": len(rows),
            "verified": sum(1 for r in rows if r["verification_status"] == "verified"),
            "uncertain": sum(1 for r in rows if r["verification_status"] == "uncertain"),
            "rejected": sum(1 for r in rows if r["verification_status"] == "rejected"),
            "unreviewed": sum(1 for r in rows if r["verification_status"] == "unreviewed")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"verifier model: {GeminiVisualVerifier().model}", flush=True)
    summaries = []
    for i, (drawing_id, image_path) in enumerate(DRAWINGS, start=1):
        assert image_path.exists(), f"missing fixture image: {image_path}"
        print(f"[{i}/{len(DRAWINGS)}] {drawing_id}...", flush=True)
        summaries.append(_verify_drawing(drawing_id, image_path))

    totals = {
        "note": ("Development check only -- NOT the DOAR thesis benchmark. Verifier ran on the "
                 "OBSERVER results already saved from the prior (unmodified) Gemini observer run; "
                 "the observer itself was never rerun or retuned."),
        "observer_results_source": str(OBSERVER_RESULTS_DIR),
        "per_drawing": summaries,
        "total_candidates": sum(s["candidate_count"] for s in summaries),
        "total_verified": sum(s["verified"] for s in summaries),
        "total_uncertain": sum(s["uncertain"] for s in summaries),
        "total_rejected": sum(s["rejected"] for s in summaries),
        "total_unreviewed": sum(s["unreviewed"] for s in summaries),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(f"\nSaved all crops/annotated images/results to: {OUT_DIR}", flush=True)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
