"""Adapter: real existing extractors -> canonical evidence schema (Phase 1).

Wires the outputs already produced by `analysis.py::analyze_image`
(quality, segmentation, composition, colour) into `EvidenceItem`s. This
module does not compute anything new -- Phase 1 Section 4 requires
validating and reusing existing extractors before introducing new ones,
and these are the only extractors currently proven against real drawings
(`END_TO_END_INFERENCE_TRACE.md`).

It also emits explicit `unavailable` evidence items for every canonical
evidence category with no working extractor (semantic objects, object
components/body parts, primitive-geometry semantics, spatial
relationships, OCR/text, corrections/erasures) -- required so the
canonical schema honestly represents the full category list from Phase 1
Section 1, rather than silently omitting categories that have no data.
"""

from __future__ import annotations

from typing import Any

from .evidence_schema import EvidenceItem, EvidenceLocation, EvidenceSet, unavailable_item

# Segmentation-confidence threshold below which composition/colour evidence
# is reported as `experimental` rather than `available` -- mirrors the
# existing convention in schemas.py (segmentation.status: "verified" if
# confidence >= 0.6 else "uncertain").
SEG_CONFIDENCE_AVAILABLE_THRESHOLD = 0.6

# feature_id -> (category, human extractor label) for every canonical
# category that has no working extractor today. Kept centralized so a
# future real extractor can be removed from this list in one place instead
# of being newly wired throughout the codebase.
UNIMPLEMENTED_FEATURES: dict[str, dict[str, str]] = {
    "semantic.face.eyes": {"category": "semantic_objects", "reason": "No face/eye detector is implemented, adopted, or validated in this release."},
    "semantic.object.animal_species": {"category": "semantic_objects", "reason": "No animal-species classifier is implemented, adopted, or validated in this release."},
    "semantic.object.symbol": {"category": "semantic_objects", "reason": "No symbol (star/heart/flower/cloud/sun) classifier is implemented, adopted, or validated in this release."},
    "semantic.object.category": {"category": "semantic_objects", "reason": "No general object-category classifier is implemented, adopted, or validated in this release."},
    "geometry.primitive_shape": {"category": "primitive_geometry", "reason": "OpenCV-style contour extraction exists for stroke maps, but shape-semantic classification (circle vs. rectangle vs. line) is not wired to any evidence output."},
    "geometry.shape_repetition": {"category": "primitive_geometry", "reason": "Shape-repetition counting requires shape-semantic classification, which does not exist yet (features.py's shape.* fields are honestly NaN for this reason)."},
    "components.body_parts": {"category": "object_components_body_parts", "reason": "No object-component/body-part detector is implemented in this release; detectors/ is schema-only scaffolding."},
    "spatial.relationships": {"category": "spatial_relationships", "reason": "Spatial relationships between detected objects require object detection, which does not exist yet."},
    "ocr.text_regions": {"category": "ocr_text", "reason": "OCR is not wired into analyze_image in this release (self-disclosed in streamlit_app.py's Objects/OCR tabs)."},
    "corrections.erasures": {"category": "corrections_erasures", "reason": "No erasure/overwriting/correction detector is implemented in this release."},
}


def _bbox_to_xywh(bounding_box: list[int] | None, width: int, height: int) -> tuple[float, float, float, float] | None:
    if bounding_box is None or not width or not height:
        return None
    x0, y0, x1, y1 = bounding_box
    return (x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height)


def build_evidence_set(analysis: dict[str, Any], artifacts: dict[str, Any] | None = None) -> EvidenceSet:
    """Build a canonical EvidenceSet from a real `analysis.json`-shaped
    dict (as produced by `analyze_image(...).to_dict()`), or an equivalent
    dict with `quality`/`segmentation`/`composition`/`colour`/`artifacts`
    keys. `artifacts` may be passed separately if the caller already
    resolved relative paths."""
    quality = analysis["quality"]
    segmentation = analysis["segmentation"]
    composition = analysis["composition"]
    colour = analysis["colour"]
    art = artifacts if artifacts is not None else analysis.get("artifacts", {})
    overlay_ref = art.get("feature_overlay")
    normalized_ref = art.get("normalized_image")

    width, height = quality["width"], quality["height"]
    quality_status = quality.get("quality_status", "supported")
    seg_conf = float(segmentation.get("confidence", 0.0))

    items: list[EvidenceItem] = []

    items.append(EvidenceItem(
        evidence_id="ev2_quality_gate",
        feature_id="quality.overall_status",
        value=quality_status,
        unit=None,
        status="available",
        confidence=1.0,
        extractor="analysis.py::_quality",
        extractor_version="v1",
        validation_status="thresholds_not_empirically_validated",
        location=None,
        visualization_reference=normalized_ref,
        reason="Deterministic resolution/blur/contrast gate computed directly from the image.",
        limitations=["Thresholds are engineering defaults, not calibrated against this dataset (quality.thresholds_validated_on_real_dataset=false)."],
        category="image_page_quality",
    ))

    if quality_status == "unsupported":
        reason = "Image quality gate marked this image unsupported: " + "; ".join(quality.get("unsupported_reasons", []))
        items.append(EvidenceItem(
            evidence_id="ev2_bbox_coverage", feature_id="composition.bounding_box_coverage",
            value=None, unit="ratio", status="insufficient_evidence", confidence=None,
            extractor="analysis.py::_segment+_composition", extractor_version="v1",
            validation_status="not_applicable", location=None, visualization_reference=overlay_ref,
            reason=reason, limitations=[], category="global_composition",
        ))
        items.append(EvidenceItem(
            evidence_id="ev2_centroid", feature_id="composition.centroid_normalized",
            value=None, unit="normalized_fraction", status="insufficient_evidence", confidence=None,
            extractor="analysis.py::_segment+_composition", extractor_version="v1",
            validation_status="not_applicable", location=None, visualization_reference=overlay_ref,
            reason=reason, limitations=[], category="size_placement",
        ))
    else:
        conclusive_status = "available" if seg_conf >= SEG_CONFIDENCE_AVAILABLE_THRESHOLD else "experimental"
        bbox = composition.get("bounding_box")
        bbox_xywh = _bbox_to_xywh(bbox, width, height)
        items.append(EvidenceItem(
            evidence_id="ev2_bbox_coverage", feature_id="composition.bounding_box_coverage",
            value=composition["bounding_box_coverage"], unit="ratio", status=conclusive_status,
            confidence=seg_conf, extractor="analysis.py::_segment+_composition", extractor_version="v1",
            validation_status="thresholds_partially_sourced",
            location=EvidenceLocation(region="bounding_box", bbox_xywh=bbox_xywh),
            visualization_reference=overlay_ref,
            reason="Fraction of the drawing's bounding box relative to the full page, from the foreground/background segmentation mask.",
            limitations=["Segmentation is a foreground/background split, not object-aware; it cannot distinguish multiple figures from one large figure."],
            category="global_composition",
        ))
        centroid = composition.get("centroid_normalized")
        if centroid is None:
            items.append(EvidenceItem(
                evidence_id="ev2_centroid", feature_id="composition.centroid_normalized",
                value=None, unit="normalized_fraction", status="insufficient_evidence", confidence=None,
                extractor="analysis.py::_segment+_composition", extractor_version="v1",
                validation_status="not_applicable", location=None, visualization_reference=overlay_ref,
                reason="No foreground pixels were detected; centroid is undefined for a blank mask.",
                limitations=[], category="size_placement",
            ))
        else:
            cx, cy = float(centroid[0]), float(centroid[1])
            items.append(EvidenceItem(
                evidence_id="ev2_centroid", feature_id="composition.centroid_normalized",
                value=[cx, cy], unit="normalized_fraction", status=conclusive_status,
                confidence=seg_conf, extractor="analysis.py::_segment+_composition", extractor_version="v1",
                validation_status="threshold_invented_no_source_anchor",
                location=EvidenceLocation(region="centroid", centroid_xy=(cx, cy)),
                visualization_reference=overlay_ref,
                reason="Normalized centroid of the foreground mask (intensity-weighted mean pixel position).",
                limitations=["The top/left/right split boundaries (0.4/0.6) used by placement rules have no numeric support in the source PDF."],
                category="size_placement",
            ))

    if quality_status == "unsupported":
        items.append(EvidenceItem(
            evidence_id="ev2_dominant_colour", feature_id="colour.dominant",
            value=None, unit=None, status="insufficient_evidence", confidence=None,
            extractor="analysis.py::_colour", extractor_version="v1",
            validation_status="not_applicable", location=None, visualization_reference=overlay_ref,
            reason="Suppressed by the quality gate.", limitations=[], category="colour",
        ))
    else:
        items.append(EvidenceItem(
            evidence_id="ev2_dominant_colour", feature_id="colour.dominant",
            value=colour.get("dominant_colour"), unit=None, status="available",
            confidence=seg_conf, extractor="analysis.py::_colour", extractor_version="v1",
            validation_status="qualitative_only", location=None, visualization_reference=overlay_ref,
            reason="Coarse colour-bin classification of foreground pixels.",
            limitations=["Colour naming is a coarse fixed-bin heuristic, not perceptual/calibrated colour classification."],
            category="colour",
        ))

    for idx, (feature_id, info) in enumerate(UNIMPLEMENTED_FEATURES.items()):
        items.append(unavailable_item(
            evidence_id=f"ev2_unavailable_{idx:02d}",
            feature_id=feature_id,
            category=info["category"],
            reason=info["reason"],
        ))

    return EvidenceSet(items=items)
