"""Phase 2C.7 / DOAR MVP real-model runtime wiring.

Single shared place that turns the frozen `visual_detector_policy.json`
into loaded, callable models -- used by both `scripts/phase2c7_run_sanity_
check.py` and the DOAR MVP app (`doar_prototype_app.py`), so the routing
logic (which model answers which target, including the composite
"grounding_dino_parts+owlv2_parts_fallback" eye config) exists in exactly
one place. Moved here, unchanged, from the sanity-check script during the
DOAR MVP build so the app can reuse it without duplicating routing logic.

Also adds `load_open_vocab_query_fn`, a dynamic single-query variant of
`phase2c4.detectors.load_real_grounding_dino` for the MVP's broad-scan
extras and on-demand Q&A visual search -- same checkpoint, same frozen
threshold, just a caller-supplied text prompt instead of the fixed
10-class vocabulary. Not a new detector family.

Never imported by the automated test suite -- every `load_real_*`/
`load_open_vocab_query_fn` call here touches real model weights (mirrors
every other `load_real_*` module in this project).
"""
from __future__ import annotations

from typing import Callable

from ..phase2c4 import detectors as det4_mod
from ..phase2c4.calibration import GROUNDING_DINO_ORIGINAL_THRESHOLD
from ..phase2c5 import proposals as prop5_mod
from ..phase2c5.ontology import PART_TARGETS


def object_class_predict_fn(detector) -> Callable[[str], dict]:
    """`detector.predict_image` (phase2c4.detectors.OpenVocabDetector)
    already returns {class_name: (detected, score)} -- no bbox is tracked
    for the object-class pathway (Phase 2C.4 never needed one), so this
    reports bbox=None honestly rather than fabricating one."""
    def predict(image_path: str) -> dict:
        presence = detector.predict_image(image_path)
        return {cls: (detected, score, None) for cls, (detected, score) in presence.items()}
    return predict


def part_predict_fn(raw_predict_fn) -> Callable[[str], dict]:
    """Wraps a phase2c5.proposals raw predict_fn (returns
    list[RawBoxDetection]) directly -- bypasses `build_part_proposals`
    (which builds annotation-schema `PartInstance`s with no confidence
    field, the wrong shape for live evidence) so the real per-detection
    confidence score is preserved in the output."""
    label_to_targets = prop5_mod.make_label_to_target(PART_TARGETS)

    def predict(image_path: str) -> dict:
        raw = raw_predict_fn(image_path)
        best: dict[str, tuple[bool, float, tuple]] = {t: (False, 0.0, None) for t in PART_TARGETS}
        for d in raw:
            for target in label_to_targets(d.label):
                if target in best and d.score > best[target][1]:
                    best[target] = (True, d.score, d.bbox_xywh_normalized)
        return best
    return predict


def combined_fallback_predict_fn(primary_predict, fallback_predict) -> Callable[[str], dict]:
    """Reproduces the exact deployed eye policy (Phase 2C.6/2C.7): primary
    result per target, falling back to the secondary model ONLY for
    targets the primary found nothing for -- same semantics as
    src/doar/phase2c6/proposal_batch.py::process_one_image, just at
    single-image, live-inference granularity instead of a batch CSV."""
    def predict(image_path: str) -> dict:
        primary = primary_predict(image_path)
        fallback = fallback_predict(image_path)
        merged = dict(primary)
        for target, (detected, _score, _bbox) in primary.items():
            if not detected and fallback.get(target, (False, 0.0, None))[0]:
                merged[target] = fallback[target]
        return merged
    return predict


def build_real_model_predict_fns(eye_best_model: str) -> dict:
    """Loads exactly the real models the frozen policy actually needs --
    never more. `eye_best_model` is one of 'owlv2_parts',
    'grounding_dino_parts', or the composite
    'grounding_dino_parts+owlv2_parts_fallback' (Phase 2C.7's actual
    frozen eye config -- Grounding DINO primary, OWLv2 fallback for
    targets Grounding DINO missed)."""
    fns = {}

    gd_object = det4_mod.load_real_grounding_dino()
    fns["grounding_dino_object_classes"] = object_class_predict_fn(gd_object)

    owl_object = det4_mod.load_real_owlv2()
    fns["owlv2_object_classes"] = object_class_predict_fn(owl_object)

    needs_gd_parts = "grounding_dino_parts" in eye_best_model
    needs_owl_parts = "owlv2_parts" in eye_best_model
    gd_parts_fn = owl_parts_fn = None
    if needs_gd_parts:
        gd_parts_predict, _meta = prop5_mod.load_real_grounding_dino_parts()
        gd_parts_fn = part_predict_fn(gd_parts_predict)
    if needs_owl_parts:
        owl_parts_predict, _meta = prop5_mod.load_real_owlv2_parts()
        owl_parts_fn = part_predict_fn(owl_parts_predict)

    if "+" in eye_best_model and gd_parts_fn and owl_parts_fn:
        fns[eye_best_model] = combined_fallback_predict_fn(gd_parts_fn, owl_parts_fn)
    elif gd_parts_fn:
        fns[eye_best_model] = gd_parts_fn
    elif owl_parts_fn:
        fns[eye_best_model] = owl_parts_fn
    return fns


def _xyxy_pixels_to_xywh_normalized(box_xyxy, image_width: float, image_height: float
                                     ) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0.0, min(x0, image_width)), max(0.0, min(x1, image_width))
    y0, y1 = max(0.0, min(y0, image_height)), max(0.0, min(y1, image_height))
    return (x0 / image_width, y0 / image_height,
            max(1e-6, (x1 - x0) / image_width), max(1e-6, (y1 - y0) / image_height))


def load_open_vocab_query_fn(*, model_id: str = "IDEA-Research/grounding-dino-tiny",
                              threshold: float = GROUNDING_DINO_ORIGINAL_THRESHOLD,
                              text_threshold: float = GROUNDING_DINO_ORIGINAL_THRESHOLD,
                              device: str = "cpu") -> Callable[[str, str], list[tuple]]:
    """Dynamic single-query variant of phase2c4.detectors.load_real_
    grounding_dino: same checkpoint, same frozen thresholds, but the text
    prompt is built from the caller's `target` string at CALL time instead
    of the fixed 10-class vocabulary -- lets one loaded model answer any
    query (broad-scan extras, on-demand Q&A search) without reloading per
    query. Returns real boxes (the fixed-vocabulary object-class loader
    discards them -- Phase 2C.4 never needed them for that pathway).

    Return shape matches visual_evidence.py's expected 6-tuple:
    (detected, confidence, bbox, model_name, checkpoint, prompt)."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def query(image_path: str, target: str) -> list[tuple]:
        from PIL import Image
        text = f"{target.strip().lower()}."
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[img.size[::-1]])[0]
        w, h = img.size
        return [(True, float(score), _xyxy_pixels_to_xywh_normalized(box, w, h),
                  f"grounding_dino_open_query:{model_id}", model_id, text)
                for score, box in zip(results["scores"], results["boxes"])]

    return query
