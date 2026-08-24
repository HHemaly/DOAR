"""Phase 2C.5 part-box proposal generation -- reuses Phase 2C.4's frozen,
unmodified detector loading pattern (src/doar/phase2c4/detectors.py) but
targets the 5-part vocabulary (ontology.PART_TARGETS) instead of the
10-class object ontology, and additionally records each detection's
BOUNDING BOX (Phase 2C.4 only needed presence/score, never boxes).

Mirrors phase2c4/detectors.py's injectable-backend pattern exactly:
`load_real_*` functions are the only code that touches `transformers`, are
NEVER called by the automated test suite, and every prompt/threshold is a
named, documented constant -- reusing Phase 2C.4's OWN already-frozen
operating points (grounding_dino threshold=0.25/0.25, owlv2 threshold=0.1)
rather than inventing new ones, per instruction to use predeclared
settings and not prompt-engineer after seeing results.

A proposal produced here is NEVER ground truth -- `build_part_proposals`
always returns `PartInstance`s with `bbox_source="model_proposed"`; only
explicit human review (the annotation app, phase2c5/app.py) can change
that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..phase2c4.calibration import GROUNDING_DINO_ORIGINAL_THRESHOLD, OWLV2_ORIGINAL_THRESHOLD
from ..phase2c4.detectors import grounding_dino_text_labels
from .ontology import PART_TARGETS
from .schema import PartInstance

PredictFn = Callable[[str], "list[RawBoxDetection]"]


@dataclass(frozen=True)
class RawBoxDetection:
    label: str
    score: float
    bbox_xywh_normalized: tuple[float, float, float, float]


PART_PHRASES: dict[str, str] = {t: t for t in PART_TARGETS}  # bare noun, e.g. "eye", "mouth"


def owlv2_part_text_queries(targets: tuple[str, ...] = PART_TARGETS) -> list[str]:
    """Identical template to phase2c4.detectors.owlv2_text_queries -- 'a {noun}'."""
    return [f"a {PART_PHRASES[t]}" for t in targets]


def grounding_dino_part_text_prompt(targets: tuple[str, ...] = PART_TARGETS) -> str:
    """Identical format to phase2c4.detectors.grounding_dino_text_prompt --
    lowercase phrases separated by '. ', each ending in a period."""
    return ". ".join(PART_PHRASES[t] for t in targets) + "."


def make_label_to_target(targets: tuple[str, ...] = PART_TARGETS) -> Callable[[str], list[str]]:
    """Same whole-word regex containment matching as
    phase2c4.detectors.make_default_label_to_classes, for the same reason:
    a detector can merge adjacent phrase matches into one label string."""
    patterns = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in targets}

    def label_to_targets(label: str) -> list[str]:
        return [t for t, pattern in patterns.items() if pattern.search(label)]

    return label_to_targets


def build_part_proposals(raw_detections: list[RawBoxDetection], *, model_name: str,
                          checkpoint: str, prompt: str, threshold: float, timestamp: str,
                          targets: tuple[str, ...] = PART_TARGETS,
                          label_to_targets: Callable[[str], list[str]] | None = None,
                          ) -> dict[str, list[PartInstance]]:
    """{target_name: [PartInstance, ...]} -- every instance carries
    bbox_source='model_proposed' and full provenance. No deduplication /
    NMS across overlapping detections is attempted here (out of scope for
    a proposal that a human will review); instance_index is assigned in
    detection order per target."""
    label_to_targets = label_to_targets or make_label_to_target(targets)
    by_target: dict[str, list[PartInstance]] = {t: [] for t in targets}
    for d in raw_detections:
        for target in label_to_targets(d.label):
            if target not in by_target:
                continue
            idx = len(by_target[target])
            by_target[target].append(PartInstance(
                instance_index=idx, bbox=d.bbox_xywh_normalized, bbox_source="model_proposed",
                proposal_model=model_name, proposal_checkpoint=checkpoint,
                proposal_prompt=prompt, proposal_threshold=threshold, proposal_timestamp=timestamp,
            ))
    return by_target


def _xyxy_pixels_to_xywh_normalized(box_xyxy, image_width: float, image_height: float
                                     ) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0.0, min(x0, image_width)), max(0.0, min(x1, image_width))
    y0, y1 = max(0.0, min(y0, image_height)), max(0.0, min(y1, image_height))
    return (x0 / image_width, y0 / image_height,
            max(1e-6, (x1 - x0) / image_width), max(1e-6, (y1 - y0) / image_height))


# ---------------------------------------------------------------------------
# Real-weight loaders. Never called by the test suite.
# ---------------------------------------------------------------------------

def load_real_grounding_dino_parts(*, model_id: str = "IDEA-Research/grounding-dino-tiny",
                                    threshold: float = GROUNDING_DINO_ORIGINAL_THRESHOLD,
                                    text_threshold: float = GROUNDING_DINO_ORIGINAL_THRESHOLD,
                                    device: str = "cpu"):
    """Same checkpoint and same thresholds as Phase 2C.4's frozen
    grounding_dino benchmark (src/doar/phase2c4/calibration.py's own
    constants) -- only the prompt vocabulary changes (5 part nouns instead
    of the 10 object classes), per instruction to reuse predeclared
    settings rather than invent new ones for this pilot."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    text = grounding_dino_part_text_prompt()

    def predict_fn(image_path: str) -> list[RawBoxDetection]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[img.size[::-1]])[0]
        w, h = img.size
        return [RawBoxDetection(label=label, score=float(score),
                                 bbox_xywh_normalized=_xyxy_pixels_to_xywh_normalized(box, w, h))
                for label, score, box in zip(grounding_dino_text_labels(results), results["scores"], results["boxes"])]

    return predict_fn, {"model_name": f"grounding_dino:{model_id}", "checkpoint": model_id,
                         "prompt": text, "threshold": threshold}


def load_real_owlv2_parts(*, model_id: str = "google/owlv2-base-patch16-ensemble",
                           threshold: float = OWLV2_ORIGINAL_THRESHOLD, device: str = "cpu"):
    """Same checkpoint and same threshold as Phase 2C.4's frozen owlv2
    benchmark -- only the prompt vocabulary changes."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    import torch

    processor = Owlv2Processor.from_pretrained(model_id, use_fast=False)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    texts = [owlv2_part_text_queries()]

    def predict_fn(image_path: str) -> list[RawBoxDetection]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = processor(text=texts, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=texts)[0]
        w, h = img.size
        return [RawBoxDetection(label=label, score=float(score),
                                 bbox_xywh_normalized=_xyxy_pixels_to_xywh_normalized(box, w, h))
                for label, score, box in zip(results["text_labels"], results["scores"], results["boxes"])]

    return predict_fn, {"model_name": f"owlv2:{model_id}", "checkpoint": model_id,
                         "prompt": " | ".join(owlv2_part_text_queries()), "threshold": threshold}
