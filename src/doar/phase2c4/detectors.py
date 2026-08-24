"""Phase 2C.4 open-vocabulary detector wrappers -- OWLv2, Grounding DINO,
YOLO-World, Florence-2.

Mirrors src/doar/phase2b/inference.py's established pattern exactly:
every model's real-weight loader (`load_real()`) is the ONLY code path that
touches `transformers`/`ultralytics`/`torch`, and is never called by the
automated test suite; all decision logic (`build_class_presence`) is a
plain, injectable-backend, fully unit-testable function operating on a
model-agnostic `RawDetection` list, so tests never need real model weights
or network access.

No threshold search anywhere in this module. Each `load_real_*` function
uses that model's own documented default operating threshold (recorded in
its docstring, cited from the model's own example usage) -- a detection is
"present" simply if it appears in the raw detection list `predict_fn`
returns, which already reflects that fixed threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..phase2b.ontology import CLASS_NAMES

PredictFn = Callable[[str], "list[RawDetection]"]


@dataclass(frozen=True)
class RawDetection:
    label: str  # free-text label as returned by the model (e.g. "a circle", "circle")
    score: float


def make_default_label_to_classes(class_names: tuple[str, ...] = CLASS_NAMES) -> Callable[[str], list[str]]:
    """Returns a `label -> [class_name, ...]` function matching every class
    name that appears as a whole word in `label` (case-insensitive).
    Returns a list, not a single class, because some detectors (observed
    with Grounding DINO) occasionally merge adjacent phrase matches into
    one label string (e.g. "a person a face" for two separate detections
    sharing a box) -- whole-word containment recovers every real match
    from a merged label instead of silently dropping all but one."""
    patterns = {c: re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE) for c in class_names}

    def label_to_classes(label: str) -> list[str]:
        return [c for c, pattern in patterns.items() if pattern.search(label)]

    return label_to_classes


def build_class_presence(raw_detections: list[RawDetection], class_names: tuple[str, ...] = CLASS_NAMES,
                          label_to_classes: Callable[[str], list[str]] | None = None) -> dict[str, tuple[bool, float]]:
    """{class_name: (detected, best_score)} for every class in `class_names`
    -- best_score is 0.0 (never None) for a class with zero raw detections,
    matching phase2b's own convention of always emitting a comparable
    numeric score per class. `label_to_classes` defaults to
    `make_default_label_to_classes(class_names)` if not supplied."""
    label_to_classes = label_to_classes or make_default_label_to_classes(class_names)
    best: dict[str, float] = {c: 0.0 for c in class_names}
    seen: dict[str, bool] = {c: False for c in class_names}
    for d in raw_detections:
        for cls in label_to_classes(d.label):
            if cls in best:
                seen[cls] = True
                best[cls] = max(best[cls], d.score)
    return {c: (seen[c], best[c]) for c in class_names}


class OpenVocabDetector:
    """Model-agnostic wrapper: any `predict_fn(image_path) -> list[RawDetection]`
    (already filtered to that model's own default threshold) becomes a
    per-class presence/score predictor. Used identically for every one of
    the four new candidates -- only `predict_fn` differs per model."""

    def __init__(self, predict_fn: PredictFn, *, model_name: str, class_names: tuple[str, ...] = CLASS_NAMES):
        self.predict_fn = predict_fn
        self.model_name = model_name
        self.class_names = class_names

    def predict_image(self, image_path: str) -> dict[str, tuple[bool, float]]:
        raw = self.predict_fn(image_path)
        return build_class_presence(raw, self.class_names)


# ---------------------------------------------------------------------------
# Prompt vocabulary -- identical noun phrasing across every model in this
# benchmark, differing only in the surrounding template each model's own
# documented usage expects. Recorded here once so every model's exact
# prompt is auditable from a single place.
# ---------------------------------------------------------------------------

CLASS_PHRASES: dict[str, str] = {c: c for c in CLASS_NAMES}  # bare noun, e.g. "person", "circle"


def owlv2_text_queries(class_names: tuple[str, ...] = CLASS_NAMES) -> list[str]:
    """OWLv2's own documented example usage queries with a short noun
    phrase per class (e.g. "a photo of a cat" in the model card; drawing-
    domain equivalent used here: "a {cls}")."""
    return [f"a {CLASS_PHRASES[c]}" for c in class_names]


def grounding_dino_text_prompt(class_names: tuple[str, ...] = CLASS_NAMES) -> str:
    """Grounding DINO's documented input format: lowercase phrases separated
    by '. ', each ending in a period -- exactly the format its own model
    card/example scripts use."""
    return ". ".join(CLASS_PHRASES[c] for c in class_names) + "."


def florence2_prompt_for_class(class_name: str) -> str:
    """Florence-2's <OPEN_VOCABULARY_DETECTION> task takes one free-text
    phrase per call -- queried once per class (10 calls/image), the bare
    noun phrase, matching the other three models' phrasing exactly."""
    return CLASS_PHRASES[class_name]


# ---------------------------------------------------------------------------
# Real-weight loaders. Never called by the test suite (mirrors
# ZeroShotClipDetector.load_real in phase2b/inference.py exactly).
# ---------------------------------------------------------------------------

def grounding_dino_text_labels(results: dict) -> list:
    """`GroundingDinoProcessor.post_process_grounded_object_detection`
    currently returns the same string labels under both `results["labels"]`
    (deprecated -- transformers warns this key will hold integer class ids
    in a future release) and `results["text_labels"]` (the documented,
    stable, string-typed replacement). Always prefer `text_labels`; only
    fall back to `labels` for an older/hypothetical transformers build that
    does not populate `text_labels` at all. Never used to convert a future
    integer id into a string -- if `text_labels` is absent, `labels` is
    still assumed to be the same string type this codebase has always
    required for semantic object names."""
    return results["text_labels"] if "text_labels" in results else results["labels"]


def load_real_owlv2(*, model_id: str = "google/owlv2-base-patch16-ensemble",
                     threshold: float = 0.1, device: str = "cpu") -> OpenVocabDetector:
    """threshold=0.1 matches the value used throughout OWLv2's own
    HuggingFace model-card example usage for
    `post_process_grounded_object_detection` -- not tuned against this
    benchmark's results."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    import torch

    processor = Owlv2Processor.from_pretrained(model_id, use_fast=False)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    texts = [owlv2_text_queries()]

    def predict_fn(image_path: str) -> list[RawDetection]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = processor(text=texts, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=texts)[0]
        return [RawDetection(label=label, score=float(score))
                for label, score in zip(results["text_labels"], results["scores"])]

    return OpenVocabDetector(predict_fn, model_name=f"owlv2:{model_id}")


def load_real_grounding_dino(*, model_id: str = "IDEA-Research/grounding-dino-tiny",
                              threshold: float = 0.25, text_threshold: float = 0.25,
                              device: str = "cpu") -> OpenVocabDetector:
    """threshold=0.25/text_threshold=0.25 are `transformers`'
    `GroundingDinoProcessor.post_process_grounded_object_detection`'s own
    documented function-signature defaults (verified against the installed
    transformers==4.57.6 API this session -- an earlier draft of this
    function used the older `box_threshold` kwarg name and a hand-picked
    0.3, both replaced here with the library's actual current default) --
    not tuned against this benchmark's results. `grounding-dino-tiny` is
    the smallest official checkpoint on the HuggingFace hub, chosen for
    CPU feasibility."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    text = grounding_dino_text_prompt()

    def predict_fn(image_path: str) -> list[RawDetection]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[img.size[::-1]])[0]
        return [RawDetection(label=label, score=float(score))
                for label, score in zip(grounding_dino_text_labels(results), results["scores"])]

    return OpenVocabDetector(predict_fn, model_name=f"grounding_dino:{model_id}")


def load_real_florence2(*, model_id: str = "microsoft/Florence-2-base",
                         device: str = "cpu") -> OpenVocabDetector:
    """Florence-2-base (232M params) -- the smaller of the two official
    checkpoints, chosen for CPU feasibility over Florence-2-large (771M).
    Uses the <OPEN_VOCABULARY_DETECTION> task token, queried once per class
    per image (Florence-2's documented interface takes one phrase per
    call) -- Florence-2 returns any matching region with no separate
    confidence score exposed in its decoded output, so `score` is fixed at
    1.0 for every returned detection (present/absent only; ranking-
    separation is not computable for this model, recorded as None
    downstream, never fabricated).

    Two workarounds are required, not optional, on the installed
    transformers==4.57.6 -- both confirmed by reproducing the failure and
    the fix directly this session, not assumed from documentation:
    (1) `attn_implementation="eager"`: Florence-2's own custom remote
    modeling code (fetched via trust_remote_code=True) predates
    transformers' newer attention-implementation dispatch internals and
    raises `AttributeError: 'Florence2ForConditionalGeneration' object has
    no attribute '_supports_sdpa'` without this flag.
    (2) `use_cache=False` in `generate()`: Florence-2's custom
    `prepare_inputs_for_generation` still assumes the legacy raw-tuple
    `past_key_values` format and raises `AttributeError: 'NoneType' object
    has no attribute 'shape'` against transformers' newer Cache-object
    format otherwise. This disables KV-caching during generation, which is
    the direct cause of this model's much higher per-image cost relative
    to the other three candidates (~9.6s per class per image measured this
    session, vs. a single batched forward pass for OWLv2/Grounding
    DINO/YOLO-World) -- a real, load-bearing runtime cost of working
    around this incompatibility, not an unrelated slowdown."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="eager").to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def predict_for_class(image_path: str, class_name: str) -> bool:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        prompt = "<OPEN_VOCABULARY_DETECTION>" + florence2_prompt_for_class(class_name)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=256, num_beams=1, do_sample=False, use_cache=False)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text, task="<OPEN_VOCABULARY_DETECTION>", image_size=img.size)
        result = parsed.get("<OPEN_VOCABULARY_DETECTION>", {})
        return len(result.get("bboxes", [])) > 0

    def predict_fn(image_path: str) -> list[RawDetection]:
        return [RawDetection(label=cls, score=1.0) for cls in CLASS_NAMES
                if predict_for_class(image_path, cls)]

    return OpenVocabDetector(predict_fn, model_name=f"florence2:{model_id}")


def load_real_yolo_world(*, checkpoint: str = "yolov8s-worldv2.pt",
                          confidence_threshold: float = 0.1) -> OpenVocabDetector:
    """confidence_threshold=0.1 matches Ultralytics' own documented
    open-vocabulary usage examples for YOLO-World (`model.set_classes(...)`
    + default `.predict(conf=0.1)` shown in their own docs for open-set
    prompts, which are naturally lower-confidence than the closed-set COCO
    classes YOLO-World also supports) -- not tuned against this benchmark's
    results. Uses Ultralytics' packaging of YOLO-World (`ultralytics`
    PyPI package), not the original AILab-CVC/YOLO-World repository
    (GPL-v3, heavier mmyolo/mmdetection dependency stack) -- documented
    substitution, not a silent one: same published weights, a different,
    actively-maintained, MIT-licensed inference wrapper."""
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    model.set_classes([CLASS_PHRASES[c] for c in CLASS_NAMES])

    def predict_fn(image_path: str) -> list[RawDetection]:
        results = model.predict(image_path, conf=confidence_threshold, verbose=False)
        out = []
        r = results[0]
        names = r.names
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            label = names[cls_idx]
            score = float(box.conf[0])
            out.append(RawDetection(label=label, score=score))
        return out

    return OpenVocabDetector(predict_fn, model_name=f"yolo_world:{checkpoint}")
