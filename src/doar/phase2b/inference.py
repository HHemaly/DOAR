"""Phase 2B detector baselines.

Two independent baselines, per docs/PHASE2B_MODEL_SELECTION.md:
  * ZeroShotClipDetector -- Category A, open-vocabulary CLIP zero-shot
    region/image classification, covers all 10 ontology classes.
  * detect_circles_classical -- Category D, classical-CV contour
    circularity, `circle` only.

Both accept injectable backends so automated tests never need real model
weights or network access (mirrors the established pattern in
tests/test_deep_compare.py::RunnerTests::test_runner_with_injected_synthetic_trainer).
Real inference (used only by scripts/phase2b_run_baseline.py, never by
the test suite) is a thin wrapper that loads real open_clip weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ontology import CLASS_NAMES

# A CLIP text-prompt template. Kept broad and non-diagnostic -- this is a
# raw visual-similarity query, not a psychological claim.
PROMPT_TEMPLATE = "a child's drawing containing a {cls}"

EmbedFn = Callable[[str], "list[float]"]  # path/text -> embedding vector


@dataclass
class ZeroShotPrediction:
    class_name: str
    similarity: float  # raw cosine similarity, -1..1
    status: str  # "detected" / "not_detected" / "uncertain"


class ZeroShotClipDetector:
    """CLIP zero-shot classifier over the fixed Phase 2B class vocabulary.

    `image_embed_fn`/`text_embed_fn` are injectable so tests can run with
    a deterministic synthetic embedding function -- no real CLIP weights
    or network access required for CI. `load_real()` is the only entry
    point that touches `open_clip`/`torch`, and it is never called by the
    automated test suite.
    """

    def __init__(self, image_embed_fn: EmbedFn, text_embed_fn: EmbedFn,
                 *, class_names: tuple[str, ...] = CLASS_NAMES,
                 detect_threshold: float = 0.28, uncertain_margin: float = 0.03,
                 model_name: str = "unspecified", model_version: str = "unspecified") -> None:
        self.image_embed_fn = image_embed_fn
        self.text_embed_fn = text_embed_fn
        self.class_names = class_names
        self.detect_threshold = detect_threshold
        self.uncertain_margin = uncertain_margin
        self.model_name = model_name
        self.model_version = model_version
        self._text_cache: dict[str, list[float]] = {}

    @classmethod
    def load_real(cls, *, model_name: str = "ViT-B-32", pretrained: str = "openai",
                   cache_dir: str | None = None, device: str = "cpu",
                   detect_threshold: float = 0.28) -> "ZeroShotClipDetector":
        """Loads real open_clip weights. Requires network access on first
        call (or an already-populated cache_dir) -- never invoked by the
        test suite; used only for the real pilot run."""
        import open_clip
        import torch
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()

        def image_embed_fn(path: str) -> list[float]:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                v = model.encode_image(img)
                v = v / v.norm(dim=-1, keepdim=True)
            return v[0].cpu().tolist()

        def text_embed_fn(text: str) -> list[float]:
            tok = tokenizer([text])
            with torch.no_grad():
                v = model.encode_text(tok)
                v = v / v.norm(dim=-1, keepdim=True)
            return v[0].cpu().tolist()

        return cls(image_embed_fn, text_embed_fn, detect_threshold=detect_threshold,
                   model_name=model_name, model_version=pretrained)

    def _text_embedding(self, class_name: str) -> list[float]:
        if class_name not in self._text_cache:
            self._text_cache[class_name] = self.text_embed_fn(PROMPT_TEMPLATE.format(cls=class_name))
        return self._text_cache[class_name]

    def predict_image(self, image_path: str) -> list[ZeroShotPrediction]:
        image_vec = self.image_embed_fn(image_path)
        preds = []
        for class_name in self.class_names:
            text_vec = self._text_embedding(class_name)
            sim = _cosine(image_vec, text_vec)
            if sim >= self.detect_threshold + self.uncertain_margin:
                status = "detected"
            elif sim >= self.detect_threshold - self.uncertain_margin:
                status = "uncertain"
            else:
                status = "not_detected"
            preds.append(ZeroShotPrediction(class_name=class_name, similarity=sim, status=status))
        return preds


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class CircleDetection:
    detected: bool
    circularity: float  # 4*pi*area / perimeter^2, 1.0 = perfect circle
    count: int


def detect_circles_classical(image_path: str, *, circularity_threshold: float = 0.75,
                              min_contour_area: int = 30) -> CircleDetection:
    """Category D baseline: classical contour-circularity detection for
    the `circle` ontology class only. CPU-only, no model weights, uses
    opencv-python-headless (already declared in the `cv` extra)."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path!r}")
    edges = cv2.Canny(img, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    best_circularity = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_contour_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        best_circularity = max(best_circularity, circularity)
        if circularity >= circularity_threshold:
            count += 1

    return CircleDetection(detected=count > 0, circularity=round(float(best_circularity), 4), count=count)
