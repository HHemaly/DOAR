# Phase 2B Model Selection

Compares candidates across the 4 categories the task specifies, using
real, tested facts from this environment (not assumed), before choosing
at most 2 baselines.

## Category A — open-vocabulary bootstrap baseline

**Candidate**: CLIP-based zero-shot region classification (`open-clip-torch`,
already declared in `pyproject.toml`'s `embeddings` extra — no new
dependency added). This is not object detection with learned bounding-box
regression; it is zero-shot *image/region-level* classification against a
text-prompt vocabulary (`"a drawing of a {class}"`), which fits Phase 2B's
own ontology decision (Section "appropriate label type" in
`docs/OBJECT_EVIDENCE_ONTOLOGY.md`) that most classes need presence-only
labels, not precise boxes.

- **Why chosen**: it is the only category that needs *zero* DOAR-specific
  training data to produce a first real result — directly addresses
  `PHASE2B_AUDIT_AND_PLAN.md` Section 4's central finding (no object-level
  annotation has ever existed for this dataset).
- **Real feasibility check performed this session** (not assumed):
  `huggingface.co` is reachable from this environment; a real 605,225,782-byte
  CLIP ViT-B/32 (OpenAI weights, via `open_clip`/`timm`) checkpoint was
  downloaded successfully via direct HTTPS, but at **~2.9 MB/s**, taking
  **~206 seconds**. An initial attempt through `open_clip`'s own
  `huggingface_hub`-backed downloader stalled (0 bytes transferred in 90s)
  for reasons not fully diagnosed — direct HTTPS GET worked reliably, the
  library's own resumable-download path did not, within the time budget
  available to investigate.
- **Implication for design**: real weight downloads are possible but slow
  and not something any automated test may depend on. `inference.py`'s
  zero-shot detector class accepts an *injectable* classifier callable
  precisely so unit/CI tests never need network access or real weights —
  matching this codebase's own established pattern (`tests/test_deep_compare.py::RunnerTests::test_runner_with_injected_synthetic_trainer`).
- **License/version**: CLIP ViT-B/32, OpenAI pretrained weights, MIT-licensed
  `open_clip` library. Model card: general zero-shot classification,
  trained on natural photographs -- **not** on children's line drawings, a
  documented domain gap (Section "limitations" below).
- **Hardware**: ~600 MB checkpoint, runs on CPU or the available Quadro
  P3200 (6.44 GB VRAM) comfortably; well within budget.

## Category B — lightweight supervised detector

**Rejected for this pilot.** A supervised detector (even a small one, e.g.
YOLO-nano-class) needs a properly sized, class-balanced training set.
`PHASE2B_AUDIT_AND_PLAN.md` Section 4 and this pilot's own real annotation
pass (`artifacts/phase2b/class_frequency_audit.csv`) confirm only 2 of 10
candidate classes (`person`, `face`) reached even 5 positive examples in a
20-image sample — nowhere near enough to train any supervised model
without massive overfitting. Explicitly deferred to Phase 2C, contingent
on a much larger annotation campaign this pilot's own findings would
justify requesting.

## Category C — small-parts detection/segmentation

**Rejected for this pilot.** Body-part/component detection (hands, faces
as keypoint sets) is the exact capability `PHASE3_DETECTOR_EVALUATION_PLAN.md`
already found has no verified domain-appropriate (children's line-drawing)
dataset or pretrained model (Section 2 of that doc: MediaPipe/dlib/
keypointrcnn are all photograph-domain; ChildlikeSHAPES authenticity is
unconfirmed). Re-attempting this survey would duplicate completed work
rather than extend it. Deferred to Phase 2C alongside the omission/
exaggeration rules that would consume it (`PHASE2B_AUDIT_AND_PLAN.md`
Section 3).

## Category D — classical/shape baseline for symbols

**Candidate, chosen**: contour-circularity detection for the `circle`
class specifically, via OpenCV (`opencv-python-headless`, already declared
in the `cv` extra). Zero new dependency, zero network/weight requirement,
CPU-only, deterministic. Directly reuses
`PHASE3_DETECTOR_EVALUATION_PLAN.md` Section 8's own recommended first
experiment ("classical-CV circularity pilot for `circles`... directly
tests whether existing segmentation produces clean-enough component
crops") rather than re-deriving it independently.

- **Method**: `cv2.findContours` on a binarized/edge image, filtering by
  isoperimetric ratio (`4*pi*area / perimeter^2`, near 1.0 for a true
  circle) above a fixed threshold.
- **Real limitation found in this pilot's own annotation**: `circle` had
  only 1 real positive example in the 20-image sample (the sun in
  `p2b_0019`, drawn with radiating rays around the circular disc — the
  rays themselves interfere with a clean contour read). This is reported
  honestly in the evaluation, not smoothed over.

## Chosen baselines (2, matching the task's "at most two unless strongly justified")

1. **Category A**: CLIP zero-shot region/image classification — the
   general-purpose baseline covering all 10 ontology classes.
2. **Category D**: classical-CV contour circularity — a cheap, zero-shot-
   independent cross-check specifically for `circle`, since it is the one
   class category (primitive-geometry) with an established, non-learned
   detection method matching the task's own precedent.

## Limitations recorded up front

- CLIP was trained on natural photographs; children's line drawings are a
  substantial domain shift the model has never seen during training. Zero-
  shot performance on this dataset is an open empirical question this
  pilot exists to answer, not an assumption.
- The real annotation sample (n=20) is far below what is needed to
  estimate precision/recall with usable confidence intervals for 8 of the
  10 classes — this is reported as a finding, not concealed
  (`artifacts/phase2b/class_frequency_audit.csv`).
- No GPU-memory instrumentation exists in this repository's pipeline
  (`PHASE2B_AUDIT_AND_PLAN.md` Section 8); this pilot records what it
  observes manually rather than relying on an automated measurement that
  does not exist.
