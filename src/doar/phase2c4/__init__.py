"""Phase 2C.4 -- modern open-vocabulary detector benchmark.

Runs OWLv2, Grounding DINO, Florence-2, and YOLO-World (via injectable,
model-agnostic wrappers in detectors.py) against the complete 80-image
Phase 2C.1 pilot, evaluated against genuine Phase 2C.1 human annotations
only (never legacy Phase 2B provisional labels). Existing Phase 2B/2C.2
CLIP zero-shot + classical-CV baselines are reused unchanged, never
rerun. No threshold is tuned anywhere in this package; no rule is
activated; no detector output is treated as ground truth.
"""
