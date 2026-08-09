"""Phase 2C.7 -- visual-model finalization.

Evaluates the real, human-reviewed Phase 2C.5/2C.6 eye annotations
(outputs/phase2c5/exports/phase2c5_part_annotations.csv) against the
stored Grounding DINO / OWLv2 eye proposals, freezes a single eye
detector configuration on a group-disjoint development split (never the
protected holdout), and combines that result with the existing frozen
Phase 2C.4/2C.4A object-detector benchmark into one class-by-class visual
detector policy (VALIDATED_AUTOMATIC / EXPERIMENTAL_AUTOMATIC / DISABLED).
Ships a single image-only inference API (`visual_detector.analyze`) that
dispatches to the frozen policy internally -- callers never choose a
model or supply annotation.

Never modifies Phase 2C.4/2C.4A's frozen benchmark artifacts, never
fine-tunes, never activates a psychological rule, and never reads an
emotion-class label anywhere.
"""
