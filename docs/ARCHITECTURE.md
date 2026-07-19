# Architecture

DOAR v3 separates objective evidence, learned predictions, rule evaluation,
automated judging, and clinician review. Confidence values from these layers are
never merged into a diagnostic probability. Stable evidence IDs connect claims
to measurements.

The first executable slice contains typed analysis records, dependency-light
foreground segmentation, composition and foreground-only colour analysis,
artifact generation, and existing-split dataset manifests.

Version 3.1 adds a deterministic three-candidate segmentation ensemble
(colour-distance, adaptive grayscale, and global grayscale), disagreement-aware
confidence, exact bounding-box evidence, and a cached numerical objective
feature table. Feature extraction may include test rows for one-time caching,
but training code must filter to `train`; model selection must filter to
`valid`.
