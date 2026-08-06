# Current-to-Target Gap (v7) — DOAR-TRACE Phase 2B

**Status: post-implementation record.** Supersedes `CURRENT_TO_TARGET_GAP_V6.md`
(Phase 2A.2) for the areas Phase 2B touched — an object/symbol-detection
*evidence pilot only*, not a psychological validation and not a
production capability. No analysis, rule, feature, threshold, or
aggregation logic changed; no rule in `rules_registry_v2.json` was
activated.

## What Phase 2B's audit found (presented before implementation)

1. Of 41 registry-v2 rules, 10 are executable today, none object-related;
   31 are disabled, of which only 24 are even theoretically
   detector-addressable from a static image (7 need drawing-process
   info, longitudinal comparison, or a physical scale reference no
   detector could ever supply).
2. Of those 24, 12 are presence-only (the realistic pilot candidates);
   12 need a relational/state/omission judgment substantially harder and
   higher-stakes than presence detection.
3. Species-level animal rules, face/eye-state rules, and omission rules
   must stay disabled even with a perfect detector — either explicitly
   forbidden by the task or deferred to Phase 2C on scientific/ethical
   grounds independent of technical feasibility.
4. **No object-level annotation had ever existed for this dataset** —
   "which classes occur often enough" was unanswerable in advance and
   became this pilot's own central finding, not a precondition.
5. The Phase 7B clean-split gate is not green (`duplicate_policy_approved`
   and `manifest_frozen` both fail) — Phase 2B reuses the existing
   duplicate-group machinery without depending on that gate.
6. Hardware (Quadro P3200, 6.44GB VRAM) fits a small zero-shot or
   classical-CV baseline; a real network-speed check found weight
   downloads work but slowly (~2.9 MB/s, ~3.5 min for a 600MB CLIP
   checkpoint) — not something any automated test may depend on.
7. A prior "Phase 3" detector-evaluation track already completed the
   candidate survey and literature review (planning only, nothing
   implemented) — Phase 2B continues from exactly where it stopped.

## Gaps Phase 2B closed

1. **A candidate object-evidence ontology exists** (10 active classes,
   2 explicitly postponed) with per-class inclusion/exclusion criteria,
   ambiguous-case handling, and human-review requirements
   (`docs/OBJECT_EVIDENCE_ONTOLOGY.md`).
2. **A real, hand-annotated pilot sample exists**: 80 images selected and
   blind-copied (group-disjoint, non-conflicted, blind to emotion label),
   20 of them fully annotated against all 10 classes (200 real judgments,
   single annotator, honestly disclosed) — `artifacts/phase2b/annotation_manifest.csv`.
3. **Two real baselines were implemented and run** against real DOAR
   images: CLIP zero-shot classification (Category A) and classical-CV
   contour circularity (Category D) — `src/doar/phase2b/inference.py`,
   real predictions in `artifacts/phase2b/raw_predictions.csv`, an 80-image
   inference smoke test with 0 failures.
4. **Honest evaluation exists**: per-class precision/recall/F1, a
   ranking-separation (AUC-like) statistic, and threshold sensitivity —
   revealing a genuine negative finding (`face`'s CLIP similarity ranks
   *worse than random* against ground truth on this dataset) alongside a
   genuine calibration finding (`person`'s fixed threshold never fires
   despite real ranking signal) — `docs/PHASE2B_EVALUATION_PROTOCOL.md`.
5. **A detector evidence schema exists**
   (`src/doar/phase2b/schemas.py::DetectorEvidenceRecord`) with a status
   vocabulary where a missing detection is structurally distinct from a
   confirmed negative, and a Technical View integration
   (`src/doar/phase2b/technical_view.py`) that displays real pilot
   results with un-skippable experimental-status caveats.
6. **Leakage control reuses, not reinvents**, Phase 7A/7B's existing
   duplicate-group computation — verified directly that all 20 annotated
   images fall in 20 distinct duplicate groups.

## Gaps that remain (honestly, not silently)

1. **8 of 10 candidate classes have insufficient positive support** (< 5
   real examples) at n=20 — no reliability claim is made for them.
2. **Neither baseline is validated for evidence use** — both are
   pilot-stage, explicitly marked `pilot_unvalidated`, with zero rule
   activation.
3. **The Phase 7B clean-split gate is still not green** — unchanged by
   this phase, a separate, already-tracked piece of prior work.
4. **60 of the 80 selected pilot images remain unannotated** — available
   for a follow-up pass without needing to reselect.
5. **`eye`/`mouth` classes and all relational/omission/state rules are
   explicitly deferred to Phase 2C** — no attempt was made at them here.
6. **No supervised detector training was attempted** — correctly, per the
   audit's own finding that no annotated set at production scale exists
   yet.
