# DOAR-TRACE Phase 2B Implementation Report

**Status: real, executed, tested. Not psychologically validated.**
Branch `feature/doar-phase2b-object-evidence-pilot`, checkpoint tag
`checkpoint/pre-doar-trace-phase2b`, starting commit `047d39b` (Stage 1
CI repair, verified green on GitHub: both `Core` and `ML` jobs passed on
run [31057964595](https://github.com/HHemaly/DOAR/actions/runs/31057964595)).
This is a feasibility pilot for depicted-object detection as traceable
visual evidence — explicitly not a psychological-validation phase, not a
production detector deployment, and no rule in `rules_registry_v2.json`
was activated by any result produced here.

## 1. Audit presented before implementation

`PHASE2B_AUDIT_AND_PLAN.md`, built from real, cited evidence across
`rules_registry_v2.json`, `evidence_rule_engine.py`, `judges.py`,
`dataset_gate.py`, `construct_registry.json`, and the prior Phase 3
detector-evaluation planning track. Central finding: no object-level
annotation has ever existed for this dataset, so "which classes occur
often enough" could not be answered in advance — it became this pilot's
own deliverable.

## 2. Ontology

`docs/OBJECT_EVIDENCE_ONTOLOGY.md` / `src/doar/phase2b/ontology.py`: 10
active, presence-only candidate classes (person, face, hand, animal,
house, tree, heart, star, circle, vehicle), 2 explicitly postponed to
Phase 2C (eye, mouth — too fine-grained for a presence-only zero-shot
pilot, and their consuming rules need state/omission judgments regardless
of detection). No species-level animal distinction is made, per explicit
task constraint.

## 3. Real annotation pass

80 images were selected (`src/doar/phase2b/dataset.py::select_pilot_sample`,
seed 2026) — one per distinct duplicate group, from the non-conflicted
pool of the existing Phase 7B partition manifest, blind-copied under
opaque `p2b_XXXX` filenames with no emotion-class trace. **20 of the 80
were fully hand-annotated** against all 10 classes by directly viewing
each image (200 real judgments, single annotator, disclosed as such —
`docs/PHASE2B_ANNOTATION_PROTOCOL.md`). This is below the task's
suggested 200-500 range — a real, stated session constraint, not padded
with unexamined labels. `artifacts/phase2b/annotation_manifest.csv`
(200 rows, validated complete); `artifacts/phase2b/class_frequency_audit.csv`
shows only `person` (9 positive) and `face` (15 positive) reach the
5-positive sufficient-support bar; the other 8 classes do not.

## 4. Model selection

`docs/PHASE2B_MODEL_SELECTION.md`: 2 baselines chosen (task's "at most
two"), both already-declared dependencies, no new package added —
CLIP zero-shot classification (`open-clip-torch`, Category A) and
classical-CV contour circularity (`opencv-python-headless`, Category D).
Categories B/C rejected for this pilot given the real annotation
shortfall found in Section 3. A real network-speed check is recorded:
huggingface.co download works but at ~2.9 MB/s (~3.5 min for a 600MB
checkpoint) — never something an automated test depends on.

## 5. Implementation

`src/doar/phase2b/{ontology,dataset,duplicate_groups,annotations,inference,evaluation,schemas,technical_view}.py`
and `scripts/phase2b_{audit_dataset,run_baseline,evaluate}.py`. Both
baselines accept injectable backends so the automated test suite never
needs network access or real model weights (mirrors
`test_deep_compare.py`'s existing dependency-injection pattern).
`scripts/phase2b_evaluate.py` and `scripts/phase2b_audit_dataset.py` were
both directly verified to reproduce identical output on re-run.

## 6. Real results (honest, not manufactured)

Real CLIP inference was run against all 20 annotated images (and, as a
broader smoke test, all 80 selected images — 0 failures). Evaluated
against real ground truth:

- `face`: ranking-separation 0.18 — **worse than random** (0.5), a
  genuine negative finding for CLIP zero-shot on this drawing domain.
- `person`: recall 0.0 at the fixed 0.28 threshold, but ranking
  separation 0.70 — the underlying similarity score carries real signal;
  the fixed threshold was simply miscalibrated.
- `house`/`tree`/`star`: promising ranking separation (0.93-1.0) but on
  1-3 positive examples each — not generalizable at this sample size.
- Classical-CV circularity for `circle`: recall 1.0, precision 0.059 —
  massively over-fires on non-circle contours.

Full detail: `docs/PHASE2B_EVALUATION_PROTOCOL.md`,
`artifacts/phase2b/{per_class_metrics,threshold_sensitivity,error_analysis,model_manifest.json}`.
**No class's result is strong enough to justify any rule activation, and
none is claimed to be** — per the task's own explicit allowance, this
negative-to-weak result is reported as a valid finding, not hidden.

## 7. Evidence schema and Technical View

`src/doar/phase2b/schemas.py::DetectorEvidenceRecord` — a missing
detection (`not_detected`) is a structurally distinct status from a
confirmed negative; non-conclusive statuses (`uncertain`,
`not_assessable`, `failed`) can never carry a `confidence` value
(enforced in `__post_init__`, tested). `technical_view.py` adds a
Technical-View-only, read-only summary of real pilot artifacts, always
captioned with the same experimental/unvalidated warning. **Parent View
is unmodified** (confirmed by diff) — this pilot's own results do not
clear the bar the task sets for showing anything there.

## 8. Tests and safety

77 new tests total (`tests/test_phase2b_*.py`): ontology validation,
annotation-row invariants (including that a non-`present` status can
never carry a positive count), deterministic manifest creation, CPU-safe
inference smoke tests via injected backends, evaluation-logic tests,
duplicate-group/leakage tests (skip-guarded for the private manifest),
a structural test confirming no `src/doar/phase2b/*` module imports the
rule-evaluation engine, and a diagnostic-language scan reusing
`judges.py`'s own `DIAGNOSTIC_PATTERNS`. Full suite: **851/851 unittest,
852 pytest, ruff clean (verified against both the pinned local ruff and a
fresh 0.16.1 install), compileall clean, AppTest 5/5** — all run in both
the full local `.venv` and a from-scratch CORE-job-equivalent minimal
venv.

## 9. Remaining scientific risks and honest limitations

1. 8 of 10 candidate classes have inadequate positive support to say
   anything reliable about them yet.
2. Neither baseline is validated — both are explicitly `pilot_unvalidated`.
3. The Phase 7B clean-split gate remains ungreen, unrelated to this phase.
4. 60 of the 80 selected (blinded, leakage-safe) images remain
   unannotated — the natural, lowest-friction next step for a larger
   Phase 2C annotation pass.
5. `eye`/`mouth` and every relational/omission/state rule are deferred to
   Phase 2C, not attempted here.

## 10. Exact recommendation for Phase 2C

**Do not begin automatically — this report is the stopping point.** If
authorized: (a) annotate the remaining 60 already-blinded, already-
leakage-safe images (no reselection needed) to raise support for the 8
insufficient classes; (b) investigate `face`'s anti-correlated ranking
before trusting CLIP for any face-related evidence; (c) tune `person`'s
detection threshold only once a genuinely held-out validation slice
exists (not against this pilot's own 20 images); (d) only then consider
Category B/C (supervised or keypoint-based) detectors, and only for
classes with real, adequate annotated support. None of this was
attempted here, and none was claimed to be.
