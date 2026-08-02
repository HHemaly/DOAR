# DOAR — Proposed Architecture (post-audit)

This describes the target architecture to satisfy the current working spec, built as an extension of what's already in `DOAR-main`, not a rewrite. Every item is tagged **RETAIN**, **EXTEND**, **BUILD NEW**, or **DECISION NEEDED** (the last requires explicit approval before any code is written, per the working rules). Nothing in this document has been implemented yet except where explicitly noted as already-existing.

---

## 1. Guiding constraint

DOAR analyses **spontaneous free drawings**. No drawing prompt is required or assumed. This shapes every layer below — most importantly, it means Tier 2/3 content-conditional and prompt-dependent rules must be structurally incapable of firing unless their precondition (a positively detected object, or known prompt context) is actually met, not just documented as such.

## 2. Nineteen-stage pipeline mapped to current code

| Spec stage | Current code | Status |
|---|---|---|
| 1. Upload + immutable provenance | `streamlit_app.py` (upload widget, added this session), `main.py analyze-image` | **EXTEND** — upload exists; provenance is not yet immutable (§6 below) |
| 2. Checksum + duplicate ID | `leakage.py` (`sha256`, `phash`) | **RETAIN** — real, tested, exercised |
| 3. Image-quality gate | `analysis.py::_quality()` | **RETAIN** — real resolution/blur/contrast gating, not hardcoded (verified) |
| 4. Page detection | Not implemented as a distinct stage | **BUILD NEW** — currently segmentation assumes the whole image is the page |
| 5. Perspective/illumination correction | Not implemented | **BUILD NEW** |
| 6. Foreground segmentation | `analysis.py::_segment()` (multi-candidate: colour-distance / adaptive-grayscale / global-grayscale) | **RETAIN** — real, multi-strategy, confidence-scored |
| 7. Objective feature extraction | `features.py` (59 features/image, composition/colour/stroke/segmentation families; shape family honestly `NaN`) | **RETAIN** |
| 8. Object/symbol/figure/face detection | Does not exist (`detections.json` explicitly `unavailable`) | **BUILD NEW** — the single largest gap; see §4 |
| 9. Independent emotion inference | `emotion.py`, `deep/inference.py`, three model families (classical/deep/fusion) | **RETAIN** — verified working end-to-end this session |
| 10. Versioned rule execution | `rules.py` | **EXTEND** — needs tiering + content-conditional gating (§3) |
| 11. Evidence-family aggregation | `concerns.py` | **EXTEND** — logic exists and is tested in isolation but is disconnected (audit §5.3); needs the source-type bug fixed and a real aggregation-strength policy (§5) |
| 12. Contradiction/missing-evidence handling | Partially present (`missing_evidence`, `not_matched` statuses in `rules.py`) | **EXTEND** |
| 13. Deterministic report validation | `judges.py` | **RETAIN** — real, tested safety scanning (EN+AR) |
| 14. Optional LLM audit | Does not exist | **BUILD NEW**, optional, off by default (§7) |
| 15. Arabic/English report generation | `reports.py`, `localization.py` | **RETAIN** |
| 16. Evidence-grounded Q&A | `qa.py` | **RETAIN** — deterministic, keyword-routed, evidence-cited |
| 17. Append-only persistence | `review.py` (reviews only) | **EXTEND** — case outputs are currently mutable (audit §6); needs versioning |
| 18. Professional review | `review.py`, Streamlit "Psychologist Review" tab | **RETAIN**, extend with label-conflict/adjudication queue (§6) |
| 19. Research export/evaluation | `thesis.py`, `evaluation.py`, `experiments.py` | **RETAIN** — verified working this session |

## 3. Rule registry — tiered restructuring

**Current state**: one flat list of 19 rules, no tier concept, 68% permanently `missing_detector` (`CURRENT_STATE_AUDIT.md` §5.2).

**Proposed**: restructure `rules_registry.json` into the three tiers the spec defines, without discarding any existing rule (per "retired, not deleted"):

- **Tier 1** (prompt-independent, currently measurable): the 6 coverage/placement rules, plus new candidates — colour statistics, brightness/saturation, colour diversity, symmetry, connected-component count/separation, edge/stroke density, line continuity, shading/blackening, scribbling/fragmentation. Most of the *features* for these already exist in `features.py` (colour, stroke families); what's missing is *rule* wording/citations for each, which requires the literature-sourcing pass described in `IMPLEMENTATION_PLAN.md` Phase 1 — not fabricated here.
- **Tier 2** (content-conditional): the 13 currently-dead rules (eyes, animals, symbols), gated behind detectors that do not yet exist (§4). Each rule's activation must check a real "positively detected" flag with a documented confidence threshold before evaluating — not just check that a feature key exists.
- **Tier 3** (prompt-/age-dependent): none currently exist in the registry; this tier is a placeholder for any future HTP/DAP/KFD-style omission or standardized-scoring rule, and must always return `NOT_ASSESSABLE_CONTEXT_UNKNOWN`/`NOT_ASSESSABLE_AGE_UNKNOWN` under the free-drawing default (`SCIENTIFIC_LIMITATIONS.md` §4).

`rules.py::_status_for()` needs a tier-aware dispatcher; today it has no concept of tier and treats every rule identically once matched.

## 4. Detector layer — the largest genuinely new component

Nothing here exists today. This is real, non-trivial computer-vision scope, not a config change. Recommended approach, in priority order (cheapest/most defensible first):

1. **Face/person detection** — a pretrained, general-purpose face/person detector (e.g. a lightweight face-detection model) applied to line-drawing images. **Known risk**: general photograph-trained detectors often fail on line art/scribbles; this must be validated on a held-out sample of the actual dataset before being called anything but `experimental`, per the spec's explicit prohibition on "substitute a general photograph detector and describe it as validated for children's drawings."
2. **Shape/symbol classifier** (circles, stars, hearts, simple geometric shapes) — plausibly buildable from classical CV (contour approximation + shape descriptors) given the connected-component infrastructure already in `features.py`, likely higher near-term reliability than a learned animal classifier.
3. **Animal-species classifier** (tiger/wolf/fox/squirrel/lion) — hardest of the three: children's line drawings of animals are stylistically far from any existing animal-classification training distribution (which is almost universally photographs). This may not be buildable to an "adequate documented performance" bar at all with realistic effort/data — this should be evaluated and possibly abandoned (with the corresponding rules staying `DETECTOR_UNAVAILABLE` / `CATALOGUED_RESEARCH_ONLY` permanently) rather than shipped as a low-quality detector presented as functional.

**DECISION NEEDED**: which detector(s), if any, to actually build, and with what validation bar, is a scope/cost decision — not made here. `IMPLEMENTATION_PLAN.md` treats this as its own approval-gated phase.

## 5. Aggregation — from "always []" to a real, capped policy

Fix the `source_type` bug (`concerns.py`/`rules.py`, audit §5.3) so evidence genuinely tagged by its real source (objective measurement vs. clinician-symbolic hypothesis vs. model prediction vs. future detector) can be told apart. Then implement the spec's four-level strength vocabulary (`INSUFFICIENT` / `WEAK_HYPOTHESIS` / `POSSIBLE_FOR_EXPLORATION` / `PROFESSIONAL_REVIEW_SUGGESTED`) as a direct replacement for the current binary emit-or-not logic, evidence-family-based (not raw rule-count-based) as the spec requires. This is a **behavior-preserving extension** of already-tested code (`tests/test_concerns_and_ingest.py`), not a rewrite — low risk, no approval needed beyond this document, since it doesn't touch model choice, DB, API, or UI framework.

## 6. Persistence — DECISION NEEDED

Current: flat JSON/CSV, mutable per-case files, one append-only CSV (`review_master.csv`). This works for the single-analyst, single-machine use this session demonstrated, but the spec requires: immutable case history, append-only original-label tracking, a label-conflict/adjudication queue, a training-candidate queue, and eventually multi-user review concurrency (two reviewers rating the same case without clobbering each other's CSV rows — the current `append_reviews()` has no row-level locking).

Two realistic paths:

- **(a) Stay file-based, add versioning by convention** — e.g. `analysis.v1.json`, `analysis.v2.json`, an `index.json` pointing at the current version, revision metadata. Zero new dependencies, fits the existing "everything is a file" mental model, easy to keep working offline with one command as the spec requires. Downside: concurrent-write safety, query-ability (e.g. "show me every case with `POSSIBLE_CONFLICT` status") get progressively more awkward as case count grows, and it's DIY-fragile — easy to reintroduce the exact overwrite bug found in `case_output.py` this session if a future contributor isn't careful.
- **(b) Introduce SQLite** (still zero external services, still one file, still no new infra to run) as an *additional* index/query layer over the same JSON artifacts — cases, labels, reviews, and audit events get rows; large blobs (images, embeddings, checkpoints) stay as files referenced by path+checksum. This directly gives transactional appends, real queries for the label-audit/adjudication workflows, and a natural place for the `original_source_label` + audit-status fields from spec §5. Downside: it's a new dependency and a new migration story (the spec explicitly asks for "safe database migrations"), and it's the kind of "replace the database" decision the working rules say must be approved before implementation, even though SQLite is about as low-risk as a database choice gets (single file, stdlib `sqlite3`, no server process, trivial to back up alongside the rest of `outputs/`).

**Recommendation**: (b), SQLite as an index layer, not a replacement for file storage of large artifacts. **This is flagged as a material decision per the working rules ("replacing... the database... requires approval") — do not start on this until you confirm.**

## 7. Optional external-AI audit layer

**BUILD NEW**, off by default, per spec §13. Proposed shape once approved: a thin function that takes a drafted report string + the case's stored evidence dict, and returns either the report unchanged or `CANNOT_DETERMINE` per unsupported sentence, using a deterministic checker (evidence-ID existence, numeric match against stored values, prohibited-language scan reusing the already-built `judges.py::safety_judge` patterns) *before* anything reaches the user. No model selection (which LLM, if any) is made here — that is itself a decision requiring approval, and is out of scope until the rest of the pipeline is solid, since the spec explicitly requires the system work without any external LLM at all.

## 8. API layer — not currently planned unless requested

No REST API exists today (audit §7) and the spec does not explicitly require one (it asks for "API-contract tests" under testing, which could equally apply to the CLI's argument contract or an internal Python API surface). Recommendation: do not build a new HTTP API/framework choice speculatively — it's exactly the kind of "public API" decision the working rules flag as requiring approval, and nothing in the current interface requirements (Streamlit + CLI) needs one. Revisit only if a concrete need (e.g. a separate frontend) emerges.

## 9. What is explicitly NOT changing without separate approval

- The four-class emotion label schema (Angry/Fear/Happy/Sad) — changing this is a research-protocol decision.
- The choice of `mobilenet_v3_small` as the current best deep model — this was selected on a 2-model, 1-seed reduced sweep for time budget reasons this session, not a considered final architecture decision; §11 of the working spec's model-comparison work (resnet18, EfficientNet, ConvNeXt-Tiny, DINOv2, SigLIP2 candidates) is future work, to be scoped before any "final model" claim is made.
- The UI framework (Streamlit) — retained as-is per "preserve a working existing interface where reasonable."
