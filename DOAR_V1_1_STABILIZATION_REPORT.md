# DOAR V1.1 — Stabilization Report

**Status: execution phase, complete.** Branch `feature/doar-phase2c-annotation-expansion`,
starting HEAD `35a2bfd` (verified clean). Fixes the 7 real problems (A–G) a genuine manual test
through `doar_prototype_app.py` exposed, removes every user-facing model/checkpoint/threshold
control, and makes the app's capability claims and confidence numbers honest. No new detector work,
no fine-tuning, no LLM integration — pure stabilization of the existing V1 system.

## 1. Starting branch / HEAD

`feature/doar-phase2c-annotation-expansion` at `35a2bfd` ("DOAR V1: connect validated visual
detections to the real rule engine"), with `d4fd655` (DOAR MVP) and `5d8fde3` (Phase 2C.7) as
ancestors — all three confirmed present in `git log`.

## 2. Preflight state

`pwd` confirmed the correct repository. `git status` was clean before any change. HEAD matched
`35a2bfd` exactly.

## 3. Audit of problems A–G

Traced from source before any edit (not assumed from the bug description):

| # | Root cause found |
|---|---|
| A | `analysis.py` correctly saves artifacts under `case_dir/artifacts/` and stores case-relative paths in `analysis.json` (e.g. `"artifacts/foreground_mask.png"`). But `doar_prototype_app.py`'s Technical View re-computed objective features live via `objective_feature_row(str(original_image), analysis)` using the loaded (case-relative-path) `analysis` dict directly — `features.py` then calls `Image.open(analysis["artifacts"]["foreground_mask"])`, which resolves against the **app's process cwd** (repo root, wherever `streamlit run` was launched), not the case directory. |
| B | `outputs/phase5/` (containing every trained emotion checkpoint) does not exist anywhere in this repository clone — confirmed by a full-repo search for `*.pt`/`*.pth`/`*.ckpt`/`*.safetensors`. This is an environment/deployment gap, not a code bug: `emotion.py::predict` already handles a missing checkpoint file correctly (`status="failed"`, `reason="Checkpoint does not exist: ..."`) — the actual UI bug was that Technical View never displayed that `reason`, and the sidebar offered 3 checkpoint choices that don't exist on this machine as if they were live options. |
| C | Confirmed: the sidebar had a live `st.selectbox("Emotion model", ...)` plus a custom-checkpoint-path `st.text_input` — genuine user-facing model controls. |
| D | `parent_view.py`'s `_CAPABILITY_STATUS` was a static dict predating the DOAR MVP's real visual-detection work — still claimed "General object detection: NOT AVAILABLE" and `build_overall_result_summary` unconditionally emitted "Objects and relationships were not analyzed in this version," even for a case where the visual scan genuinely ran. |
| E | `render_phase2b_pilot_summary` (a real, already-honestly-labeled 20-image research pilot) was called inline as numbered section "10b." directly inside the live per-case Technical View flow, positioned identically to real case data. |
| F | `page_frame.py`'s automatic classical-CV branches compute `confidence = min(1.0, 0.6 + 0.4*uniformity)` (and similar formulas) — reaching 0.94–1.00 whenever a branch fires, while the SAME function's own `limitations` field states the method "cannot distinguish a genuinely full page with a very thin margin from a tightly cropped photo." Confirmed on the exact real case from the bug report: `page_frame_status="cropped_or_content_only"`, `confidence=1.0`. |
| G | Verified, not a bug: Parent View was already exactly 5 sections with zero technical leakage (re-confirmed here with the real, unmodified app via `streamlit.testing.v1.AppTest` — 18/18 pre-existing Parent-View tests pass unchanged). |

## 4. Exact root cause of the foreground-mask failure

`doar_prototype_app.py`, Technical View section "3. Objective features":
```python
feature_row = objective_feature_row(str(original_image), analysis)   # BEFORE
```
`analysis["artifacts"]["foreground_mask"]` is the string `"artifacts/foreground_mask.png"`
(case-relative, by design — `analysis.py` itself relativizes every artifact path so a case directory
stays portable across machines). `features.py::objective_feature_row` opens it directly with
`Image.open(...)`, which is resolved by Python/Pillow relative to `os.getcwd()` — the Streamlit
process's cwd, not the case directory. Confirmed reproducible on the real bug-report case
(`outputs/prototype_cases/h38_1786305027/`).

**Fix**: new module `src/doar/case_artifacts.py` — `resolve_artifact_path`/`resolve_analysis_artifacts`,
resolving every artifact path (flat or nested, e.g. `candidate_masks`) against `case_dir` explicitly,
never relying on cwd. Applied at the exact call site:
```python
resolved_analysis = resolve_analysis_artifacts(analysis, case_dir)   # AFTER
feature_row = objective_feature_row(str(original_image), resolved_analysis)
```
The app's pre-existing single-file `artifact()` helper (used for image previews) was refactored to
reuse the same utility — one source of truth, not two slightly-different resolution implementations.
Verified on the real case (`scripts/doar_v1_1_stabilization_e2e_check.py`, step 5): 60 objective
features computed successfully from the app's own repo-root cwd, zero `[Errno 2]` failures.

## 5. Exact expressive-model failure and resolution

**Not a loading/preprocessing/architecture bug.** `emotion.py::predict` already correctly detects a
missing checkpoint file and returns `status="failed"`, `reason="Checkpoint does not exist: <path>"` —
verified this logic is sound by direct inspection and by the real bug-report case's own `emotion.json`
matching this exact reason string. The actual gaps were: (1) Technical View never displayed the
`reason` field, so the user only saw "status: failed, No probabilities" with no explanation; (2) the
sidebar presented 3 checkpoint choices as if they were live, working options, when none of the
referenced files exist on this machine.

**Resolution**: `production_config.py` (§6) resolves ONE frozen expressive-model identifier
(`efficientnet_b0_seed_42_calibrated`, the checkpoint `CURRENT_CAPABILITY_AUDIT.md` itself documents
as "used in this session's own trace", macro-F1 0.7364) and checks its existence ONCE. If missing, it
passes `checkpoint=None` to `analyze_image_with_timing` — reaching `emotion.py`'s ALREADY-CORRECT
`unavailable()` branch (`status="unavailable"`, not the less-honest `"failed"`) instead of pointing at
a known-dead path. Technical View now shows the `reason` text explicitly. **No fallback checkpoint was
invented; no probabilities are ever fabricated** — confirmed by test
(`test_production_config.py::EmotionUnavailableBranchTests`) and by the real E2E run (§14): the
branch honestly reports `status="unavailable"`, `reason="No emotion checkpoint was supplied."`,
every probability `None`.

## 6. Production model/config policy

New module `src/doar/production_config.py`:
```python
@dataclass(frozen=True)
class ProductionAnalysisConfig:
    expressive_model_identifier: str
    expressive_model_checkpoint: str | None      # None if not present on this machine
    expressive_model_available: bool
    expressive_model_unavailable_reason: str | None
    visual_detector_policy_path: str             # Phase 2C.7's frozen policy artifact
    visual_detector_policy_version: str
```
`resolve_production_config()` is the single function anything in the pipeline calls — resolved once,
reused for every case, written to each case's `production_config.json` for provenance. It never
invents a new model choice: the expressive-model identifier is exactly the one prior phases already
called "recommended" (`CURRENT_CAPABILITY_AUDIT.md`); the visual detector policy path is exactly
Phase 2C.7's already-frozen `artifacts/phase2c7/visual_detector_policy.json`. It only decides WHETHER
the asset is present on the current machine — never substitutes a different, less-validated one.

## 7. User-visible model controls removed

From `doar_prototype_app.py`'s sidebar:
- `st.selectbox("Emotion model", list(KNOWN_CHECKPOINTS.keys()))` — removed.
- `st.text_input("...or a custom checkpoint path...")` — removed.
- The `KNOWN_CHECKPOINTS` dict itself — removed (0 remaining references, confirmed by grep).

Confirmed by a real `AppTest` run against the actual app file
(`tests/test_v1_1_stabilization_app.py::NoModelSelectionControlsTests`): no `st.selectbox` label
contains "emotion model"/"checkpoint"/"efficientnet"/"grounding dino"/"owlv2"; no `st.text_input`
label contains "checkpoint path"; zero `st.slider`/`st.number_input` widgets exist anywhere in the
app. The only remaining `st.selectbox` widgets are "Age range" (parent-supplied display-only
context), "Or reopen a previous case" (case selection), and "Action" (expert-review action choice) —
none are model/threshold controls.

## 8. Page-frame confidence changes and rationale

`page_frame.py`: added `MAX_AUTOMATIC_CONFIDENCE = 0.90` and a `_cap()` helper, applied to every
AUTOMATIC classical-CV branch's confidence formula. **Not a cosmetic markdown** — a principled
ceiling reflecting the method's own documented, irreducible limitation ("cannot distinguish a
genuinely full page with a very thin margin from a tightly cropped photo"): no automatic estimate
from this heuristic may ever be represented as more certain than 0.90. Genuine certainty (1.0)
remains reserved exclusively for a real human decision — `page_reference.py`'s
`user_confirmed_full_frame`/`user_defined_page_corners` paths, which record an explicit user
assertion, not a machine estimate, and were deliberately left untouched (verified by test:
`test_page_frame_confidence.py::UserConfirmationUntouchedByCapTests`). Page-relative rule abstention
(`apply_page_frame_gating`) was re-verified unchanged and still correctly gates on
`page_relative_features_assessable`, not on the raw confidence number. Confirmed on the real
bug-report case: `page_frame` confidence went from the previously-recorded `1.0` to `0.9` on rerun.

## 9. Stale UI/capability cleanup

- `parent_view.py::_CAPABILITY_STATUS`: moved "General object detection", "Face/body-part detection",
  "Arbitrary and unknown object extraction", and "Searchable region evidence" from `not_available` to
  `working` (all genuinely real since the DOAR MVP + Phase 2C.7); added "Expert review workflow" to
  `working`; kept "Spatial relationships" (renamed for precision), "Parent-context updating", "LLM
  explanation", "Visual AI consistency judge" in `not_available` (still genuinely unbuilt, explicitly
  out of this phase's scope).
- `build_overall_result_summary` gained an optional `capabilities` parameter (the case's real
  `judges.json["module_availability"]`): when `capabilities["visual_detection"] == "available"`, the
  sentence changes from "Objects and relationships were not analyzed in this version" (now false) to
  an accurate "The drawing was automatically scanned for known objects and visual elements; spatial
  relationships... are not yet analyzed." `capabilities=None` (or unavailable) keeps the original,
  still-accurate wording — verified both directions never contradict actual case state
  (`test_capability_state.py::OverallResultSummaryCapabilityAwareTests`).
- `judges.py::run_judges`'s `module_availability` dict extended with the canonical, per-case keys:
  `objective_features` (available/partial/unavailable, computed from the real `analysis` at judge
  time), `visual_detection`, `open_world_search`, `expressive_model` (mirrors `emotion_model`),
  `rules` (mirrors `psychologist_rules`), `expert_review` (mirrors `clinician_review`) — **this
  extends the EXISTING mechanism** (`case_output.refresh_module_availability`, built in a prior
  phase for `detection`/`clinician_review`), not a second, redundant schema.

## 10. Research/live-case UI separation

`render_phase2b_pilot_summary` and the "Measurement validation status" subsection (both cross-case,
not per-case, already honestly captioned as such) are now under a single collapsed
`st.expander(...)`, inside a new, clearly-labeled "11. Research / Validation" header, explicitly
captioned "NOT specific to this case, and never mixed into the case-level evidence/rule sections
above." Nothing was deleted — `render_phase2b_pilot_summary`'s own internal `st.header("10b. ...")`
was changed to `st.subheader(...)` (better nesting inside the expander; its test fixture updated to
match) and every subsequent Technical View section renumbered sequentially (11→13, confirmed by grep:
1 through 13, no gaps).

## 11. Parent View verification

Re-verified with the REAL, modified app via `streamlit.testing.v1.AppTest` (not a mock):
`test_phase2a2_parent_view.py`'s pre-existing 18 tests (exactly 5 numbered subheaders; no raw rule_id,
evidence_id, or `detector_absent` string in any Parent-tab widget) all pass unchanged against the
stabilized app. New tests confirm the updated capability-aware wording still contains zero technical
internals (`test_v1_1_stabilization_app.py::CapabilityWordingRegressionTests`,
`test_capability_state.py::...no_technical_internals_leak...`).

## 12. Technical View verification

Confirmed via real `AppTest` runs: "3. Objective features" renders without error (Problem A fixed);
"5. Expressive-content model" now shows the `reason` text when unavailable, plus a new "Production
configuration / provenance" subsection reading `production_config.json`; "11. Research / Validation"
exists and contains the Phase 2B pilot inside a collapsed expander; all provenance (model names,
checkpoints, detector policy version, rule IDs, evidence IDs, bboxes) remains fully visible here —
nothing was removed, only correctly kept out of Parent View.

## 13. Rule-safety regression verification

The DOAR V1 rule-integration work (commit `35a2bfd`) was not touched by this phase. Re-ran its full
targeted suite (`test_rule_engine_v2_visual.py`, `test_visual_rule_integration.py`,
`test_visual_canonical_evidence.py`) as part of the full suite (§16) — all pass unchanged.
Experimental/unknown visual evidence still cannot activate a rule; the registry gate
(`allowed_output_level`) is still respected exactly as before; on-demand search still never triggers
rule integration. No behavior in that subsystem changed.

## 14. Real end-to-end case result

`scripts/doar_v1_1_stabilization_e2e_check.py`, run against **the exact real drawing from the manual
bug report** (`outputs/prototype_cases/h38_1786305027/h38.jpg` — confirmed by inspecting that case's
own saved `analysis.json`: `emotion.status="failed"` with the identical "Checkpoint does not exist"
reason, `artifacts.foreground_mask="artifacts/foreground_mask.png"`, and `page_frame.confidence=1.0`
on a `cropped_or_content_only` read — the exact fingerprints of Problems A/B/F). All 13 steps
**PASSED**:

| Step | Result |
|---|---|
| Production config resolved | `efficientnet_b0_seed_42_calibrated`, `available=False` (genuinely absent on this machine) |
| Real analysis pipeline | `analysis.json` written successfully |
| Foreground mask path (Problem A) | **60 objective features computed with zero path failures**, from the app's own repo-root cwd |
| Expressive model (Problem B) | Honestly `status="unavailable"`, `reason="No emotion checkpoint was supplied."`, every probability `None` — no fabrication |
| No model controls (Problem C) | Confirmed absent from app source |
| Page-frame confidence (Problem F) | `cropped_or_content_only`, confidence **0.9** (was 1.0 on the original manual-test run) |
| Capability state (Problem D) | All canonical `module_availability` keys present and internally consistent |
| Parent View summary | Correct, honest wording; zero technical-internal leakage |
| Technical View (real AppTest) | 0 exceptions, "11. Research / Validation" present, no stale "Feature computation failed" |
| Case reload | `analysis.json`/`judges.json` identical after reload |

## 15. Runtime

13-step script: **6.0 seconds** (no real model loading needed — expressive checkpoint genuinely
absent, visual scan out of this phase's scope, already covered by the separate rule-integration
phase's own real-model E2E run).

## 16. Tests

46 new/changed tests across 6 files (`test_case_artifacts.py`, `test_production_config.py`,
`test_capability_state.py`, `test_page_frame_confidence.py`, `test_v1_1_stabilization_app.py`, plus 1
net-new test in `test_parent_view.py` replacing a now-stale assertion) covering all 18 requested
categories. `ruff check` — all checks passed on every changed file. `compileall` — clean. Targeted
suite (18 files touching every changed module) — **253 passed, 2 pre-existing skips, 0 failed**. Full
repository suite — **1481 passed, 7 pre-existing skips, 0 failed** (1435 baseline + exactly 46 new
tests), 572s wall clock.

## 17. Remaining limitations

- The expressive-model checkpoint genuinely does not exist in this environment (`outputs/` is
  git-ignored and unpopulated on this machine) — this is an infrastructure/deployment fact, not
  something further code changes can fix; the same `production_config.py` identifier will resolve
  automatically the moment that checkpoint file is present on any machine that runs this code.
- `judges_v2.json`/`generated_claims.json`/`verification_report.json` are not regenerated by a plain
  reopen of an old case analyzed before this phase — only newly-analyzed cases get the full canonical
  `module_availability` key set; an old case's `judges.json` simply lacks the new keys (`.get()`-based
  reads elsewhere already handle this gracefully, no crash).
- "Spatial relationships between detected objects" remains explicitly `not_available`, per this
  phase's own scope boundary (no relation/scene-graph work).

## 18. Visual Knowledge V2 — design-only migration plan

**Current representation** (`visual_evidence.py::VisualFinding`, frozen dataclass): `label,
finding_id, free_form_label, bbox, confidence, detector, checkpoint, prompt, validation_status,
evidence_status, rule_mapping_status, related_rule_ids, source, query, timestamp`.

**Target** (`VisualEntity`, NOT implemented this phase): a superset, additive migration —

| New field | Derivation from current `VisualFinding` |
|---|---|
| `entity_id` | = today's `finding_id`, renamed for clarity (stable provenance id, unchanged computation) |
| `canonical_label` | = today's `label` |
| `candidate_labels` | new: `[label]` today (single-candidate list); populated with multiple ranked candidates once a resolver step exists |
| `aliases_en` / `aliases_ar` | new: empty by default; populated from a future English/Arabic synonym table keyed by `canonical_label` |
| `broader_categories` / `possible_subtypes` | new: empty by default; a coarse static mapping (e.g. `"dog"` → `broader_categories=["animal"]`) could be added additively without touching existing fields |
| `visual_similarities` | new: empty; requires an embedding index (Visual Memory, explicitly out of scope) |
| `entity_type` | new: derivable today from `validation_status`/`rule_mapping_status` (e.g. `"object"` vs `"mark"`) — a pure function of existing fields, no new detection needed |
| `bbox` / `crop_ref` | `bbox` unchanged; `crop_ref` new (a path to a saved crop image — additive, computable from `bbox` + the case image at write time) |
| `dominant_colors` | new: computable today from the existing `bbox` + case image with the same colour logic `analysis.py::_dominant_colours` already implements — no new detector |
| `relative_size` / `page_position` | new: computable today from `bbox` + `page_reference.py`'s already-resolved page polygon (§8's own machinery) |
| `shape_features` / `line_features` | new: out of scope for this migration; would reuse `analysis.py`'s existing stroke/shape feature computation, scoped to the entity's crop |
| `detector_provenance` | = today's `detector` + `checkpoint` + `prompt`, grouped |
| `model_validation_status` | = today's `validation_status` (renamed for clarity vs. the new `case_verification_status`) |
| `case_verification_status` | new: `verified`/`uncertain`/`rejected` — populated by expert review (`expert_review.py` already has the `confirm`/`reject`/`rename` actions; this field is where their outcome would be recorded per-entity instead of only in `clinician_review.json`'s flat history) |
| `rule_links` | = today's `related_rule_ids` + `rule_mapping_status`, grouped |
| `expert_review` | new: a structured back-reference to the matching `clinician_review.json` history entries for this entity |
| `memory_status` | new: out of scope (requires Visual Memory) |

**Migration strategy**: additive only. `VisualEntity` would subsume `VisualFinding` field-for-field
(no information loss), with every NEW field defaulting to `None`/`[]`/`"unresolved"` for findings
produced by the CURRENT detectors (which cannot populate them yet) — never a breaking schema change.
`to_dict()`/`from_dict()` would follow the exact same backward-compatible pattern already used for
`finding_id` in this session (an old `detections.json` without the new fields reloads by
regenerating safe defaults, not crashing). The canonical-evidence adapter
(`visual_finding_to_evidence`) and the rule engine (`evaluate_visual_object_presence_rules`) would
need zero changes — they only ever read `label`/`bbox`/`confidence`/`evidence_status`/`validation_status`
today, all of which remain present, unchanged in meaning, on `VisualEntity`.

## 19. Next recommended implementation step

**Implement `VisualEntity` (§18) as its own phase**, starting with the zero-new-detector fields
(`dominant_colors`, `relative_size`, `page_position`, `entity_type`) since those are pure functions of
data DOAR already computes — the highest-value, lowest-risk slice of Visual Knowledge V2, and the
natural foundation the future Q&A tool-planner (§7 of the original task) and rule-management system
(§8) both depend on for richer per-entity reasoning.

## 20. Commit hash

`1736cc9` on `feature/doar-phase2c-annotation-expansion` (parent: `35a2bfd`).

## 21. Push status

**Nothing was pushed at any point in this phase.**
