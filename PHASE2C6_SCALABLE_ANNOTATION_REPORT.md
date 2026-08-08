# Phase 2C.6 — Scalable Rule-Critical Annotation

**Status: execution phase.** Branch `feature/doar-phase2c-annotation-expansion`, starting HEAD `66e5119`
(Phase 2C.5, verified clean). Upgrades the Phase 2C.5 annotation app to a graphical bounding-box
editor, builds a resumable/checkpointed proposal-generation pipeline, prepares a real ~300-image
blinded expansion manifest, and runs the first 50-image proposal batch. **Does not process the
remaining 250 images** — explicitly gated on your review, per the stop condition.

## 0. Preflight

| Check | Result |
|---|---|
| Branch | `feature/doar-phase2c-annotation-expansion` |
| HEAD (starting) | `66e5119ccd7aaecdad055c68623e5e62f54796a1` |
| Working tree | clean |
| `PHASE2C5_RULE_CRITICAL_ANNOTATION_REPORT.md` / `artifacts/phase2c5/` exist | yes |

**Direction change acknowledged**: the Phase 2C.5 stock-photo/corpus-content observation remains
documented as a limitation only (§ below) — no watermark detector, no source audit, no restart of
prior experiments. A stock-sourced or non-spontaneous image is not automatically excluded from
detector-proposal work; only images that are corrupted/unreadable or essentially blank are
auto-skipped (`skipped_not_assessable`, §5).

---

## 1. Stage 1 — Graphical bounding-box UI

**Dependency**: `streamlit-drawable-canvas==0.9.3` (MIT, github.com/andfanilo/streamlit-drawable-canvas),
added as the `phase2c6` optional extra in `pyproject.toml`. Chosen over a hand-rolled JS component or a
fuller CV-annotation suite (e.g. label-studio) as the smallest, most widely-used package that does
exactly click/drag/resize/delete boxes on an image with no server component of its own. Its last PyPI
release predates Streamlit 1.36+ and pins only `streamlit>=0.63` (no upper bound) — installs cleanly
and imports without error against this project's `streamlit==1.61.1`, but **was not interactively
verified in a live browser this session** (no browser access in this environment). What WAS verified:
every piece of the coordinate/reconciliation logic, both with synthetic fixtures (28 unit tests,
`tests/test_phase2c6_canvas_helpers.py`) and against the real Phase 2C.5 15-image pilot's real image
dimensions and real proposal boxes (§4).

**Coordinate normalization** (`src/doar/phase2c6/canvas_helpers.py`): the canvas is rendered at a
fixed pixel size computed by `compute_display_size` (downscales to fit within 800px on the longer
side, upscales small images to at least 400px so tiny drawings stay editable — never distorts aspect
ratio, since the background image is pre-resized to exactly that size before being handed to the
canvas). Every conversion between canvas pixel space and this project's normalized `(x, y, w, h)` in
`[0, 1]` bbox convention is re-derived from the CURRENT display size on every call — nothing
accumulates across saves, so a browser-window resize between two save cycles cannot corrupt a box
(tested explicitly, `test_round_trip_survives_a_resize_between_conversions`).

**Reconciliation**: Fabric.js's own JSON serialization does not reliably preserve custom fields
attached to a seeded box, so this module never relies on object identity. Instead, after each canvas
interaction, every box the canvas currently shows is IoU-matched against the boxes that were seeded
into it (`match_canvas_boxes_to_instances`). An unmatched seed was deleted (rejected); an unmatched
canvas box is newly drawn (`human_drawn`); a matched pair that moved becomes `human_edited`; a matched
pair left untouched **stays `model_proposed`** — explicit acceptance is a separate action (an
"Accept as-is" / "Reject" checklist rendered below the canvas for any box still awaiting review), never
implied by merely displaying it. `savable_instances` drops any remaining `model_proposed` box before a
save — the same "a proposal is never ground truth until a human acts on it" invariant Phase 2C.5
established, now re-verified against the graphical workflow (28 tests, plus the real-pilot
verification in §4).

### UX checklist against Stage 1's requirements

| Requirement | Status |
|---|---|
| Display drawing | Yes |
| Show detector proposal boxes | Yes (dashed, distinct color) |
| Click/drag to create a new box | Yes (`drawing_mode="rect"`) |
| Move / resize existing box | Yes (`drawing_mode="transform"`) |
| Delete box | Yes (canvas's own native toolbar) |
| Accept / edit / reject proposed box | Yes (canvas edit = accept-with-edit; explicit checklist = accept-as-is / reject) |
| Multiple boxes per target | Yes (unchanged from Phase 2C.5, now graphically editable) |
| Previous/next image | Yes (unchanged from Phase 2C.5) |
| Previous/next target | **New this phase** — sidebar buttons, replaces the free-form selectbox |
| Keyboard shortcuts | **Partial** — Streamlit has no native global-keyboard-shortcut API without another custom component; the canvas's own Fabric.js toolbar has its own keyboard bindings (e.g. Delete key removes a selected object) but this app adds none of its own. Not pursued further this phase — flagged, not silently ignored. |
| Continuous save | Yes (unchanged: every Save/Save & Next is an atomic full-store rewrite) |
| Safe resume | Yes (unchanged store mechanism; verified again in §4) |
| Visible progress | Yes (unchanged progress bar) |
| Undo last unsaved action | **Provided by the canvas's own native toolbar** (Fabric.js undo/redo/clear), not a separate app-level undo stack — judged sufficient for in-progress canvas edits; not extended to the Accept/Reject checklist buttons (a mis-click there just needs a Reject/re-accept, not a true undo) |

**Allowed eye attributes preserved unchanged from Phase 2C.5** (`eye_state`, `eye_detail`) — no new
attribute was added, and the app still uses `TARGET_ALLOWED_ATTRIBUTE_KEYS` to prevent any other
target from acquiring a subjective attribute. No psychological label was introduced anywhere.

---

## 2. Stage 2 — Verification against the real 15-image pilot

`scripts/phase2c6_verify_pilot_ui.py` — a technical (non-browser) verification against the REAL
Phase 2C.5 pilot's real images and real Grounding DINO/OWLv2 proposals. **13/13 checks passed**:
locked-test exclusion, proposal coordinate round-trip at real image display sizes, resize-survival,
edit persistence with correct provenance, rejected-proposal exclusion, manual-box creation,
multi-instance handling (verified against a real multi-box case, `p2b_0079`/person), store
resume-after-restart, export, and confirmation that Phase 2C.1's own store is never touched (no
`phase2c1.store` writer is even imported by this script — checked at the AST level in
`tests/test_phase2c6_safety.py`). Full output reproduced in the Verification section below.

---

## 3. Stage 3 — 300-image blinded expansion manifest

**Executed for real** (not just designed, per this phase's instructions) against the real Phase 7B
duplicate-group manifest (3,688 images, 3,350 non-conflicted). Method:
`src/doar/phase2c6/expansion_manifest.py::build_expansion_manifest`, which:

1. Excludes every `image_id` already used by the Phase 2C.1 80-image pilot (loaded from that pilot's
   own private mapping CSV) — **this automatically excludes the 7 locked-test images too**, since
   they were themselves drawn from the same original 80-image selection.
2. Calls Phase 2C.5's own `expansion_sampling.select_expansion_sample` (never reimplemented) —
   group-disjoint, non-conflicted, deterministic, no emotion-class label ever read.
3. Blind-copies the 300 selected images to opaque `p2c6_0000`..`p2c6_0299` IDs — a prefix
   **deliberately distinct** from the existing pilot's `p2b_` namespace so the two ID spaces can
   never collide or be confused (tested).

**Result: 300 images selected and blinded**, seed `20260809` (Phase 2C.5's own `EXPANSION_SEED_STAGE_B`
constant, reused unchanged). Private outputs (blinded images, re-identification mapping) live under
`outputs/phase2c6/` (gitignored). Public artifact `artifacts/phase2c6/expansion_manifest_summary.json`
contains only the 300 opaque pilot_ids, the seed, and counts — no `image_id`, `group_id`, or
`original_path` (verified both by a real-data test and an AST-level check that the summary dict
literal cannot contain those keys).

**Emotion labels confirmed unused**: `select_expansion_sample`'s only inputs are `conflict_status`
and `group_id` from the duplicate-group manifest — no emotion-class field exists in that manifest at
all, so there is nothing for the selection to condition on even in principle (confirmed structurally:
no `If`/`Compare` AST node in the selection path mentions any of Angry/Fear/Happy/Sad).

---

## 4. Stage 4 — Resumable, checkpointed, batched proposal pipeline

`src/doar/phase2c6/proposal_batch.py`. Checkpoint file (`outputs/phase2c6/
proposal_batch_checkpoint_private.csv`) is rewritten (atomic replace) after **every batch**, never
only at the end of a run; the raw-proposals CSV is appended to per batch, never held in memory until
the end. Per-image status ∈ `{pending, completed, failed, skipped_not_assessable}`. A restart re-reads
the checkpoint and only processes pilot_ids not already `completed`/`skipped_not_assessable` —
`failed` images ARE retried (a failure may have been transient).

**Verified with a genuine simulated interruption** (not just described): a unit test
(`test_interrupted_mid_run_resumes_only_unfinished_images`) runs the pipeline on only the first 5 of
12 images (simulating a kill), then calls it again with the full 12-image list, and asserts the second
call processes exactly the remaining 7, and that the proposals CSV ends up with exactly one header
line despite being written across two separate runs. 15 tests total for this module, all passing.

`is_assessable` (`scripts/phase2c6_run_proposal_batch.py`) checks only what is objectively checkable
per instruction — image fails to open (corrupted/unreadable), or grayscale pixel std-dev below 2.0
(essentially blank) — never attempts to judge "unrelated content," which needs human judgment.

---

## 5. Stage 5 — First real 50-image proposal batch

Ran `scripts/phase2c6_run_proposal_batch.py --limit 50 --batch-size 25` against the real 300-image
expansion manifest. **Grounding DINO is the primary proposal source** (Phase 2C.4/2C.4A/2C.5's own
consistent finding: higher recall). **OWLv2 is loaded once and invoked only as a fallback**, per
image, for whichever targets Grounding DINO returned nothing for — never a duplicate full inference
pass over every image. Both reuse Phase 2C.4's frozen checkpoints/thresholds (`threshold=0.25/0.25`,
`threshold=0.1`) via Phase 2C.5's own loaders, unchanged.

**Result: 50/50 images completed, 0 failed, 0 skipped_not_assessable.** 369 total proposal boxes.
OWLv2 fallback triggered on 10/50 images (20%) — used only where needed, not a duplicate full pass.

| Target | Proposal boxes | Images with ≥1 proposal (/50) | Proposal rate |
|---|---|---|---|
| person | 103 | 42 | 84% |
| face | 92 | 43 | 86% |
| eye | 98 | 39 | 78% |
| hand | 40 | 22 | 44% |
| mouth | 36 | 24 | 48% |

**Reading this honestly, per instruction**: these are PROPOSAL counts, not accuracy — no human has
reviewed this batch yet, so nothing here is a claim about correctness, only about how often each
target produced *something*. The relative pattern (person/face/eye proposing far more often than
hand/mouth) is directionally consistent with both Phase 2C.4's object-level findings and Phase 2C.5's
15-image pilot — hand and mouth remain the two targets needing the most human review per box, not
just per image.

Full table: `artifacts/phase2c6/proposal_batch_summary.csv`.

---

## 6. Stage 6 — First-batch readiness

| Metric | Value |
|---|---|
| Images processed | 50 |
| Images failed | 0 |
| Images skipped (not assessable) | 0 |
| Proposal boxes/image (mean) | 7.38 |
| Runtime/image (mean / min / max) | 40.18s / 14.96s / 78.81s |
| Total runtime, 50 images (uncontended, this session) | 2009.3s (≈33.5 min) |
| **Projected runtime, remaining 250 images** | ≈167.6 min (≈2.8 hours) |
| **Projected total runtime, all 300 images** | ≈12,054s (≈3.35 hours) |
| Storage: raw-proposals CSV (50 images) | 36 KB → projected ≈216 KB for 300 |
| Storage: checkpoint CSV (50 images) | 8 KB → projected ≈48 KB for 300 |
| Storage: blinded images (all 300, already copied) | 13 MB total |

**Is Grounding DINO practical at this scale?** Yes, on this CPU-only machine, for a batch job run in
the background — ~40s/image is far too slow for live in-app inference (which this pipeline never
does) but entirely practical for an offline, resumable, checkpointed batch: the full 300-image run is
a ~3.35-hour background job, comfortably splittable across several `--limit` invocations if needed,
and safe to interrupt/resume at any point (§4).

**Target-specific pattern**: no failures or crashes on any of the 50 images — the pipeline itself is
robust. The only target-specific pattern worth flagging is proposal RATE (§5's table), not a failure
mode — hand and mouth simply produce a proposal far less often than person/face/eye, consistent with
Phase 2C.4/2C.5's own prior findings, not a new problem introduced by this larger batch.

Full JSON: `artifacts/phase2c6/proposal_runtime_summary.json`.

---

## 7. Stage 7 — Human annotation workflow

Launch: `python -m streamlit run phase2c5_part_annotation_app.py` with environment variables pointed
at the expansion batch:

```
$env:DOAR_PHASE2C1_IMAGES_DIR = "outputs/phase2c6/expansion_images_private"
$env:DOAR_PHASE2C5_PROPOSALS_PATH = "outputs/phase2c6/expansion_raw_proposals_private.csv"
python -m streamlit run phase2c5_part_annotation_app.py
```

(`DOAR_PHASE2C5_STORE_PATH` is left at its default, `outputs/phase2c5/part_annotation_store.csv` —
deliberately the SAME store file used for the 15-image pilot verification, since `annotation_id` is
keyed by `pilot_id__target__annotator_id` and the `p2b_`/`p2c6_` prefixes can never collide; this lets
Stage 8's fine-tuning-readiness count reviewed evidence from both rounds together.)

Workflow exactly as specified: drawing → proposal shown (dashed) → Accept/Edit/Reject → add missing
boxes → set status/eye attribute → Save → next target/image. Progress is calculated automatically
(sidebar). Provenance breakdown (human_accepted / human_edited / human_drawn / rejected / uncertain /
not_assessable) is available via `src/doar/phase2c5/quality.py::proposal_review_outcomes` and
`support_counts`, unchanged from Phase 2C.5, now applicable to this larger batch once you begin
reviewing it.

**Your private annotation data will be stored** at `outputs/phase2c5/part_annotation_store.csv`
(gitignored, local-only, atomic-write-per-save). Blinded expansion images live at
`outputs/phase2c6/expansion_images_private/` (gitignored). Nothing under `outputs/` is ever committed.

---

## 8. Stage 8 — Fine-tuning readiness criterion (design only)

`src/doar/phase2c6/finetuning_readiness.py`. Predeclared thresholds (not tuned against any specific
dataset size):

| Criterion | Threshold |
|---|---|
| Reviewed images | ≥ 150 |
| Positive instances per target | ≥ 40 |
| Bbox coverage among present cases | ≥ 95% |
| Support-imbalance ratio (max/min positive count across targets) | ≤ 8.0 |
| Held-out validation split | ≥ 30 images, disjoint from training |

`evaluate_finetuning_readiness(store, reviewed_pilot_ids)` reports `overall_ready` plus a specific,
itemized list of `blocking_reasons` when not ready — never silently lowers the bar. **Currently not
ready**: 0 human-reviewed rows exist yet in the store (no human has used the upgraded app this
session — that is the explicit next step, not something this automated session can produce). See
`artifacts/phase2c6/finetuning_readiness_criteria.json` for the full criterion definition evaluated
against the current (empty) store.

---

## 9. Stage 9 — Scope discipline

Confirmed NOT implemented anywhere in this phase: SAM/SAM2, MediaPipe, dlib, any new detector family,
relationship modeling, scene graphs, global-context modeling, feature fusion, fuzzy rules,
evidence-weighted rules, psychological inference, rule activation, fine-tuning, PEFT, adapters,
training, or hyperparameter search (all checked structurally — no such import/call exists anywhere in
`src/doar/phase2c6/`, tested).

---

## Verification

`scripts/phase2c6_verify_pilot_ui.py` output (real 15-image pilot):

```
PASS: no locked-test image among the 15 verified pilot_ids
PASS: all real proposal boxes round-trip exactly at their real image's display size
PASS: found a real multi-instance case in the pilot data to verify against
PASS: box survives a display resize between two save cycles
PASS: at least one real seeded proposal for p2b_0079/person
PASS: edited proposal shows human_edited provenance (not silently accepted)
PASS: new hand-drawn box appears as human_drawn
PASS: accept_proposal_instance preserves original proposal provenance
PASS: savable_instances excludes any still-unreviewed model_proposed box
PASS: store resume after simulated restart recovers the saved record
PASS: resumed record's provenance is unchanged
PASS: export produces a readable file
PASS: Phase 2C.1 store file untouched by this script (no write call exists)

13 passed, 0 failed
```

New tests: `tests/test_phase2c6_{canvas_helpers,proposal_batch,finetuning_readiness,
expansion_manifest,safety}.py` — coordinate conversion and resize-survival, provenance through
accept/edit/reject/draw, multi-instance reconciliation, atomic checkpoint save/resume, a genuine
simulated mid-batch interruption and resume, batch-vs-header-write correctness across repeated runs,
locked-test exclusion (structural + real-data), emotion-label blindness (structural AST check on the
selection path, not just a text scan), private-path/opaque-ID leakage (AST-level dict-key check),
expansion-manifest determinism, no-rule-activation, no-fine-tuning, and backward compatibility with
the unmodified Phase 2C.5 schema/store.

68 new tests, all passing (`test_phase2c6_canvas_helpers.py`: 28, `test_phase2c6_proposal_batch.py`: 15,
`test_phase2c6_finetuning_readiness.py`: 5, `test_phase2c6_expansion_manifest.py`: 5,
`test_phase2c6_safety.py`: 15). Ruff and `compileall` clean. Full repository suite:
**1246 passed, 7 skipped, 0 failures** (1178 prior + 68 new, exact match), 331.5s.
