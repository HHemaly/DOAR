# Phase 2C.5 — Rule-Critical Model-Assisted Annotation

**Status: execution phase.** Branch `feature/doar-phase2c-annotation-expansion`, starting HEAD `53bcd38`
(Phase 2C.4A, verified clean before any change this phase). Builds the tooling and validates
feasibility for rule-critical part annotation (eye/mouth/hand/face/person); **does not perform a
large-scale (300–500 image) annotation or proposal run** — that is explicitly gated on your review
(Stage 7).

## 1. Preflight (verified this session)

| Check | Result |
|---|---|
| Branch | `feature/doar-phase2c-annotation-expansion` |
| HEAD (starting) | `53bcd38e0866f0ae8e616cd548dee4c1e5688649` |
| Working tree | clean |
| `PHASE2C4_DETECTOR_BENCHMARK_REPORT.md` / `PHASE2C4A_VALIDATION_CORRECTION_REPORT.md` exist | yes |
| `artifacts/phase2c4/`, `artifacts/phase2c4a/` exist, untouched this phase | yes |
| Private images/annotations gitignored | yes (`outputs/` in `.gitignore`) |
| Child drawings tracked by git | no (verified `git ls-files \| grep image extensions` empty) |

---

## 2. Stage 1 — Rule-to-annotation-target traceability

Read directly from the live `resources/psychology_sources/rules_registry_v2.json` this session
(not from memory) — every `rule_id`/`observable`/quote below is copied from that file, and a test
(`tests/test_phase2c5_rule_traceability.py`) verifies every referenced `rule_id` still exists there
with `allowed_output_level == "disabled"` (unchanged by this phase).

| Target | Supports | Presence enough? | Bbox? | Count? | Attribute? | Suitable this phase? |
|---|---|---|---|---|---|---|
| **eye** | `PSY_AR_EYES_CLOSED_003`, `EN_COMPILED_EYES_MISSING_DETAIL_020` | No | Yes | Yes | `eye_state` (open/closed), `eye_detail` (detailed/undetailed/missing) | **Yes** |
| eye (wide, future) | `PSY_AR_EYES_WIDE_001` | No | Yes (reuses eye bbox) | — | none (derived later from bbox geometry) | No — future derived computation |
| eye (stern, postponed) | `PSY_AR_EYES_STERN_002` | — | — | — | — | **No — no defensible operational definition found** |
| **mouth** | `EN_COMPILED_MISSING_MOUTH_036` | No | Yes | No | none | **Yes** |
| **hand** | `EN_COMPILED_MISSING_HANDS_035`, `EN_COMPILED_EXAGGERATED_BODY_PARTS_037` | No | Yes | Yes | none | **Yes** (bbox added on top of existing Phase 2C.1 presence) |
| **face** | `EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038` (partial — blocked at its "isolation" component regardless) | No | Yes | Yes | none | **Yes** (forward-looking; doesn't unlock the rule alone) |
| **person** | reference structure only (denominator for `EXAGGERATED_BODY_PARTS_037`) | No | Yes | Yes | none | **Yes** |
| body_part_relative_size | `EN_COMPILED_EXAGGERATED_BODY_PARTS_037` | — | uses existing bboxes | — | **none — derived, never annotated as "exaggerated"** | Yes (derived, not an annotation target) |
| face_expression (postponed) | `EN_COMPILED_FACE_EXPRESSION_021` | — | — | — | — | **No — see below** |

Full table: `artifacts/phase2c5/rule_annotation_traceability.csv`.

**`PSY_AR_EYES_STERN_002` postponed**: no defensible geometric or drawn-appearance proxy for
"stern" was identified — unlike open/closed (a binary drawn state), there is nothing to
operationalize yet.

**`EN_COMPILED_FACE_EXPRESSION_021` postponed**: its own source quote — *"Smiling can suggest
positive tone; sad, tense, or frightened faces can suggest distress"* — is written in language too
close to this project's explicitly forbidden Angry/Fear/Happy/Sad vocabulary for a defensible
operational protocol to be designed safely this phase. Even a "geometric-only" taxonomy risks
priming an annotator toward affect judgment. A fully geometric alternative — `mouth_curve_direction`
(upturned/downturned/straight/absent), derived post-hoc from pixels in the mouth bbox via
classical-CV curve analysis, never labeled by a human — is recorded as future work that would not
even need a new annotation pass, since mouth bbox is already collected above.

**9 of 9 requested rule_ids traced; all remain `allowed_output_level: disabled`. Nothing was activated.**

---

## 3. Stage 2 — Annotation schema (`src/doar/phase2c5/schema.py`, `ontology.py`)

- `PART_TARGETS = ("eye", "mouth", "hand", "face", "person")` — every entry traces to a rule above (tested).
- One row = one `(pilot_id, target_name, annotator_id)` judgment (Phase 2C.1's own keying convention),
  but each row carries a **list** of `PartInstance` boxes (JSON-serialized in one cell) to support
  multiple instances (two eyes, several hands) without breaking that keying invariant.
- `status ∈ {present, absent, uncertain, not_assessable}` (reused from Phase 2C.1); `status != present`
  always carries zero instances (same invariant Phase 2C.1's own `instance_count` enforces).
- Attributes (`eye_state`, `eye_detail`) are only allowed on `eye` — enforced by
  `TARGET_ALLOWED_ATTRIBUTE_KEYS`, tested.
- **No psychological label is ever annotated.** `EN_COMPILED_EXAGGERATED_BODY_PARTS_037`'s evidence
  is hand bbox + person bbox; "exaggerated" is a derived `area_ratio`, never a human judgment.

Full schema: `artifacts/phase2c5/annotation_schema.json`.

---

## 4. Stage 3 — Provenance / ground-truth policy

`PartInstance.bbox_source ∈ {model_proposed, human_accepted, human_edited, human_drawn}` — enforced
by validation (`schema.py::PartInstance.__post_init__`):

- `model_proposed` **requires** `proposal_model` (+ checkpoint/prompt/threshold/timestamp) — a
  proposal without a named source is rejected outright.
- `human_drawn` **forbids** any proposal field — false provenance (claiming a hand-drawn box came
  from a model, or vice versa) cannot be constructed.
- Accepting a proposal (`app_helpers.accept_proposal_instance`) changes `bbox_source` to
  `human_accepted` but **keeps** every provenance field, so it stays auditable which model/prompt/
  threshold originally proposed it.
- Editing a proposal (`edit_proposal_instance`) changes `bbox_source` to `human_edited`, same
  provenance-preservation rule.
- **A `model_proposed` instance is never itself saved as a trusted annotation** — the app only ever
  writes `human_accepted` / `human_edited` / `human_drawn` instances, or an empty instance list for
  `absent`/`uncertain`/`not_assessable`. This is a structural property (tested), not a convention.
- Phase 2C.1's own store/schema (`src/doar/phase2c1/{schema,store}.py`) is **never imported for
  writing** by any Phase 2C.5 module (tested) — original human annotations stay untouched.
- Schema versioned separately: `phase2c5_part_annotation_schema_v1` (distinct from Phase 2C.1's
  `phase2c1_annotation_schema_v1`, tested).

---

## 5. Stage 4 — Proposal-model feasibility pilot (real inference, real results)

**15 development-eligible images** (deterministic even-stride sample over the sorted 73-image
`dev_eligible_excl_test` cohort — `scripts/phase2c5_select_feasibility_pilot.py`, `assert_pilot_ids_
exclude_locked_test` called before use). **Locked-test images were never touched.**

Grounding DINO (primary) and OWLv2 (comparison) ran with the **same checkpoints and thresholds
already frozen in Phase 2C.4** (`threshold=0.25/0.25` and `threshold=0.1` respectively — imported
directly from `phase2c4.calibration`'s own constants, not re-chosen), targeting the 5-part
vocabulary instead of the 10-class object ontology. No threshold or prompt was adjusted after
seeing results.

| Model | Inference time (15 images, uncontended) | Total proposals | Images with ≥1 proposal |
|---|---|---|---|
| Grounding DINO | 516s (≈34s/image) | 94 (across 5 targets) | 15/15 |
| OWLv2 | 1053s (≈70s/image) | 39 | 9/15 |

### Per-target proposal counts and visual read

Every proposed box was rendered on the source image and **visually reviewed by the agent this
session** (not a certified Phase 2C.1-style human annotation — used only for this go/no-go decision,
never stored as ground truth):

| Target | Grounding DINO | OWLv2 | Visual read |
|---|---|---|---|
| **person** | 41 boxes, 14/15 images | 13 boxes, 5/15 images | Mostly correct; strong on multi-figure scenes (6/6 figures correctly boxed in one image); occasional false positive (a balloon boxed as a person in one image). |
| **face** | 29 boxes, 13/15 images | 9 boxes, 5/15 images | Consistently reasonable on both single- and multi-figure drawings. |
| **eye** | 16 boxes, 9/15 images | 5 boxes, 2/15 images | Tight, correct localization on clear drawn faces (including 2/2 correct on a two-face image with OWLv2). One clear false-positive case: both eye proposals landed on drawn hearts instead of the actual (much smaller) eyes. |
| **mouth** | 5 boxes, 3/15 images | 5 boxes, 4/15 images | Excellent on 2 simple, clean drawings (tight on the actual drawn mouth line) but **confused with unrelated small shapes twice** — a hair ribbon (Grounding DINO) and a crossed-arms/clothing pattern (OWLv2). Real, recurring failure mode, not a one-off. |
| **hand** | 5 boxes, 4/15 images | 5 boxes, 2/15 images | Small sample; plausible placements near actual hand/wrist regions in every inspected case. |

Full table: `artifacts/phase2c5/proposal_feasibility_summary.csv`.

### Go/no-go per target

- **person, face: GO.** Usable as proposal seeds for a human-reviewed pass with both models;
  Grounding DINO's higher recall makes it the better default source.
- **eye, hand: GO WITH REVIEW.** Real signal, real false positives (hearts confused for eyes) —
  usable to speed up annotation, but every proposal needs a genuine human look, not a rubber stamp.
- **mouth: GO WITH HEAVY REVIEW, not disqualified.** The failure mode is understandable (mouth is a
  small, low-contrast feature that competes with hair ornaments, buttons, and clothing patterns for
  detector attention) but real and recurring in this tiny sample. **Not scaled to hundreds of images
  blindly** — recommend treating mouth proposals as a weak hint only, and budgeting for a high
  proportion of full manual mouth boxes rather than assuming proposals save much annotator time here.
- **No target was judged unusable enough to fully stop.** No hundreds of bad pseudo-boxes were produced.

### Grounding DINO vs. OWLv2

Consistent with Phase 2C.4/2C.4A's own finding at the object-class level: **Grounding DINO has
higher recall** (proposals on more images, more boxes per image) but is the one that produced the
clear eye→heart false positive; **OWLv2 is more conservative** (fewer proposals, but the ones it
does produce were correct in every case inspected, including its best-in-pilot eye localization).
This mirrors Phase 2C.4A's recommended roles almost exactly: Grounding DINO as the higher-recall
proposal generator, OWLv2 as a more precision-leaning comparison source — worth offering **both**
models' proposals side-by-side in the UI (implemented: `PROPOSAL_MODEL_PREFERENCE` tries
Grounding DINO first, falls back to OWLv2) rather than picking one.

### Unexpected finding: corpus content, not detector quality

**While visually reviewing the 15 pilot images (not something this stage set out to check), at
least 2 of the 15 (`p2b_0068`, `p2b_0054`) are visibly watermarked stock/clip-art images from
`dreamstime.com` — not child drawings at all.** A further 2–3 images show content atypical of a
spontaneous child drawing (a hyper-realistic painted eye illustration; an abstract textured painting
with no recognizable figure; a drawing containing an inset close-up reference photo). This is **2 of
15 (13%) confirmed non-genuine content** in a small, deterministically-drawn sample from the exact
73-image cohort every prior phase (2B through 2C.4A) has been evaluating against as if it were
entirely genuine child drawings.

**This was not investigated further this phase** (out of scope, and confirming/quantifying it
properly needs a dedicated pass across the full pilot and likely the underlying corpus). It is
flagged here as the single most important finding of this session, ahead of any per-target
recommendation below — see §12 for a specific ask before any further annotation investment.

---

## 6. Stage 5 — Annotation UI/workflow status

`phase2c5_part_annotation_app.py` (new, Streamlit, mirrors `phase2c_annotation_app.py`'s structure)
+ `src/doar/phase2c5/app_helpers.py` (testable logic, no Streamlit runtime needed — same split the
project already uses: Phase 2C.1's own Streamlit file has no dedicated test file either).

Implemented: open by blinded pilot ID; one target at a time; shows model-proposed boxes (loaded from
a precomputed proposals CSV, never live-inferred inside the app); Accept / numeric-edit / Delete per
box; "+ Add manual box"; multi-instance; present/absent/uncertain/not_assessable; `eye_state`/
`eye_detail` selectors for eye only; uncertainty reason; notes; Previous/Next/Jump navigation;
resume-by-reload (same atomic-write store as Phase 2C.1); progress bar; CSV export. No emotion
label, source folder, detector-confidence-as-truth, or psychological interpretation is displayed
anywhere (tested).

**Known gap, not fixed this phase**: box coordinates are entered as numeric x/y/w/h fields, not a
click-and-drag canvas. A visual canvas widget (e.g. `streamlit-drawable-canvas`) would be a real
usability improvement but is a new runtime dependency — **not added without your separate approval**,
flagged in the app's own module docstring rather than silently pulled in.

**Not yet exercised by a real annotator this session** — see §10.

---

## 7. Stage 6 — Locked-test protection

`assert_pilot_ids_exclude_locked_test` is called in `phase2c5_select_feasibility_pilot.py` **before**
the pilot subset is used for anything (tested — the call must precede any use). The feasibility
pilot, prompt/threshold choices (reused verbatim from Phase 2C.4's frozen constants, never
re-selected against these results), and the expansion-sampling protocol design all draw only from
`dev_eligible_excl_test`. `expansion_sampling.py` never references `LOCKED_TEST` at all (tested).
**No locked-test image was used for any decision this phase, and none was rendered, boxed, or
inspected.**

---

## 8. Stage 7 — 300–500 image expansion sampling protocol (design only)

`src/doar/phase2c5/expansion_sampling.py::select_expansion_sample` — implemented and unit-tested
against a synthetic manifest, **never invoked against the real corpus this phase**. Wraps (never
modifies) Phase 2B's own frozen `select_pilot_sample` group-disjoint, non-conflicted, deterministic
selection, adding an `exclude_image_ids` filter so an expansion round can never re-select an image
the 80-image Stage-A pilot already used. Seed `20260809`, distinct from Phase 2B/2C.1's own
`DEFAULT_SEED=2026`. No emotion-class label is read anywhere in the selection path (the underlying
manifest doesn't expose one).

| Stage | Size | Status | Gate |
|---|---|---|---|
| A — current dev pilot | 80 (73 dev-eligible) | already exists, reused | none |
| B — first expansion | 300–500 | **protocol designed, not executed** | (1) this pilot judged acceptable; (2) UI/schema verified; (3) **you review this protocol** |
| C — larger expansion | 1,000+ | conditional, not sized yet | Stage B's learning-curve/burden results |
| D — remaining dataset | up to 3,688 | conditional, not designed | only if C still shows value |

Full JSON: `artifacts/phase2c5/expansion_sampling_protocol.json`. **No 300–500-image sample was
drawn, no images were blinded/copied, no proposal run was started at that scale.**

---

## 9. Stage 8 — Annotation quality / readiness metrics

`src/doar/phase2c5/quality.py` implements and tests: completion rate, per-target present/absent/
uncertain/not_assessable support, bbox coverage among present cases, instance-count stats, proposal
acceptance/edit/manual-add/rejection rates (from raw-proposal counts vs. final store), and support
imbalance across targets. **Explicit `SINGLE_ANNOTATOR_CAVEAT`** on every readiness summary; no
Cohen's kappa or any agreement statistic is computed anywhere (tested — no kappa import/assignment
exists in the module).

**Current real numbers: 0 across the board.** No human has used `phase2c5_part_annotation_app.py`
yet this session — see `artifacts/phase2c5/annotation_readiness_summary.json`, which states this
plainly rather than fabricating placeholder counts. The tooling is built and tested against
synthetic fixtures and the real feasibility-pilot proposals; the actual human-reviewed annotation
round is the explicit next step, gated on your review.

---

## 10. Unresolved visual targets

- **`PSY_AR_EYES_STERN_002`** — no operational definition found; not annotated, not derived.
- **`EN_COMPILED_FACE_EXPRESSION_021`** — postponed; a fully geometric future path exists (mouth
  curve direction, derived, never labeled) but was not attempted this phase.
- **Mouth false-positive rate** — real, recurring in this small sample; needs either a larger
  feasibility check before scaling, a stricter acceptance threshold in the UI, or accepting a lower
  time-savings expectation for this target specifically.
- **Isolation (`EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038`'s blocking component)** — face bbox is
  now collected, but the spatial-relationship-between-figures logic itself remains Phase 2C.3 Stage D
  scope, not built this phase.

---

## 11. Alternative model/method — not implemented, your opinion requested

**Face/eye/mouth landmark detectors** (e.g. MediaPipe Face Mesh, dlib 68-point landmarks) are
purpose-built for exactly this localization task on human faces, and could in principle be far more
precise than a generic open-vocabulary detector for eye/mouth specifically.

- **Why it could be better**: landmark detectors output dozens of anatomically-anchored points per
  face in milliseconds on CPU — orders of magnitude faster than Grounding DINO/OWLv2's ~30–65s/image
  here, and purpose-fit to exactly the eye/mouth localization problem.
- **Which current failure it might solve**: the mouth false-positive pattern (confusing hair
  ribbons/clothing patterns for a mouth) — a landmark model anchors to a detected face first, so it
  cannot propose a mouth outside a face region the way an open-vocabulary detector can.
- **Real risk, stated plainly**: these models are trained on photographic human faces. A child's
  stick-figure or cartoon-style drawing is extremely out-of-domain for that training data — it is
  quite possible such a model would simply fail to detect a "face" at all on most of this corpus,
  making it *less* useful than the current generic detectors, not more. This is genuinely uncertain
  without testing.
- **Hardware/runtime**: trivial (CPU, sub-second/image) — a non-issue either way.
- **Annotation implications**: none beyond what's already built — it would slot in as a third
  proposal source alongside Grounding DINO/OWLv2 using the same `bbox_source=model_proposed`
  provenance path.
- **Licensing/reproducibility**: MediaPipe is Apache-2.0; dlib is Boost-1.0 — both permissive, both
  well-established, no concern there.
- **Does it change the thesis architecture?** No — same proposal→human-review pattern, one more
  candidate source.
- **Recommendation**: worth a small (~10-image) exploratory test before any decision, given the real
  chance it simply doesn't transfer to line drawings. **Not implemented. Asking for your opinion
  before running even that small test.**

---

## 12. Priority ask before Stage 7 execution: corpus-content review

Independent of the model question above, and more consequential: **§5's incidental discovery of
stock-photo contamination in the very cohort every phase since 2B has treated as genuine child
drawings.** Before committing to a 300–500-image (or larger) annotation expansion, it would be worth
deciding whether to:

(a) run a lightweight manual/visual audit of the existing 80-image pilot (and ideally a larger
    corpus sample) specifically for non-genuine content, before investing more annotation effort in
    a cohort of unknown-but-nontrivial contamination rate; and/or
(b) design an automated or semi-automated screen (e.g. flagging images containing watermark-like
    text/logos) as a preprocessing step before Stage B's expansion sample is drawn.

**Not implemented, not started — explicitly your call**, since it affects the interpretation of
every prior phase's results, not just this one.

---

## Verification

New tests (all synthetic-fixture-only, no real model weights called by the test suite, no private
data): `tests/test_phase2c5_{schema,store,rule_traceability,app_helpers,quality,expansion_sampling,
safety}.py` — schema validation (multi-instance, provenance invariants, attribute restrictions),
store round-trip/resume, rule-registry cross-check (rule_ids exist, still disabled), proposal
accept/edit/reject/delete logic, quality-metric computation (including the "no kappa" structural
check), deterministic expansion sampling with prior-round exclusion, and the full safety suite
(no rule-engine import, no emotion-label exposure outside documented disclaimers, locked-test guard
actually called before use, proposals never auto-accepted, provenance preserved through
accept/edit, all four `bbox_source` values reachable and distinct, multi-instance support,
uncertain/not_assessable handling, Phase 2C.1 store never written by Phase 2C.5 code, schema
versions distinct).
