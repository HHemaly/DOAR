# Phase 2B Audit and Plan — Depicted-Object Evidence Pilot

**Status: pre-implementation audit, presented before any detector code was
written, per instruction.** Branch `feature/doar-phase2b-object-evidence-pilot`,
checkpoint tag `checkpoint/pre-doar-trace-phase2b`, starting commit `047d39b`
(Stage 1 CI repair, verified green on GitHub: both `Core` and `ML` jobs
passed on run
[31057964595](https://github.com/HHemaly/DOAR/actions/runs/31057964595)).

This document answers the required audit questions from real, cited
evidence in the repository — `resources/psychology_sources/rules_registry_v2.json`,
`src/doar/registry_v2_build.py`, `src/doar/rule_schema.py`,
`src/doar/evidence_rule_engine.py`, `src/doar/judges.py`,
`src/doar/dataset.py`, `src/doar/dataset_gate.py`,
`resources/psychology_sources/construct_registry.json`,
`PHASE3_DETECTOR_EVALUATION_PLAN.md`, `PHASE3A_RESULTS.md`,
`PHASE7A_DATASET_READINESS.md`, `PHASE7B_DUPLICATE_POLICY.md`, and files
under `outputs/phase7b/` — not invented. No detector code, ontology
finalization, or annotation has been produced yet; this is read-only
investigation.

## 1. Which disabled rules need an object/face/body-part/symbol detector

41 rules total in `rules_registry_v2.json`; **10 executable today** (the 6
original tier-1 composition/placement rules plus 4 Phase 2A line/placement
proxies) — **none of these 10 are object-related**. Of the **31 disabled
rules**, 7 are not addressable by any image detector regardless of Phase 2B's
success, because they need information no static image can carry:

| Class | Count | Rules | Why no detector helps |
|---|---|---|---|
| `process_required` | 3 | `PSY_AR_FLOWERS_CLOUDS_SUN_010`, `EN_COMPILED_LINE_OVER_ERASING_034`, `EN_COMPILED_REFUSAL_TO_DRAW_039` | Needs the child's behavior *while drawing* (distraction, erasure frequency, refusal) — invisible in the finished artifact |
| `longitudinal_required` | 3 | `EN_COMPILED_REPEATED_MONSTERS_DANGER_025`, `EN_COMPILED_REPEATED_FAMILY_CONFLICT_027`, `EN_COMPILED_REPEATED_ISOLATION_THEMES_028` | Needs repetition *across multiple drawings over time*, not one image |
| `not_operational` | 1 | `EN_COMPILED_VERY_SMALL_DRAWING_026` | Needs an absolute physical size reference no scan/photo carries |

The remaining **24 rules are `static_detector`-class** — genuinely
addressable, in principle, from a single image, given a real object/face/
body-part/symbol detector:

`PSY_AR_EYES_WIDE_001`, `PSY_AR_EYES_STERN_002`, `PSY_AR_EYES_CLOSED_003`,
`PSY_AR_ANIMAL_TIGER_WOLF_004`, `PSY_AR_ANIMAL_FOX_005`,
`PSY_AR_ANIMAL_SQUIRREL_006`, `PSY_AR_ANIMAL_LION_007`,
`PSY_AR_GEOMETRY_008`, `PSY_AR_STARS_009`, `PSY_AR_CIRCLES_011`,
`PSY_AR_TRANSPORT_012`, `PSY_AR_HEARTS_013`,
`EN_COMPILED_EYES_MISSING_DETAIL_020`, `EN_COMPILED_FACE_EXPRESSION_021`,
`EN_COMPILED_ANIMAL_CHOICE_GENERAL_022`, `EN_COMPILED_HOUSE_023`,
`EN_COMPILED_TREE_024`, `EN_COMPILED_LINE_ZIGZAG_033`,
`EN_COMPILED_MISSING_HANDS_035`, `EN_COMPILED_MISSING_MOUTH_036`,
`EN_COMPILED_EXAGGERATED_BODY_PARTS_037`,
`EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038`,
`EN_COMPILED_EXCESSIVE_DETAIL_040`, `EN_COMPILED_NEGLECT_BACKGROUND_041`.

Every one of these currently carries `required_feature_ids: []` — no
feature ID exists for any of them yet. The *older* 19-rule registry
(`src/doar/rule_schema.py:71-100`) already reserves a placeholder feature-ID
namespace for several of them even though no extractor populates it:
`semantic.face.eyes`, `semantic.object.animal_species`,
`geometry.shape_repetition`, `semantic.object.symbol`,
`geometry.primitive_shape`, `semantic.object.category` — Phase 2B's evidence
output should populate this existing namespace, not invent a parallel one.

## 2. Which need relationships/judgments, not just presence

Splitting the 24 static-detector rules by what kind of detector output they
actually need:

**Presence-only (12 rules)** — the natural, lowest-risk Phase 2B candidates:
`PSY_AR_ANIMAL_TIGER_WOLF_004/005/006/007`, `EN_COMPILED_ANIMAL_CHOICE_GENERAL_022`,
`PSY_AR_STARS_009`, `PSY_AR_CIRCLES_011`, `PSY_AR_HEARTS_013`,
`PSY_AR_TRANSPORT_012`, `EN_COMPILED_HOUSE_023`, `EN_COMPILED_TREE_024`,
`EN_COMPILED_LINE_ZIGZAG_033` (a stroke pattern, arguably not really an
"object" — flagged for likely exclusion from the ontology, see Section 10
of `docs/OBJECT_EVIDENCE_ONTOLOGY.md` below).

**Relational / state / comparative judgment (12 rules)** — need a detector
*plus* a further judgment on top of detection, all substantially harder and
higher-stakes:
- Eye/face **state** classification: `PSY_AR_EYES_WIDE_001/STERN_002/CLOSED_003`,
  `EN_COMPILED_EYES_MISSING_DETAIL_020`, `EN_COMPILED_FACE_EXPRESSION_021`.
- **Omission** (absence, not presence): `EN_COMPILED_MISSING_HANDS_035`,
  `EN_COMPILED_MISSING_MOUTH_036`, `EN_COMPILED_NEGLECT_BACKGROUND_041`.
- **Comparative sizing**: `EN_COMPILED_EXAGGERATED_BODY_PARTS_037`.
- **Repetition counting**: `PSY_AR_GEOMETRY_008`.
- **Compound / weakest-link**: `EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038`
  (needs face + isolation detection on top of a colour feature that already
  works).
- **Subjective threshold**: `EN_COMPILED_EXCESSIVE_DETAIL_040` ("excessive"
  has no natural detector output).

A ready-made **omission primitive already exists and is tested**:
`evidence_rule_engine.py::evaluate_omission()` (lines 200-219), gating on
`parent_object_reliably_detected`, `region_visible`,
`component_extractor_applicable_and_reliable`, and
`absence_criterion_evaluated`. Its own comment states it was "provided as a
ready, independently-tested primitive for the first component-omission rule
that gets a real body-part/component detector" — this is a direct, unused
hook Phase 2B's evidence schema should integrate with, not duplicate.

## 3. Which must stay disabled even if detected (explicit task constraint)

This is a scientific/ethical judgment independent of technical feasibility:

- **All 4 species-specific animal rules** (`tiger_or_wolf`, `fox`,
  `squirrel`, `lion`) must stay disabled even with a perfect species
  classifier — the task explicitly forbids "species-level animal
  interpretations in this phase," and the construct registry maps them
  straight to serious constructs (`tension_or_anger_pattern`,
  `protection_or_coping`) on the strength of a single, non-replicated
  psychological claim. Only `EN_COMPILED_ANIMAL_CHOICE_GENERAL_022`
  ("any animal, general presence") is compatible with Phase 2B's
  non-species-level ontology, and even that rule stays **disabled**
  during the pilot (Section 16 forbids activating any detector-dependent
  rule regardless).
- **Eye/face-state and expression rules** (`001/002/003/020/021`): the
  hardest technical capability among the 24 (expression classification on
  a *drawn* face, not a photo), feeding directly into
  `low_mood_or_emotional_distress_pattern`/`tension_or_anger_pattern`.
  Left for a dedicated Phase 2C effort, not this pilot.
- **Omission rules** (`035/036/041`) and **exaggerated body parts** (`037`):
  need reliable person/body-part keypoint detection on line drawings, for
  which Phase 3's own detector survey (Section 6 below) found no verified
  domain-appropriate dataset. High psychological stakes, unverified
  technical feasibility — Phase 2C candidates, not Phase 2B.
- **Compound rule 038**: blocked at its weakest link (face + isolation
  detection) regardless of what Phase 2B builds for its colour component.

None of the above will be activated in Phase 2B under any circumstance —
Section 16's constraint ("Do not activate detector-dependent psychological
rules during this pilot") makes the point moot for *every* rule, but this
section records that even the *technical capability*, if it existed today,
would not be sufficient grounds to activate these specific rules without a
dedicated review this pilot does not attempt.

## 4. Which candidate classes occur often enough

**Unknown — and this is itself the central finding of this audit.** Per
Sections 6 and 10, **no object-level annotation has ever been produced for
this dataset**; every detector candidate surveyed in `PHASE3_DETECTOR_EVALUATION_PLAN.md`
was pretrained on *other* data. There is no existing per-class prevalence
count for house/tree/heart/star/circle/animal/person/hand/eye anywhere in
this repository. Phase 2B's own annotation manifest (Section 11 below) is
the mechanism that answers this question — it is a Phase 2B *deliverable*,
not a precondition that can be checked in advance. The ontology
(`docs/OBJECT_EVIDENCE_ONTOLOGY.md`) is therefore written with explicit
per-class "insufficient support" language and a documented merge/postpone/
remove decision rule (Section 13), rather than assuming any class survives
contact with real annotation counts.

## 5. Roboflow augmentations / near-duplicates

Extensively characterized by prior work — Phase 2B must **reuse, not
reinvent**, this machinery:

- `src/doar/dataset.py` already implements aHash (`NEAR_DUP_THRESHOLD = 5`)
  and dHash (`DHASH_THRESHOLD = 6`, Phase 7B) perceptual hashing.
- `src/doar/partition.py` (Phase 7A) computes full-dataset union-find
  duplicate groups: 2,106 groups, 730 with 2+ members, largest group 137
  images at the aHash-5 default — with **documented chaining artifacts**
  (13 of the 15 largest aHash-5 groups span 2-4 different emotion classes,
  i.e. the threshold over-merges unrelated images).
- Phase 7B's blinded 87-pair manual audit found aHash precision collapses
  from 100% (Hamming 0-1) to 0% at distance 5 (the project's own default
  boundary) — the dHash switch and threshold-6→lower correction is **still
  not finalized**: `outputs/phase7b/APPROVED_POLICY.json` and
  `outputs/phase7b/final_partition/FROZEN.json` **do not exist**.
- `check_clean_split_gate()` (`dataset_gate.py`) currently reports
  **`gate_passed = False`**, specifically failing checks 3
  (`duplicate_policy_approved`) and 4 (`manifest_frozen`); the other 6
  checks pass against the unlocked dHash-6 partition.

**Implication for Phase 2B**: Phase 2B cannot depend on a locked clean
split existing. Its own leakage-control document
(`docs/PHASE2B_LEAKAGE_CONTROL.md`) must independently apply the *same*
union-find duplicate-group computation to whatever image sample it selects
for annotation, and must not claim or require the Phase 7B gate to be
green — this is stated as an explicit Phase 2B stop condition already
(Section 18 item "CI not green" is separate from dataset-gate state; see
Section 9 below for how this pilot proceeds without the gate).

## 6. Source-image family grouping for leakage

Directly reuse `src/doar/partition.py`'s union-find `source_image_group`
concept (Phase 7A) — every image selected for Phase 2B annotation gets the
*same* group ID Phase 7A/7B already computed, recorded in
`artifacts/phase2b/duplicate_groups.csv`. Two images in the same group must
never be split across Phase 2B's own train/valid-equivalent partition (used
only if a supervised baseline, Category B, is attempted) — this reuses
existing, tested infrastructure rather than building a second
duplicate-detector.

## 7. Unsuitable or ambiguous images

- **0 unreadable files** in the raw 3,688-image dataset
  (`PHASE3A_RESULTS.md`) — no corrupt-image exclusion needed at intake.
- **674 images (18.3%) carry a label-conflict flag** (Phase 7A's broader
  any-group-member-disagrees check) — Phase 2B's pilot sample should be
  drawn preferentially from the *non-conflicted* 3,014-image pool, both to
  keep the pilot on the cleanest subset and because a conflicted image's
  emotion label is unreliable, which is a confound Phase 2B doesn't need to
  inherit even though it never uses emotion labels itself.
- **Blind annotation is a hard requirement**: `PHASE3_DETECTOR_EVALUATION_PLAN.md`
  Section 5 (round-2 correction) requires annotators have no visibility
  into the emotion-class folder or any detector/model output. Phase 2B's
  sample-selection tooling must physically strip the emotion label from
  whatever an annotator sees (e.g. copy images into a flat, randomized,
  re-IDed folder) — this is a concrete implementation requirement for
  `scripts/phase2b_audit_dataset.py`.

## 8. Hardware / memory available

- GPU: Quadro P3200, **6.44 GB VRAM** (confirmed via
  `torch.cuda.get_device_properties`, `PHASE7_RESULTS.md:62`).
- Disk: **47 GB free**; phases 3a-6 cumulative footprint ≈1.65 GB.
- CPU fallback proven to work for every existing code path
  (`MODEL_EXPERIMENT_AUDIT.md`: "No GPU-only code path exists that would
  break on CPU").
- Peak GPU memory is **not instrumented** anywhere in the pipeline — a
  known, pre-existing gap, not something Phase 2B needs to fix, but Phase
  2B's own model-selection document should record approximate VRAM use
  manually (from `nvidia-smi` at run time) since no automatic instrument
  exists to rely on.
- No system-RAM figure is recorded anywhere in the repo; only VRAM and
  disk are measured historically. Phase 2B should record whatever it
  observes locally, not assume a number.

**Implication**: 6.44 GB VRAM comfortably fits a small supervised detector
(e.g. YOLO-nano-class) or a CPU/light-GPU open-vocabulary zero-shot model
at modest resolution, but rules out large open-vocabulary models (e.g.
Grounding DINO-Large, SAM-ViT-H) without careful batching — consistent
with the existing batch-4/grad-accum-4 discipline already used for deep
training in this repo.

## 9. Smallest scientifically useful Phase 2B experiment

Given Sections 4-8 above — no existing annotation, no locked clean split,
modest but workable hardware, and a strict "presence-only, non-relational,
non-species-level" scope — the smallest defensible pilot is:

1. A **broad, presence-only ontology** of ~8-10 classes drawn from the
   12 presence-only rules in Section 2 (Section 10 below finalizes the
   list, dropping the zigzag-line candidate as not really an "object").
2. A **~200-300 image annotation sample** (within the task's stated
   200-500 range, biased toward the low end because this is explicitly a
   *feasibility* pilot, not a full annotation campaign — matches Phase 3's
   own Stage A "feasibility-only, ~30-50 images" starting point scaled up
   modestly for a slightly broader class set), drawn from the
   non-conflicted 3,014-image pool, blind to emotion label, with
   duplicate-group awareness from Section 6.
3. A **zero-shot open-vocabulary baseline (Category A)** as the primary
   approach — the only category that requires *no* DOAR-specific training
   data to produce a first result, directly addressing the "no annotation
   exists yet" constraint rather than waiting on it. A **classical-CV
   contour-circularity baseline** for the `circle` class specifically,
   reusing Phase 3's own recommended first experiment (zero-dependency,
   already scoped, testable against the pilot annotation directly).
4. Honest per-class evaluation with real support counts, explicit
   "insufficient support" reporting for any class with too few positive
   examples, and no rule activation whatsoever.

## 10. What must wait for Phase 2C

- Species-level animal classification (Section 3).
- Any face/eye state or expression classification.
- Any omission (absence) rule, even though the schema primitive exists.
- Any comparative-sizing or repetition-counting rule.
- Any supervised detector *training* at production scale (needs a properly
  sized, class-balanced, leakage-controlled annotated set this pilot's
  small sample will not produce).
- Any reliance on the Phase 7B clean-split gate being green (it currently
  is not, and locking it is explicitly out of Phase 2B's scope — a
  separate, already-tracked piece of prior work).
- Any actual activation of a detector-dependent psychological rule, under
  any confidence level, per Section 16's explicit constraint.

---

**Audit conclusion**: Phase 2B should proceed as a narrow, presence-only,
zero-shot-first pilot over ~8-10 broad object/symbol classes, evaluated
honestly against a small (~200-300 image), leakage-aware, blind-annotated
sample — producing evidence-schema output only, never a rule activation.
This is the smallest scope that is still a genuine scientific test (does
detection work reliably enough on *this* dataset's line-drawing domain at
all) rather than a re-statement of Phase 3's already-completed planning
work.
