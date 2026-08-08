# Phase 2C.3 Audit and Design Report — Expanded Visual Ontology, Context Modeling, and Open-Vocabulary Detector Benchmark

**Status: audit and design only. Nothing installed, downloaded, trained, fine-tuned,
or annotated. STOP point — awaiting approval before any implementation.** Branch
`feature/doar-phase2c-annotation-expansion`, starting commit `ad4011f` (Phase 2C.2,
1011/1011 tests passing). This document extends, and does not redesign, the existing
DOAR-TRACE architecture — every design proposal below is stated in terms of the
repository's own existing modules (`trace_evidence.py`, `phase2b/`, `phase2c1/`,
`phase2c2/`, `page_reference.py`, `rules_registry_v2.json`), not a parallel scheme.

---

## 1. Repository / State Verification (Stage 0)

Verified directly this session, not assumed:

| Check | Result |
|---|---|
| Branch | `feature/doar-phase2c-annotation-expansion` |
| HEAD | `ad4011fc181aefdbf935c2c5a4c3d59f2fa2f0bf` (matches the stated Phase 2C.2 commit exactly) |
| `git status` | clean, up to date with origin |
| `artifacts/phase2c2/` exists | yes — `class_metrics_by_cohort.csv`, `comparison_to_phase2b_20.csv`, `fp_fn_examples.csv`, `model_manifest.json`, `raw_predictions_80.csv` |
| `outputs/phase2c1/annotation_store.csv` exists locally | yes (228,821 bytes) |
| Private outputs gitignored | yes — `git check-ignore` confirms `outputs/phase2c1/annotation_store.csv`, `outputs/phase2c1/private_images`, `outputs/phase2c2/raw_predictions_80_private.csv` all match `.gitignore:5:outputs/` |
| Drawings tracked by git | none — `git ls-files` matching `.jpg/.jpeg/.png/.bmp/.webp` returns empty |

Read/re-grounded this session before proposing anything below:
`DOAR_TRACE_MASTER_SPEC.md`, `resources/psychology_sources/rules_registry_v2.json`
(all 41 rules, structured extraction), `resources/psychology_sources/construct_registry.json`
(all 12 constructs), `docs/OBJECT_EVIDENCE_ONTOLOGY.md`, `PHASE2B_AUDIT_AND_PLAN.md`,
`PHASE2B_IMPLEMENTATION_REPORT.md`, `PHASE2C1_IMPLEMENTATION_REPORT.md`,
`PHASE2C1_POST_ANNOTATION_REPORT.md`, `PHASE2C2_EVALUATION_REPORT.md`,
`src/doar/trace_evidence.py` (the existing `EvidenceRecordV2`/`EvidenceSetV2`
architecture every design below builds on), `src/doar/phase2b/schemas.py`
(`DetectorEvidenceRecord`), `src/doar/phase2c1/schema.py`, `src/doar/phase2c2/evaluation.py`.

---

## 2. Current Bounding-Box / Localization Readiness Audit (Stage A)

Computed directly against the real, private `outputs/phase2c1/annotation_store.csv`
this session (never fabricated):

| Metric | Value |
|---|---|
| Genuine-human rows (`annotator_type == "human"`) | 800 |
| PRESENT judgments | 235 |
| PRESENT with a valid bounding box | **0** |
| Overall bbox coverage | **0 / 235 = 0.0%** |

**PRESENT count per class**: person 59, face 71, hand 43, animal 13, house 14,
tree 15, heart 5, star 6, circle 4, vehicle 5 (identical to
`PHASE2C1_POST_ANNOTATION_REPORT.md` §B — re-verified, not re-derived from a
different source).

**Instance-count coverage**: fully populated for every PRESENT row (required by
`AnnotationRecord`'s own schema — `status == "present"` requires `instance_count >= 1`).
Real distribution: 115 rows at count=1, 48 at count=2, 24 at count=3, 24 at count=4,
1 at count=5, 13 at count=6, 2 at count=7, 6 at count=8, 2 at count=12. This is a
genuine, usable multi-instance signal — not degenerate.

**Bounding boxes**: the annotation app's bbox field (`phase2c_annotation_app.py`,
added during Phase 2C.1) was optional and was never used by the annotator. Zero
bboxes exist anywhere in the real store.

### Conclusion (stated per instruction, not softened)

| Evaluation type | Supported? |
|---|---|
| (a) Presence-only evaluation | **Yes** — already exercised in Phase 2C.2 |
| (b) Count evaluation | **Partially** — ground truth exists and is real, but no current baseline predicts a comparable instance count (CLIP zero-shot has no counting head; only the classical-CV circle detector emits a `count` field, and Phase 2C.2 already found it unreliable) |
| (c) Localization evaluation (bbox-level) | **No** — 0% ground-truth bbox coverage |
| (d) IoU / mAP evaluation | **No** — mathematically undefined without ground-truth boxes |

**The Phase 2C.3 benchmark must remain presence-level (and, where meaningful,
count-level) initially.** No IoU/AP/mAP number may be computed or reported as a
primary metric until genuine human bounding-box ground truth exists at adequate
volume — this is exactly why Stage J (model-assisted annotation) is designed below,
rather than assuming a second all-manual bbox pass.

---

## 3. Visual / Object Ontology V2 (Stage B)

**Method**: every candidate below is either (a) already in the Phase 2B/2C.1
10-class ontology, (b) required by a specific existing rule in
`rules_registry_v2.json` (verified by direct structured extraction of all 41 rules
this session, not memory), or (c) explicitly flagged as **not currently justified**
by any repository requirement — several of the task's own suggested candidates
(door, window, roof, road, furniture, grass) do **not** appear anywhere in the rule
registry, master spec, or construct docs, and are listed here only to record that
they were checked and found unsupported, pending Stage F's blinded discovery
process (which could surface them from real dataset evidence — a different,
legitimate justification route this document explicitly keeps open).

### 3.1 Rule-registry census (verified this session)

41 rules total, by `observability_class`:
- `static_detector` (needs an object/face/part/symbol detector): **24**
- `static_direct` (already executable, composition/placement, no detector): 7
- `static_proxy` (already executable, line-quality proxies): 3
- `process_required` (needs drawing-process observation, no detector helps): 3
- `longitudinal_required` (needs multiple drawings over time): 3
- `not_operational` (no detector could ever help): 1

Only the 24 `static_detector` rules (plus the 3 `process_required`/3
`longitudinal_required`/1 `not_operational`, listed for completeness since a future
architecture must still explain why they stay disabled) drive Ontology V2 candidates.

### 3.2 Candidate visual concepts already covered (Ontology V1, unchanged)

`person, face, hand, animal (general), house, tree, heart, star, circle, vehicle` —
10 classes, `docs/OBJECT_EVIDENCE_ONTOLOGY.md`, real support established in Phase 2C.1/2C.2.

### 3.3 New candidates directly required by existing, currently-disabled rules

| Candidate | Concept type | Driving rule(s) |
|---|---|---|
| `eye` (as a locatable part, not just presence) | facial part | `PSY_AR_EYES_WIDE_001/STERN_002/CLOSED_003`, `EN_COMPILED_EYES_MISSING_DETAIL_020` |
| `mouth` | facial part | `EN_COMPILED_MISSING_MOUTH_036` |
| `face_expression` (a *state*, not a part) | facial attribute | `EN_COMPILED_FACE_EXPRESSION_021` |
| animal **species** distinction (tiger/wolf, fox, squirrel, lion) | object attribute (sub-class) | `PSY_AR_ANIMAL_TIGER_WOLF_004/FOX_005/SQUIRREL_006/LION_007` — **explicitly forbidden to activate** even if detected (Phase 2B Section 3 constraint, restated here, still binding) |
| repeated geometric shape **count** | count / shape symbolism | `PSY_AR_GEOMETRY_008` |
| `hand` **omission** reasoning (not just presence) | omission | `EN_COMPILED_MISSING_HANDS_035` |
| body-part **relative size** (hands/teeth/eyes named explicitly in the source quote) | object attribute (relative size) | `EN_COMPILED_EXAGGERATED_BODY_PARTS_037` |
| `background` presence/extent (as a region, not an object) | environmental / compositional | `EN_COMPILED_NEGLECT_BACKGROUND_041` |
| stroke pattern (`zigzag`) | symbol / line pattern — **flagged in Phase 2B's own ontology doc as arguably not a real "object"** | `EN_COMPILED_LINE_ZIGZAG_033` |
| `sun`, `cloud`, `flower` (as a set) | environmental element | `PSY_AR_FLOWERS_CLOUDS_SUN_010` — **but this rule is `process_required`, not `static_detector`**; detecting these objects would not activate the rule as written. Retained here only because they are the one repository-justified environmental-element cluster; their actual rule use needs a process signal no detector can supply. |
| `monster` / threatening-figure presence | symbol (open-vocabulary-leaning) | `EN_COMPILED_REPEATED_MONSTERS_DANGER_025` — **`longitudinal_required`**; single-image detection alone cannot activate this rule either, but a `monster` presence class is still a legitimate candidate for descriptive/discovery purposes independent of that rule |
| compound face+isolation region | derived (needs `face` + spatial isolation judgment) | `EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038` |

### 3.4 Candidates explicitly checked and found **not currently justified**

`nose, arm, leg, feet/foot, grass, door, window, roof, road, furniture, sky, sea,
water, balloon, butterfly, ghost, weapon, gun, knife, torso, finger, ear, hair,
text/writing` — **zero mentions** anywhere in `rules_registry_v2.json` (checked via
direct regex extraction with word boundaries, not a fuzzy substring match). These
must not be added to Ontology V2 on the strength of this audit alone. They remain
open candidates for Stage F's blinded discovery process, which can supply the
*other* legitimate justification route this document keeps open: real, recurring
dataset evidence.

### 3.5 `unknown` / `other` — a required, not optional, bucket

Per Stage F's requirement below, Ontology V2 must include an explicit
`unknown_or_unclassified` route for anything a discovery pass finds that maps to no
current class — never silently forced into the nearest existing label.

---

## 4. Ontology-to-Rule Traceability Table (Stage B, continued)

| candidate_visual_concept | concept_type | source_rule_or_requirement | currently_supported | requires_object_presence | requires_part_detection | requires_count | requires_attribute | requires_relative_size | requires_position | requires_relationship | requires_omission_reasoning | requires_global_context | clinical_or_rule_relevance | scientific_priority | technical_feasibility | annotation_requirement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| person | object | `EN_COMPILED_*` (implicit, ontology v1) | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | supports omission/size rules once parts exist | High (evaluation_supported) | High (existing CLIP baseline) | presence+count done; bbox needed for part rules |
| face | object | v1 | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | gates `face_expression`, `dark_colors_...` | High | High | presence+count done |
| hand | object | v1; `MISSING_HANDS_035` | Yes (v1) | Yes | No | Yes | No | Yes (037) | No | No | **Yes** (035) | No | omission + relative-size rules | High | High | needs bbox for omission/size judgment |
| animal (general) | object | v1; `ANIMAL_CHOICE_GENERAL_022` | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | general presence only, never species | Medium | High | presence+count done |
| house | object | v1; `HOUSE_023` | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | Medium | High | High | presence+count done |
| tree | object | v1; `TREE_024` | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | Medium | High | High | presence+count done |
| heart | symbol | v1; `HEARTS_013` | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | Low-Medium | Medium | High | presence+count done |
| star | symbol | v1; `STARS_009` | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | Low | Medium | Medium (n=6, exploratory) | presence+count done |
| circle | symbol | v1; `CIRCLES_011` | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | Low-Medium | Medium | Low (both baselines weak) | presence+count done |
| vehicle | object | v1; `TRANSPORT_012` | Yes (v1) | Yes | No | Yes | No | No | No | No | No | No | Low-Medium | Low (n=5, worse-than-random CLIP) | Low | presence+count done |
| eye | facial part | `EYES_WIDE_001/STERN_002/CLOSED_003`, `EYES_MISSING_DETAIL_020` | No | Yes | **Yes** | No | **Yes** (state: wide/stern/closed) | No | No | No | Possibly (missing detail) | No | 4 rules gate on this — highest new-class priority | **High** | Unproven on line drawings (hardest sub-part class) | needs presence + state label + bbox for state judgment |
| mouth | facial part | `MISSING_MOUTH_036` | No (postponed, Phase 2B) | Yes | **Yes** | No | No | No | No | No | **Yes** | No | gates a mood-related rule | Medium-High | Unproven | needs presence + omission judgment |
| face_expression | facial attribute (state) | `FACE_EXPRESSION_021` | No | No (needs `face` first) | No | No | **Yes** | No | No | No | No | No | gates mood construct directly | Medium (hardest technically) | Low (expression-on-drawn-face is unproven anywhere) | needs a labeled expression taxonomy, not just presence |
| animal species (tiger/wolf/fox/squirrel/lion) | object attribute (sub-class) | `ANIMAL_TIGER_WOLF_004/FOX_005/SQUIRREL_006/LION_007` | No — **and must stay disabled even if detected** | Yes | No | No | Yes (species) | No | No | No | No | No | **Forbidden to activate per explicit task constraint, unchanged from Phase 2B** | N/A (excluded by policy) | Low (no matching off-the-shelf vocabulary, per Phase 3's own finding) | out of scope |
| repeated geometric shape count | count / shape symbolism | `GEOMETRY_008` | No | Yes | No | **Yes** | No | No | No | No | No | No | repetition/rigidity construct | Medium | Medium (classical-CV shape counting more tractable than semantic detection) | needs per-instance counting, not just presence |
| body-part relative size | object attribute | `EXAGGERATED_BODY_PARTS_037` | No | Yes (hand/eye/etc.) | Yes (the specific part) | No | No | **Yes** | No | No | No | No | fear/insecurity construct | Medium | Low (needs part bbox + a reference scale) | needs bbox on both the part and a reference (e.g. whole figure) |
| background presence/extent | environmental / compositional | `NEGLECT_BACKGROUND_041` | No | No (a region, not an object) | No | No | Yes (extent) | No | No | No | **Yes** (subjective "neglect" threshold) | **Yes** | disorganisation construct | Low-Medium | Low (region segmentation, not object detection) | needs a region-level, not object-level, judgment |
| sun / cloud / flower | environmental element | `FLOWERS_CLOUDS_SUN_010` | No | Yes | No | No | No | No | No | No | No | No | positive-affect construct — but rule is `process_required`, detector alone cannot activate it | Low (detector work here cannot activate the rule it maps to) | Medium (visually distinct symbols) | presence-only would suffice if ever revisited |
| monster / threatening figure | symbol (open-vocabulary) | `REPEATED_MONSTERS_DANGER_025` | No | Yes | No | No | No | No | No | No | No | **Yes** (longitudinal) | fear construct — rule is `longitudinal_required`, single-image detection cannot activate it | Low (for rule activation); Medium (for descriptive/discovery use) | Low (open-vocabulary, ill-defined class boundary) | presence-only, if pursued at all |
| zigzag line pattern | symbol / line pattern | `LINE_ZIGZAG_033` | No — flagged by Phase 2B as "arguably not an object" | Debatable | No | No | No | No | No | No | No | No | tension/anger construct | Low | Medium (classical-CV line-shape analysis, not a detector) | presence-only |
| unknown / other | open-vocabulary bucket | Stage F requirement | No (does not exist yet) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | prevents forced misclassification | **High (infrastructure)** | N/A | structural, not a detector target |

**Not added** (checked, zero registry support): nose, arm, leg, feet, grass, door,
window, roof, road, furniture, sky, sea, water, balloon, butterfly, ghost, weapon,
text/writing. See §3.4.

---

## 5. Object-Combination / Co-occurrence Design (Stage C)

**A combination is an observation, never an interpretation.** No co-occurrence
pattern below may be described in output text as psychologically meaningful on its
own — only an existing, evidence-graded rule in `rules_registry_v2.json` may do
that, exactly as today.

### 5.1 Representation, reusing the existing evidence architecture

`trace_evidence.py::EvidenceRecordV2` already supports exactly what co-occurrence
needs: `source_evidence_ids` (a list, for recording *which* individual-object
evidence records a derived record aggregates) and `evidence_type` (to tag the
layer). Proposed, **not implemented**, new `evidence_type` value:
`"object_cooccurrence_observation"`.

```
EvidenceRecordV2(
    evidence_id="ev_cooc_p2b_0001_person_house",
    evidence_type="object_cooccurrence_observation",
    producer="phase2c3.cooccurrence",
    value={"classes_present": ["person", "house"], "counts": {"person": 2, "house": 1}},
    unit=None, confidence=None,   # co-occurrence is a set fact, not a probabilistic claim
    method="set_intersection_over_object_evidence_records",
    status="PASS",  # or ABSTAIN if any contributing object evidence itself failed
    source_evidence_ids=["ev_p2c3_person_...", "ev_p2c3_house_..."],
    ...
)
```

### 5.2 What this must answer (queryable, not narrative)

- Which object classes co-occur in a given image (the full present-class set).
- Scene composition: the multiset of `(class, count)` pairs.
- Frequency of specific combinations across the corpus (`person+house`,
  `person+house+tree`, `multiple people`, `person+animal`, etc.) — a descriptive
  corpus statistic, computed exactly the way `Phase2C1's` `class_support`/
  `annotation_status_distribution` artifacts already compute per-class frequencies,
  extended to co-occurrence.
- Common vs. uncommon combinations — a plain frequency ranking, not a judgment.
- Multi-instance-of-one-class detection (`person` count >= 2) — already directly
  computable from Phase 2C.1's existing `instance_count` field, no new detector work
  needed for *this* piece.

### 5.3 Storage

A `EvidenceSetV2` already validates `source_evidence_ids` reference known records
and rejects duplicate `evidence_id`s (`__post_init__`, existing, unmodified) — the
co-occurrence layer gets this validation for free by using the same container, not
a parallel one.

---

## 6. Object-Relationship / Scene-Graph Design (Stage D)

### 6.1 What is derivable directly from boxes (once bboxes exist — they do not yet, §2)

| Relation | Derivable from boxes alone? | Formula sketch |
|---|---|---|
| `left_of` / `right_of` | Yes | compare bbox center-x |
| `above` / `below` | Yes | compare bbox center-y |
| `inside` / `contains` | Yes | bbox containment test |
| `overlap` | Yes | IoU > 0 |
| `touching` | Partially — bbox-adjacency is a coarse proxy; true touching needs mask/contour adjacency, not just box edges |
| `near` / `far` / `relative_distance` | Yes (numeric), but "near"/"far" as categorical needs a chosen threshold — must not be tuned on the locked-test images |
| `relative_size` / `larger_than` / `smaller_than` | Yes | compare bbox area |
| `grouped` / `isolated` | Yes, as a derived statistic over pairwise distances (e.g. a person far from every other detected object) |
| `count` | Yes | already available (`instance_count`, §2) |
| page-region placement | **Already implemented, unrelated to this design** — `src/doar/page_reference.py` already computes page-relative position with 6 modes; any object-level position feature should reuse this module, not reinvent it |

### 6.2 What needs a learned model or VLM, not geometry alone

- Semantic relations beyond geometry (`holding`, `riding`, `talking to`) — no
  current DOAR infrastructure supports this at all; would require a
  relationship-classification model (e.g. a scene-graph-generation model) or a VLM
  prompted for relational captions (Florence-2's region-captioning task is a
  plausible zero-shot starting point, **not run in this audit**).
- `touching` at pixel/contour precision — needs segmentation masks, not boxes
  (masks do not exist yet either; see §9/§10, only Grounding DINO/Florence-2/OWLv2
  offer any mask-adjacent capability among the audited candidates, and even then
  only via a secondary segmentation step).

### 6.3 Proposed structure (design only)

```
nodes = [{"node_id": "person_1", "class": "person", "bbox": [...], "evidence_id": "ev_..."}, ...]
edges = [
    {"from": "person_1", "to": "house_1", "relation": "left_of", "evidence_id": "ev_rel_..."},
    {"from": "person_1", "to": "person_2", "relation": "far_from", "evidence_id": "ev_rel_..."},
    {"from": "tree_1", "to": "house_1", "relation": "right_of", "evidence_id": "ev_rel_..."},
]
```

Each edge is itself an `EvidenceRecordV2` (`evidence_type="object_relationship_observation"`,
`source_evidence_ids=[node_a_evidence_id, node_b_evidence_id]`) — same reuse
principle as §5.

### 6.4 Existing rules that would benefit once this layer exists

- `EN_COMPILED_EXAGGERATED_BODY_PARTS_037` — needs `relative_size` between a body
  part and a reference (the whole figure or another part).
- `EN_COMPILED_MISSING_HANDS_035`/`MISSING_MOUTH_036` — omission reasoning is
  strictly easier once a `person`/`face` node reliably exists to check "does this
  node have an expected child part" against (this is exactly what
  `evidence_rule_engine.py::evaluate_omission()`'s existing, already-tested
  primitive was built for — Phase 2B's audit flagged it as an unused hook; this is
  the connection point).
- `PSY_AR_CIRCLES_011`/isolation-flavoured constructs — `grouped`/`isolated`
  relational statistics between `person` nodes give a real, geometric measurement
  for a concept ("isolation") current rules can currently only proxy through colour
  or size features.
- `EN_COMPILED_NEGLECT_BACKGROUND_041` — needs a figure-vs-background spatial
  relation (does the detected-object region dominate the frame vs. leave background
  visible), not a new object class per se.

---

## 7. Whole-Image / Global-Context Design (Stage E)

### 7.1 Existing and candidate global representations (audited, none run/chosen)

| Representation | Status in DOAR today | What it captures that per-object boxes cannot |
|---|---|---|
| Existing DOAR deep emotion-classification embedding (Phase 3A–7 CNN checkpoints) | **Already exists and is trained**, currently used only for the 4-class emotion head | Whole-composition style/energy the model learned end-to-end; already DOAR-specific (trained on this exact dataset family), unlike every other candidate below |
| CLIP global image embedding | **Already computed as a side-effect** of the existing zero-shot pipeline (`image_embed_fn` in `phase2b/inference.py`) — currently discarded after per-class cosine similarity, never persisted as its own evidence record | Zero marginal cost to persist; a natural first global-branch candidate |
| DINOv2 (self-supervised ViT embeddings) | Not used anywhere in DOAR yet | Purely visual structure/style embedding with no text-conditioning bias — could reveal composition patterns CLIP's language-supervised embedding might miss or over-weight toward "objects" specifically |
| Florence-2 global caption / dense description | Not used anywhere in DOAR yet | A free-text, human-readable global description — closest existing candidate to "what would a person say the whole drawing shows," useful for the discovery task (§8) as much as for a global-context feature |

**No winner is chosen here** — per instruction, no external benchmark table may
decide this; the choice must come from the Stage H-style benchmark, run on DOAR
drawings specifically, not asserted from any paper's numbers.

### 7.2 Proposed 3-way future experiment

`GLOBAL IMAGE ONLY` vs. `OBJECTS ONLY` vs. `OBJECTS + GLOBAL IMAGE` (fusion), all
evaluated against the same genuine-human presence labels, same cohort split
(full/dev-eligible/locked-test) as Phase 2C.2 already established. This is Family 5
in the ablation matrix (§14).

### 7.3 What global context could capture that per-object detection structurally cannot

Overall page-fill/energy gestalt; relative proportion and balance across the whole
composition (already partially covered by existing composition features —
`composition.bounding_box_coverage`, `composition.centroid_normalized` — but those
are single-figure-focused, not whole-scene); elements too small, faint, or
ambiguous to clear any per-class detection threshold but still visually present;
overall stylistic/energy cues the existing line-quality proxies (`stroke.intensity_proxy`,
`stroke.fragmentation`) already measure locally but a global embedding might
capture more holistically. **This is a hypothesis to test (§14 Family 5), not a
claim already validated.**

---

## 8. Blinded Object-Discovery Methodology (Stage F)

**Design only — not run.** Purpose: find recurring visual concepts the current
10-class ontology (and even the rule-justified additions in §3.3) misses, using
*dataset* evidence as the second legitimate justification route (§3, method note).

### 8.1 Procedure

1. **Corpus**: the same `outputs/phase2c1/private_images/` blinded workspace
   already built and verified reproducible (`PHASE2C1_ANNOTATION_PROVENANCE_AUDIT.md`)
   — reused, not reselected. A representative subset (e.g. all 80, or a further
   seeded sub-sample if compute-constrained) — **decision deferred to
   implementation, not fixed here**.
2. **Model**: Florence-2's dense-region-caption or open-vocabulary detection task
   (candidate; not the only option — OWLv2/Grounding DINO could also be prompted
   with a broad vocabulary list as a complementary discovery pass). **Not run in
   this audit.**
3. **Blinding**: identical discipline to every other Phase 2C tool —
   `p2b_XXXX` filenames only, no emotion folder, no original path, ever exposed to
   the model or recorded in output (mirrors `workspace.list_workspace_images`'s
   existing guarantee).
4. **Output schema** (proposed, not implemented):
   `{candidate_concept, frequency_across_corpus, mean_confidence, example_pilot_ids
   (a small sample, not every occurrence), mapped_ontology_v2_class_or_UNKNOWN}`.
5. **The `UNKNOWN`/`OTHER` route is mandatory**, not optional — a candidate that
   doesn't cleanly map to an Ontology V2 class must be recorded as unknown, never
   silently coerced into the nearest existing label (this directly implements §3.5).

### 8.2 Human review workflow (design only)

Mirrors the existing Phase 2C.1 review-mode anti-anchoring pattern
(`phase2c_annotation_app.py`'s "reveal after your own judgment" gate) rather than
inventing a new UI paradigm: a reviewer sees each discovered candidate concept with
its example images and frequency, and chooses **accept** (promote to a real
Ontology V2 class, with the same inclusion/exclusion-criteria authoring
`docs/OBJECT_EVIDENCE_ONTOLOGY.md` already requires for every class), **reject**
(discard), **merge** (fold into an existing class), or **defer** (keep as `UNKNOWN`
pending more examples). **Automatic output is a proposal only, never treated as
ground truth** — restated because it is the single most important constraint on
this stage.

---

## 9. Modern Detector Benchmark Audit (Stage G)

**No package installed, no checkpoint downloaded.** Facts below were verified via
web search this session (license/repo/size), not asserted from training-data
memory alone, matching this project's own established practice
(`DECISION_LOG.md`'s Phase 3 entry: "verified via web search rather than asserted
from memory").

| | CLIP (existing baseline) | Grounding DINO | OWLv2 | YOLO-World | Florence-2 |
|---|---|---|---|---|---|
| Repository | `open_clip_torch` (already a DOAR dependency) | `IDEA-Research/GroundingDINO` | `google/owlv2-*` (HuggingFace `transformers`) | `AILab-CVC/YOLO-World` | `microsoft/Florence-2-*` (HuggingFace `transformers`) |
| Checkpoint used/considered | ViT-B-32, `openai` weights (already run, Phase 2B/2C.2) | Swin-T variant is the standard small entry point | `owlv2-base-patch16-ensemble` | small/medium/large (`yolo-world-s/m/l`) | `Florence-2-base` (232M) or `Florence-2-large` (771M) |
| License | MIT-family (OpenCLIP) | **Apache 2.0** | **Apache 2.0** | **GPL-v3** (copyleft — real constraint for any downstream distribution, thesis/research use is unaffected) | **MIT** |
| Approx. model size | ~350MB (already downloaded, cached) | ~700MB (Swin-T backbone class, unverified exact figure — must be confirmed at actual download time, not assumed) | ~800MB (base, 0.2B params, confirmed via HF model card) | tens of MB (s) to ~200MB (l) — YOLO-family models are the smallest candidates here | ~460MB (base, 232M params) to ~1.5GB (large, 771M params) |
| Dependency complexity | Low (already integrated) | Medium (needs `torch`, `transformers` or the original repo's own detection stack; text-encoder fusion adds setup vs. plain classification) | Low-Medium (fully in `transformers`, same ecosystem already used) | Medium (Ultralytics-family tooling, some divergence from plain `transformers`) | Low (fully in `transformers`, single `AutoModelForCausalLM`-style load) |
| CPU feasibility | **Proven** — this machine, ~21s/80 images (Phase 2C.2) | Plausible but unverified — detection-head architectures are typically heavier per-image than CLIP's single embedding pass | Plausible, same CLIP-family cost class as the existing baseline, likely somewhat heavier due to the detection head | Designed for speed even on modest hardware (real-time on GPU; CPU numbers not verified here) | Plausible for base (232M); large (771M) likely materially slower on CPU-only — **must be measured locally before adoption**, not assumed from any paper |
| GPU preferable? | No (already proven fine on CPU) | Preferred but not required for a small-batch, 80-image evaluation | Preferred but not required at this scale | Preferred but not required at this scale | Preferred, especially for the 771M variant |
| Open-vocabulary | Yes (already exercised) | **Yes** (its core design point — arbitrary category names or phrases) | **Yes** (text-conditioned) | **Yes** (prompt-then-detect) | **Yes** (via task prompts, including open-vocab detection/grounding tasks) |
| Localization / bbox | No (classification-only, as currently used) | **Yes** (its primary output) | **Yes** | **Yes** | **Yes** (dedicated detection/grounding task prompts) |
| Segmentation | No | No natively (commonly paired with SAM/SAM2 in "Grounded-SAM," a separate integration, not audited here) | No | No | **Partial** — Florence-2 supports region-to-segmentation-adjacent tasks in its unified prompt set, degree not verified here |
| Zero-shot | Yes | Yes | Yes | Yes | Yes |
| Fine-tuning capability | Yes (not attempted here) | Yes (documented fine-tuning workflows exist) | Yes (standard `transformers` fine-tuning) | Yes | Yes (documented fine-tuning workflows exist) |
| PEFT/LoRA-style capability | Plausible (linear probe on frozen embeddings, already a known pattern) | Plausible (transformer backbone) | Plausible (transformer backbone, same class as CLIP) | Less standard in the current ecosystem tooling | Plausible (transformer backbone) |
| Confidence/ranking output | Yes (cosine similarity, already used) | Yes (per-box confidence) | Yes (per-box confidence) | Yes (per-box confidence) | Yes (task-dependent scoring) |
| Reproducibility | **Proven** (byte-identical predictions confirmed this session, Phase 2C.2 §0) | Should be deterministic given fixed weights/input, unverified in this repo | Should be deterministic, unverified in this repo | Should be deterministic, unverified in this repo | Should be deterministic, unverified in this repo |
| Suitability for line drawings (a priori) | **Measured, weak-to-moderate** (Phase 2C.2: real signal for `house`/`tree`, anti-correlated for `face`/`vehicle`) | Unknown — trained predominantly on photographic data, same domain-gap risk as CLIP | Unknown — same CLIP-family domain-gap risk, since it shares CLIP's visual backbone | Unknown — trained on photographic detection data (Objects365/GQA/etc.) | Unknown — trained on a very large, but still predominantly photographic + document/OCR, corpus |
| Integration complexity into DOAR | None (done) | Medium — new dependency, new evidence-record shape for boxes | Low-Medium — same `transformers` ecosystem as nothing else in DOAR yet, but conceptually closest to the existing CLIP integration | Medium — a different tooling stack than the rest of DOAR's ML dependencies | Low-Medium — single unified interface could plausibly replace/complement more than one future need (detection, captioning, discovery) with one dependency |

**No winner is declared.** This table is feasibility-only, per instruction — actual
comparison must happen on DOAR drawings (§10/§11), not COCO/LVIS numbers, none of
which are cited above as a ranking signal.

---

## 10. Fair Zero-Shot Model Benchmark Protocol (Stage H)

Design only — mirrors Phase 2C.2's own protocol exactly (same reproducibility
discipline, same cohort split), extended to multiple candidate models:

1. **Corpus**: identical 80-image blinded workspace, identical genuine-human
   reference labels (`annotator_type == "human"` only — never legacy provisional).
2. **Vocabulary**: the same Ontology V1 10-class vocabulary for the first
   comparison round, so results are directly comparable to Phase 2C.2's own
   numbers; Ontology V2 classes (§3) only after §3's candidates are formally
   accepted (§8.2), not before.
3. **Prompts**: equivalent phrasing across models where each model's own prompting
   convention allows (e.g. Grounding DINO/OWLv2/YOLO-World all accept free-text
   category phrases close to CLIP's existing `"a child's drawing containing a
   {cls}"` template; exact wording per model to be finalized at implementation
   time, not tuned against these results).
4. **Blinding**: no emotion label, ever, to any model or in any output — identical
   guarantee already enforced and tested for the existing pipeline.
5. **Leakage/duplicate controls**: preserved — the same duplicate-group-disjoint
   80-image selection already verified reproducible.
6. **Cohorts**: identical three-way split (`full_80` / `dev_eligible_excl_test` /
   `locked_test_descriptive_only`) — the 7 locked-test images reported separately,
   never used for model selection.
7. **No threshold tuning** in this first pass — every model's own suggested
   default operating point is used once, exactly as Phase 2C.2 did for CLIP;
   ranking-separation (threshold-independent) is the primary honest signal, exactly
   as Phase 2C.2 established.

### Metrics (identical vocabulary to Phase 2C.2's own, extended)

Presence precision, recall, F1, specificity, balanced accuracy (new — a natural
addition given some classes have skewed present/absent ratios), ranking/confidence
quality where the model exposes a score, false positives/negatives by pilot_id,
inference time, model size, memory usage, CPU/GPU requirement. **IoU/AP/mAP are
explicitly excluded** from this first pass — §2 already established 0% bbox
ground-truth coverage; computing these would be reporting an undefined metric as if
it were meaningful.

---

## 11. Future Fine-Tuning Experiment Design (Stage I)

**Not performed in this audit.** Ladder, applied per candidate model:

| Tier | Description | Requires |
|---|---|---|
| FT-0 | Zero-shot (as-is) | Nothing beyond what exists now — **the only tier actually run so far** (CLIP, Phase 2C.2) |
| FT-1 | Frozen backbone + a lightweight classification/detection head trained on top | A held-out development split with real labels (not the locked test); feasible once Ontology V2 presence labels exist at reasonable volume per class (Phase 2B's own `MIN_POSITIVE_SUPPORT=5` floor, reused throughout Phase 2C, is the absolute minimum — nowhere near enough for a trained head; dozens to low-hundreds of positives per class is the realistic floor, informed by Phase 7's own effective-sample-size findings on the much larger emotion-classification task) |
| FT-2 | Partial fine-tuning (last transformer layers unfrozen) | More annotation than FT-1, plus compute headroom this CPU-only machine likely lacks for anything beyond the smallest candidate models |
| FT-3 | Parameter-efficient tuning (LoRA/adapters) | Architecture-dependent — plausible for all four new candidates (all are transformer-backbone models) but unverified in practice here; substantially reduces the compute/annotation-volume bar vs. FT-2 |
| FT-4 | Full fine-tuning | Only justified if annotation volume becomes large and leakage-safe — explicitly **not** appropriate at the current 80-image, 0%-bbox-coverage state |

**Hard constraints restated**: no fine-tuning on the 7 locked-test images at any
tier; no hyperparameter selection using the test split; both already enforced
structurally for training/tuning via `workspace.assert_pilot_ids_exclude_locked_test`
(existing, unmodified, not yet invoked by anything — this is exactly the future
call site it was built for).

---

## 12. Model-Assisted Annotation Plan (Stage J)

Given §2's 0% bbox coverage and the project's own stated preference to avoid
another large fully-manual annotation effort, the recommended path (design only):

1. Run the eventual best zero-shot model (chosen via §10/§11's benchmark, not
   assumed) over the blinded workspace to propose, per candidate detection:
   `class, bbox, optional mask, count`.
2. Present each proposal to a human via an extension of the existing
   `phase2c_annotation_app.py` (same app, new mode) — the human may **accept**,
   **reject**, **edit** (adjust the box), **relabel** (wrong class, right box),
   **mark uncertain**, or **mark not_assessable** — the exact same status
   vocabulary Phase 2C.1's schema already enforces (`OBJECT_STATUSES`), extended
   with a `bbox_source: "model_proposed" | "human_drawn" | "human_edited"`
   provenance field so a model-assisted box is always structurally distinguishable
   from one a human drew from scratch — mirroring the same discipline
   `annotator_type` already applies to distinguish genuine-human from legacy Phase
   2B rows.
3. **A model proposal is never ground truth until a human accepts or edits it** —
   restated because it is the load-bearing constraint of this entire stage.
4. This workflow is strictly cheaper per-box than drawing from scratch (accept/
   reject/nudge vs. draw), making bbox-level ground truth realistically obtainable
   for the first time.

### Extending Ontology V2 safely through this workflow

A class discovered via Stage F (§8) only becomes a first-class Ontology V2 member
after (a) a human review step accepts it (§8.2) and (b) it accumulates enough
model-assisted-and-human-confirmed annotations to clear the same
`MIN_POSITIVE_SUPPORT` floor every existing class is already held to
(`phase2b/evaluation.py::MIN_POSITIVE_SUPPORT`, reused, not replaced) — never
promoted on the strength of the model's own confidence score alone.

---

## 13. Thesis Experiment / Ablation Matrix (Stage K)

For every family: research question (RQ), independent variable (IV), controlled
variables (CV), dependent metrics (DV), required data, expected contribution, and
explicitly what conclusion would/would not be justified.

### Family 1 — Object Model Comparison (CLIP vs. Grounding DINO vs. OWLv2 vs. YOLO-World vs. Florence-2)

- **RQ**: Which open-vocabulary detector best separates present/absent object
  classes on DOAR children's line drawings specifically?
- **IV**: detector architecture/checkpoint.
- **CV**: same 80-image cohort, same genuine-human labels, same vocabulary/prompt
  intent, same cohort split, no threshold tuning in the first pass.
- **DV**: precision/recall/F1/specificity/balanced-accuracy/ranking-separation per
  class, inference time, model size.
- **Data required**: presence-level labels only (already sufficient, §2).
- **Contribution**: the thesis's own, DOAR-specific answer to "which off-the-shelf
  open-vocabulary detector transfers best to this domain" — not assumed from any
  COCO/LVIS leaderboard.
- **Justified conclusion**: "on this 80-image DOAR cohort, model X shows the best
  [specific metric] for [specific classes]."
- **Not justified**: "model X is the best open-vocabulary detector" (unqualified,
  domain-general) — the sample is small and DOAR-specific by design.

### Family 2 — Model Adaptation (zero-shot vs. calibrated zero-shot vs. frozen/linear vs. partial FT vs. PEFT vs. full FT)

- **RQ**: How much does each adaptation tier improve over zero-shot, per unit of
  additional annotation/compute cost?
- **IV**: adaptation tier (§11's FT-0…FT-4 ladder).
- **CV**: same model backbone held fixed while tier varies; same dev-eligible
  split; locked-test never used for tuning at any tier.
- **DV**: same metrics as Family 1, plus annotation volume consumed and
  wall-clock/compute cost per tier.
- **Data required**: FT-1 onward needs real bbox ground truth (§12) at growing
  volume — **not available today**.
- **Contribution**: an empirical cost/benefit curve for adaptation depth on this
  specific, small, sketch-domain dataset.
- **Justified**: "tier Y gives Z% improvement over zero-shot for N additional
  annotated boxes."
- **Not justified**: claiming a tier is "production-ready" from this scale alone.

### Family 3 — Object Representation (presence only vs. presence+count vs. bboxes vs. masks)

- **RQ**: Does adding count and/or localization information change what a
  downstream rule/fusion stage can support, independent of detector accuracy?
- **IV**: representation granularity.
- **CV**: same detector, same images.
- **DV**: which existing/candidate rules become evaluable at each granularity (a
  qualitative dependent variable, not just a number) — e.g. §6.4's rule list.
- **Data required**: presence+count already exist; bbox/mask do not (§2, §12).
- **Contribution**: a concrete map from "what we annotate" to "what we can
  scientifically claim."
- **Justified**: "X rules become evaluable once bboxes exist, vs. Y today."
- **Not justified**: assuming finer representation automatically improves
  downstream psychological validity — it only improves measurement precision.

### Family 4 — Context (individual objects vs. objects+co-occurrence vs. objects+relationships)

- **RQ**: Does adding co-occurrence/relationship evidence improve construct-level
  agreement with any existing rule's own evaluation criteria, beyond per-object
  presence alone?
- **IV**: context layer included.
- **CV**: same object-detection layer held fixed across conditions.
- **DV**: rule-level evaluability/coverage (qualitative) plus any quantitative
  proxy the specific rule under test defines.
- **Data required**: relationship layer needs bboxes (§6.1) — not available today.
- **Contribution**: tests whether the architecture's own premise (context matters
  beyond individual objects) is empirically supported for *this* system, not
  assumed.
- **Justified**: "adding co-occurrence changes evaluability for rule R."
- **Not justified**: "objects + context proves X psychological pattern" — context
  evidence stays observation-only (Stage L invariant).

### Family 5 — Global Context (objects only vs. global only vs. objects+global fusion)

- **RQ**: Does a whole-image embedding capture presence-relevant or
  construct-relevant signal that per-object detection misses?
- **IV**: branch(es) included.
- **CV**: same 80-image cohort/labels; same fusion mechanism held simple (e.g.
  concatenation or late-fusion) to isolate the information-source question from a
  fusion-architecture question.
- **DV**: same presence metrics as Family 1, computed per branch and per fusion.
- **Data required**: none beyond what exists (global embeddings are a
  representation choice, not a new label type).
- **Contribution**: directly answers §7's open question empirically.
- **Justified**: "global-only achieves [X] vs. objects-only [Y] vs. fusion [Z] on
  this cohort."
- **Not justified**: assuming fusion always wins — Family 6 explicitly guards
  against this same fallacy at the whole-system level.

### Family 6 — Full DOAR Fusion (A=handcrafted features, B=objects, C=relationships/context, D=global)

- **RQ**: Which subset(s) of {A, B, C, D} actually improve a downstream target
  (initially the existing emotion-classification task, as the only target with
  enough labeled data to train against; construct-level targets once rule
  evaluability from Family 4 is established) — not assumed to be "more is better."
- **IV**: which of A/B/C/D are included (ablation grid: A, B, C, D, A+B, B+C, B+D,
  A+B+C, B+C+D, A+B+C+D — exactly the set specified).
- **CV**: same train/valid/test protocol already governing every existing DOAR
  model (validation-only tuning, locked test only at final evaluation).
- **DV**: task performance metric appropriate to whatever target is being
  predicted, held identical across all ablation cells.
- **Data required**: A and D exist now; B exists at presence-level now; C does not
  exist yet (needs bboxes, §6).
- **Contribution**: the thesis's central empirical claim — a real ablation, not an
  assumed architecture.
- **Justified**: "adding C improved [metric] over B alone by [X]" — cell-by-cell,
  evidenced.
- **Not justified**: "the full fusion A+B+C+D is the correct final architecture" —
  only if it is the empirically best cell, and even then framed as best-on-this-data,
  not proven-optimal-in-general.

### Family 7 — Ontology (current 10-class vs. expanded Ontology V2)

- **RQ**: What additional rule-evaluability/scientific coverage does Ontology V2
  buy, and at what reliability/annotation-cost tradeoff?
- **IV**: ontology version.
- **CV**: same detector/evaluation protocol across both ontology versions.
- **DV**: number of rules made evaluable (from §4's traceability table), per-class
  support/reliability (reusing `quality.classify_detector_readiness`'s existing,
  tested tiering), annotation cost incurred.
- **Data required**: Ontology V2 classes need their own annotation pass (§8/§12).
- **Contribution**: a direct, quantified answer to "was expanding the ontology
  worth it," not an assumption that more classes is automatically better.
- **Justified**: "V2 adds evaluability for rules {...} at a cost of {...} additional
  annotated instances, with reliability tier {...} per new class."
- **Not justified**: claiming V2 improves psychological validity — it only expands
  what can be *measured*, exactly like Family 3's boundary.

---

## 14. Final Target Architecture (Stage L)

Proposed, **not implemented**:

```
                             DRAWING
                                |
          -----------------------------------------
          |                    |                    |
          v                    v                    v
 objective/handcrafted    object detector       global encoder
 features (existing,      (Ontology V1 today,    (CLIP embedding today,
 features.py)             V2 candidate, §3)       DINOv2/Florence-2
          |                    |                    candidate, §7)
          |                    v
          |              object evidence
          |             (EvidenceRecordV2,
          |              evidence_type=
          |              "object_detection",
          |              existing schema)
          |                    |
          |             ----------------
          |             |              |
          |             v              v
          |       co-occurrence   relationships
          |       (§5, new         (§6, new
          |        evidence_type)   evidence_type)
          |             |              |
          |             ------+---------
          |                   |
          -----------+--------+--------+
                      v
                evidence fusion
              (Family 6 ablation, §14 —
               which subset wins is an
               empirical question, not
               assumed here)
                      |
                      v
             evidence/rule engine
           (rules_registry_v2.json,
            unchanged; CONCERNS_ENABLED
            stays False until this
            whole layer is validated)
                      |
           ---------------------
           |                   |
           v                   v
      Technical View       Parent View
      (unchanged this      (unchanged this
       phase and until      phase — hard
       validated)            constraint)
```

**Invariant, restated because it governs every box above**: object evidence,
co-occurrence evidence, relationship evidence, and global-image evidence all stay
`status`-graded observations (`PASS/WARN/FAIL/ABSTAIN`, the existing
`EvidenceRecordV2` vocabulary) until a specific, already-defined rule in
`rules_registry_v2.json` — with its existing evidence-grade and
multi-family-convergence requirements (`construct_registry.json`'s
`minimum_independent_evidence_families: 2`, unchanged) — consumes them. Nothing in
this architecture activates a rule or a construct on its own.

---

## 15. Expected Hardware / Runtime Constraints

- **This machine**: CPU-only, Intel UHD Graphics, no CUDA — unchanged from every
  prior phase's stated constraint.
- **Proven so far**: CLIP ViT-B-32 zero-shot inference over 80 images in ~21s once
  weights are cached (Phase 2C.2, real measurement).
- **Expected for the four new candidates** (estimates, not measurements — must be
  verified locally before any adoption decision, exactly as this document's own
  Stage G table already flags): OWLv2-base and Grounding DINO (Swin-T class) are
  likely a small-integer multiple of CLIP's per-image cost, given both run a
  comparable-or-larger backbone plus a detection head; YOLO-World's small variants
  are likely the fastest of the four; Florence-2-large (771M) is the most likely to
  be materially slower on CPU-only hardware than the others.
- **Download sizes** (approximate, network-dependent, same caveat Phase 2B's own
  model-selection doc already recorded for the CLIP checkpoint — ~600MB at
  ~2.9MB/s took ~3.5 minutes; this session's own CLIP re-download took ~22s,
  suggesting current network conditions are faster, but this must be re-measured
  per model, not assumed): roughly 460MB–1.5GB depending on which Florence-2
  variant, ~700MB-class for Grounding DINO's Swin-T checkpoint, ~800MB for
  OWLv2-base, tens-to-~200MB for YOLO-World variants.
- **Recommendation**: benchmark exactly one small-checkpoint candidate first
  (e.g. YOLO-World-s or OWLv2-base) to get a real local CPU-latency measurement
  before committing to evaluating all four at full scale — this is a scheduling
  recommendation for the next phase, not something this audit performed.

---

## 16. Exact Files Recommended to Create/Modify (future phases — none created this session)

- `src/doar/phase2c3/ontology_v2.py` — Ontology V2 candidate registry, mirroring
  `phase2b/ontology.py`'s existing structure (frozen dataclasses, `CLASSES`,
  `POSTPONED_CLASSES`, explicit `UNKNOWN` handling).
- `src/doar/phase2c3/cooccurrence.py` — §5's `EvidenceRecordV2`-based co-occurrence
  builder.
- `src/doar/phase2c3/relationships.py` — §6's geometric-relation functions
  (`left_of`, `above`, `relative_size`, etc.), pure functions over bbox pairs, unit
  tested with synthetic boxes exactly like every existing Phase 2C module.
- `src/doar/phase2c3/global_context.py` — wrappers for whichever global embedding(s)
  Stage H's benchmark ends up comparing, injectable-backend pattern (mirrors
  `phase2b/inference.py`'s existing `image_embed_fn`/`text_embed_fn` injection so
  tests never need real model weights).
- `src/doar/phase2c3/discovery.py` — §8's blinded-discovery orchestration
  (model-agnostic interface, injectable backend).
- `scripts/phase2c3_run_detector_benchmark.py` — mirrors
  `scripts/phase2b_run_baseline.py`'s existing structure, parameterized over which
  detector to run.
- `scripts/phase2c3_discover_concepts.py` — §8's discovery-pass driver.
- `docs/OBJECT_EVIDENCE_ONTOLOGY_V2.md` — the human-readable mirror of
  `ontology_v2.py`, exactly matching how `docs/OBJECT_EVIDENCE_ONTOLOGY.md` already
  mirrors `phase2b/ontology.py`.
- `docs/RELATIONSHIP_SCHEMA.md`, `docs/COOCCURRENCE_SCHEMA.md` — design docs for §5/§6's
  evidence-record shapes, at the same level of detail `docs/PAGE_REFERENCE_MODEL.md`
  already documents for the existing page-position feature.
- Extension to `phase2c_annotation_app.py` (or a new `phase2c3_bbox_review_app.py`)
  for §12's model-assisted bbox review workflow.
- `tests/test_phase2c3_*.py` — synthetic-fixture tests for every module above,
  matching the existing Phase 2C.1/2C.2 testing discipline exactly.

**None of these exist yet. None should be created before this document is approved.**

---

## 17. Recommended Execution Order for Remaining Phases

1. **Approval of this document** (current stop point).
2. **Stage F execution** (blinded discovery, one small model, one subset of
   images) — cheapest way to get real evidence on the §3.4 "not currently
   justified" candidates before committing to Ontology V2's final shape.
3. **Stage G/H execution, smallest candidate first** (§15's recommendation:
   one small-checkpoint model, measured locally) — establishes real DOAR-specific
   detector comparison data before investing in all four candidates.
4. **Ontology V2 finalization** — informed by both 2 and 3, not decided in
   isolation from either.
5. **Stage J (model-assisted bbox annotation)** using whichever detector Stage 3
   showed the most promise, to finally close the §2 bbox-coverage gap.
6. **Families 3–5 experiments** (representation/context/global) — only once bboxes
   exist (Family 3/4) or as soon as the global-branch benchmark is feasible
   (Family 5, no bbox dependency).
7. **Family 6 (full fusion ablation)** — the thesis's central empirical result,
   deliberately last, since it depends on every earlier stage's real data, not
   assumptions.
8. **Family 2 (fine-tuning ladder)** — only after Families 1–6 establish which
   model/representation/context combination is worth the additional annotation
   investment fine-tuning would require.

---

## Hard safety rules — restated, unchanged, and honored in every section above

No diagnosis; no psychological inference directly from any model prediction; no
abuse/trauma inference from object presence or co-occurrence; no Parent View
change; no detector/co-occurrence/relationship/global evidence activates any rule
in this phase; no emotion folder label used as supervision anywhere in this design;
no locked-test model selection/tuning (guarded structurally,
`assert_pilot_ids_exclude_locked_test`, unchanged); no fabricated labels; no
fabricated bounding boxes (§2's 0% coverage is reported honestly, not padded); no
fabricated second reviewer; no model-generated proposal treated as ground truth
(§8.2, §12); no large-scale annotation performed; no fine-tuning performed; no
automatic threshold tuning performed; no stable-branch merge; nothing pushed.

**This document is the stop point. Awaiting approval before any Stage F–L
implementation begins.**
