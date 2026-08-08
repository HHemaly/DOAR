# Phase 2C.4 — Modern Open-Vocabulary Detector Benchmark

**Status: execution phase. Real models installed and run on this CPU-only machine.**
Branch `feature/doar-phase2c-annotation-expansion`. Baseline reused unchanged:
Phase 2C.2's CLIP zero-shot + classical-CV results (commit `ad4011f`) — **not rerun**.

## 0. Preflight (verified this session)

| Check | Result |
|---|---|
| Branch | `feature/doar-phase2c-annotation-expansion` |
| HEAD (starting) | `9e52e13453f5d0f347f87752658fd7f548784237` |
| Working tree | clean |
| `PHASE2C3_AUDIT_AND_DESIGN.md` exists | yes |
| 80-image private pilot exists | yes (`outputs/phase2c1/private_images/`, 80 files) |
| Genuine-human annotations exist | yes (`outputs/phase2c1/annotation_store.csv`) |
| Private outputs gitignored | yes (`outputs/phase2c1/*`, `outputs/phase2c4/*` all match `.gitignore:5:outputs/`) |

## 1. Environment and installation outcomes

Added `pyproject.toml` optional group `phase2c4 = ["transformers>=4.40,<5",
"accelerate>=0.30,<1", "einops>=0.7,<1", "ultralytics>=8.1,<9"]`, installed via
`pip install -e ".[phase2c4]"`. Resolved versions: `transformers==4.57.6`,
`accelerate==0.34.2`, `einops==0.8.2`, `ultralytics==8.4.116`, `torch==2.13.0+cpu`
(unchanged — already installed, not upgraded).

**Side effect observed and verified non-breaking**: `ultralytics` pulled in
`opencv-python==5.0.0`, which resolves ahead of the project's existing
`opencv-python-headless` dependency for the shared `cv2` import name. Verified
directly: `cv2.__version__` reports `5.0.0`, `pip check` reports no broken
requirements, and Phase 2B's `detect_circles_classical()` reproduces its exact
previous output (`circularity=0.8984, count=38` for `p2b_0000`, matching Phase
2C.2's committed `raw_predictions_80.csv` byte-for-byte on that field).

**Disk space**: pre-session free space on `C:` was ~5.18 GiB. It dipped to
~2.75 GB during the concurrent model downloads/pip installs (four checkpoints
plus new packages, all within normal bounds for that footprint) and recovered
to ~7.48 GB once downloads finished. No unexplained usage was found on
read-only investigation; nothing was deleted or modified.

### Per-model installation/load outcome

| Model | Outcome | Blocker found | Fix applied |
|---|---|---|---|
| **OWLv2** (`google/owlv2-base-patch16-ensemble`, `transformers`) | **Ran successfully** | None | — |
| **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`, `transformers`) | **Ran successfully** | `post_process_grounded_object_detection()` doesn't accept the `box_threshold` kwarg name I initially used (an older API assumption) | Corrected to the installed `transformers==4.57.6`'s actual parameter name, `threshold` (verified via `inspect.signature` directly) |
| **Florence-2** (`microsoft/Florence-2-base`, `transformers`, `trust_remote_code=True`) | **Ran successfully, but only after two real fixes** | (1) `AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'` — Florence-2's custom remote modeling code predates transformers' newer attention-dispatch internals. (2) `AttributeError: 'NoneType' object has no attribute 'shape'` in `prepare_inputs_for_generation` — Florence-2's custom code still assumes the legacy raw-tuple `past_key_values` format, incompatible with transformers' newer Cache-object format. | (1) `attn_implementation="eager"`. (2) `use_cache=False` in `generate()` — this second fix is also the direct cause of Florence-2's much higher per-call cost (no KV-cache reuse across generation steps) |
| **YOLO-World** (`yolov8s-worldv2.pt` via `ultralytics`, not the original AILab-CVC/YOLO-World repo) | **Ran successfully, but with near-zero usable signal** | None (technical) — but confidence scores are far below its own documented default threshold on this domain (max ~0.05 observed even at a diagnostic `conf=0.001`, vs. the documented `conf=0.1` default) | Documented substitution, not silent: used Ultralytics' own MIT-licensed packaging of YOLO-World rather than the original AILab-CVC/YOLO-World repository (GPL-v3, heavier `mmyolo`/`mmdetection` dependency stack) — same published weights, different, actively-maintained inference wrapper |

**No model was silently omitted.** Every blocker above was reproduced, diagnosed,
and either fixed (with the fix documented in `src/doar/phase2c4/detectors.py`'s own
docstrings) or reported as a real, measured domain-transfer finding (YOLO-World).
**No GPU/Colab run was required to get all four models running** — all four run on
this CPU-only machine — but see §6 for where a GPU would materially help
(Florence-2's runtime specifically).

## 2. Prompts (exact, documented)

Identical bare noun-phrase vocabulary across all five models (CLIP included, for
reference), differing only in each model's own required template:

| Model | Template | Example (`circle`) |
|---|---|---|
| CLIP (Phase 2C.2, unchanged) | `"a child's drawing containing a {cls}"` | `"a child's drawing containing a circle"` |
| OWLv2 | `"a {cls}"` | `"a circle"` |
| Grounding DINO | `"{cls}. "` joined, one string for all 10 classes | `"person. face. hand. animal. house. tree. heart. star. circle. vehicle."` |
| Florence-2 | bare noun, one call per class | `"circle"` (with task token `<OPEN_VOCABULARY_DETECTION>`) |
| YOLO-World | bare noun via `model.set_classes([...])` | `"circle"` |

No prompt was engineered after seeing any result. `src/doar/phase2c4/detectors.py`'s
`owlv2_text_queries`/`grounding_dino_text_prompt`/`florence2_prompt_for_class`
functions are the single, auditable source of every prompt actually sent.

## 3. Operating thresholds (exact, documented, never tuned against these results)

| Model | Threshold(s) | Source |
|---|---|---|
| OWLv2 | `threshold=0.1` | OWLv2's own HuggingFace model-card example usage for `post_process_grounded_object_detection` |
| Grounding DINO | `threshold=0.25, text_threshold=0.25` | `transformers`' own function-signature defaults for `GroundingDinoProcessor.post_process_grounded_object_detection`, verified via `inspect.signature` this session |
| Florence-2 | N/A (task-based, no separate confidence threshold exposed) | `<OPEN_VOCABULARY_DETECTION>` returns any matching region directly; presence = at least one returned bbox |
| YOLO-World | `conf=0.1` | Ultralytics' own documented open-vocabulary usage examples |

## 4. Rule visual-coverage analysis

All 41 rules in `rules_registry_v2.json` classified (`src/doar/phase2c4/rule_coverage.py`,
tested against the live registry file for completeness):

| Classification | Count |
|---|---|
| `already_measurable_by_objective_features` | 10 |
| `potentially_measurable_by_benchmarked_detector` | 7 |
| `requires_new_object_or_part_class` | 7 |
| `requires_object_attribute` | 7 |
| `requires_bbox_localization` | 3 |
| `requires_process_information` | 3 |
| `requires_longitudinal_information` | 3 |
| `remains_not_operational` | 1 |

**7 of 41 rules (17%) are, in principle, already coverable by the benchmarked
detector vocabulary** (plain presence of an existing ontology class):
`PSY_AR_STARS_009`, `PSY_AR_CIRCLES_011`, `PSY_AR_TRANSPORT_012`,
`PSY_AR_HEARTS_013`, `EN_COMPILED_ANIMAL_CHOICE_GENERAL_022`,
`EN_COMPILED_HOUSE_023`, `EN_COMPILED_TREE_024`. Full table with per-rule
rationale: `artifacts/phase2c4/RULE_VISUAL_COVERAGE_AFTER_DETECTOR_BENCHMARK.csv`.
**No rule was activated or had its `allowed_output_level` changed by this
analysis.**

---

## 5. Per-model / per-class results (full 80-image cohort, descriptive)

Ground truth throughout: genuine Phase 2C.1 human annotation only
(`annotator_type == "human"`), never legacy Phase 2B provisional labels
(structurally enforced and tested, `ground_truth.genuine_human_ground_truth`).
CLIP row reused unchanged from Phase 2C.2 (`ad4011f`) — **not rerun**.

| Model | Class | n present | TP | FP | FN | TN | Precision | Recall | Specificity | Balanced acc. | Ranking sep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP (baseline) | person | 59 | 2 | 0 | 57 | 21 | 1.000 | 0.034 | 1.000 | 0.517 | 0.531 |
| CLIP | face | 71 | 4 | 0 | 67 | 9 | 1.000 | 0.056 | 1.000 | 0.528 | 0.264 |
| CLIP | house | 14 | 4 | 0 | 10 | 66 | 1.000 | 0.286 | 1.000 | 0.643 | 0.740 |
| CLIP | tree | 15 | 4 | 1 | 11 | 63 | 0.800 | 0.267 | 0.984 | 0.625 | 0.738 |
| **OWLv2** | person | 59 | 18 | 0 | 41 | 21 | 1.000 | 0.305 | 1.000 | 0.653 | 0.653 |
| **OWLv2** | face | 71 | 27 | 1 | 44 | 8 | 0.964 | 0.380 | 0.889 | 0.635 | 0.618 |
| **OWLv2** | hand | 43 | 9 | 2 | 34 | 35 | 0.818 | 0.209 | 0.946 | 0.578 | 0.582 |
| **OWLv2** | house | 14 | 10 | 0 | 4 | 66 | 1.000 | 0.714 | 1.000 | **0.857** | 0.857 |
| **OWLv2** | tree | 15 | 7 | 0 | 8 | 64 | 1.000 | 0.467 | 1.000 | 0.733 | 0.733 |
| **OWLv2** | heart | 5 | 5 | 8 | 0 | 67 | 0.385 | 1.000 | 0.893 | **0.947** | 0.947 |
| **OWLv2** | circle | 4 | 0 | 15 | 4 | 61 | 0.000 | 0.000 | 0.803 | 0.401 | 0.401 |
| Grounding DINO | person | 59 | 48 | 8 | 11 | 13 | 0.857 | 0.814 | 0.619 | 0.716 | 0.783 |
| Grounding DINO | face | 71 | 51 | 1 | 20 | 8 | 0.981 | 0.718 | 0.889 | 0.804 | 0.836 |
| Grounding DINO | animal | 13 | 11 | 35 | 2 | 32 | 0.239 | 0.846 | 0.478 | 0.662 | 0.606 |
| Grounding DINO | circle | 4 | 1 | 48 | 3 | 28 | 0.020 | 0.250 | 0.368 | 0.309 | 0.312 |
| Florence-2 | *every class* | — | =n_present | huge | **0** | **0** | ≈base rate | **1.000** | **0.000** | ≈0.500 | undefined |
| YOLO-World | face | 71 | 5 | 0 | 66 | 9 | 1.000 | 0.070 | 1.000 | 0.535 | 0.535 |
| YOLO-World | animal/house/heart/star/vehicle | — | 0 | 0 | all | all | — | 0.000 | 1.000 | 0.500 | 0.500 |

Full 120-row table (4 models x 10 classes x 3 cohorts):
`artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv`. False-positive/
false-negative pilot_ids: `artifacts/phase2c4/phase2c4_fp_fn_examples.csv`.

### Macro-averaged comparison (mean across all 10 classes, `full_80` cohort)

| Model | Macro balanced accuracy | Macro F1 | Macro ranking-separation | # classes with bal.acc. > 0.6 |
|---|---|---|---|---|
| **OWLv2** | **0.642** | **0.467** | 0.642 | 6 / 10 |
| Grounding DINO | 0.604 | 0.399 | **0.668** | **7 / 10** |
| CLIP (baseline) | 0.539 | 0.246 | 0.535 | 2 / 10 |
| YOLO-World | 0.518 | 0.218 | 0.518 | 0 / 10 |
| Florence-2 | 0.501 | 0.386 | undefined | 0 / 10 |

**Reading this table honestly (per instruction, not inferring quality from raw
counts)**: Florence-2's macro F1 (0.386) looks numerically competitive with
Grounding DINO's, but this is an artifact of recall being fixed at 1.000 for
every class — Florence-2's `<OPEN_VOCABULARY_DETECTION>` task returns at least
one bounding box for essentially every prompted class on essentially every image
(79-80/80), regardless of true content. Its balanced accuracy (0.501, exactly
chance level) is the metric that actually reveals this: **Florence-2-base, used
this way, provides no discriminative presence/absence signal on this dataset at
all.** This is the concrete reason §7 does not recommend it.

## 6. Runtime and feasibility (measured, not estimated)

**Caveat, stated plainly**: OWLv2, Grounding DINO, and Florence-2 ran with
overlapping wall-clock windows (concurrent background processes competing for
this machine's CPU) to fit within the session — their measured per-image times
below are inflated by mutual contention, not clean isolated measurements.
YOLO-World's run also overlapped with the tail of the other two. Relative
ordering (which model is fastest) is still informative; absolute numbers are
upper bounds, not best-case figures.

| Model | Load time | Total inference (80 img) | Mean/image | Checkpoint size class |
|---|---|---|---|---|
| CLIP (Phase 2C.2, isolated run) | ~22s | ~21s | **~0.26s** | ~350MB |
| YOLO-World | 92.2s | 283.4s | **3.54s** | ~25MB (+ ~338MB CLIP text encoder, auto-installed) |
| OWLv2 | 44.3s | 3978.0s | 49.72s (contended) | ~800MB |
| Grounding DINO | 251.4s | 3622.1s | 45.28s (contended) | ~700MB |
| Florence-2 | 37.3s | 16357.9s | **204.47s** (contended, `use_cache=False`) | ~460MB |

**Florence-2 is the clear outlier** even accounting for contention — its
`use_cache=False` workaround (§1) means every one of its 800 class-queries
(10 classes x 80 images) re-computes generation from scratch with no KV-cache
reuse, a structural cost the other three models don't pay (each does one batched
forward pass per image covering all 10 classes at once). A GPU would help every
model, but would help Florence-2 proportionally the most, since generation-based
decoding parallelizes far better on GPU than CPU.

**Peak memory**: all four models' processes stayed in the ~1.0-1.2GB working-set
range (measured via `Get-Process`), comfortable on this machine even under
3-way concurrency.

**Installation/dependency complexity** (recap from §1): OWLv2 and Grounding DINO
were the smoothest (`transformers`-native, one real bug each, both fixed in
minutes). YOLO-World required an unprompted auto-install of a second package
(`ultralytics/CLIP`) the first time `set_classes()` was called with open
vocabulary. Florence-2 required two non-obvious workarounds for its
`trust_remote_code=True` custom modeling code's incompatibility with current
`transformers` internals.

## 7. Model selection — DOAR-specific, not COCO/LVIS-derived

Evaluated on the 8 requested criteria, from this session's own real results only:

1. **Detection quality (macro balanced accuracy)**: OWLv2 > Grounding DINO > CLIP
   > YOLO-World ≈ Florence-2 (chance).
2. **Recall** (important — missed evidence is harmful): Grounding DINO is
   substantially highest on person/animal/house/tree/heart (0.7-0.93+), at a real
   precision cost on the non-face/person classes. OWLv2 has moderate recall gains
   over CLIP on house/tree/heart. YOLO-World and CLIP are both very conservative
   (low recall almost everywhere).
3. **Precision** (avoid false evidence entering rules): OWLv2 and YOLO-World are
   the most precise where they fire at all (often 0.8-1.0). Grounding DINO's
   precision collapses below 0.25 on 6 of 10 classes — a real risk if used to feed
   evidence directly into a rule without a stronger gate. Florence-2's precision
   equals base rate everywhere (no useful signal).
4. **Performance across object types**: no single model is best on every class.
   OWLv2 leads on `house`/`heart`/`tree`. Grounding DINO leads on `person`/`face`/
   `animal` (recall-heavy). All five models are weak-to-uninformative on `circle`,
   `star`, `vehicle` — consistent with Phase 2C.2's own finding that these classes
   lack real signal in *any* baseline tried so far, not just CLIP.
5. **Runtime/resource requirements**: YOLO-World is fastest by a wide margin;
   Florence-2 is the clear outlier, impractical for CPU-only iteration at this
   configuration.
6. **Reproducibility**: all four are deterministic given fixed weights/input (no
   sampling used anywhere — `do_sample=False`/`num_beams=1` throughout); not
   independently re-verified by a second run this session due to time, unlike
   CLIP's confirmed byte-identical reproduction in Phase 2C.2.
7. **Localization for future rule-critical features**: OWLv2 and Grounding DINO
   both return real bounding boxes (not used for evaluation this phase, per
   instruction, but available); YOLO-World also returns boxes; Florence-2's boxes
   are undermined by its lack of discrimination — a box for "circle" on an image
   with no circle is not useful evidence regardless of its coordinates.
8. **Ontology expansion / fine-tuning support**: all four are open-vocabulary by
   design (arbitrary new class names require no architecture change) and all sit
   on transformer backbones amenable to the FT-1/FT-3 tiers in §10.

### Recommendation

- **Best overall candidate: OWLv2.** Highest macro balanced accuracy and F1,
  best precision/recall balance, materially beats the CLIP baseline (0.642 vs.
  0.539 macro balanced accuracy; 6 vs. 2 classes clearing a reasonable bar) at a
  runtime cost that's real but not prohibitive.
- **Best lightweight candidate: YOLO-World** (by far the fastest), but its
  recall is too low on this domain to recommend over OWLv2 for actual evidence
  quality — a candidate for a future recall-boost pass (e.g. a lower threshold),
  not for this phase's fixed-threshold protocol.
- **Best candidate for future rule-critical (localization) work: OWLv2 or
  Grounding DINO** — both return real, usable boxes; Grounding DINO's much higher
  recall could make it the better *candidate generator* in a model-assisted
  annotation workflow (§ Phase 2C.3 Stage J) even though its raw precision is too
  low to trust unreviewed.
- **Not worth continuing at this configuration: Florence-2-base.** Zero
  discriminative value as an open-vocabulary presence detector on this dataset,
  and the slowest by more than an order of magnitude. A different Florence-2 task
  prompt (e.g. `<CAPTION_TO_PHRASE_GROUNDING>` instead of
  `<OPEN_VOCABULARY_DETECTION>`) or the larger checkpoint might behave
  differently — not tested this session, and not a small change given the
  measured runtime cost.

**Does the best model materially beat CLIP?** Yes — OWLv2 improves macro
balanced accuracy by +0.103 and macro F1 by +0.221 over the existing baseline,
with 3x as many classes clearing a reasonable usefulness bar (6 vs. 2). This is
a real, measured improvement, not a marginal one.

## 8. Error analysis (false positives / false negatives, `full_80` cohort)

| Model | Dominant failure mode |
|---|---|
| OWLv2 | Under-detection (high FN) on `person`/`face`/`hand` (conservative threshold costs recall); real over-firing on `circle`/`heart`/`star` (FP 8-15) |
| Grounding DINO | Severe over-firing (FP 34-51 of ~65-76 negatives) on `animal`/`house`/`tree`/`heart`/`star`/`circle` — the model appears to treat these small-symbol classes as almost always "present"; `person`/`face` are comparatively well-behaved |
| Florence-2 | Total over-firing — FP equals (n_absent) almost exactly on every class, FN is always 0 |
| YOLO-World | Severe under-firing — FN dominates almost every class; several classes never fire at all (0 TP and 0 FP) |

Every model's false positives/negatives are traceable to specific opaque
`pilot_id`s in `artifacts/phase2c4/phase2c4_fp_fn_examples.csv` — no example is
summarized here beyond aggregate counts, consistent with keeping this document
free of anything that could re-identify a specific drawing's content pattern.

---

*Sections 9-11 below are design-only, written earlier in this session, and do
not depend on the results above.*

## 9. Next rule-driven annotation extension (design only — not begun)

Derived directly from §4's rule-coverage table, prioritizing by how many disabled
rules each addition would unlock and how directly the repository's own audits
(Phase 2B, Phase 2C.3) already flagged the gap — **not an arbitrary expansion**:

1. **`eye` as a locatable part + a small state vocabulary** (present/absent,
   open/closed) — unlocks 4 rules (`PSY_AR_EYES_WIDE_001/STERN_002/CLOSED_003`,
   `EN_COMPILED_EYES_MISSING_DETAIL_020`), the single highest-value addition by
   rule count.
2. **`mouth` as a locatable part** — unlocks 1 rule directly
   (`EN_COMPILED_MISSING_MOUTH_036`) and is a prerequisite the existing
   `dark_colors_sad_faces_isolation` (038) compound rule also depends on
   indirectly via `face`.
3. **Bounding boxes for the already-benchmarked `hand`** — unlocks the
   relative-size half of `EN_COMPILED_EXAGGERATED_BODY_PARTS_037` and the
   omission half of `EN_COMPILED_MISSING_HANDS_035`, without adding a single new
   class.
4. **A `face_expression` state label** (a small, closed vocabulary — e.g.
   happy/neutral/sad/angry-looking, drawn expression only, never inferred mood) on
   top of the existing `face` class — unlocks `EN_COMPILED_FACE_EXPRESSION_021`.
   Flagged explicitly as the technically hardest of the four (no existing labeled
   taxonomy for drawn facial expression anywhere in this project).

**Not proposed**: a general-purpose 50-class ontology, species-level animal
labels (explicitly forbidden regardless of technical feasibility, unchanged
policy), or any class not traced to a specific rule or a Stage-F discovery result.

### Model-assisted workflow for this extension (design only)

Per Phase 2C.3 §12 (Stage J) and this session's real results (§5-7 below): the
detector `predicted_positive_from_*` output for `hand`/`eye`-region candidates
would seed proposed boxes in an extension of `phase2c_annotation_app.py`, with the
same accept/reject/edit/relabel/uncertain/not_assessable vocabulary Phase 2C.1's
schema already enforces, plus a `bbox_source` provenance field distinguishing
`model_proposed` from `human_drawn`/`human_edited` — **a proposal is never ground
truth until a human accepts or edits it.** Not implemented this session.

## 10. Future fine-tuning feasibility (design only — not attempted)

| Tier | OWLv2 | Grounding DINO | Florence-2 | YOLO-World |
|---|---|---|---|---|
| FT-0 (zero-shot) | **Done this session** | **Done this session** | **Done this session** | **Done this session** |
| FT-1 (frozen backbone + lightweight head) | Feasible — standard `transformers` linear-probe pattern over its CLIP-family visual backbone | Feasible — same transformer-backbone pattern | Feasible in principle, but its autoregressive generation interface makes a simple linear probe less natural than for the other three (would need to probe an intermediate encoder representation instead of the generation head) | Feasible via Ultralytics' own fine-tuning API |
| FT-2 (partial/last-layer fine-tuning) | Feasible, needs more annotated data than FT-1 and more CPU time than this machine comfortably offers (untested here) | Same | Same caveat as FT-1, compounded | Feasible, same data/compute caveat |
| FT-3 (PEFT/LoRA/adapters) | Plausible (transformer backbone) — not verified in practice this session | Plausible — not verified | Plausible (transformer backbone exists under the generation head) — not verified | Less standard in the current Ultralytics tooling for the open-vocab text branch specifically |
| FT-4 (full fine-tuning) | Only if annotation volume grows far past current levels (§ below) | Same | Same, and least likely to be worth the CPU cost given FT-0's own ~50s+/image cost already measured this session | Same |

**Additional human-labeled data each tier would require** (informed by real support
counts from Phase 2C.1/2C.2, not invented): FT-1 needs bounding boxes (§2 of
`PHASE2C1_POST_ANNOTATION_REPORT.md` — currently 0% coverage) at minimum for
whichever classes are targeted, at a volume well past the existing
`MIN_POSITIVE_SUPPORT=5` floor — realistically dozens to low-hundreds of boxed
positives per class, consistent with Phase 2C.3's own FT-1 estimate. FT-2/FT-3
need more again, plus a genuinely held-out validation slice never touching the 7
locked-test images. **No fine-tuning was performed or hyperparameter-selected in
this phase**, and none may use the locked test set for either purpose, per
instruction (`workspace.assert_pilot_ids_exclude_locked_test`, unchanged,
available for whenever this work actually begins).

## 11. Future rule-engine experiment protocol (design only — not implemented)

Comparing three rule-execution strategies once the visual evidence layer above is
validated — **not run in this phase**:

| | R1: current hard deterministic | R2: fuzzy continuous-measurement handling | R3: evidence/confidence-weighted aggregation with safety gates |
|---|---|---|---|
| **What stays identical** | The rule registry itself (`rules_registry_v2.json`), the construct definitions (`construct_registry.json`), the `minimum_independent_evidence_families: 2` convergence requirement, every safety wording constraint (`judges.py::DIAGNOSTIC_PATTERNS`), and — critically — **the ontology and detector layer are the same evidence source for all three; only how a rule consumes that evidence changes.** |
| **What changes** | A rule fires or doesn't based on a fixed threshold on a boolean/categorical measurement (today's actual behavior for the 10 already-executable rules). | A rule's activation strength becomes a continuous function of the underlying measurement (e.g. how far `bounding_box_coverage` sits past its threshold), rather than a step function — still deterministic, no learned weights. | A rule's contribution to a construct is weighted by the evidence record's own `confidence` field (already part of `EvidenceRecordV2`, unused for this purpose today), with an explicit, hand-authored floor below which a rule may never fire regardless of aggregate weight (the "safety gate"). |
| **Novelty vs. today** | None — this is the existing, already-shipped behavior, kept as the control condition. | Genuinely new: requires defining a continuous response curve per rule, not just a threshold — itself a research question (linear? sigmoid? something threshold-anchored?). | Genuinely new: requires deciding how confidence combines across independent evidence families without letting several low-confidence signals fake a high-confidence conclusion — the safety gate exists specifically to prevent that failure mode. |

### How psychologist expert review evaluates the three

A blinded comparison: the same set of real (already-annotated) drawings run through
all three strategies, with a psychologist reviewer seeing only the resulting
observations/hypotheses (never which strategy produced which, mirroring this
project's existing anti-anchoring review pattern from Phase 2C.1). Metrics:

- **Expert agreement**: does the reviewer judge the output as a reasonable
  reading of the drawing, independent of strategy.
- **False activations**: constructs/rules the reviewer judges as NOT supported by
  the drawing but which fired anyway.
- **Missed relevant rules**: constructs the reviewer judges as plausibly supported
  but which did not fire.
- **Stability**: how much a strategy's output changes under small, meaning-
  preserving input perturbations (the existing `docs/FEATURE_ROBUSTNESS_RESULTS.md`
  transformation-invariance discipline, extended to rule-level output rather than
  feature-level).
- **Abstention rate**: how often each strategy correctly declines to produce any
  hypothesis (an `ABSTAIN`-status outcome) versus forcing a low-confidence one —
  R1's binary nature may abstain "correctly" more often almost by construction,
  which the comparison must account for, not treat as an automatic win.
- **Explanation traceability**: whether a human can reconstruct *why* a given
  hypothesis fired from the recorded evidence chain (`source_evidence_ids`) alone
  — R3's confidence-weighted aggregation is the hardest to keep traceable and
  should be judged partly on whether it stays auditable, not just accurate.

**Hard constraint, restated**: no learned or fuzzy rule strategy (R2/R3) may use
the locked test set for tuning any parameter, threshold curve, or weight — the
existing `assert_pilot_ids_exclude_locked_test` guard applies here exactly as it
would to any detector training. Nothing in this section has been implemented.
