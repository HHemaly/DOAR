# Phase 3 — Detector Evaluation Plan (planning + validation prep only)

Status: **planning document, nothing implemented**. No detector has been adopted,
downloaded, or enabled. No rule's `activation_status` has changed from Phase 2.
Companion to `DETECTOR_DEPENDENCY_MATRIX.csv` (per-rule detail) and
`DECISION_LOG.md` (approvals).

---

## 1. Method

For each of the 13 Tier-2 rules, `DETECTOR_DEPENDENCY_MATRIX.csv` records the
required object/feature, detector capability, task type, free-drawing
applicability, candidate models/datasets, licence/compute, required
annotations, validation metrics, and — critically — whether the rule stays
scientifically unvalidated even with a working detector (it always does; see
`RULE_SOURCE_REGISTER.csv`; a detector can validate a *geometric observation*,
never the *psychological interpretation* attached to it in the source PDF).

Candidates were researched, not recalled from memory alone — several claims
below (COCO's exact class list, licence terms, dataset existence) were
verified against primary sources this session; each is marked accordingly.

---

## 2. Candidate comparison by detector category

### 2.1 Person / face / body-structure (3 rules: wide_eyes, stern_eyes, closed_eyes)

| Candidate | Domain | Verified facts | Fit for DOAR |
|---|---|---|---|
| MediaPipe Face Mesh / dlib 68-point landmarks | Photo | Apache 2.0 / Boost licence, CPU-feasible, extremely mature | Trained on photographs of real faces; performance on children's line-drawn faces is **untested** and plausibly poor — face geometry conventions in child drawings (circle + dots + lines) differ structurally from photographic faces |
| Haar cascade face detector (OpenCV) | Photo | Ships with full `opencv-python` but **not** with `opencv-python-headless` (confirmed: `cv2.data.haarcascades` is empty in this venv) — would need a small separate XML download | Same domain-gap concern as above, likely weaker than MediaPipe/dlib, but nearly free to pilot once the cascade file is obtained |
| ChildlikeSHAPES + CharSegNet (Meta, arXiv:2504.08022, Apr 2025) | Drawing (unconfirmed authenticity) | 16,075 pixel-annotated "childlike" figure drawings, 25 semantic parts, built for a figure-**animation** pipeline | Closest domain match found — **but** whether the drawings are real scanned children's drawings or artist-created "childlike style" art for animation is not established from the paper abstract and must be checked at the source before relying on it. License/access terms also unconfirmed. |
| torchvision `keypointrcnn_resnet50_fpn` (person keypoints) | Photo | Included in the already-installed `torchvision`, but pretrained **weights require a first-use download** (hundreds of MB) — not fetched this session per your "no downloads yet" instruction | COCO-person-keypoint domain, same photo-vs-drawing gap |

No candidate here is drawing-specific *and* confirmed to be real child-drawn
data. **This category needs a source-verification step (confirm ChildlikeSHAPES'
provenance and licence) before any pilot is worth running.**

### 2.2 Shapes, symbols, and common objects (6 rules: repeated_geometric_shapes, stars, flowers_clouds_sun, circles, vehicles, hearts)

| Candidate | Domain | Verified facts | Fit for DOAR |
|---|---|---|---|
| Quick, Draw! (Google) | Sketch (adult/general-user-drawn, not children) | **Verified**: 345 categories, CC-BY 4.0 licence, 50M drawings; includes star/circle/flower/cloud/sun/heart-type categories and vehicle categories (exact spelling to confirm against `categories.txt`) | Per your instruction, treated as a **classification-of-isolated-sketch candidate only** — but DOAR already has connected-component extraction (`features.py`) that could supply the "isolated crop" a QuickDraw-style classifier expects, narrowing (not closing) the detection-vs-classification gap you flagged |
| ESRA (Roboflow Universe) | Children's drawings (claimed) | Found this session: ~3,002 images, "objects inside kids' drawings," ships with a pretrained detector | Most directly on-domain candidate found — but licence, annotation quality, and exact category list are **unverified community-hosted data**; must be checked before any reliance |
| Classical CV (contour circularity / Hu moments) | N/A — no model | Zero licence/compute/download cost; `features.py` already extracts connected components | Plausible for `circles` specifically (geometric circularity is directly measurable) without any learned model at all — cheapest possible first experiment for one rule |
| COCO-pretrained detector | Photo | **Verified**: COCO's vehicle classes are car/truck/bus/train/airplane/boat/bicycle/motorcycle — directly matches the `vehicles` rule's vocabulary | Only rule in this whole matrix where an off-the-shelf class vocabulary already lines up; photo-vs-line-drawing domain gap still untested |

**This is the most feasible category**: cheapest compute, most directly-relevant
public data (QuickDraw), an existing DOAR component-extraction step to build on,
and one rule (`circles`) plausibly solvable with zero learned model.

### 2.3 Animals (4 rules: tiger_or_wolf, fox, squirrel, lion)

| Candidate | Domain | Verified facts | Fit for DOAR |
|---|---|---|---|
| COCO-pretrained detector | Photo | **Verified**: COCO's 10 animal classes are bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe. **None of tiger/wolf/fox/squirrel/lion are COCO classes.** | Not usable out of the box for any of these 4 rules |
| iNaturalist-trained species classifiers (iNat2021 etc.) | Photo (nature photography) | Cover these species | Real photographs of animals in nature — very large, untested domain gap to a child's 6-line drawing of a fox |
| Quick, Draw! | Sketch | Per your instruction: classification-only, isolated-sketch candidate; exact category coverage for these 5 species not yet confirmed | Weakest fit of the three categories even before considering the domain gap |
| Drawing-specific / children's-drawing-specific animal dataset | — | **None found** in this session's search | — |

**This is the least feasible category.** No candidate's class vocabulary
matches, no drawing-domain data was found, and fine-grained species
identification in child line-art (tiger vs. wolf vs. dog) is plausibly a hard
research problem in its own right, not a routine transfer-learning exercise.
Recommend treating annotation-effort estimation itself as the first gating
question here, before any modeling work — see §5.

---

## 3. Prioritization

| Category | Rules | Feasibility | Best candidate found | Priority |
|---|---|---|---|---|
| Shapes/symbols/objects | 6 | **Highest** — cheap compute, relevant public data, reuses existing DOAR code, one rule (circles) needs no model at all | QuickDraw (classifier only) over DOAR's own component crops; classical CV for circles | **1st** |
| Person/face/eyes | 3 | Medium — mature face-detection tech exists, but eye-*state* classification has no precedent and the one promising drawing-domain dataset (ChildlikeSHAPES) has unconfirmed provenance | ChildlikeSHAPES (pending verification) or photo-domain landmarks as a weaker fallback | 2nd |
| Animals | 4 | **Lowest** — zero matching pretrained vocabulary, no drawing-domain data found, likely genuine research problem | None confident; iNaturalist as a weak experimental starting point only | 3rd, possibly permanently deferred |

This ordering is driven by feasibility and rule-count, as you asked — not by
"scientific value" in the sense of validating any psychological claim, since
**no candidate in any category can do that** (§1). "Value" here means: does a
successful detector let DOAR make *any* additional honest, evidence-traceable
geometric observation. All three categories would, equally; feasibility breaks
the tie.

---

## 4. Should annotation and validation precede implementation?

**Yes, unambiguously**, for a reason specific to this project: DOAR's own
dataset (`Combined_Drawing`) has no object-level annotations at all — every
candidate above was pretrained on *other* data. There is currently no way to
know whether any candidate works on DOAR's actual images without first
building a small held-out annotated sample to test against. Skipping straight
to "download and try a detector" would mean judging it against nothing.

Recommended sequence:
1. Build one shared held-out annotation sample (not three separate ones) covering
   all three categories, so the one annotation effort serves every experiment.
2. For the animal category specifically, measure **human inter-rater
   agreement** on that sample *before* any model is judged — if two adult
   raters can't agree whether a child's drawing shows a fox or a dog, no
   detector should be held to a higher bar than that agreement level.
3. Only then pilot the highest-priority category (shapes/symbols) against the
   sample.

---

## 5. Annotation plan (schema in `docs/PHASE3_ANNOTATION_SCHEMA.md`)

- **Sample size**: propose ~150–200 images stratified across the 4 emotion
  folders and both leakage-clean splits, drawn from the already-leakage-cleaned
  `outputs/full_run/leakage/clean_dataset` so no annotation effort is wasted on
  images that would be quarantined anyway.
- **Labels per image**: face-present (bool) + face bbox; per detected
  connected-component, a shape/symbol class from a fixed vocabulary (star,
  circle, heart, flower, cloud, sun, vehicle, geometric-shape, none/other);
  animal-present (bool) + species from a fixed vocabulary (tiger, wolf, fox,
  squirrel, lion, other-animal, none).
- **Raters**: minimum 2 independent raters per image for a subsample large
  enough to compute Cohen's kappa (the `review.py` agreement machinery already
  built in Phase 0/1 can be reused directly — no new code needed for this
  part).
- **This does not require a clinician** — these are *descriptive* labels
  ("is there a circle here"), not psychological judgments, so any careful
  annotator can do it; clinician time stays reserved for Phase 6's confidence-
  ceiling validation.

---

## 6. Proposed acceptance thresholds

Stated as *proposed starting points*, not derived from any existing DOAR data
(none exists yet) — to be revisited once the annotation sample produces real
inter-rater numbers:

| Detector | Minimum precision | Minimum recall | Notes |
|---|---|---|---|
| Face-presence | 0.80 | 0.70 | Binary presence is the easiest sub-task in the whole matrix |
| Eye-state classification | macro-F1 ≥ 0.60, AND ≥ human inter-rater agreement | — | Must not exceed what humans agree on |
| Shape/symbol classification (per class) | 0.75 | 0.65 | Per-class, not averaged — a class that fails badly should not be hidden by others doing well |
| Vehicle detection (COCO classes) | 0.70 | 0.60 | Lower bar acknowledging untested photo-to-drawing transfer |
| Animal species classification | Not set | Not set | Do not set a bar until human inter-rater agreement is measured (§4) |

A rule only leaves `DETECTOR_UNAVAILABLE` if its detector clears its bar *and*
you separately approve enabling it (per your standing instruction — detector
validation alone is not sufficient).

---

## 7. Expected compute / time (rough, to be refined once scoped)

| Task | Compute | Time estimate |
|---|---|---|
| Annotation (150–200 images, 2 raters, 3 categories) | None (human time) | The dominant cost of this whole phase — likely several hours of careful human labeling, not engineering time |
| Classical-CV circularity pilot (circles) | CPU, seconds | Trivial |
| QuickDraw-classifier pilot over DOAR crops | CPU or small GPU, minutes to train a small CNN | Low |
| Face/landmark pilot (MediaPipe/dlib) | CPU, seconds per image | Low, but first requires the ChildlikeSHAPES provenance check (§2.1) or falls back to the weaker photo-domain option |
| Animal species pilot | Not recommended before annotation + inter-rater step | — |

Nothing above requires a model download yet — all are either classical CV
(zero download), or would use MediaPipe/dlib/a small custom CNN, each a
deliberate, separately-flagged download when you approve moving from planning
to piloting.

---

## 8. Recommended first experiment

**Circles, via classical CV (contour circularity on existing connected
components), validated against a ~30–50 image hand-labeled subsample.** This is
the only candidate in the entire matrix that needs no new dependency, no
download, and no model — it directly tests whether DOAR's existing
segmentation pipeline produces clean enough component crops to make *any*
shape classification approach viable at all, which is a prerequisite finding
for every other shape/symbol rule regardless of which classifier is eventually
used for them. If this fails (e.g., because real children's circles rarely
form one clean connected component), that is itself the most valuable thing to
learn before investing in QuickDraw-classifier or ESRA-dataset integration
work.

---

## 9. Better alternative discovered this session

Two datasets surfaced during research that were not in the original
architecture proposal and are plausibly better starting points than the
QuickDraw-only assumption in `PROPOSED_ARCHITECTURE.md` §4:

- **ESRA** (Roboflow Universe) — directly "objects inside kids' drawings,"
  the closest on-domain match found for the shapes/symbols/vehicles category.
  Not yet verified for licence or annotation quality.
- **ChildlikeSHAPES** (Meta, 2025) — the closest domain match found for
  figure/face structure, but its authenticity as real child-drawn data is
  unconfirmed and must be checked before relying on it.
- **SceneDAPR** (ACM Web Conference 2024) — considered and **not recommended**:
  it is scene-level free-hand drawing data including children, but it is
  fundamentally a Draw-A-Person-in-the-Rain dataset — an *instructed*,
  prompt-specific psychological test. Per the project's own free-drawing
  constraint (`SCIENTIFIC_LIMITATIONS.md` §4), training on instructed-prompt
  data risks learning compositional conventions (a person, rain, sometimes an
  umbrella) that don't generalize to genuinely spontaneous drawings, and using
  it would blur exactly the line the project is built to keep clean. Access
  also requires a non-commercial research agreement via direct email request,
  which is a decision for you, not something to pursue without your sign-off.

Recommendation: **experiment** with ESRA (cheap to check), **defer**
ChildlikeSHAPES pending provenance verification, **reject** SceneDAPR for
this project's purposes despite its relevance to the broader field.
