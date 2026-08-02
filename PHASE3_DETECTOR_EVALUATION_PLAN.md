# Phase 3 — Detector Evaluation Plan (planning + validation prep only)

Status: **planning document, nothing implemented**. No detector has been adopted,
downloaded, or enabled. No rule's `activation_status` has changed from Phase 2.
Companion to `DETECTOR_DEPENDENCY_MATRIX.csv` (per-rule detail),
`LITERATURE_CANDIDATE_REGISTER.csv` and `IMPROVEMENT_REGISTER.md` (literature
review round 1), and `DECISION_LOG.md` (approvals).

> **Round 2 corrections (2026-08-02)**: sections 3, 4, 6, 8, and 9 below were
> revised after (a) a literature review round found the face/eye rule family's
> supporting evidence to be weaker than this plan's original priority ranking
> assumed, and (b) methodological corrections to the validation design itself
> (annotator blinding, sample-size justification, and a statistic-comparison
> error). Original text is preserved below each correction, marked
> superseded, not deleted, per the project's append-only documentation policy.

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

> **CORRECTED (round 2)**: the original table below prioritized by feasibility
> and rule-count only, as instructed at the time. A subsequent literature
> review (`LITERATURE_CANDIDATE_REGISTER.csv`, `IMPROVEMENT_REGISTER.md`)
> found that the broader human-figure/facial-indicator literature — not just
> the one citation already in the eyes rules — is consistently weak,
> instructed-protocol-only, and in at least one well-controlled study
> internally inconsistent across statistical methods for the same feature
> (`LIT_HFD_DEPRESSION_ADULT_004`: eye-shading significant by logistic
> regression, p=.034, but *not* significant by ROC/AUC, p=.285). This doesn't
> change feasibility, but it lowers the *scientific value* of ever finishing
> the face/eye detector, since a successful detector there would still only
> support an unusually weakly-evidenced interpretation even by this
> literature area's own low bar. Per your instruction to prioritize scientific
> value over rule count, face/eyes is demoted below shapes/symbols.
>
> **Further correction (2026-08-02, second pass)**: the "developmental-stage
> feature ranks 1st" claim below itself overreached — see
> `LITERATURE_CANDIDATE_REGISTER.csv`'s corrected `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`
> row. The p<.001 finding is robust *in its source study's population,
> protocol, and CNN-based measurement* — not in DOAR's classical
> handcrafted features on genuinely free drawings, which is untested. This
> table row is retracted; the corrected ranking (developmental-stage feature
> included as one candidate among several, not a presumed winner) is in the
> final chat report and will be folded back into this document next revision.

| Category / candidate | Rules enabled if successful | Feasibility | Scientific value if successful | Priority |
|---|---|---|---|---|
| Objective developmental-stage feature (`LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`) | 0 rules — not tied to any Tier-2 rule; a new Tier-1 objective feature / possible emotion-model covariate | **Unknown, not "high"** — reuses existing DOAR *code paths* but whether they measure the right *construct* is an untested hypothesis, not a given | **RETRACTED ranking**: the source evidence (p<.001) is real but applies to a different population/protocol/measurement approach than DOAR's; transfer is unvalidated | **Not ranked 1st — see final report for the corrected comparison** |
| Shapes/symbols/objects | 6 | High — cheap compute, relevant public data, reuses existing DOAR code, one rule (circles) needs no model at all | Low-moderate — a working detector only ever yields a geometric observation; the attached interpretation stays unsupported (§1) regardless | **2nd** |
| Person/face/eyes | 3 | Medium — mature face-detection tech exists, but eye-*state* classification has no precedent and the one promising drawing-domain dataset (ChildlikeSHAPES) has unconfirmed provenance | **Lowered this round** — even the best available instructed-protocol literature for this rule family is weak and internally inconsistent (see correction note above) | 3rd |
| Animals | 4 | Lowest — zero matching pretrained vocabulary, no drawing-domain data found | Lowest — no literature at all, in any protocol, supports any of the 4 specific species interpretations | 4th, possibly permanently deferred |

Original (round 1) ordering, preserved for the record: shapes/symbols 1st,
face/eyes 2nd, animals 3rd — driven by feasibility and rule-count only, before
the literature review existed to weigh scientific value against it.

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

> **CORRECTED (round 2) — annotator blinding and staged validation**:
> - **Annotators must be blind to the dataset's folder label (the emotion
>   class) and to any model/detector output** when producing ground truth.
>   Nothing in round 1 explicitly required this; it must be stated as a hard
>   requirement, not an implied one — an annotator who can see "this image is
>   from the `Angry` folder" or a detector's own guess could anchor on it,
>   contaminating the very ground truth used to judge that detector.
> - **A 30–50 image subsample is a feasibility check only** (does the
>   pipeline run end-to-end, are crops sane, is the schema usable) — **not**
>   a validation result and never to be reported as one. Round 1's §6
>   thresholds table wrongly implied a small subsample could be scored
>   against those numbers; corrected in §6 below.

---

## 5. Annotation plan (schema in `docs/PHASE3_ANNOTATION_SCHEMA.md`)

> **CORRECTED (round 2)**: round 1 proposed a single ~150–200 image sample
> size up front. That was premature — sample size for a validation study
> should follow from the question being asked (expected prevalence of each
> class, the precision needed on the resulting confidence interval, and the
> cost of a false positive vs. a false negative), none of which are known yet.
> The staged plan below replaces the single-number proposal.

- **Stage A — feasibility only (~30–50 images)**: confirms the pipeline runs
  end-to-end (crops are sane, the schema is usable, annotators can actually
  apply it consistently) and produces a *rough* prevalence estimate per class
  (e.g., "roughly how often does a component even look like a recognizable
  shape at all"). **Not used to accept or reject any detector.**
- **Stage B — sample-size decision**: using Stage A's rough prevalence
  estimates, compute the sample size actually needed for the confidence
  interval width the acceptance decision requires (e.g., a rare class near
  5% prevalence needs a much larger sample than a common class near 40% to
  bound its precision/recall CI to a useful width) — standard proportion
  confidence-interval sample-size calculation, done once real prevalence
  numbers exist, not before.
- **Stage C — full annotation** at the size Stage B determines, executed with
  the blinding requirement from §4.
- **Labels per image**: unchanged from round 1 — face-present (bool) + face
  bbox; per detected connected-component, a shape/symbol class from a fixed
  vocabulary (star, circle, heart, flower, cloud, sun, vehicle,
  geometric-shape, none/other); animal-present (bool) + species from a fixed
  vocabulary (tiger, wolf, fox, squirrel, lion, other-animal, none).
- **Raters**: minimum 2 independent, blinded raters per image, large enough to
  compute Cohen's kappa with a reportable confidence interval (`review.py`'s
  existing agreement machinery is reused, not rebuilt).
- **This does not require a clinician** — these are *descriptive* labels, not
  psychological judgments; clinician time stays reserved for Phase 6.

---

## 6. Proposed acceptance thresholds

> **CORRECTED (round 2)**: round 1's table below stated specific numeric
> precision/recall bars as "proposed starting points." Two problems with that,
> per your review: (1) it compared a detector's macro-F1 directly against a
> human inter-rater kappa as if the two were on the same scale — **they are
> not**. Kappa is chance-corrected agreement between two raters; macro-F1 is
> an unweighted per-class harmonic mean of precision and recall against a
> single reference labeling. Neither is a valid stand-in for the other, and
> "detector F1 must exceed human kappa" is not a coherent requirement. (2)
> Thresholds should be set only after prevalence, inter-rater agreement,
> confidence-interval width, and the real-world cost of a false positive vs.
> false negative are known for each specific class — none of which exist yet.
> The numeric table is retracted; below is the corrected process.

**Corrected acceptance process, per detector/class:**
1. From the annotated sample (§5), compute class prevalence and the human
   inter-rater kappa **with its confidence interval**.
2. Compute the detector's precision, recall, and per-class confusion,
   **each with its own confidence interval** (not a point estimate alone —
   a small sample makes a single number misleading).
3. State, explicitly and separately for each class, the cost of a false
   positive (e.g., a shape/animal/expression wrongly reported to a parent)
   vs. a false negative (a real signal missed) — these costs are not
   symmetric and not the same across classes (a false "closed eyes" is a
   different cost than a false "fox").
4. Only then set a numeric acceptance bar for that specific class, informed
   by 1–3 — not a single blanket 0.75/0.65-style number applied uniformly
   across unrelated classes, which round 1 incorrectly proposed.
5. A rule only leaves `DETECTOR_UNAVAILABLE` if its detector clears the bar
   set this way *and* you separately approve enabling it — detector
   validation alone remains insufficient, unchanged from round 1.

Original (round 1) numeric table, preserved for the record as an example of
the retracted approach: face-presence P≥0.80/R≥0.70; eye-state macro-F1≥0.60
"and ≥ human inter-rater agreement" (the invalid comparison); shape/symbol
per-class P≥0.75/R≥0.65; vehicles P≥0.70/R≥0.60; animals — none set.

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

> **SUPERSEDED — see the final chat report for the corrected comparison.**
> This section's round-2 revision overreached by presenting the
> developmental-stage feature as though its evidence (p<.001) directly
> supported DOAR feasibility — it does not; that evidence applies to the
> source study's own population, protocol, and CNN-based measurement, and
> transfer to DOAR's classical features on free drawings is an untested
> hypothesis, exactly like every Tier-2 detector candidate. The comparison
> across circles, the developmental-stage feature, emotion-model improvement,
> and label-quality auditing is redone on equal footing in the final report
> and will be folded back into this document next revision. Text below is
> preserved as the record of what was previously (incorrectly) recommended.

**If the goal is Phase-3 Tier-2 detector groundwork**: circles, via classical
CV (contour circularity on existing connected components), feasibility-checked
against a ~30–50 image subsample per the corrected §5 process (not treated as
a validation result). This is the only candidate in the entire matrix that
needs no new dependency, no download, and no model — it directly tests
whether DOAR's existing segmentation pipeline produces clean enough component
crops to make *any* shape classification approach viable at all. If this
fails (e.g., real children's circles rarely form one clean connected
component), that is itself the most valuable thing to learn before investing
in QuickDraw-classifier or ESRA-dataset integration work.

**If the goal is the highest scientific-value-per-effort experiment overall**:
the scribble-vs-representational objective feature (§3's new 1st-priority row)
— equally cheap, equally zero-dependency, but grounded in a robust,
unconflicted p<.001 finding rather than an eventually-still-unvalidated
psychological rule, and it produces something usable (a descriptive feature,
a possible model covariate) even in the best case, not just a prerequisite
check for later work.

---

## 9. Better alternative discovered this session

> **UPDATED (round 2) — primary-source verification actually attempted**, per
> your instruction not to recommend these on search-summary claims alone.

- **ESRA** (Roboflow Universe) — **verification attempted and blocked**: the
  Roboflow Universe page returned HTTP 403 to a direct fetch this session.
  Its licence, real annotation quality, and exact category list remain
  **genuinely unverified**, more so than round 1's "unverified" note implied —
  this was an active attempt, not an omission. Do not rely on it without a
  human visiting the page directly (or an authenticated tool) to confirm
  terms before any download.
- **ChildlikeSHAPES** (Meta, arXiv:2504.08022) — **verification attempted and
  inconclusive**: the arXiv abstract page was fetched directly; it states
  "16,000 childlike drawings with pixel-level annotations across 25 semantic
  categories" but **does not disclose** whether these are scanned real
  children's drawings, artist-created "childlike style" art, or synthetic
  images, nor any licence/availability terms. This requires the full paper or
  project page, not yet fetched. Authenticity remains unconfirmed.
- **SceneDAPR** (ACM Web Conference 2024) — **fully verified via its GitHub
  page this session**: licence is **CC BY-NC 4.0** (non-commercial only,
  access requires emailing the authors with project details, ~7 day
  response); exactly **148 total object categories**, of which 6 are the
  core DAPR categories (Person, Rain, Umbrella, Cloud, Puddle, Lightning);
  **1,399 scene sketches total** — 462 from children & adolescents (ages
  8–18, 33%), 650 from adults (46%), 287 from seniors (20%); protocol is
  explicitly the instructed Draw-A-Person-in-the-Rain task, confirmed not
  free drawing; ships with pretrained YOLOv8 detection weights and
  SVG/COCO/YOLO format-conversion utilities. This confirms round 1's rejection
  on stronger grounds than before: the "children" group is specifically
  8–18-year-olds (adolescent-inclusive, not matched to DOAR's likely younger
  population either), and the task is unambiguously instructed, not free.
  **Reject stands, now on verified rather than search-summarized facts.**

Recommendation, revised: do **not** experiment with ESRA or ChildlikeSHAPES
until their licence/provenance is actually confirmed by a human visiting the
source (both blocked automated verification this session) — round 1's
"experiment"/"defer" recommendations are downgraded to **defer both** pending
that manual step. **Reject SceneDAPR stands**, now fully verified.
