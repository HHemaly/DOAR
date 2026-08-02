# Experiment Prioritization — Detector Work vs. Model Evaluation vs. Dataset/Label Validation

Answers the question posed directly: given everything found in the audit
(Phase 0-2) and both literature-review rounds, where should the next real
engineering effort go? Compares four concrete candidate experiments,
spanning three different phases of `IMPLEMENTATION_PLAN.md` (Phase 3
detectors, Phase 5 model sweep, Phase 6 label audit), on equal footing. This
document does not assume detector work is the answer — that assumption is
explicitly tested below and rejected.

Nothing here is implemented. This is a planning/ranking artifact.

---

## The four candidates, compared on the same four axes

| | Scientific value | Transfer validity | Annotation burden | Technical feasibility |
|---|---|---|---|---|
| **A. Objective scribble/fragmentation pilot** (test whether DOAR's classical features can capture the developmental-stage construct from `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`) | Moderate — non-diagnostic by construction, so zero interpretation-validity ceiling if it works; genuinely useful covariate | **Unknown — this IS the open question**: source study used CNN features on a stimulus-completion task; DOAR would use classical features on free drawings. Real risk of non-transfer. | Low-moderate — some DOAR-specific validation data needed, but a rough correlational check is cheaper than full annotation | High — reuses existing `features.py`, zero new dependency |
| **B. Circle/shape detection pilot** (classical-CV contour circularity, Phase 3's original recommendation) | **Low** — even perfect success only supports an unvalidated psychological interpretation (`PSY_AR_CIRCLES_011`, ceiling 0.10); ceiling is capped by design, not by execution quality | Directly testable, well-scoped question (do real components form clean contours) | Low — 30-50 image feasibility check per the corrected Phase 3 process | High — zero dependency, classical CV only |
| **C. Complete the four-class emotion-model baseline** (run the full documented multi-seed, multi-architecture sweep already approved in `IMPLEMENTATION_PLAN.md` Phase 5, instead of this session's time-constrained 2-model/1-seed reduction) | **High** — this is DOAR's one already-demonstrated, actually-working, real-data contribution (locked-test macro-F1 0.637 this session). More engineering here directly improves a number that's already part of a legitimate thesis result, unlike A/B which serve an interpretive layer that stays capped regardless of effort | N/A — not a transfer question; this optimizes DOAR's own pipeline on DOAR's own data | None beyond compute time — dataset and code already exist and are verified working | High — `compare-deep-models` etc. are built, tested, and were run successfully this session; only needs full-scope compute time |
| **D. Label-quality / duplicate audit** (out-of-fold disagreement, multi-seed disagreement, optional Confident-Learning-style ranking — Phase 6, not yet started) | **Highest** — addresses a problem we *already know is real and large*: 41% of the raw dataset was removed this session for cross-split duplication or contradictory labels (`CURRENT_STATE_AUDIT.md` §2). That was leakage detection, not label-*correctness* detection — an image can be unique and still mislabeled, and nothing currently measures that. This directly bounds how much of Candidate C's 70.4%/0.637 numbers reflects real signal vs. label noise. | **N/A — lowest risk of all four**: this is entirely about DOAR's own data, with no external population/protocol to transfer from at all | **Can be near-zero**: out-of-fold and multi-seed disagreement need no new human annotation, only re-running already-built multi-seed training with cross-validation splits. A blind professional-review queue (optional, higher-value extension) does need human time. | High for the algorithmic part (existing training infra); moderate for a review-queue UI |

---

## Ranking

1. **D — Label-quality/duplicate audit.** Highest scientific value, lowest
   risk (no external-transfer question at all — it's entirely internal to
   data we already have), and the algorithmic core can be built essentially
   for free by re-running existing training code with cross-validation folds.
   This is the one candidate that turns a problem we've already *proven
   exists* (via the leakage work) into a measured, bounded quantity instead
   of an open question. It also directly de-risks Candidate C: running the
   full model sweep on data of unknown label quality risks reporting
   optimistic numbers that don't hold up to scrutiny.
2. **C — Complete the emotion-model baseline.** High value, zero technical
   risk, extends the one part of DOAR that is already demonstrably working
   on real data. Best sequenced *after* (or concurrently with, if time-boxed)
   D, so the model comparison is evaluated on a dataset with a known,
   documented label-quality profile rather than an unexamined one.
3. **A — Objective scribble/fragmentation pilot.** Genuinely interesting,
   cheap, zero-dependency, and honestly scoped now that its overreach has
   been corrected — but its payoff (one possible covariate feature) is
   smaller in scope than C or D, and its central premise (classical-feature
   transfer from a CNN-measured, stimulus-completion-task finding) is the
   least certain of the four.
4. **B — Circle/shape detection.** Lowest thesis value of the four, despite
   being the cheapest and technically easiest. Its ceiling is fixed by the
   interpretation-validity problem documented throughout this review
   (`RULE_SOURCE_REGISTER.csv`: every one of the 13 Tier-2 rules stays
   scientifically unvalidated even with a working detector) — success here
   only ever produces infrastructure validation ("do our image crops make
   sense"), not a result that strengthens the thesis's actual scientific
   contribution.

## Direct answer to "detector development, model evaluation, or dataset/label
## validation — which would the thesis gain more from?"

**Dataset/label validation, then model evaluation. Detector development
last**, and likely last by a wide margin — not because detectors are
uninteresting, but because every detector in `DETECTOR_DEPENDENCY_MATRIX.csv`
serves rules that stay scientifically unvalidated regardless of detector
success (confirmed independently, per-rule, in `RULE_SOURCE_REGISTER.csv`),
while dataset/label validation and model evaluation both directly strengthen
a result DOAR has already produced and can defend. This reverses the implicit
sequencing `IMPLEMENTATION_PLAN.md` had been drifting toward (Phase 3 next).
Recommend re-ordering Phases 3/5/6 there once this ranking is approved — not
done yet, since that's itself a plan change worth your sign-off, not a
unilateral edit.

## What this does NOT mean

- It does not mean Phase 3 planning work was wasted — the dependency matrix,
  candidate comparison, and annotation schema remain correct and reusable
  whenever detector work is eventually justified.
- It does not mean the literature review found detectors unnecessary forever
  — `LIT_EMOTION_FACE_ENCODING_007` (round 2's most promising lead) is
  detector-dependent and worth pursuing *if* round-3 full-text verification
  confirms it holds up. This ranking is about sequencing, not permanent
  exclusion.
