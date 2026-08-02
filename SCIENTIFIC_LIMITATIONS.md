# DOAR — Scientific Limitations

This document states, plainly, what DOAR can and cannot claim about a child's drawing. It exists so that no report, UI screen, Q&A answer, or thesis chapter oversells the system. It is a living document — update it whenever a capability, detector, or validation status changes (see `DECISION_LOG.md` for the change process; `RETIRED_NOT_DELETED` status applies here too — never delete a limitation because a feature improved, note that it was superseded and when).

## 1. The core scientific claim (and only this claim)

> DOAR is an evidence-traceable decision-support system that analyses spontaneous children's drawings and generates limited emotional-expression hypotheses to support non-verbal communication and professional review. It is not a diagnostic instrument.

Every other sentence in this document exists to keep the system inside that claim.

## 2. What DOAR may hypothesize about (with hedged, non-diagnostic wording)

Positive/warm expression; happy/cheerful expression; sad/subdued expression; fear- or tension-related expression; anger- or frustration-related expression; withdrawal or inhibition; possible loneliness or social disconnection; possible need for reassurance or support; high emotional intensity; calm expression; mixed or unclear expression; insufficient evidence.

Every one of these must be presented as a *hypothesis for a supportive conversation*, never as a finding, using wording of the form: *"Several independent visual characteristics are consistent with a possible [X] expression. The evidence is limited and should be explored through a neutral conversation rather than treated as proof of the child's emotional state."*

## 3. What DOAR must never infer or diagnose from a drawing alone

Depression; anxiety disorder; ADHD; autism; PTSD; psychosis; abuse or neglect; self-harm or suicide risk; any psychiatric or developmental diagnosis. Directly, visibly safety-relevant content (e.g. an explicit disclosure written on the page) may be described neutrally and routed for professional review — the drawing itself cannot establish *why* it is present or whether a real event occurred.

DOAR must never produce an "anxiety percentage," "abuse probability," or similar clinical score. The 4-class emotion-model probabilities are model estimates over a fixed label schema (Angry/Fear/Happy/Sad) learned from imperfect dataset folder labels — they are not, and must never be presented as, probabilities of the child's true internal emotional state. `emotion.py`'s output already carries `"Model probabilities are not psychological or diagnostic confidence"` as a limitation string on every prediction — this must be preserved in every future refactor.

**Why this boundary is non-negotiable, not just cautious:** a 2022 empirical study using deep neural networks found the House-Tree-Person test — a far more established, standardized instrument than anything in DOAR's current rule registry — "is not valid for the prediction of mental health" ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0001691822002499)). A systematic review and meta-analysis of HTP indicators found low cross-study consistency and concluded the HTP indicator system is not reliable ([Guo et al., PMC9848786](https://pmc.ncbi.nlm.nih.gov/articles/PMC9848786/)). A systematic review of projective drawings specifically for identifying child sexual abuse is the reference the user supplied for exactly this reason ([PMID 22467642](https://pubmed.ncbi.nlm.nih.gov/22467642/)) — projective drawing tests have been found unable to reliably indicate abuse, and DOAR must never suggest otherwise. ADHD diagnosis requires a structured, multi-informant, multi-setting clinical process per the American Academy of Pediatrics guideline ([PMC7067282](https://pmc.ncbi.nlm.nih.gov/articles/PMC7067282/)) — nothing a single drawing can approximate.

If a validated instrument with decades of research behind it does not reliably predict mental health from drawings, an unattributed two-page Arabic pop-psychology article (DOAR's current rules source, see `CURRENT_STATE_AUDIT.md` §5.1) certainly cannot. This is why every rule derived from it carries a confidence ceiling of 0.05–0.25 and `scientific_support: not_found_for_specific_claim` — and why 68% of that registry (13/19 rules) currently cannot even activate, for lack of a detector (`RULE_COVERAGE_MATRIX.csv`).

## 4. Free-drawing constraint — why omission-based inference is invalid here

DOAR analyses **spontaneous free drawings**, not a House-Tree-Person, Draw-a-Person, Kinetic-Family-Drawing, or any other instructed-prompt test. This matters scientifically, not just procedurally: omission-based interpretation ("the child didn't draw a window, which means...") is only meaningful when the child was *asked* to draw a house (or similar) and chose not to include a standard element. In a free drawing, a missing house, tree, person, or family member carries no information at all — the child was never asked to draw one.

Consequently:
- Absence of an object must never be scored as `NOT_MATCHED` or treated as negative evidence. It must be `NOT_ASSESSABLE_CONTEXT_UNKNOWN` (the drawing prompt is unknown) or, once prompt metadata is optionally supplied, `OBSERVED_ABSENCE` (only meaningful if a prompt is actually known).
- Two of the three cited size-related rules currently in the registry (`coverage_about_half` at minimum) are themselves sourced from studies that used **instructed** tasks (children were told what to draw and their drawing was compared against that instruction) — see `RULE_SOURCE_REGISTER.csv` rows `PSY_AR_SIZE_HALF_014`/`015`/`016`. Their applicability to genuinely spontaneous free drawing is itself an open question the current registry does not flag. This is a gap, not yet a documented limitation, and is added here for the first time by this audit.
- Any HTP/DAP/KFD-style omission rule, standardized scoring, or age-normative developmental interpretation belongs in the "Tier 3: prompt- or age-dependent research rules" catalogue only, and must return `NOT_ASSESSABLE_CONTEXT_UNKNOWN` / `NOT_ASSESSABLE_AGE_UNKNOWN` rather than silently applying to every image.

## 4a. Phase 2 update (2026-08-02) — rule tiers and concern aggregation

Every rule in `rules_registry.json` now carries an explicit `tier` (`tier_1_prompt_independent` / `tier_2_content_conditional` / `tier_3_prompt_or_age_dependent`) and `activation_status` field. **No rule's user-facing behavior changed** — the same 6 coverage/placement rules produce `weak_support` at the same confidence ceilings, the same 13 eyes/animals/shapes/symbols rules produce `missing_detector`, exactly as before. The tier field only makes the *reason* explicit and machine-checkable.

The concern-aggregation engine (`concerns.py`) now implements four strength levels (`INSUFFICIENT` / `WEAK_HYPOTHESIS` / `POSSIBLE_FOR_EXPLORATION` / `PROFESSIONAL_REVIEW_SUGGESTED`) instead of a binary converge-or-nothing check, and a real defect was fixed: `evaluate_rules()` previously discarded the emotion model's independent prediction (`del colour, evidence`), so no genuinely independent second source could ever reach the convergence check. It no longer does. **This remains fully disabled in production** (`CONCERNS_ENABLED = False`) — the fix makes the logic correct and testable, not user-facing. See `DECISION_LOG.md` 2026-08-02 for a correction to this session's own earlier (inaccurate) description of what the defect was.

Two things this fix does *not* do, stated explicitly so neither is assumed: it does not detect contradiction between the emotion model and rule hypotheses (the registry's rules have no defined mapping onto the Angry/Fear/Happy/Sad label space, so no such comparison is implemented — inventing one would itself be an overclaim); and `PROFESSIONAL_REVIEW_SUGGESTED` (the strongest level) is only reachable with synthetic 3-source test data today, since only two source types (clinician rules, emotion model) exist anywhere in the real pipeline until Phase 3 detectors ship.

## 5. What is empirically validated in DOAR itself, right now — and what is not

**Empirically demonstrated this session (real numbers, real run, see `CURRENT_STATE_AUDIT.md` §2):** the ML pipeline can distinguish the 4 dataset emotion labels above chance on a locked test split (macro-F1 0.637, accuracy 70.4%, n=98). This is evidence the *dataset labels* (not necessarily "true" child emotion) are learnable from image content to a moderate degree, comparable to similarly-scaled published work on children's-drawing emotion classification (see the results report generated earlier this session for citations).

**Not empirically validated in DOAR, at any point:**
- That the dataset's folder labels reflect a child's actual felt emotion at the time of drawing (they are, at best, a third party's classification of the drawing — provenance of the original dataset's labeling process is unknown and out of DOAR's control).
- That any of the 6 currently-active coverage/placement rules produce clinically meaningful output — their confidence ceilings (0.10–0.20) and `not_found_for_...`/`weak_not_specific` scientific-support labels already say this explicitly; this document just makes it load-bearing.
- That the fusion model's calibration (temperature scaling, valid-only) generalizes beyond this specific dataset and split.
- That any concern-profile output is meaningful — none has ever been generated (`CURRENT_STATE_AUDIT.md` §5.3).

## 6. Reliability of the underlying dataset labels

41% of the raw local dataset (1,517 of 3,688 images) was removed this session for cross-split duplication or contradictory labeling (the same image filed under two different emotions) — see `CURRENT_STATE_AUDIT.md` §2 and §4. This is disclosed here because it directly bounds what can be claimed: a dataset with this much internal inconsistency cannot be assumed to carry high-quality ground truth for the remaining images either. Per Section 12 of the working spec, remaining labels should be treated as "dataset labels of uncertain quality," not verified psychological truth, until a label-quality audit (multi-model disagreement, out-of-fold ranking, blind professional review) is built and run — this does not exist yet (see `IMPLEMENTATION_PLAN.md`).

## 6a. Capture-medium and material confounds (added 2026-08-02, literature review round 2)

DOAR's objective features — `coverage_*` (page-coverage/bounding-box) and
`colour_*` (colour proportions, dominant colour) most directly — are computed
from a final flattened image with no record of how it was produced. Literature
found this session (`LIT_MATERIAL_MEDIUM_CONFOUND_010`) reports that paper
size (small paper measurably limits completion/detail; oversized paper leaves
excess blank space that would read as low "coverage") and drawing medium
(marker vs. touchscreen-finger vs. stylus; watercolor pens vs. pencils
measurably differ in colour brightness) materially affect exactly these
features, independent of anything psychological. DOAR's `Combined_Drawing`
dataset almost certainly aggregates images captured under inconsistent
conditions (scans, phone photos, different paper sizes) with no metadata
recording which — meaning a `coverage_small` or `colour.dominant_colour`
reading could reflect the physical page or the drawing tool as easily as
anything about the child. This is a first-class limitation on the *objective
features themselves*, not only on the psychological interpretations attached
to them, and it existed before this literature review — it was simply not
previously documented. No fix is proposed here beyond disclosure; a future
fix would mean capturing (or estimating) paper size/medium at intake time.

## 7. External AI (if/when added)

No external LLM is currently wired into DOAR (`CURRENT_STATE_AUDIT.md` §7). If one is added later (per the spec's Section 13, for translation/explanation/Q&A phrasing only), it must never originate a measurement, count, mask, probability, rule activation, or citation — only rephrase what deterministic code already computed, subject to a deterministic post-hoc validator that rejects any claim not traceable to a stored evidence ID. This is a design requirement for future work, not a current capability to describe.

## 8. Standing rule for every future contributor

Before any rule, detector, or claim is enabled for real users:
1. Its required feature must be measurable by an implemented, tested detector.
2. The detector must have a documented, honest performance/reliability statement (not just "it runs without crashing").
3. It must apply to spontaneous free drawings, or be strictly content-conditional on a positively detected object.
4. Its evidence limitations must be recorded in this document and in `RULE_SOURCE_REGISTER.csv`.
5. Its output must be capped below any diagnostic-sounding confidence.
6. Its activation and deactivation must be tested.
7. Its use must be explicitly approved (see `DECISION_LOG.md`).

"It is cited in a real paper" is necessary, never sufficient.
