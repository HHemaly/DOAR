# DOAR-TRACE Master Specification

**Status: living design document for the DOAR-TRACE architecture.** This
document is the policy layer; `PHASE1_IMPLEMENTATION_REPORT.md` and
`PHASE1_5_IMPLEMENTATION_REPORT.md` record what has actually been built
against it so far. It extends, and does not contradict,
`TARGET_APPLICATION_ARCHITECTURE.md` (the prior session's design for the
same end state) — read that document for the module dependency diagram;
this document adds the rule-ontology taxonomy, the evidence-v2 contract,
and the safety wording policy that diagram assumed but never fully
specified.

**Phase 1.5 update**: Sections 2 and the allowed-output table below were
written before the compiled English PDF existed in this repository and
described a 19-rule, single-construct-per-rule design. That design has
since been fully superseded — see `docs/AGGREGATION_POLICY.md` for the
current Level A/B/C policy (41 rules, 12 shared constructs,
`individual_heuristic_only` replacing the old single-rule
`question_generating`/`combined_hypothesis_only` conflation) and
`docs/CONSTRUCT_MAPPING_RATIONALE.md` for how rules now map to
constructs. Section 2 below is left as the original taxonomy definition
(still accurate for the 8 observability classes) with corrections marked
inline where Phase 1.5 changed something it originally assumed.

## 1. Final-product policy

The final product must eventually show, for every case: all extracted
features, all detected objects, all spatial relationships, all rule
statuses, the possible meaning attached to each matched rule, combined
drawing-level hypotheses from independently-converging rules, references,
evidence strength, limitations, contradictions, uncertainty and
unavailable information — plus an LLM layer for explanation and chat, and
two clearly separate artifacts: a drawing-only analysis and a
context-enhanced analysis (once child-profile context exists downstream).

### Allowed vs. forbidden wording

**The system MAY produce a cautious, non-diagnostic, multi-feature
hypothesis**, e.g.:

> "Several features in this drawing form a possible low-mood or
> emotional-distress pattern. This does not mean that the child is
> depressed. It may be worth discussing the drawing gently, watching
> whether similar themes repeat, and seeking professional advice if there
> are persistent concerns about the child's mood or behaviour."

> "This drawing contains several visual features associated with
> prominence in the depicted scene. Whether this reflects the child's
> feelings or personality cannot be determined from the drawing alone."

**The system MUST NOT** produce a diagnostic or definitive claim, in
English or Arabic, including but not limited to: "The child is
depressed," "The child has anxiety," "The child is superior," "The
drawing proves abuse," or any equivalent Arabic statement (تشخيص، يعاني
من، مصاب، اضطراب، اكتئاب، قلق مرضي، صدمة نفسية، إساءة).

**This is enforced mechanically, not just by prompt/policy text**:
`judges.py::DIAGNOSTIC_PATTERNS` / `_ARABIC_DIAGNOSTIC` already scan
every rule-derived narrative string for exactly these patterns
(`judges.py::run_judges`'s `safety_judge`) and this increment's claim
verifier (4F) reuses the identical regex set for any new generated text,
so the same forbidden-wording list cannot silently drift between the two
enforcement points.

The parent-facing drawing-level-hypothesis goal is **not removed** by
this caution — it is reached only through evidence grades, aggregation
thresholds (reusing `concerns.py`'s existing 2-evidence/2-source
convergence policy, §4D), explicit limitations, references, and the
wording constraints above.

## 2. Rule ontology taxonomy (registry-v2)

Every candidate rule (from the current registry, and from every source
PDF row) is classified into exactly one **observability class**:

| Class | Meaning |
|---|---|
| `static_direct` | Computed directly from the image with an existing, real extractor (e.g. bounding-box coverage) |
| `static_detector` | Computable from the single static image, but requires a detector that does not exist yet (e.g. animal-species classifier) |
| `static_proxy` | The observable is a *stand-in* for a different physical quantity that cannot actually be measured (e.g. line darkness/thickness as a proxy for pencil pressure — never claim the proxy IS the physical quantity) |
| `prompt_required` | Requires knowing the drawing task/prompt given to the child |
| `age_required` | Requires the child's age/developmental stage to interpret |
| `process_required` | Requires observing the drawing *process* (e.g. "while distracted"), not just the finished image |
| `longitudinal_required` | Requires multiple drawings over time |
| `not_operational` | None of the above apply cleanly; not currently a usable observable at all |

Every rule also receives exactly one **allowed-output level** (Phase 1.5
adds `individual_heuristic_only`, the correct level for a single
executable rule shown on its own — see `docs/AGGREGATION_POLICY.md`
Level B):

| Level | Meaning |
|---|---|
| `observation_only` | State the raw measurement, no interpretation |
| `question_generating` | May be phrased as a gentle, cautious question/observation for a parent |
| `individual_heuristic_only` | A single rule's own cautious reading, shown alone (Level B) — **never** displayed as a combined theme |
| `combined_hypothesis_only` | Only usable as part of a multi-rule, multi-family converged theme (Level C, `docs/AGGREGATION_POLICY.md`) — **never** emitted from a single rule alone |
| `professional_only` | Shown only in the technical/clinician view, never the parent view |
| `disabled` | Not shown anywhere (no detector, or precondition unobservable) |

Weak or speculative rules (lion, fox, squirrel, stars, circles, etc.) are
**never discarded** for being weak — they remain in the ontology with
their speculative evidence level explicitly visible, `allowed_output_level
= disabled` only because their required detector doesn't exist yet, not
because the rule itself was judged unworthy.

**Wording discipline**: a stroke/line-darkness feature is never described
as physical pencil pressure — always "dark/thick-line appearance proxy."
`EN_COMPILED_LINE_HEAVY_PRESSURE_030`/`EN_COMPILED_LINE_LIGHT_PRESSURE_031`
are exactly this case, `observable` named
`heavy_line_pressure_appearance`/`light_line_pressure_appearance` (never
just "pressure"). **Phase 2A update**: these 2 rules, plus
`EN_COMPILED_LINE_SHAKY_BROKEN_032` and `EN_COMPILED_PLACEMENT_CENTER_029`,
are now `allowed_output_level: individual_heuristic_only` — wired to a
real, additive, parallel evaluator (`rule_engine_v2.py`), never modifying
`rules.py`/`rules_registry.json`. See `docs/STATIC_PROXY_RULE_POLICY.md`
for the full per-rule activation audit and
`docs/PAGE_FRAME_ASSESSABILITY.md`/`docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md`
for the measurement validation and threshold provenance behind that
decision.

## 3. Evidence-v2 contract (summary — full schema in `trace_evidence.py`)

Every `EvidenceRecordV2` carries: `evidence_id, evidence_type, producer,
producer_version, value, unit, confidence, method, status
(PASS/WARN/FAIL/ABSTAIN), limitations, provenance, source_evidence_ids`.
`status` is a **judge-style verdict on the evidence itself** (was this
measurement trustworthy), distinct from and complementary to
`evidence_schema.py`'s existing `EvidenceItem.status`
(available/experimental/unavailable/failed/insufficient_evidence/
requires_manual_review), which answers "was this ever measured at all."
Both schemas exist in the codebase; see `CURRENT_TO_TARGET_GAP_V2.md` for
why they were not merged in this increment.

## 3.5. Construct system and Level A/B/C policy (Phase 1.5, new)

Rules no longer map 1:1 to a bespoke construct name (the Phase 1 design,
which is exactly why a single rule could render as a "theme"). A small,
fixed set of 12 drawing-level constructs
(`resources/psychology_sources/construct_registry.json`) is defined
once; multiple rules map to a shared construct only when genuinely
defensible (`docs/CONSTRUCT_MAPPING_RATIONALE.md`). Output is split into
three explicit levels — A (observation), B (individual rule suggestion,
mandatory for every triggered rule), C (combined hypothesis, only when a
construct's own policy is satisfied by >=2 independent contributors) —
specified in full in `docs/AGGREGATION_POLICY.md`, including the ordinal
0-4 scale and the `global_expressive_content_model` evidence family.

## 3.6. Page-frame assessability and measurement validation (Phase 2A, new)

A page-relative rule (page-coverage or placement) is only meaningful
when the whole physical page is visible in the uploaded image.
`page_frame.py` adds an explicit, classical-CV-only (no trained
detector) assessment with 5 statuses; only `full_page_detected`/
`likely_full_page` permit normal evaluation. Otherwise, the rule's
status becomes `not_assessable` — a status distinct from both
`not_matched` (evaluated, found false) and `missing_detector` (no
evaluator exists at all) — never silently reinterpreted as either. A
120-image real audit found only 12.5% of the current dataset
assessable; see `docs/PAGE_FRAME_ASSESSABILITY.md`.

Every executable rule's underlying feature is validated two ways before
activation: synthetic ground-truth checks (does the pipeline recover an
analytically known value?) and transformation-invariance checks (does
the feature survive resize/rotation/compression/lighting/crop noise
within a stated tolerance?) — `docs/FEATURE_MEASUREMENT_VALIDATION.md`
and `docs/FEATURE_ROBUSTNESS_RESULTS.md`. Both validate measurement
implementation only, never psychological validity — that distinction is
stated in both documents' own text, not left implicit.

Every executable rule's threshold carries one of 5 documented
provenance categories (`directly_sourced` through
`empirically_exploratory`) and a real sensitivity sweep on non-test
data — `docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md`. Real trigger
counts for all 41 registry-v2 rules (10 executable, 31 not) are
recorded in `artifacts/phase2a/rule_trigger_distribution.csv`, never
interpreted as psychological prevalence.

## 4. Governing invariants (unchanged from `TARGET_APPLICATION_ARCHITECTURE.md`)

1. No module writes to the original dataset or any existing
   `outputs/phase3a`–`outputs/phase7b`/`outputs/thesis_run` artifact.
2. No module accesses `split == "test"` without both `test_guard.py` flags.
3. `CONCERNS_ENABLED` stays `False`; this increment does not flip it.
4. `rules_registry.json` (the production registry) is never written by
   any automated path — `rules_registry_v2.json` (4C) is a **draft**,
   sitting beside it, never auto-promoted.
5. Every user-facing claim carries an explicit status from `{measured,
   model_output, executed_rule, user_provided, approved_knowledge_base,
   not_available}` — never an unlabeled assertion.
