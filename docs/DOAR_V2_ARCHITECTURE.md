# DOAR V2 architecture: from V1 to the ontology foundation

## 0. Terminology (read this first — the repository already overloads "V2")

This repository's own history uses "v2"/"Phase 1" for a *prior* piece of
work that is **not** the same thing this document calls "V2":

| Term used here | What it actually refers to | Where |
|---|---|---|
| **V1** | The original pipeline: `features.py` (59 objective measurements) + `rules.py` (19-rule dispatcher) + `resources/psychology_sources/rules_registry.json` + `models.py`/`deep/*` (Happy/Sad/Angry/Fear classifier) + `concerns.py` (disabled). Baseline commit for comparison purposes: `247562d` (this branch's `HEAD` at the start of this milestone — see `DOAR_V2_MIGRATION.md` §0 for the exact git state recorded before any V2 work began). | Pre-existing |
| **the evidence pipeline** (repo-internal "Phase 1"/`RuleV2`/"evidence schema v2") | `evidence_schema.py`, `evidence_adapter.py`, `rule_schema.py`, `evidence_rule_engine.py`, `evidence_pipeline.py` — a real, tested, already-working image-only evidence-to-rule path added in a prior session, wired to CLI as `main.py evidence-pipeline`. It re-expresses the same 19 rules with explicit source citation (page/section of the one Arabic PDF), an honest 7-outcome vocabulary (`triggered`/`not_triggered`/`insufficient_evidence`/`not_applicable`/`extractor_unavailable`/`requires_manual_review`/`conflicting_evidence`), and a hard invariant that a missing extractor can never resolve to `triggered`/`not_triggered`. | Pre-existing, **unmodified by this milestone** |
| **V2 ontology foundation** (this document, this milestone) | The new `src/doar/ontology/` package: a declarative feature ontology, a source-agnostic `FeatureObservation`, a multi-source append-only `FeatureStore`, a `CaseContext`, a composable declarative `Rule` schema with explicit `EMPIRICAL_EXPRESSIVE`/`CLASSICAL_PROJECTIVE` tagging, and a three-valued (TRUE/FALSE/UNKNOWN) rule engine with full evaluation traces. | **New in this milestone** |

The evidence pipeline already solved image-only traceability for the
*specific* 19-rule registry. What it does not provide — and what the
task's V2 milestone actually asks for — is: a feature description
language independent of any one extractor; a store that can hold more
than one source's opinion per feature and never erase a correction;
context (age/task/narrative) as a first-class, genuinely optional input
that gates rule applicability generically; and rules whose premises are
composable data (`AllOf`/`AnyOf`/`NotOf` over feature conditions) rather
than one Python branch per rule. The ontology foundation adds exactly
that layer, on top of — not instead of — the evidence pipeline.

## 1. Why a new layer instead of extending the evidence pipeline in place

Extending `evidence_rule_engine.py` directly was considered and
rejected for this milestone:

- Its `RuleResult`/outcome vocabulary and `_evaluate_condition` dispatch
  are hand-written per observable (`_COVERAGE_OBSERVABLES`,
  `_PLACEMENT_SPLIT`) specifically for this one 19-rule registry. Adding
  compound premises, a context object, and a multi-source store to it
  would have meant rewriting most of the module — exactly the
  "replace experimentally validated code" the task says not to do
  without a documented reason.
- Its evidence model (`EvidenceItem`/`EvidenceSet`) is scoped to one
  pipeline run producing one flat list for one image. The task requires
  a store that holds *multiple sources'* claims about the *same*
  feature over time, including human corrections that must never erase
  prior evidence — a materially different data model, not an additive
  field.

Instead, `src/doar/ontology/legacy_adapters.py` is the seam: it
converts real `EvidenceItem`s (produced by the unmodified, tested
`evidence_adapter.build_evidence_set`) into `FeatureObservation`s, and
converts the unmodified, tested `rule_schema.load_rules_v2()` registry
into declarative `ontology.rule.Rule`s. Both existing modules are
reused, not rewritten; `tests/test_evidence_rule_engine_v2.py` (the
pre-existing 396-line suite) continues to pass unchanged.

## 2. Target architecture (this milestone's slice)

```
DRAWING
   |
   v
analyze_image (existing, unmodified)  -- quality / segmentation / composition / colour
   |
   v
build_evidence_set (existing, unmodified)  -- EvidenceItem/EvidenceSet, canonical categories
   |
   v
evidence_item_to_observation (new, ontology/legacy_adapters.py)
   |
   v
FeatureObservation  (new, ontology/observation.py)
   |
   v
FeatureStore  (new, ontology/store.py)  -- append-only, multi-source,
   |                                        CORROBORATED/SINGLE_SOURCE/AI_ONLY/
   |                                        CONFLICT/NOT_ASSESSABLE/HUMAN_CONFIRMED/
   |                                        HUMAN_CORRECTED
   v
Rule (declarative, ontology/rule.py)  +  CaseContext (ontology/context.py)
   |                                        age / sex / drawing_task / narrative
   |                                        -- all Optional, never inferred
   v
evaluate_rule (three-valued engine, ontology/engine.py)
   |
   v
RuleEvaluation + full trace
   MATCHED / MATCHED_PROVISIONAL / NOT_MATCHED / NOT_ASSESSABLE /
   NOT_APPLICABLE_TO_TASK / NOT_APPLICABLE_TO_AGE / MISSING_PREREQUISITE /
   CONFLICTING_EVIDENCE
```

This is deliberately the *first* slice, matching the task's Section 5
scope limit: no new detector, no VLM/LLM integration, no report
generator, no scoring/aggregation across rules. `ontology/engine.py`
evaluates one rule at a time and returns its evidence; nothing here
combines multiple rules into a "concern" or a psychological conclusion
(that remains `concerns.py`'s job, and it stays disabled —
`CONCERNS_ENABLED = False` — untouched by this milestone).

## 3. The one architectural rule, enforced mechanically, in three places

*"Psychological interpretation must never be performed directly inside
feature extraction."* Enforced by:

1. `features.py`/`evidence_adapter.py`: verified by audit
   (`docs/DOAR_V2_FEATURE_AUDIT.md` §2) — every one of the 59+6
   implemented features is a plain numeric/categorical measurement, zero
   interpretation mixed in.
2. `ontology/observation.py`/`ontology/store.py`: a `FeatureObservation`
   has no field for a psychological claim — only `value`, `assessable`,
   confidence, and provenance. There is structurally nowhere in this
   schema to put "the child is sad."
3. `ontology/rule.py`: interpretation only ever appears in
   `Rule.professional_reasoning`/`parent_safe_wording`, attached to a
   rule (with a citation, a framework tag, and a confidence ceiling),
   never attached to a feature. `Rule.__post_init__` additionally
   refuses to let a rule claim `EMPIRICAL_EXPRESSIVE` without a real
   (non-`TODO`) citation — a second, independent guard against
   fabricated scientific support, on top of the first.

## 4. Omission safety, worked end to end

The task's own example — `hands_detected=false` alone must not become
`hands_missing=true` — is implemented generically, not as a special
case:

- `config/feature_ontology.yaml` declares `hands_present` with
  `requirements: [human_figure_present, hands_region_visible]`.
- `ontology/engine.py::_prerequisites_satisfied` checks each declared
  requirement's own *effective* value in the `FeatureStore` before ever
  looking at the feature's own reported value.
- If any requirement is not `TRUE`-and-assessable, the condition
  resolves to `TriState.UNKNOWN` — **regardless of what the feature's
  own observation says** — and the rule resolves to
  `MISSING_PREREQUISITE`, never `MATCHED`/`NOT_MATCHED`.
- Proven by `tests/test_ontology_rule_engine.py::OmissionSafetyTests`,
  which constructs exactly the task's scenario (figure present, hands
  region visibility never established, detector reports
  `hands_present=False`) and asserts the outcome is
  `MISSING_PREREQUISITE`, not a confirmed omission.

## 5. Context is one object, not two pipelines

`ontology.context.CaseContext` has one shape whether every field is
`None` or fully populated. `ontology/engine.py::evaluate_rule` takes it
as a plain parameter; there is no "image-only mode" vs. "with-context
mode" branch anywhere in the engine. Setting `context.age = 9` and
re-calling `evaluate_rule`/`evaluate_all_rules` with the same
`FeatureStore` re-evaluates every age-gated rule correctly with no other
code change — proven by
`tests/test_ontology_rule_engine.py::ContextPrerequisiteTests::test_re_running_with_context_later_supplied_changes_outcome_no_code_change`.
No rule in the current 19-rule registry actually uses age/task/narrative
gating yet (`docs/DOAR_V2_RULE_AUDIT.md` §1 — all 19 are `IMAGE_ONLY`),
so this is proven with a synthetic rule in tests; it is real,
executable machinery waiting for the first context-dependent rule
(e.g. an HTP/KFD/DAP-specific rule), not a placeholder.

## 6. What is explicitly deferred (per the task's own Section 14 scope limit)

No new detector, no VLM/LLM integration, confidence calibration beyond
simple value-equality/status precedence, evidence-family aggregation
across rules, final psychological scoring, a report-writing LLM, a Q&A
agent, a dashboard, or literature expansion (Freud/Machover/Hammer,
graphology, age estimation). See `docs/DOAR_V2_MIGRATION.md` §"NEXT" for
what is recommended once this foundation lands.
