# DOAR — Benchmark Schema and Metric Definitions (Development Stage)

Defines the comparison schema and metrics for the "Direct Gemini vs.
DOAR" benchmark. Implemented by `src/doar/benchmark_metrics.py` (metric
code) and `scripts/run_development_benchmark.py` (the runner) — both
validated on real, already-saved Observer/Verifier development data with
zero new API calls, but not yet run against real human annotations
(none exist yet; see `ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md` and
`scripts/annotate_development_set.py`). Applies first to the 15-image
development set (`DEVELOPMENT_SET_15.json`), never yet to the 100-image
held-out set.

## 1. Comparison design

Two conditions evaluated per image, both against the SAME human ground
truth (`ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md`'s Pass-1/Pass-2 records):

- **Condition A — Direct Gemini**: the raw `GeminiVisualObserver` output
  alone (open-world candidates, no verifier, no DOAR reasoning chain).
- **Condition B — DOAR full pipeline**: Observer → Verifier → (this
  phase's) reasoning chain, i.e. only `case_verification_status ==
  "verified"` entities count as DOAR's asserted findings for metric
  purposes.

## 2. Per-image result record schema

```json
{
  "image_id": "h38",
  "condition": "direct_gemini | doar_full_pipeline",
  "raw_candidates": ["... VisualObserverCandidate dicts ..."],
  "verified_entities": ["... VisualEntity dicts (condition B only) ..."],
  "eligible_rule_matches": ["... EligibleAtomicRuleMatch (condition B only) ..."],
  "candidate_hypotheses": ["... CandidateHypothesis (condition B only) ..."],
  "runtime_seconds": 0.0,
  "human_ground_truth_pass1": ["... annotator items ..."],
  "human_ground_truth_pass2": ["... annotator items ..."]
}
```

## 3. Visual metrics (per image, then averaged across the development set)

| Metric | Definition | Ground truth used |
|---|---|---|
| Precision | matched candidates / total candidates reported | Pass 2 (exhaustive) |
| Recall | matched candidates / total human-listed items | Pass 2 (exhaustive) |
| F1 | harmonic mean of precision/recall | Pass 2 |
| Salient recall | matched candidates / Pass-1 items | Pass 1 (salient-only) |
| Hallucination rate | candidates matching NO human-listed item (Pass 2, either annotator) / total candidates reported | Pass 2 |
| Abstention rate | candidates with `entity_type` in `{unknown, scribble, abstract_mark}` OR verifier `case_verification_status in {uncertain, unreviewed}` / total candidates | N/A (self-reported honesty measure) |
| Verifier correction rate | candidates that were `verified` by the observer's own confidence but `rejected`/`uncertain` by the independent verifier / total observer candidates with a bbox | N/A (condition B only — measures how often the verifier actually changes the outcome) |

**Matching rule** (candidate vs. human item): the SAME deterministic
token-overlap matcher already built for observer/verifier label
comparison (`visual_observer._labels_plausibly_match`) — reused, not
reinvented, so "matched" means the same thing here as it already does
throughout the Observer/Verifier pipeline.

**Implemented** in `src/doar/benchmark_metrics.py`
(`visual_precision_recall_f1`/`salient_recall`/`hallucination_rate`/
`abstention_rate`/`verifier_correction_rate`). No number here reflects a
real result yet — computation requires the human annotations
(`ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md`), which do not exist yet; the
runner reports every metric as `null` until they do.

## 4. Reasoning metrics (condition B only — direct Gemini has no reasoning layer)

| Metric | Definition |
|---|---|
| Evidence-backed claims | Count of `EligibleAtomicRuleMatch` objects whose `matched_entity_ids` is non-empty (i.e. every claim traces to a real, verified entity — by `reasoning_chain.py`'s own construction, this is always 100%; this metric exists to catch a FUTURE regression, not to measure today's code) |
| Unsupported claims | Count of any hypothesis-package field NOT traceable to a `rule_id` in `RULE_EVIDENCE_MATRIX.csv` or a `reference_id` in `external_literature_register.json` — expected to be 0 by construction; a non-zero count is a bug report, not a benchmark finding |
| Rule/reference traceability | % of `CandidateHypothesis.supporting_rule_ids` that resolve to a real row in `RULE_EVIDENCE_MATRIX.csv` (expected 100%) |
| Hypothesis derivability | Count of images where >=1 `CandidateHypothesis` was produced, broken down by `support_level` (`WEAK_HYPOTHESIS` / `POSSIBLE_FOR_EXPLORATION` / `PROFESSIONAL_REVIEW_SUGGESTED`) — reports HOW OFTEN the reasoning chain reaches each strength level on real drawings, never a claim about correctness |

**Implemented** in `src/doar/benchmark_metrics.py`
(`evidence_backed_claim_rate`/`unsupported_claim_rate`/
`rule_reference_traceability`/`hypothesis_derivability`), confirmed
1.0/0.0/1.0-as-expected on the 5 already-exposed anchor images'
real saved data during this phase's dry run (2 rule references, 9
eligible matches, zero unsupported).

These "reasoning metrics" are largely **construction-time guarantees**
(traceability is enforced by `reasoning_chain.py`'s own design — see
`RULE_FREEZE_REPORT.md`), not something a benchmark run discovers after
the fact. Their inclusion here is for the future 100-image run, where
running the pipeline at scale is the only way to confirm the guarantee
holds under real, varied input, not just under this phase's smoke test.

## 5. Inter-annotator visual agreement

Two independent annotators (`ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md`)
each complete Pass 1 and Pass 2 for all 15 images, entirely
independently, with no consensus step. **Their two item lists are never
merged into one combined ground truth** — every visual metric in Section
3 is computed against EACH annotator's list separately (see
`src/doar/benchmark_metrics.py`'s module docstring). This section defines
a SEPARATE metric measuring how well the two annotators agree with each
other, not a way of producing a single truth set.

**Method — normalized concept-set pairwise F1/Jaccard + adjudication
rate** (implemented in `benchmark_metrics.inter_annotator_agreement`):

1. Normalize both annotators' item labels the same way every other label
   comparison in DOAR already works — `visual_observer._labels_plausibly_
   match` (whole-word token overlap, not raw substring matching).
2. Greedily match each of annotator A's items to at most one still-
   unclaimed item of annotator B's (so two "sun" items in B never both
   match the one "sun" item in A) — `matched_pairs`.
3. `jaccard = matched_pairs / (len(A) + len(B) - matched_pairs)`
4. `f1 = 2 * matched_pairs / (len(A) + len(B))`
5. `adjudication_rate = unmatched_items / (len(A) + len(B))` — the
   fraction of ALL recorded items (either annotator) with no counterpart.
   This is a flag for which items a human adjudicator would need to look
   at before either annotator's list could be trusted alone — this
   function does not resolve the disagreement itself.

Computed once per image, separately for Pass 1 and Pass 2 (Pass 2 is
expected to show higher agreement, having no time pressure).

## 6. What this schema explicitly does NOT define

- Any pass/fail threshold — the 15-image development set is for building
  and debugging the comparison itself, not for judging DOAR "good enough."
- Any combination of visual + reasoning metrics into a single score.
- Statistical significance testing — reserved for the 100-image held-out
  run's own, separate protocol (not written yet, out of this phase's scope).

## 7. Explicit non-goal

This schema, and any future script that implements it, must never write
back into `RULE_EVIDENCE_MATRIX.csv`, `CONCERN_DOMAIN_MAP.json`, or any
other frozen file based on development-set results — per the parent
task's instruction ("Do NOT tune scientific policy from these images").
Development-set results may surface **implementation bugs** to fix; they
must never silently become a scientific conclusion.
