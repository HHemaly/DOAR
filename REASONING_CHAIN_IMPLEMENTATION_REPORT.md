# DOAR — Reasoning Chain + Development Benchmark Scaffold: Implementation Report (V1.6)

## What was implemented

**`src/doar/reasoning_chain.py`** — new, entirely additive module implementing
the frozen chain: verified visual observation → eligible atomic rule →
evidence family → concern domain → candidate clinical hypothesis. Reads
`RULE_EVIDENCE_MATRIX.csv` / `RULE_RELATIONSHIP_GRAPH.json` /
`CONCERN_DOMAIN_MAP.json` read-only. **Never imports or calls
`rules.py::evaluate_rules` or `concerns.py::derive_concerns`** (proven by a
dedicated source-level test) — reuses only the pure
`concerns.py::_aggregation_strength` helper. Nothing this module produces
is written into a case's real `analysis.json` `rule_evaluations`/`concerns`.

Two safety properties enforced and tested:

1. **Only `case_verification_status == "verified"` entities can satisfy a
   rule's visual precondition.** `uncertain`/`rejected`/`unreviewed`
   entities never do (`VisualPreconditionTests.test_unverified_entity_
   never_satisfies_a_precondition`, parametrized over all three).
2. **VISUALLY VERIFIED != PSYCHOLOGICALLY VALIDATED at the aggregation
   level, not just per-rule.** Every rule-sourced evidence item is tagged
   source_type `clinician_symbolic` regardless of whether its visual
   precondition was Gemini-verified — confirming an object is PRESENT is
   not independent confirmation of the PSYCHOLOGICAL CLAIM a PDF attached
   to it. Two rules from the same concern domain, both visually verified,
   still cap at `WEAK_HYPOTHESIS` (never higher) unless a genuinely
   independent second source (`model_evidence`, e.g. the emotion model)
   is also supplied — proven directly by
   `CandidateHypothesisSafetyTests.test_two_verified_rules_same_domain_
   cap_at_weak_hypothesis` and
   `test_independent_model_evidence_can_raise_the_ceiling`.

Also implemented: `EligibleAtomicRuleMatch` (copies every field verbatim
from the frozen matrix — invents nothing), evidence-family deduplication
(`deduplicate_by_evidence_family`), concern-domain aggregation
(`aggregate_by_concern_domain`), `CandidateHypothesis` construction, and
`build_clinician_package` / `build_parent_package` /
`build_llm_writer_payload` matching `CLINICIAN_OUTPUT_SCHEMA.md` /
`PARENT_OUTPUT_SCHEMA.md` / `LLM_SYNTHESIS_POLICY.md` exactly.
`neutral_descriptive_only(_no_construct_proposed)` and
`maltreatment_or_safety_concern` domains are explicitly excluded from ever
producing a hypothesis regardless of evidence volume.

## Rules now visually usable vs. blocked (exact counts, verified
programmatically against all 41 rules in the frozen matrix — not
hand-counted)

| Category | Count | Rule IDs |
|---|---|---|
| **Now visually checkable** (positive-presence via Gemini-verified entities) | 16 | Eyes ×3, animals ×4 + animal-choice-general, geometry, stars, circles, transport, hearts, face_expression, house, tree |
| **Now visually checkable** (absence pattern: person verified + part not verified) | 2 | missing_hands, missing_mouth |
| **Blocked — structural** (`requires_process_data`/`requires_longitudinal_data`/`requires_absolute_scale_or_context` = yes; unrelated to entity presence) | 7 | flowers_clouds_sun, repeated_monsters_danger, very_small_drawing_absolute, repeated_family_conflict, repeated_isolation_themes, over_erasing, refusal_to_draw |
| **Blocked — needs an unbuilt feature** (quality/proportion/density judgment, not identity) | 6 | eyes_missing_or_undetailed, zigzag_lines, exaggerated_body_parts, dark_colors_sad_faces_isolation, excessive_detail, neglect_of_background |
| **Not applicable to this module** (Tier-1 composition or stroke-proxy rules — already evaluated by `rules.py` directly against composition/stroke features, not VisualEntity presence; unaffected by this phase) | 10 | coverage_half/full/small, placement_top/left/right/center, heavy/light pressure, shaky/broken lines |

16 + 2 + 7 + 6 + 10 = 41.

**"Now visually checkable" means the PRECONDITION can be satisfied** — it
does not mean the rule's interpretation is validated. All 18 checkable
rules keep their frozen `allowed_output_level` (`disabled` for 14 of them,
`individual_heuristic_only` for the eyes/face_expression/house/tree that
happen to also carry that flag) — no rule was promoted.

**Known limitation, not a bug**: the three eye rules (wide/stern/closed)
share the identical minimal precondition ("an eye is visibly present") —
this module has no eye-QUALITY classifier (open vs. narrowed vs. shut) to
distinguish which specific rule's claim is more apt, so all three
co-satisfy whenever any verified eye entity exists. This is an honest,
stated design limitation (documented in `reasoning_chain.py` and this
report), not silently hidden, and not something to "fix" without building
a real detector — out of this phase's scope.

## Bug found and fixed during the smoke test

`VisualEntity.matches_search_term()`'s liberal both-directions substring
matching (designed for open-ended search recall) produced two real false
positives when reused for precondition-checking: an entity labeled
"eyebrow" wrongly satisfied the `wide_eyes` precondition ("eye" is a
substring of "eyebrow"), and a verifier-supplied alternative label
"cartoon eye" wrongly satisfied `PSY_AR_TRANSPORT_012`'s precondition
("car" is a substring of "cartoon"). **Fixed** by adding a dedicated
whole-word token matcher (`_tokenize`/`_entity_tokens`/`_find_matches`)
specific to this module — never reusing the search-oriented substring
check for precondition purposes again. Confirmed fixed by rerunning the
3-image smoke test (see below) before and after.

## 15-image development set

`DEVELOPMENT_SET_15.json`: the 5 required anchors (h38, p2b_0001..0004)
plus 10 additional images selected by a fully reproducible, non-random
systematic sample (evenly-spaced indices, step = floor(75/10) = 7, into
the 75-image remaining pool of `outputs/phase2c1/private_images`,
excluding p2b_0000 and the 4 anchors) — `p2b_0005/0012/0019/0026/0033/
0040/0047/0054/0061/0068`. All 15 files verified to exist. Permanently
development-only per `DEVELOPMENT_SET_15.json`'s own note.

## Smoke test result (3 images: h38, p2b_0001, p2b_0003 — already-exposed development images, no new API calls)

| Image | Entities loaded | Verified | Preconditions satisfied | Eligible matches | Hypotheses |
|---|---|---|---|---|---|
| h38 | 11 | 10 | 0/41 | 0 | 0 |
| p2b_0001 | 7 | 3 | 2/41 | 2 | 0 (single-evidence domains, correctly `INSUFFICIENT`) |
| p2b_0003 | 11 | 5 | 6/41 | 6 | 1 (`depressive_or_low_mood_related`, `WEAK_HYPOTHESIS`) |

h38 produced zero matches — correct, not a bug: h38's content (sun, cloud,
butterfly, traffic light, car, person, ground) matches none of the 41
rules' vocabulary, and its one borderline case (the car, whose bbox was
never `verified` due to the known localization overshoot documented in
the prior bbox-stabilization phase) correctly never reached the
precondition check at all. Full trace (image → entities → preconditions →
matches → families → domains → hypothesis → clinician/parent/LLM
packages) completed for all 3 images with zero exceptions, both before
and after the token-matching bug fix.

## Tests

30 new tests in `tests/test_reasoning_chain.py`, all passing. Ruff and
compileall clean. Full suite: see final report for totals (run in
background alongside this report).

## Deliverables

`src/doar/reasoning_chain.py`, `tests/test_reasoning_chain.py`,
`DEVELOPMENT_SET_15.json`, `ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md`,
`BENCHMARK_SCHEMA.md`, this report.
