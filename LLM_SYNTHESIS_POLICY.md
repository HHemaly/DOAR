# DOAR — LLM Synthesis / Q&A Policy

Answers the task's question directly: **yes, an LLM is the right tool for
natural, tailored phrasing** — but only as a *writer* over data DOAR
already computed, never as a *source* of observations, rules, hypotheses,
or references. This document extends the pre-existing
`LLM_GROUNDING_AND_SAFETY_DESIGN.md` (which already designs the general
`ChatProvider`/`ClaimVerifier`/`SafetyChecker` architecture, implemented
today only via the deterministic, no-API-key `DeterministicChatProvider`
in `src/doar/chat.py`) with the specific evidence-package schema and
research-candidate-queue behavior this audit's clinical-hypothesis layer
needs. **No LLM is wired in by this document. Design/schema only.**

## 1. The structured evidence package (what the LLM is allowed to see)

Exactly the fields the task lists — no more, no less:

```json
{
  "case_id": "...",
  "verified_observations": ["obs_... (CLINICIAN_OUTPUT_SCHEMA.md cards)"],
  "measurements": {"composition": "...", "colour": "..."},
  "eligible_rules": ["rule_id, source_claim, allowed_output_level"],
  "evidence_strength": {"...": "..."},
  "candidate_hypotheses": ["hyp_... (CLINICIAN_OUTPUT_SCHEMA.md cards)"],
  "contradictions": ["RULE_RELATIONSHIP_GRAPH.json literature_level_contradictions, if relevant"],
  "alternative_explanations": ["from the triggered rules' own alternative_explanations field"],
  "missing_information": ["from the hypothesis card's missing_clinical_information field"],
  "references": ["external_literature_register.json entries actually cited"],
  "allowed_claim_level": "LEVEL_0 | LEVEL_1 | LEVEL_2 (never LEVEL_3, per RULE_EVIDENCE_AUDIT.md §9)",
  "audience": "clinician | parent"
}
```

This is the **only** input the LLM step receives about the case. It never
sees the raw image, the raw model logits, or any data not already reduced
to one of the fields above.

## 2. What the LLM may do

- Explain, summarize, and connect the package's own fields in flowing prose.
- Adapt tone: `CLINICIAN_OUTPUT_SCHEMA.md` wording for `audience: clinician`,
  `PARENT_OUTPUT_SCHEMA.md` wording (and its hard content restrictions,
  Section 3 of that document) for `audience: parent`.
- Answer follow-up questions **using only the same evidence package** —
  each turn re-supplies the package, the LLM does not accumulate outside
  knowledge across turns.

## 3. What the LLM must never do (extends `LLM_GROUNDING_AND_SAFETY_DESIGN.md` §2 verbatim)

- Invent an observation, measurement, rule, hypothesis, reference, or
  DOI/PMID not present in the supplied package.
- State a claim at a higher `allowed_output_level` than the package
  specifies (a `LEVEL_1` rule's finding can never be phrased as a `LEVEL_2`
  named hypothesis by the LLM rewriting it more confidently).
- Strengthen a hypothesis's support level beyond what
  `EVIDENCE_AGGREGATION_POLICY.md`'s transparent count already established
  — "the evidence is compelling" is not an LLM's call to make.
- Produce a `LEVEL_3` diagnostic statement under any framing.
- Answer a question about content NOT in the package by guessing.

## 4. Verification pipeline (reuses `LLM_GROUNDING_AND_SAFETY_DESIGN.md` §3–4 exactly)

Draft → `ClaimExtractor.extract()` → `ClaimVerifier.verify()` against the
evidence package (each claim must map to a field in Section 1's schema or
it is `UNSUPPORTED` and dropped/replaced) → `SafetyChecker` (diagnostic-
language regex, extended to also catch a claim exceeding its rule's
`allowed_output_level`) → display. No new component is introduced; the
existing `ClaimVerifier`'s "evidence bundle" is simply this document's
Section 1 package instead of (or in addition to) `qa.py`'s existing
evidence set.

## 5. Grounded Q&A / RAG layer (schema only, per task instruction — not implemented)

Retrieval is scoped to exactly two sources, both already governed:

1. **Case evidence** — Section 1's package, per-case.
2. **Validated DOAR knowledge base** — `RULE_EVIDENCE_MATRIX.csv` +
   `resources/psychology_sources/external_literature_register.json`, i.e.
   only rules/references that have gone through this audit's (or a future
   session's) verification, never raw web content.

If a user's question requests information not present in either source:

> **"Not currently in the validated DOAR knowledge base. Search literature?"**

If the user opts in, any live web/literature search result is:

- Shown for that turn only, clearly labelled `unverified_external_search
  — not yet in the DOAR knowledge base`.
- **Never** written into `external_literature_register.json` or any rule
  registry automatically.
- Queued as a `RESEARCH_CANDIDATE` entry (same status vocabulary already
  used throughout `LITERATURE_CANDIDATE_REGISTER.csv` and this audit's own
  new entries) for a **future, separate, human-reviewed session** to verify
  and formally add — exactly `LLM_GROUNDING_AND_SAFETY_DESIGN.md` §6's
  pre-existing knowledge-base-governance principle
  ("`EvidenceRetriever` only ever reads already-approved records... no
  component in this design writes to the knowledge base"), restated here
  because it is the load-bearing constraint for this specific feature.

**Never**: web search result → automatically becomes a production rule or
citation. There is no code path today that does this, and this policy
requires none be added without the full workflow above.

## 6. What this document does not do

- Does not select or configure a real LLM provider (still deferred, per
  `LLM_GROUNDING_AND_SAFETY_DESIGN.md` §1).
- Does not implement `ClaimExtractor`/`ClaimVerifier`/`ResponseJudge` for
  the clinical-hypothesis case — those remain typed stubs, same as today.
- Does not build the RAG retrieval index, the "search literature?" UI
  affordance, or the research-candidate-queue storage — schema/policy only,
  per `RULE_EVIDENCE_AUDIT.md` §11 (out of scope this phase).
