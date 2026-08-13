# DOAR — Rule + Clinical Reasoning Corpus Freeze Report

## What was frozen this session

The **rule + literature + reasoning-architecture corpus**, as a documented,
verified, internally-consistent snapshot — not a new implementation. No
executable code changed. `resources/psychology_sources/rules_registry.
json` (19-rule production registry) and `src/doar/rules.py`/`concerns.py`
are byte-for-byte unchanged; `CONCERNS_ENABLED` remains `False`.

## Counts (verified this session, not assumed)

- Source rows: **67** (19 Arabic PDF + 48 English PDF), both read directly.
- Unique consolidated rules (draft): **41**.
- Unique active rules (production): **19**.
- Executable in production (confirmed by reading `rules.py`'s
  `_TIER1_DISPATCH`): **6**.
- Executable in the draft registry: **10** — this is what "~10 executable"
  in the prior estimate actually referred to; do not conflate with the
  production figure of 6.
- Blocked, `DETECTOR_UNAVAILABLE`: **27** across the draft registry (13 of
  which are also in production).
- Every one of the 41 rules now has a fully traceable provenance chain:
  `rule_id` → `source_entry_ids` → PDF/page/section → `RULE_EVIDENCE_
  MATRIX.csv` row → (where applicable) `external_literature_register.json`
  entry. Verified by direct cross-reference, not spot-checked.

## Evidence honesty check

38 of 41 rules: `evidence_direction = insufficient`. 2: `partial`. 1:
`conflicting`. **0: `support` or `contradict`.** No rule reaches
`context_transfer_justification = directly_applicable`. This is the
expected, honest result for a corpus built from an unattributed 2-page
handout and a self-described "not diagnostic... speculative" compiled
guide — the audit's job was to confirm this stays visible, not to improve
the numbers.

## What is new and ready for future use (design only, nothing activated)

- A concern-domain taxonomy (`CONCERN_DOMAIN_MAP.json`) with 8 real domains
  covering positive/neutral, anxiety/stress, low-mood, aggression, social
  withdrawal, and developmental/attention themes — explicitly excluding
  ADHD/abuse diagnostic labels per the AAP guideline and Allen & Tussey
  review — plus a 9th domain (`maltreatment_or_safety_concern`)
  intentionally left with **zero** candidate rules.
- A relationship graph (`RULE_RELATIONSHIP_GRAPH.json`) distinguishing
  `REQUIRES` (detector preconditions), `SAME_EVIDENCE_FAMILY` (anti-double-
  counting), `CONTEXT_LIMITED_BY` (confound literature), and one real
  literature-level `CONTRADICTS` finding (colour/emotion) that touches no
  currently active rule.
- A transparent, non-numeric evidence-aggregation policy extending
  `concerns.py`'s existing, tested 4-level strength scale.
- Clinician and parent output schemas with a hard content-separation rule
  (no diagnostic labels, no abuse framing, no technical fields in parent
  output — reusing the existing, already-tested Parent View leak-check
  pattern).
- An LLM synthesis policy extending the existing `LLM_GROUNDING_AND_
  SAFETY_DESIGN.md`/`chat.py` architecture to hypothesis narration, with an
  explicit research-candidate queue instead of any web-search → production
  path.

## Explicit gaps, stated not hidden

- Abuse/maltreatment: no PDF-sourced or literature-sourced positive
  indicator exists to expose — the domain stays empty by design, not by
  omission.
- `LIT_EMOTION_FACE_ENCODING_007` (PubMed 17165414) remains unverified —
  full-text fetch was blocked twice across two sessions; still the most
  promising unverified lead for a future, narrowly-scoped depicted-
  expression rule.
- No compound (Level 4) rule was defined — no literature found this
  session justified one beyond `concerns.py`'s existing generic
  convergence mechanism.
- The Tree-imagery meta-analysis (item 2) was only partially re-verified
  (full text blocked); its participant count is taken from the task's own
  supplied figure.

## RULE + CLINICAL REASONING CORPUS FREEZE: **YES**

Every one of the 41 rules is traceable to its source, honestly graded
against independent literature (0 rules overclaim support), and mapped
into a documented — not activated — reasoning hierarchy with an explicit,
non-numeric aggregation policy and a hard clinician/parent output
separation. Nothing here changes production behavior, so freezing the
corpus carries no runtime risk; the open items above (unverified emotion-
encoding lead, empty abuse domain, no compound rules) are correctly
scoped as future work, not blockers to using this snapshot as the
benchmark's fixed reference.
