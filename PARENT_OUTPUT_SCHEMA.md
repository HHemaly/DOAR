# DOAR — Parent Output Schema

Same underlying evidence as `CLINICIAN_OUTPUT_SCHEMA.md`, different tone
and a hard content restriction. **Schema only — no UI built here.**

## 1. Core rule

Parent output **never states an unconfirmed psychiatric/diagnostic label**,
even hedged. The clinician schema may say "Possible depressive/low-mood
presentation" (a LEVEL 2 clinical-hypothesis phrase, still non-diagnostic
but naming a clinical construct); parent wording stays one level more
general, describing the observation and its ordinariness, never the named
construct.

| Clinician wording | Parent wording |
|---|---|
| "Possible depressive/low-mood presentation" | "Several repeated signs related to low mood or emotional difficulty were observed." |
| "Possible anxiety- or stress-related presentation" | "Several repeated signs related to worry, caution, or nervous feelings were observed." |
| "Possible social withdrawal or isolation theme" | "A few signs sometimes linked to feeling alone or left out were observed." |
| "Possible aggression- or threat-related theme" | "A few signs sometimes linked to frustration or tension were observed." |
| "Possible attention/developmental-related pattern" | "A few signs related to focus or task style were observed — these have many ordinary explanations, including age and drawing experience." |
| Maltreatment/safety domain | **Never surfaced to a parent as a DOAR-generated hypothesis at all** — this domain has zero candidate rules (`CONCERN_DOMAIN_MAP.json`) and DOAR must never suggest it to a parent under any circumstances. |

This table is sourced directly from `CONCERN_DOMAIN_MAP.json`'s
`example_parent_wording` field per domain — one source of truth, not a
second copy that could drift.

## 2. Required structure per parent-facing section

```json
{
  "what_was_observed": "A plain description of the drawing feature, no jargon (reuses parent_safe_wording already in rules_registry_v2.json).",
  "simple_explanation": "Why some people think this might mean something, in one hedged sentence.",
  "uncertainty": "This is not a diagnosis, and the same feature has other common, non-clinical explanations. (Verbatim pattern already used by rules_registry_v2.json's parent_safe_wording field.)",
  "gentle_questions_to_ask": ["The rule's own question_template field, e.g. 'Would you be willing to ask your child about the wide eyes in this drawing?'"],
  "what_to_monitor": "Whether this comes up again in future drawings, or in how your child talks about their day.",
  "when_professional_review_may_help": "If this pattern keeps repeating, or if you notice it alongside other changes in mood or behaviour, mentioning it to your child's pediatrician or a counselor is a reasonable next step -- not because anything is confirmed, but because a professional can look at the fuller picture."
}
```

Every field maps to data DOAR already has per rule
(`parent_safe_wording`, `question_template` in `rules_registry_v2.json`)
or is a fixed, reviewed sentence template (`uncertainty`,
`when_professional_review_may_help`) — not generated freely per case.

## 3. What parent output must never include

- A named psychiatric/developmental diagnosis or disorder, confirmed or
  hypothetical ("your child may have depression/ADHD/autism").
- Any numeric score, percentage, or "risk level."
- Technical fields: `rule_id`, `bbox`, `evidence_family`, DOI/PMID,
  `confidence_ceiling`, or any internal status enum — these stay in
  Technical View / the clinician schema only, matching the existing,
  already-tested separation (`DOAR_V1_1_STABILIZATION_REPORT.md`'s
  Parent-View leak check: `assertNotIn` on `checkpoint`/`grounding_dino`/
  `bbox`/`rule_id` in Parent View text — this schema extends that same
  tested boundary to the new hypothesis output, does not relax it).
- Abuse/maltreatment framing of any kind, hedged or not (see Section 1's
  table — this is an absolute exclusion, not a wording choice).

## 4. Relationship to existing Parent View code

`parent_view.py`'s `build_overall_result_summary` already generates hedged,
capability-aware sentences from structured evidence with tested leakage
guarantees. This schema is additive to that existing pattern for the new
hypothesis layer (Level 5 of `RULE_EVIDENCE_AUDIT.md`'s hierarchy) — it
does not replace or redesign the existing Parent View summary logic, and
implementing it is out of this audit's scope (`RULE_EVIDENCE_AUDIT.md` §11:
no UI work this phase).
