# Parent-View Information Policy (DOAR-TRACE Phase 2A.2)

## Why this exists

Phase 2A.2's audit (Section 2) found the Parent view had grown to 12
mixed sections, several of them showing internal identifiers (`rule_id`,
`observable`, `evidence_id`, `reference_ids`) and full raw tables (all
41 rules, all 60 features) that a parent has no way to interpret and
that were never meant for a lay audience. This document is the policy
those 12 sections are replaced against; `doar_prototype_app.py` is the
implementation.

## The 5 required sections

Parent View contains **exactly 5** main result sections, in this order:

1. **Overall result** -- the first thing a parent reads. Generated
   entirely by `parent_view.build_overall_result_summary()` from real
   per-case data (never hard-coded): whether a combined pattern exists,
   the expressive-model result when available, how many individual
   suggestions exist, that objects/relationships were not analyzed, and
   whether the complete page was assessable.
2. **What was observed** -- `plain_language_observations()`, now
   page-reference-aware (see below).
3. **Possible meaning** -- combined patterns and individual
   observations, each in its own expander, titled by a friendly
   observation number + family name (never a raw `rule_id`).
4. **Questions and next steps** -- natural, open-ended questions (never
   the mechanical "ask about the X" pattern -- see Section 8's fix in
   `registry_v2_build.py`), with a safe generic fallback
   ("Can you tell me what is happening in this picture?") when no
   rule-specific question exists for this case.
5. **Limitations** -- the capability-status summary sentence, the
   page-not-assessable warning when relevant, and the permanent
   disclaimer.

A follow-up chat panel and a single "More details about this result"
expander (pointing to Technical view) sit below the 5 sections but are
not themselves counted as result sections.

## What is hidden from Parent View by default (Section 4)

Never shown in the Parent tab: evidence IDs, internal rule IDs, feature
IDs, dependency groups, schema versions, producer names,
`detector_absent:*` strings, the full 41-rule table, the full 60-feature
table, threshold enums, raw judge records, verification implementation
details. Every one of these remains fully visible in Technical View --
**nothing is deleted, only reorganized out of the Parent tab**.

The one Parent-view expander that surfaces rule-level detail ("Observation
N: <friendly family>") may contain only: the matched rule's own
plain-language explanation (`parent_safe_wording`), a friendly source
name (`friendly_source_name()` -- never the raw PDF filename), the
evidence level as written in the source, alternative explanations, and
limitations.

## Page-reference-aware wording (Section 5)

`plain_language_observations()` and `build_overall_result_summary()`
both take the real, resolved `page_reference` for the case. When
`page_reference.page_relative_features_assessable` is `False`, neither
function will ever produce "covers X% of the page", "centred on the
page", or "placed at the top/left/right of the page" -- instead:

> "The complete sheet could not be confirmed, so page use and placement
> were not interpreted."

The raw, image-frame-relative measurements (`composition.bounding_box_coverage`,
centroid, margins) remain visible in Technical View's objective-features
table, explicitly labelled:

> "Measured relative to the uploaded image frame; not interpreted as
> physical-page usage."

## Minimal page-reference user control (Section 6)

A single, clearly optional sidebar control: "Does this image show the
complete sheet of paper?" with 4 choices (`PARENT_PAGE_DECLARATION_LABELS`
in `page_reference.py`) mapped to the real API via
`user_page_declaration_from_choice()`:

| Choice | API declaration | Resolved mode |
|---|---|---|
| Let the system decide | `None` | `auto_detected_page` / automatic outcome |
| Yes, the complete sheet is visible | `{"mode": "user_confirmed_full_frame"}` | `user_confirmed_full_frame` |
| No, this image is cropped | `{"mode": "user_declared_cropped"}` | `cropped_or_content_only` |
| I am not sure | `{"mode": "user_declared_uncertain"}` | `uncertain` |

No corner editor is implemented in this phase (`user_defined_page_corners`
exists in the API and is fully tested, but has no UI entry point yet --
this is stated explicitly in Technical View, not hidden). The chosen
declaration is **persisted and traceable purely from the saved
`page_reference` fields** (`obtained_via` + `page_reference_mode`) --
`page_reference.describe_declaration_choice()` reconstructs which of
the 4 choices was made without any separate declaration file, and
Technical View's "Parent declaration used" metric shows it for every
case.

## Capability status (Section 7)

`parent_view.capability_status()` returns the real WORKING/LIMITED/NOT
AVAILABLE inventory (exact lists in the module). Parent View shows only
`capability_status_summary_text()` -- one or two plain sentences.
Technical View's "Missing capabilities" section shows the complete,
itemized 3-tier list.

## Natural question wording (Section 8)

Every one of the 10 executable rules now has a hand-written
`question_template_override` in `registry_v2_build.py` -- natural,
open-ended, grouped by theme (page-coverage rules ask what shaped how
much of the page was used; placement rules ask what shaped where on the
page; line-intensity rules ask what tool was used; the fragmentation
rule asks the child to narrate the scene). The old mechanically-generated
"Would you be willing to ask your child about the `<raw_observable>`"
pattern remains only for the 31 disabled rules, which never reach the
Parent view.

## Technical View (Section 9)

Nothing removed from Phase 2A/2A.1's Technical View -- reorganized under
11 named subsections (Input and page reference / Image quality and
segmentation / Objective features / Canonical features / Expressive-
content model / Rule evaluations / Combined-pattern calculation /
Evidence and provenance / Judges and verification / Missing capabilities
/ Sources and registry), with new content this phase adds (page
reference, canonical features, capability status, declaration
traceability) folded into the section it belongs to.

## What "psychologically validated" still never means

No wording change in this phase claims scientific/psychological
validation of any interpretation -- this phase changes *where*
information is shown and *how plainly* it is worded, never the
underlying evidence grade, confidence ceiling, or validation status of
any rule.
