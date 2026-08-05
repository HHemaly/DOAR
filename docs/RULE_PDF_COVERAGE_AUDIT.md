# Rule–PDF Coverage Audit (DOAR-TRACE 4C)

**Status: real, verified coverage audit.** Every claim below was checked
against the actual PDF text and the actual `rules_registry_v2.json` file
generated this session (`src/doar/registry_v2_build.py`), not assumed.

## Source-file discrepancy (report first, per the task's own instruction)

The task that requested this audit names `child_drawing_rules_compiled.pdf`
as the ontology source. **That file does not exist anywhere in this
repository** — a full-tree search (`find . -iname "*.pdf"`, excluding
`.venv`) returns exactly one PDF:

```
./التحليل النفسي للصور.pdf
```

This is the same PDF read and fully extracted (via `pypdf`, since
`pdftoppm`/poppler is not installed here) in an earlier session this
engagement, and independently re-verified for this audit. It is a short,
informal, 2-page Arabic document ("Psychological Analysis of Drawings")
— no page numbers printed in the source itself, no numbered sections, no
academic citations of its own, and `resources/psychology_sources/
rules_registry.json`'s own metadata already self-declares
`scientifically_validated: false`.

**This audit uses that PDF as the sole ontology source.** If a different,
more extensive `child_drawing_rules_compiled.pdf` exists outside this
repository (e.g. supplied separately and not yet copied in), it was not
available to this session and none of its content could be included here
— this is a real, disclosed blocker, not a silent substitution.

## Coverage result

The PDF contains **19 distinct bullet-point rules**, spread across 2
pages under 8 informal section headings (رسم العيون / رسم الحيوانات /
الأشكال الهندسية [+ 3 subsections: النجوم، الزهور والسحب والشمس، الدوائر]
/ رسم وسائل النقل / رسم القلوب / حجم الرسم / موقع الرسم). Every one of
these 19 bullets was already present, 1:1, in the production
`rules_registry.json` before this session began (confirmed by direct
page-by-page reading against the registry's `arabic`/`english` fields in
an earlier task this engagement, re-spot-checked here).

**Result: 19 of 19 PDF rows covered. 0 rows missing from the registry. 0
registry rules without a traceable PDF source.** There is no 20th row to
add — `rules_registry_v2.json`'s 19 entries are exactly the PDF's 19
bullets, each now carrying the full registry-v2 field set (observability
class, allowed-output level, evidence family, target construct,
dependency group, alternative explanations, etc. — see
`DOAR_TRACE_MASTER_SPEC.md` Section 2 for the taxonomy). Every entry's
`registry_v2_status` is `already_in_production_registry`, never the
task's default `candidate_unreviewed` label, because none of them are
actually new candidates.

## Source-page traceability (all 19)

| Rule ID | Page | Section |
|---|---|---|
| PSY_AR_EYES_WIDE_001 | 1 | رسم العيون |
| PSY_AR_EYES_STERN_002 | 1 | رسم العيون |
| PSY_AR_EYES_CLOSED_003 | 1 | رسم العيون |
| PSY_AR_ANIMAL_TIGER_WOLF_004 | 1 | رسم الحيوانات |
| PSY_AR_ANIMAL_FOX_005 | 1 | رسم الحيوانات |
| PSY_AR_ANIMAL_SQUIRREL_006 | 1 | رسم الحيوانات |
| PSY_AR_ANIMAL_LION_007 | 1 | رسم الحيوانات |
| PSY_AR_GEOMETRY_008 | 1 | الأشكال الهندسية |
| PSY_AR_STARS_009 | 1 | الأشكال الهندسية — النجوم |
| PSY_AR_FLOWERS_CLOUDS_SUN_010 | 1 | الأشكال الهندسية — الزهور والسحب والشمس |
| PSY_AR_CIRCLES_011 | 1 | الأشكال الهندسية — الدوائر |
| PSY_AR_TRANSPORT_012 | 1 | رسم وسائل النقل |
| PSY_AR_HEARTS_013 | 2 | رسم القلوب |
| PSY_AR_SIZE_HALF_014 | 2 | حجم الرسم |
| PSY_AR_SIZE_FULL_015 | 2 | حجم الرسم |
| PSY_AR_SIZE_SMALL_016 | 2 | حجم الرسم |
| PSY_AR_PLACE_TOP_017 | 2 | موقع الرسم |
| PSY_AR_PLACE_LEFT_018 | 2 | موقع الرسم |
| PSY_AR_PLACE_RIGHT_019 | 2 | موقع الرسم |

## Observability classification summary

| Class | Count | Rules |
|---|---|---|
| `static_direct` | 6 | the 3 size + 3 placement rules — real, computed measurements |
| `static_detector` | 12 | 3 eyes + 4 animals + geometry + stars + circles + transport + hearts — need a detector that does not exist |
| `process_required` | 1 | flowers/clouds/sun — precondition ("while distracted") is not observable from a static finished image at all, even with a perfect detector |
| `prompt_required` / `age_required` / `longitudinal_required` / `not_operational` | 0 each | no rule in this source needs these |

## Allowed-output level summary

| Level | Count | Why |
|---|---|---|
| `question_generating` | 6 | `static_direct` rules — a real (if unvalidated-threshold) measurement exists |
| `disabled` | 13 | `static_detector` (12, no detector) + `process_required` (1, unobservable precondition) |
| `observation_only` / `combined_hypothesis_only` / `professional_only` | 0 each | no single rule in this registry qualifies for these — see below |

`combined_hypothesis_only` is **never** assigned to a single rule by
design (`DOAR_TRACE_MASTER_SPEC.md` Section 2) — it is reserved for the
dependency-aware aggregator (`structured_report.py`, Section 4D) when
≥2 independently-sourced rules converge on a related theme.

## Duplicates / near-duplicates

None. Every one of the 19 rules maps to a distinct PDF bullet and a
distinct `observable` value — there is no pair of rules in this source
that restates the same observation. (Rules `004`–`007`, the 4
animal-species rules, share an identical *structural pattern* — one
species implies one trait — but each names a different species and is a
distinct rule, not a duplicate.)

## Weak/speculative rules — retained, not discarded

Per the task's explicit instruction, `lion`, `fox`, `squirrel`, `stars`,
and `circles` all remain in `rules_registry_v2.json` with
`evidence_level: "speculative_symbolic"` and `allowed_output_level:
"disabled"` (blocked only by the missing detector, never removed from
the ontology).

## Wording discipline check

No rule in this source PDF concerns line darkness, thickness, or
pressure — the "dark/thick-line appearance proxy" wording requirement
(`DOAR_TRACE_MASTER_SPEC.md` Section 2) does not currently apply to any
of the 19 rules. It is recorded as a standing rule for any future
proxy-type feature (e.g. if a `stroke.intensity_proxy`-based rule is ever
added — `features.py` already names that exact feature `intensity_proxy`,
not "pencil pressure," so the existing extractor already follows this
discipline).
