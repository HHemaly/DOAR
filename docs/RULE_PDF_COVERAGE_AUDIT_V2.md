# Rule-PDF Coverage Audit v2 (DOAR-TRACE Phase 1.5)

**Status: real, verified.** Supersedes `docs/RULE_PDF_COVERAGE_AUDIT.md`
(Phase 1, which covered only `التحليل النفسي للصور.pdf`'s 19 rows since
`child_drawing_rules_compiled.pdf` did not exist in the repository at
that time). Both PDFs are now present and both are fully catalogued.

## Source files (both verified present and read page by page)

| File | Path | Pages | Read via |
|---|---|---|---|
| Arabic psychologist notes | `التحليل النفسي للصور.pdf` (repo root) | 2 | `pypdf` extraction |
| Compiled English guide | `resources/psychology_sources/child_drawing_rules_compiled.pdf` | 3 | `pypdf` extraction |

Both files are checked for existence by
`tests/test_source_catalog_and_registry_v2.py::SourcePDFPresenceTests`,
which fails loudly (not silently) if either is ever missing or
substituted.

## Source row counts (exact)

| Source | Total rows | Candidate-rule rows | General-guidance rows |
|---|---|---|---|
| Arabic notes | 19 | 19 | 0 |
| Compiled English guide | 48 | 43 | 5 (4 "how to use the guide" principles + 1 closing "Bottom line" caution) |
| **Total** | **67** | **62** | **5** |

Full row-by-row detail, including original text, page, section, and the
source's own evidence-level/notes columns where present, is in
`resources/psychology_sources/source_rule_catalog.json`
(`source_catalog_build.py`).

## Duplicates and near-duplicates (all 31 explicit, none silent)

- **19 cross-document pairs** where an Arabic rule and an English rule
  describe the same observable (e.g. `wide_eyes`) -- mostly
  `near_duplicate` (the two sources differ in wording/certainty), one
  `related_but_distinct` (fox: "malicious intent" vs. "sneakiness/cunning").
- **3 internal-to-the-English-PDF duplicates**: the PDF itself restates
  "repeated monsters/danger" across two sections (page 2 and page 3),
  and "eyes with little detail or missing eyes" across two sections
  (page 1 and page 3); "very small drawing" and "uses only a small part
  of the page" (page 2) describe closely related but distinct observables.
- **9 `expands` relationships** marking genuinely new English-only
  observable families (line quality/pressure — 5 rules; missing/
  exaggerated body parts; house/tree; centering; the "mood or concern
  flags" family).

## Unique rule count and classification (from `rules_registry_v2.json`)

**41 total rules**: 19 original production IDs (unchanged) + 22 new
`EN_COMPILED_*` IDs.

| Observability class | Count | Meaning |
|---|---|---|
| `static_direct` | 7 | Real, computed measurement exists (6 wired to an evaluator; 1, `placement_center`, computable from an existing feature but not yet wired to any evaluator) |
| `static_detector` | 24 | Single static image, needs a detector that does not exist |
| `static_proxy` | 3 | A real proxy feature exists (`stroke.intensity_proxy`, `stroke.fragmentation`) but is not wired to any evaluator, and must never be described as the literal physical quantity (e.g. "pencil pressure") |
| `process_required` | 3 | Precondition describes the drawing *process*, not observable from the finished image (`flowers_clouds_sun`'s "while distracted", `over_erasing`, `refusal_to_draw`) |
| `longitudinal_required` | 3 | The source's own word "repeated" requires comparing multiple drawings over time (the 3 "repeated ___ scenes/themes" rules) |
| `not_operational` | 1 | `very_small_drawing` (absolute figure size) — no physical scale reference exists in a scanned/photographed drawing |

**Directly measurable rules today**: the original 6 tier-1
composition/placement rules (`coverage_small/about_half/full`,
`placement_top/left/right`) — `allowed_output_level:
individual_heuristic_only`. **Every other rule (35 of 41) is
`allowed_output_level: disabled`** — no new detector, proxy wiring, or
evaluator was built in this phase (explicitly out of scope).

## Threshold provenance (unchanged from Phase 1, re-verified)

| Threshold | Status |
|---|---|
| `coverage_small` <=20% | Directly sourced (PDF states "20%" exactly) |
| `coverage_about_half` 40-60% | Center sourced ("about 50%"), tolerance band invented |
| `coverage_full` >=90% | Invented stand-in for "covers the whole page" |
| `placement_top/left/right` 0.4/0.6 splits | Entirely invented — neither source gives a number |

## Constructs currently capable of real convergence

**Exactly one**: `fear_or_insecurity_pattern`, via `coverage_small` +
the expressive-content model's Fear output (see
`docs/AGGREGATION_POLICY.md`). Every other construct remains
theoretical until either (a) a real detector unlocks a second
`static_detector`/`static_proxy` rule mapped to the same construct as an
executable rule, or (b) two executable tier-1 rules are ever remapped to
share a construct (not done in this phase, to avoid forcing
convergence — see `docs/CONSTRUCT_MAPPING_RATIONALE.md`).

## What this audit does NOT claim

Per the task's explicit instruction, the compiled PDF's own evidence
labels (`Weak/mixed`, `Speculative`, `Suggestive only`, `Moderate for
basic emotion only`, `Weak/common-sense`) are preserved verbatim in
`rules_registry_v2.json`'s `evidence_level_as_written` field and never
upgraded. Neither PDF is treated as validated psychological fact — both
sources self-describe as informal/non-diagnostic
(`resources/psychology_sources/rules_registry.json`'s
`scientifically_validated: false`; the English PDF's own introduction:
"not diagnostic rules; many are speculative or weakly supported").
