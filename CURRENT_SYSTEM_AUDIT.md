# Current System Audit — Phase 0 (this task)

**Status: synthesis + fresh verification.** This document answers, point by
point, the Phase-0 question list for the evidence-and-rule-system task. It
**reuses, and does not re-derive**, the prior session's
`CURRENT_CAPABILITY_AUDIT.md` and `END_TO_END_INFERENCE_TRACE.md` (both
still accurate — re-checked, not re-run in full, except where noted) and
adds what is new to this task: verification of the completed human review
(`HUMAN_REVIEW_EXPORT_VERIFICATION.md`), a full read of the sole supplied
psychology PDF, and the resulting rule-by-rule feature/threshold audit
(`RULE_FEATURE_COVERAGE.md`). Nothing here was assumed from documentation
alone — every claim below cites the file/function/session artifact behind
it.

---

## 1. Which review-export files exist, and whether genuinely complete

**Fully re-verified, not re-assumed, this task.** See
`HUMAN_REVIEW_EXPORT_VERIFICATION.md` for the complete evidence trail: 7
files (`decisions.json`, `items_registry.json`,
`human_pair_reviews.csv`, `human_group_reviews.csv`,
`reviewer_agreement_report.json`, `threshold_precision_summary.json`,
`unresolved_items.csv`), **225/225 items decided (100%)**, 0 malformed
records, 15 cross-category duplicate reviews found and resolved (14
consistent, 1 genuine contradiction flagged and conservatively resolved
toward merge in `flagged_contradictions.json`). This directly supersedes
the prior audit's finding (`CURRENT_CAPABILITY_AUDIT.md` §5): *"zero human
review decisions have been recorded as of this audit"* was true when
written (an earlier session's zombie Streamlit process on port 8501 was
silently masking new writes) and is now false — the review is genuinely
complete.

## 2. Which manifest/split currently uses those decisions

**A new candidate partition**, `outputs/phase7b/candidate_partition_review_based/`,
built this task directly from the 225 decisions via the new
`compute_duplicate_groups_from_review()` (`src/doar/partition.py`). It is
**not yet the live/approved partition** — `outputs/phase7b/final_partition/`
(threshold-6, pre-human-review) is untouched, and no code anywhere
auto-promotes a candidate to final. Two of the 8 dataset-gate checks
(`src/doar/dataset_gate.py`) remain intentionally open pending your
explicit approval: `duplicate_policy_approved`, `manifest_frozen`. No
existing trained checkpoint (§4 below) was trained on this or any
leakage-safe partition — all predate Phase 7B entirely.

## 3. Which features are genuinely calculated

Unchanged from `CURRENT_CAPABILITY_AUDIT.md` §1 item 6 / §8: **53 real
numeric features** in `features.py::objective_feature_row`
(quality/segmentation/composition/colour/stroke/shape families), each with
its own `FeatureValue.confidence`/`missing` flag; 2 shape features are
honestly hardcoded `NaN`/`missing=True` (no shape detector exists). This
task adds one new finding (`RULE_FEATURE_COVERAGE.md`, final section):
**of these 53+ real features, only 2 are ever read by the rule engine** —
`bounding_box_coverage` and `centroid_normalized`/`placement`. The other
51 are computed, stored, and available in every case's `analysis.json`,
but currently unused downstream. This is real, idle evidence, not a
missing capability — the fastest lever for expanding rule coverage without
building any new extractor (see §9 "Immediate next action" reasoning).

## 4. Which outputs are placeholders/incomplete/misleading

Unchanged and re-confirmed: `detections.json` is a hardcoded
`{"status": "unavailable", "detections": []}` written unconditionally for
every case (`case_output.py:61`, `CURRENT_CAPABILITY_AUDIT.md` §7,
`END_TO_END_INFERENCE_TRACE.md` §4.8) — not a bug, a real absent
capability, honestly labeled. `concerns.py` is disabled by design
(`CONCERNS_ENABLED = False`). Nothing new this task contradicts either
finding.

## 5. Which rules exist and where defined

**19 rules**, `resources/psychology_sources/rules_registry.json`. This
task adds the piece the prior audit did not have: **the sole source PDF
itself was read in full** (`التحليل النفسي للصور.pdf`, 2 pages, extracted
via `pypdf`) and every one of the 19 rules was matched 1:1 to a specific
PDF bullet, with exact page number and informal Arabic section heading —
see `RULE_FEATURE_COVERAGE.md`'s matrix. No rule in the registry lacks a
traceable source sentence; no PDF bullet was left unmapped to a rule
either (19 registry rules ↔ 19 distinct source bullets, a clean
one-to-one correspondence).

## 6. Whether rules are hard-coded/config-driven/only documented

**Config-driven content, hard-coded evaluation logic** — unchanged from
`CURRENT_CAPABILITY_AUDIT.md` §4: rule *text* (title, Arabic/English
wording, tier, activation status, references) lives in the JSON registry;
rule *evaluation* (which composition field to read, which comparison
operator, which numeric threshold) is hard-coded in
`rules.py::_tier1_status`. This task's new finding (below, §7) is exactly
about that hard-coded threshold layer.

## 7. Whether numerical thresholds were invented without source support

**Yes, precisely characterized for the first time this task** —
`RULE_FEATURE_COVERAGE.md`'s "Threshold or qualitative criterion" /
"Threshold source" columns and its summary section:

| Threshold | Status |
|---|---|
| `coverage_small` ≤20% | **Directly sourced** — PDF states "20%" exactly |
| `coverage_about_half` 40–60% | **Center sourced ("about 50%"), tolerance band invented** |
| `coverage_full` ≥90% | **Invented** stand-in for the PDF's qualitative "covers the whole page" |
| `placement_top` centroid_y<0.4 | **Entirely invented** — PDF gives no number for "top" |
| `placement_left` centroid_x<0.4 | **Entirely invented** — PDF gives no number for "left" |
| `placement_right` centroid_x>0.6 | **Entirely invented** — PDF gives no number for "right" |

All 6 are already tagged `IMPLEMENTED_UNVALIDATED` with a
`confidence_ceiling` ≤0.20 in the registry, and every rule result already
carries that ceiling downstream — so no invented threshold is currently
presented as validated. But the registry did not previously distinguish
*which* thresholds are sourced numbers versus invented operationalizations;
that distinction is now explicit and is the basis for Phase 1's evidence
schema (a `threshold_source` field, not just a validation-status flag).

## 8. Which PDF contains each rule (document, page, section)

Fully answered in `RULE_FEATURE_COVERAGE.md`'s matrix, column-by-column,
for all 19 rules. Only one PDF was supplied
(`التحليل النفسي للصور.pdf`, repo root) — there is no second source
document in the repository to cross-check against; `RULE_SOURCE_REGISTER.csv`'s
35 rows are a superset of *reference citations* (external literature used
for the cautionary `scientific_support` annotations), not additional
source PDFs — confirmed by reading the CSV's `citation_title` column
against `rules_registry.json`'s `references` block, which lists the same
6 external papers (PMID/DOI-bearing) repeated across the 35 rows as
per-rule applicability assessments, not 35 independent sources.

## 9. Whether existing feature/rule coverage docs agree with executable code

**Yes, re-verified.** The prior session's `RULE_AND_FEATURE_COVERAGE.md`
(113 lines, tier/activation-status table) and this task's new,
more-detailed `RULE_FEATURE_COVERAGE.md` (full PDF-page-level
provenance, per-rule) agree on every load-bearing fact: 6/19 executable,
13/19 blocked by missing detectors, zero rules scientifically validated.
No discrepancy found between either document and `rules.py`'s actual
dispatch logic (`_tier1_status`, `evaluate_rules`).

## 10. Vague, contradictory, duplicated, unsupported, non-operational rules

New findings from the full PDF read, not previously documented:

- **`PSY_AR_FLOWERS_CLOUDS_SUN_010` is non-operational in principle, not just
  unimplemented** — its precondition ("drawn while distracted/absent-minded,"
  Arabic شارد) describes the child's mental state during drawing, which no
  static image can ever measure. It should be reclassified, not left as an
  ordinary "awaiting a detector" tier-2 rule.
- **4 animal rules (`004`–`007`) are structurally identical** (one species →
  one trait) and represent one capability question (a reliable
  child-drawing animal-species classifier — itself unvalidated for this
  domain), not four independent ones.
- **No rule in the registry is internally contradictory or duplicated**
  against another rule — each of the 19 maps to a distinct source bullet.
- **The PDF itself is explicit that it is informal notes**, not an academic
  source: no citations, no numbered sections, no methodology section, no
  sample/population description anywhere in its own 2 pages. This matches
  `rules_registry.json`'s own self-declared
  `source.scientifically_validated: false`.

## 11. Genuinely working extractors/models/interfaces (re-confirmed, unchanged)

Per `CURRENT_CAPABILITY_AUDIT.md` §1/§6–11 and `END_TO_END_INFERENCE_TRACE.md`
(both re-read, not re-run, this task — no code in the traced path was
touched): quality gate, segmentation (`colour_distance` heuristic),
composition, colour binning, 53-feature objective extraction, the rule
dispatcher, 6 deterministic judges, bilingual HTML reports (with the known
parent/professional rule-visibility gap), deterministic grounded Q&A, and
case persistence/versioning are all real and exercised by tests or the
prior session's live trace. Object/component/OCR detection remains
schema-only, never wired to any executable path.

## 12. Whether unavailable evidence is incorrectly treated as pass/false/zero

**Partially, one real gap confirmed, not previously stated this bluntly**:
`detections.json`'s hardcoded `{"status": "unavailable", "detections": []}"`
is *labeled* unavailable (good) but its shape — an empty list — is
indistinguishable, to any naive downstream consumer, from "zero objects
were found by a working detector." Nothing in the current schema stops a
report generator from reading `detections: []` and writing "no objects
detected" as if that were a confirmed negative. No such misreport
currently exists in `reports.py` (checked: it never reads `detections.json`
at all today), but the schema itself does not yet prevent it. This is
exactly the risk Phase 1's `status` vocabulary (`unavailable` must never
collapse to an empty/zero value with no distinguishing status field) is
designed to close.

## 13. Which rules depend on unavailable OCR/semantic objects/body parts/shapes/spatial evidence

All 13 tier-2 rules (`RULE_FEATURE_COVERAGE.md` rows 1–13): 3 eye/face
rules, 4 animal-species rules, 6 shape/symbol rules (geometric-shape
repetition, stars, flowers/clouds/sun, circles, vehicles, hearts). Zero
rules currently depend on OCR or spatial/body-part relationships — none of
the 19 supplied rules reference text, body-part-to-body-part distance, or
multi-object spatial layout at all; that entire evidence family
(explicitly requested in the Phase 1 canonical schema) has no rule to
consume it yet in this registry.

## 14. Summary: what Phase 0 concludes

The system's real, working core (segmentation → composition/colour →
feature extraction → rule dispatch → judges → bilingual report → QA) is
solid and honestly instrumented. The two real gaps are (a) 51 of 53
already-computed features are unused by any rule, and (b) all rule
thresholds beyond the one directly-sourced percentage are either
partially or fully invented operationalizations of qualitative source
language, correctly capped at low confidence but not yet labeled with a
`threshold_source` field distinguishing *why* they're unvalidated.
Phase 1 targets both gaps directly: a canonical evidence schema that
carries every already-computed feature (not just the 2 currently wired)
plus an explicit `threshold_source` field in the rule registry.
