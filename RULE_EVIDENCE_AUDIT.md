# DOAR — Rule + Literature + Clinical Reasoning Audit (V1.5)

Final audit before the thesis benchmark. Confirms actual repository state
(never assumed prior estimates), audits both source PDFs directly, verifies
external literature against primary sources where reachable, and designs
the hierarchical reasoning model (Sections 6–13) the clinician/parent output
layers will eventually consume. **This session did not activate any new
rule, detector, or output path.** Every design artifact here is a schema/
definition, not a change to `rules.py`, `concerns.py`, or `rule_engine_v2.py`.

Companion files: `RULE_EVIDENCE_MATRIX.csv` (one row per unique rule),
`resources/psychology_sources/external_literature_register.json`
(verified external literature), `RULE_RELATIONSHIP_GRAPH.json`,
`CONCERN_DOMAIN_MAP.json`, `EVIDENCE_AGGREGATION_POLICY.md`,
`CLINICIAN_OUTPUT_SCHEMA.md`, `PARENT_OUTPUT_SCHEMA.md`,
`LLM_SYNTHESIS_POLICY.md`, `RULE_FREEZE_REPORT.md`.

---

## 1. Verified current state (do not reuse the pre-session estimate)

The task's own prior estimate — "~67 source rows, ~41 unique rules, ~10
executable rules" — is **accurate for the full draft/candidate landscape**,
but conflates two different things that must be reported separately:

| Layer | File | Count | Status |
|---|---|---|---|
| Raw source rows (both PDFs, pre-consolidation) | `resources/psychology_sources/source_rule_catalog.json` | **67** (19 Arabic PDF + 48 English PDF) | `candidate_rule_entry_count: 62` (5 rows are methodology statements, not observable-feature rules) |
| Unique consolidated rules (draft) | `resources/psychology_sources/rules_registry_v2.json` | **41** | `status: "draft -- NOT the production registry"` |
| Unique ACTIVE rules (production) | `resources/psychology_sources/rules_registry.json`, consumed by `src/doar/rules.py` | **19** | All from the Arabic PDF only |
| Executable in the DRAFT (41-rule) registry | `allowed_output_level: individual_heuristic_only` / `validation_status: IMPLEMENTED_UNVALIDATED` | **10** | This is what "~10 executable" actually refers to |
| Executable in PRODUCTION (19-rule) registry, confirmed independently by reading `rules.py`'s `_TIER1_DISPATCH` | `coverage_about_half/full/small` + `placement_top/left/right` | **6** | The only rules that can ever produce `weak_support` today |
| Blocked, `DETECTOR_UNAVAILABLE`, in production | Eyes ×3, animals ×4, geometry, stars, flowers/clouds/sun, circles, transport, hearts | **13** | Structurally return `missing_detector`, never `not_matched` |
| Engine-level compound mechanism | `concerns.py::derive_concerns` | 1 (`concern_converged_001`) | Implemented, unit-tested, **`CONCERNS_ENABLED = False`** in production |

**The corpus was already reduced/corrected before this session** (dated
2026-08-02 per `RULE_SOURCE_REGISTER.csv`/`RULE_COVERAGE_MATRIX.csv`'s own
file suffixes `*_pre_structural_fix_*`/`*_pre_dev_stage_correction_*`).
Reporting "67/41/~10" without this distinction would overstate what is
actually live. This audit's `RULE_EVIDENCE_MATRIX.csv` covers all **41**
draft-registry rules (the most complete existing per-rule dataset), and
notes production-vs-candidate status per row.

## 2. PDF audit (both PDFs read directly this session, not re-derived from the JSON transcription alone)

**`التحليل النفسي للصور.pdf`** (2 pages, Arabic, unattributed psychologist
handout). Content: eyes (wide/stern/closed), animals (tiger-or-wolf, fox,
squirrel, lion), geometric shapes (general + stars, flowers/clouds/sun,
circles), transport, hearts, size (half/full/small page coverage),
placement (top/left/right). **19 distinct claims** — read directly this
session and cross-checked word-for-word against `source_rule_catalog.json`'s
`SRC_AR_001`–`019` transcriptions: **faithful, no discrepancy found.**

**`resources/psychology_sources/child_drawing_rules_compiled.pdf`** (3 pages,
English, self-described as "not diagnostic rules; many are speculative or
weakly supported"). Content, by section: eyes/face (5, incl. face
expression and missing/undetailed eyes), animals (5, incl. general animal
choice), geometric/symbolic shapes (4), hearts (1), animals/objects with
stories (4: transport, house, tree, repeated monsters/danger/injury),
size/page use (4), placement (4, incl. centering), line quality/pressure
(5), missing/altered details (4), repeated themes (3: monsters, family
conflict, isolation), mood/concern flags (4: dark colours+sad+isolation,
refusal to draw, excessive detail, background neglect). Plus a 4-row
"how to use the guide" methodology table (single drawing vs. patterns vs.
child's explanation vs. age/skill) — correctly excluded from the rule
registry as non-observable-feature content.

**Finding: no useful rule content from either PDF is missing from the
41-rule draft registry.** Every category above maps to an existing
`PSY_AR_*` (production) or `EN_COMPILED_*` (candidate) rule_id — verified
by direct cross-reference, item by item, not assumed. Duplicates across the
two PDFs (e.g. both describe wide eyes, closed eyes, tiger/wolf, stars) were
**already explicitly mapped**, not silently merged: `source_rule_catalog.
json`'s own `relationships` array records 21 `near_duplicate`, 3
`related_but_distinct`, and 7 `expands` (English-only, no Arabic
counterpart) relationships at the source-row level, all resolved into
single `rule_id`s via each rule's `source_entry_ids` field (e.g.
`PSY_AR_EYES_WIDE_001.source_entry_ids == ["SRC_AR_001", "SRC_EN_001"]`).
`RULE_RELATIONSHIP_GRAPH.json` re-expresses the handful of genuinely
**unmerged** related-but-distinct pairs (e.g. relative page-coverage vs.
absolute figure size) at the rule level.

**Indicator-category coverage check** (per this task's Section 2 checklist):
positive/neutral (flowers/clouds/sun, hearts, face expression-positive,
placement/size) ✓ present; anxiety/stress (fear/insecurity target_construct,
protection/coping, caution/low-energy) ✓ present; low-mood/depression (face
expression-distress, missing mouth, dark-colours+sad+isolation, repeated
family conflict) ✓ present; social withdrawal (circles, isolation
placements, repeated isolation themes) ✓ present; aggression/threat (stern
eyes, tiger/wolf, heavy pressure, zigzag lines) ✓ present; developmental/
attention-adjacent (repetition/rigidity, disorganisation/fragmentation,
excessive detail, background neglect) ✓ present, **but never labelled ADHD
anywhere in either source**; **abuse/maltreatment — absent from both PDFs
entirely.** Neither source proposes a single abuse-specific indicator; this
is a genuine gap, addressed via external literature only (Section 7 below),
never as a PDF-sourced rule.

## 3. Scientific evidence audit — methodology

For every one of the 41 rules, `RULE_EVIDENCE_MATRIX.csv` distinguishes the
**source claim** (what the PDF literally says — `source_claim` column) from
**external evidence** (`external_evidence_summary`, `evidence_direction`,
`evidence_strength_as_written` columns). The result, verified by direct
count from the generated matrix:

| `evidence_direction` | Count | Meaning |
|---|---|---|
| `insufficient` | 38 | No independent study found (or found but explicitly `not_found_for_[X]_claim`) that supports the SPECIFIC claim as written |
| `partial` | 2 | `PSY_AR_EYES_STERN_002`, `PSY_AR_SIZE_SMALL_016` — an indirect literature thread exists but does not validate the specific claim |
| `conflicting` | 1 | `PSY_AR_SIZE_HALF_014` — cited size/depth research is itself internally mixed |
| `support` / `contradict` | 0 | **No rule in this corpus reaches either extreme** — consistent with the corpus's own self-description as a heuristic guide, not a validated instrument |

`context_transfer_justification` (new column, this audit): `expert_review_
required` (27 rules — bare clinician/heuristic claims with no independent
source study behind them at all, so neither "applicable" nor
"task-specific" is defensible, only "needs expert sign-off before any use"),
`insufficient_evidence` (7 — the underlying construct is not even
measurable today: `process_required`/`longitudinal_required`/
`not_operational`), `partially_applicable` (7 — the OBSERVABLE is
prompt-independent, but the CITED SOURCE STUDIES behind it used an
instructed task: `PSY_AR_SIZE_*`, `PSY_AR_PLACE_*`,
`EN_COMPILED_PLACEMENT_CENTER_029`, `EN_COMPILED_VERY_SMALL_DRAWING_026`).
**`directly_applicable` was never assigned to any of the 41 rules** — this
audit found no rule with both a prompt-independent observable AND a
free-drawing-validated source study.

## 4. Required external literature — verified this session, not assumed

All 7 items the task specified were checked via independent live web search
this session (title/authors/journal/DOI/PMID cross-confirmed across ≥2
sources where possible; exact statistics taken from search-result summaries
of the source abstract, not invented). Full detail, including limitations
and `context_transfer_justification`, is in
`resources/psychology_sources/external_literature_register.json`
(`required_seven_items`).

1. **HTP meta-analysis** (Guo et al. 2023, PMID 36683989, PMC9848786,
   *Frontiers in Psychiatry*) — **confirmed**: 30 studies, 665 effects,
   6,295 participants, matching the task's figures exactly. 50 characteristics
   recurred ≥3×; 39 of those were significant in ≥1 study, but cross-study
   consistency was low. `task_specific_not_transferable`.
2. **Tree imagery meta-analysis** (PMID 42437128, DOI `10.1155/da/9571222`,
   *Depression and Anxiety*, 2024) — **confirmed** via two independent
   searches (title, journal, DOI, PMID all match); 42 studies, 24
   tree-specific characteristics found predictive. Full text not fetched
   (blocked); participant count (8,552) taken from the task's own figure,
   not independently re-derived. `task_specific_not_transferable`.
3. **Allen & Tussey abuse review** (DOI `10.1177/1524838012440339`, PMID
   22467642, *Trauma, Violence, & Abuse*, 2012) — **confirmed**: no graphic
   indicator or scoring system reliably discriminates abused from
   non-abused children. This is the boundary reference that grounds DOAR's
   absolute prohibition on abuse inference from a drawing.
4. **AAP ADHD Clinical Practice Guideline** (Wolraich et al., *Pediatrics*
   2019;144(4):e20192528, PMID 31570648, PMC7067282) — **confirmed**.
   Defines the multi-informant, multi-setting clinical process a real ADHD
   diagnosis requires — a bar no drawing-only system approaches.
5. **Burkitt, Barrett & Davis 2003** (DOI `10.1111/1469-7610.00134`,
   *Journal of Child Psychology and Psychiatry* 44(3):445–455) —
   **confirmed**. N=330, ages 4–11, **instructed** colouring-in-a-template
   task. **This is a DIFFERENT paper from the pre-existing
   `LIT_COLOUR_EMOTION_001` entry** (DOI `10.1080/01443410.2013.785059`,
   *Educational Psychology* 2014) — both real, distinct, now both catalogued
   without conflation.
6. **Crawford et al. 2012** (DOI `10.1002/icd.742`, *Infant and Child
   Development* 21:198–215) — **confirmed**. Direct empirical caution: even
   in an instructed, externally-framed task, colour use only partially and
   inconsistently tracked assigned emotional valence. Compared directly
   against item 5 (and the existing `LIT_COLOUR_EMOTION_001`) in
   `RULE_RELATIONSHIP_GRAPH.json`'s `literature_level_contradictions` — the
   colour/emotion literature is **genuinely contested**, not simply
   supportive-with-a-footnote.
7. **Philippsen, Tsuji & Nagai 2022** (DOI `10.3389/fpsyg.2022.783446`,
   PMID 36438392, PMC9697182, *Frontiers in Psychology*) — **confirmed**
   (journal previously unstated in the carried-forward entry; now pinned).
   N=104 children, 621 drawings — matches the task's figures exactly.
   Purely developmental (representational content ↑, scribbling ↓ with age,
   p<.001); explicitly not a psychiatric measure. `partially_applicable`
   (source used CNN features on a stimulus-completion paradigm, not DOAR's
   classical features on genuinely unrestricted free drawing).

**Pre-existing DOAR references audited and preserved, not re-derived**: all
13 `LIT_*` entries in `LITERATURE_CANDIDATE_REGISTER.csv` and the 3
boundary references in `RULE_SOURCE_REGISTER.csv` rows 21–23 are carried
forward into `external_literature_register.json` with their **original**
verification status intact (never silently upgraded to "verified" without
an independent check performed in the session that wrote it).

**New this session** — 2 published scoring/combination systems, for
Section 5 below: Koppitz Emotional Indicators (multi-source reliability/
validity search) and Naglieri/McNeish/Bardos's DAP:SPED. See
`external_literature_register.json`'s `scoring_system_references_new_
this_session`.

**No new PDF-adjacent rule was found missing from the registry during
literature review** — the additional search (Section 4's "search for
relevant literature we missed") surfaced confound/context literature
(cultural, motor-skill, material/medium — all pre-existing in
`LITERATURE_CANDIDATE_REGISTER.csv`) and the two scoring systems above, not
a new observable feature absent from both PDFs.

## 5. Rule combinations / published scoring systems

Investigated per the task's explicit request — not previously covered by
this corpus's own registers.

**Koppitz Emotional Indicators** (30-item composite HFD scoring system,
DAP protocol). Real, standardized, widely studied. Inter-scorer reliability
acceptable (phi 0.75–0.81 in one validation study) but **test–retest
reliability poor** (phi <0.43 for 28/30 individual indicators and the total
score). Discriminates severely disturbed children from controls, but
**fails to discriminate mildly disturbed children from typical peers on
average scores** — one study found school psychologists could not tell
counseling-referred from non-referred pupils' drawings by inspection alone.
**Not transferable**: instructed DAP protocol; would require a full
body-part-presence/distortion/size-anomaly detector far beyond DOAR's
roadmap; the system's own native-context reliability is weak enough that a
structurally similar DOAR mechanism would inherit the same concern even if
the detector existed.

**DAP:SPED** (Naglieri, McNeish & Bardos 1991) — a standardized, quantitative
scoring system for emotional-disturbance screening from three Draw-A-Person
figures. Mixed validity: some studies find it a significant, moderate
predictor of internalizing disturbance; others report correlations "at best
small, resulting in inadequate diagnostic accuracy." **Not transferable**
for the same structural reasons as Koppitz.

**What IS transferable — the general principle only, never a borrowed
weight or cutoff**: both are real precedents for *structured, multi-
indicator, transparently-reported* drawing assessment as a methodology.
`EVIDENCE_AGGREGATION_POLICY.md` builds DOAR's own transparent-count
approach on this precedent explicitly, while refusing every one of their
specific items, weights, or validated cutoffs (none apply to free drawing,
none apply to DOAR's available features, and even in their native context
their psychometrics are weak). **No published weight, effect size, or
cutoff from either system is used anywhere in this audit's design work.**

## 6. Hierarchical rule model (design, not implementation)

| Level | Content | DOAR status today |
|---|---|---|
| 1. Verified observations | e.g. `sad face`, `person`, `missing hand`, `small size`, `dark line appearance` | Partial — 6 composition observables real; 27 content-conditional observables need detectors that don't exist |
| 2. Atomic rules | The 41 rules in `RULE_EVIDENCE_MATRIX.csv` | 19 active (production), 22 draft candidates |
| 3. Evidence families | `evidence_family` field, 15 families (e.g. `facial_feature_style`, `missing_body_part`, `size_composition`) — `RULE_RELATIONSHIP_GRAPH.json`'s `same_evidence_family_edges` | Already exists in the draft registry; formalized here as explicit graph edges so a future aggregator never double-counts (e.g. frown + tears + sad-face-label all belong to one `facial_feature_style`/`colour_mood_flags`-adjacent family, not three independent signals) |
| 4. Compound rules | None defined yet — this audit found no literature-justified, DOAR-measurable compound rule strong enough to encode; `concerns.py`'s existing convergence engine is the only compound mechanism, and it is generic (any ≥2 sources), not indicator-specific | Implemented, tested, **disabled** |
| 5. Concern domains / candidate hypotheses | `CONCERN_DOMAIN_MAP.json` — 8 real domains + `neutral_descriptive_only(_no_construct_proposed)`, each mapped to its candidate atomic rules | Design only — every domain is `NOT_CURRENTLY_PRODUCIBLE` |
| 6. Longitudinal modifier | `first_occurrence` / `repeated` / `persistent` / `increasing` / `decreasing` — see below | Design only; 3 rules already flagged `longitudinal_required` in the draft registry (repeated monsters/danger, repeated family conflict, repeated isolation) but no multi-drawing case model exists in DOAR today |

**Longitudinal modifier definition** (no probability invented): a
`longitudinal_required` rule's evidence, when multiple dated drawings from
the same case exist, carries one of — `first_occurrence` (seen once),
`repeated` (seen ≥2 times, not necessarily consecutive), `persistent` (seen
in the majority of a case's drawings), `increasing` /`decreasing` (frequency
trend across drawings ordered by date). This is a **descriptive tag on
existing evidence**, never a new confidence number — "persistent" is not
"more likely true," it is "observed more often," and the aggregation policy
(`EVIDENCE_AGGREGATION_POLICY.md`) treats it exactly that transparently.

## 7. Abuse / maltreatment — display-only feature list

Per the task's explicit instruction ("DOAR SHOULD expose all verified
features that literature has linked to possible abuse/maltreatment... but
NEVER convert them into proof"). Neither PDF proposes any abuse-specific
indicator. The literature found this session and previously
(`REF_ALLEN_TUSSEY_ABUSE_2012`, `LIT_DRAWN_STORIES_MALTREATMENT_012`) is
**uniformly negative-or-inconclusive evidence about drawings' ability to
detect abuse** — there is no positive, validated "abuse indicator list" to
expose from primary research. What DOAR can honestly display, if this is
built in a future phase, is: (a) that `EN_COMPILED_REPEATED_MONSTERS_
DANGER_025`/`EN_COMPILED_REPEATED_FAMILY_CONFLICT_027`
(longitudinal, general distress/threat themes — **not** an abuse-specific
construct) exist in the corpus, always labelled as general distress
indicators, never abuse-specific; (b) `REF_ALLEN_TUSSEY_ABUSE_2012` and
`LIT_DRAWN_STORIES_MALTREATMENT_012` themselves, presented as the reason
DOAR does NOT and cannot claim drawing-based abuse detection. `CONCERN_
DOMAIN_MAP.json`'s `maltreatment_or_safety_concern` domain is intentionally
**empty of candidate atomic rules** — this is accurate, not an oversight.

## 8. Visual evidence ≠ psychological validation (unchanged, reaffirmed)

The frozen Observer → Verifier pipeline (`943859f`) determines whether an
object is **visually present** (e.g. "lion detected + independently
verified"). This audit's entire hierarchy sits **downstream** of that: a
verified `VisualEntity` only ever becomes a candidate INPUT to Level 1
("lion" observed) — whether Level 2's `PSY_AR_ANIMAL_LION_007` attaches any
interpretation to it is governed entirely by that rule's own `allowed_
output_level` (`disabled` — `DETECTOR_UNAVAILABLE` in the 19-rule
production registry regardless; the animal-species classifier this rule
would need does not exist). **"Lion detected + verified" does not, and
under the current registry cannot, produce "superiority."** No change to
the Observer/Verifier pipeline was made or is proposed by this audit.

## 9. Rule output policy — see `RULE_EVIDENCE_MATRIX.csv`'s `allowed_output_level` column

Per-rule, not a blanket policy: `allowed_output_level` in the draft
registry is `disabled` (31 rules) or `individual_heuristic_only` (10
rules) — corresponding to this task's LEVEL 0 (objective fact — always
available, e.g. "hands were not detected," independent of any rule),
LEVEL 1 (cautious literature-linked interpretation), LEVEL 2 (clinician
review flag / candidate hypothesis), LEVEL 3 (diagnostic). **No rule in
this 41-rule corpus is assigned LEVEL 3, and none should be** — Section 3's
own evidence-direction tally (38/41 `insufficient`) is the reason: nothing
here clears the bar Section 13 sets for LEVEL 3.

## 10. Human-like, evidence-grounded output — see `LLM_SYNTHESIS_POLICY.md`

The task asks whether an LLM could phrase output naturally rather than with
fixed template sentences. `LLM_GROUNDING_AND_SAFETY_DESIGN.md` (pre-
existing, `src/doar/chat.py`) already designs exactly this architecture for
Q&A; `LLM_SYNTHESIS_POLICY.md` (new, this audit) extends it specifically to
clinician/parent hypothesis narration — a structured evidence package in,
natural prose out, verified claim-by-claim against that package before
display, using the SAME `ClaimVerifier`/`SafetyChecker` components already
designed. No LLM is wired in by this audit; this is a schema/policy
document only.
