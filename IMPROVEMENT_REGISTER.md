# Improvement Register

Append-only, like `DECISION_LOG.md`. Tracks proposed improvements to DOAR's
rule catalogue and objective-feature set as literature review proceeds.
Nothing recorded here is activated or user-facing — see
`RULE_COVERAGE_MATRIX.csv` / `rules_registry.json` for what actually runs.

---

## 2026-08-02 — Literature review round 1

Source material: `LITERATURE_CANDIDATE_REGISTER.csv` (6 candidates, full
detail per candidate). Comparison against the 19 existing rules in
`rules_registry.json` / `RULE_SOURCE_REGISTER.csv`, per the 5 categories
requested.

### 1. Genuinely supported additions (new, activatable psychological rule candidates)

**None found in this pass.** Every psychological-interpretation candidate
researched this round was either directly self-contradictory in the
literature (`LIT_COLOUR_EMOTION_001`), structurally unmeasurable from DOAR's
static-image input (`LIT_PENCIL_PRESSURE_002`), far beyond current detector
scope with confounded evidence (`LIT_HFD_KOPPITZ_INDICATORS_003`), or
population/protocol-mismatched to DOAR entirely (`LIT_HFD_DEPRESSION_ADULT_004`,
adults not children, instructed not spontaneous). This is reported honestly
rather than manufacturing a candidate to fill this category — see
`LITERATURE_CANDIDATE_REGISTER.csv` for the full reasoning per candidate.

**One candidate looked like a genuinely supported addition, but on review its
transfer to DOAR is itself an unvalidated hypothesis, not an established
fact** — correction applied 2026-08-02, see below. `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`
(scribble-vs-representational developmental stage) IS robustly evidenced
(p<.001, age effect) *in its source study's population, protocol, and
measurement approach*. What is NOT established: the source measured this via
CNN-extracted features on a stimulus-completion task, not DOAR's classical
handcrafted features on unrestricted free drawings — whether DOAR's existing
feature infrastructure can capture the same construct at all, or whether the
finding transfers to genuinely free drawing, are both open questions
requiring their own DOAR-specific pilot. This still directly answers the
request to find "objective observations that improve description or model
performance even when no psychological
interpretation is justified" — but as a **hypothesis to pilot**, not an
operational candidate. Downgraded from an earlier draft of this register
that called it "recommended" without that caveat. If ever piloted, it stays
a Tier-1 objective-feature candidate and possible emotion-model covariate at
most — **never** a rule with any emotional/personality/diagnostic
interpretation attached.

### 2. Duplicates or correlated indicators

- `LIT_COLOUR_EMOTION_001` overlaps with an **existing DOAR feature**, not an
  existing rule: `colour.py`/`features.py` already compute `colour_proportions`
  and `dominant_colour`, but no registry rule currently uses them. The
  literature is too self-contradictory to justify attaching an interpretive
  rule to this already-measured feature — flagged for research-only status,
  not implementation.
- `LIT_SELFESTEEM_METHOD_RELIABILITY_005` is not a new candidate at all — it
  is **directly correlated with / reinforcing** the existing rule
  `PSY_AR_SIZE_FULL_015` (`coverage_full` → "high self-esteem"), which
  already carries `scientific_support: not_found_for_threshold_claim` and a
  0.15 ceiling. This meta-analysis is additional citation-strength for that
  existing scepticism, recorded in `RULE_SOURCE_REGISTER.csv`, not a new row
  in the rule registry.

### 3. Rules supported only in HTP/DAP/family or other instructed protocols

Confirmed and reinforced for candidates researched this round:
`LIT_PENCIL_PRESSURE_002` (Draw-A-Person/Bender/car), `LIT_HFD_KOPPITZ_INDICATORS_003`
(Draw-A-Person), `LIT_HFD_DEPRESSION_ADULT_004` (Draw-A-Person, adults) are
**all** exclusively instructed-protocol literature. None apply to DOAR's
spontaneous free-drawing scope per `SCIENTIFIC_LIMITATIONS.md` Section 4 —
consistent with why the existing registry's own `PSY_AR_SIZE_*` and
`PSY_AR_PLACE_*` rules already flag their own source studies
(`REF_SIZE_2004`, `REF_SIZE_DEPTH_2013`, `REF_PLACEMENT_1992`) as
instructed-task-derived. This is now a broader, repeated pattern across
every drawing-assessment literature area searched, not an isolated caveat on
three rules — worth stating plainly: **the overwhelming majority of rigorous
empirical drawing-interpretation research is instructed-protocol research.**
Spontaneous free-drawing-specific interpretive literature is scarce; the one
directly on-point paper found this round (`LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`)
turned out, on close reading, to also not be pure free drawing (a stimulus-
completion paradigm) — see its `spontaneous_or_instructed` field for the
exact distinction. This scarcity is itself a documented evidence gap, not a
search failure — see the final report for what it implies.

### 4. Unsupported rules for which detector development would have little scientific value

The 4 animal rules (`PSY_AR_ANIMAL_TIGER_WOLF_004`, `PSY_AR_ANIMAL_FOX_005`,
`PSY_AR_ANIMAL_SQUIRREL_006`, `PSY_AR_ANIMAL_LION_007`) remain the clearest
case: no literature — in this round or the Phase 3 planning round — supports
any animal-species-specific psychological interpretation in any protocol,
spontaneous or instructed. Building a species-level detector for these would
only ever produce a geometric observation ("this drawing contains an
animal resembling X") with zero attachable validated interpretation. If any
detector work is ever justified here, a coarse "animal present" flag would
capture the only defensible signal at far lower cost/risk than fine-grained
species identification — species-level detail buys no additional scientific
value given the interpretation attached to it is unsupported regardless of
species.

### 5. Existing rules that should remain catalogued but be retired from implementation priority

**Revising the Phase 3 priority order** based on this round's findings: the
3 eye rules (`PSY_AR_EYES_WIDE_001`, `PSY_AR_EYES_STERN_002`,
`PSY_AR_EYES_CLOSED_003`) were previously ranked priority #2 (medium
feasibility) in `PHASE3_DETECTOR_EVALUATION_PLAN.md`. This round's deeper
look at the broader HFD/facial-expression-indicator literature
(`LIT_HFD_KOPPITZ_INDICATORS_003`, `LIT_HFD_DEPRESSION_ADULT_004`) shows that
even the most rigorous, best-controlled instructed-protocol studies in this
space produce weak, sometimes statistically fragile, internally inconsistent
findings (e.g. the shaded-eyes result significant by logistic regression but
not by ROC/AUC in `LIT_HFD_DEPRESSION_ADULT_004`). Combined with the
project's own existing citation for eye style
(`REF_EYES_EXPRESSION_2007`, already correctly limited to "children CAN
intentionally depict emotion," not "eye style reveals true felt emotion"),
the interpretive ceiling for this rule family is now shown to be lower than
its Phase 3 feasibility ranking implied. **Recommendation: keep these 3
rules catalogued (never delete, per policy), but deprioritize
face/eye-detector work below where it was ranked in the Phase 3 plan** — a
successful detector here would still only support a weakly-evidenced
interpretation. See the final report for the revised priority order.

---

---

## 2026-08-02 — Literature review round 2 (broader pass)

Source material: `LITERATURE_SEARCH_LOG.md` (10 pre-specified queries across 7
topic areas, reproducibly logged with screening decisions) and 7 new rows in
`LITERATURE_CANDIDATE_REGISTER.csv` (candidates 007–013). Round 1's 6
candidates are re-included below for a single complete picture. Per your
instruction, every candidate is sorted into exactly the 6 categories you
specified — plus a **confounders** note, which doesn't fit any of the 6
cleanly but was too load-bearing to omit; flagged as an addition, not a
substitution.

### Useful objective observation

- `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006` (scribble-vs-representational stage)
  — **status downgraded this session**: the underlying developmental finding
  is robust in its source population/protocol, but transfer to DOAR's
  classical features is an unvalidated hypothesis, not an established fact
  (see the correction applied to both CSVs and `DECISION_LOG.md`).
- `LIT_FRAGMENTATION_LOCAL_PROCESSING_013` (fragmentation) — the *feature*
  (`stroke.fragmentation`) already exists in DOAR for unrelated purposes; see
  below for why its *interpretation* is rejected outright, not merely
  deprioritized.

### Possible emotion-expression feature

- `LIT_EMOTION_FACE_ENCODING_007` (standard facial-feature conventions for
  depicted emotion: mouth/eyebrows/eyes) — **the single most promising lead
  found across both rounds**, because it targets DOAR's actual mandate
  (emotional-expression depiction) rather than a personality trait, unlike
  every one of the existing 3 eye rules. Explicitly flagged
  `SEARCH SUMMARY ONLY` — the primary source (PubMed 17165414) blocked full-
  text verification this session (cookie wall). **Do not act on this beyond
  cataloguing until round 3 secures the full text.**

### Instructed-protocol research only

`LIT_PENCIL_PRESSURE_002`, `LIT_HFD_KOPPITZ_INDICATORS_003`,
`LIT_HFD_DEPRESSION_ADULT_004`, `LIT_KFD_VALIDITY_011`,
`LIT_DRAWN_STORIES_MALTREATMENT_012`. Five of thirteen candidates across both
rounds — confirming round 1's observation that this is the dominant category
in the field, not an artifact of a small sample.

### Technically unmeasurable (from DOAR's static-image input)

`LIT_PENCIL_PRESSURE_002` (also instructed-protocol — the two categories
aren't exclusive; physical pressure requires a digitizer at capture time,
which DOAR never has).

### Conflicting or unsupported

`LIT_COLOUR_EMOTION_001` (directly contradictory studies),
`LIT_SELFESTEEM_METHOD_RELIABILITY_005` (meta-analytic evidence *against* the
method class an existing rule depends on), `LIT_FRAGMENTATION_LOCAL_PROCESSING_013`'s
*interpretation* specifically (not its underlying feature — see below).

### Irrelevant to DOAR

`LIT_DRAWN_STORIES_MALTREATMENT_012` (mandatory verbal/narrative component
outside DOAR's image-only scope) doubles into this category alongside
instructed-protocol-only.

### Confounders (cross-cutting — not one of the 6 requested categories, recorded because omitting it would be dishonest)

- `LIT_CULTURAL_CONFOUND_008` — culture/country and gender-linked patterns in
  size, pose, shading, and subject choice (e.g. vehicles). **Action**: add as
  an explicit limitation to `PSY_AR_SIZE_*` and `PSY_AR_TRANSPORT_012`.
- `LIT_MOTOR_SKILL_CONFOUND_009` — fine motor skill as an independent
  predictor of drawing-assessment scores. Reinforces limitations already
  present on some rules; not new in kind, new in citation strength.
- `LIT_MATERIAL_MEDIUM_CONFOUND_010` — **genuinely new finding, not
  previously documented anywhere in DOAR's registers**: paper size and
  drawing medium (marker/tablet/finger, watercolor vs. pencil) measurably
  affect completion, detail, and colour brightness. DOAR's `coverage_*` and
  `colour_*` features have no way to account for this — the pipeline never
  records capture medium or paper size. **Action**: add to
  `SCIENTIFIC_LIMITATIONS.md` as a first-class limitation on the *objective
  features themselves*, not just on their psychological interpretation.

### A flag worth stating plainly: `LIT_FRAGMENTATION_LOCAL_PROCESSING_013`

This candidate is recorded specifically as a **do-not-pursue flag**, not a
finding to build on. A single unverified search-summary sentence associated
drawing fragmentation with a "local processing bias" — a construct with
autism-spectrum-cognition connotations in parts of the literature. DOAR's
`stroke.fragmentation` feature already exists and stays in use for its
current, unrelated, non-diagnostic purposes — but this specific
*interpretation* must never be proposed as a rule, given the explicit,
absolute prohibition on autism inference from a drawing
(`SCIENTIFIC_LIMITATIONS.md` Section 3). Recorded here so a future
contributor who finds the same association in stronger form sees this flag
before proposing anything.

### Comparison against the 19 existing rules (updating round 1's 5 categories)

1. **Genuinely supported additions**: still none with full verification.
   `LIT_EMOTION_FACE_ENCODING_007` is the first candidate across both rounds
   worth actively pursuing full-text verification for — but it is not yet
   supported, only promising.
2. **Duplicates/correlated indicators**: `LIT_CULTURAL_CONFOUND_008` and
   `LIT_MOTOR_SKILL_CONFOUND_009` correlate with limitations several existing
   rules already list in general terms — round 2 supplies citable specifics.
3. **Instructed-protocol-only**: confirmed again as the dominant pattern (5 of
   7 new candidates).
4. **Unsupported rules where detector development has little scientific
   value**: unchanged from round 1 — the 4 animal rules remain the clearest
   case; no new evidence this round changes that.
5. **Retired from implementation priority**: unchanged from round 1 (face/eye
   rules demoted) — **but note the tension**: `LIT_EMOTION_FACE_ENCODING_007`
   is about EXPRESSION depiction (well-matched to DOAR's mandate) while the
   existing 3 eye rules are about PERSONALITY TRAITS (poorly matched,
   correctly demoted). If round 3 verifies candidate 007, it would justify a
   *new*, differently-worded Tier-2 rule — not un-deprioritizing the existing
   trait-based ones, which stay demoted on their own (separate) merits.

---

## Standing rule

No entry in this register changes any rule's `activation_status` or any
detector's status. Only `DECISION_LOG.md`-recorded approvals do that, and
only after the detector/evidence requirements in `SCIENTIFIC_LIMITATIONS.md`
Section 8 are met.
