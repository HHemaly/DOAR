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

**One genuinely supported addition was found, but it is explicitly
non-psychological**: `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006` (scribble-vs-
representational developmental stage) is robustly evidenced (p<.001, age
effect), buildable from DOAR's existing feature infrastructure with no new
dependency, and directly answers the request to find "objective observations
that improve description or model performance even when no psychological
interpretation is justified." Recommended as a Tier-1 objective-feature
candidate and possible emotion-model covariate — **not** as a rule with any
emotional/personality/diagnostic interpretation attached, ever.

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

## Standing rule

No entry in this register changes any rule's `activation_status` or any
detector's status. Only `DECISION_LOG.md`-recorded approvals do that, and
only after the detector/evidence requirements in `SCIENTIFIC_LIMITATIONS.md`
Section 8 are met.
