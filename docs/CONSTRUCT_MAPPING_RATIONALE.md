# Construct Mapping Rationale

**Status: real decisions, recorded for review.** This document explains
why each of the 41 registry-v2 rules was, or was not, mapped to one of
the 12 drawing-level constructs in `construct_registry.json`, and
explicitly logs the alternatives considered and rejected. Every mapping
is a single (`rule_id -> construct_id`, `direction: supports`) pair —
Phase 1.5 does not attempt multi-construct mappings per rule; that
simplification is stated here, not hidden.

## Why this exists

Phase 1's registry (`RULE_FEATURE_COVERAGE.md`/`rules_registry_v2.json`
draft-1) used one bespoke `target_construct` per rule (e.g.
`PSY_AR_SIZE_FULL_015 -> "self_esteem"`), which is exactly why a single
rule could be shown as a "candidate drawing-level theme" — grouping by a
1:1 construct means any single triggered rule automatically forms a
"group of one." Phase 1.5 replaces this with a small, fixed set of 12
broad constructs (`construct_registry.json`) that multiple *independent*
rules can converge on — convergence becomes meaningful again because it
requires genuinely different rules to agree, not just one rule restating
itself as its own "construct."

## Rules mapped to a construct (27 of 41)

| Construct | Rules |
|---|---|
| `affiliation_or_connection` | `PSY_AR_TRANSPORT_012`, `PSY_AR_HEARTS_013`, `PSY_AR_PLACE_RIGHT_019` |
| `caution_or_low_visual_energy` | `EN_COMPILED_LINE_LIGHT_PRESSURE_031` |
| `disorganisation_or_fragmentation` | `EN_COMPILED_LINE_SHAKY_BROKEN_032`, `EN_COMPILED_NEGLECT_BACKGROUND_041` |
| `fear_or_insecurity_pattern` | `PSY_AR_SIZE_SMALL_016`, `EN_COMPILED_REPEATED_MONSTERS_DANGER_025`, `EN_COMPILED_VERY_SMALL_DRAWING_026`, `EN_COMPILED_EXAGGERATED_BODY_PARTS_037` |
| `low_mood_or_emotional_distress_pattern` | `EN_COMPILED_FACE_EXPRESSION_021`, `EN_COMPILED_REPEATED_FAMILY_CONFLICT_027`, `EN_COMPILED_MISSING_MOUTH_036`, `EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038` |
| `positive_affective_tone` | `PSY_AR_FLOWERS_CLOUDS_SUN_010` |
| `protection_or_coping` | `PSY_AR_ANIMAL_SQUIRREL_006` |
| `repetition_or_rigidity` | `PSY_AR_GEOMETRY_008`, `EN_COMPILED_EXCESSIVE_DETAIL_040` |
| `social_distance_or_isolation` | `PSY_AR_CIRCLES_011`, `PSY_AR_PLACE_LEFT_018`, `EN_COMPILED_REPEATED_ISOLATION_THEMES_028` |
| `tension_or_anger_pattern` | `PSY_AR_EYES_STERN_002`, `PSY_AR_ANIMAL_TIGER_WOLF_004`, `EN_COMPILED_LINE_HEAVY_PRESSURE_030`, `EN_COMPILED_LINE_ZIGZAG_033` |
| `visual_dominance_or_prominence` | `PSY_AR_SIZE_FULL_015`, `EN_COMPILED_PLACEMENT_CENTER_029` |
| `high_visual_energy_or_expansiveness` | *(none directly — see note below)* |

Note: `high_visual_energy_or_expansiveness` has no rule mapped to it as a
*primary* construct in this pass — `PSY_AR_SIZE_FULL_015`'s "expansiveness/
strong energy/self-assertion" reading (from the English source) would
also fit here, but it was assigned to `visual_dominance_or_prominence`
instead because that is the closer match to the *Arabic* source's own
"self-esteem" framing, and a rule maps to exactly one construct in this
pass. Flagged here for expert review, not silently decided.

## Rules deliberately left unmapped (14 of 41)

Per the task's explicit instruction — "do not map rules merely to force
convergence" — the following were left `target_construct: null` because
their interpretation is a **personality trait**, not a **drawing-level
emotional/social/energy pattern**, or because the source's own wording is
**bidirectional** (points two different ways depending on context):

| Rule | Why left unmapped |
|---|---|
| `PSY_AR_EYES_WIDE_001` | Bidirectional: "openness/alertness" **or** "fear/surprise" in the same source sentence. |
| `PSY_AR_EYES_CLOSED_003` | Specific, idiosyncratic reading ("refusing to look inward") — doesn't fit any of the 12 broad patterns. |
| `PSY_AR_ANIMAL_FOX_005` | "Sneakiness/cunning" is a trait judgment, not an emotional/social drawing-level pattern. |
| `PSY_AR_ANIMAL_LION_007` | "Belief in superiority" is a trait judgment. |
| `PSY_AR_STARS_009` | "Wanting attention/admiration" is a trait judgment, not one of the 12 patterns. |
| `PSY_AR_SIZE_HALF_014` | Bidirectional by the source's own wording ("outgoing at times, introverted at other times"). |
| `PSY_AR_PLACE_TOP_017` | "Dreamy personality/difficulty adapting" doesn't cleanly fit any of the 12. |
| `EN_COMPILED_EYES_MISSING_DETAIL_020` | Overlaps `low_mood_or_emotional_distress_pattern` loosely ("withdrawal") but is graded "Weak/mixed" for a non-specific bundle of readings (withdrawal, low interest, *or simply limited drawing skill*) — too diffuse to assign confidently. |
| `EN_COMPILED_ANIMAL_CHOICE_GENERAL_022` | Explicitly non-specific in the source itself ("interests, story play, fear, identification, or a projective theme"). |
| `EN_COMPILED_HOUSE_023`, `EN_COMPILED_TREE_024` | Each spans 3-4 unrelated readings (safety, growth, vulnerability, stability) with no single dominant direction. |
| `EN_COMPILED_LINE_OVER_ERASING_034` | Process-required (not observable from a static image) — mapping would be premature before it can ever fire. |
| `EN_COMPILED_MISSING_HANDS_035` | "Difficulty acting/social difficulty/avoidance" is diffuse enough across three different domains to resist a single-construct fit. |
| `EN_COMPILED_REFUSAL_TO_DRAW_039` | Process-required (describes something the child did *not* draw, not visible in any image) — same reasoning as over-erasing. |

## Worked examples from the task, evaluated (not blindly accepted)

- *"small page use + frightened/sad expression + Fear/Sad expressive
  model may contribute to a fear/insecurity or low-mood pattern"* —
  **accepted**: `PSY_AR_SIZE_SMALL_016` → `fear_or_insecurity_pattern`,
  `EN_COMPILED_FACE_EXPRESSION_021` → `low_mood_or_emotional_distress_pattern`
  (a close neighbor, not the same construct — the two are genuinely
  different patterns, not merged), and the expressive-content model's
  Sad/Fear output is wired as its own independent family in
  `structured_report_v2.py`, per Section 6.
- *"full-page use + central prominence + relatively large main figure may
  contribute to visual prominence"* — **accepted**:
  `PSY_AR_SIZE_FULL_015` and `EN_COMPILED_PLACEMENT_CENTER_029` both map
  to `visual_dominance_or_prominence`. ("Relatively large main figure"
  requires object-level size, which needs a detector that does not exist
  — not added as a third contributor.)
- *"sad face + isolation + dark-colour concentration may contribute to a
  low-mood/distress pattern"* — **accepted**: `EN_COMPILED_FACE_EXPRESSION_021`
  and `EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038` both map to
  `low_mood_or_emotional_distress_pattern`. ("Isolation" as a standalone
  scene-content signal maps instead to `social_distance_or_isolation`,
  a related but distinct construct — not folded in here to avoid
  overstating convergence.)
- *"smiling face + positive symbols + Happy expressive model may
  contribute to positive affective tone"* — **accepted**:
  `EN_COMPILED_FACE_EXPRESSION_021` (smiling branch) and
  `PSY_AR_FLOWERS_CLOUDS_SUN_010` both map to `positive_affective_tone`;
  the Happy expressive-model output is wired as an independent family.

## Honest current reach

**All of the rules mapped to `low_mood_or_emotional_distress_pattern`,
`fear_or_insecurity_pattern`, and every other construct except
`visual_dominance_or_prominence` are `allowed_output_level: disabled`
today** (no detector exists for any of them — see
`docs/RULE_PDF_COVERAGE_AUDIT_V2.md`). The only construct with 2 rules
that are *both currently executable* is `visual_dominance_or_prominence`
(`PSY_AR_SIZE_FULL_015`, executable; `EN_COMPILED_PLACEMENT_CENTER_029`,
disabled — not wired to an evaluator despite being computable) — so even
this construct cannot reach Level C (combined hypothesis) on real data
today, since it needs 2 *executed* rules from 2 *independent evidence
families*, not just 2 mapped rules. This is stated plainly, not
discovered as a surprise later: see `PHASE1_5_IMPLEMENTATION_REPORT.md`
for exactly which constructs can converge on real evidence today (answer:
none, without the expressive-content model's contribution — see
`docs/AGGREGATION_POLICY.md`).
