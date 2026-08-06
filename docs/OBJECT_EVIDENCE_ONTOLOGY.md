# Phase 2B Object-Evidence Ontology

**Status: candidate list, not a forced final taxonomy.** Built from
`PHASE2B_AUDIT_AND_PLAN.md` Sections 1-4 and the task's own suggested
candidate set (person, face, eye, mouth, hand, animal, house, tree, heart,
star), extended with two rule-driven candidates the audit found
(`circle`, `vehicle`). Classes may be merged, postponed, or removed once
real annotation counts exist (Section 13) — nothing here is presumed to
survive contact with data.

## What a detection means, and does not mean

An object detection means only: **"the drawing contains visual evidence
resembling this depicted class."** It is not, and must never be treated
as, evidence of: abuse, depression, anxiety, fear, aggression,
intelligence, superiority, family relationships, or any emotional state.
No species-level animal interpretation is made in this phase (`animal`
is a single broad class — tiger vs. fox vs. squirrel is never
distinguished here, per `PHASE2B_AUDIT_AND_PLAN.md` Section 3). A
detection is raw visual evidence; any psychological meaning is a
*separate*, later, expert-reviewed decision this phase does not make.

## Candidate classes

| Class | Inclusion criteria | Exclusion criteria | Partial-object policy | Repeated-instance policy | Minimum visible evidence | Appropriate label type |
|---|---|---|---|---|---|---|
| `person` | Any recognizable human figure (stick figure or detailed) | Non-human figures (animals, monsters) even if humanoid posture | Count as present if head + at least one limb-bearing torso segment is drawn | Record instance count, not just presence | A head shape plus one connected body element | presence + count; bbox if a region-proposal backend is used |
| `face` | A person's face with at least one recognizable feature (eyes, nose, or mouth marks) | A person figure with no facial marks at all (blank oval) | A face is "present" even if only partially detailed; expression is NOT assessed in this phase | Record count if multiple faces | One closed head-shape with ≥1 internal mark | presence only (no expression/state label) |
| `hand` | A discernible hand shape (with or without individual fingers) attached to or near a person figure | An arm ending in a plain line with no hand shape at all | Presence-only; "missing hand" (omission) is explicitly out of scope for this phase (needs the person to be reliably detected first, per `evidence_rule_engine.py::evaluate_omission`) | Record count | A rounded or fingered terminal shape at a limb end | presence only, experimental (see limitations below) |
| `animal` | Any four-legged or clearly non-human creature, general category only | Do not attempt species identification (tiger/wolf/fox/lion/squirrel are explicitly forbidden distinctions in this phase) | Count as present if body + at least one identifying feature (ears, tail, legs) is drawn | Record instance count | One connected animal-like body shape | presence + count |
| `house` | A building-like structure (walls + roof, however simple) | A single rectangle with no roof marking (ambiguous with abstract shapes) | Count as present if roof + wall boundary both drawn | Rare; record count if it occurs | Roof shape + enclosed wall area | presence only |
| `tree` | A trunk-and-canopy or trunk-and-branches structure | A single vertical line with no canopy/branch structure | Present if trunk + any canopy/branch mark exists | Record count | Trunk shape + canopy or branch mark | presence + count |
| `heart` | The conventional two-lobed heart symbol | Any other rounded shape (circle, oval) without the lobed/pointed heart outline | Present if the heart's characteristic silhouette is recognizable, even if small | Record count | The lobed-top, pointed-bottom silhouette | presence + count |
| `star` | The conventional pointed star symbol (5+ points typical) | Asterisk-like scribbles with no closed/near-closed star outline | Present if a star-like radiating/pointed shape is recognizable | Record count | ≥4 radiating points from a shared center | presence + count |
| `circle` | A closed or near-closed round shape, used as a standalone symbol (not e.g. a head or wheel already counted elsewhere) | A head, wheel, or sun already counted under `person`/`vehicle` — do not double-count the same drawn shape under two classes | Present if the loop is >75% closed | Record count | A closed or near-closed round contour | presence + count; this is also the class used by the classical-CV circularity baseline (Section 12) |
| `vehicle` | Cars, trucks, boats, planes, bicycles — any depicted mode of transport | Non-vehicle mechanical objects | Present if wheels/wings/hull plus a body shape are drawn | Record count | Body shape + at least one class-typical feature (wheel, wing, sail) | presence only |

## Explicitly postponed from this pilot (recorded, not silently dropped)

- **`eye`** and **`mouth`**: both appear in the task's original candidate
  list. Postponed to Phase 2C because (a) they are small, fine-grained
  facial components that a coarse, zero-shot, presence-only baseline is
  unlikely to localize reliably at the resolution this dataset's images
  are captured/scanned at, and (b) every rule that would actually consume
  them (`PSY_AR_EYES_WIDE_001/STERN_002/CLOSED_003`,
  `EN_COMPILED_EYES_MISSING_DETAIL_020`, `EN_COMPILED_MISSING_MOUTH_036`)
  needs a *state or omission* judgment, not mere presence — exactly the
  category `PHASE2B_AUDIT_AND_PLAN.md` Section 3 defers to Phase 2C
  regardless of detector feasibility. Attempting them now would spend
  this pilot's small annotation budget on classes whose downstream rules
  cannot be activated even on a technical success.

## Ambiguous cases (recorded for the annotation protocol)

- A drawn sun with rays vs. a `star`: a sun is **not** a `star` for this
  ontology — annotators mark `uncertain` if genuinely unclear which was
  intended, never guess.
- A humanoid animal (e.g. a cartoon bear standing upright, wearing
  clothes): counted as `animal`, not `person` — the presence of
  non-human features (tail, snout, ears) takes precedence.
- A wheel drawn alone, unattached to any vehicle body: not counted as
  `circle` or `vehicle` — too ambiguous to assign either label
  confidently; marked `not_assessable` per the annotation protocol.
- A face with no body (floating head): counts as both `person` (a person
  is present) and `face`.

## Human-review requirement

Per the task's explicit constraint, the following classes/rule pairings
must never be consumed by a psychological rule without dedicated expert
review, **regardless of Phase 2B's detector performance**:
`animal` → any species-specific interpretation (forbidden outright, not
just gated); `hand` → any omission-based rule; `face` → any
expression/state rule. This phase produces detection evidence only; no
rule in `rules_registry_v2.json` is activated by this phase under any
confidence level (`PHASE2B_AUDIT_AND_PLAN.md` Section 3, restated as a
hard constraint here).
