# Page-Reference Model (DOAR-TRACE Phase 2A.1, Section 3)

## Why this exists

Phase 2A's `page_frame.py` answers one question: "is the whole page
likely visible?" (5 statuses, `docs/PAGE_FRAME_ASSESSABILITY.md`). It
never answers a second, distinct question every page-relative
measurement actually needs: **which region of the image counts as "the
page"?** Before this phase, every page-relative feature
(`composition.bounding_box_coverage`, centroid, margins) silently
divided by the whole uploaded image's own dimensions -- an unexamined
assumption, not a decision. `page_reference.py` makes that decision
explicit, recorded, and (for the first time) overridable by a real
human declaration.

## The 6 modes

| Mode | Polygon | Assessable? | How obtained |
|---|---|---|---|
| `auto_detected_page` | Whole image `[[0,0],[1,0],[1,1],[0,1]]` | Yes | `page_frame.py`'s classical-CV heuristic judged `full_page_detected`/`likely_full_page` |
| `user_confirmed_full_frame` | Whole image | Yes | An explicit user assertion ("this image shows the full page edge-to-edge") -- overrides any automatic reading |
| `user_defined_page_corners` | A real, possibly smaller, user-supplied quadrilateral | Yes | 4 user-marked pixel corners, geometrically validated and normalized |
| `cropped_or_content_only` | `None` | No | `page_frame.py` judged the page is not fully visible -- no boundary is manufactured |
| `uncertain` | `None` | No | Evidence was ambiguous -- never guessed toward either side |
| `failed` | `None` | No | The automatic assessment itself errored |

An explicit user declaration (`user_confirmed_full_frame`/
`user_defined_page_corners`) always takes precedence over the automatic
assessment -- a human decision is never silently overridden by a
heuristic. For the current batch/dataset pipeline (no interactive
user), only the 4 automatic-path modes are ever produced; the 2
user-declared modes are a real, tested schema/API for the future
interactive application, exercised by
`tests/test_page_reference.py`/`tests/test_phase2a1_integration.py`
against real corner geometry, not merely typed and left unused.

## What every `PageReference` records

`page_reference_mode`, `page_polygon` (normalized `[[x,y], ...]` or
`None`), `confidence`, `obtained_via` (a provenance string, e.g.
`classical_cv_border_uniformity_v1`/`explicit_user_assertion_v1`/
`user_supplied_corners_v1`), `page_relative_features_assessable`
(boolean -- the single field every gating decision downstream actually
reads), `limitations`, `evidence_id`. Construction itself enforces the
invariant that `page_relative_features_assessable=True` requires a real
`page_polygon`, and that only the 3 assessable modes may ever be marked
assessable (`ValueError` otherwise) -- this cannot be constructed
inconsistently.

## Never infers missing parts of a cropped page

For `cropped_or_content_only`/`uncertain`/`failed`, `page_polygon` is
always `None`. No code path in `page_reference.py` ever extrapolates,
crops, or guesses a boundary for a partially-visible page. Folder
labels are never consulted (the module has no access to them at all --
`resolve_page_reference` takes only the real `page_frame` assessment
and, optionally, a real user declaration).

## User-defined corners: validation

4 pixel corners are required exactly; each must fall within the
image's own bounds; the resulting normalized polygon's area (shoelace
formula) must be at least 1% of the image area, or `InvalidPagePolygonError`
is raised -- rejecting an obvious data-entry mistake (e.g. all 4
corners clicked near the same point) rather than silently accepting a
near-zero-area "page." Given the same 4 corners, the resulting
normalized polygon is bit-for-bit deterministic
(`tests/test_page_reference.py::UserDefinedPageCornersTests::test_same_corners_always_produce_identical_polygon`).

## `page_relative_bounding_box_coverage`

The one concrete feature this model unlocks so far
(`features.py`, `segmentation.page_relative_bounding_box_coverage`):
`image_relative_coverage / page_area_fraction`. For the two whole-image
modes, `page_area_fraction == 1.0`, so this is numerically identical to
the existing `bounding_box_coverage` -- the distinction only becomes
visible once a real, smaller `user_defined_page_corners` polygon
exists, which a real end-to-end test confirms produces a genuinely
different (and correctly larger) coverage value
(`tests/test_phase2a1_integration.py::test_user_defined_smaller_page_polygon_changes_the_coverage_value`).
`None` (never a silently-defaulted number) whenever no page reference
is assessable.

## What consumes `page_reference` (and a bug this phase found and fixed)

`rule_engine_v2.apply_page_frame_gating`/`evaluate_v2_rules`'s
`EN_COMPILED_PLACEMENT_CENTER_029` gating and
`rule_engine_v2.redefine_coverage_full` all gate on
`page_reference.page_relative_features_assessable` -- **not** the raw
automatic `page_frame` status directly. This distinction matters: a
first implementation of `apply_page_frame_gating`/`evaluate_v2_rules`
(carried over unchanged from Phase 2A) still read `page_frame` directly,
which meant an explicit `user_confirmed_full_frame` declaration was
being correctly *resolved* into `page_reference` but never actually
*consulted* by gating -- a user override had no effect. Caught by a
dedicated integration test
(`tests/test_phase2a1_integration.py::test_user_confirmed_full_frame_overrides_automatic_cropped_classification`)
and fixed; `page_frame_judge` (the 9th judge) was updated the same way,
since it would otherwise have flagged a correctly-overridden case as a
violation.

`page_frame.py` itself, and the display-only `page_frame_assessment`
field in `structured_analysis.json`, are unchanged -- `page_reference`
is the layer built on top that turns a status into an actual usable
region.

## What this does NOT do

No trained page/corner detector was built (explicitly out of scope --
`auto_detected_page` reuses Phase 2A's classical-CV heuristic
unchanged). No folder label is ever read. No rule's `allowed_output_level`
changes because of this model -- it only changes *whether* a
page-relative rule is evaluated at all, never whether it is shown more
prominently.

## Phase 2A.2: the parent-facing control and 2 new negative-declaration modes

Section 6's minimal, optional Parent-view control ("Does this image
show the complete sheet of paper?") has 4 choices, all real API calls,
no UI-only stub:

| Choice | `user_page_declaration` | Resolved `page_reference_mode` |
|---|---|---|
| Let the system decide | `None` | automatic (`auto_detected_page` or whatever `page_frame.py` finds) |
| Yes, the complete sheet is visible | `{"mode": "user_confirmed_full_frame"}` | `user_confirmed_full_frame` (unchanged from Phase 2A.1) |
| No, this image is cropped | `{"mode": "user_declared_cropped"}` | `cropped_or_content_only` |
| I am not sure | `{"mode": "user_declared_uncertain"}` | `uncertain` |

The 2 new modes (`user_declared_cropped`/`user_declared_uncertain`)
resolve to the **same** `page_reference_mode` string and outcome
(`page_relative_features_assessable=False`) as the equivalent automatic
reading — the gating behaviour is identical either way. The only
difference is `obtained_via` (`explicit_user_assertion_v1` instead of
`classical_cv_border_uniformity_v1`), which makes the human decision
traceable and distinguishable from an automatic one in the saved case
output. `describe_declaration_choice()` reverses this mapping for
Technical-view display, reconstructing which of the 4 choices was made
purely from the saved `PageReference` fields — no separate declaration
file is written.

`user_page_declaration_from_choice()` is the only function this phase
adds that maps a UI-facing string to the API; `resolve_page_reference()`
itself gained no new required parameters, keeping every existing caller
(the Phase 2A.1 batch pipeline, which always passes `None`) unaffected.

**No corner-editor UI was built** — `user_defined_page_corners` remains
a real, tested API capability with no UI entry point in this prototype,
stated explicitly in Technical View rather than silently absent.
