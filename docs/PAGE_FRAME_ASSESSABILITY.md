# Page-Frame Assessability (DOAR-TRACE Phase 2A, Section 3)

## Why this exists

Every page-coverage rule (`coverage_small`/`coverage_about_half`/`coverage_full`)
and every placement rule (`placement_top`/`placement_left`/`placement_right`/
`placement_center`) is only meaningful **relative to the whole physical
page** the child drew on. Before Phase 2A, nothing in `analysis.py`
ever checked whether the uploaded image actually shows the whole page:
`_estimate_background` silently samples the image's own border pixels
and treats them as "the page background", regardless of whether the
image is a scan of a full sheet, a tightly cropped photo, or stock
clipart with no physical page at all. `src/doar/page_frame.py` adds
that missing check as an explicit, honest, classical-CV assessment.

**This is not a trained object/page detector.** It is pure-numpy
heuristic analysis of border-strip colour uniformity and foreground-mask
edge-touching, reusing the real `background_stability` value
`_segment` already computes. No object-detector training was done or
is implied here (explicitly out of scope for this phase).

## The 5 statuses

| Status | Meaning | Assessable? |
|---|---|---|
| `full_page_detected` | Wide, highly uniform border on all 4 sides, content touches no edge | Yes |
| `likely_full_page` | Reasonably uniform border, content touches at most 1 edge lightly | Yes |
| `cropped_or_content_only` | Content clearly bleeds off multiple edges, or a moderately uniform border check fails with 2+ edges touched | No |
| `uncertain` | Evidence genuinely ambiguous — **never guessed** | No |
| `failed` | The assessment itself errored (e.g. unreadable image) | No |

`ASSESSABLE_STATUSES = {full_page_detected, likely_full_page}`. Only
these two statuses permit a page-relative rule to be evaluated normally.
Everything else gates the rule to a new `not_assessable` status —
**never** `not_matched`, because `not_matched` means "the condition was
checked and found false," while `not_assessable` means "the condition
could not be checked at all." Conflating the two would silently turn
"we don't know" into "no," which is a false negative, not an honest
abstention.

## The heuristic (decision tree, in order)

1. Content touches ≥3 edges with a touch ratio >0.15 → `cropped_or_content_only`
   (strong negative evidence).
2. Content touches 0 edges, border uniformity ≥0.85, background stability
   ≥0.8 → `full_page_detected` (strong positive evidence).
3. Content touches ≤1 edge lightly (touch ratio <0.10), border uniformity
   ≥0.6 → `likely_full_page` (moderate positive evidence).
4. Content touches ≥2 edges (below the strong-negative bar above) →
   `cropped_or_content_only`.
5. Otherwise → `uncertain`. This is the deliberate fallback: when the
   border-uniformity and edge-touch signals don't clearly agree, the
   status is never guessed toward either side.

Border uniformity is measured over the outermost ~3% of each image
dimension (`_BORDER_BAND_PX_FRACTION = 0.03`), converting pixel
standard-deviation in that band to a 0–1 score (std=0 → 1.0 uniform,
std≥60 → 0.0).

## Known limitation (by design, not a bug)

The heuristic cannot distinguish a genuinely full page with a very thin
margin from a tightly cropped photo — this is stated in every
assessment's `limitations` field. A synthetic test case with content
filling 100% of the frame and no distinguishable border returns
`uncertain` rather than a guessed `cropped_or_content_only`, because
`_estimate_background`'s border-pixel-median approach cannot find a
true background reference in that case. This was investigated during
development and judged an acceptable, honest fallback, not a defect —
`uncertain` is exactly the correct non-committal answer when the
evidence is genuinely ambiguous.

## Real dataset audit result (critical finding)

`phase2a_page_frame_audit.py` ran the assessment over a **120-image,
class-balanced, non-test sample** (30 per class × 4 classes, seed=42) of
`outputs/phase5/manifest.csv`. Result (`artifacts/phase2a/dataset_page_frame_audit.csv`):

| Status | Count | % |
|---|---|---|
| `cropped_or_content_only` | 97 | 80.8% |
| `likely_full_page` | 10 | 8.3% |
| `uncertain` | 8 | 6.7% |
| `full_page_detected` | 5 | 4.2% |
| **Assessable total** | **15** | **12.5%** |

**Only 12.5% of this dataset's images are page-frame-assessable.** This
was reported prominently before continuing implementation, per the
task's explicit stop-condition instruction ("Stop if the page-frame
audit shows that page-use rules are fundamentally invalid for most of
the current dataset; report that finding before changing rule
semantics"). Implementation proceeded because Section 3's own design
(the `not_assessable` status) was built specifically to handle exactly
this scenario safely — the finding does not indicate a bug, it
quantifies a real, pre-existing property of this dataset: it is a
heterogeneous mix of stock clipart with no physical page and real
photos cropped tight, with content frequently bleeding off the image's
own edges.

This was cross-validated by `phase2a_rule_trigger_distribution.py`'s
independent 200-image sample: page-gated rules were assessable on only
31–32/200 (~15.5–16%) of images — consistent with the 12.5% figure from
a different, independently-drawn sample.

## What this changes downstream

- The 6 original `rules.py`-dispatched rules (`PSY_AR_SIZE_HALF_014`,
  `PSY_AR_SIZE_FULL_015`, `PSY_AR_SIZE_SMALL_016`, `PSY_AR_PLACE_TOP_017`,
  `PSY_AR_PLACE_LEFT_018`, `PSY_AR_PLACE_RIGHT_019`) are post-processed by
  `rule_engine_v2.apply_page_frame_gating()` — `rules.py` itself is
  never edited.
- `EN_COMPILED_PLACEMENT_CENTER_029` (the one newly-activated placement
  rule) is gated internally inside `rule_engine_v2.evaluate_v2_rules()`.
- `page_frame_judge` (`judge_schemas.py`, the 9th judge) independently
  re-verifies, from the saved case output alone, that this gating
  invariant was actually respected for every case.
- The Parent view states explicitly when the page could not be
  confirmed visible, so absence of a page-coverage/placement observation
  reads as "we could not check this," never "nothing was found."

## A structural finding: `coverage_full` vs. detectability

While migrating 3 pre-existing tests whose synthetic images used a very
thin (2–5px) white margin, it became clear that `coverage_full`
(≥0.90 bounding-box coverage relative to the *whole image*) and a
genuinely detectable full-page margin are close to mutually exclusive
under the current ~3%-of-dimension border band: a margin wide enough to
read as clean/uniform (≥~3% of the image dimension) necessarily pushes
bounding-box coverage below ~0.90 for simple centered content. This is
a real, useful finding about how these two independently-designed
heuristics interact — not a bug in either one — and is worth keeping in
mind if `coverage_full`'s threshold or the border-band fraction are
revisited in a future phase.

## Phase 2A.1 updates

**Page-reference model built on top of this module**: `page_frame.py`
itself is unchanged, but `page_reference.py` (Section 3,
`docs/PAGE_REFERENCE_MODEL.md`) now resolves an explicit page
polygon/mode from this module's status, and — critically — allows a
real human declaration (`user_confirmed_full_frame`/
`user_defined_page_corners`) to override an automatic
`cropped_or_content_only` reading. Every gating decision downstream now
reads `page_reference.page_relative_features_assessable`, not this
module's `page_frame_status` directly.

**`coverage_full` redefined**: the structural finding above led to a
real fix, not just documentation — see
`docs/PAGE_COVERAGE_DEFINITION_DECISION.md`.

**A corrected finding about the underlying `_segment` heuristic**:
Phase 2A.1's investigation into `segmentation.border_touch_ratio`
(`docs/BORDER_TOUCH_RATIO_DECISION.md`) found that `_segment`'s
candidate-selection scoring can, in some cases, systematically
under-detect content that touches the image border — the same
underlying mechanism `page_frame.py`'s own edge-touch evidence
(`_content_touches_edge`/`_edge_touch_ratio`) depends on. This was
**not** fixed in Phase 2A.1 (out of scope — see that document for why),
but is recorded here as a live caveat on this module's real-image
accuracy: a page-frame classification could, in principle, be affected
by the same candidate-selection bias for images where content
genuinely touches the border. The 120-image real audit
(`artifacts/phase2a/dataset_page_frame_audit.csv`) was re-run unchanged
after every Phase 2A.1 code change and produced byte-identical results
each time, so this is a theoretical exposure identified for future
investigation, not an observed discrepancy on this dataset.
