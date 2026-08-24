# DOAR Objective Feature Explorer -- Supervisor Demo Guide

Standalone demo of DOAR's objective/formal drawing-feature layer
(`src/doar/formal_features.py`, Phase G0 / Appendix G), proving it works
independently of the rule engine, Ask DOAR, Gemini, and evidence
aggregation. Script: `scripts/objective_feature_explorer.py`.

## Launch

```
.venv\Scripts\python.exe -m streamlit run scripts\objective_feature_explorer.py
```

## Selected examples

All three basenames were checked by exact-basename match against
`experiments/E1_visual_representation/raw/e1a_test_paths_LOCKED_DO_NOT_USE.csv`
(512 unique locked basenames) and confirmed **not** part of the Locked
Test split. All three are reused from `SUPERVISOR_DEMO_CASES` in
`doar_prototype_app.py`, already audited in a prior session for the same
Locked Test exclusion; selection below was made purely on visual/technical
suitability (plausible feature values, no obvious segmentation failure),
never by psychological label.

| Slot | Case ID | Image | Why chosen |
|---|---|---|---|
| Example Drawing 1 | `e2e_check_1786239075` | `outputs/prototype_cases/e2e_check_1786239075/p2b_0000.jpg` | Simple line/pencil drawing -- near-monochrome (`colour.diversity` = 1), low chromatic coverage, sparse linework: the clearest "line drawing" of the three candidates. |
| Example Drawing 2 | `a111_1787479358` | `outputs/prototype_cases/a111_1787479358/a111.jpeg` | Colourful drawing -- highest chromatic coverage (0.96) and colour diversity (4/5 bins) of the three candidates; a saturated, multi-colour painted piece. |
| Example Drawing 3 | `h38_1786305027` | `outputs/prototype_cases/h38_1786305027/h38.jpg` | Complex/detailed drawing -- many distinct drawn objects (vehicle, two figures, sun, clouds, five butterflies, a traffic light); highest direction-cluster count (18/18) and orientation entropy (0.99) of the three, plus the highest detail density. |

## Feature coverage

All **22/22** current formal features (`src/doar/formal_features.py::compute_formal_features`)
rendered with valid, non-missing values for all three examples -- verified
both by direct script probe and by the AppTest suite's dataframe-shape
assertion (`tests/test_objective_feature_explorer.py`). Nothing was
missing/unavailable for any of the three; the explorer's
"not measurable for this image" path exists in the code for
`FeatureValue.missing` but was not naturally triggered by these examples
(exercised separately with a synthetic value in
`test_missing_values_would_render_gracefully`).

## Best measurements to point out, per drawing

**Example Drawing 1 -- simple line/pencil drawing**
- `line.mean_width` = 4.46 px -- thin, consistent linework.
- `colour.diversity` = 1, `colour.chromatic_coverage` = 0.27 -- confirms the near-monochrome, pencil/line character.
- `line.continuity` = 0.14 -- the largest connected mark covers only 14% of the drawn area; many separate small strokes rather than one dominant shape.
- `line.fragmentation_rate` = 0.33 -- moderate fragmentation, consistent with sketch-like linework.
- `line.orientation_entropy` = 0.75 -- directions are fairly spread, not a single dominant stroke direction.
- `symmetry.bilateral` = 0.82 -- a fairly left-right symmetric composition.
- **QC limitation**: `segmentation.status` = "uncertain" (confidence 0.51), `foreground_coverage` only 0.105 (10.5% of the page has any mark) -- the Measurement Quality panel shows a warning; treat values with more caution than a confidently segmented drawing.

**Example Drawing 2 -- colourful drawing**
- `colour.chromatic_coverage` = 0.96 -- almost all marked pixels are saturated colour, not black/grey/white.
- `colour.diversity` = 4/5, `colour.saturation` = 0.61 -- vivid, varied palette.
- `line.direction_cluster_count` = 18/18, `line.orientation_entropy` = 0.98 -- painterly texture with directions spread across essentially the whole range.
- `line.continuity` = 0.53 -- one dominant painted mass (the face) rather than many separate marks.
- `symmetry.bilateral` = 0.59 -- lower symmetry, consistent with an off-centre composition.
- **QC status**: `segmentation.status` = "verified" (confidence 0.67) -- the panel shows a green confirmation, no warning.

**Example Drawing 3 -- complex/detailed drawing**
- `line.direction_cluster_count` = 18/18, `line.orientation_entropy` = 0.99 -- the most direction-varied of the three, consistent with many separately-oriented objects.
- `scene.detail_density` = 0.093 -- the highest edge-detail-per-foreground-area of the three examples.
- `stroke.junction_corner_density_proxy` = 0.90 -- many corners/junctions, consistent with several distinct small shapes (butterflies, wheels, window).
- `colour.chromatic_coverage` = 0.64, `colour.diversity` = 2/5 -- colourful but less saturated overall than Example 2.
- `symmetry.bilateral` = 0.64 -- moderate symmetry.
- **QC limitation**: `segmentation.status` = "uncertain" (confidence 0.56) -- warning shown; not a catastrophic failure, but flagged for caution same as Example 1.

## ~2-minute spoken explanation

"This layer is intentionally deterministic and interpretable. Instead of
asking an AI whether the drawing looks chaotic or symmetric, I quantify
properties such as line continuity, orientation entropy, colour diversity
and bilateral symmetry. These measurements remain objective until their
psychological relevance is separately validated.

Let me show you three drawings that are visually very different from each
other. The first is a simple, mostly monochrome line drawing -- and the
numbers agree: colour diversity is 1 out of 5 possible bins, chromatic
coverage is low, and the largest connected stroke covers only 14% of the
marked area, meaning it's built from many small, separate strokes rather
than one continuous shape. The second is a colourful, saturated painted
piece -- chromatic coverage jumps to 96%, colour diversity to 4 out of 5,
and continuity rises to 53%, because most of the colour sits in one
dominant painted mass. The third is the busiest of the three -- a scene
with a car, two people, a sun, clouds, five butterflies and a traffic
light -- and that complexity shows up directly in the numbers: direction-
cluster count and orientation entropy both hit their maximum, meaning
strokes point in almost every direction rather than one or two dominant
ones, and detail density is the highest of the three examples.

Every one of these twenty-two numbers comes from plain computer-vision
statistics -- distance transforms, Sobel gradients, connected-component
analysis, HSV colour thresholds -- the same deterministic code path for
every image, every time, with no learned weights and no network call.
Where DOAR's own segmentation step isn't confident about the foreground --
which happens for two of these three drawings -- the interface says so
explicitly, with a measurement-quality warning, rather than presenting
uncertain numbers as if they were reliable.

None of this claims to know what a child felt or intended. It doesn't
measure physical pencil pressure, and it never touches the rule engine or
any psychological interpretation. It simply describes, in a repeatable
way, what is visibly true about the drawing -- so that any psychological
claim built on top of it later can be checked against a stable, objective
foundation rather than an AI's impression."

## Testing

Test file: `tests/test_objective_feature_explorer.py` -- 9 tests, all
passing, run via Streamlit's `AppTest` harness (real widget interaction,
same pattern as `tests/test_supervisor_demo_smoke.py`):

```
.venv\Scripts\python.exe -m pytest tests\test_objective_feature_explorer.py -v
```

Covers: app opens and renders for all 3 examples; all 22 current formal
features render with valid values (dataframe-shape assertion); missing
values handled gracefully (direct unit check of the rendering helper);
QC warnings render correctly (Examples 1 and 3 warn, Example 2
confirms); a structural import-guard (source-level AST check + runtime
`sys.modules` check) confirms zero rule-engine/Ask-DOAR/Gemini imports;
zero tracebacks anywhere on any page.
