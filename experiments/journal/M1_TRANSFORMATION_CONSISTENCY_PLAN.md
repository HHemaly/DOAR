# M1 — Transformation consistency of the explicit visual measurements (analysis plan)

Status: **exploratory, added after B4** (no manual annotation; see PROTOCOL_DEVIATIONS.md D6).
This file fixes every rule *before* the experiment is run on the real data. The rules are
implemented in `experiments/journal/scripts/m1_transformation_consistency.py`; any later change
must be logged in PROTOCOL_DEVIATIONS.md.

## Question
Do the archived explicit measurements (E2-A0, 49 numeric columns) give consistent values on two
digital versions of the **same** drawing, after allowing for the geometric relation between them?

What it is: a **transformation-consistency (repeatability) analysis** on digital copies.
What it is not: real-world test-retest (no re-photographing) and not accuracy (a stable but wrong
measurement passes). No psychological meaning is tested.

## Data
* Images: T1_v2 train + valid only (the locked test split is not read).
* Pairs: images sharing a Roboflow filename stem (`rf_stem`; pair review: 40/40 same drawing, 0/20
  controls). Within a stem group the reference is the first image id in sorted order; it is paired
  with every other member. The stem group is the resampling unit (dependence between pairs of
  one drawing is respected).
* Extractor: archived `extract_features` (`_frozen_e2_features.py`), unchanged, A0 mask inside.

## Pair registration (fixed rules)
* The relation between the two images is estimated among the 8 dihedral transforms
  (identity, flips, 90/180/270 rotations, transposes): both images to grayscale, the transformed
  reference and the copy resized to 64 x 64, z-scored, Pearson correlation; best transform kept.
* `identical`: same shape and identical pixels → excluded (trivial).
* `unregistered`: best correlation < 0.95 (crop, shear, arbitrary-angle rotation, heavy noise)
  → counted, not analysed per feature.
* `geometric`: correlation ≥ 0.95 and photometrically equal: after alignment and resizing,
  every RGB channel mean differs by ≤ 2.0 grey levels and every channel std ratio is in
  [0.97, 1.03]. **Primary set.**
* `geometric+photometric`: correlation ≥ 0.95, photometric condition fails. **Secondary set.**

## Feature roles (fixed in code: `FEATURE_ROLES`)
* `excluded` (canvas metadata, resolution-dependent): canvas_width_px, canvas_height_px,
  canvas_aspect_ratio, foreground_pixel_count.
* `invariant_geometric`: value must not change (foreground_fraction, bbox area/fill, centroid
  distance from centre, 4x4 occupied grid, component count/fractions, thickness proxies).
* `equivariant_geometric`: value must change in a known way; the expected value is computed
  from the reference by the exact coordinate mapping of the registered transform
  (x→1−x, y→1−y, x↔y; e.g. left↔right margins, quadrant permutation, centroid_x→1−centroid_x,
  vertical↔horizontal symmetry under a transpose).
* `invariant_photometric`: colour/intensity/edge features; must not change in the primary set;
  **not comparable** in the secondary set (the image colours were changed).

## Metrics (per feature, per set)
* Error e = |observed(copy) − expected(from reference)|; normalised error e / IQR of the feature
  over the reference images of the analysed pairs (std if IQR = 0; 1 if both 0).
* A pair is **consistent** if normalised error ≤ τ = 0.10 (disagreement under one tenth of the
  between-drawing spread). One value missing (NaN) and the other present = inconsistent; both
  missing = excluded and counted.
* Consistency rate: mean over stem groups of the within-group share of consistent pairs.
  95% CI: percentile bootstrap over stem groups, 2000 resamples, seed 20261106.
* Verdict: **CONSISTENT** if CI low ≥ 0.90; **INCONSISTENT** if CI high < 0.90; otherwise
  **INCONCLUSIVE**. Fewer than 10 stem groups → `TOO_FEW`.
* Also reported (descriptive): median absolute and normalised error, 90th percentile, ICC(2,1) of
  (expected, observed), and the consistency rate per transform type when ≥ 10 groups.
* Mask agreement: IoU between the A0 mask of the copy and the aligned (nearest-neighbour
  resized) A0 mask of the reference; group-weighted median and share ≥ 0.90, with CI.
  Spearman correlation between pair mask IoU and the pair's mean normalised error over the
  invariant geometric features (descriptive).

## Interpretation rules (fixed)
* CONSISTENT = numerically repeatable under digital transforms only; not validated, not accurate.
* INCONSISTENT = the measurement changes when only the digital copy changes: it cannot be
  trusted as a property of the drawing, whatever its accuracy.
* No threshold, tolerance or set definition is changed after the outputs are seen.
