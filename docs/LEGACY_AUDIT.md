# Legacy audit

## Verified findings

- `pipeline.py` contains useful safety, reporting, and claim-validation ideas but
  passes large unvalidated dictionaries.
- The preserved worktree fix updates the analysis document instead of replacing
  it with the emotion heuristic result.
- The preserved dataset fix uses `class _TD(DrawingDataset, Dataset)`.
- Legacy dataset code uses `val`; v3 uses the physical folder name `valid`.
- Legacy feature extraction mixes page pixels into colour ratios and can select
  negligible colours as dominant.
- Placement logic needs explicit unavailable state on blank/failed segmentation.
- The checked-in `.venv` is machine-specific and unusable in this environment.
- The psychologist PDF is not present in the legacy checkout.

Legacy files remain reference-only and are not overwritten by v3.
