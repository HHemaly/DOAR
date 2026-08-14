# Quarantined: pre-UI console/timer annotation experiment

These 3 files are what existed under `annotations/A1/` before this phase
replaced the console/timed Pass-1+Pass-2 workflow with the one-page-per-
drawing Streamlit UI. They were produced while validating the OLD
console tool (`scripts/annotate_development_set.py`, retired this
phase), not real annotation work by Annotator A1:

- `h38_pass1.json` -- 0 items, `timed_out: true` (60s elapsed, nothing
  recorded before the deadline).
- `p2b_0001_pass1.json` -- 2 items (`"big house with 2 windows"`,
  `"sun rays"`), `timed_out: true`.
- `p2b_0002_pass1.json` -- 0 items, `elapsed_seconds: 9.7`,
  `timed_out: false` (ended early via blank line).

None of these use the current item schema (`salient`, `confidence`,
`note` per item) or the current one-file-per-image path convention
(`annotations/<annotator_id>/<image_id>.json`); they are Pass-1-only
records under the old `<image_id>_pass<1|2>.json` scheme. They cover
only 3 of the 15 development images, and only Pass 1 -- never a
complete or valid annotation set under any protocol version.

**Quarantined, not deleted, and never loaded by any current tool** --
`scripts/run_development_benchmark.py`'s `load_human_annotations` only
reads `annotations/<annotator_id>/<image_id>.json`, which does not match
this directory's naming, so these can never be silently picked up. Kept
here only as a record of what the old experiment produced. Safe to
delete once no longer of interest.
