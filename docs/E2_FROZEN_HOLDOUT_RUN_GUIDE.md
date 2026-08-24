# E2-A0-R1-V2 Frozen Unseen Holdout — Run Guide

This guide explains how to run `colab/paper/E2_A0_R1_V2_FROZEN_UNSEEN_HOLDOUT.ipynb` from a
completely fresh Google Colab runtime. The notebook re-assembles the frozen
"E2-A0-R1-V2 — FROZEN UNSEEN QC HOLDOUT" experiment for the conference paper. Every
function body, threshold, path, and rubric label it uses is extracted verbatim from the
archived notebook at `colab/archive/DOAR_21_8_26_V1.ipynb` (cell indices 22, 23, and 24) — see
each cell's provenance banner for the exact source.

## 1. What must exist in Google Drive before you start

Mount the Google account that holds the DOAR Masters Drive folder. This notebook does **not**
create any of the frozen source artifacts below — it only reads/verifies them. If any are
missing, CELL 4 will report `MISSING` and the notebook cannot proceed.

Required, already-frozen inputs (exact paths, verbatim from the archive):

- `/content/drive/MyDrive/Masters/Datasets/Combined_Drawing` — the drawing image dataset
  (`DATASET_ROOT`).
- `/content/drive/MyDrive/Masters/DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES/tables/E2A_T1_INTERPRETABLE_FEATURES.csv`
  — the frozen E2-A0 feature table (`SOURCE_FEATURE_FILE`). Its SHA256 is checked in CELL 4
  against the archive-declared value; if it does not match, the notebook stops.
- `/content/drive/MyDrive/Masters/DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES/E2A0_R1_PILOT_V1/tables/E2A0_R1_PILOT_MASK_RESULTS.csv`
  — the 84-image R1-V1 development-pilot results (`V1_RESULT_FILE`); part of the 84-image
  development set excluded from the holdout.
- `/content/drive/MyDrive/Masters/DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES/E2A0_R1_V2_PILOT/tables/E2A0_R1_V2_PILOT_RESULTS.csv`
  — the 84-image R1-V2 development-pilot results (`V2_DEV_RESULT_FILE`); the other half of the
  84-image development set.

The notebook creates its own output directory tree under
`/content/drive/MyDrive/Masters/DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES/E2A0_R1_V2_UNSEEN_HOLDOUT/`
(`HOLDOUT_ROOT`) the first time it runs — you do not need to create this yourself.

## 2. Cell execution order

Run cells top to bottom, in order, in a single fresh runtime. Do not skip cells except where
noted in step 3 below.

| # | Cell | What it does |
|---|------|---------------|
| — | Title (markdown) | Notebook overview |
| 1 | Mount Drive + imports | Mounts Drive, imports all libraries the frozen functions need |
| 2 | Repository/version provenance | Records git commit, filenames, dates, installed library versions |
| 3 | Frozen A0/V2/QC/helper definitions | Defines `build_a0_mask`, `build_v2_mask`, `measurement_policy`, etc. — verbatim from the archive |
| 4 | Frozen artifact/path verification | Checks all required Drive paths exist + SHA256 of the frozen E2-A0 table |
| 5 | Holdout integrity checks | Deterministically rebuilds the 80-image holdout (40 A0_FLAGGED + 40 A0_OK) and asserts N=80, 40/40 split, zero overlap with the 84-image development set, Train/Validation-only, no Locked Test access, all paths resolve |
| 6 | Freeze manifest | Hashes the running implementation, holdout composition, and records library versions |
| 7 | Execute A0 + frozen V2 | Runs both mask extractors on all 80 holdout images |
| 8 | Save per-image results | Writes the results CSV + QC/support/measurement-policy summary tables |
| 9 | Internal visual QC contact sheet | Generates ORIGINAL \| A0 MASK \| V2 MASK PDF/PNG contact sheets (one set for A0_FLAGGED, one for A0_OK) |
| 10 | Blank human visual-review template | Writes `E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv` with empty review columns |
| 11 | Visual-review statistics (conditional) | **Only computes anything once the human review below is complete** |
| 12 | Numerical acceptance rule (markdown) | States plainly that no numeric acceptance rule exists in the archive for this experiment, and quotes what does exist |
| 13 | Deterministic publication example selection | Picks Case A / Case B by sorted `image_id` — **depends on the completed human review** |
| 14 | Paper-ready output generation | Writes `paper_results_summary.md` with everything computed so far |

## 3. The manual step required before CELL 11 can run

CELL 10 writes a blank CSV at:

```
.../E2A0_R1_V2_UNSEEN_HOLDOUT/tables/E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv
```

Before running CELL 11, you must:

1. Open this CSV (e.g. download it, open in Sheets/Excel, or edit in Colab).
2. For every one of the 80 rows, look at the corresponding case in the CELL 9 contact
   sheets (ORIGINAL \| A0 \| V2) and fill in:
   - `visual_mask_quality` — one of: `GOOD`, `PARTIAL`, `FAILED`, `OVERSEGMENTED`,
     `UNDERSEGMENTED`, `APPROPRIATELY_AMBIGUOUS`, `UNCERTAIN` (this is the frozen rubric
     found verbatim in the archive — do not use any other label).
   - `support_classification_quality` — your assessment of whether V2's inferred
     support/background regime looks correct for that image (free text is acceptable here;
     the archive does not define a closed vocabulary for this column).
   - `visual_notes` — free-text notes (optional).
   - `review_complete` — set to `True` once you have reviewed that row.
3. Save/upload the completed CSV back to the same path in Drive, overwriting the blank
   template CELL 10 wrote.
4. Re-run CELL 11 (and CELL 13, CELL 14). CELL 11 explicitly checks that
   `review_complete` is `True` for all 80 rows before computing any statistics; if it is not,
   it prints a message and computes nothing rather than producing partial numbers.

## 4. Expected output files

All outputs land under `HOLDOUT_ROOT` =
`/content/drive/MyDrive/Masters/DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES/E2A0_R1_V2_UNSEEN_HOLDOUT/`:

- `tables/E2A0_R1_V2_UNSEEN_HOLDOUT_MANIFEST.csv` — the 80-image holdout manifest
- `tables/E2A0_R1_V2_UNSEEN_HOLDOUT_RESULTS.csv` — per-image A0/V2 results
- `tables/E2A0_R1_V2_HOLDOUT_QC_SUMMARY.csv`, `..._SUPPORT_SUMMARY.csv`,
  `..._MEASUREMENT_POLICY_SUMMARY.csv` — summary tables
- `tables/E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv` — blank, then human-completed, review file
- `figures/flagged_contact_sheets/` and `figures/control_contact_sheets/` — PDF + PNG
  ORIGINAL \| A0 \| V2 contact sheets
- `PAPER_FREEZE_MANIFEST.json` — implementation/holdout hashes and library versions
- `PAPER_VISUAL_REVIEW_STATS.json` — recovery/retention/oversegmentation/review-required/
  unreliable rates with Wilson 95% CIs (only after human review is complete)
- `PAPER_PUBLICATION_EXAMPLES.json` — the deterministically selected Case A / Case B image ids
  (only after human review is complete)
- `paper_results_summary.md` — the final paper-ready summary

## 5. What this notebook deliberately does NOT do

No model training. No E2-A1 experiment. No evaluation against the Locked Test set (Test
images are never materialized or accessed — only their count is referenced, exactly as in the
archive). No parameter tuning or threshold search — the A0 and V2 thresholds are frozen and
copied verbatim. No segmentation method beyond the frozen A0 and V2 functions extracted from
`colab/archive/DOAR_21_8_26_V1.ipynb`.

## 6. Known limitation carried over from the archive

The archive does not document a numeric acceptance rule for this specific experiment (the
80-image frozen holdout) — only a qualitative rule ("no catastrophic systematic failure").
See CELL 12 in the notebook for the full explanation and the exact archive quotes. Do not
apply the 85%/90%/5% numeric gate found elsewhere in the archive (archive cell 22) to this
experiment's results — that gate was declared for a different, earlier, superseded pilot
(R1-V1 on the 84-image development set), not for this frozen unseen holdout.
