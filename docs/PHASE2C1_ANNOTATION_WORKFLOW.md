# Phase 2C.1 Annotation Workflow

**Status: infrastructure ready, no human annotation performed yet.** This
document is the operational guide for the human annotator/reviewer using
`phase2c_annotation_app.py`. See `PHASE2C1_ANNOTATION_PROVENANCE_AUDIT.md`
for why the 80-image working set is trustworthy, and
`docs/OBJECT_EVIDENCE_ONTOLOGY.md` for the 10 object classes themselves
(unchanged from Phase 2B).

## What exists before you start

- **Real dataset**: read-only, never modified. Every Phase 2C.1 tool opens
  it in read mode only (verified structurally in
  `tests/test_phase2c1_safety.py` and by `git status` staying clean of
  anything under `outputs/`).
- **Private blinded workspace**: `outputs/phase2c1/private_images/` — 80
  opaque `p2b_XXXX.<ext>` files, no emotion label, no original filename, no
  original folder structure. Built once this session via
  `doar.phase2c1.workspace.build_pilot_workspace`, reproducing the exact
  original Phase 2B 80-image selection (seed 2026) — see the provenance
  audit for the verification.
- **Re-identification mapping**: `outputs/phase2c1/private_pilot_mapping.csv`
  — kept in a SEPARATE file from the images themselves, never read by the
  annotation app's UI-facing code except to populate the opaque
  `image_id`/`source_image_group` columns (never the original path or
  emotion label) on saved rows.
- **Seed annotation store**: `outputs/phase2c1/annotation_store.csv` —
  pre-loaded with the 20 already-annotated Phase 2B images
  (`p2b_0000`-`p2b_0019`), migrated from `artifacts/phase2b/
  annotation_manifest.csv` as **provisional, unreviewed** rows (their
  `review_status` stays `unreviewed`; `artifacts/phase2b/
  annotation_manifest.csv` itself is never modified). The remaining 60
  images (`p2b_0020`-`p2b_0079`) have no rows yet.
- All of the above lives under `outputs/`, which is fully gitignored —
  confirmed via `git status` after every stage of this session's work.

## Launching the app

```powershell
cd DOAR-work
.\.venv\Scripts\Activate.ps1
python -m streamlit run phase2c_annotation_app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`).

## Primary annotation mode

1. Enter your annotator ID in the sidebar (required — an annotation can
   never be unattributed; there is no default).
2. The app opens on your first incomplete image (fewer than 10 of the 10
   ontology classes judged by you).
3. For each of the 10 classes, choose a status
   (Present / Absent / Uncertain / Not assessable). **Uncertain and Not
   assessable are never converted to Absent anywhere in this pipeline** —
   pick them whenever genuinely unclear, per
   `docs/OBJECT_EVIDENCE_ONTOLOGY.md`'s own documented ambiguous cases
   (e.g. sun vs. star, a wheel drawn alone).
4. If Present: set the instance count, optionally mark partial/occluded,
   and optionally enter a normalized bounding box (`x,y,w,h`, each 0-1,
   e.g. `0.1,0.2,0.3,0.4`) — invalid input is flagged and simply not saved
   as a bbox rather than blocking the rest of the save.
5. Click **Save** (stays on this image) or **Save & Next** (advances).
   Every click writes the entire store to disk immediately — closing the
   browser tab or terminal never loses saved work; re-launching resumes
   exactly where you left off.
6. Use **Previous / Next / Jump to image # / First unannotated** to
   navigate.

## Review mode

Intended for a **second, independent annotator** (yourself at a different
time, or someone else). To reduce anchoring bias:

1. Switch to "Review" in the sidebar, enter your own (reviewer) annotator
   ID.
2. For each class, record **your own independent judgment first** — the
   original annotator's label is not shown yet.
3. Click "Save my independent judgment" — this creates its own annotation
   row under your annotator ID (a real second annotator, not a fabricated
   one).
4. Only now does "Reveal primary annotator's label" unlock.
5. Compare, then record **Agreement**, **Disagreement (unresolved)**, or
   **Disagreement (resolved)** — this updates the *original* annotator's
   row's review metadata in place (`review_status`, `reviewer_id`,
   `review_timestamp`, `adjudication_status`); it does not create a
   duplicate row.

## Quality metrics (sidebar → "Show quality report")

Computed live from whatever is in the store:

- Per-class support (present/absent/uncertain/not_assessable counts).
- Completion rate per annotator.
- Review coverage.
- **Percent agreement and Cohen's kappa — only once >= 5 matched judgments
  exist between the same two annotators** (`quality.MIN_PAIRS_FOR_KAPPA`).
  Below that, or with fewer than 2 distinct annotators in the store, the
  report says so explicitly rather than showing a number.
- Instance-count disagreement (among matched Present/Present pairs).
- Unresolved disagreements (adjudication_status ==
  `disagreement_unresolved`).

**No inter-annotator statistic is ever fabricated.** With a single
annotator (the current real state — only the original Phase 2B session
annotator exists in the seed store), the report says exactly that.

## Export

Sidebar → "Export CSV + JSON" writes
`outputs/phase2c1/exports/phase2c1_annotations.{csv,json}` — safe to run at
any point, reflects whatever is saved so far, never fabricates a row for
an unannotated image/class.

## What this phase does NOT do

- No supervised detector training.
- No use of the locked test split for anything.
- No rule activation, no psychological interpretation, no diagnostic
  language anywhere in this pipeline's output (mechanically checked,
  `tests/test_phase2c1_safety.py`).
- No modification of Parent View (`doar_prototype_app.py`) — confirmed
  unmodified, structurally tested.
- No modification of `artifacts/phase2b/annotation_manifest.csv`.

## What you (the human annotator) do next

1. Launch the app as above.
2. Annotate the remaining 60 images (`p2b_0020`-`p2b_0079`) as the primary
   annotator, or review the existing 20 provisional Phase 2B judgments in
   Review mode as a second, independent annotator.
3. Once real dual-annotator data exists, re-open the sidebar's quality
   report for a real (not single-annotator) agreement statistic.
4. Decide, from real per-class support counts once annotation is further
   along, which classes (if any) are ready for a Phase 2C.2 detector
   experiment — this phase does not make that call for you.
