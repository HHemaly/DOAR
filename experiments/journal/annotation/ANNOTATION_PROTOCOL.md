# Reference-mask annotation protocol (C2/C3)

**Purpose.** Human reference masks against which every foreground method is
measured. The annotator never sees class labels, file names, or any method's
mask.

## Who

- **Primary annotator**: all ~120 tasks.
- **Second annotator** (supervisor or a colleague): the 40 tasks listed in
  `double_annotation_tasks.csv`, independently, without seeing the primary
  annotator's masks.

## What counts as a drawing mark

Mark every pixel that the child put on the support: pencil, crayon, marker,
paint, ink, including faint and light-coloured strokes, filled areas and
written text.

Do **not** mark: paper texture, shadows, folds, page edges, the table or
background outside the paper, scanner noise, Roboflow-style added noise
(isolated black/white specks), black wedges from rotation, watermarks.

## Steps per image (about 2–4 minutes)

1. **Assessability.** Choose one:
   - `ASSESSABLE` — a support (paper) and marks on it can be told apart.
   - `NOT_ASSESSABLE_FULL_CANVAS` — paint or colour covers (nearly) the whole page.
   - `NOT_ASSESSABLE_PHOTO_SCENE` — a photograph where other objects dominate.
   - `NOT_ASSESSABLE_NOT_A_CHILD_DRAWING` — stock illustration, clip art, adult artwork.
   - `NOT_ASSESSABLE_BLANK` — no visible marks.
   - `NOT_ASSESSABLE_OTHER` — explain in the note.
2. For `ASSESSABLE` images, click 2–5 points on clean **paper** (different
   lighting areas if the page is shaded).
3. Move the threshold until the overlay covers the marks with as little paper
   as possible.
4. Brush green over missed marks; brush red over anything that is not a mark.
   Zoom the browser if strokes are thin.
5. Save. You can reopen and re-save a task; earlier versions are kept.

## Running the tool

```
pip install -e ".[cv,ml,ui,phase2c6]" scikit-image
streamlit run experiments/journal/annotation/reference_mask_app.py -- --batch <path to the C2 batch folder>
```

Enter your initials in the sidebar. Masks and records are written into the
batch folder; upload the whole folder back to Google Drive when finished.

## Known limitation (reported in the paper)

The starting proposal comes from the annotator's own paper samples and
threshold, which resembles a Lab-distance method (like A0) and can anchor
the reference toward such methods. The second annotator, the per-pixel
brush corrections and the boundary metric are the checks on this.
