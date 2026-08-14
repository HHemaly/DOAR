# DOAR — Human Annotation Protocol for the 15-Image Development Set

For **TWO independent annotators**, each working alone (no discussion until
both have annotated all 15 images). Applies only to
`DEVELOPMENT_SET_15.json`'s 15 images — **never** the future held-out
benchmark set. Purpose: an independent, human ground truth for what is
*visually present* in each drawing, to compare against the Observer/
Verifier pipeline's output. **Not a psychological or clinical annotation
task.**

## Non-negotiable rule for every annotator

> **Annotate only what is visibly drawn. Never write a personality trait,
> emotion diagnosis, or psychological interpretation of any kind.**
> "Sun", "person", "yellow car", "traffic light" — yes.
> "Happy child", "the child seems anxious", "outgoing personality" — no,
> under any circumstances, even as a passing comment.

If an annotator is unsure whether a mark is a specific object or an
abstract/scribble mark, write **"unclear region"** or **"unidentified
mark"** — never guess a specific object to fill the field, and never guess
a psychological meaning.

## The protocol: one page per drawing, no timer, no separate passes

Use the visual annotation tool (`scripts/annotate_ui.py`) — see
"How to annotate" below. For each of the 15 drawings, the image stays
visible on screen the whole time. There is no time limit and no separate
"quick pass" / "exhaustive pass": in one sitting per drawing, list
**every** item you can identify, however small or incidental, and for
each one flag directly whether it struck you as **salient** (immediately
obvious, would-notice-at-a-glance) or not.

For each visible item, record:

- **label** — a short, plain description (e.g. "yellow car", "traffic
  light", "two people").
- **location** — an approximate region (e.g. "top-left", "center",
  "bottom edge") — no need for exact coordinates or a pixel bounding box.
- **salient** — `yes` if this is something you'd notice at a glance,
  `no` if it's a smaller/incidental detail you only saw on closer look.
- **confidence** — `clear` (confident what it is) or `ambiguous`
  (visible mark, but you are not sure what it depicts).
- **note** — optional free-text, only if something needs a short
  clarifying remark (still subject to the non-negotiable rule above —
  never a psychological remark).

### Example

```
person       | center     | salient=yes | clear
sun          | top-right  | salient=yes | clear
small mark   | bottom     | salient=no  | ambiguous
```

## How to annotate

Launch one instance per annotator (each instance only ever reads/writes
that annotator's own `annotations/<annotator_id>/` subtree — an A1
session can never see A2's annotations, or vice versa):

```
streamlit run scripts/annotate_ui.py -- --annotator A1
streamlit run scripts/annotate_ui.py -- --annotator A2
```

The image is shown large on the left; an editable table on the right
lists items for the current drawing — add a row, fill in the five
fields, check "Salient" if applicable, pick a confidence level. Rows can
be edited or deleted directly in the table. Every edit autosaves; closing
the browser tab loses nothing. Use **Previous** / **Save & Next** to move
between drawings, and check **Mark this drawing complete** when you are
done with a drawing (the sidebar shows which of the 15 are complete, so
you can always see progress and resume exactly where you left off).

## What annotators do NOT need to do

- No timer, no separate quick/exhaustive passes.
- No bounding-box pixel coordinates (approximate region text is enough).
- No agreement/consensus step during annotation — inter-annotator
  comparison happens afterward, by whoever runs the benchmark comparison,
  not by the annotators themselves.
- No rating of "how psychologically meaningful" anything is — out of
  scope for this protocol entirely (see the non-negotiable rule above).
- No console typing — all annotation happens in the on-screen table.

## Storage schema

One file per `(annotator, image_id)` —
`annotations/<annotator_id>/<image_id>.json`:

```json
{
  "annotator_id": "A1",
  "image_id": "h38",
  "items": [
    {"label": "sun", "location": "top-right", "salient": true, "confidence": "clear", "note": ""},
    {"label": "small mark", "location": "bottom", "salient": false, "confidence": "ambiguous", "note": ""}
  ],
  "complete": true
}
```

## How this will be used later (not part of the annotator's task)

Every recorded item becomes part of that annotator's **exhaustive
reference** (`src/doar/benchmark_metrics.py::visual_precision_recall_f1`);
items with `salient=true` become that annotator's **salient reference**
(`salient_recall`). Across both annotators, `SALIENT` = the normalized
union of salient items from either annotator, and `CORE_SALIENT` = the
normalized intersection of salient items from both
(`normalized_salience`, `primary_and_sensitivity_salient_recall`) — never
a merged "ground truth" item list, only a separate agreement/salience
measure. The two annotators' item lists are never combined into a single
truth set for precision/recall purposes; both remain independently
available. An LLM is never used for annotation matching or as ground
truth — all matching reuses the same deterministic label matcher
(`visual_observer._labels_plausibly_match`) already used throughout the
Observer/Verifier pipeline.

## History: retired console/timed workflow

An earlier version of this protocol used a console tool
(`scripts/annotate_development_set.py`, now removed) with a hard
60-second timed Pass 1 and a separate untimed Pass 2. That workflow was
found difficult to use in practice and never produced a valid or complete
annotation set (at most 3 of 15 images touched, one with 0 items
recorded before its timer elapsed). Those old-schema files were
quarantined, not reused, under
`annotations/_quarantined_pre_ui_timer_experiment/` — see the `README.md`
there for exactly what existed. This document now describes only the
current one-page-per-drawing protocol above.
