# DOAR — Human Annotation Protocol for the 15-Image Development Set

For **TWO independent annotators**, each working alone (no discussion until
both passes are complete for all 15 images). Applies only to
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

## Pass 1 — 60-second salient items (per image)

1. Look at the image for **60 seconds only** (use a timer).
2. Write down every **salient** (immediately obvious, would-notice-at-a-
   glance) visual item you saw. No minimum or maximum count — write what
   you actually noticed.
3. For each item, record: a short label (2-4 words, plain description,
   e.g. "yellow car", "traffic light", "two people"), and an approximate
   location in the drawing (e.g. "top-left", "center", "bottom edge") —
   no need for exact coordinates in Pass 1.
4. Stop at 60 seconds even if you think you might notice more — that is
   the point of this pass (salience, not completeness).

## Pass 2 — exhaustive visible items (per image, separate sitting)

1. Look at the image for as long as needed.
2. List **every** distinct item you can identify, however small or
   incidental (individual butterflies, background marks, decorative
   details, text-like marks, scribbles) — not just the salient ones from
   Pass 1.
3. For each item: label, approximate bounding region (a rough box
   description is enough, e.g. "roughly the left third, upper half" — a
   precise pixel bbox is not required from human annotators), and a
   **confidence flag**: `clear` (confident what it is) or `ambiguous`
   (visible mark, but you are not sure what it depicts).
4. Re-list every Pass-1 item too (so Pass 2 is a complete, standalone
   record, not a diff against Pass 1).

## Template (per image, per annotator, per pass)

```
Annotator: A1 | A2
Image ID: <from DEVELOPMENT_SET_15.json, e.g. "h38">
Pass: 1 | 2
Timestamp started:
Timestamp finished:

Items:
  1. label: <plain description>
     location: <approximate region>
     confidence: clear | ambiguous   (Pass 2 only)
  2. ...
```

Save as one file per `(annotator, image_id, pass)` —
`annotations/<annotator_id>/<image_id>_pass<1|2>.json` (or `.md`, either is
fine, kept plain-text/structured so it is easy to diff later) — schema
suggestion:

```json
{
  "annotator_id": "A1",
  "image_id": "h38",
  "pass": 1,
  "items": [
    {"label": "sun", "location": "top-left", "confidence": "clear"},
    {"label": "yellow car", "location": "center-left", "confidence": "clear"}
  ]
}
```

## What annotators do NOT need to do

- No bounding-box pixel coordinates (approximate region text is enough).
- No agreement/consensus step during annotation — inter-annotator
  comparison happens afterward, by whoever runs the benchmark comparison,
  not by the annotators themselves.
- No rating of "how psychologically meaningful" anything is — out of
  scope for this protocol entirely (see the non-negotiable rule above).

## How this will be used later (not part of the annotator's task)

Pass-1 items become the **salient recall** ground truth (Section 7's
visual metrics); Pass-2 items become the **exhaustive precision/recall**
ground truth. Two annotators exist so a simple agreement measure
(e.g. item-label overlap between A1 and A2) can flag genuinely ambiguous
images before they are used for any metric — not implemented in this
phase (`BENCHMARK_SCHEMA.md` defines the metric, the comparison script
itself is future work, out of scope per this phase's instructions).
