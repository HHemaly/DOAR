"""Click-through review tools that run INSIDE Colab (no CSV editing by hand).

pair_review(...)     A5: blinded image-pair review -> writes review_<stratum>.csv
holdout_review(...)  C1: frozen-V2 holdout mask review -> fills the existing
                     E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv

Every click is saved to Drive immediately (atomic replace), so closing the
browser or a Colab disconnect loses nothing; re-running the cell resumes at
the first unanswered item. The pair review never shows file names, splits or
which pairs are controls.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import numpy as np

PAIR_VERDICTS = ("SAME_DRAWING", "DIFFERENT_DRAWING", "UNSURE")
PAIR_TRANSFORMS = ("none", "rotation", "flip", "crop_or_zoom", "noise", "brightness_or_colour", "multiple")
MASK_RUBRIC = ("GOOD", "PARTIAL", "FAILED", "OVERSEGMENTED", "UNDERSEGMENTED", "APPROPRIATELY_AMBIGUOUS",
               "UNCERTAIN")
SUPPORT_RUBRIC = ("CORRECT", "INCORRECT", "UNCERTAIN")


# ---------------------------------------------------------------- pure helpers

def read_rows(path: str | Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write_rows(path: str | Path, fields: list[str], rows: list[dict]) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def first_open(rows: list[dict], done) -> int:
    for i, r in enumerate(rows):
        if not done(r):
            return i
    return len(rows)


def pair_done(r: dict) -> bool:
    return r.get("verdict") in PAIR_VERDICTS


def holdout_done(r: dict) -> bool:
    return str(r.get("review_complete", "")).strip().lower() == "true" and r.get("visual_mask_quality") in MASK_RUBRIC


def record_pair(rows: list[dict], i: int, verdict: str, transform: str, reviewer: str, notes: str = "") -> None:
    if verdict not in PAIR_VERDICTS:
        raise ValueError(verdict)
    rows[i].update({"verdict": verdict,
                    "transform_seen": transform if verdict == "SAME_DRAWING" else "not_applicable",
                    "reviewer": reviewer, "notes": notes})


def record_holdout(rows: list[dict], i: int, quality: str, support: str, notes: str = "") -> None:
    if quality not in MASK_RUBRIC:
        raise ValueError(quality)
    rows[i].update({"visual_mask_quality": quality, "support_classification_quality": support,
                    "visual_notes": notes, "review_complete": "True"})


def _png(img: np.ndarray, max_side: int = 520) -> bytes:
    from PIL import Image
    im = Image.fromarray(img)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _resolve(dataset_root: Path, rel: str) -> Path:
    rel = rel.replace("\\", "/")
    p = dataset_root / rel
    if p.exists():
        return p
    return dataset_root.joinpath(*rel.split("/")[-3:])


# ---------------------------------------------------------------- widgets

def pair_review(review_dir: str | Path, dataset_root: str | Path, stratum: str, reviewer: str):
    """Show blinded pairs of one stratum (rf_stem, name_key, dihedral, embedding)."""
    import ipywidgets as W
    from IPython.display import display

    from .foreground import load_rgb
    review_dir, dataset_root = Path(review_dir), Path(dataset_root)
    if not reviewer.strip():
        raise ValueError("set REVIEWER to your initials first")
    key_fields, key_rows = read_rows(review_dir / f"pairs_{stratum}.csv")   # paths only, type never shown
    form_path = review_dir / f"review_{stratum}.csv"
    fields, rows = read_rows(form_path)
    paths = {k["pair_id"]: (k["path_a"], k["path_b"]) for k in key_rows}
    state = {"i": first_open(rows, pair_done)}

    title = W.HTML()
    left, right = W.Image(format="png"), W.Image(format="png")
    transform = W.Dropdown(options=PAIR_TRANSFORMS, value="none", description="If same:")
    notes = W.Text(placeholder="optional note", layout=W.Layout(width="420px"))
    buttons = [W.Button(description=v.replace("_", " ").title(), button_style=s)
               for v, s in zip(PAIR_VERDICTS, ("success", "danger", "warning"))]
    back = W.Button(description="◀ Previous")
    msg = W.HTML()

    def show():
        i = state["i"]
        done = sum(pair_done(r) for r in rows)
        if i >= len(rows):
            title.value = f"<h3>{stratum}: all {len(rows)} pairs reviewed ✓ — run the score cell</h3>"
            left.value = right.value = b""
            return
        pid = rows[i]["pair_id"]
        a, b = paths[pid]
        title.value = (f"<h3>{stratum} — pair {pid} ({i + 1} of {len(rows)}; {done} answered)</h3>"
                       "<p>Same child's drawing (possibly rotated, flipped, cropped, noisy)?</p>")
        left.value = _png(load_rgb(_resolve(dataset_root, a)))
        right.value = _png(load_rgb(_resolve(dataset_root, b)))
        prev = rows[i].get("verdict")
        msg.value = f"<i>current answer: {prev}</i>" if prev else ""
        notes.value = rows[i].get("notes", "")

    def answer(verdict):
        def _(_btn):
            record_pair(rows, state["i"], verdict, transform.value, reviewer, notes.value)
            write_rows(form_path, fields, rows)
            transform.value = "none"
            state["i"] = first_open(rows, pair_done) if state["i"] + 1 >= len(rows) else state["i"] + 1
            show()
        return _

    for v, btn in zip(PAIR_VERDICTS, buttons):
        btn.on_click(answer(v))
    back.on_click(lambda _b: (state.update(i=max(0, state["i"] - 1)), show()))
    show()
    display(W.VBox([title, W.HBox([left, right]), W.HBox([transform, notes]),
                    W.HBox([*buttons, back]), msg]))


def holdout_review(review_csv: str | Path, dataset_root: str | Path):
    """Show original | A0 mask | frozen V2 mask for each of the 80 holdout images."""
    import ipywidgets as W
    from IPython.display import display

    from . import foreground as fg
    review_csv, dataset_root = Path(review_csv), Path(dataset_root)
    fields, rows = read_rows(review_csv)
    state = {"i": first_open(rows, holdout_done)}
    title = W.HTML()
    ims = [W.Image(format="png") for _ in range(3)]
    caps = [W.HTML(value=c) for c in ("<b>original</b>", "<b>A0 mask</b> (white = marks)",
                                      "<b>frozen V2 mask</b> (white = marks)")]
    support = W.Dropdown(options=SUPPORT_RUBRIC, value="UNCERTAIN", description="Support regime:")
    notes = W.Text(placeholder="optional note", layout=W.Layout(width="420px"))
    buttons = [W.Button(description=q.replace("_", " ").title(),
                        button_style="success" if q == "GOOD" else ("danger" if q == "FAILED" else ""))
               for q in MASK_RUBRIC]
    back = W.Button(description="◀ Previous")

    def show():
        i = state["i"]
        done = sum(holdout_done(r) for r in rows)
        if i >= len(rows):
            title.value = f"<h3>All {len(rows)} holdout images reviewed ✓ — run the C1 cell</h3>"
            return
        r = rows[i]
        img = fg.load_rgb(_resolve(dataset_root, r["relative_path"]))
        a0 = fg.a0(img).mask
        v2 = fg.v2(img)
        title.value = (f"<h3>Holdout image {i + 1} of {len(rows)} ({done} answered)</h3>"
                       f"<p>Judge the <b>V2</b> mask. V2 says support = <b>{v2.meta['support_regime']}</b>, "
                       f"confidence {v2.meta['support_confidence']}.</p>")
        for w, x in zip(ims, (img, (a0 * 255).astype(np.uint8), (v2.mask * 255).astype(np.uint8))):
            w.value = _png(x, 360)
        notes.value = r.get("visual_notes", "")

    def answer(q):
        def _(_btn):
            record_holdout(rows, state["i"], q, support.value, notes.value)
            write_rows(review_csv, fields, rows)
            support.value = "UNCERTAIN"
            state["i"] = first_open(rows, holdout_done) if state["i"] + 1 >= len(rows) else state["i"] + 1
            show()
        return _

    for q, b in zip(MASK_RUBRIC, buttons):
        b.on_click(answer(q))
    back.on_click(lambda _b: (state.update(i=max(0, state["i"] - 1)), show()))
    show()
    display(W.VBox([title, W.HBox([W.VBox([c, im]) for c, im in zip(caps, ims)]),
                    W.HBox([support, notes]), W.HBox(buttons[:4]), W.HBox([*buttons[4:], back])]))
