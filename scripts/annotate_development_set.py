#!/usr/bin/env python
"""DOAR 15-image development-set annotation tool.

Lightweight, reusable CLI for the two independent human annotators
(`ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md`) -- no manual JSON editing
required. Walks the 15 images in `DEVELOPMENT_SET_15.json`, running
Pass 1 (60-second salient visual inventory) then Pass 2 (exhaustive
visual inventory) for each, and writes each pass to
`annotations/<annotator_id>/<image_id>_pass<1|2>.json` in the exact
schema the protocol defines. Resumable: re-running skips any
image/pass file that already exists, so an annotator can stop and
continue later without redoing finished work.

Two annotators run this tool independently (different --annotator
ids) and their outputs are never combined by this script -- see
`ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md` and
`src/doar/benchmark_metrics.py` for why (an LLM, and no single
annotator's list alone, is ever treated as the sole ground truth).

**Pass-1 timing protocol** (hard-enforced, not just warned about):
the image is never displayed before the annotator explicitly starts the
60-second window (`_prompt_and_wait` runs strictly before
`open_pass1_image_for_duration` in `run_pass1`), and once 60 seconds
have elapsed, `collect_items` stops reading further annotation lines
even if one is already sitting in the input queue -- nothing typed after
the deadline is ever recorded. The image itself is shown in a window
this process owns (Tk, not the OS default viewer via `os.startfile`) so
it can be auto-closed at the same deadline; if Tk/Pillow display is
unavailable in a given environment this falls back to `os.startfile`
(logged, not silent), which the input-side cutoff still protects even
though the fallback window itself cannot be force-closed early.

Usage:
    python scripts/annotate_development_set.py --annotator A1
    python scripts/annotate_development_set.py --annotator A2 --pass 1
    python scripts/annotate_development_set.py --annotator A1 --image h38
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_SET_PATH = ROOT / "DEVELOPMENT_SET_15.json"
ANNOTATIONS_DIR = ROOT / "annotations"

PASS1_DURATION_SECONDS = 60.0

NON_NEGOTIABLE_REMINDER = (
    "REMINDER (non-negotiable, see ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md):\n"
    "  Record ONLY what is visibly drawn -- objects, marks, shapes, people, "
    "animals, scenery.\n"
    "  Do NOT record emotions, personality traits, mood, or any psychological "
    "interpretation.\n"
    "  If unsure what something is, use the label 'unclear region' or "
    "'unidentified mark'."
)


def load_development_set() -> list[dict]:
    data = json.loads(DEV_SET_PATH.read_text(encoding="utf-8"))
    return data["images"]


def output_path(annotator_id: str, image_id: str, pass_number: int) -> Path:
    out_dir = ANNOTATIONS_DIR / annotator_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_id}_pass{pass_number}.json"


def open_image(path: Path) -> None:
    """Untimed fallback: hands the image to the OS default viewer. This
    process gets no handle to that window and cannot close it early --
    only used when the self-owned Tk display (see
    `open_pass1_image_for_duration`) is unavailable, or for the untimed
    Pass 2."""
    try:
        os.startfile(str(path))  # Windows-only; this tool targets this project's Windows environment.
    except Exception as exc:
        print(f"  (could not auto-open image -- please open it manually: {path}  [{exc}])")


def _show_image_window_for_duration(image_path: Path, duration_seconds: float) -> None:
    """Displays `image_path` in a window this process owns and controls,
    auto-destroying it after `duration_seconds` (`root.after`) so the
    60-second cutoff applies to VIEWING the image, not just to typing --
    something `os.startfile`'s externally-owned viewer window can never
    give this process the ability to do."""
    import tkinter as tk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"Pass 1 ({duration_seconds:.0f}s) -- {image_path.name}")
    img = Image.open(image_path)
    img.thumbnail((1000, 1000))
    photo = ImageTk.PhotoImage(img, master=root)
    label = tk.Label(root, image=photo)
    label.pack()
    root.after(int(duration_seconds * 1000), root.destroy)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


def open_pass1_image_for_duration(image_path: Path, duration_seconds: float) -> threading.Thread:
    """Starts the self-owned, auto-closing Tk display in a background
    thread and returns it (daemon, so it never blocks process exit).
    Falls back to the uncontrollable OS default viewer, loudly, if Tk/
    Pillow display raises (e.g. no display attached to this environment)."""
    def _worker() -> None:
        try:
            _show_image_window_for_duration(image_path, duration_seconds)
        except Exception as exc:
            print(f"  (self-closing image window unavailable ({exc}); falling back to the default "
                  f"viewer -- note this process cannot auto-close that window at {duration_seconds:.0f}s, "
                  f"but no further items will be accepted after the deadline regardless.)")
            open_image(image_path)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def _stdin_reader_loop(out_queue: "queue.Queue[str | None]") -> None:
    """The ONE thread that ever reads stdin for the whole run -- every
    prompt (start gates, Pass-1 timed collection, Pass-2 untimed
    collection) pulls from `out_queue` instead of calling input()
    directly, so there is never a race between two readers competing for
    the same terminal input."""
    while True:
        line = sys.stdin.readline()
        if line == "":
            out_queue.put(None)  # EOF
            return
        out_queue.put(line.rstrip("\n"))


def _prompt_and_wait(prompt: str, line_queue: "queue.Queue[str | None]") -> None:
    print(prompt)
    line_queue.get()


def _parse_annotation_line(line: str, pass_number: int) -> dict | None:
    parts = [p.strip() for p in line.split("|")]
    label = parts[0] if parts else ""
    if not label:
        return None
    location = parts[1] if len(parts) > 1 else ""
    entry = {"label": label, "location": location}
    if pass_number == 2:
        confidence = parts[2] if len(parts) > 2 and parts[2] in ("clear", "ambiguous") else "clear"
        entry["confidence"] = confidence
    return entry


def collect_items(
        line_queue: "queue.Queue[str | None]", pass_number: int, *,
        start_time: float | None = None, duration_seconds: float | None = None,
        clock=time.monotonic,
) -> tuple[list[dict], bool]:
    """Pulls parsed annotation items off `line_queue` until either a
    blank line arrives (annotator finished early) or, when
    `duration_seconds` is given, that many seconds have elapsed since
    `start_time` on `clock` -- whichever comes first.

    HARD cutoff: once the deadline has passed, this returns immediately
    without consuming another line from the queue, even if one is
    already waiting there -- nothing typed after the deadline is ever
    recorded ("do not merely warn after >60 seconds"). Pass 2 calls this
    with `duration_seconds=None`, which blocks with no time limit at
    all, identical to the old behaviour.

    Pure w.r.t. its inputs otherwise (no globals, injectable clock) so
    the deadline behaviour is unit-testable without any real waiting or
    a real terminal -- see `tests/test_annotate_development_set.py`.
    """
    items: list[dict] = []
    while True:
        if duration_seconds is not None:
            remaining = duration_seconds - (clock() - start_time)
            if remaining <= 0:
                return items, True
        else:
            remaining = None
        try:
            line = line_queue.get(timeout=remaining)
        except queue.Empty:
            return items, True
        if line is None or line == "":
            return items, False
        entry = _parse_annotation_line(line, pass_number)
        if entry:
            items.append(entry)


def show_pass1_items_for_reference(annotator_id: str, image_id: str) -> None:
    pass1_path = output_path(annotator_id, image_id, 1)
    if not pass1_path.exists():
        return
    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
    print("Your Pass 1 items for this image (Pass 2 must re-list every one of these, plus anything new):")
    for item in pass1["items"]:
        print(f"    - {item['label']}" + (f" ({item['location']})" if item.get("location") else ""))


def run_pass1(
        annotator_id: str, image: dict, image_root: Path, line_queue: "queue.Queue[str | None]", *,
        open_fn=open_pass1_image_for_duration, clock=time.monotonic,
        duration_seconds: float = PASS1_DURATION_SECONDS,
) -> None:
    """`open_fn` is called ONLY after `_prompt_and_wait` returns -- this
    ordering is the actual enforcement of "image is not exposed before
    Pass-1 timing begins", not just a comment; tests assert this call
    order directly by injecting a recording `open_fn`."""
    image_id = image["image_id"]
    out_path = output_path(annotator_id, image_id, 1)
    if out_path.exists():
        print(f"[skip] {image_id} pass 1 already recorded -> {out_path}")
        return

    image_path = image_root / image["relative_path"]
    print(f"\n{'=' * 70}\n{image_id}  (Pass 1)\n{'=' * 70}")
    print(NON_NEGOTIABLE_REMINDER)
    print("\nPASS 1: 60-SECOND SALIENT VISUAL INVENTORY.")
    print(f"The image stays hidden until you start. Once started you have exactly "
          f"{duration_seconds:.0f} seconds to view it and type what stands out -- "
          f"input stops being accepted the instant time is up.")

    _prompt_and_wait("Press Enter to begin your 60 seconds...", line_queue)
    start_time = clock()
    open_fn(image_path, duration_seconds)

    print("Enter one item per line as: <label> | <location>  (blank line also ends the pass early)")
    items, timed_out = collect_items(line_queue, 1, start_time=start_time, duration_seconds=duration_seconds, clock=clock)
    elapsed = clock() - start_time
    print(f"\n  Pass 1 window closed ({elapsed:.1f}s elapsed, timed_out={timed_out}).")

    record = {
        "annotator_id": annotator_id, "image_id": image_id, "pass": 1, "items": items,
        "elapsed_seconds": round(elapsed, 1), "timed_out": timed_out,
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved -> {out_path}")


def run_pass2(annotator_id: str, image: dict, image_root: Path, line_queue: "queue.Queue[str | None]") -> None:
    image_id = image["image_id"]
    out_path = output_path(annotator_id, image_id, 2)
    if out_path.exists():
        print(f"[skip] {image_id} pass 2 already recorded -> {out_path}")
        return

    image_path = image_root / image["relative_path"]
    print(f"\n{'=' * 70}\n{image_id}  (Pass 2)\n{'=' * 70}")
    print(NON_NEGOTIABLE_REMINDER)
    print("\nPASS 2: EXHAUSTIVE VISUAL INVENTORY.")
    print("Take as long as you need. List every item you can identify, however small.")
    show_pass1_items_for_reference(annotator_id, image_id)

    _prompt_and_wait("Press Enter to open the image...", line_queue)
    open_image(image_path)

    print("Enter one item per line as: <label> | <location> | <confidence: clear/ambiguous>")
    print("Blank line finishes this pass.")
    items, _timed_out = collect_items(line_queue, 2)

    record = {"annotator_id": annotator_id, "image_id": image_id, "pass": 2, "items": items}
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DOAR 15-image development-set annotation tool (Pass 1 + Pass 2).")
    parser.add_argument("--annotator", required=True, help="Annotator id, e.g. A1 or A2")
    parser.add_argument("--pass", dest="pass_number", type=int, choices=(1, 2), default=None,
                         help="Restrict to a single pass; default runs Pass 1 then Pass 2 across all 15 images.")
    parser.add_argument("--image", dest="image_id", default=None,
                         help="Restrict to a single image_id (useful for resuming or testing the tool itself).")
    args = parser.parse_args()

    images = load_development_set()
    if args.image_id:
        images = [im for im in images if im["image_id"] == args.image_id]
        if not images:
            print(f"Unknown image_id: {args.image_id}")
            sys.exit(1)

    passes = [args.pass_number] if args.pass_number else [1, 2]

    print(f"Annotator: {args.annotator}  |  Images: {len(images)}  |  Passes: {passes}")
    print(NON_NEGOTIABLE_REMINDER)

    line_queue: "queue.Queue[str | None]" = queue.Queue()
    threading.Thread(target=_stdin_reader_loop, args=(line_queue,), daemon=True).start()

    for pass_number in passes:
        for image in images:
            if pass_number == 1:
                run_pass1(args.annotator, image, ROOT, line_queue)
            else:
                run_pass2(args.annotator, image, ROOT, line_queue)

    print("\nAll requested annotation passes complete (or already recorded).")
    print(f"Output directory: {output_path(args.annotator, '', 1).parent}")


if __name__ == "__main__":
    main()
