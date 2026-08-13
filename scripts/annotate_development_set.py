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

Usage:
    python scripts/annotate_development_set.py --annotator A1
    python scripts/annotate_development_set.py --annotator A2 --pass 1
    python scripts/annotate_development_set.py --annotator A1 --image h38
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_SET_PATH = ROOT / "DEVELOPMENT_SET_15.json"
ANNOTATIONS_DIR = ROOT / "annotations"

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
    try:
        os.startfile(str(path))  # Windows-only; this tool targets this project's Windows environment.
    except Exception as exc:
        print(f"  (could not auto-open image -- please open it manually: {path}  [{exc}])")


def prompt_items(pass_number: int) -> list[dict]:
    field_hint = "<label> | <location>" + (" | <confidence: clear/ambiguous>" if pass_number == 2 else "")
    print(f"Enter one item per line as: {field_hint}")
    print("Blank line finishes this pass.")
    items = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = [p.strip() for p in line.split("|")]
        label = parts[0] if parts else ""
        if not label:
            continue
        location = parts[1] if len(parts) > 1 else ""
        entry = {"label": label, "location": location}
        if pass_number == 2:
            confidence = parts[2] if len(parts) > 2 and parts[2] in ("clear", "ambiguous") else "clear"
            entry["confidence"] = confidence
        items.append(entry)
    return items


def show_pass1_items_for_reference(annotator_id: str, image_id: str) -> None:
    pass1_path = output_path(annotator_id, image_id, 1)
    if not pass1_path.exists():
        return
    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
    print("Your Pass 1 items for this image (Pass 2 must re-list every one of these, plus anything new):")
    for item in pass1["items"]:
        print(f"    - {item['label']}" + (f" ({item['location']})" if item.get("location") else ""))


def run_pass(annotator_id: str, image: dict, pass_number: int, image_root: Path) -> None:
    image_id = image["image_id"]
    out_path = output_path(annotator_id, image_id, pass_number)
    if out_path.exists():
        print(f"[skip] {image_id} pass {pass_number} already recorded -> {out_path}")
        return

    image_path = image_root / image["relative_path"]
    print(f"\n{'=' * 70}\n{image_id}  (Pass {pass_number})\n{'=' * 70}")
    print(NON_NEGOTIABLE_REMINDER)
    open_image(image_path)

    start = None
    if pass_number == 1:
        print("\nPASS 1: 60-SECOND SALIENT VISUAL INVENTORY.")
        print("Glance at the drawing and list only what stands out immediately -- do not study it in detail.")
        input("Press Enter when you are ready to begin your 60 seconds...")
        start = time.monotonic()
    else:
        print("\nPASS 2: EXHAUSTIVE VISUAL INVENTORY.")
        print("Take as long as you need. List every item you can identify, however small.")
        show_pass1_items_for_reference(annotator_id, image_id)

    items = prompt_items(pass_number)
    record = {"annotator_id": annotator_id, "image_id": image_id, "pass": pass_number, "items": items}

    if pass_number == 1 and start is not None:
        elapsed = time.monotonic() - start
        record["elapsed_seconds"] = round(elapsed, 1)
        if elapsed > 90:
            print(f"  (note: {elapsed:.0f}s elapsed -- Pass 1 is meant to be quick; recorded as-is.)")

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

    for pass_number in passes:
        for image in images:
            run_pass(args.annotator, image, pass_number, ROOT)

    print("\nAll requested annotation passes complete (or already recorded).")
    print(f"Output directory: {output_path(args.annotator, '', 1).parent}")


if __name__ == "__main__":
    main()
