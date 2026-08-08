#!/usr/bin/env python
"""Phase 2C.6 Stage 4/5: resumable, checkpointed, batched proposal
generation over the expansion manifest. Grounding DINO is the primary
proposal source (matches Phase 2C.4/2C.4A's own finding: higher recall);
OWLv2 is loaded once and only invoked, per image, as a fallback for
targets Grounding DINO returned nothing for -- never a full duplicate
inference pass over every image, per instruction.

`--limit N` restricts the run to the first N pending pilot_ids (by sorted
pilot_id) -- used this session to run exactly the first 50 images and
stop, never proceeding to the rest automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5 import proposals as prop_mod  # noqa: E402
from doar.phase2c5.ontology import PART_TARGETS  # noqa: E402
from doar.phase2c5.schema import utc_now_iso  # noqa: E402
from doar.phase2c6 import proposal_batch as pb  # noqa: E402

MIN_STD_FOR_ASSESSABLE = 2.0  # 0-255 grayscale std-dev floor below which an image is "essentially blank"


def is_assessable(image_path: Path) -> bool:
    """Corrupted/unreadable or essentially blank -> not assessable. Does
    NOT attempt to judge 'unrelated content' -- that requires human
    judgment, not an automated check, per instruction."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L")
        import statistics
        pixels = list(img.getdata())
        if len(pixels) < 2:
            return False
        return statistics.pstdev(pixels) >= MIN_STD_FOR_ASSESSABLE
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="process at most this many PENDING images this run, then stop")
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    images_dir = ROOT / "outputs/phase2c6/expansion_images_private"
    checkpoint_path = ROOT / "outputs/phase2c6/proposal_batch_checkpoint_private.csv"
    proposals_csv_path = ROOT / "outputs/phase2c6/expansion_raw_proposals_private.csv"

    all_pilot_ids = sorted(p.stem for p in images_dir.glob("p2c6_*"))
    print(f"{len(all_pilot_ids)} images in expansion manifest")

    checkpoint = pb.load_checkpoint(checkpoint_path)
    todo = pb.pending_pilot_ids(all_pilot_ids, checkpoint)
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(todo)} pilot_ids to process this run (limit={args.limit})")
    if not todo:
        print("nothing to do")
        return

    print("loading grounding_dino (primary)...", flush=True)
    gd_predict, gd_meta = prop_mod.load_real_grounding_dino_parts()
    print("loading owlv2 (fallback only)...", flush=True)
    owl_predict, owl_meta = prop_mod.load_real_owlv2_parts()

    process_fn = partial(
        _process, predict_primary=gd_predict, primary_meta=gd_meta,
        predict_fallback=owl_predict, fallback_meta=owl_meta, targets=PART_TARGETS)

    t0 = time.time()
    summary = pb.run_batches(todo, images_dir, checkpoint_path=checkpoint_path,
                              proposals_csv_path=proposals_csv_path, batch_size=args.batch_size,
                              process_image_fn=process_fn)
    elapsed = time.time() - t0
    print(f"SUMMARY {summary} elapsed_s={elapsed:.1f} mean_s_per_image={elapsed / len(todo):.2f}")


def _process(pilot_id, image_path, *, predict_primary, primary_meta, predict_fallback, fallback_meta, targets):
    return pb.process_one_image(
        pilot_id, image_path, predict_primary=predict_primary, primary_meta=primary_meta,
        predict_fallback=predict_fallback, fallback_meta=fallback_meta, targets=targets,
        build_proposals=prop_mod.build_part_proposals, is_assessable=is_assessable,
        timestamp_fn=utc_now_iso)


if __name__ == "__main__":
    main()
