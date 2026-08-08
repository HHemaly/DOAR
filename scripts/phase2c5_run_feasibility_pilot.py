#!/usr/bin/env python
"""Phase 2C.5 Stage 4: runs Grounding DINO (primary) and OWLv2 (comparison)
part-box proposals over the small feasibility-pilot subset selected by
phase2c5_select_feasibility_pilot.py, and renders each image with its
proposed boxes overlaid so the proposals can be visually reviewed for the
go/no-go feasibility decision. Writes ONLY to outputs/phase2c5/ (private,
gitignored) -- proposal images and raw box data are never committed.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5 import proposals as prop_mod  # noqa: E402
from doar.phase2c5.ontology import PART_TARGETS  # noqa: E402
from doar.phase2c5.schema import utc_now_iso  # noqa: E402

COLORS = {"eye": (220, 40, 40), "mouth": (40, 140, 220), "hand": (30, 160, 30),
          "face": (200, 130, 20), "person": (140, 40, 200)}


def draw_boxes(image_path: Path, by_target: dict, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for target, instances in by_target.items():
        color = COLORS.get(target, (0, 0, 0))
        for inst in instances:
            x, y, bw, bh = inst.bbox
            box = [x * w, y * h, (x + bw) * w, (y + bh) * h]
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0] + 2, max(0, box[1] - 12)), f"{target} {inst.proposal_threshold}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def run_model(model_key: str, predict_fn, meta: dict, pilot_ids: list[str],
              images_dir: Path, out_dir: Path) -> list[dict]:
    rows = []
    for pilot_id in pilot_ids:
        image_path = images_dir / f"{pilot_id}.png"
        if not image_path.exists():
            candidates = list(images_dir.glob(f"{pilot_id}.*"))
            if not candidates:
                print(f"WARNING: no image file found for {pilot_id}")
                continue
            image_path = candidates[0]
        t0 = time.time()
        raw = predict_fn(str(image_path))
        elapsed = time.time() - t0
        by_target = prop_mod.build_part_proposals(
            raw, model_name=meta["model_name"], checkpoint=meta["checkpoint"],
            prompt=meta["prompt"], threshold=meta["threshold"], timestamp=utc_now_iso())
        draw_boxes(image_path, by_target, out_dir / f"{pilot_id}__{model_key}.png")
        for target, instances in by_target.items():
            for inst in instances:
                rows.append({"model": model_key, "pilot_id": pilot_id, "target": target,
                             "instance_index": inst.instance_index,
                             "bbox_x": round(inst.bbox[0], 4), "bbox_y": round(inst.bbox[1], 4),
                             "bbox_w": round(inst.bbox[2], 4), "bbox_h": round(inst.bbox[3], 4)})
        print(f"[{model_key}] {pilot_id}: "
              f"{sum(len(v) for v in by_target.values())} proposals in {elapsed:.1f}s", flush=True)
    return rows


def main() -> None:
    ids_path = ROOT / "outputs/phase2c5/feasibility_pilot_ids_private.json"
    pilot_ids = json.loads(ids_path.read_text(encoding="utf-8"))["pilot_ids"]
    images_dir = ROOT / "outputs/phase2c1/private_images"
    out_dir = ROOT / "outputs/phase2c5/feasibility_pilot_private"

    print(f"loading grounding_dino (targets={PART_TARGETS})...", flush=True)
    gd_predict, gd_meta = prop_mod.load_real_grounding_dino_parts()
    gd_rows = run_model("grounding_dino", gd_predict, gd_meta, pilot_ids, images_dir, out_dir)

    print("loading owlv2...", flush=True)
    owl_predict, owl_meta = prop_mod.load_real_owlv2_parts()
    owl_rows = run_model("owlv2", owl_predict, owl_meta, pilot_ids, images_dir, out_dir)

    all_rows = gd_rows + owl_rows
    out_csv = ROOT / "outputs/phase2c5/feasibility_pilot_raw_proposals_private.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "pilot_id", "target", "instance_index",
                                           "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} proposal rows -> {out_csv}")
    print(f"rendered overlay images -> {out_dir}")


if __name__ == "__main__":
    main()
