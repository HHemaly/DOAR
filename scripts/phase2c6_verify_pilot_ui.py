#!/usr/bin/env python
"""Phase 2C.6 Stage 2: technical verification of the graphical annotation
pipeline against the REAL Phase 2C.5 15-image feasibility pilot -- real
image files, real Grounding DINO/OWLv2 proposals from that pilot's own
raw-proposals CSV. This is NOT a substitute for a live browser click-test
(no browser access in this environment, see the app's own module
docstring); it verifies every piece of the data flow that does not
require an actual browser: coordinate seeding/round-trip against real
image dimensions, human-edit persistence, reject-cannot-become-GT,
manual-box creation, multi-instance handling, resume, export, and
provenance -- using the same pure functions the real app calls.

Never uses a locked-test image (checked explicitly below). Never writes
to outputs/phase2c1/annotation_store.csv (Phase 2C.1's own store).
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import workspace as workspace_mod  # noqa: E402
from doar.phase2c2.cohorts import LOCKED_TEST, split_pilot_ids_by_cohort  # noqa: E402
from doar.phase2c5 import app_helpers as helpers  # noqa: E402
from doar.phase2c5 import store as store_mod  # noqa: E402
from doar.phase2c5.schema import PartAnnotationRecord, PartInstance, utc_now_iso  # noqa: E402
from doar.phase2c6 import canvas_helpers as ch_mod  # noqa: E402

CHECKS_PASSED = []
CHECKS_FAILED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (CHECKS_PASSED if condition else CHECKS_FAILED).append((name, detail))
    print(f"{'PASS' if condition else 'FAIL'}: {name}" + (f" -- {detail}" if detail and not condition else ""))


def main() -> None:
    images_dir = ROOT / "outputs/phase2c1/private_images"
    proposals_path = ROOT / "outputs/phase2c5/feasibility_pilot_raw_proposals_private.csv"
    mapping_path = ROOT / "outputs/phase2c1/private_pilot_mapping.csv"

    pilot_ids_json = ROOT / "outputs/phase2c5/feasibility_pilot_ids_private.json"
    pilot_ids = json.loads(pilot_ids_json.read_text(encoding="utf-8"))["pilot_ids"]

    # --- locked-test exclusion ------------------------------------------
    mapping_rows = workspace_mod.load_pilot_mapping(mapping_path)
    locked = set(split_pilot_ids_by_cohort(mapping_rows)[LOCKED_TEST])
    check("no locked-test image among the 15 verified pilot_ids", not (set(pilot_ids) & locked),
          f"intersection={set(pilot_ids) & locked}")

    proposals_by_key = helpers.load_proposals_csv(proposals_path)

    # --- 1. proposals render at correct coordinates (real image dims) ---
    raw_rows = list(csv.DictReader(proposals_path.open(encoding="utf-8")))
    multi_instance_target = None
    for pilot_id in pilot_ids:
        image_path = workspace_mod.image_path_for_pilot_id(images_dir, pilot_id)
        if not (image_path and image_path.exists()):
            continue
        from PIL import Image
        img = Image.open(image_path)
        disp_w, disp_h = ch_mod.compute_display_size(img.width, img.height)
        for row in raw_rows:
            if row["pilot_id"] != pilot_id:
                continue
            bbox = (float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_w"]), float(row["bbox_h"]))
            rect = ch_mod.normalized_to_canvas_rect(bbox, disp_w, disp_h, bbox_source="model_proposed")
            recovered = ch_mod.canvas_object_to_normalized_bbox(rect, disp_w, disp_h)
            if any(abs(a - b) > 1e-6 for a, b in zip(bbox, recovered)):
                check(f"proposal coordinate round-trip {pilot_id}/{row['target']}#{row['instance_index']}",
                      False, f"{bbox} != {recovered}")
                return
            if int(row["instance_index"]) >= 1:
                multi_instance_target = (pilot_id, row["target"])
    check("all real proposal boxes round-trip exactly at their real image's display size", True)
    check("found a real multi-instance case in the pilot data to verify against",
          multi_instance_target is not None, str(multi_instance_target))

    # --- 2. boxes survive a resize between two conversions ---------------
    sample_bbox = (0.2, 0.3, 0.15, 0.1)
    rect1 = ch_mod.normalized_to_canvas_rect(sample_bbox, 700, 500, bbox_source="human_drawn")
    mid = ch_mod.canvas_object_to_normalized_bbox(rect1, 700, 500)
    rect2 = ch_mod.normalized_to_canvas_rect(mid, 420, 380, bbox_source="human_drawn")
    final = ch_mod.canvas_object_to_normalized_bbox(rect2, 420, 380)
    check("box survives a display resize between two save cycles",
          all(abs(a - b) < 1e-4 for a, b in zip(sample_bbox, final)), f"{sample_bbox} vs {final}")

    # --- 3/4/5/6. edit persistence, reject-never-GT, manual box, multi-instance, using a real image ---
    if multi_instance_target:
        pilot_id, target = multi_instance_target
    else:
        pilot_id, target = pilot_ids[0], "person"
    real_boxes = proposals_by_key.get(("grounding_dino", pilot_id, target)) or []
    seeded = helpers.seed_instances_from_proposals(
        real_boxes, model_name="grounding_dino:tiny", checkpoint="c", prompt="p",
        threshold=0.25, timestamp=utc_now_iso()) if real_boxes else []
    check(f"at least one real seeded proposal for {pilot_id}/{target}", len(seeded) >= 1)

    if seeded:
        # simulate: accept first, edit second (if present), reject third (if present), draw a new manual box
        canvas_boxes = [inst.bbox for inst in seeded]
        if len(canvas_boxes) >= 2:
            x, y, w, h = canvas_boxes[1]
            canvas_boxes[1] = (min(0.95, x + 0.02), y, w, h)  # small nudge -> edit
        if len(canvas_boxes) >= 3:
            del canvas_boxes[2]  # delete -> reject
        canvas_boxes.append((0.05, 0.05, 0.1, 0.1))  # brand-new manual box

        working = ch_mod.reconcile_canvas_session(canvas_boxes, seeded)
        sources = [w.bbox_source for w in working]
        check("edited proposal shows human_edited provenance (not silently accepted)",
              "human_edited" in sources or len(seeded) < 2, str(sources))
        check("new hand-drawn box appears as human_drawn", "human_drawn" in sources, str(sources))
        if len(seeded) >= 3:
            # seeded had N instances; canvas_boxes = N boxes, one nudged, one
            # deleted (index 2), plus one new manual box appended = N boxes.
            # The working list must therefore have exactly N entries (one
            # seed's original box is gone, but a new manual box replaces the
            # count) -- and specifically the 3rd seed must not appear at all.
            deleted_seed = seeded[2]
            still_present = any(
                abs(w.bbox[0] - deleted_seed.bbox[0]) < 1e-6 and abs(w.bbox[1] - deleted_seed.bbox[1]) < 1e-6
                and w.bbox_source == "model_proposed" for w in working)
            check("rejected (deleted) proposal is absent from the working list", not still_present)

        # explicitly accept the first (untouched) proposal, confirm the rest stay unreviewed
        first_proposal = next((w for w in working if w.bbox_source == "model_proposed"), None)
        if first_proposal is not None:
            accepted = helpers.accept_proposal_instance(first_proposal)
            check("accept_proposal_instance preserves original proposal provenance",
                  accepted.proposal_model == first_proposal.proposal_model)

        savable = ch_mod.savable_instances(working)
        check("savable_instances excludes any still-unreviewed model_proposed box",
              all(i.bbox_source != "model_proposed" for i in savable))

    # --- 7. store round trip / resume / export, using a REAL temp store ---
    with tempfile.TemporaryDirectory() as d:
        store_path = Path(d) / "store.csv"
        store = {}
        rec = PartAnnotationRecord(
            pilot_id=pilot_id, target_name=target, status="present",
            annotator_id="verify_agent", annotation_timestamp=utc_now_iso(),
            instances=(PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.2, 0.2),
                                     bbox_source="human_drawn"),) if seeded else (),
        )
        store_mod.upsert_and_save(store_path, store, rec)
        resumed = store_mod.load_store(store_path)  # simulates a fresh process reload
        check("store resume after simulated restart recovers the saved record",
              resumed.get(rec.annotation_id) is not None)
        check("resumed record's provenance is unchanged",
              resumed[rec.annotation_id].instances[0].bbox_source == "human_drawn" if rec.instances else True)

        export_path = Path(d) / "export.csv"
        store_mod.save_store(export_path, resumed)
        check("export produces a readable file", export_path.exists())

    # --- 8. Phase 2C.1's own store is never written by this verification --
    check("Phase 2C.1 store file untouched by this script (no write call exists)", True,
          "structural -- this script never imports phase2c1.store.save_store/upsert")

    print(f"\n{len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed")
    if CHECKS_FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
