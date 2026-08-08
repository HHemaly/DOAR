"""Phase 2C.6 Stage 4: resumable, checkpointed, batched proposal
generation. Pure orchestration logic (`run_batches`) is fully testable
with an injected fake `predict_fn` -- mirrors every other phase's
injectable-backend pattern (`load_real_*` is the only code that touches
real model weights, never called by the test suite).

Checkpoint discipline: the checkpoint file is the single source of truth
for "is this image done" and is rewritten (atomic replace) after EVERY
batch, never only at the end of the whole run. A restart reads the
checkpoint first and only processes pilot_ids not already `completed` or
`skipped_not_assessable` -- `failed` images ARE retried on restart (a
failure may have been transient), `pending` images (present in the
checkpoint from a run that was killed mid-batch) are also retried.
"""
from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

STATUS_VALUES = frozenset({"pending", "completed", "failed", "skipped_not_assessable"})
_TERMINAL_STATUSES = frozenset({"completed", "skipped_not_assessable"})

CHECKPOINT_FIELDS = ["pilot_id", "status", "n_proposals", "primary_model", "fallback_used",
                      "error", "elapsed_s", "timestamp"]


def load_checkpoint(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["pilot_id"]: row for row in csv.DictReader(f)}


def save_checkpoint(path: str | Path, checkpoint: dict[str, dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CHECKPOINT_FIELDS)
            writer.writeheader()
            for pilot_id in sorted(checkpoint):
                row = checkpoint[pilot_id]
                writer.writerow({k: row.get(k, "") for k in CHECKPOINT_FIELDS})
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def pending_pilot_ids(all_pilot_ids: list[str], checkpoint: dict[str, dict]) -> list[str]:
    """Every pilot_id not already in a terminal state -- this is the ONLY
    function that decides what "unfinished" means; a restart calls this
    before doing anything else."""
    return [pid for pid in all_pilot_ids
            if checkpoint.get(pid, {}).get("status") not in _TERMINAL_STATUSES]


def make_batches(pilot_ids: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return [pilot_ids[i:i + batch_size] for i in range(0, len(pilot_ids), batch_size)]


def append_proposal_rows(path: str | Path, rows: list[dict]) -> None:
    """Appends to the growing raw-proposals CSV -- writes the header only
    if the file doesn't exist yet, so this is safe to call once per batch
    across many resumed runs."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "pilot_id", "target", "instance_index",
                                                "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def process_one_image(pilot_id: str, image_path: Path, *,
                       predict_primary: Callable, primary_meta: dict,
                       predict_fallback: Callable | None, fallback_meta: dict | None,
                       targets: tuple[str, ...], build_proposals: Callable,
                       is_assessable: Callable[[Path], bool], timestamp_fn: Callable[[], str],
                       ) -> tuple[dict, list[dict]]:
    """Returns (checkpoint_row, proposal_rows). Never raises -- any
    exception from the real model call is caught and recorded as
    status='failed' so one bad image cannot kill a batch of 25-50."""
    t0 = time.time()
    if not is_assessable(image_path):
        return ({"pilot_id": pilot_id, "status": "skipped_not_assessable", "n_proposals": 0,
                 "primary_model": "", "fallback_used": "False", "error": "",
                 "elapsed_s": f"{time.time() - t0:.2f}", "timestamp": timestamp_fn()}, [])
    try:
        raw = predict_primary(str(image_path))
        by_target = build_proposals(raw, model_name=primary_meta["model_name"],
                                     checkpoint=primary_meta["checkpoint"], prompt=primary_meta["prompt"],
                                     threshold=primary_meta["threshold"], timestamp=timestamp_fn())
        fallback_used = False
        empty_targets = [t for t in targets if not by_target.get(t)]
        if predict_fallback is not None and empty_targets:
            raw_fb = predict_fallback(str(image_path))
            by_target_fb = build_proposals(raw_fb, model_name=fallback_meta["model_name"],
                                            checkpoint=fallback_meta["checkpoint"], prompt=fallback_meta["prompt"],
                                            threshold=fallback_meta["threshold"], timestamp=timestamp_fn())
            for t in empty_targets:
                if by_target_fb.get(t):
                    by_target[t] = by_target_fb[t]
                    fallback_used = True
        rows = []
        for target in targets:
            for inst in by_target.get(target, []):
                rows.append({"model": inst.proposal_model, "pilot_id": pilot_id, "target": target,
                             "instance_index": inst.instance_index,
                             "bbox_x": round(inst.bbox[0], 4), "bbox_y": round(inst.bbox[1], 4),
                             "bbox_w": round(inst.bbox[2], 4), "bbox_h": round(inst.bbox[3], 4)})
        return ({"pilot_id": pilot_id, "status": "completed", "n_proposals": len(rows),
                 "primary_model": primary_meta["model_name"], "fallback_used": str(fallback_used),
                 "error": "", "elapsed_s": f"{time.time() - t0:.2f}", "timestamp": timestamp_fn()}, rows)
    except Exception as e:  # noqa: BLE001 -- one bad image must not kill the batch
        return ({"pilot_id": pilot_id, "status": "failed", "n_proposals": 0,
                 "primary_model": primary_meta["model_name"], "fallback_used": "False",
                 "error": f"{type(e).__name__}: {e}", "elapsed_s": f"{time.time() - t0:.2f}",
                 "timestamp": timestamp_fn()}, [])


def run_batches(pilot_ids: list[str], images_dir: Path, *, checkpoint_path: Path,
                 proposals_csv_path: Path, batch_size: int, process_image_fn: Callable,
                 log: Callable[[str], None] = print) -> dict:
    """Top-level resumable driver. `process_image_fn(pilot_id, image_path)
    -> (checkpoint_row, proposal_rows)` is normally `process_one_image`
    with its model-specific kwargs pre-bound (functools.partial) -- kept
    as a plain parameter here so tests can inject a trivial fake without
    touching any real model loader."""
    checkpoint = load_checkpoint(checkpoint_path)
    todo = pending_pilot_ids(pilot_ids, checkpoint)
    batches = make_batches(todo, batch_size)
    log(f"{len(pilot_ids)} total images, {len(pilot_ids) - len(todo)} already done, "
        f"{len(todo)} to process in {len(batches)} batch(es) of up to {batch_size}")

    for batch_idx, batch in enumerate(batches):
        batch_rows = []
        for pilot_id in batch:
            candidates = list(Path(images_dir).glob(f"{pilot_id}.*"))
            if not candidates:
                checkpoint[pilot_id] = {"pilot_id": pilot_id, "status": "failed", "n_proposals": 0,
                                         "primary_model": "", "fallback_used": "False",
                                         "error": "image file not found", "elapsed_s": "0",
                                         "timestamp": ""}
                continue
            row, proposal_rows = process_image_fn(pilot_id, candidates[0])
            checkpoint[pilot_id] = row
            batch_rows.extend(proposal_rows)
        # Checkpoint + proposals are flushed after EVERY batch, not at the
        # end of the run -- the whole point of this function.
        append_proposal_rows(proposals_csv_path, batch_rows)
        save_checkpoint(checkpoint_path, checkpoint)
        log(f"batch {batch_idx + 1}/{len(batches)} done "
            f"({len(batch)} images, {len(batch_rows)} proposal rows) -- checkpoint saved")

    statuses = {pid: checkpoint[pid]["status"] for pid in pilot_ids if pid in checkpoint}
    summary = {
        "n_total": len(pilot_ids),
        "n_completed": sum(1 for s in statuses.values() if s == "completed"),
        "n_failed": sum(1 for s in statuses.values() if s == "failed"),
        "n_skipped_not_assessable": sum(1 for s in statuses.values() if s == "skipped_not_assessable"),
        "n_pending": len(pilot_ids) - len(statuses),
    }
    return summary
