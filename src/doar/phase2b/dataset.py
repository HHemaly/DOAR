"""Phase 2B pilot-sample selection and blind re-identification.

Selects a small, group-disjoint (one image per duplicate group), non-
conflicted sample from the existing Phase 7B partition manifest, then
copies the selected images into a flat, randomized, opaquely-renamed
working directory with no folder-name or filename trace of the original
emotion-class label -- the blind-annotation requirement carried over from
`PHASE3_DETECTOR_EVALUATION_PLAN.md` Section 5 (annotators must have no
visibility into the emotion-class folder).
"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from .duplicate_groups import load_group_lookup

DEFAULT_SEED = 2026


def select_pilot_sample(n: int, *, seed: int = DEFAULT_SEED,
                         manifest_path: Path | None = None) -> list[dict]:
    """Deterministically selects up to `n` images, one per duplicate group,
    from the non-conflicted pool of the existing Phase 7B partition
    manifest. Returns a list of {"image_id", "path", "group_id"} dicts,
    sorted by the opaque pilot_id assigned in `blind_copy_sample`'s caller
    (sorting here is by image_id, for reproducibility independent of
    dict-ordering)."""
    lookup = load_group_lookup(manifest_path)
    clean = {iid: info for iid, info in lookup.items() if info["conflict_status"] == "clean"}
    by_group: dict[str, list[str]] = {}
    for iid, info in clean.items():
        by_group.setdefault(info["group_id"], []).append(iid)

    group_ids = sorted(by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    if len(group_ids) < n:
        raise ValueError(
            f"Only {len(group_ids)} distinct clean duplicate-groups are "
            f"available, fewer than the requested sample size {n}.")
    selected = []
    for gid in group_ids[:n]:
        members = sorted(by_group[gid])
        image_id = members[0]  # deterministic: lowest image_id in the group
        selected.append({"image_id": image_id, "path": clean[image_id]["path"], "group_id": gid})
    selected.sort(key=lambda r: r["image_id"])
    return selected


def blind_copy_sample(selected: list[dict], output_dir: Path, *,
                       seed: int = DEFAULT_SEED) -> list[dict]:
    """Copies each selected image into `output_dir` under an opaque
    `p2b_XXXX.<ext>` filename, in an order independently shuffled from the
    selection order (so file-listing order carries no residual information
    about class or group). Returns the same records with a `pilot_id`
    field added; writes no emotion-class or split information into the
    output directory at all. The mapping back to image_id/original path is
    returned to the caller only -- it must not be written next to the
    blinded images themselves, or the blinding is defeated.

    Reproducibility note: the shuffle consumes `len(selected)` draws from
    `random.Random(seed + 1)`, so the resulting pilot_id assignment is NOT
    invariant to the length of `selected` -- calling this with a 20-image
    `selected` list does not reproduce the first 20 pilot_ids of an
    80-image call with the same seed. Always reselect with the same `n`
    used originally to reproduce a specific run's pilot_id assignment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(len(selected)))
    random.Random(seed + 1).shuffle(order)
    out = []
    for rank, idx in enumerate(order):
        rec = dict(selected[idx])
        src = Path(rec["path"])
        pilot_id = f"p2b_{rank:04d}"
        dest = output_dir / f"{pilot_id}{src.suffix.lower()}"
        shutil.copyfile(src, dest)
        rec["pilot_id"] = pilot_id
        rec["blind_path"] = str(dest)
        out.append(rec)
    out.sort(key=lambda r: r["pilot_id"])
    return out


def write_pilot_mapping_csv(records: list[dict], output_path: Path) -> Path:
    """The re-identification key (pilot_id -> real image_id/path/group_id).
    Kept separate from the blinded image directory and from the annotation
    manifest an annotator would see while labeling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pilot_id", "image_id", "source_image_group", "original_path"])
        for r in records:
            writer.writerow([r["pilot_id"], r["image_id"], r["group_id"], r["path"]])
    return output_path
