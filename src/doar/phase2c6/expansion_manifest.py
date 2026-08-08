"""Phase 2C.6 Stage 3: builds the blinded ~300-image development-eligible
expansion manifest.

Reuses Phase 2C.5's own `expansion_sampling.select_expansion_sample`
(never reimplemented) for the actual group-disjoint, non-conflicted,
deterministic, emotion-blind selection. The only new logic here is
blinding (copying selected images to opaque IDs) with a prefix
(`p2c6_XXXX`) DISTINCT from the existing Phase 2C.1 80-image pilot's own
`p2b_XXXX` namespace, so the two blinded ID spaces can never collide or
be confused with each other -- mirrors `phase2b.dataset.blind_copy_sample`
exactly (frozen, never modified), just with a different id prefix and its
own independent shuffle-seed derivation.

Everything this module writes lives under `outputs/` (gitignored) except
`write_expansion_summary`, which writes ONLY opaque pilot_ids and counts,
never an `image_id`, `group_id`, or `original_path`.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

from ..phase2b.dataset import write_pilot_mapping_csv
from ..phase2c5.expansion_sampling import EXPANSION_SEED_STAGE_B, select_expansion_sample

PILOT_ID_PREFIX = "p2c6"


def blind_copy_expansion_sample(selected: list[dict], output_dir: Path, *, seed: int) -> list[dict]:
    """Identical algorithm to `phase2b.dataset.blind_copy_sample` (frozen,
    never modified) -- independently shuffled copy order, opaque
    zero-padded IDs -- except the ID prefix is `p2c6_` instead of `p2b_`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(len(selected)))
    random.Random(seed + 1).shuffle(order)
    out = []
    for rank, idx in enumerate(order):
        rec = dict(selected[idx])
        src = Path(rec["path"])
        pilot_id = f"{PILOT_ID_PREFIX}_{rank:04d}"
        dest = output_dir / f"{pilot_id}{src.suffix.lower()}"
        shutil.copyfile(src, dest)
        rec["pilot_id"] = pilot_id
        rec["blind_path"] = str(dest)
        out.append(rec)
    out.sort(key=lambda r: r["pilot_id"])
    return out


def load_used_image_ids(*mapping_paths: Path) -> set[str]:
    """Every `image_id` already used by a prior round (Phase 2C.1's 80-image
    pilot at minimum) -- read from private pilot-mapping CSVs so a new
    expansion round can never re-select an image already annotated."""
    used: set[str] = set()
    for path in mapping_paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("image_id"):
                    used.add(row["image_id"])
    return used


def build_expansion_manifest(n: int, *, seed: int = EXPANSION_SEED_STAGE_B,
                              exclude_image_ids: set[str], corpus_manifest_path,
                              images_output_dir: Path, mapping_output_path: Path) -> list[dict]:
    """Runs the full Stage 3 pipeline: select -> blind-copy -> write the
    private re-identification mapping. Returns the blinded records
    (pilot_id/image_id/group_id/path/blind_path) -- callers must never
    pass this return value into anything that gets committed; only
    `write_expansion_summary`'s output (opaque pilot_ids/counts only) is
    safe to commit."""
    selected = select_expansion_sample(n, seed=seed, exclude_image_ids=exclude_image_ids,
                                        manifest_path=corpus_manifest_path)
    blinded = blind_copy_expansion_sample(selected, images_output_dir, seed=seed)
    write_pilot_mapping_csv(blinded, mapping_output_path)
    return blinded


def write_expansion_summary(blinded: list[dict], out_path: Path, *, seed: int,
                             n_excluded_prior_round: int) -> Path:
    """Public, non-private artifact: opaque pilot_ids and counts only --
    no image_id, group_id, or original_path."""
    summary = {
        "n_images": len(blinded),
        "seed": seed,
        "pilot_id_prefix": PILOT_ID_PREFIX,
        "n_excluded_prior_round_image_ids": n_excluded_prior_round,
        "pilot_ids": sorted(r["pilot_id"] for r in blinded),
        "selection_method": "expansion_sampling.select_expansion_sample "
                             "(group-disjoint, non-conflicted, excludes prior-round image_ids, "
                             "no emotion-label filtering)",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
