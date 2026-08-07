"""Phase 2C.1 Stage D -- private blinded annotation workspace.

Wraps Phase 2B's own, unmodified `select_pilot_sample` / `blind_copy_sample`
(src/doar/phase2b/dataset.py) rather than reimplementing selection --
PHASE2C1_ANNOTATION_PROVENANCE_AUDIT.md documents why this reproduces the
exact original 80-image pilot. Nothing here touches
artifacts/phase2b/annotation_manifest.csv.

Everything this module writes lives under an `outputs/`-rooted directory
(gitignored, private, local-only) -- never under a path git would track.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..phase2b.dataset import DEFAULT_SEED, blind_copy_sample, select_pilot_sample
# Referenced read-only, purely to resolve the SAME default manifest path
# `select_pilot_sample`/`load_group_lookup` already use, so this module's
# own original_split lookup (below) can never silently drift from the file
# the selection itself was actually computed against. Does not modify
# src/doar/phase2b/duplicate_groups.py in any way.
from ..phase2b.duplicate_groups import _PRECOMPUTED_MANIFEST as _DEFAULT_PARTITION_MANIFEST
from .schema import AnnotationRecord

PILOT_MANIFEST_VERSION = "phase2c1_pilot_v1_seed2026_n80"

# Phase 2B's own recorded annotator id, preserved verbatim when migrating --
# never relabeled as if it were a Phase 2C.1 annotator.
PHASE2B_ANNOTATOR_ID = "single_annotator_session_2026-08-06"

# original_split values that must never silently enter a future detector
# training/tuning/calibration/model-selection set -- see
# assert_pilot_ids_exclude_locked_test below. Recorded here, not inline,
# so a future caller has one obvious place to import the definition from.
LOCKED_TEST_SPLIT_VALUE = "test"


def build_pilot_workspace(*, n: int = 80, seed: int = DEFAULT_SEED,
                           images_dir: str | Path, mapping_path: str | Path,
                           partition_manifest_path: str | Path | None = None) -> dict:
    """Selects the n-image group-disjoint pilot sample and blind-copies it
    into `images_dir` under opaque p2b_XXXX filenames, writing the
    pilot_id -> real image_id/group/path re-identification key to
    `mapping_path` -- a SEPARATE file from `images_dir`, never inside it,
    so the blinded folder itself never carries a de-anonymization trail.

    Safe to re-run: `select_pilot_sample`/`blind_copy_sample` are
    deterministic given the same seed and partition manifest, so re-running
    this reproduces the identical pilot_id assignment (verified in
    PHASE2C1_ANNOTATION_PROVENANCE_AUDIT.md) rather than drifting.
    """
    images_dir = Path(images_dir)
    mapping_path = Path(mapping_path)
    resolved_manifest_path = Path(partition_manifest_path) if partition_manifest_path else _DEFAULT_PARTITION_MANIFEST
    selected = select_pilot_sample(n, seed=seed, manifest_path=resolved_manifest_path)
    copied = blind_copy_sample(selected, images_dir, seed=seed)
    original_split_by_image_id = _load_original_split_lookup(resolved_manifest_path)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pilot_id", "image_id", "source_image_group", "original_split",
                          "original_path", "blind_path"])
        for r in sorted(copied, key=lambda r: r["pilot_id"]):
            writer.writerow([
                r["pilot_id"], r["image_id"], r["group_id"],
                original_split_by_image_id.get(r["image_id"], "unknown"),
                r["path"], r["blind_path"],
            ])

    n_test_split = sum(
        1 for r in copied if original_split_by_image_id.get(r["image_id"]) == LOCKED_TEST_SPLIT_VALUE
    )
    return {
        "n_selected": len(copied),
        "images_dir": str(images_dir),
        "mapping_path": str(mapping_path),
        "seed": seed,
        "pilot_manifest_version": PILOT_MANIFEST_VERSION,
        "n_from_locked_test_split": n_test_split,
    }


def _load_original_split_lookup(partition_manifest_path: str | Path) -> dict[str, str]:
    """Reads image_id -> original_split (train/valid/test) directly from
    the full Phase 7B partition manifest -- independent of
    `duplicate_groups.load_group_lookup`, which deliberately does not pass
    this column through. This is READ-ONLY and does not modify
    src/doar/phase2b/duplicate_groups.py or the manifest file itself."""
    path = Path(partition_manifest_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["image_id"]: row["original_split"] for row in csv.DictReader(f)}


def pilot_ids_from_locked_test_split(mapping_path: str | Path) -> set[str]:
    """The pilot_ids among the private workspace whose ORIGINAL dataset
    split was 'test' -- the locked split. Annotation of these images is
    explicitly permitted in Phase 2C.1 (this is an annotation-expansion
    phase, not model development), but any future detector
    training/threshold-tuning/model-selection/calibration code must
    exclude this set before using annotated pilot images for those
    purposes -- see assert_pilot_ids_exclude_locked_test."""
    rows = load_pilot_mapping(mapping_path)
    return {r["pilot_id"] for r in rows if r.get("original_split") == LOCKED_TEST_SPLIT_VALUE}


def assert_pilot_ids_exclude_locked_test(pilot_ids: set[str] | list[str], mapping_path: str | Path) -> None:
    """Future-readiness guard for Phase 2C.2+ detector development: raises
    ValueError if any of `pilot_ids` originated in the locked test split.
    Intended to be called by any future training/tuning/model-selection/
    calibration code path BEFORE it uses a set of annotated pilot images --
    not wired into anything in Phase 2C.1 itself, since no detector
    training happens in this phase."""
    locked = pilot_ids_from_locked_test_split(mapping_path)
    violating = set(pilot_ids) & locked
    if violating:
        raise ValueError(
            f"{len(violating)} pilot_id(s) originate from the original LOCKED TEST split "
            f"and must never be used for detector training/threshold-tuning/model-selection/"
            f"calibration: {sorted(violating)}"
        )


def load_pilot_mapping(mapping_path: str | Path) -> list[dict]:
    return list(csv.DictReader(Path(mapping_path).open(encoding="utf-8")))


def list_workspace_images(images_dir: str | Path) -> list[str]:
    """Returns sorted pilot_ids present as image files in `images_dir` --
    the only thing the annotation app needs to enumerate images; it never
    reads `mapping_path` (that would defeat the blind)."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in exts)


def image_path_for_pilot_id(images_dir: str | Path, pilot_id: str) -> Path | None:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return None
    for p in images_dir.iterdir():
        if p.stem == pilot_id:
            return p
    return None


def migrate_phase2b_provisional(annotation_manifest_path: str | Path,
                                 mapping_path: str | Path) -> list[AnnotationRecord]:
    """One-time, read-only migration: turns the 200 rows already committed
    in artifacts/phase2b/annotation_manifest.csv (20 images x 10 classes)
    into Phase 2C.1 AnnotationRecords, so the app can load them as
    PROVISIONAL review targets -- never as if a Phase 2C.1 annotator
    produced them. review_status stays 'unreviewed' (matching the source
    file's own column) and annotator_id is preserved as the real Phase 2B
    session id, not relabeled.

    annotator_type is set to 'legacy_provisional_human' (never 'human') --
    this is the structural signal quality.py's agreement tooling uses to
    exclude these rows from human-human Cohen's kappa / percent agreement.
    These are NOT confirmed independent ground truth: a single annotator,
    unreviewed, disclosed as such in docs/PHASE2B_ANNOTATION_PROTOCOL.md.
    They remain available for an explicit, separately-labeled "human vs.
    provisional reference" comparison (quality.compute_provisional_reference_comparison).

    Never writes artifacts/phase2b/annotation_manifest.csv -- read-only.
    Returns records for the caller to upsert into a Phase 2C.1 store; does
    not touch any store itself.
    """
    manifest_rows = list(csv.DictReader(Path(annotation_manifest_path).open(encoding="utf-8")))
    mapping_by_pilot = {r["pilot_id"]: r for r in load_pilot_mapping(mapping_path)}

    records = []
    for row in manifest_rows:
        pilot_id = row["pilot_id"]
        group_row = mapping_by_pilot.get(pilot_id)
        source_image_group = group_row["source_image_group"] if group_row else row.get("source_image_group", "")
        image_id = group_row["image_id"] if group_row else row.get("image_id", "")
        status = row["status"]
        instance_count = int(row.get("instance_count") or 0)
        bbox_raw = (row.get("bbox") or "").strip()
        bbox = tuple(float(v) for v in bbox_raw.split(",")) if bbox_raw else None
        records.append(AnnotationRecord(
            pilot_id=pilot_id,
            image_id=image_id,
            source_image_group=source_image_group,
            class_name=row["class"],
            status=status,
            annotator_id=row.get("annotator") or PHASE2B_ANNOTATOR_ID,
            annotator_type="legacy_provisional_human",
            annotation_timestamp=row.get("annotation_timestamp") or "2026-08-06T00:00:00+00:00",
            instance_count=instance_count,
            bbox=bbox,
            partial_or_occluded=str(row.get("partial_or_occluded", "False")).strip().lower() == "true",
            uncertainty_reason=row.get("uncertain_reason", "") or "",
            review_status=row.get("review_status") or "unreviewed",
            notes=row.get("notes", "") or "",
            source_manifest_version="phase2b_annotation_manifest_v1_migrated",
        ))
    return records


def remaining_unannotated_pilot_ids(images_dir: str | Path, store_pilot_ids: set[str]) -> list[str]:
    """The 60 (of 80) pilot images with no annotation row yet, per
    CURRENT_TO_TARGET_GAP_V7.md's own accounting -- never pre-filled with
    placeholder rows, they simply don't appear in the store until a human
    annotates them through the app."""
    return sorted(set(list_workspace_images(images_dir)) - store_pilot_ids)
