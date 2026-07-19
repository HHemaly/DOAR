from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

CLASSES = ("Angry", "Fear", "Happy", "Sad")
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Perceptual (average) hash size and the max Hamming distance still counted as a
# near-duplicate. 64-bit hash; <=5 bits differing is a conservative near-dup.
_PHASH_SIZE = 8
NEAR_DUP_THRESHOLD = 5


def _average_hash(path: Path) -> str | None:
    """Dependency-light perceptual average-hash for near-duplicate detection."""
    try:
        with Image.open(path) as image:
            small = image.convert("L").resize((_PHASH_SIZE, _PHASH_SIZE), Image.BILINEAR)
            pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p > avg else "0" for p in pixels)
        return f"{int(bits, 2):016x}"
    except Exception:
        return None


def _hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 999


def build_manifest(dataset: str | Path, output_csv: str | Path) -> dict:
    root = Path(dataset).resolve()
    rows = []
    counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    hashes: dict[str, list[str]] = {}
    for split in SPLITS:
        for cls in CLASSES:
            folder = root / split / cls
            if not folder.exists():
                continue
            for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS):
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                readable, width, height = True, None, None
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                        image.verify()
                except Exception:
                    readable = False
                image_id = hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:16]
                phash = _average_hash(path) if readable else None
                row = {
                    "image_id": image_id,
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "split": split,
                    "class": cls,
                    "sha256": digest,
                    "phash": phash or "",
                    "width": width,
                    "height": height,
                    "file_size": len(raw),
                    "readable": readable,
                    "provenance": "physical_existing_split",
                }
                rows.append(row)
                counts[split][cls] += 1
                hashes.setdefault(digest, []).append({"image_id": image_id, "split": split})
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "image_id", "path", "relative_path", "split", "class", "sha256",
        "width", "height", "file_size", "readable", "provenance",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # ── Duplicate + cross-split leakage analysis ──────────────────────────
    exact_groups = [members for members in hashes.values() if len(members) > 1]

    # Exact duplicates that span more than one split => hard leakage.
    cross_split_exact = []
    for members in exact_groups:
        splits_involved = sorted({m["split"] for m in members})
        if len(splits_involved) > 1:
            cross_split_exact.append({
                "image_ids": [m["image_id"] for m in members],
                "splits": splits_involved,
            })

    # Near-duplicates across splits (perceptual hash within threshold).
    # Exclude degenerate hashes from flat/near-constant images (all-0 / all-f):
    # average-hash cannot distinguish flat images and would false-positive.
    _degenerate = {"0000000000000000", "ffffffffffffffff"}
    hashed_rows = [r for r in rows if r.get("phash") and r["phash"] not in _degenerate]
    cross_split_near = []
    for i in range(len(hashed_rows)):
        for j in range(i + 1, len(hashed_rows)):
            a, b = hashed_rows[i], hashed_rows[j]
            if a["split"] == b["split"]:
                continue
            dist = _hamming(a["phash"], b["phash"])
            if dist <= NEAR_DUP_THRESHOLD:
                cross_split_near.append({
                    "image_ids": [a["image_id"], b["image_id"]],
                    "splits": sorted({a["split"], b["split"]}),
                    "hamming": dist,
                })

    leakage_ok = not cross_split_exact and not cross_split_near

    summary = {
        "dataset_root": str(root),
        "counts": counts,
        "total": len(rows),
        "exact_duplicate_groups": [[m["image_id"] for m in members] for members in exact_groups],
        "cross_split_exact_leakage": cross_split_exact,
        "cross_split_near_duplicate_leakage": cross_split_near,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "leakage_ok": leakage_ok,
        "leakage_status": "PASS" if leakage_ok else "FAIL_LEAKAGE_DETECTED",
        "split_names": list(SPLITS),
        "test_locked": True,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
