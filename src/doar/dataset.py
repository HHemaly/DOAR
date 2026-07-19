from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

CLASSES = ("Angry", "Fear", "Happy", "Sad")
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
                row = {
                    "image_id": image_id,
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "split": split,
                    "class": cls,
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "file_size": len(raw),
                    "readable": readable,
                    "provenance": "physical_existing_split",
                }
                rows.append(row)
                counts[split][cls] += 1
                hashes.setdefault(digest, []).append(image_id)
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
    summary = {
        "dataset_root": str(root),
        "counts": counts,
        "total": len(rows),
        "exact_duplicate_groups": [ids for ids in hashes.values() if len(ids) > 1],
        "split_names": list(SPLITS),
        "test_locked": True,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
