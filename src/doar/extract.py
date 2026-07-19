from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path

from .analysis import analyze_image
from .features import objective_feature_row, serialize_feature_row


def extract_features(manifest: str | Path, output: str | Path) -> dict:
    output = Path(output)
    cases = output / "cases"
    output.mkdir(parents=True, exist_ok=True)
    with open(manifest, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    flat_rows, failures, schema = [], [], {}
    for record in manifest_rows:
        if record.get("readable", "").lower() not in ("true", "1"):
            failures.append({"image_id": record["image_id"], "path": record["path"], "error": "manifest_unreadable"})
            continue
        try:
            analysis = analyze_image(record["path"], cases / record["image_id"])
            feature_row = objective_feature_row(record["path"], analysis.to_dict())
            structured = serialize_feature_row(feature_row)
            (cases / record["image_id"] / "features.json").write_text(
                json.dumps(structured, indent=2), encoding="utf-8"
            )
            flat = {
                "image_id": record["image_id"],
                "path": record["path"],
                "split": record["split"],
                "class": record["class"],
                **{name: item.value for name, item in feature_row.items()},
            }
            flat_rows.append(flat)
            schema.update({name: {
                "version": item.version, "method": item.method,
                "valid_min": item.valid_min, "valid_max": item.valid_max,
            } for name, item in feature_row.items()})
        except Exception as exc:
            failures.append({
                "image_id": record.get("image_id"), "path": record.get("path"),
                "error": str(exc), "traceback": traceback.format_exc(),
            })
    metadata_fields = ["image_id", "path", "split", "class"]
    fields = metadata_fields + sorted(schema)
    with (output / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    with (output / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "path", "error", "traceback"])
        writer.writeheader()
        writer.writerows(failures)
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    missing = {
        name: sum(not str(row.get(name, "")).strip() or str(row.get(name)).lower() == "nan" for row in flat_rows)
        for name in schema
    }
    summary = {
        "manifest": str(Path(manifest).resolve()),
        "processed": len(flat_rows),
        "failures": len(failures),
        "feature_count": len(schema),
        "splits": sorted({row["split"] for row in flat_rows}),
        "test_features_extracted_but_never_used_for_training": True,
        "missing_values": missing,
    }
    (output / "extraction_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
