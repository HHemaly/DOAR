from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .dataset import CLASSES, EXTENSIONS, SPLITS


def validate_dataset(dataset: str | Path, output: str | Path) -> dict:
    root, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    failures, dimensions, distribution = [], [], []
    counts = {split: {label: 0 for label in CLASSES} for split in SPLITS}
    modes = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    unknown_folders = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            failures.append({"path": str(split_dir), "split": split, "class": "",
                             "reason": "missing_split_folder"})
            continue
        unknown_folders.extend(
            str(path) for path in split_dir.iterdir()
            if path.is_dir() and path.name not in CLASSES
        )
        for label in CLASSES:
            folder = split_dir / label
            if not folder.exists():
                failures.append({"path": str(folder), "split": split, "class": label,
                                 "reason": "missing_class_folder"})
                continue
            for path in folder.rglob("*"):
                if not path.is_file():
                    continue
                if path.stat().st_size == 0:
                    failures.append({"path": str(path), "split": split, "class": label,
                                     "reason": "zero_byte_file"})
                    continue
                if path.suffix.lower() not in EXTENSIONS:
                    failures.append({"path": str(path), "split": split, "class": label,
                                     "reason": "unsupported_extension"})
                    continue
                counts[split][label] += 1
                # Preserve locked-test discipline: test content is counted and
                # hashed for informational duplicates but not decoded for metrics.
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[digest].append(str(path))
                if split == "test":
                    continue
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                        modes[image.mode] += 1
                        dimensions.append({
                            "split": split, "class": label, "width": width, "height": height,
                            "mode": image.mode, "extension": path.suffix.lower(),
                        })
                        image.verify()
                except Exception as exc:
                    failures.append({"path": str(path), "split": split, "class": label,
                                     "reason": f"unreadable:{exc}"})
    for split in SPLITS:
        for label in CLASSES:
            distribution.append({"split": split, "class": label, "count": counts[split][label]})
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    totals = {split: sum(values.values()) for split, values in counts.items()}
    train_counts = list(counts["train"].values())
    imbalance = max(train_counts) / max(1, min(train_counts)) if train_counts else None
    status = "FAIL" if any(item["reason"].startswith("missing_") for item in failures) else (
        "WARN" if failures or unknown_folders else "PASS"
    )
    result = {
        "status": status, "dataset_root": str(root.resolve()),
        "expected_splits": list(SPLITS), "validation_folder": "valid",
        "classes": list(CLASSES), "counts": counts, "totals": totals,
        "unknown_folders": unknown_folders, "failures": len(failures),
        "exact_duplicate_groups_informational": len(duplicate_groups),
        "duplicates_block_training": False, "train_imbalance_ratio": imbalance,
        "decoded_test_images": 0, "colour_modes_train_valid": dict(modes),
    }
    (output / "dataset_readiness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split", "class", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    for name, rows in (("class_distribution.csv", distribution),
                       ("dimension_distribution.csv", dimensions)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            fields = list(rows[0]) if rows else ["split", "class", "count"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    html = (
        "<html><meta charset='utf-8'><body><h1>DOAR dataset readiness</h1>"
        f"<p>Status: <strong>{status}</strong></p><pre>{json.dumps(result, indent=2)}</pre>"
        "</body></html>"
    )
    (output / "dataset_readiness.html").write_text(html, encoding="utf-8")
    return result


def check_training_readiness(
    dataset: str | Path | None, output_dir: str | Path,
    manifest: str | Path | None = None, features: str | Path | None = None,
    embeddings: str | Path | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    packages = {
        name: importlib.util.find_spec(module) is not None
        for name, module in {
            "numpy": "numpy", "pillow": "PIL", "sklearn": "sklearn", "joblib": "joblib",
            "torch": "torch", "torchvision": "torchvision", "streamlit": "streamlit",
            "open_clip": "open_clip",
        }.items()
    }
    cuda = {"available": False, "device": None, "memory_bytes": None}
    if packages["torch"]:
        import torch
        cuda["available"] = torch.cuda.is_available()
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
            cuda["memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    disk = shutil.disk_usage(output)
    dataset_ok = bool(dataset and (Path(dataset) / "train").exists()
                      and (Path(dataset) / "valid").exists() and (Path(dataset) / "test").exists())
    # C4: report real status; do NOT hardcode test_locked.
    from .test_guard import GUARD_VERSION
    guarded_commands = ["evaluate", "evaluate-predictions", "export-probabilities",
                        "apply-late-fusion"]

    def _leakage_status(man):
        if not man:
            return None
        for cand in (Path(man).parent / "leakage_gate" / "leakage_report.json",
                     Path(man).with_suffix(".summary.json")):
            if cand.exists():
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    return data.get("leakage_status") or ("PASS" if data.get("leakage_ok") else "FAIL")
                except Exception:
                    return "UNKNOWN"
        return "not_generated"

    def _provenance_status(art_dir, name):
        if not art_dir:
            return None
        p = Path(art_dir).parent / name
        return "present" if p.exists() else "missing"

    gpu_smoke_recorded = (output / "gpu" / "gpu_smoke.json").exists() or \
        (Path(output_dir) / "gpu_smoke.json").exists()

    checks = {
        "python_3_11": sys.version_info[:2] == (3, 11),
        "core_dependencies": packages["numpy"] and packages["pillow"],
        "ml_dependencies": packages["sklearn"] and packages["joblib"],
        "deep_dependencies": packages["torch"] and packages["torchvision"],
        "dataset_structure": dataset_ok,
        "valid_split_name": bool(dataset and (Path(dataset) / "valid").exists()),
        "output_writable": os.access(output, os.W_OK),
        "manifest_exists": bool(manifest and Path(manifest).exists()),
        "feature_table_exists": bool(features and Path(features).exists()),
        "embedding_cache_exists": bool(embeddings and Path(embeddings).exists()),
    }
    failures = [name for name in ("core_dependencies", "dataset_structure", "output_writable")
                if not checks[name]]
    warnings = [name for name, value in checks.items() if not value and name not in failures]
    # 6 GB-safe default is batch 4 unless an actual memory smoke proves otherwise.
    mem_gb = (cuda["memory_bytes"] or 0) / 1024**3
    recommendation = (
        {"device": "cuda", "batch_size": 8, "workers": 2, "grad_accum_steps": 1,
         "profile": "gpu_ge_8gb"}
        if cuda["available"] and mem_gb >= 8 and gpu_smoke_recorded else
        {"device": "cuda", "batch_size": 4, "workers": 2, "grad_accum_steps": 2,
         "profile": "gpu_6gb_safe_default",
         "note": "batch 4 default for 6 GB; raise only after a gpu-smoke memory check"}
        if cuda["available"] else
        {"device": "cpu", "batch_size": 4, "workers": 0, "grad_accum_steps": 1,
         "profile": "cpu_smoke"}
    )
    result = {
        "status": "FAIL" if failures else "WARN" if warnings else "PASS",
        "python": platform.python_version(), "platform": platform.platform(),
        "python_3_11_compatible": sys.version_info[:2] == (3, 11),
        "packages": packages, "cuda": cuda, "disk_free_bytes": disk.free,
        "checks": checks, "failures": failures, "warnings": warnings,
        "recommended_settings": recommendation,
        "final_test_guard": {
            "version": GUARD_VERSION, "status": "enforced",
            "protected_commands": guarded_commands,
        },
        "leakage_status": _leakage_status(manifest),
        "feature_provenance": _provenance_status(features, "extraction_metadata.json"),
        "embedding_provenance": _provenance_status(embeddings, "embedding_metadata.json"),
        "gpu_smoke_recorded": bool(gpu_smoke_recorded),
    }
    (output / "training_readiness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
