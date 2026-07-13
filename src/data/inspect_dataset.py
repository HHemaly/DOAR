"""
inspect_dataset.py — inspect the Combined_Drawing dataset BEFORE any modelling.

Auto-discovers class folders (label-agnostic), measures every image, detects
corrupted files and exact/near duplicates, and writes the CSVs + JSON + figures
the thesis requires. It NEVER fabricates statistics — everything is measured from
the actual files on disk.

Outputs (under <out>/dataset_analysis/):
    dataset_summary.csv       one row per image (path, class, w, h, ar, blur, ...)
    class_distribution.csv    images per class
    duplicates.csv            exact + near-duplicate groups (perceptual hash)
    corrupted_files.csv       files that failed to load
    dataset_statistics.json   aggregate stats
    figures/                  class distribution, w/h/ar/quality distributions
"""

from __future__ import annotations
import os
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_images(dataset_root: str) -> list[dict]:
    """
    Recursively find images. The immediate parent folder name is the class.
    Returns list of {path, class, filename}.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    records = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            records.append({
                "path":     str(p),
                "class":    p.parent.name,
                "filename": p.name,
            })
    return records


def discover_classes(records: list[dict]) -> list[str]:
    return sorted({r["class"] for r in records})


# ---------------------------------------------------------------------------
# Per-image measurement
# ---------------------------------------------------------------------------

def _measure_image(path: str) -> dict | None:
    """Measure one image. Return None if the file is corrupted/unreadable."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        # quality score in [0,1]: blur normalised, penalise tiny images
        quality = min(1.0, blur / 500.0) * (0.5 if (h < 200 or w < 200) else 1.0)
        return {
            "width":      w,
            "height":     h,
            "aspect_ratio": round(w / h, 4) if h else 0.0,
            "blur":       round(blur, 2),
            "brightness": round(brightness, 2),
            "contrast":   round(contrast, 2),
            "quality_score": round(quality, 4),
            "megapixels": round(w * h / 1e6, 4),
        }
    except Exception:
        return None


def _phash(path: str, hash_size: int = 8) -> str | None:
    """Perceptual hash (average-hash) for near-duplicate detection."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize(
            (hash_size, hash_size), Image.BILINEAR
        )
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p > avg else "0" for p in pixels)
        return f"{int(bits, 2):0{hash_size*hash_size//4}x}"
    except Exception:
        return None


def _md5(path: str) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 999


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------

def inspect_dataset(dataset_root: str, out_dir: str,
                    near_dup_threshold: int = 5) -> dict:
    """
    Full dataset inspection. Writes all CSVs/JSON/figures under
    out_dir/dataset_analysis/. Returns the statistics dict.

    near_dup_threshold: max Hamming distance between perceptual hashes to be
    considered a near-duplicate (0 = identical pHash). Default 5 of 64 bits.
    """
    analysis_dir = os.path.join(out_dir, "dataset_analysis")
    fig_dir = os.path.join(analysis_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[inspect] Discovering images under {dataset_root} ...")
    records = discover_images(dataset_root)
    classes = discover_classes(records)
    print(f"[inspect] Found {len(records)} images across {len(classes)} classes.")

    rows = []
    corrupted = []
    phashes = {}
    md5s = defaultdict(list)

    for i, rec in enumerate(records, 1):
        if i % 200 == 0:
            print(f"[inspect]   measured {i}/{len(records)}")
        meas = _measure_image(rec["path"])
        if meas is None:
            corrupted.append(rec)
            continue
        ph = _phash(rec["path"])
        m5 = _md5(rec["path"])
        if ph:
            phashes[rec["path"]] = ph
        if m5:
            md5s[m5].append(rec["path"])
        rows.append({**rec, **meas, "phash": ph or "", "md5": m5 or ""})

    # ── Exact duplicates (same md5) ──────────────────────────────
    exact_groups = [paths for paths in md5s.values() if len(paths) > 1]

    # ── Near duplicates (pHash within threshold, different md5) ──
    near_pairs = []
    items = list(phashes.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            p1, h1 = items[i]
            p2, h2 = items[j]
            d = _hamming(h1, h2)
            if 0 < d <= near_dup_threshold:
                near_pairs.append((p1, p2, d))

    # ── Write dataset_summary.csv ────────────────────────────────
    summary_csv = os.path.join(analysis_dir, "dataset_summary.csv")
    _write_csv(summary_csv, rows, [
        "path", "class", "filename", "width", "height", "aspect_ratio",
        "blur", "brightness", "contrast", "quality_score", "megapixels",
        "phash", "md5",
    ])

    # ── class_distribution.csv ───────────────────────────────────
    class_counts = defaultdict(int)
    for r in rows:
        class_counts[r["class"]] += 1
    class_rows = [{"class": c, "count": n} for c, n in sorted(class_counts.items())]
    _write_csv(os.path.join(analysis_dir, "class_distribution.csv"),
               class_rows, ["class", "count"])

    # ── duplicates.csv ───────────────────────────────────────────
    dup_rows = []
    for g in exact_groups:
        for p in g:
            dup_rows.append({"type": "exact", "path": p,
                             "group_key": g[0], "distance": 0})
    for p1, p2, d in near_pairs:
        dup_rows.append({"type": "near", "path": p2,
                         "group_key": p1, "distance": d})
    _write_csv(os.path.join(analysis_dir, "duplicates.csv"),
               dup_rows, ["type", "path", "group_key", "distance"])

    # ── corrupted_files.csv ──────────────────────────────────────
    _write_csv(os.path.join(analysis_dir, "corrupted_files.csv"),
               corrupted, ["path", "class", "filename"])

    # ── Aggregate statistics ─────────────────────────────────────
    widths  = [r["width"]  for r in rows]
    heights = [r["height"] for r in rows]
    quals   = [r["quality_score"] for r in rows]
    n_valid = len(rows)

    def _stats(vals):
        if not vals:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        s = sorted(vals)
        return {
            "min": min(vals), "max": max(vals),
            "mean": round(sum(vals) / len(vals), 3),
            "median": s[len(s) // 2],
        }

    imbalance_ratio = (max(class_counts.values()) / min(class_counts.values())
                       if class_counts else 0)

    stats = {
        "dataset_root":       dataset_root,
        "total_files_found":  len(records),
        "valid_images":       n_valid,
        "corrupted_images":   len(corrupted),
        "num_classes":        len(classes),
        "classes":            classes,
        "images_per_class":   dict(sorted(class_counts.items())),
        "class_imbalance_ratio": round(imbalance_ratio, 2),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_images": sum(len(g) for g in exact_groups),
        "near_duplicate_pairs":   len(near_pairs),
        "width_stats":        _stats(widths),
        "height_stats":       _stats(heights),
        "quality_stats":      _stats(quals),
        "near_dup_threshold": near_dup_threshold,
    }
    with open(os.path.join(analysis_dir, "dataset_statistics.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Figures ──────────────────────────────────────────────────
    _make_figures(fig_dir, rows, class_counts)

    print(f"[inspect] Done. Outputs in {analysis_dir}")
    print(f"[inspect]   valid={n_valid} corrupted={len(corrupted)} "
          f"exact_dupes={stats['exact_duplicate_images']} "
          f"near_dupe_pairs={len(near_pairs)} imbalance={imbalance_ratio:.1f}x")
    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _make_figures(fig_dir: str, rows: list[dict], class_counts: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[inspect] matplotlib not available; skipping figures.")
        return

    def _save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Class distribution
    if class_counts:
        fig, ax = plt.subplots(figsize=(9, 5))
        items = sorted(class_counts.items())
        bars = ax.bar([c for c, _ in items], [n for _, n in items],
                      color="#3498db", edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Images per Class", fontsize=13, fontweight="bold")
        ax.set_xlabel("Class"); ax.set_ylabel("Number of images")
        plt.xticks(rotation=30, ha="right")
        _clean(ax); _save(fig, "class_distribution")

    # Width / height / aspect-ratio / quality distributions
    for key, title, color in [
        ("width",  "Image Width Distribution (px)",  "#2980b9"),
        ("height", "Image Height Distribution (px)", "#27ae60"),
        ("aspect_ratio", "Aspect-Ratio Distribution (w/h)", "#8e44ad"),
        ("quality_score", "Image-Quality Score Distribution", "#e67e22"),
    ]:
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(vals, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(key.replace("_", " ")); ax.set_ylabel("Count")
        _clean(ax); _save(fig, f"{key}_distribution")

    # Class imbalance (horizontal, sorted)
    if class_counts:
        fig, ax = plt.subplots(figsize=(9, 5))
        items = sorted(class_counts.items(), key=lambda x: x[1])
        bars = ax.barh([c for c, _ in items], [n for _, n in items],
                       color="#e74c3c", edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Class Imbalance (sorted)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of images")
        _clean(ax); _save(fig, "class_imbalance")


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
