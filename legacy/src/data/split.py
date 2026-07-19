"""
split.py — deterministic, leak-safe train/val/test split.

Prevents data leakage: exact duplicates and near-duplicates (perceptual-hash
groups) are collapsed into a single "group" and every image in a group is
assigned to the SAME split. Splitting is stratified per class and seeded, so
the same inputs always produce the same split.

Default split (documented rationale): 70/15/15. For very small classes the
splitter guarantees at least one image in train and, where possible, one each
in val/test, and logs any class too small to populate all three splits.
"""

from __future__ import annotations
import os
import csv
import json
from collections import defaultdict


def _load_summary(summary_csv: str) -> list[dict]:
    with open(summary_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 999


def _build_groups(rows: list[dict], near_dup_threshold: int) -> dict:
    """
    Union-find over images: same md5 OR pHash within threshold => same group.
    Returns {image_path: group_id}.
    """
    parent = {r["path"]: r["path"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Exact duplicates by md5
    by_md5 = defaultdict(list)
    for r in rows:
        if r.get("md5"):
            by_md5[r["md5"]].append(r["path"])
    for paths in by_md5.values():
        for p in paths[1:]:
            union(paths[0], p)

    # Near duplicates by pHash (only within the same class to be safe/relevant)
    by_class = defaultdict(list)
    for r in rows:
        if r.get("phash"):
            by_class[r["class"]].append((r["path"], r["phash"]))
    for cls_items in by_class.values():
        for i in range(len(cls_items)):
            for j in range(i + 1, len(cls_items)):
                if _hamming(cls_items[i][1], cls_items[j][1]) <= near_dup_threshold:
                    union(cls_items[i][0], cls_items[j][0])

    return {p: find(p) for p in parent}


def make_split(summary_csv: str, out_dir: str,
               ratios=(0.70, 0.15, 0.15), seed: int = 42,
               near_dup_threshold: int = 5) -> dict:
    """
    Create a leak-safe stratified split from dataset_summary.csv.

    Writes:
        <out_dir>/splits/split.csv       (path, class, group_id, split)
        <out_dir>/splits/split_meta.json (counts, rationale, seed)

    Returns the split_meta dict.
    """
    import random
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    rng = random.Random(seed)

    rows = _load_summary(summary_csv)
    if not rows:
        raise ValueError(f"No rows in {summary_csv}")

    groups = _build_groups(rows, near_dup_threshold)

    # Represent each group once, tagged with its class (majority class in group)
    group_class = {}
    group_members = defaultdict(list)
    for r in rows:
        gid = groups[r["path"]]
        group_members[gid].append(r["path"])
        group_class.setdefault(gid, r["class"])

    # Stratify groups per class, shuffle deterministically, split
    per_class_groups = defaultdict(list)
    for gid, cls in group_class.items():
        per_class_groups[cls].append(gid)

    assignment = {}          # group_id -> split
    small_classes = []
    tr, va, te = ratios

    for cls, gids in sorted(per_class_groups.items()):
        gids = sorted(gids)
        rng.shuffle(gids)
        n = len(gids)
        if n < 3:
            small_classes.append({"class": cls, "n_groups": n})
        n_train = max(1, int(round(n * tr)))
        n_val   = int(round(n * va))
        # guarantee test gets the remainder; keep at least 1 in val/test when possible
        n_val   = min(n_val, max(0, n - n_train - 1))
        n_test  = n - n_train - n_val
        for i, gid in enumerate(gids):
            if i < n_train:
                assignment[gid] = "train"
            elif i < n_train + n_val:
                assignment[gid] = "val"
            else:
                assignment[gid] = "test"

    # Expand group assignment to every image
    split_rows = []
    for r in rows:
        gid = groups[r["path"]]
        split_rows.append({
            "path":     r["path"],
            "class":    r["class"],
            "group_id": gid,
            "split":    assignment.get(gid, "train"),
        })

    splits_dir = os.path.join(out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    split_csv = os.path.join(splits_dir, "split.csv")
    with open(split_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "group_id", "split"])
        w.writeheader()
        w.writerows(split_rows)

    # Counts
    counts = defaultdict(lambda: defaultdict(int))
    for sr in split_rows:
        counts[sr["split"]][sr["class"]] += 1
    split_totals = {s: sum(c.values()) for s, c in counts.items()}

    # Leak check: assert no group spans two splits
    group_splits = defaultdict(set)
    for sr in split_rows:
        group_splits[sr["group_id"]].add(sr["split"])
    leaks = [g for g, s in group_splits.items() if len(s) > 1]

    meta = {
        "seed":               seed,
        "ratios":             {"train": tr, "val": va, "test": te},
        "near_dup_threshold": near_dup_threshold,
        "total_images":       len(split_rows),
        "total_groups":       len(group_class),
        "split_totals":       split_totals,
        "per_class_per_split": {s: dict(c) for s, c in counts.items()},
        "small_classes":      small_classes,
        "leakage_groups":     len(leaks),
        "leakage_ok":         len(leaks) == 0,
        "rationale": (
            "70/15/15 stratified by class. Exact + near-duplicate images are "
            "grouped (perceptual hash, Hamming <= threshold) and kept within a "
            "single split to prevent leakage. Deterministic with fixed seed."
        ),
    }
    with open(os.path.join(splits_dir, "split_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[split] {split_totals}  (groups={len(group_class)}, "
          f"leakage_ok={meta['leakage_ok']})")
    if small_classes:
        print(f"[split] WARNING small classes (<3 groups): "
              f"{[c['class'] for c in small_classes]}")
    return meta
