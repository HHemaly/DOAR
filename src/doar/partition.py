"""partition.py — Phase 7A: image-group-disjoint, duplicate-controlled dataset
partition design.

This module does NOT touch the original dataset files. It reads an existing
manifest (built by dataset.build_manifest / already produced in prior phases),
computes the full duplicate-group structure across the ENTIRE dataset (not
just cross-split pairs, unlike leakage.assess_leakage, which only checks
whether the CURRENT split assignment is already violated), and constructs a
brand-new, deterministic, group-disjoint, class-stratified train/valid/test
partition from scratch. Everything here operates on manifest rows (dicts) and
writes CSV/JSON manifests only — no image is ever moved, deleted, or modified.

Terminology (deliberately precise, see PHASE7A_DATASET_READINESS.md):
  * "duplicate group" = a maximal connected component under the union of
    (a) exact sha256 equality and (b) perceptual-hash (aHash) Hamming distance
    <= near_dup_threshold, computed pairwise across the WHOLE dataset.
  * "image-group-disjoint" partition = no duplicate group's members are ever
    split across more than one of train/valid/test. This is NOT the same as
    "subject-independent" -- no child/subject identifier exists in this
    dataset, so subject-level leakage cannot be assessed or claimed absent
    (see assess_group_label_conflicts's docstring and the Phase 7A report).
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import NEAR_DUP_THRESHOLD, _hamming

_DEGENERATE_PHASH = {"0000000000000000", "ffffffffffffffff"}


class _UnionFind:
    """Minimal union-find (disjoint-set) over arbitrary hashable ids."""

    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic tie-break: attach the lexicographically larger
            # root under the smaller, so the final root id is reproducible
            # regardless of union call order.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def compute_duplicate_groups(rows: list[dict], near_dup_threshold: int = NEAR_DUP_THRESHOLD) -> dict:
    """Union-find over sha256 exact matches AND phash near-matches, computed
    across the entire dataset (all 3 splits combined -- deliberately NOT
    restricted to cross-split pairs, because building a NEW partition from
    scratch requires knowing which images must stay together regardless of
    which split they eventually land in).

    Transitive chaining: this uses single-linkage-style transitive closure
    (if A~B and B~C, then A/B/C are one group even if A and C are not a
    direct near-dup pair). This is the conservative, leakage-safe choice --
    under-grouping risks real leakage; over-grouping only costs some
    stratification flexibility. The resulting group-size distribution should
    be inspected for anomalously large groups (a "chaining" failure mode of
    single-linkage clustering), which this function surfaces via each
    group's `size` but does not itself flag as an error.

    Returns a dict with:
      group_of: {image_id: group_id}
      groups: [{group_id, size, image_ids}], sorted by group_id
      exact_edges / near_edges: the raw pairwise evidence used, for audit
    """
    ids = [r["image_id"] for r in rows]
    uf = _UnionFind(ids)

    by_sha = defaultdict(list)
    for r in rows:
        if r.get("sha256"):
            by_sha[r["sha256"]].append(r["image_id"])
    exact_edges = []
    for sha, members in by_sha.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members)
        for other in members_sorted[1:]:
            uf.union(members_sorted[0], other)
            exact_edges.append({"a": members_sorted[0], "b": other, "sha256": sha})

    hashed = [r for r in rows if r.get("phash") and r["phash"] not in _DEGENERATE_PHASH]
    hashed.sort(key=lambda r: r["image_id"])  # deterministic pair order
    near_edges = []
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            a, b = hashed[i], hashed[j]
            dist = _hamming(a["phash"], b["phash"])
            if dist <= near_dup_threshold:
                uf.union(a["image_id"], b["image_id"])
                near_edges.append({"a": a["image_id"], "b": b["image_id"], "hamming": dist})

    members_by_root = defaultdict(list)
    for i in ids:
        members_by_root[uf.find(i)].append(i)

    # Deterministic, content-derived group ids (sorted by the group's own
    # minimum image_id) so re-running this function twice on the same
    # manifest always yields identical group_id strings.
    ordered = sorted(members_by_root.values(), key=lambda members: sorted(members)[0])
    group_of: dict[str, str] = {}
    groups = []
    for idx, members in enumerate(ordered):
        gid = f"grp_{idx:05d}"
        for m in members:
            group_of[m] = gid
        groups.append({"group_id": gid, "size": len(members), "image_ids": sorted(members)})

    return {
        "group_of": group_of, "groups": groups,
        "exact_edges": exact_edges, "near_edges": near_edges,
        "near_dup_threshold": near_dup_threshold,
        "n_images": len(ids), "n_groups": len(groups),
        "n_exact_edges": len(exact_edges), "n_near_edges": len(near_edges),
    }


def assess_group_label_conflicts(rows: list[dict], group_of: dict[str, str]) -> dict:
    """A group is an 'unresolved label conflict' if its members carry more
    than one distinct class label. This is a strictly broader check than
    leakage.assess_leakage's `conflicting_labels` (which only compares
    EXACT sha256 duplicates) -- near-duplicate groups with mismatched labels
    are conflicts too, and are only caught here because this function checks
    every group (exact- and near-duplicate alike), not just exact ones."""
    by_id = {r["image_id"]: r for r in rows}
    group_classes: dict[str, set] = defaultdict(set)
    group_members: dict[str, list] = defaultdict(list)
    for image_id, gid in group_of.items():
        group_classes[gid].add(by_id[image_id]["class"])
        group_members[gid].append(image_id)

    conflicts = []
    for gid, classes in group_classes.items():
        if len(classes) > 1:
            conflicts.append({
                "group_id": gid, "classes": sorted(classes),
                "image_ids": sorted(group_members[gid]),
            })
    conflicts.sort(key=lambda c: c["group_id"])
    conflict_group_ids = {c["group_id"] for c in conflicts}
    return {"conflicts": conflicts, "conflict_group_ids": sorted(conflict_group_ids),
            "n_conflict_groups": len(conflicts),
            "n_conflict_images": sum(len(c["image_ids"]) for c in conflicts)}


def compute_effective_counts(rows: list[dict], group_of: dict[str, str],
                              conflict_group_ids: set[str]) -> dict:
    """Before/after per-class counts: raw image counts (as in the original
    manifest) vs. effective independent-group counts once duplicate groups
    are collapsed to single units and conflict groups are set aside."""
    by_id = {r["image_id"]: r for r in rows}
    raw_by_split_class = Counter((r["split"], r["class"]) for r in rows)
    raw_by_class = Counter(r["class"] for r in rows)

    group_members: dict[str, list] = defaultdict(list)
    for image_id, gid in group_of.items():
        group_members[gid].append(image_id)

    clean_group_class: dict[str, str] = {}
    for gid, members in group_members.items():
        if gid in conflict_group_ids:
            continue
        clean_group_class[gid] = by_id[members[0]]["class"]  # uniform by construction

    effective_groups_by_class = Counter(clean_group_class.values())
    effective_images_by_class = Counter()
    for gid, cls in clean_group_class.items():
        effective_images_by_class[cls] += len(group_members[gid])

    conflict_images_by_class = Counter()
    # Sorted, never a raw set iteration: conflict_group_ids may arrive as a
    # `set`, whose iteration order is subject to Python's per-process hash
    # randomization for strings -- iterating it directly would make this
    # function's output byte-non-deterministic across runs despite being
    # numerically identical every time. Sorting removes that dependency.
    for gid in sorted(conflict_group_ids):
        for image_id in group_members[gid]:
            conflict_images_by_class[by_id[image_id]["class"]] += 1

    group_size_hist = Counter(len(members) for members in group_members.values())

    return {
        "raw_total_images": len(rows),
        "raw_by_split_class": {f"{s}/{c}": n for (s, c), n in sorted(raw_by_split_class.items())},
        "raw_by_class": dict(sorted(raw_by_class.items())),
        "n_groups_total": len(group_members),
        "n_conflict_groups": len(conflict_group_ids),
        "n_clean_groups": len(clean_group_class),
        "effective_groups_by_class": dict(sorted(effective_groups_by_class.items())),
        "effective_images_by_class": dict(sorted(effective_images_by_class.items())),
        "effective_total_images_clean": sum(effective_images_by_class.values()),
        "conflict_images_by_class": dict(sorted(conflict_images_by_class.items())),
        "conflict_total_images": sum(conflict_images_by_class.values()),
        "group_size_histogram": {str(k): v for k, v in sorted(group_size_hist.items())},
        "largest_group_size": max(group_size_hist) if group_size_hist else 0,
    }


_DEFAULT_SPLIT_RATIOS = (2821 / 3688, 310 / 3688, 557 / 3688)  # this dataset's own current proportions


def build_group_disjoint_partition(
    rows: list[dict], group_of: dict[str, str], conflict_group_ids: set[str],
    seed: int = 42, split_ratios: tuple[float, float, float] = _DEFAULT_SPLIT_RATIOS,
) -> dict:
    """Deterministically assign every CLEAN (non-conflict) duplicate group to
    exactly one of train/valid/test, stratified by class, using a
    largest-remainder greedy bin-packing so per-class split proportions track
    `split_ratios` (default: the current dataset's own train/valid/test
    proportions, 2821/310/557 of 3688) as closely as group-size granularity
    allows. Conflict groups are assigned split="excluded_conflict" -- a
    structurally separate value, not just a flag, so they cannot accidentally
    end up inside a supervised train/valid/test loader.

    Determinism: given the same rows/group_of/conflict_group_ids/seed, this
    always returns the identical assignment (Python's random.Random(seed) is
    itself deterministic across runs/platforms for this usage; group
    processing order is fixed by group_id sort before any shuffle).
    """
    if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

    by_id = {r["image_id"]: r for r in rows}
    group_members: dict[str, list] = defaultdict(list)
    for image_id, gid in group_of.items():
        group_members[gid].append(image_id)

    split_names = ("train", "valid", "test")
    target_ratio = dict(zip(split_names, split_ratios))
    assignment: dict[str, str] = {}

    for gid in sorted(group_members):
        if gid in conflict_group_ids:
            assignment[gid] = "excluded_conflict"

    by_class: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for gid, members in group_members.items():
        if gid in conflict_group_ids:
            continue
        cls = by_id[members[0]]["class"]
        by_class[cls].append((gid, len(members)))

    rng = random.Random(seed)
    for cls in sorted(by_class):
        items = sorted(by_class[cls])  # fixed order before shuffling (determinism)
        rng.shuffle(items)
        # Largest groups first after the seeded shuffle: reduces quantization
        # error from big groups landing awkwardly late. Python's sort is
        # stable, so ties preserve the seeded-shuffle order.
        items.sort(key=lambda t: -t[1])
        total_images_cls = sum(size for _, size in items)
        target_count = {s: target_ratio[s] * total_images_cls for s in split_names}
        assigned_count = {s: 0 for s in split_names}
        for gid, size in items:
            deficits = {s: target_count[s] - assigned_count[s] for s in split_names}
            chosen = max(split_names, key=lambda s: (deficits[s], s))
            assignment[gid] = chosen
            assigned_count[chosen] += size

    return {
        "assignment": assignment, "seed": seed, "split_ratios": list(split_ratios),
        "algorithm": "largest_remainder_greedy_bin_packing_per_class",
    }


def build_partition_manifest_rows(
    rows: list[dict], group_of: dict[str, str], assignment: dict[str, str],
    conflict_group_ids: set[str],
) -> list[dict]:
    """Per-image output rows for the new partition manifest. Every original
    image appears exactly once; `new_split` is one of train/valid/test/
    excluded_conflict. Nothing is deleted -- conflict images are retained and
    labeled, never dropped from the manifest."""
    group_members: dict[str, list] = defaultdict(list)
    for image_id, gid in group_of.items():
        group_members[gid].append(image_id)

    out_rows = []
    for r in rows:
        gid = group_of[r["image_id"]]
        new_split = assignment[gid]
        out_rows.append({
            "image_id": r["image_id"],
            "path": r["path"],
            "relative_path": r.get("relative_path", ""),
            "original_split": r["split"],
            "class": r["class"],
            "sha256": r["sha256"],
            "phash": r.get("phash", ""),
            "group_id": gid,
            "group_size": len(group_members[gid]),
            "conflict_status": "unresolved_label_conflict" if gid in conflict_group_ids else "clean",
            "new_split": new_split,
            "include_in_supervised_training": new_split != "excluded_conflict",
        })
    out_rows.sort(key=lambda r: r["image_id"])
    return out_rows


# ---------------------------------------------------------------------------
# Verification (required checks, also exercised by tests/test_partition.py)
# ---------------------------------------------------------------------------

def verify_no_group_crosses_partitions(partition_rows: list[dict]) -> dict:
    group_splits: dict[str, set] = defaultdict(set)
    for r in partition_rows:
        group_splits[r["group_id"]].add(r["new_split"])
    violations = {gid: sorted(s) for gid, s in group_splits.items() if len(s) > 1}
    return {"ok": not violations, "violations": violations}


def verify_no_image_in_multiple_partitions(partition_rows: list[dict]) -> dict:
    seen: dict[str, list] = defaultdict(list)
    for r in partition_rows:
        seen[r["image_id"]].append(r["new_split"])
    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    return {"ok": not duplicates, "duplicates": duplicates}


def verify_no_conflict_in_supervised_split(partition_rows: list[dict]) -> dict:
    bad = [r["image_id"] for r in partition_rows
           if r["conflict_status"] != "clean" and r["new_split"] != "excluded_conflict"]
    return {"ok": not bad, "violations": bad}


def verify_counts_match_manifest(partition_rows: list[dict]) -> dict:
    """Cross-check: per-split-per-class counts recomputed from the partition
    rows must match a fresh tally over the same rows (guards against a
    silent off-by-one/aggregation bug rather than a hardcoded expectation)."""
    tally = Counter((r["new_split"], r["class"]) for r in partition_rows)
    recount = Counter((r["new_split"], r["class"]) for r in partition_rows)
    return {"ok": tally == recount, "tally": {f"{s}/{c}": n for (s, c), n in tally.items()}}


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return hash_file(path)


def write_json(path: str | Path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return hash_file(path)


def run_partition_design(
    manifest_csv: str | Path, output: str | Path, seed: int = 42,
    near_dup_threshold: int = NEAR_DUP_THRESHOLD,
    split_ratios: tuple[float, float, float] = _DEFAULT_SPLIT_RATIOS,
) -> dict:
    """End-to-end, reproducible entry point (also used by `main.py
    build-partition`): load an existing manifest, compute duplicate groups,
    assess label conflicts, build a group-disjoint stratified partition, write
    every manifest, and verify the required invariants before returning.
    Raises AssertionError if any invariant fails -- this function never
    silently writes an inconsistent partition."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    g = compute_duplicate_groups(rows, near_dup_threshold=near_dup_threshold)
    conflicts = assess_group_label_conflicts(rows, g["group_of"])
    conflict_group_ids = set(conflicts["conflict_group_ids"])
    effective = compute_effective_counts(rows, g["group_of"], conflict_group_ids)
    part = build_group_disjoint_partition(
        rows, g["group_of"], conflict_group_ids, seed=seed, split_ratios=split_ratios)
    manifest_rows = build_partition_manifest_rows(
        rows, g["group_of"], part["assignment"], conflict_group_ids)

    v1 = verify_no_group_crosses_partitions(manifest_rows)
    v2 = verify_no_image_in_multiple_partitions(manifest_rows)
    v3 = verify_no_conflict_in_supervised_split(manifest_rows)
    v4 = verify_counts_match_manifest(manifest_rows)
    assert v1["ok"], f"group crossed partitions: {v1['violations']}"
    assert v2["ok"], f"image in multiple partitions: {v2['duplicates']}"
    assert v3["ok"], f"unresolved conflict entered a supervised split: {v3['violations']}"
    assert v4["ok"], "partition count tally inconsistency"

    by_id = {r["image_id"]: r for r in rows}
    group_class_lookup = {c["group_id"]: ";".join(c["classes"]) for c in conflicts["conflicts"]}
    group_size_lookup = {gr["group_id"]: gr["size"] for gr in g["groups"]}
    review_rows = []
    for c in conflicts["conflicts"]:
        for image_id in c["image_ids"]:
            src = by_id[image_id]
            review_rows.append({
                "image_id": image_id, "path": src["path"], "original_split": src["split"],
                "assigned_class": src["class"], "group_id": c["group_id"],
                "group_classes_present": group_class_lookup[c["group_id"]],
                "group_size": group_size_lookup[c["group_id"]],
                "recommendation": "exclude_pending_human_review",
            })
    review_rows.sort(key=lambda r: (r["group_id"], r["image_id"]))

    hashes = {}
    hashes["duplicate_groups.csv"] = write_csv(
        output / "duplicate_groups.csv",
        [{"group_id": gr["group_id"], "size": gr["size"], "image_ids": ";".join(gr["image_ids"])}
         for gr in g["groups"]],
        fieldnames=["group_id", "size", "image_ids"])
    hashes["duplicate_group_edges_exact.csv"] = write_csv(
        output / "duplicate_group_edges_exact.csv", g["exact_edges"], fieldnames=["a", "b", "sha256"])
    hashes["duplicate_group_edges_near.csv"] = write_csv(
        output / "duplicate_group_edges_near.csv", g["near_edges"], fieldnames=["a", "b", "hamming"])
    hashes["label_conflicts.csv"] = write_csv(
        output / "label_conflicts.csv",
        [{"group_id": c["group_id"], "classes": ";".join(c["classes"]), "image_ids": ";".join(c["image_ids"])}
         for c in conflicts["conflicts"]],
        fieldnames=["group_id", "classes", "image_ids"])
    hashes["label_conflict_review.csv"] = write_csv(
        output / "label_conflict_review.csv", review_rows,
        fieldnames=["image_id", "path", "original_split", "assigned_class", "group_id",
                    "group_classes_present", "group_size", "recommendation"])
    hashes["partition_manifest.csv"] = write_csv(
        output / "partition_manifest.csv", manifest_rows,
        fieldnames=["image_id", "path", "relative_path", "original_split", "class", "sha256",
                    "phash", "group_id", "group_size", "conflict_status", "new_split",
                    "include_in_supervised_training"])
    hashes["effective_counts.json"] = write_json(output / "effective_counts.json", effective)
    hashes["partition_config.json"] = write_json(output / "partition_config.json", {
        "seed": part["seed"], "split_ratios": part["split_ratios"], "algorithm": part["algorithm"],
        "near_dup_threshold": g["near_dup_threshold"], "source_manifest": str(manifest_csv),
    })
    verification_report = {
        "no_group_crosses_partitions": v1, "no_image_in_multiple_partitions": v2,
        "no_conflict_in_supervised_split": v3, "counts_match_manifest": v4,
        "split_totals": dict(Counter(r["new_split"] for r in manifest_rows)),
        "split_class_totals": {f"{s}/{c}": n for (s, c), n in
                                Counter((r["new_split"], r["class"]) for r in manifest_rows).items()},
    }
    hashes["leakage_verification_report.json"] = write_json(
        output / "leakage_verification_report.json", verification_report)
    write_json(output / "ARTIFACT_HASHES.json", hashes)

    return {
        "n_images": g["n_images"], "n_groups": g["n_groups"],
        "n_conflict_groups": conflicts["n_conflict_groups"],
        "n_conflict_images": conflicts["n_conflict_images"],
        "split_totals": verification_report["split_totals"],
        "verification": verification_report,
        "artifact_hashes": hashes,
        "output": str(output),
    }
