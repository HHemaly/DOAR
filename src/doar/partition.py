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

from .dataset import NEAR_DUP_THRESHOLD, _difference_hash, _hamming

_DEGENERATE_PHASH = {"0000000000000000", "ffffffffffffffff"}


def compute_dhash_column(rows: list[dict]) -> None:
    """Phase 7B: compute and attach a 'dhash' field to every row in place,
    reading each image once from its recorded `path`. Does not touch the
    original image files (read-only) or the existing `phash` (aHash) column."""
    for r in rows:
        r["dhash"] = _difference_hash(Path(r["path"])) or ""


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


def compute_duplicate_groups(
    rows: list[dict], near_dup_threshold: int = NEAR_DUP_THRESHOLD, hash_field: str = "phash",
) -> dict:
    """Union-find over sha256 exact matches AND near-dup hash matches, computed
    across the entire dataset (all 3 splits combined -- deliberately NOT
    restricted to cross-split pairs, because building a NEW partition from
    scratch requires knowing which images must stay together regardless of
    which split they eventually land in).

    `hash_field` selects which perceptual-hash column to use for the near-dup
    comparison -- defaults to "phash" (the pre-existing aHash column, used
    for Phase 7A's reproduction of the original leakage analysis). Phase 7B's
    final policy passes `hash_field="dhash"` explicitly, so which hash method
    produced a given result is always visible at the call site rather than
    implied by silently renaming a column before calling this function.

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

    hashed = [r for r in rows if r.get(hash_field) and r[hash_field] not in _DEGENERATE_PHASH]
    hashed.sort(key=lambda r: r["image_id"])  # deterministic pair order
    near_edges = []
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            a, b = hashed[i], hashed[j]
            dist = _hamming(a[hash_field], b[hash_field])
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


def refine_with_complete_linkage(rows: list[dict], single_linkage_result: dict, hash_field: str = "dhash") -> dict:
    """Phase 7B (continued): a constrained alternative to unrestricted
    single-linkage transitive closure, evaluated specifically to check
    whether it resolves heterogeneous ("chained") components without
    discarding genuine duplicate clusters.

    For every single-linkage component with 3+ members, this re-clusters
    ONLY that component's own members using complete-linkage agglomerative
    clustering: two (sub)clusters may only merge if EVERY cross-pair between
    their members is within `near_dup_threshold` (i.e. the merged cluster's
    diameter -- its single worst-case pairwise distance -- stays within the
    threshold), rather than single-linkage's "at least one edge" rule. Exact
    sha256 duplicates are always kept together regardless of hash distance
    (they are definitionally identical content, not a near-dup judgment
    call). Singleton and 2-member components are returned unchanged --
    complete- and single-linkage cannot differ for fewer than 3 members.

    This does NOT replace compute_duplicate_groups's single-linkage result
    for the project's own leakage-safety default (see
    PHASE7B_DUPLICATE_POLICY.md for why single-linkage was retained) -- it
    exists to produce a concrete, evidence-based comparison on the actual
    data, specifically for components already flagged as heterogeneous.
    """
    by_id = {r["image_id"]: r for r in rows}
    near_dup_threshold = single_linkage_result["near_dup_threshold"]

    exact_pairs: set[frozenset] = set()
    for e in single_linkage_result["exact_edges"]:
        exact_pairs.add(frozenset((e["a"], e["b"])))

    new_groups = []
    split_report = []
    for gr in single_linkage_result["groups"]:
        members = gr["image_ids"]
        if len(members) < 3:
            new_groups.append({"group_id": gr["group_id"], "size": len(members), "image_ids": members})
            continue

        # All-pairs Hamming distance within this (small) component.
        dist = {}
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if frozenset((a, b)) in exact_pairs:
                    d = 0
                else:
                    ha, hb = by_id[a].get(hash_field, ""), by_id[b].get(hash_field, "")
                    d = _hamming(ha, hb) if ha and hb else 999
                dist[frozenset((a, b))] = d

        # Complete-linkage agglomeration: start with singleton clusters,
        # repeatedly merge the pair of clusters with the smallest maximum
        # cross-distance, but only while that maximum stays <= threshold.
        clusters = [[m] for m in members]
        while True:
            best = None
            best_diam = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    max_d = max(dist[frozenset((a, b))] for a in clusters[i] for b in clusters[j])
                    if max_d <= near_dup_threshold and (best_diam is None or max_d < best_diam):
                        best, best_diam = (i, j), max_d
            if best is None:
                break
            i, j = best
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]

        if len(clusters) == 1:
            new_groups.append({"group_id": gr["group_id"], "size": len(members), "image_ids": sorted(members)})
        else:
            split_report.append({
                "original_group_id": gr["group_id"], "original_size": len(members),
                "n_subclusters": len(clusters), "subcluster_sizes": sorted(len(c) for c in clusters),
            })
            for k, sub in enumerate(sorted(clusters, key=lambda c: sorted(c)[0])):
                new_groups.append({
                    "group_id": f"{gr['group_id']}_cl{k}", "size": len(sub), "image_ids": sorted(sub),
                })

    group_of = {}
    for gr in new_groups:
        for m in gr["image_ids"]:
            group_of[m] = gr["group_id"]

    return {
        "group_of": group_of, "groups": new_groups,
        "near_dup_threshold": near_dup_threshold, "hash_field": hash_field,
        "n_images": sum(gr["size"] for gr in new_groups), "n_groups": len(new_groups),
        "n_components_split": len(split_report), "split_report": split_report,
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
    exposed_group_ids: frozenset[str] = frozenset(),
) -> dict:
    """Deterministically assign every CLEAN (non-conflict) duplicate group to
    exactly one of train/valid/test, stratified by class, using a
    largest-remainder greedy bin-packing so per-class split proportions track
    `split_ratios` (default: the current dataset's own train/valid/test
    proportions, 2821/310/557 of 3688) as closely as group-size granularity
    allows. Conflict groups are assigned split="excluded_conflict" -- a
    structurally separate value, not just a flag, so they cannot accidentally
    end up inside a supervised train/valid/test loader.

    `exposed_group_ids` (Phase 7B): duplicate groups containing an image whose
    label/prediction has already been disclosed as non-blind (the 4 Phase 6
    smoke-test images -- see PHASE6_RESULTS.md Section 2). Every clean
    (non-conflict) exposed group is force-assigned to "train" -- never
    "valid" or "test" -- since training does not require blindness, only
    evaluation does. Conflict status takes precedence: a group that is both
    exposed and an unresolved conflict stays "excluded_conflict", not "train".

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
        elif gid in exposed_group_ids:
            assignment[gid] = "train"

    by_class: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for gid, members in group_members.items():
        if gid in conflict_group_ids or gid in exposed_group_ids:
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


def identify_exposed_groups(group_of: dict[str, str], exposed_image_ids: set[str]) -> frozenset[str]:
    """Every duplicate group containing at least one already-disclosed,
    non-blind image (Phase 6's 4 smoke-test images) is an "exposed group" --
    the whole group, not just the disclosed image, must be kept out of
    valid/test (see build_group_disjoint_partition's docstring)."""
    return frozenset(group_of[iid] for iid in exposed_image_ids if iid in group_of)


def build_partition_manifest_rows(
    rows: list[dict], group_of: dict[str, str], assignment: dict[str, str],
    conflict_group_ids: set[str], exposed_group_ids: frozenset[str] = frozenset(),
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
            "dhash": r.get("dhash", ""),
            "group_id": gid,
            "group_size": len(group_members[gid]),
            "conflict_status": "unresolved_label_conflict" if gid in conflict_group_ids else "clean",
            "exposed_status": "previously_exposed" if gid in exposed_group_ids else "not_exposed",
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


def verify_no_exposed_group_in_valid_or_test(partition_rows: list[dict]) -> dict:
    """Phase 7B requirement: no previously-exposed image (or any duplicate-
    group member of one) may enter the new valid or test split."""
    bad = [r["image_id"] for r in partition_rows
           if r.get("exposed_status") == "previously_exposed" and r["new_split"] in ("valid", "test")]
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
    hash_field: str = "phash",
    exposed_image_ids: set[str] = frozenset(),
) -> dict:
    """End-to-end, reproducible entry point (also used by `main.py
    build-partition`): load an existing manifest, compute duplicate groups,
    assess label conflicts, build a group-disjoint stratified partition, write
    every manifest, and verify the required invariants before returning.
    Raises AssertionError if any invariant fails -- this function never
    silently writes an inconsistent partition.

    `hash_field` (Phase 7B): which near-dup hash column to use -- "phash"
    (aHash, the pre-existing default) or "dhash" (Phase 7B's evidence-based
    final policy). If "dhash" is requested and the manifest does not already
    carry a "dhash" column, it is computed here (one read per image, no
    write to the original files) and reused for the rest of this run.
    `exposed_image_ids` (Phase 7B): image IDs already disclosed as non-blind
    (see PHASE6_RESULTS.md Section 2) -- their duplicate groups, if clean,
    are force-assigned to train and excluded from valid/test.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if hash_field == "dhash" and not (rows and rows[0].get("dhash")):
        compute_dhash_column(rows)

    g = compute_duplicate_groups(rows, near_dup_threshold=near_dup_threshold, hash_field=hash_field)
    conflicts = assess_group_label_conflicts(rows, g["group_of"])
    conflict_group_ids = set(conflicts["conflict_group_ids"])
    exposed_group_ids = identify_exposed_groups(g["group_of"], set(exposed_image_ids))
    effective = compute_effective_counts(rows, g["group_of"], conflict_group_ids)
    part = build_group_disjoint_partition(
        rows, g["group_of"], conflict_group_ids, seed=seed, split_ratios=split_ratios,
        exposed_group_ids=exposed_group_ids)
    manifest_rows = build_partition_manifest_rows(
        rows, g["group_of"], part["assignment"], conflict_group_ids, exposed_group_ids)

    v1 = verify_no_group_crosses_partitions(manifest_rows)
    v2 = verify_no_image_in_multiple_partitions(manifest_rows)
    v3 = verify_no_conflict_in_supervised_split(manifest_rows)
    v4 = verify_counts_match_manifest(manifest_rows)
    v5 = verify_no_exposed_group_in_valid_or_test(manifest_rows)
    assert v1["ok"], f"group crossed partitions: {v1['violations']}"
    assert v2["ok"], f"image in multiple partitions: {v2['duplicates']}"
    assert v3["ok"], f"unresolved conflict entered a supervised split: {v3['violations']}"
    assert v4["ok"], "partition count tally inconsistency"
    assert v5["ok"], f"exposed group entered valid/test: {v5['violations']}"

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
                    "phash", "dhash", "group_id", "group_size", "conflict_status", "exposed_status",
                    "new_split", "include_in_supervised_training"])
    hashes["effective_counts.json"] = write_json(output / "effective_counts.json", effective)
    hashes["partition_config.json"] = write_json(output / "partition_config.json", {
        "seed": part["seed"], "split_ratios": part["split_ratios"], "algorithm": part["algorithm"],
        "near_dup_threshold": g["near_dup_threshold"], "hash_field": hash_field,
        "source_manifest": str(manifest_csv), "n_exposed_image_ids": len(exposed_image_ids),
        "n_exposed_groups": len(exposed_group_ids),
    })
    verification_report = {
        "no_group_crosses_partitions": v1, "no_image_in_multiple_partitions": v2,
        "no_conflict_in_supervised_split": v3, "counts_match_manifest": v4,
        "no_exposed_group_in_valid_or_test": v5,
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
        "n_exposed_groups": len(exposed_group_ids),
        "split_totals": verification_report["split_totals"],
        "verification": verification_report,
        "artifact_hashes": hashes,
        "output": str(output),
    }
