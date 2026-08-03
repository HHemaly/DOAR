"""Phase 7A -- image-group-disjoint, duplicate-controlled partition design.

Covers: duplicate-group computation (exact + near, transitive closure),
label-conflict detection at the group level, deterministic group-disjoint
stratified partitioning, and the required verification checks (no image in
multiple partitions, no group crossing partitions, no unresolved conflict
in a supervised split, counts consistency)."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _row(image_id, split, cls, sha256, phash="0f0f0f0f0f0f0f0f", path=None):
    return {
        "image_id": image_id, "path": path or f"/x/{image_id}.png",
        "relative_path": f"{split}/{cls}/{image_id}.png",
        "split": split, "class": cls, "sha256": sha256, "phash": phash,
    }


class DuplicateGroupTests(unittest.TestCase):
    def test_exact_duplicates_grouped_regardless_of_split(self):
        from doar.partition import compute_duplicate_groups
        rows = [
            _row("a", "train", "Happy", "h1", phash="0000000000000001"),
            _row("b", "test", "Happy", "h1", phash="0000000000000002"),   # exact dup of a, different split
            _row("c", "valid", "Sad", "h2", phash="ffff000000000000"),
        ]
        result = compute_duplicate_groups(rows, near_dup_threshold=0)
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertNotEqual(result["group_of"]["a"], result["group_of"]["c"])
        self.assertEqual(result["n_exact_edges"], 1)

    def test_near_duplicates_grouped_within_same_split_too(self):
        # leakage.assess_leakage() only ever compares CROSS-split pairs; this
        # module must also group near-dups within the SAME split, because a
        # from-scratch partition needs the full duplicate structure, not just
        # violations of the old split assignment.
        from doar.partition import compute_duplicate_groups
        rows = [
            _row("a", "train", "Happy", "h1", phash="0f0f0f0f0f0f0f0f"),
            _row("b", "train", "Happy", "h2", phash="0f0f0f0f0f0f0f0e"),  # 1 bit off, SAME split
        ]
        result = compute_duplicate_groups(rows, near_dup_threshold=5)
        self.assertEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_near_edges"], 1)

    def test_transitive_chain_forms_one_group(self):
        # a~b (dist small), b~c (dist small), a~c not directly within
        # threshold -- must still all end up in one group (single-linkage
        # transitive closure), the conservative/leakage-safe choice.
        from doar.partition import compute_duplicate_groups
        rows = [
            _row("a", "train", "Happy", "ha", phash="0000000000000001"),
            _row("b", "valid", "Happy", "hb", phash="0000000000000003"),
            _row("c", "test", "Happy", "hc", phash="0000000000000007"),
        ]
        # hamming(a,b)=1 (bit2), hamming(b,c)=1 (bit3), hamming(a,c)=2
        result = compute_duplicate_groups(rows, near_dup_threshold=1)
        gid_a = result["group_of"]["a"]
        self.assertEqual(gid_a, result["group_of"]["b"])
        self.assertEqual(gid_a, result["group_of"]["c"])

    def test_degenerate_phash_excluded_from_near_dup(self):
        from doar.partition import compute_duplicate_groups
        rows = [
            _row("a", "train", "Happy", "ha", phash="0000000000000000"),
            _row("b", "valid", "Happy", "hb", phash="0000000000000000"),
        ]
        result = compute_duplicate_groups(rows, near_dup_threshold=5)
        self.assertNotEqual(result["group_of"]["a"], result["group_of"]["b"])
        self.assertEqual(result["n_near_edges"], 0)

    def test_group_ids_deterministic_across_runs(self):
        from doar.partition import compute_duplicate_groups
        rows = [
            _row("z", "train", "Happy", "h1"),
            _row("a", "test", "Happy", "h1"),
            _row("m", "valid", "Sad", "h2"),
        ]
        r1 = compute_duplicate_groups(rows, near_dup_threshold=0)
        r2 = compute_duplicate_groups(rows, near_dup_threshold=0)
        self.assertEqual(r1["group_of"], r2["group_of"])
        self.assertEqual(r1["groups"], r2["groups"])


class LabelConflictTests(unittest.TestCase):
    def test_exact_duplicate_with_different_labels_is_a_conflict(self):
        from doar.partition import compute_duplicate_groups, assess_group_label_conflicts
        rows = [
            _row("a", "train", "Happy", "h1"),
            _row("b", "test", "Sad", "h1"),  # same bytes, different label
        ]
        g = compute_duplicate_groups(rows, near_dup_threshold=0)
        conflicts = assess_group_label_conflicts(rows, g["group_of"])
        self.assertEqual(conflicts["n_conflict_groups"], 1)
        self.assertIn("a", conflicts["conflicts"][0]["image_ids"])
        self.assertIn("b", conflicts["conflicts"][0]["image_ids"])

    def test_near_duplicate_label_conflict_is_caught(self):
        # This is a real gap in leakage.assess_leakage: it only flags
        # conflicting_labels for EXACT sha256 matches, never near-duplicates.
        # assess_group_label_conflicts must catch this case too.
        from doar.partition import compute_duplicate_groups, assess_group_label_conflicts
        rows = [
            _row("a", "train", "Angry", "ha", phash="0f0f0f0f0f0f0f0f"),
            _row("b", "test", "Fear", "hb", phash="0f0f0f0f0f0f0f0e"),  # near-dup, different label
        ]
        g = compute_duplicate_groups(rows, near_dup_threshold=5)
        conflicts = assess_group_label_conflicts(rows, g["group_of"])
        self.assertEqual(conflicts["n_conflict_groups"], 1)

    def test_clean_group_has_no_conflict(self):
        from doar.partition import compute_duplicate_groups, assess_group_label_conflicts
        rows = [_row("a", "train", "Happy", "h1"), _row("b", "test", "Happy", "h1")]
        g = compute_duplicate_groups(rows, near_dup_threshold=0)
        conflicts = assess_group_label_conflicts(rows, g["group_of"])
        self.assertEqual(conflicts["n_conflict_groups"], 0)


class EffectiveCountsDeterminismTests(unittest.TestCase):
    def test_dict_outputs_are_key_sorted_not_set_iteration_order(self):
        # Regression test: compute_effective_counts() used to iterate a raw
        # `set` of conflict_group_ids directly (`for gid in conflict_group_ids`)
        # to build conflict_images_by_class. A Python set's iteration order
        # depends on per-process string-hash randomization, NOT insertion
        # order, so the resulting dict's key order -- and therefore the exact
        # bytes written to effective_counts.json -- was not reproducible
        # across separate process runs even though the counts were always
        # numerically correct. Found via `main.py build-partition` producing
        # a different effective_counts.json hash on two separate real-dataset
        # runs of the identical command. Fixed by sorting before iterating/
        # building every dict this function returns. This test checks the
        # guarantee directly: every dict-valued field must already be in
        # sorted-key order, which is order-independent of however the caller
        # constructed the conflict_group_ids set.
        from doar.partition import compute_duplicate_groups, assess_group_label_conflicts, compute_effective_counts
        rows = _synthetic_dataset()
        g = compute_duplicate_groups(rows, near_dup_threshold=5)
        conflicts = assess_group_label_conflicts(rows, g["group_of"])
        result = compute_effective_counts(rows, g["group_of"], set(conflicts["conflict_group_ids"]))
        for field in ("raw_by_class", "effective_groups_by_class",
                     "effective_images_by_class", "conflict_images_by_class"):
            keys = list(result[field].keys())
            self.assertEqual(keys, sorted(keys), f"{field} keys not sorted: {keys}")


def _synthetic_dataset(n_per_class=40, dup_rate=0.1, conflict_rate=0.02, seed=7):
    """A synthetic manifest-row set with controlled exact/near-dup/conflict
    structure, large enough to exercise partitioning meaningfully."""
    import random
    rng = random.Random(seed)
    rows = []
    counter = 0
    classes = ("Angry", "Fear", "Happy", "Sad")
    base_hashes = []
    for cls in classes:
        for _ in range(n_per_class):
            counter += 1
            image_id = f"img_{counter:05d}"
            sha = f"sha_{counter:05d}"
            phash = format(rng.getrandbits(64), "016x")
            rows.append(_row(image_id, "train", cls, sha, phash=phash))
            base_hashes.append((image_id, sha, phash, cls))
    # Inject exact duplicates (same sha256, different image_id/split) at dup_rate
    n_dups = int(len(base_hashes) * dup_rate)
    for k in range(n_dups):
        src_id, sha, phash, cls = base_hashes[k]
        counter += 1
        dup_id = f"img_{counter:05d}"
        rows.append(_row(dup_id, "valid" if k % 2 == 0 else "test", cls, sha, phash=phash))
    # Inject a handful of label conflicts (exact dup, different class)
    n_conf = max(1, int(len(base_hashes) * conflict_rate))
    other_classes = {"Angry": "Fear", "Fear": "Happy", "Happy": "Sad", "Sad": "Angry"}
    for k in range(n_dups, n_dups + n_conf):
        src_id, sha, phash, cls = base_hashes[k % len(base_hashes)]
        counter += 1
        conf_id = f"img_{counter:05d}"
        rows.append(_row(conf_id, "test", other_classes[cls], sha, phash=phash))
    return rows


class PartitionConstructionTests(unittest.TestCase):
    def setUp(self):
        self.rows = _synthetic_dataset()

    def _build(self, seed=42):
        from doar.partition import (
            compute_duplicate_groups, assess_group_label_conflicts,
            build_group_disjoint_partition, build_partition_manifest_rows,
        )
        g = compute_duplicate_groups(self.rows, near_dup_threshold=5)
        conflicts = assess_group_label_conflicts(self.rows, g["group_of"])
        conflict_ids = set(conflicts["conflict_group_ids"])
        part = build_group_disjoint_partition(
            self.rows, g["group_of"], conflict_ids, seed=seed,
            split_ratios=(0.7, 0.1, 0.2))
        manifest_rows = build_partition_manifest_rows(
            self.rows, g["group_of"], part["assignment"], conflict_ids)
        return g, conflicts, part, manifest_rows

    def test_deterministic_split_reproduction(self):
        _, _, part1, rows1 = self._build(seed=42)
        _, _, part2, rows2 = self._build(seed=42)
        self.assertEqual(part1["assignment"], part2["assignment"])
        self.assertEqual(rows1, rows2)

    def test_different_seed_can_differ(self):
        _, _, part1, _ = self._build(seed=42)
        _, _, part2, _ = self._build(seed=123)
        # Not a strict requirement that they differ, but with this many
        # groups it would be suspicious if a different seed produced an
        # identical assignment -- guards against an accidentally-ignored
        # seed parameter.
        self.assertNotEqual(part1["assignment"], part2["assignment"])

    def test_no_image_assigned_to_multiple_partitions(self):
        from doar.partition import verify_no_image_in_multiple_partitions
        _, _, _, manifest_rows = self._build()
        result = verify_no_image_in_multiple_partitions(manifest_rows)
        self.assertTrue(result["ok"], result["duplicates"])

    def test_no_duplicate_group_crosses_partitions(self):
        from doar.partition import verify_no_group_crosses_partitions
        _, _, _, manifest_rows = self._build()
        result = verify_no_group_crosses_partitions(manifest_rows)
        self.assertTrue(result["ok"], result["violations"])

    def test_no_unresolved_conflict_enters_supervised_split(self):
        from doar.partition import verify_no_conflict_in_supervised_split
        _, conflicts, _, manifest_rows = self._build()
        self.assertGreater(conflicts["n_conflict_groups"], 0, "fixture should include conflicts")
        result = verify_no_conflict_in_supervised_split(manifest_rows)
        self.assertTrue(result["ok"], result["violations"])

    def test_every_original_image_present_exactly_once(self):
        _, _, _, manifest_rows = self._build()
        self.assertEqual(len(manifest_rows), len(self.rows))
        self.assertEqual(
            {r["image_id"] for r in manifest_rows},
            {r["image_id"] for r in self.rows},
        )

    def test_conflict_images_never_dropped_only_relabeled(self):
        # Conflicting-label images must be RETAINED (split="excluded_conflict"),
        # never silently removed from the manifest.
        _, conflicts, _, manifest_rows = self._build()
        conflict_image_ids = {iid for c in conflicts["conflicts"] for iid in c["image_ids"]}
        by_id = {r["image_id"]: r for r in manifest_rows}
        for iid in conflict_image_ids:
            self.assertIn(iid, by_id)
            self.assertEqual(by_id[iid]["new_split"], "excluded_conflict")
            self.assertFalse(by_id[iid]["include_in_supervised_training"])

    def test_counts_consistency_check(self):
        from doar.partition import verify_counts_match_manifest
        _, _, _, manifest_rows = self._build()
        result = verify_counts_match_manifest(manifest_rows)
        self.assertTrue(result["ok"])

    def test_split_ratios_are_validated(self):
        from doar.partition import build_group_disjoint_partition, compute_duplicate_groups
        g = compute_duplicate_groups(self.rows, near_dup_threshold=5)
        with self.assertRaises(ValueError):
            build_group_disjoint_partition(self.rows, g["group_of"], set(), split_ratios=(0.5, 0.5, 0.5))


class ManifestIOTests(unittest.TestCase):
    def test_write_csv_and_json_hash_and_paths_resolve(self):
        from doar.partition import write_csv, write_json, hash_file
        with tempfile.TemporaryDirectory() as d:
            rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
            csv_path = Path(d) / "out.csv"
            digest = write_csv(csv_path, rows, fieldnames=["a", "b"])
            self.assertTrue(csv_path.exists())
            self.assertEqual(digest, hash_file(csv_path))

            json_path = Path(d) / "out.json"
            jdigest = write_json(json_path, {"x": 1})
            self.assertTrue(json_path.exists())
            self.assertEqual(jdigest, hash_file(json_path))

    def test_manifest_paths_resolve_to_original_locations(self):
        # The partition manifest must carry the ORIGINAL image path through
        # unchanged -- this is what "generate manifests rather than moving
        # images" means in practice: the path column must still point at the
        # real, untouched file.
        from doar.partition import (
            compute_duplicate_groups, assess_group_label_conflicts,
            build_group_disjoint_partition, build_partition_manifest_rows,
        )
        with tempfile.TemporaryDirectory() as d:
            real_file = Path(d) / "train" / "Happy" / "img.png"
            real_file.parent.mkdir(parents=True)
            real_file.write_bytes(b"not a real image, just a path-resolution test")
            rows = [_row("a", "train", "Happy", "h1", path=str(real_file))]
            g = compute_duplicate_groups(rows, near_dup_threshold=0)
            conflicts = assess_group_label_conflicts(rows, g["group_of"])
            part = build_group_disjoint_partition(
                rows, g["group_of"], set(conflicts["conflict_group_ids"]), seed=1)
            manifest_rows = build_partition_manifest_rows(
                rows, g["group_of"], part["assignment"], set(conflicts["conflict_group_ids"]))
            self.assertTrue(Path(manifest_rows[0]["path"]).exists())
            self.assertEqual(Path(manifest_rows[0]["path"]).read_bytes(),
                             b"not a real image, just a path-resolution test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
