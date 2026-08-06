"""Requires the local, private Phase 7B partition manifest (gitignored,
derived from the private dataset) -- these are local-only integration
tests, explicitly marked via a skip guard, per the task's own requirement
that no test may require an unavailable private dataset otherwise."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.duplicate_groups import (
    groups_available, load_group_lookup, write_pilot_duplicate_groups_csv,
)

_AVAILABLE = groups_available()
_REASON = "requires the local, private Phase 7B partition manifest (gitignored, not available in CI)"


class GroupsAvailabilityGuardTests(unittest.TestCase):
    def test_missing_manifest_raises_a_clear_filenotfounderror(self):
        with self.assertRaises(FileNotFoundError):
            load_group_lookup(Path("does/not/exist.csv"))


@unittest.skipUnless(_AVAILABLE, _REASON)
class LoadGroupLookupTests(unittest.TestCase):
    def test_returns_a_nonempty_lookup(self):
        lookup = load_group_lookup()
        self.assertGreater(len(lookup), 0)

    def test_every_entry_has_the_expected_fields(self):
        lookup = load_group_lookup()
        sample = next(iter(lookup.values()))
        self.assertIn("group_id", sample)
        self.assertIn("group_size", sample)
        self.assertIn("conflict_status", sample)


@unittest.skipUnless(_AVAILABLE, _REASON)
class WritePilotDuplicateGroupsCsvTests(unittest.TestCase):
    def test_writes_one_row_per_selected_image(self):
        lookup = load_group_lookup()
        some_ids = list(lookup.keys())[:5]
        with tempfile.TemporaryDirectory() as d:
            path = write_pilot_duplicate_groups_csv(some_ids, Path(d) / "groups.csv")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 6)  # header + 5

    def test_unknown_image_id_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(KeyError):
                write_pilot_duplicate_groups_csv(["not_a_real_image_id"], Path(d) / "groups.csv")


if __name__ == "__main__":
    unittest.main()
