"""Automated schema-integrity checks for the project's root-level CSV
registers (RULE_SOURCE_REGISTER.csv, RULE_COVERAGE_MATRIX.csv,
DETECTOR_DEPENDENCY_MATRIX.csv, LITERATURE_CANDIDATE_REGISTER.csv).

Added after a real incident (2026-08-02, see DECISION_LOG.md): three rows in
RULE_SOURCE_REGISTER.csv contained an unquoted comma inside a parenthetical
remark, which any CSV parser reads as an extra field boundary -- a strict
csv.DictWriter later crashed on this and briefly truncated the file before
being caught and reverted via git. This test would have caught the
malformed rows immediately, before they were ever committed, rather than
relying on a later script crashing on them.

These tests are read-only: they never write to the registers.
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# The first column of each register is treated as its unique identifier
# column for the duplicate-ID check.
REGISTERS = {
    "RULE_SOURCE_REGISTER.csv": "rule_id",
    "RULE_COVERAGE_MATRIX.csv": "rule_id",
    "DETECTOR_DEPENDENCY_MATRIX.csv": "rule_id",
    "LITERATURE_CANDIDATE_REGISTER.csv": "candidate_id",
}


class CsvRegisterIntegrityTests(unittest.TestCase):
    def _rows(self, filename: str) -> tuple[list[str], list[list[str]]]:
        path = ROOT / filename
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        return header, rows

    def test_every_register_file_exists(self):
        for filename in REGISTERS:
            self.assertTrue((ROOT / filename).exists(), f"{filename} is missing")

    def test_every_row_has_exactly_the_declared_column_count(self):
        for filename in REGISTERS:
            with self.subTest(filename=filename):
                header, rows = self._rows(filename)
                expected = len(header)
                bad = [
                    (i, len(row), row[0] if row else "(empty row)")
                    for i, row in enumerate(rows, start=2)
                    if len(row) != expected
                ]
                self.assertEqual(
                    bad, [],
                    f"{filename}: rows with wrong field count (expected {expected}): {bad}",
                )

    def test_no_duplicate_ids_within_a_register(self):
        for filename, id_column in REGISTERS.items():
            with self.subTest(filename=filename):
                header, rows = self._rows(filename)
                id_index = header.index(id_column)
                ids = [row[id_index] for row in rows]
                seen = set()
                duplicates = {i for i in ids if i in seen or seen.add(i)}
                self.assertEqual(duplicates, set(), f"{filename}: duplicate {id_column} values: {duplicates}")

    def test_no_row_has_an_empty_id(self):
        for filename, id_column in REGISTERS.items():
            with self.subTest(filename=filename):
                header, rows = self._rows(filename)
                id_index = header.index(id_column)
                empties = [i for i, row in enumerate(rows, start=2) if not row[id_index].strip()]
                self.assertEqual(empties, [], f"{filename}: rows with an empty {id_column} at lines {empties}")

    def test_header_is_non_empty_and_has_no_duplicate_column_names(self):
        for filename in REGISTERS:
            with self.subTest(filename=filename):
                header, _ = self._rows(filename)
                self.assertTrue(header)
                self.assertEqual(len(header), len(set(header)), f"{filename}: duplicate column names in header")

    def test_dict_reader_round_trip_matches_row_reader(self):
        # A stricter check: DictReader (used by every script that consumes
        # these registers) must produce a dict with no None key for every
        # row -- None appears as a key exactly when a row has more fields
        # than the header, which is the failure mode this whole test file
        # exists to prevent from recurring silently.
        for filename in REGISTERS:
            with self.subTest(filename=filename):
                path = ROOT / filename
                with open(path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                offenders = [row.get("rule_id") or row.get("candidate_id") for row in rows if None in row]
                self.assertEqual(offenders, [], f"{filename}: rows with extra unmapped fields: {offenders}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
