"""DOAR MVP: case_output.py::refresh_module_availability -- patches specific
judges.json["module_availability"] fields in place after a real detection
run or an expert-review submission, both of which happen strictly after
finalize_case's own run_judges() snapshot."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.case_output import refresh_module_availability  # noqa: E402


class RefreshModuleAvailabilityTests(unittest.TestCase):
    def test_patches_given_keys_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "judges.json").write_text(json.dumps({
                "module_availability": {"detection": "unavailable", "ocr": "unavailable"},
                "quality_judge": {"status": "pass"},
            }), encoding="utf-8")
            refresh_module_availability(case_dir, detection="available")
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["detection"], "available")
            self.assertEqual(judges["module_availability"]["ocr"], "unavailable")
            self.assertEqual(judges["quality_judge"]["status"], "pass")

    def test_no_op_when_judges_file_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            refresh_module_availability(Path(tmp), detection="available")
            self.assertFalse((Path(tmp) / "judges.json").exists())

    def test_creates_module_availability_key_if_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "judges.json").write_text(json.dumps({}), encoding="utf-8")
            refresh_module_availability(case_dir, clinician_review="submitted")
            judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
            self.assertEqual(judges["module_availability"]["clinician_review"], "submitted")

    def test_archives_previous_version_via_write_versioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "judges.json").write_text(
                json.dumps({"module_availability": {"detection": "unavailable"}}), encoding="utf-8")
            refresh_module_availability(case_dir, detection="available")
            self.assertTrue((case_dir / "versions").exists())


if __name__ == "__main__":
    unittest.main()
