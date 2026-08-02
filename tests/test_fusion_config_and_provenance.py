"""B4 — fusion config plumbing (accepted-but-unused rejected; constraints enforced).

CLI-level tests use the real primary_fusion.toml. Provenance (B5) tested in
test_provenance.py.
"""

from __future__ import annotations
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _run(*args):
    return subprocess.run([sys.executable, "main.py", *args], cwd=ROOT,
                          capture_output=True, text=True)


class FusionConfigTests(unittest.TestCase):
    def test_shipped_toml_all_fields_consumed(self):
        # Every accepted field in primary_fusion.toml must be mapped (no
        # accepted-but-unused). It should fail on inputs/paths, NOT on an
        # unconsumed-config assertion.
        proc = _run("train-fusion-model", "--config",
                    "configs/training/primary_fusion.toml")
        combined = proc.stdout + proc.stderr
        self.assertNotIn("accepted but never used", combined)

    def test_bad_selection_split_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "bad.toml"
            cfg.write_text(textwrap.dedent("""
                [inputs]
                features = "f.csv"
                embeddings = "e.npz"
                [experiment]
                methods = ["early_scaled_concat"]
                seeds = [42]
                selection_split = "test"
                primary_metric = "macro_f1"
                psychologist_rules_used = false
                concern_profiles_used = false
                [output]
                directory = "out"
                calibration = "none"
            """), encoding="utf-8")
            proc = _run("train-fusion-model", "--config", str(cfg))
            self.assertIn("selection_split must be 'valid'", proc.stdout + proc.stderr)

    def test_psychologist_rules_true_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "bad.toml"
            cfg.write_text(textwrap.dedent("""
                [inputs]
                features = "f.csv"
                embeddings = "e.npz"
                [experiment]
                methods = ["early_scaled_concat"]
                seeds = [42]
                selection_split = "valid"
                primary_metric = "macro_f1"
                psychologist_rules_used = true
                concern_profiles_used = false
                [output]
                directory = "out"
                calibration = "none"
            """), encoding="utf-8")
            proc = _run("train-fusion-model", "--config", str(cfg))
            self.assertIn("must be false", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
