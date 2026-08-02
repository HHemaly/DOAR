"""B1/B2/B3 — calibrated export, universal exporter alignment, shared test guard."""

from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

try:
    import sklearn  # noqa
    _SK = True
except Exception:
    _SK = False


class TestGuardTests(unittest.TestCase):
    def test_non_test_split_is_allowed(self):
        from doar.test_guard import require_test_access
        self.assertIsNone(require_test_access(
            "valid", unlock_test=False, confirm_final_evaluation=False,
            initiated_by=None, command="x", audit_dir="/tmp/na"))

    def test_test_split_blocked_without_all_flags(self):
        from doar.test_guard import require_test_access, TestAccessDenied
        with tempfile.TemporaryDirectory() as d:
            for kw in (
                dict(unlock_test=False, confirm_final_evaluation=True, initiated_by="a"),
                dict(unlock_test=True, confirm_final_evaluation=False, initiated_by="a"),
                dict(unlock_test=True, confirm_final_evaluation=True, initiated_by=""),   # empty
                dict(unlock_test=True, confirm_final_evaluation=True, initiated_by=None),
            ):
                with self.assertRaises(TestAccessDenied):
                    require_test_access("test", command="x", audit_dir=d, **kw)

    def test_test_access_granted_writes_audit(self):
        from doar.test_guard import require_test_access
        with tempfile.TemporaryDirectory() as d:
            ev = require_test_access(
                "test", unlock_test=True, confirm_final_evaluation=True,
                initiated_by="Ahmed", command="evaluate", audit_dir=d,
                timestamp="2026-01-01T00:00:00Z", model="m.pt")
            self.assertEqual(ev["initiated_by"], "Ahmed")
            log = (Path(d) / "final_test_unlock_log.jsonl").read_text().strip()
            self.assertIn("Ahmed", log)
            self.assertIn("evaluate", log)


class CLIGuardTests(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([sys.executable, "main.py", *args],
                              cwd=ROOT, capture_output=True, text=True)

    def test_evaluate_predictions_blocks_test_by_default(self):
        # Build a minimal export with a test row, then try to evaluate test split.
        with tempfile.TemporaryDirectory() as d:
            from doar.evaluation import save_probability_export
            exp = Path(d) / "e.json"
            save_probability_export(
                exp, sample_ids=["s0"], splits=["test"], y_true=[0],
                proba=np.eye(4)[[0]], model_id="m", checkpoint_hash="h",
                calibration_status="uncalibrated")
            proc = self._run("evaluate-predictions", "--export", str(exp),
                             "--split", "test", "--output", str(Path(d) / "out"))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("locked", (proc.stdout + proc.stderr).lower())


@unittest.skipUnless(_SK, "sklearn not installed")
class CalibratedExportTests(unittest.TestCase):
    def test_calibrated_probabilities_differ_from_raw(self):
        # A fusion-like bundle with a temperature != 1 must export calibrated
        # probabilities (used for eval) distinct from raw_probabilities (B1).
        import joblib
        from sklearn.linear_model import LogisticRegression
        from doar.probability_export import export_probabilities
        with tempfile.TemporaryDirectory() as d:
            # tiny feature CSV
            import csv
            fp = Path(d) / "features.csv"
            rng = np.random.RandomState(0)
            with open(fp, "w", newline="") as h:
                w = csv.writer(h)
                w.writerow(["image_id", "path", "split", "class", "f0", "f1"])
                for i in range(20):
                    cls = ["Angry", "Fear", "Happy", "Sad"][i % 4]
                    w.writerow([f"s{i}", "", "valid", cls, rng.randn(), rng.randn()])
            X = rng.randn(20, 2)
            y = np.array([i % 4 for i in range(20)])
            clf = LogisticRegression(max_iter=500).fit(X, y)
            bundle = {"model": clf, "model_name": "objfeat",
                      "calibration": {"status": "calibrated", "temperature": 2.5,
                                      "method": "temperature_scaling"}}
            mp = Path(d) / "m.joblib"
            joblib.dump(bundle, mp)
            out = Path(d) / "exp.json"
            export_probabilities(str(mp), str(fp), None, str(out), splits=["valid"])
            data = json.loads(out.read_text())
            self.assertEqual(data["calibration_status"], "calibrated")
            self.assertEqual(data["temperature"], 2.5)
            row = data["predictions"][0]
            self.assertIn("raw_probabilities", row)
            # calibrated (used) != raw for temperature != 1
            used = list(row["probabilities"].values())
            raw = list(row["raw_probabilities"].values())
            self.assertNotAlmostEqual(used[0], raw[0], places=4)

    def test_fusion_missing_embedding_raises(self):
        import joblib
        import csv
        from sklearn.linear_model import LogisticRegression
        from doar.probability_export import export_probabilities
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d) / "features.csv"
            with open(fp, "w", newline="") as h:
                w = csv.writer(h)
                w.writerow(["image_id", "path", "split", "class", "f0"])
                for i in range(6):
                    w.writerow([f"s{i}", "", "valid", "Happy", 0.1 * i])
            # embeddings only for s0..s2 -> s3..s5 missing
            npz = Path(d) / "emb.npz"
            np.savez(npz, image_ids=np.array(["s0", "s1", "s2"]),
                     embeddings=np.random.randn(3, 4),
                     splits=np.array(["valid"] * 3), labels=np.array(["Happy"] * 3))
            clf = LogisticRegression(max_iter=200).fit(np.random.randn(6, 5),
                                                       np.array([0, 1, 2, 3, 0, 1]))
            bundle = {"model": clf, "checkpoint_type": "doar_fusion_bundle_v1"}
            mp = Path(d) / "f.joblib"
            joblib.dump(bundle, mp)
            with self.assertRaises(ValueError) as ctx:
                export_probabilities(str(mp), str(fp), str(npz), str(Path(d) / "e.json"),
                                     splits=["valid"])
            self.assertIn("alignment", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
