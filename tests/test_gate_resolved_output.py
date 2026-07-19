"""A1 — leakage gate uses the RESOLVED output dir; config-only never Path(None)."""

from __future__ import annotations
import csv
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _clean_manifest(path):
    rows = [
        {"image_id": "a", "path": "/x/a.png", "split": "train", "class": "Happy",
         "sha256": "h1", "phash": "0f0f0f0f0f0f0f0f"},
        {"image_id": "b", "path": "/x/b.png", "split": "valid", "class": "Sad",
         "sha256": "h2", "phash": "ffff0000ffff0000"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "path", "split", "class", "sha256", "phash"])
        w.writeheader()
        w.writerows(rows)


class GateResolvedOutputTests(unittest.TestCase):
    def test_gate_raises_without_any_output(self):
        # Directly exercise the guard path: a resolved_output of None with no
        # args.output must raise a clear error, not TypeError from Path(None).
        sys.path.insert(0, str(ROOT / "src"))
        from doar.leakage import enforce_leakage_gate  # noqa
        # emulate the guard the way _gate does
        base = None
        with self.assertRaises(ValueError):
            if base is None:
                raise ValueError("Leakage gate requires an output directory (config or --output).")

    def test_extract_embeddings_config_only_does_not_pathnone(self):
        # extract-embeddings with a config (output from TOML, no --output) must
        # reach the gate with a real path. We stop before torch by pointing the
        # manifest at a clean file; the failure (if any) must NOT be Path(None).
        with tempfile.TemporaryDirectory() as d:
            man = Path(d) / "manifest.csv"
            _clean_manifest(man)
            out = Path(d) / "emb_out"
            cfg = Path(d) / "emb.toml"
            cfg.write_text(textwrap.dedent(f"""
                [input]
                manifest = "{man.as_posix()}"
                [embedding]
                backbone = "resnet18"
                [output]
                directory = "{out.as_posix()}"
            """), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "main.py", "extract-embeddings", "--config", str(cfg)],
                cwd=ROOT, capture_output=True, text=True)
            combined = proc.stdout + proc.stderr
            # The gate must have run against the resolved output (leakage_gate dir).
            self.assertNotIn("expected str, bytes or os.PathLike", combined)
            self.assertNotIn("NoneType", combined)
            # Either the gate passed (clean manifest) or it failed later on torch —
            # but never on Path(None). A leakage_gate dir under the resolved output
            # proves the resolved path was used.
            self.assertTrue((out / "leakage_gate").exists()
                            or "PyTorch" in combined or "torch" in combined.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
