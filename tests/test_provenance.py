"""B5 — artifact provenance + cross-artifact verification (CPU)."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _feat(manifest_sha, sid_hash, status="PASS", classes=None):
    return {"artifact": "objective_features", "manifest_sha256": manifest_sha,
            "sample_id_hash": sid_hash, "leakage_status": status,
            "class_order": classes or ["Angry", "Fear", "Happy", "Sad"]}


def _emb(manifest_sha, sid_hash, status="PASS", classes=None):
    return {"artifact": "embeddings", "manifest_sha256": manifest_sha,
            "sample_id_hash": sid_hash, "leakage_status": status,
            "class_order": classes or ["Angry", "Fear", "Happy", "Sad"]}


class ProvenanceTests(unittest.TestCase):
    def test_sample_id_hash_order_sensitive(self):
        from doar.provenance import sample_id_hash
        self.assertNotEqual(sample_id_hash(["a", "b"]), sample_id_hash(["b", "a"]))
        self.assertEqual(sample_id_hash(["a", "b"]), sample_id_hash(["a", "b"]))

    def test_matching_artifacts_pass(self):
        from doar.provenance import verify_artifacts
        r = verify_artifacts(_feat("m1", "s1"), _emb("m1", "s1"))
        self.assertTrue(r["ok"])

    def test_different_manifest_rejected(self):
        from doar.provenance import verify_artifacts, ProvenanceError
        with self.assertRaises(ProvenanceError):
            verify_artifacts(_feat("m1", "s1"), _emb("m2", "s1"))

    def test_different_sample_ids_rejected(self):
        from doar.provenance import verify_artifacts, ProvenanceError
        with self.assertRaises(ProvenanceError):
            verify_artifacts(_feat("m1", "s1"), _emb("m1", "s2"))

    def test_bad_leakage_status_rejected(self):
        from doar.provenance import verify_artifacts, ProvenanceError
        with self.assertRaises(ProvenanceError):
            verify_artifacts(_feat("m1", "s1", status="FAIL"), _emb("m1", "s1"))

    def test_override_requires_justification_and_audits(self):
        from doar.provenance import verify_artifacts, ProvenanceError
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ProvenanceError):
                verify_artifacts(_feat("m1", "s1"), _emb("m2", "s1"),
                                 allow_override=True)  # no justification
            r = verify_artifacts(_feat("m1", "s1"), _emb("m2", "s1"),
                                 allow_override=True, override_justification="known re-scan",
                                 audit_dir=d, timestamp="2026-01-01T00:00:00Z")
            self.assertTrue(r["overridden"])
            self.assertTrue((Path(d) / "provenance_override_audit.jsonl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
