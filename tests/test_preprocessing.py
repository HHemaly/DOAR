"""Item 6 — preprocessing spec is the single source of truth; DINOv2 fixed."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class SpecTests(unittest.TestCase):
    def test_dinov2_records_its_actual_transform(self):
        from doar.deep.preprocessing import resolve_preprocessing
        spec = resolve_preprocessing("dinov2_vits14")
        # The bug was labelling resize-224 as native; now it IS resize256->crop224.
        self.assertEqual(spec["resize"], 256)
        self.assertEqual(spec["crop"], 224)
        self.assertEqual(spec["preprocessing_version"],
                         "dinov2_resize256_centercrop224_imagenet_norm")

    def test_torchvision_spec(self):
        from doar.deep.preprocessing import resolve_preprocessing
        spec = resolve_preprocessing("resnet18", 224)
        self.assertEqual(spec["resize"], [224, 224])
        self.assertIsNone(spec["crop"])
        self.assertEqual(spec["weights_id"], "DEFAULT")

    def test_openclip_spec(self):
        from doar.deep.preprocessing import resolve_preprocessing
        spec = resolve_preprocessing("openclip:ViT-B-32")
        self.assertEqual(spec["family"], "openclip")
        self.assertEqual(spec["weights_id"], "laion2b_s34b_b79k")

    def test_hash_changes_with_transform_params(self):
        from doar.deep.preprocessing import resolve_preprocessing, preprocessing_hash
        a = preprocessing_hash(resolve_preprocessing("resnet18", 224))
        b = preprocessing_hash(resolve_preprocessing("resnet18", 256))  # diff resize
        c = preprocessing_hash(resolve_preprocessing("dinov2_vits14"))
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_compatibility_check(self):
        from doar.deep.preprocessing import (resolve_preprocessing, preprocessing_hash,
                                             assert_preprocessing_compatible,
                                             PreprocessingMismatch)
        spec = resolve_preprocessing("resnet18", 224)
        good = preprocessing_hash(spec)
        # Matching hash -> ok
        assert_preprocessing_compatible(good, spec)
        # Missing hash (older checkpoint) -> allowed
        assert_preprocessing_compatible(None, spec)
        # Wrong hash -> raises
        with self.assertRaises(PreprocessingMismatch):
            assert_preprocessing_compatible("deadbeef", spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
