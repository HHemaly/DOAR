"""DOAR MVP: broad_vocabulary.py -- combines every frozen Phase 2C.7 policy
target with a small set of unvalidated "common child-drawing content"
extras. No real model weights involved, pure data-shape tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.broad_vocabulary import (  # noqa: E402
    UNVALIDATED_EXTRA_TARGETS, broad_scan_vocabulary, frozen_policy_targets, is_frozen_policy_target,
)
from doar.phase2c7.detector_policy import (  # noqa: E402
    EXPERIMENTAL_AUTOMATIC, OBJECT_CLASS_POLICY, build_eye_policy_entry,
)


def _eye_entry(status=EXPERIMENTAL_AUTOMATIC):
    return build_eye_policy_entry(
        status=status, best_model="grounding_dino_parts+owlv2_parts_fallback",
        best_model_checkpoint="ckpt", prompt="eye. mouth.", threshold=0.25,
        precision=0.88, recall=0.73, balanced_accuracy=0.57, n_ground_truth_present=128,
        localization_validated=False, rationale="test")


class FrozenPolicyTargetsTests(unittest.TestCase):
    def test_includes_eye_and_excludes_non_dispatchable(self):
        targets = frozen_policy_targets(_eye_entry())
        self.assertIn("eye", targets)
        self.assertIn("person", targets)
        self.assertNotIn("person_part_reference", targets)

    def test_disabled_targets_still_listed_here_but_never_dispatched_downstream(self):
        # frozen_policy_targets only reports policy membership -- DISABLED
        # exclusion happens in visual_detector.analyze_image, not here.
        targets = frozen_policy_targets(_eye_entry())
        self.assertIn("circle", targets)
        self.assertIn("vehicle", targets)


class BroadScanVocabularyTests(unittest.TestCase):
    def test_combines_frozen_and_extras_deduplicated(self):
        vocab = broad_scan_vocabulary(_eye_entry())
        self.assertEqual(len(vocab), len(set(vocab)))
        for extra in UNVALIDATED_EXTRA_TARGETS:
            self.assertIn(extra, vocab)
        for target in OBJECT_CLASS_POLICY:
            if target == "person_part_reference":
                continue  # non-dispatchable -- never independently scanned
            self.assertIn(target, vocab)
        self.assertIn("eye", vocab)

    def test_extras_never_overlap_frozen_policy_targets(self):
        frozen = set(OBJECT_CLASS_POLICY) | {"eye"}
        self.assertEqual(frozen & set(UNVALIDATED_EXTRA_TARGETS), set())


class IsFrozenPolicyTargetTests(unittest.TestCase):
    def test_true_for_policy_members_and_eye(self):
        self.assertTrue(is_frozen_policy_target("person"))
        self.assertTrue(is_frozen_policy_target("eye"))

    def test_false_for_unvalidated_extras_and_unknown(self):
        self.assertFalse(is_frozen_policy_target("sun"))
        self.assertFalse(is_frozen_policy_target("bicycle"))


if __name__ == "__main__":
    unittest.main()
