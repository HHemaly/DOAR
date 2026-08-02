"""Item 16 — structured psychologist review + agreement (real reviews only)."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class ReviewTests(unittest.TestCase):
    def test_no_reviews_agreement_unavailable_not_fabricated(self):
        from doar.review import init_master, compute_agreement
        with tempfile.TemporaryDirectory() as d:
            m = init_master(Path(d) / "master.csv")
            res = compute_agreement(m)
            self.assertFalse(res["has_reviews"])
            self.assertIn("unavailable", res["note"])

    def test_append_only_and_item_validation(self):
        from doar.review import init_master, append_reviews, load_master
        with tempfile.TemporaryDirectory() as d:
            m = init_master(Path(d) / "master.csv")
            append_reviews(m, [{"case_id": "c1", "item": "rules", "reviewer_id": "r1",
                                "rating": "Agree", "comment": "", "timestamp": "t0"}])
            append_reviews(m, [{"case_id": "c1", "item": "segmentation", "reviewer_id": "r1",
                                "rating": "Disagree", "comment": "", "timestamp": "t1"}])
            self.assertEqual(len(load_master(m)), 2)   # appended, not overwritten
            with self.assertRaises(ValueError):
                append_reviews(m, [{"case_id": "c1", "item": "not_an_item",
                                    "reviewer_id": "r1", "rating": "Agree"}])

    def test_synthetic_reviews_excluded(self):
        from doar.review import init_master, append_reviews, compute_agreement
        with tempfile.TemporaryDirectory() as d:
            m = init_master(Path(d) / "master.csv")
            # Only synthetic rows -> agreement stays unavailable.
            append_reviews(m, [{"case_id": "c1", "item": "rules", "reviewer_id": "SYN",
                                "rating": "Agree", "timestamp": "t", "is_synthetic": "true"}])
            self.assertFalse(compute_agreement(m)["has_reviews"])

    def test_two_reviewers_cohen_kappa(self):
        from doar.review import init_master, append_reviews, compute_agreement
        with tempfile.TemporaryDirectory() as d:
            m = init_master(Path(d) / "master.csv")
            rows = []
            for i, item in enumerate(["rules", "segmentation", "concerns"]):
                rows.append({"case_id": "c1", "item": item, "reviewer_id": "A",
                             "rating": "Agree", "timestamp": "t"})
                rows.append({"case_id": "c1", "item": item, "reviewer_id": "B",
                             "rating": "Agree" if i < 2 else "Disagree", "timestamp": "t"})
            append_reviews(m, rows)
            res = compute_agreement(m)
            self.assertTrue(res["has_reviews"])
            self.assertEqual(res["n_reviewers"], 2)
            self.assertEqual(res["kappa_type"], "cohen")

    def test_fleiss_kappa_three_raters(self):
        from doar.review import fleiss_kappa
        # perfect agreement across 3 raters on 2 items -> 1.0
        self.assertEqual(fleiss_kappa([{"Agree": 3}, {"Disagree": 3}]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
