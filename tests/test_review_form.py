"""C2 — UI-independent review form-to-row conversion + agreement consumption."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class ReviewFormTests(unittest.TestCase):
    def test_form_to_rows_covers_seven_items(self):
        from doar.review import form_to_rows, REVIEW_ITEMS
        ratings = {it: "Agree" for it in REVIEW_ITEMS}
        rows = form_to_rows("case1", "revA", ratings, {}, "t0")
        self.assertEqual(len(rows), 7)
        self.assertEqual({r["item"] for r in rows}, set(REVIEW_ITEMS))
        self.assertTrue(all(r["reviewer_id"] == "revA" for r in rows))

    def test_reviewer_required(self):
        from doar.review import form_to_rows
        with self.assertRaises(ValueError):
            form_to_rows("case1", "", {"rules": "Agree"}, {}, "t0")

    def test_skipped_items_excluded(self):
        from doar.review import form_to_rows
        rows = form_to_rows("case1", "revA", {"rules": "Agree", "concerns": None}, {}, "t0")
        self.assertEqual(len(rows), 1)

    def test_form_rows_flow_into_agreement(self):
        from doar.review import form_to_rows, init_master, append_reviews, compute_agreement
        with tempfile.TemporaryDirectory() as d:
            master = init_master(Path(d) / "review_master.csv")
            for rev, rating in (("A", "Agree"), ("B", "Agree")):
                append_reviews(master, form_to_rows(
                    "c1", rev, {"rules": rating, "segmentation": rating}, {}, "t0"))
            res = compute_agreement(master)
            self.assertTrue(res["has_reviews"])
            self.assertEqual(res["n_reviewers"], 2)
            self.assertEqual(res["kappa_type"], "cohen")

    def test_synthetic_reviews_excluded_from_agreement(self):
        from doar.review import form_to_rows, init_master, append_reviews, compute_agreement
        with tempfile.TemporaryDirectory() as d:
            master = init_master(Path(d) / "review_master.csv")
            append_reviews(master, form_to_rows(
                "c1", "A", {"rules": "Agree"}, {}, "t0", is_synthetic=True))
            self.assertFalse(compute_agreement(master)["has_reviews"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
