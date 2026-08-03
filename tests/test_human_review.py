"""Tests for the Phase 7B human-review interface's core logic
(src/doar/human_review.py). No Streamlit import here -- this exercises the
plain data/IO functions the UI layer sits on top of."""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from doar.human_review import (
    HUMAN_JUDGMENT_CHOICES,
    blind_item_for_display,
    build_threshold_precision_summary,
    compute_precision_by_distance,
    compute_reviewer_agreement,
    export_all,
    export_group_reviews_csv,
    export_pair_reviews_csv,
    export_unresolved_csv,
    find_unresolved,
    load_decisions,
    progress_by_category,
    save_decision,
    wilson_score_interval,
)


def _item(item_id, category, **overrides):
    base = {
        "item_id": item_id, "category": category, "source_pair_id": None,
        "image_rel_path": f"outputs/phase7b/human_review/review_pairs/{item_id}.jpg",
        "image_a": f"{item_id}_a", "image_b": f"{item_id}_b",
        "class_a": "Happy", "class_b": "Sad", "split_a": "train", "split_b": "train",
        "same_label": False, "source_category": "synthetic", "hamming_ahash": None,
        "hamming_dhash": None, "ai_preliminary_judgment": None, "ai_notes": None,
        "group_id": None, "group_size": None, "edge_type": "near_dup",
    }
    base.update(overrides)
    return base


class WilsonScoreIntervalTests(unittest.TestCase):
    def test_zero_n_returns_maximally_uncertain(self):
        self.assertEqual(wilson_score_interval(0, 0), (0.0, 1.0))

    def test_bounds_contain_point_estimate_and_stay_in_unit_interval(self):
        lo, hi = wilson_score_interval(5, 10)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_more_successes_shifts_interval_up(self):
        lo_low, hi_low = wilson_score_interval(1, 10)
        lo_high, hi_high = wilson_score_interval(9, 10)
        self.assertLess(hi_low, hi_high)
        self.assertLess(lo_low, lo_high)

    def test_perfect_record_interval_excludes_zero_for_larger_n(self):
        lo, _ = wilson_score_interval(20, 20)
        self.assertGreater(lo, 0.5)


class BlindingTests(unittest.TestCase):
    def test_blind_view_exposes_only_id_category_and_image_path(self):
        item = _item("c17_0001", "component_17", class_a="Angry", hamming_dhash=3,
                      ai_preliminary_judgment="definite_duplicate")
        blind = blind_item_for_display(item)
        self.assertEqual(set(blind.keys()), {"item_id", "category", "image_rel_path"})

    def test_blind_view_never_leaks_hidden_fields_regardless_of_input(self):
        item = _item("amb_0001", "ambiguous", class_a="Fear", class_b="Fear", same_label=True,
                      hamming_ahash=5, ai_preliminary_judgment="uncertain", ai_notes="secret note",
                      group_id="grp_1", group_size=33)
        blind = blind_item_for_display(item)
        leaked = set(blind.values()) & {
            "Fear", True, 5, "uncertain", "secret note", "grp_1", 33,
        }
        self.assertEqual(leaked, set())


class DecisionStoreTests(unittest.TestCase):
    def test_save_and_reload_round_trips(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "decisions.json"
            save_decision(path, "amb_0001", "definite_duplicate", "identical drawing")
            decisions = load_decisions(path)
            self.assertEqual(decisions["amb_0001"]["decision"], "definite_duplicate")
            self.assertEqual(decisions["amb_0001"]["notes"], "identical drawing")
            self.assertIn("reviewed_at", decisions["amb_0001"])

    def test_missing_decisions_file_returns_empty_dict(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "does_not_exist.json"
            self.assertEqual(load_decisions(path), {})

    def test_second_save_does_not_clobber_earlier_items(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "decisions.json"
            save_decision(path, "amb_0001", "definite_duplicate")
            save_decision(path, "amb_0002", "uncertain")
            decisions = load_decisions(path)
            self.assertEqual(set(decisions.keys()), {"amb_0001", "amb_0002"})

    def test_invalid_decision_value_rejected(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "decisions.json"
            with self.assertRaises(ValueError):
                save_decision(path, "amb_0001", "not_a_real_choice")

    def test_clearing_a_decision_removes_it(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "decisions.json"
            save_decision(path, "amb_0001", "definite_duplicate")
            save_decision(path, "amb_0001", None)
            self.assertNotIn("amb_0001", load_decisions(path))

    def test_all_choices_are_accepted(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "decisions.json"
            for i, choice in enumerate(HUMAN_JUDGMENT_CHOICES):
                save_decision(path, f"item_{i}", choice)
            decisions = load_decisions(path)
            self.assertEqual(len(decisions), len(HUMAN_JUDGMENT_CHOICES))


class ProgressTests(unittest.TestCase):
    def test_progress_counts_are_correct(self):
        categories_order = {"ambiguous": ["a1", "a2", "a3"], "component_17": ["c1"]}
        decisions = {"a1": {"decision": "uncertain"}}
        progress = progress_by_category(categories_order, decisions)
        self.assertEqual(progress["ambiguous"], {"total": 3, "reviewed": 1, "remaining": 2})
        self.assertEqual(progress["component_17"], {"total": 1, "reviewed": 0, "remaining": 1})


class ReviewerAgreementTests(unittest.TestCase):
    def _registry(self):
        items = {
            "amb_0001": _item("amb_0001", "ambiguous", ai_preliminary_judgment="definite_duplicate"),
            "amb_0002": _item("amb_0002", "ambiguous", ai_preliminary_judgment="similar_but_different"),
            "amb_0003": _item("amb_0003", "ambiguous", ai_preliminary_judgment="not_duplicate"),
            "c17_0001": _item("c17_0001", "component_17", ai_preliminary_judgment=None),
        }
        return {"items": items, "categories_order": {"ambiguous": list(items)[:3], "component_17": ["c17_0001"]}}

    def test_agreement_only_counts_items_with_both_ai_and_human_judgment(self):
        registry = self._registry()
        decisions = {
            "amb_0001": {"decision": "definite_duplicate", "notes": ""},
            "c17_0001": {"decision": "definite_duplicate", "notes": ""},  # no AI judgment -> excluded
        }
        result = compute_reviewer_agreement(registry, decisions)
        self.assertEqual(result["n_items_compared"], 1)
        self.assertEqual(result["overall_agreement_rate"], 1.0)

    def test_ai_vocabulary_collapses_onto_human_vocabulary_for_comparison(self):
        registry = self._registry()
        decisions = {
            "amb_0002": {"decision": "different_drawings", "notes": ""},  # AI said similar_but_different
            "amb_0003": {"decision": "different_drawings", "notes": ""},  # AI said not_duplicate
        }
        result = compute_reviewer_agreement(registry, decisions)
        self.assertEqual(result["n_items_compared"], 2)
        self.assertEqual(result["overall_agreement_rate"], 1.0)

    def test_disagreement_lowers_rate_and_appears_in_confusion_matrix(self):
        registry = self._registry()
        decisions = {"amb_0001": {"decision": "different_drawings", "notes": ""}}
        result = compute_reviewer_agreement(registry, decisions)
        self.assertEqual(result["overall_agreement_rate"], 0.0)
        self.assertEqual(result["confusion_matrix_ai_bucket_by_human_choice"]["definite_duplicate"]["different_drawings"], 1)

    def test_no_comparable_items_yields_none_rate_not_a_crash(self):
        registry = self._registry()
        result = compute_reviewer_agreement(registry, {})
        self.assertEqual(result["n_items_compared"], 0)
        self.assertIsNone(result["overall_agreement_rate"])


class PrecisionByDistanceTests(unittest.TestCase):
    def test_precision_computed_per_exact_distance_bucket(self):
        items = {
            "p1": _item("p1", "threshold_boundary", hamming_dhash=2),
            "p2": _item("p2", "threshold_boundary", hamming_dhash=2),
            "p3": _item("p3", "threshold_boundary", hamming_dhash=5),
        }
        registry = {"items": items, "categories_order": {}}
        decisions = {
            "p1": {"decision": "definite_duplicate"}, "p2": {"decision": "same_drawing_transformed"},
            "p3": {"decision": "different_drawings"},
        }
        table = compute_precision_by_distance(registry, decisions, "hamming_dhash", judgment_source="human")
        self.assertEqual(table["2"]["precision"], 1.0)
        self.assertEqual(table["2"]["n_judged"], 2)
        self.assertEqual(table["5"]["precision"], 0.0)

    def test_uncertain_counts_in_denominator_not_numerator(self):
        items = {"p1": _item("p1", "threshold_boundary", hamming_dhash=4)}
        registry = {"items": items, "categories_order": {}}
        decisions = {"p1": {"decision": "uncertain"}}
        table = compute_precision_by_distance(registry, decisions, "hamming_dhash")
        self.assertEqual(table["4"]["n_judged"], 1)
        self.assertEqual(table["4"]["n_duplicate_like"], 0)
        self.assertEqual(table["4"]["n_uncertain"], 1)

    def test_missing_distance_field_excluded_from_table(self):
        items = {"p1": _item("p1", "ambiguous", hamming_dhash=None)}
        registry = {"items": items, "categories_order": {}}
        table = compute_precision_by_distance(registry, {"p1": {"decision": "uncertain"}}, "hamming_dhash")
        self.assertEqual(table, {})

    def test_small_sample_flagged_insufficient(self):
        items = {"p1": _item("p1", "threshold_boundary", hamming_dhash=7)}
        registry = {"items": items, "categories_order": {}}
        table = compute_precision_by_distance(registry, {"p1": {"decision": "definite_duplicate"}},
                                               "hamming_dhash", min_n_for_confidence=5)
        self.assertTrue(table["7"]["insufficient_sample"])

    def test_ai_judgment_source_uses_preliminary_judgments_not_decisions(self):
        items = {"p1": _item("p1", "threshold_boundary", hamming_dhash=3,
                              ai_preliminary_judgment="definite_duplicate")}
        registry = {"items": items, "categories_order": {}}
        table = compute_precision_by_distance(registry, {}, "hamming_dhash", judgment_source="ai")
        self.assertEqual(table["3"]["precision"], 1.0)

    def test_invalid_judgment_source_raises(self):
        items = {"p1": _item("p1", "threshold_boundary", hamming_dhash=3)}
        registry = {"items": items, "categories_order": {}}
        with self.assertRaises(ValueError):
            compute_precision_by_distance(registry, {}, "hamming_dhash", judgment_source="bogus")


class ThresholdPrecisionSummaryTests(unittest.TestCase):
    def test_summary_includes_both_hash_fields_and_both_judgment_sources(self):
        items = {"p1": _item("p1", "threshold_boundary", hamming_dhash=3, hamming_ahash=2,
                              ai_preliminary_judgment="definite_duplicate")}
        registry = {"items": items, "categories_order": {}}
        summary = build_threshold_precision_summary(registry, {"p1": {"decision": "uncertain"}})
        for key in ("human_precision_by_dhash_distance", "human_precision_by_ahash_distance",
                    "provisional_ai_precision_by_dhash_distance", "provisional_ai_precision_by_ahash_distance"):
            self.assertIn(key, summary)


class UnresolvedItemsTests(unittest.TestCase):
    def test_undecided_and_uncertain_items_are_unresolved_decided_items_are_not(self):
        items = {
            "p1": _item("p1", "ambiguous"), "p2": _item("p2", "ambiguous"), "p3": _item("p3", "ambiguous"),
        }
        registry = {"items": items, "categories_order": {}}
        decisions = {"p2": {"decision": "uncertain"}, "p3": {"decision": "definite_duplicate"}}
        unresolved = find_unresolved(registry, decisions)
        statuses = {u["item_id"]: u["status"] for u in unresolved}
        self.assertEqual(statuses, {"p1": "not_yet_reviewed", "p2": "marked_uncertain"})


class ExportTests(unittest.TestCase):
    def _registry_and_decisions(self):
        items = {
            "amb_0001": _item("amb_0001", "ambiguous", ai_preliminary_judgment="definite_duplicate",
                               hamming_ahash=0),
            "tb_0001": _item("tb_0001", "threshold_boundary", ai_preliminary_judgment="uncertain",
                              hamming_dhash=5),
            "c17_0001": _item("c17_0001", "component_17", image_a="MEMBER_A", image_b="MEMBER_B",
                               hamming_dhash=3),
            "c17_0002": _item("c17_0002", "component_17", image_a="MEMBER_B", image_b="MEMBER_C",
                               hamming_dhash=6),
            "pc_0001": _item("pc_0001", "policy_change", source_category="candidate_threshold_2_to_4",
                              hamming_dhash=4),
        }
        registry = {
            "items": items,
            "categories_order": {
                "ambiguous": ["amb_0001"], "threshold_boundary": ["tb_0001"],
                "component_17": ["c17_0001", "c17_0002"], "policy_change": ["pc_0001"],
            },
        }
        decisions = {
            "amb_0001": {"decision": "definite_duplicate", "notes": "n1", "reviewed_at": "t1"},
            "c17_0001": {"decision": "definite_duplicate", "notes": "", "reviewed_at": "t2"},
            "c17_0002": {"decision": "different_drawings", "notes": "", "reviewed_at": "t3"},
            "pc_0001": {"decision": "same_drawing_transformed", "notes": "", "reviewed_at": "t4"},
        }
        return registry, decisions

    def test_pair_reviews_csv_has_one_row_per_item_with_correct_columns(self):
        registry, decisions = self._registry_and_decisions()
        with TemporaryDirectory() as d:
            out = Path(d) / "human_pair_reviews.csv"
            n = export_pair_reviews_csv(registry, decisions, out)
            self.assertEqual(n, 5)
            rows = list(csv.DictReader(open(out, encoding="utf-8")))
            self.assertEqual(len(rows), 5)
            row = next(r for r in rows if r["item_id"] == "amb_0001")
            self.assertEqual(row["human_judgment"], "definite_duplicate")
            self.assertEqual(row["ai_preliminary_judgment"], "definite_duplicate")
            unreviewed_row = next(r for r in rows if r["item_id"] == "tb_0001")
            self.assertEqual(unreviewed_row["human_judgment"], "")

    def test_group_reviews_csv_aggregates_component_17_members_correctly(self):
        registry, decisions = self._registry_and_decisions()
        with TemporaryDirectory() as d:
            out = Path(d) / "human_group_reviews.csv"
            export_group_reviews_csv(registry, decisions, out)
            rows = list(csv.DictReader(open(out, encoding="utf-8")))
            member_b = next(r for r in rows if r["group_or_transition_id"] == "MEMBER_B")
            # MEMBER_B appears in both c17 edges: one duplicate_like, one different
            self.assertEqual(member_b["n_internal_edges"], "2")
            self.assertEqual(member_b["n_judged_duplicate_like"], "1")
            self.assertEqual(member_b["n_judged_different"], "1")
            transition_row = next(r for r in rows if r["group_or_transition_id"] == "candidate_threshold_2_to_4")
            self.assertEqual(transition_row["n_reviewed"], "1")

    def test_unresolved_csv_excludes_decided_items(self):
        registry, decisions = self._registry_and_decisions()
        with TemporaryDirectory() as d:
            out = Path(d) / "unresolved_items.csv"
            n = export_unresolved_csv(registry, decisions, out)
            rows = list(csv.DictReader(open(out, encoding="utf-8")))
            ids = {r["item_id"] for r in rows}
            self.assertNotIn("amb_0001", ids)  # decided, not uncertain
            self.assertIn("tb_0001", ids)  # never decided
            self.assertEqual(n, len(rows))

    def test_export_all_writes_five_files_and_a_consistent_summary(self):
        registry, decisions = self._registry_and_decisions()
        with TemporaryDirectory() as d:
            registry_path = Path(d) / "items_registry.json"
            decisions_path = Path(d) / "decisions.json"
            out_dir = Path(d) / "exports"
            json.dump(registry, open(registry_path, "w", encoding="utf-8"))
            json.dump(decisions, open(decisions_path, "w", encoding="utf-8"))
            summary = export_all(registry_path, decisions_path, out_dir)
            for fname in ("human_pair_reviews.csv", "human_group_reviews.csv",
                          "reviewer_agreement_report.json", "threshold_precision_summary.json",
                          "unresolved_items.csv"):
                self.assertTrue((out_dir / fname).exists(), fname)
            self.assertEqual(summary["total_items"], 5)
            self.assertEqual(summary["total_reviewed"], 4)


if __name__ == "__main__":
    unittest.main()
