"""Tests for scripts/clinician_review_app.py -- the psychologist/clinician
review prototype.

Covers: cached-only data loading (no live Gemini calls), the VISUALLY
VERIFIED != PSYCHOLOGICALLY VALIDATED gate (uncertain/rejected evidence
stays visible but never becomes rule-eligible), parent-view language
safety (no rule IDs / internal domain identifiers / disorder-name
claims), the clinician_relevance_verdict vs clinical_diagnostic_status
separation, feedback save/load, and that p2b_0004 resolves to its fresh
cache, not the legacy stale one.
"""
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "clinician_review_app", ROOT / "scripts" / "clinician_review_app.py")
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)

from doar import reasoning_chain as rc  # noqa: E402


class NoLiveGeminiCallsTests(unittest.TestCase):
    def test_module_never_calls_the_live_observer_or_verifier(self):
        source = (ROOT / "scripts" / "clinician_review_app.py").read_text(encoding="utf-8")
        self.assertNotIn("GeminiVisualObserver(", source)
        self.assertNotIn("GeminiVisualVerifier(", source)
        self.assertNotIn("run_live_observer_and_verifier(", source)

    def test_load_case_bundle_never_touches_live_call_path(self):
        # A real case load must succeed without any network-related error --
        # proves the cached-only path actually works end to end.
        bundle = app.load_case_bundle("h38")
        self.assertIsNotNone(bundle)
        self.assertIsNotNone(bundle["rows"])


class CaseLoadingTests(unittest.TestCase):
    def test_list_case_ids_returns_all_15(self):
        ids = app.list_case_ids()
        self.assertEqual(len(ids), 15)
        self.assertIn("p2b_0003", ids)

    def test_load_case_bundle_unknown_id_returns_none(self):
        self.assertIsNone(app.load_case_bundle("not_a_real_case"))

    def test_load_case_bundle_has_expected_shape(self):
        bundle = app.load_case_bundle("p2b_0003")
        for key in ("image", "rows", "source_path", "entities", "conditions", "checks"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["image"]["image_id"], "p2b_0003")
        self.assertGreater(len(bundle["rows"]), 0)
        self.assertEqual(len(bundle["checks"]), 41)


class P2b0004FreshCacheTests(unittest.TestCase):
    def test_p2b_0004_resolves_to_the_live_cache_not_legacy(self):
        bundle = app.load_case_bundle("p2b_0004")
        self.assertIsNotNone(bundle["rows"])
        self.assertIn("development_live_cache", str(bundle["source_path"]))

    def test_p2b_0004_bboxes_are_all_valid_normalized_coordinates(self):
        bundle = app.load_case_bundle("p2b_0004")
        for row in bundle["rows"]:
            bbox = row["observer_candidate"].get("bbox")
            if bbox:
                self.assertTrue(all(0.0 <= v <= 1.0 for v in bbox),
                                 f"{row['observation_id']} has an out-of-range bbox: {bbox}")


class VerificationGateTests(unittest.TestCase):
    """VISUALLY VERIFIED != PSYCHOLOGICALLY VALIDATED: uncertain/rejected/
    unreviewed evidence must remain visible to the clinician (grouped,
    not hidden) but must never appear as matched evidence behind an
    eligible rule."""

    def test_non_verified_rows_are_grouped_not_dropped(self):
        bundle = app.load_case_bundle("p2b_0003")
        groups = app.group_rows_by_status(bundle["rows"])
        total_grouped = len(groups["verified"]) + len(groups["uncertain"]) + len(groups["other"])
        self.assertEqual(total_grouped, len(bundle["rows"]))
        # p2b_0003's real cached data has known uncertain/rejected rows.
        self.assertGreater(len(groups["uncertain"]) + len(groups["other"]), 0)

    def test_no_unverified_entity_id_appears_in_any_eligible_match(self):
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            verified_ids = {e.entity_id for e in bundle["entities"] if e.case_verification_status == "verified"}
            for m in bundle["conditions"]["doar_full_pipeline"]["eligible_matches"]:
                for entity_id in m.matched_entity_ids:
                    self.assertIn(entity_id, verified_ids,
                                  f"{case_id}: rule {m.rule_id} matched a non-verified entity {entity_id}")

    def test_verified_labels_excludes_uncertain_and_rejected(self):
        bundle = app.load_case_bundle("p2b_0003")
        doar = bundle["conditions"]["doar_full_pipeline"]
        groups = app.group_rows_by_status(bundle["rows"])
        excluded_labels = {r["observer_candidate"]["label"] for r in groups["uncertain"] + groups["other"]}
        verified_labels = set(doar["verified_labels"])
        self.assertEqual(verified_labels & excluded_labels, set())


class ParentViewSafetyTests(unittest.TestCase):
    """Parent view must never expose rule IDs, internal snake_case concern-
    domain identifiers, or disorder-name diagnostic claims -- only the
    already-vetted, code-constrained parent_package wording."""

    _RULE_ID_PATTERN = re.compile(r"\b(PSY_AR_|EN_COMPILED_)[A-Z_0-9]+\b")
    _DISORDER_WORDS = ("depression", "depressed", "anxiety disorder", "adhd", "ptsd", "bipolar",
                        "schizophrenia", "abuse", "maltreatment", "autism", "ocd")

    def _fixture_hypothesis(self, concern_domain="depressive_or_low_mood_related"):
        return rc.CandidateHypothesis(
            concern_domain=concern_domain, hypothesis_label="Possible depressive/low-mood presentation",
            support_level="WEAK_HYPOTHESIS", evidence_families_represented=("facial_feature_style",),
            supporting_rule_ids=("EN_COMPILED_FACE_EXPRESSION_021",), supporting_entity_ids=("e1",),
            contradicting_reference_ids=(), alternative_explanations=("Many other explanations exist",),
            missing_clinical_information=("Child's own explanation (not collected)",),
        )

    def test_parent_sections_contain_no_rule_ids(self):
        sections = app.build_parent_view_sections([self._fixture_hypothesis()])
        blob = json.dumps(sections)
        self.assertIsNone(self._RULE_ID_PATTERN.search(blob), f"rule ID leaked into parent view: {blob}")

    def test_parent_sections_contain_no_internal_domain_identifier(self):
        sections = app.build_parent_view_sections([self._fixture_hypothesis()])
        blob = json.dumps(sections)
        self.assertNotIn("depressive_or_low_mood_related", blob)
        self.assertNotIn("concern_domain", blob)

    def test_parent_sections_contain_no_disorder_name_claim(self):
        sections = app.build_parent_view_sections([self._fixture_hypothesis()])
        blob = json.dumps(sections).lower()
        for word in self._DISORDER_WORDS:
            self.assertNotIn(word, blob, f"disorder-name claim {word!r} leaked into parent view")

    def test_empty_hypothesis_list_yields_no_sections(self):
        self.assertEqual(app.build_parent_view_sections([]), [])

    def test_no_hypothesis_message_is_not_a_diagnostic_claim(self):
        for word in self._DISORDER_WORDS:
            self.assertNotIn(word, app.PARENT_NO_HYPOTHESIS_MESSAGE.lower())

    def test_all_real_cases_parent_view_is_safe(self):
        # End-to-end sweep across the real 15-case development set.
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            sections = app.build_parent_view_sections(bundle["conditions"]["doar_full_pipeline"]["hypotheses"])
            blob = json.dumps(sections)
            self.assertIsNone(self._RULE_ID_PATTERN.search(blob), f"{case_id}: rule ID leaked into parent view")


class RelevanceVerdictVsDiagnosisTests(unittest.TestCase):
    def test_relevance_verdict_options_never_claim_diagnosis(self):
        for option in app.RELEVANCE_VERDICT_OPTIONS:
            self.assertNotIn("diagnos", option.lower())

    def test_clinical_diagnostic_status_is_always_hardcoded_assessment_pending(self):
        # The app must never derive clinical_diagnostic_status from the
        # clinician_relevance_verdict -- it's a fixed literal in the source.
        source = (ROOT / "scripts" / "clinician_review_app.py").read_text(encoding="utf-8")
        self.assertIn('"clinical_diagnostic_status": "assessment_pending"', source)

    def test_clinician_no_hypothesis_message_does_not_claim_absence(self):
        self.assertIn("does not indicate absence", app.CLINICIAN_NO_HYPOTHESIS_MESSAGE)


class FeedbackStorageTests(unittest.TestCase):
    def test_save_and_load_round_trips(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original = app.FEEDBACK_DIR
            app.FEEDBACK_DIR = Path(tmp)
            try:
                payload = {"observation_feedback": {"h38_c00": {"observer_label": "sun", "verdict": "Confirm visually"}},
                           "hypothesis_feedback": [], "general_comments": "looks reasonable"}
                path = app.save_feedback("test_session", "h38", payload)
                self.assertTrue(path.exists())
                loaded = app.load_feedback("test_session", "h38")
                self.assertEqual(loaded["general_comments"], "looks reasonable")
                self.assertEqual(loaded["case_id"], "h38")
                self.assertEqual(loaded["session_id"], "test_session")
                self.assertIn("git_commit", loaded)
                self.assertIn("timestamp", loaded)
            finally:
                app.FEEDBACK_DIR = original

    def test_repeated_save_is_idempotent_not_duplicated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original = app.FEEDBACK_DIR
            app.FEEDBACK_DIR = Path(tmp)
            try:
                app.save_feedback("test_session", "h38", {"general_comments": "first"})
                app.save_feedback("test_session", "h38", {"general_comments": "second"})
                json_files = list(Path(tmp).glob("*.json"))
                self.assertEqual(len(json_files), 1)
                self.assertEqual(app.load_feedback("test_session", "h38")["general_comments"], "second")
            finally:
                app.FEEDBACK_DIR = original

    def test_feedback_never_stores_api_keys(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original = app.FEEDBACK_DIR
            app.FEEDBACK_DIR = Path(tmp)
            try:
                app.save_feedback("test_session", "h38", {"general_comments": "note"})
                saved_text = (Path(tmp) / "test_session__h38.json").read_text(encoding="utf-8")
                for banned in ("GEMINI_API_KEY", "api_key", "apikey"):
                    self.assertNotIn(banned, saved_text)
            finally:
                app.FEEDBACK_DIR = original

    def test_summary_csv_rebuilt_after_save(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original = app.FEEDBACK_DIR
            app.FEEDBACK_DIR = Path(tmp)
            try:
                app.save_feedback("s1", "h38", {"general_comments": "a"})
                app.save_feedback("s2", "p2b_0001", {"general_comments": "b"})
                csv_path = Path(tmp) / "summary.csv"
                self.assertTrue(csv_path.exists())
                content = csv_path.read_text(encoding="utf-8")
                self.assertIn("h38", content)
                self.assertIn("p2b_0001", content)
            finally:
                app.FEEDBACK_DIR = original

    def test_different_sessions_same_case_do_not_collide(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original = app.FEEDBACK_DIR
            app.FEEDBACK_DIR = Path(tmp)
            try:
                app.save_feedback("session_a", "h38", {"general_comments": "from A"})
                app.save_feedback("session_b", "h38", {"general_comments": "from B"})
                self.assertEqual(app.load_feedback("session_a", "h38")["general_comments"], "from A")
                self.assertEqual(app.load_feedback("session_b", "h38")["general_comments"], "from B")
            finally:
                app.FEEDBACK_DIR = original


class BboxOverlayTests(unittest.TestCase):
    def test_draw_annotated_image_returns_an_image_same_size_as_original(self):
        bundle = app.load_case_bundle("h38")
        image_path = ROOT / bundle["image"]["relative_path"]
        from PIL import Image
        original = Image.open(image_path)
        annotated = app.draw_annotated_image(image_path, bundle["rows"])
        self.assertEqual(annotated.size, original.size)

    def test_candidate_with_no_bbox_does_not_crash(self):
        bundle = app.load_case_bundle("h38")
        image_path = ROOT / bundle["image"]["relative_path"]
        rows = [dict(bundle["rows"][0])]
        rows[0] = dict(rows[0])
        rows[0]["observer_candidate"] = dict(rows[0]["observer_candidate"])
        rows[0]["observer_candidate"]["bbox"] = None
        app.draw_annotated_image(image_path, rows)  # must not raise


if __name__ == "__main__":
    unittest.main()
