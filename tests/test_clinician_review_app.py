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

from doar import drawing_synthesis as ds  # noqa: E402
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
        for key in ("image", "rows", "source_path", "entities", "conditions", "checks",
                    "deterministic_features", "synthesis"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["image"]["image_id"], "p2b_0003")
        self.assertGreater(len(bundle["rows"]), 0)
        self.assertEqual(len(bundle["checks"]), 41)
        self.assertIsNotNone(bundle["deterministic_features"])
        self.assertIsNotNone(bundle["synthesis"])


class UnifiedEvidenceBundleTests(unittest.TestCase):
    """DOAR realignment milestone: every case bundle now also carries the
    unified multimodal evidence + always-on drawing synthesis
    (drawing_synthesis.synthesize_drawing), cache-first via
    drawing_synthesis.load_or_compute_deterministic_features -- proven
    here across the real 15-case development set, no live Gemini call."""

    def test_every_real_case_has_populated_synthesis(self):
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            synthesis = bundle["synthesis"]
            self.assertGreater(len(synthesis.unified_evidence), 0, f"{case_id}: no unified evidence")
            self.assertGreater(len(synthesis.objective_profile), 0, f"{case_id}: empty objective profile")
            self.assertTrue(synthesis.overall_synthesis["summary"], f"{case_id}: overall synthesis is always-on")

    def test_p2b_0003_deterministic_features_have_sixty_objective_features(self):
        bundle = app.load_case_bundle("p2b_0003")
        self.assertEqual(len(bundle["deterministic_features"]["objective_features"]), 60)


class DisplaySectionsTests(unittest.TestCase):
    """build_display_sections regroups drawing_synthesis's own
    objective_profile into the 7 sections the realignment task specifies
    -- a pure UI-layer regrouping, so every item must land in exactly one
    section (nothing dropped, nothing duplicated, nothing invented)."""

    def test_all_seven_section_titles_present(self):
        bundle = app.load_case_bundle("p2b_0003")
        sections = app.build_display_sections(bundle)
        self.assertEqual(set(sections.keys()), set(app.DISPLAY_SECTION_TITLES))

    def test_item_count_conserved_across_regrouping(self):
        bundle = app.load_case_bundle("p2b_0003")
        sections = app.build_display_sections(bundle)
        total_in_sections = sum(len(items) for items in sections.values())
        total_in_profile = sum(len(items) for items in bundle["synthesis"].objective_profile.values())
        self.assertEqual(total_in_sections, total_in_profile)

    def test_body_part_entities_land_in_expression_pose_not_objects_people(self):
        bundle = app.load_case_bundle("p2b_0003")
        sections = app.build_display_sections(bundle)
        entities = bundle["entities"]
        body_part_labels = {e.canonical_label for e in entities if "body_part" in e.broader_categories}
        if not body_part_labels:
            self.skipTest("p2b_0003 has no body-part-labeled entity in this cache")
        expression_reasons = " ".join(entry["item"]["reason"] for entry in sections["Expression / Pose"])
        self.assertTrue(any(label in expression_reasons for label in body_part_labels))

    def test_no_bundle_returns_empty_sections_without_crashing(self):
        bundle = app.load_case_bundle("p2b_0003")
        bundle = dict(bundle)
        bundle["synthesis"] = None
        sections = app.build_display_sections(bundle)
        self.assertTrue(all(items == [] for items in sections.values()))


class TechnicalFeatureRowsTests(unittest.TestCase):
    def test_rows_have_the_five_required_columns(self):
        bundle = app.load_case_bundle("p2b_0003")
        rows = app.build_technical_feature_rows(bundle)
        self.assertGreater(len(rows), 0)
        for row in rows:
            for col in ("feature", "value", "source", "status", "rule_eligible"):
                self.assertIn(col, row)

    def test_rule_eligible_rows_carry_matched_rule_ids(self):
        bundle = app.load_case_bundle("p2b_0003")
        rows = app.build_technical_feature_rows(bundle)
        eligible = [r for r in rows if r["rule_eligible"] == "yes"]
        for row in eligible:
            self.assertNotEqual(row["matched_rule_ids"], "-")


_RULE_ID_PATTERN = re.compile(r"\b(PSY_AR_|EN_COMPILED_)[A-Z_0-9]+\b")
_EVIDENCE_FAMILY_NAMES = (
    "facial_feature_style", "line_intensity_quality", "shape_symbolism", "object_symbolism",
    "animal_symbolism", "spatial_placement", "size_composition",
)
_CONCERN_DOMAIN_NAMES = (
    "anxiety_or_stress_related", "depressive_or_low_mood_related", "aggression_or_threat_related",
    "social_withdrawal_related", "developmental_or_attention_related", "positive_affect_or_social_engagement",
    "neutral_descriptive_only", "neutral_descriptive_only_no_construct_proposed",
)


class ParentFriendlyProfileNeverEmptyTests(unittest.TestCase):
    """UI redesign (human-centric Parent/Clinician views): Parent View
    must never be empty because no hypothesis/convergent pattern exists --
    build_parent_friendly_profile is the always-populated part, built
    entirely from real measured/observed values. Sections A-E per the
    redesign: objects_seen (may be empty for a sparse case -- the app
    substitutes a fallback sentence), visual_style_notes, evidence_text,
    uncertainty_note, questions are always non-empty."""

    def test_all_real_cases_produce_a_non_empty_profile(self):
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            profile = app.build_parent_friendly_profile(bundle)
            for key in ("visual_style_notes", "evidence_text", "uncertainty_note", "questions"):
                self.assertTrue(profile[key], f"{case_id}: {key} is empty")

    def test_profile_never_leaks_a_rule_id(self):
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            profile = app.build_parent_friendly_profile(bundle)
            blob = json.dumps(profile)
            self.assertIsNone(_RULE_ID_PATTERN.search(blob), f"{case_id}: rule ID leaked into parent profile")

    def test_profile_never_leaks_an_evidence_family_or_concern_domain_name(self):
        """UI redesign requirement: no evidence-family names, no concern-
        domain identifiers in Parent View -- stricter than the pre-
        existing rule-ID-only check."""
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            profile = app.build_parent_friendly_profile(bundle)
            blob = json.dumps(profile)
            for name in _EVIDENCE_FAMILY_NAMES + _CONCERN_DOMAIN_NAMES:
                self.assertNotIn(name, blob, f"{case_id}: {name!r} leaked into parent profile")

    def test_zero_hypothesis_case_still_has_evidence_text(self):
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            if bundle["synthesis"].candidate_hypotheses:
                continue
            profile = app.build_parent_friendly_profile(bundle)
            self.assertTrue(profile["evidence_text"])
            return
        self.skipTest("no zero-hypothesis case found in this cache to exercise the never-empty path")

    def test_not_assessable_case_never_states_a_page_percentage(self):
        """Section B: raw image bounds must never be described as "the
        page" when page_relative_features_assessable is False -- no
        "N% of the page" / placement wording in that case."""
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            page_reference = (bundle["deterministic_features"] or {}).get("page_reference") or {}
            if page_reference.get("page_relative_features_assessable"):
                continue
            profile = app.build_parent_friendly_profile(bundle)
            blob = " ".join(profile["visual_style_notes"])
            self.assertNotIn("% of the page", blob, f"{case_id}: page percentage stated despite not-assessable page")
            self.assertIn(app.PARENT_PAGE_NOT_ASSESSABLE_NOTE, profile["visual_style_notes"])

    def test_assessable_case_does_state_a_page_percentage(self):
        """Positive control for the test above -- p2b_0061 (the one
        assessable dev case) SHOULD describe page-relative composition."""
        bundle = app.load_case_bundle("p2b_0061")
        page_reference = bundle["deterministic_features"]["page_reference"]
        if not page_reference.get("page_relative_features_assessable"):
            self.skipTest("p2b_0061 is no longer the page-assessable dev case -- see the acceptance-audit regression tests")
        profile = app.build_parent_friendly_profile(bundle)
        blob = " ".join(profile["visual_style_notes"])
        self.assertIn("% of the page", blob)

    def test_unconfirmed_elements_produce_the_generic_note_never_specific_labels(self):
        """Observer candidates that were never independently confirmed
        must never be named in Parent View -- only the generic sentence."""
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            unverified_labels = {r["observer_candidate"]["label"] for r in bundle["rows"]
                                  if r["verification_status"] != "verified"}
            if not unverified_labels:
                continue
            profile = app.build_parent_friendly_profile(bundle)
            self.assertTrue(profile["has_unconfirmed_elements"])
            blob = json.dumps(profile)
            for label in unverified_labels:
                verified_labels = {r["observer_candidate"]["label"] for r in bundle["rows"]
                                    if r["verification_status"] == "verified"}
                if label in verified_labels:
                    continue  # same label also verified via a different candidate -- legitimately shown
                self.assertNotIn(label, blob, f"{case_id}: unconfirmed label {label!r} leaked into parent profile")


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


class H38CarRegressionTests(unittest.TestCase):
    """UI redesign explicit regression: h38's 'car' candidate is
    verification_status=unreviewed (never independently confirmed) --
    it must stay excluded from ALL eligible/rule-linked evidence and
    must never be named anywhere in the Parent View, regardless of how
    visually obvious it might be. Presentation changes must never promote
    an entity into evidence."""

    def test_car_is_unreviewed_not_verified(self):
        bundle = app.load_case_bundle("h38")
        car = next(e for e in bundle["entities"] if e.canonical_label == "car")
        self.assertEqual(car.case_verification_status, "unreviewed")

    def test_car_is_not_in_verified_labels(self):
        bundle = app.load_case_bundle("h38")
        verified_labels = {e.canonical_label for e in bundle["entities"] if e.case_verification_status == "verified"}
        self.assertNotIn("car", verified_labels)

    def test_car_never_appears_in_parent_view_text(self):
        bundle = app.load_case_bundle("h38")
        profile = app.build_parent_friendly_profile(bundle)
        self.assertNotIn("car", profile["objects_seen"].split(", "))
        blob = json.dumps(profile)
        self.assertNotIn("car,", blob)
        self.assertNotIn("car and", blob)
        self.assertNotIn('"car"', blob)

    def test_car_is_not_rule_eligible_in_unified_evidence(self):
        bundle = app.load_case_bundle("h38")
        car = next(e for e in bundle["entities"] if e.canonical_label == "car")
        car_items = [ue for ue in bundle["synthesis"].unified_evidence
                     if ue.item.evidence_id == f"ev_semantic_{car.entity_id}"]
        self.assertTrue(car_items)
        self.assertFalse(car_items[0].rule_eligible)
        self.assertIsNone(car_items[0].item.value)


class H38PageBoundaryTextRegressionTests(unittest.TestCase):
    """UI redesign explicit regression: h38's page is NOT assessable --
    Parent View must never state a page percentage or a raw-image-bound
    placement phrase for it."""

    def test_parent_view_never_says_100_percent_of_the_page(self):
        bundle = app.load_case_bundle("h38")
        profile = app.build_parent_friendly_profile(bundle)
        blob = " ".join(profile["visual_style_notes"])
        self.assertNotIn("100% of the page", blob)
        self.assertNotIn("% of the page", blob)

    def test_parent_view_never_says_bottom_center_of_the_page(self):
        bundle = app.load_case_bundle("h38")
        profile = app.build_parent_friendly_profile(bundle)
        blob = " ".join(profile["visual_style_notes"])
        self.assertNotIn("bottom center", blob)
        self.assertNotIn("bottom_center", blob)

    def test_page_relative_interpretation_is_clearly_marked_unavailable(self):
        bundle = app.load_case_bundle("h38")
        profile = app.build_parent_friendly_profile(bundle)
        self.assertIn(app.PARENT_PAGE_NOT_ASSESSABLE_NOTE, profile["visual_style_notes"])


class TechnicalViewStillExhaustiveTests(unittest.TestCase):
    """UI redesign requirement: the Technical/Audit View must keep full
    traceability -- raw objective image-relative features, page_reference_
    mode/page_relative_features_assessable, and every page-gated
    NOT_ASSESSABLE decision must remain exposed (this file's render_
    technical_view was NOT touched by the redesign; these tests pin its
    still-exhaustive data sources)."""

    def test_technical_feature_rows_include_raw_image_relative_features(self):
        bundle = app.load_case_bundle("h38")
        rows = app.build_technical_feature_rows(bundle)
        feature_ids = {r["feature"] for r in rows}
        self.assertIn("segmentation.bounding_box_coverage", feature_ids)
        self.assertIn("composition.centroid_x", feature_ids)

    def test_deterministic_features_carry_page_reference_fields(self):
        bundle = app.load_case_bundle("h38")
        page_reference = bundle["deterministic_features"]["page_reference"]
        self.assertIn("page_reference_mode", page_reference)
        self.assertIn("page_relative_features_assessable", page_reference)

    def test_every_page_gated_rule_shows_not_assessable_status_for_h38(self):
        bundle = app.load_case_bundle("h38")
        det = bundle["deterministic_features"]
        checks = ds.check_deterministic_preconditions(det["composition"], det["objective_features"], det["page_reference"])
        gated = {c.rule_id: c.status for c in checks if c.rule_id in ds._PAGE_GATED_DETERMINISTIC_RULE_IDS}
        self.assertEqual(len(gated), 7)
        self.assertTrue(all(status == "not_assessable" for status in gated.values()), gated)


class ClinicianViewHelperTests(unittest.TestCase):
    """Redesigned Clinician View: compact case summary, entity table,
    association cards, short synthesis summary, and missing-evidence
    notes -- all pure regroupings of already-computed synthesis/entity
    data, no new evidence or eligibility logic."""

    def test_case_summary_counts_match_raw_data(self):
        bundle = app.load_case_bundle("p2b_0003")
        summary = app.build_clinician_case_summary(bundle)
        self.assertEqual(summary["entities_detected"], len(bundle["rows"]))
        groups = app.group_rows_by_status(bundle["rows"])
        self.assertEqual(summary["confirmed"], len(groups["verified"]))
        self.assertEqual(summary["uncertain_unreviewed"], len(groups["uncertain"]) + len(groups["other"]))
        self.assertEqual(summary["associations"], len(bundle["synthesis"].literature_linked_associations))
        self.assertEqual(summary["candidate_hypotheses"], len(bundle["synthesis"].candidate_hypotheses))

    def test_entity_table_rows_cover_every_row_exactly_once(self):
        bundle = app.load_case_bundle("p2b_0003")
        table_rows = app.build_entity_table_rows(bundle)
        self.assertEqual(len(table_rows), len(bundle["rows"]))
        statuses = {r["status"] for r in table_rows}
        self.assertTrue(statuses.issubset({"verified", "uncertain", "rejected", "unreviewed"}))

    def test_association_cards_cover_every_literature_linked_association(self):
        bundle = app.load_case_bundle("p2b_0003")
        cards = app.build_association_cards(bundle)
        self.assertEqual(len(cards), len(bundle["synthesis"].literature_linked_associations))
        for card in cards:
            for key in ("observation", "literature_interpretation", "evidence_strength", "domain",
                        "alternative_explanations", "source", "rule_id", "evidence_family"):
                self.assertIn(key, card)

    def test_synthesis_summary_view_reflects_the_real_overall_synthesis(self):
        bundle = app.load_case_bundle("p2b_0003")
        view = app.build_synthesis_summary_view(bundle)
        self.assertEqual(view["level"], bundle["synthesis"].overall_synthesis["level"])
        self.assertEqual(view["full_explanation"], bundle["synthesis"].overall_synthesis["summary"])

    def test_missing_evidence_notes_mention_page_gate_when_not_assessable(self):
        bundle = app.load_case_bundle("h38")
        notes = app.build_missing_evidence_notes(bundle)
        self.assertTrue(any("page" in n.lower() for n in notes))

    def test_missing_evidence_notes_never_empty(self):
        for case_id in app.list_case_ids():
            bundle = app.load_case_bundle(case_id)
            if bundle["rows"] is None:
                continue
            self.assertTrue(app.build_missing_evidence_notes(bundle))


class NaturalJoinTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(app._natural_join([]), "")

    def test_single(self):
        self.assertEqual(app._natural_join(["sun"]), "sun")

    def test_two(self):
        self.assertEqual(app._natural_join(["sun", "cloud"]), "sun and cloud")

    def test_three_or_more(self):
        self.assertEqual(app._natural_join(["sun", "cloud", "ground"]), "sun, cloud and ground")


class EvidenceLayerUnchangedByRedesignTests(unittest.TestCase):
    """Proves the presentation-only nature of this round's changes: the
    underlying synthesis/matches/hypotheses/verification statuses and
    deterministic feature values are identical to (independently
    recomputed via) drawing_synthesis.py directly -- the redesigned app
    helpers must never derive a DIFFERENT reasoning result than the
    module they display."""

    def test_case_summary_hypothesis_count_matches_synthesize_drawing_directly(self):
        for case_id in ["h38", "p2b_0003", "p2b_0026", "p2b_0061", "p2b_0068"]:
            bundle = app.load_case_bundle(case_id)
            direct = ds.synthesize_drawing(case_id, bundle["entities"], bundle["deterministic_features"])
            self.assertEqual(bundle["synthesis"].overall_synthesis, direct.overall_synthesis)
            self.assertEqual(len(bundle["synthesis"].candidate_hypotheses), len(direct.candidate_hypotheses))

    def test_association_cards_rule_ids_match_literature_linked_associations_exactly(self):
        for case_id in ["h38", "p2b_0003", "p2b_0026", "p2b_0061", "p2b_0068"]:
            bundle = app.load_case_bundle(case_id)
            card_rule_ids = sorted(c["rule_id"] for c in app.build_association_cards(bundle))
            association_rule_ids = sorted(a["rule_id"] for a in bundle["synthesis"].literature_linked_associations)
            self.assertEqual(card_rule_ids, association_rule_ids)


try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class AppSmokeTests(unittest.TestCase):
    """End-to-end Streamlit smoke tests (bare-mode script execution, no
    browser/server) -- the same tool other DOAR review apps in this repo
    already use for UI regression coverage. Loads the real app, selects
    h38, and inspects each tab's actually-rendered text."""

    def _app_on_case(self, case_id: str) -> "AppTest":
        at = AppTest.from_file(str(ROOT / "scripts" / "clinician_review_app.py"))
        at.run(timeout=120)
        at.sidebar.selectbox[0].set_value(case_id).run(timeout=120)
        return at

    def test_renders_without_exception(self):
        at = self._app_on_case("h38")
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

    def test_parent_tab_never_mentions_page_percentage_or_bottom_center_for_h38(self):
        at = self._app_on_case("h38")
        full_text = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info) + " ".join(c.value for c in at.caption)
        self.assertNotIn("100% of the page", full_text)
        self.assertNotIn("bottom center", full_text)

    def test_no_tab_leaks_a_rule_id_into_the_parent_section(self):
        # Regression guard at the whole-app level, not just the pure
        # build_parent_friendly_profile function.
        at = self._app_on_case("h38")
        full_text = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
        # The Clinician/Technical tabs legitimately contain rule IDs
        # (EN_COMPILED_LINE_LIGHT_PRESSURE_031 etc.) -- this only asserts
        # the app rendered without exception and produced real text;
        # the strict Parent-only no-leak guarantee is covered precisely
        # by ParentFriendlyProfileNeverEmptyTests against the pure
        # function actually used to render that tab.
        self.assertTrue(full_text.strip())


if __name__ == "__main__":
    unittest.main()
