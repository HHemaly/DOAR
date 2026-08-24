"""Tests for src/doar/case_presentation.py -- the presentation-layer
bilingual (English/Arabic) label module built for the pre-doctor
stabilization pass. Pure functions only, no Streamlit/network -- verifies
(1) every closed-vocabulary id actually used by case_interpretation.py's
real registries has a translation, (2) lookups never crash on an unknown
id, (3) synthesis_summary/concern_domain_reason never invent a support
level or evidence-family count beyond what was already computed."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar import case_presentation as cp


class ClosedVocabularyCoverageTests(unittest.TestCase):
    """Every observable_feature/evidence_family/concern_domain value the
    frozen registries actually contain must have a translation -- these
    tests read the SAME files case_interpretation.py reads, never a
    hardcoded copy, so they catch drift if the registries ever change."""

    def test_every_observable_feature_in_the_matrix_has_a_label(self):
        with open(ROOT / "RULE_EVIDENCE_MATRIX.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        features = {r["observable_feature"] for r in rows if r.get("observable_feature")}
        missing = features - set(cp.OBSERVATION_LABELS)
        self.assertEqual(missing, set(), f"missing OBSERVATION_LABELS entries: {sorted(missing)}")

    def test_every_evidence_family_in_the_matrix_has_a_label(self):
        with open(ROOT / "RULE_EVIDENCE_MATRIX.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        families = {r["evidence_family"] for r in rows if r.get("evidence_family")}
        missing = families - set(cp.EVIDENCE_FAMILY_LABELS)
        self.assertEqual(missing, set(), f"missing EVIDENCE_FAMILY_LABELS entries: {sorted(missing)}")

    def test_every_domain_in_the_concern_domain_map_has_a_label(self):
        domains = json.loads((ROOT / "CONCERN_DOMAIN_MAP.json").read_text(encoding="utf-8"))["domains"]
        missing = set(domains) - set(cp.DOMAIN_LABELS)
        self.assertEqual(missing, set(), f"missing DOMAIN_LABELS entries: {sorted(missing)}")

    def test_every_label_entry_has_both_languages_non_empty(self):
        for name, table in (("DOMAIN_LABELS", cp.DOMAIN_LABELS), ("OBSERVATION_LABELS", cp.OBSERVATION_LABELS),
                             ("EVIDENCE_FAMILY_LABELS", cp.EVIDENCE_FAMILY_LABELS),
                             ("SUPPORT_LEVEL_LABELS", cp.SUPPORT_LEVEL_LABELS),
                             ("VERIFICATION_STATUS_LABELS", cp.VERIFICATION_STATUS_LABELS),
                             ("CONSISTENCY_LABELS", cp.CONSISTENCY_LABELS)):
            for key, entry in table.items():
                self.assertTrue(entry.get("en", "").strip(), f"{name}[{key!r}] missing English")
                self.assertTrue(entry.get("ar", "").strip(), f"{name}[{key!r}] missing Arabic")


class LookupFallbackTests(unittest.TestCase):
    """An id outside the closed vocabulary (only possible if the frozen
    registries change) must never raise -- falls back to an
    underscore-replaced label instead."""

    def test_unknown_domain_falls_back_without_raising(self):
        self.assertEqual(cp.domain_label("some_future_domain", "en"), "some future domain")

    def test_unknown_observation_falls_back_without_raising(self):
        self.assertEqual(cp.observation_label("brand_new_feature", "ar"), "brand new feature")

    def test_unknown_support_level_falls_back(self):
        self.assertEqual(cp.support_level_label("SOMETHING_NEW", "en"), "SOMETHING NEW")


class KnownEnumCoverageTests(unittest.TestCase):
    """Every real enum value case_interpretation.py can emit (pinned from
    reading the source directly this pass) must be covered."""

    def test_support_level_enum(self):
        for level in ("NONE", "WEAK", "MODERATE", "STRONG", "INSUFFICIENT", "NOT_CURRENTLY_ASSESSED"):
            self.assertIn(level, cp.SUPPORT_LEVEL_LABELS)

    def test_verification_status_enum(self):
        for status in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
                        "INSUFFICIENT_EVIDENCE", "NOT_CURRENTLY_ASSESSED"):
            self.assertIn(status, cp.VERIFICATION_STATUS_LABELS)

    def test_consistency_enum(self):
        for status in ("CONSISTENT", "MIXED", "CONFLICT", "INSUFFICIENT_EVIDENCE",
                        "GLOBAL_LOCAL_CONSISTENT", "GLOBAL_LOCAL_MIXED", "GLOBAL_LOCAL_CONFLICT"):
            self.assertIn(status, cp.CONSISTENCY_LABELS)


class RicherDescriptorTranslationTests(unittest.TestCase):
    """Pins _derive_richer_descriptors' exact closed (label, reason)
    vocabulary (case_interpretation.py) -- these were leaking untranslated
    English into the Arabic Parent/Psychologist views before this pass."""

    def test_every_known_label_has_an_arabic_translation(self):
        for label in ("cheerful", "positive", "negative", "tense", "withdrawn-looking"):
            ar = cp.descriptor_label(label, "ar")
            self.assertNotEqual(ar, label)

    def test_static_reason_templates_translate(self):
        for reason in (
            "Happy expressive classification, together with positive-affect-related visual evidence.",
            "The expressive classifier's top class is Happy.",
            "Line-pressure appearance associated in the literature with tension/energy was observed.",
            "Placement/composition evidence associated with social withdrawal was observed.",
        ):
            self.assertNotEqual(cp.descriptor_reason(reason, "ar"), reason)

    def test_dynamic_top_emotion_reason_translates_for_every_negative_emotion(self):
        for emotion in ("Sad", "Fear", "Angry"):
            reason = f"The expressive classifier's top class is {emotion}."
            translated = cp.descriptor_reason(reason, "ar")
            self.assertNotIn(emotion, translated)
            self.assertIn(cp.emotion_label(emotion, "ar"), translated)

    def test_english_is_returned_unchanged(self):
        reason = "The expressive classifier's top class is Sad."
        self.assertEqual(cp.descriptor_reason(reason, "en"), reason)

    def test_unknown_reason_falls_back_to_english_rather_than_mistranslating(self):
        reason = "Some future descriptor reason not yet in the closed set."
        self.assertEqual(cp.descriptor_reason(reason, "ar"), reason)


def _concern(support_level="WEAK", assessable_family_count=2, independent_evidence_families=("line_intensity_quality",),
             supporting_observations=("light_line_pressure_appearance",), plain_language_reason="EN fallback text",
             domain="anxiety_or_stress_related"):
    return SimpleNamespace(
        domain=domain, support_level=support_level, assessable_family_count=assessable_family_count,
        independent_evidence_families=independent_evidence_families,
        supporting_observations=supporting_observations, plain_language_reason=plain_language_reason)


class ConcernDomainReasonTests(unittest.TestCase):
    def test_zero_assessable_uses_arabic_sentence(self):
        c = _concern(support_level="INSUFFICIENT", assessable_family_count=0,
                     independent_evidence_families=(), supporting_observations=())
        reason = cp.concern_domain_reason(c, "ar")
        self.assertIn("لا يمكن", reason)

    def test_english_never_leaks_the_raw_observable_feature_id(self):
        """item C: no raw ID as the main user-facing label, in EITHER
        language -- plain_language_reason itself embeds the raw id
        (e.g. 'light_line_pressure_appearance'); concern_domain_reason
        must not."""
        c = _concern(supporting_observations=("light_line_pressure_appearance",))
        reason = cp.concern_domain_reason(c, "en")
        self.assertNotIn("light_line_pressure_appearance", reason)
        self.assertIn(cp.observation_label("light_line_pressure_appearance", "en"), reason)

    def test_english_never_invents_a_different_family_count(self):
        c = _concern(assessable_family_count=2, independent_evidence_families=("line_intensity_quality",))
        reason = cp.concern_domain_reason(c, "en")
        self.assertIn("1", reason)
        self.assertIn("2", reason)

    def test_arabic_never_invents_a_different_family_count(self):
        c = _concern(assessable_family_count=2, independent_evidence_families=("line_intensity_quality",))
        reason = cp.concern_domain_reason(c, "ar")
        self.assertIn("1", reason)
        self.assertIn("2", reason)

    def test_arabic_includes_the_translated_observation_label(self):
        c = _concern(supporting_observations=("light_line_pressure_appearance",))
        reason = cp.concern_domain_reason(c, "ar")
        self.assertIn(cp.observation_label("light_line_pressure_appearance", "ar"), reason)
        self.assertNotIn("light_line_pressure_appearance", reason)


class ParentQuestionLabelTests(unittest.TestCase):
    """Pins structured_report.py's `_suggested_parent_questions` closed
    vocabulary (verified against RULE_EVIDENCE_MATRIX.csv directly this
    pass: 4 fixed strings + 31 instances of one template)."""

    def test_every_fixed_question_translates(self):
        for q in cp._FIXED_PARENT_QUESTIONS_AR:
            self.assertNotEqual(cp.parent_question_label(q, "ar"), q)

    def test_templated_question_translates_via_the_observation_label(self):
        q = "Would you be willing to ask your child about the light_line_pressure_appearance in this drawing?"
        translated = cp.parent_question_label(q, "ar")
        self.assertNotEqual(translated, q)
        self.assertIn(cp.observation_label("light_line_pressure_appearance", "ar"), translated)

    def test_every_real_matrix_question_is_covered(self):
        import csv
        with open(ROOT / "RULE_EVIDENCE_MATRIX.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        questions = {r["clinical_question_to_ask_child_or_caregiver"] for r in rows
                     if r.get("clinical_question_to_ask_child_or_caregiver", "").strip()}
        untranslated = [q for q in questions if cp.parent_question_label(q, "ar") == q]
        self.assertEqual(untranslated, [], f"questions not covered by the closed-vocabulary translator: {untranslated}")

    def test_english_is_unchanged(self):
        q = "Can you tell me what is happening in this picture?"
        self.assertEqual(cp.parent_question_label(q, "en"), q)

    def test_unknown_question_falls_back_to_english(self):
        q = "A brand new question not in the matrix."
        self.assertEqual(cp.parent_question_label(q, "ar"), q)


class SynthesisSummaryTests(unittest.TestCase):
    def _interp(self, top_emotion="Happy", probs=None, concern_domains=(), consistency_status="CONSISTENT",
                gemini_global_observation=None):
        profile = SimpleNamespace(availability="available", top_emotion=top_emotion,
                                   base_emotion_probabilities=probs or {"Happy": 0.9, "Sad": 0.05, "Fear": 0.03, "Angry": 0.02})
        return SimpleNamespace(expressive_profile=profile, concern_domains=concern_domains,
                                consistency_status=consistency_status,
                                gemini_global_observation=gemini_global_observation)

    def test_english_names_the_top_emotion(self):
        interp = self._interp()
        summary = cp.synthesis_summary(interp, "en")
        self.assertIn("Happy", summary)
        self.assertIn("90%", summary)

    def test_arabic_names_the_top_emotion(self):
        interp = self._interp()
        summary = cp.synthesis_summary(interp, "ar")
        self.assertIn(cp.emotion_label("Happy", "ar"), summary)

    def test_no_notable_domains_says_so_in_both_languages(self):
        interp = self._interp()
        self.assertIn("No concern domain", cp.synthesis_summary(interp, "en"))
        self.assertIn("لم يصل", cp.synthesis_summary(interp, "ar"))

    def test_notable_domain_is_named_with_its_family_count(self):
        c = _concern(support_level="WEAK", independent_evidence_families=("line_intensity_quality",))
        interp = self._interp(concern_domains=[c])
        summary_en = cp.synthesis_summary(interp, "en")
        self.assertIn(cp.domain_label("anxiety_or_stress_related", "en"), summary_en)
        self.assertIn("1", summary_en)
        summary_ar = cp.synthesis_summary(interp, "ar")
        self.assertIn(cp.domain_label("anxiety_or_stress_related", "ar"), summary_ar)

    def test_conflict_consistency_adds_a_secondary_observation_caveat(self):
        c = _concern(support_level="MODERATE")
        interp = self._interp(concern_domains=[c], consistency_status="CONFLICT")
        summary = cp.synthesis_summary(interp, "en")
        self.assertIn("secondary observations", summary)
        self.assertIn("rather than diagnostic conclusions", summary)

    def test_gemini_tone_included_when_present(self):
        interp = self._interp(gemini_global_observation={"overall_visual_tone": "cheerful"})
        summary = cp.synthesis_summary(interp, "en")
        self.assertIn("cheerful", summary)

    def test_gemini_tone_omitted_when_absent(self):
        interp = self._interp(gemini_global_observation=None)
        summary = cp.synthesis_summary(interp, "en")
        self.assertNotIn("Gemini", summary)

    def test_unavailable_expressive_profile_says_so(self):
        interp = self._interp()
        interp.expressive_profile = SimpleNamespace(availability="unavailable", top_emotion=None,
                                                      base_emotion_probabilities=None)
        summary = cp.synthesis_summary(interp, "en")
        self.assertIn("No expressive-model result", summary)


if __name__ == "__main__":
    unittest.main()
