"""Builds `resources/psychology_sources/construct_registry.json`
(DOAR-TRACE Phase 1.5, Section 5).

A small, fixed, reviewable set of drawing-LEVEL patterns -- not one
construct per rule (that was Phase 1's design, and it is exactly why a
single rule could be labeled a "theme": see
`docs/CONSTRUCT_MAPPING_RATIONALE.md` for the full before/after
comparison). Rules map to at most one construct each, and only when the
rule's own interpretation text genuinely fits the construct's definition
-- rules whose interpretation is a personality trait rather than a
drawing-level emotional/social/energy pattern (e.g. "stubbornness",
"sneakiness", "believes themselves superior") are deliberately left
unmapped (`target_construct: null`) rather than forced into one of the 12
buckets. See `docs/CONSTRUCT_MAPPING_RATIONALE.md` for the per-rule
decision and the rules that were deliberately NOT mapped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONSTRUCT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "construct_registry.json"
SCHEMA_VERSION = "construct_registry_v1"

CONSTRUCTS: list[dict[str, Any]] = [
    {
        "construct_id": "low_mood_or_emotional_distress_pattern",
        "display_name_en": "Possible low-mood or emotional-distress pattern",
        "display_name_ar": "نمط محتمل لانخفاض المزاج أو الضيق الانفعالي",
        "description": "Multiple independent drawing features that, together, can sometimes appear in drawings associated with sadness, withdrawal, or emotional distress.",
        "allowed_wording": ["possible low-mood or emotional-distress pattern", "features that can sometimes appear in drawings associated with sadness, withdrawal, or emotional distress"],
        "prohibited_wording": ["the child is depressed", "the child is sad", "this proves distress", "الطفل مكتئب", "يعاني من اكتئاب"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "at least one contributing family above purely speculative level (Weak/mixed or stronger)",
        "possible_supporting_families": ["facial_expression", "colour_mood_flags", "isolation_theme", "missing_body_part", "size_composition", "global_expressive_content_model"],
        "possible_contradicting_families": ["facial_expression_positive", "positive_affective_tone"],
        "limitations": ["Cannot establish the child's actual mood.", "Many of the same features have equally plausible non-emotional explanations (style, copying, task, chance)."],
        "contextual_questions": ["Would it be alright to ask your child what is happening in this scene?", "Have you noticed similar themes in other recent drawings?"],
        "professional_review_recommendation": "Recommended if this pattern recurs across multiple drawings or is accompanied by real-world behavioural concerns.",
    },
    {
        "construct_id": "fear_or_insecurity_pattern",
        "display_name_en": "Possible fear or insecurity pattern",
        "display_name_ar": "نمط محتمل للخوف أو عدم الأمان",
        "description": "Multiple independent drawing features that can sometimes appear in drawings associated with fear, anxiety, or a lack of confidence.",
        "allowed_wording": ["possible fear or insecurity pattern", "features that can sometimes appear in drawings associated with fear or low confidence"],
        "prohibited_wording": ["the child is anxious", "the child has anxiety", "الطفل قلق", "يعاني من قلق"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "at least one contributing family above purely speculative level",
        "possible_supporting_families": ["size_composition", "line_quality", "facial_expression", "repeated_theme", "missing_body_part", "global_expressive_content_model"],
        "possible_contradicting_families": ["high_visual_energy_or_expansiveness"],
        "limitations": ["Small size or shaky lines can equally reflect motor skill, care, or drawing style.", "Cannot establish the child's actual emotional state."],
        "contextual_questions": ["Is there anything about this drawing your child seemed hesitant or careful about?"],
        "professional_review_recommendation": "Recommended if this pattern recurs or coincides with real-world signs of anxiety.",
    },
    {
        "construct_id": "tension_or_anger_pattern",
        "display_name_en": "Possible tension or anger pattern",
        "display_name_ar": "نمط محتمل للتوتر أو الغضب",
        "description": "Multiple independent drawing features that can sometimes appear in drawings associated with tension, frustration, or anger.",
        "allowed_wording": ["possible tension or anger pattern", "features that can sometimes appear in drawings associated with tension or anger"],
        "prohibited_wording": ["the child is angry", "the child is aggressive", "الطفل غاضب", "عدواني"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "at least one contributing family above purely speculative level",
        "possible_supporting_families": ["facial_expression", "line_quality", "animal_symbolism", "global_expressive_content_model"],
        "possible_contradicting_families": ["positive_affective_tone"],
        "limitations": ["Heavy line pressure and animal choice both have strong non-emotional alternative explanations (motor style, media exposure, interest)."],
        "contextual_questions": ["Was there a story behind this drawing your child could tell you?"],
        "professional_review_recommendation": "Recommended if this pattern recurs alongside real-world behavioural concerns.",
    },
    {
        "construct_id": "visual_dominance_or_prominence",
        "display_name_en": "Possible visual prominence pattern",
        "display_name_ar": "نمط محتمل للبروز البصري",
        "description": "The main subject is drawn large and/or centrally placed, making it visually dominant on the page. Describes how the scene is organised, not the child's personality.",
        "allowed_wording": ["possible visual prominence pattern", "these observations describe how the scene is visually organised"],
        "prohibited_wording": ["the child has a dominant personality", "the child feels superior", "الطفل يشعر بالتفوق"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "no minimum beyond speculative -- this construct is compositional, not clinical",
        "possible_supporting_families": ["size_composition", "spatial_placement"],
        "possible_contradicting_families": ["caution_or_low_visual_energy"],
        "limitations": ["Page occupation reflects many things besides self-image: task instructions, available space, drawing order, or simple preference (REF_SIZE_DEPTH_2013)."],
        "contextual_questions": ["What made your child decide how much of the page to use?"],
        "professional_review_recommendation": "Not typically needed on its own; relevant mainly in combination with mood-related patterns.",
    },
    {
        "construct_id": "social_distance_or_isolation",
        "display_name_en": "Possible social-distance or isolation pattern",
        "display_name_ar": "نمط محتمل للتباعد الاجتماعي أو العزلة",
        "description": "Multiple independent drawing features that can sometimes appear in drawings associated with loneliness or social withdrawal.",
        "allowed_wording": ["possible social-distance or isolation pattern"],
        "prohibited_wording": ["the child is lonely", "the child is isolated", "الطفل وحيد"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "at least one contributing family above purely speculative level",
        "possible_supporting_families": ["shape_symbolism", "isolation_theme", "spatial_placement"],
        "possible_contradicting_families": ["affiliation_or_connection"],
        "limitations": ["Circle preference and left-side placement both have weak, non-standardized literature support."],
        "contextual_questions": ["Who else, if anyone, does your child imagine being part of this scene?"],
        "professional_review_recommendation": "Recommended if this pattern recurs or coincides with real-world social withdrawal.",
    },
    {
        "construct_id": "affiliation_or_connection",
        "display_name_en": "Possible affiliation or connection pattern",
        "display_name_ar": "نمط محتمل للانتماء أو التواصل",
        "description": "Multiple independent drawing features that can sometimes appear in drawings associated with sociability or warmth toward others.",
        "allowed_wording": ["possible affiliation or connection pattern"],
        "prohibited_wording": ["the child is very social", "الطفل اجتماعي جداً"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "no minimum beyond speculative",
        "possible_supporting_families": ["object_symbolism", "spatial_placement"],
        "possible_contradicting_families": ["social_distance_or_isolation"],
        "limitations": ["Hearts and vehicles are culturally common symbols, not psychometric indicators."],
        "contextual_questions": ["Who is this drawing for, or who does it remind your child of?"],
        "professional_review_recommendation": "Not typically needed on its own.",
    },
    {
        "construct_id": "protection_or_coping",
        "display_name_en": "Possible protection or coping theme",
        "display_name_ar": "نمط محتمل للحماية أو التكيف",
        "description": "Drawing content that can sometimes reflect a wish for protection, safety, or care.",
        "allowed_wording": ["possible protection or coping theme"],
        "prohibited_wording": ["the child feels unsafe", "الطفل يشعر بعدم الأمان"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "no minimum beyond speculative",
        "possible_supporting_families": ["animal_symbolism"],
        "possible_contradicting_families": [],
        "limitations": ["Squirrel/protection association has no fixed universal rule found in the source material."],
        "contextual_questions": ["Does your child ever talk about wanting to feel safer or cared for in a specific situation?"],
        "professional_review_recommendation": "Not typically needed on its own.",
    },
    {
        "construct_id": "positive_affective_tone",
        "display_name_en": "Possible positive affective tone",
        "display_name_ar": "نمط محتمل لنبرة انفعالية إيجابية",
        "description": "Multiple independent drawing features that can sometimes appear in drawings associated with a positive or cheerful mood.",
        "allowed_wording": ["possible positive affective tone", "features that can sometimes appear in drawings associated with a positive mood"],
        "prohibited_wording": ["the child is happy", "proves the child is fine", "الطفل سعيد"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "no minimum beyond speculative",
        "possible_supporting_families": ["facial_expression", "shape_symbolism", "global_expressive_content_model"],
        "possible_contradicting_families": ["low_mood_or_emotional_distress_pattern"],
        "limitations": ["A smiling face or a sunny scene does not confirm a child's actual emotional state."],
        "contextual_questions": ["What is happening in this happy-looking scene?"],
        "professional_review_recommendation": "Not applicable -- positive findings do not require professional follow-up.",
    },
    {
        "construct_id": "disorganisation_or_fragmentation",
        "display_name_en": "Possible disorganisation or fragmentation pattern",
        "display_name_ar": "نمط محتمل لعدم التنظيم أو التشتت",
        "description": "Multiple independent drawing features that can sometimes appear in drawings that look unsettled, incomplete, or difficult to organize.",
        "allowed_wording": ["possible disorganisation or fragmentation pattern"],
        "prohibited_wording": ["the child has a disorder", "اضطراب"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "at least one contributing family above purely speculative level",
        "possible_supporting_families": ["line_quality", "missing_body_part", "composition_omission"],
        "possible_contradicting_families": [],
        "limitations": ["Shaky lines and background neglect are strongly confounded with age and motor development."],
        "contextual_questions": ["Was your child tired, rushed, or interrupted while drawing this?"],
        "professional_review_recommendation": "Recommended only if strongly recurrent and paired with real-world concerns.",
    },
    {
        "construct_id": "repetition_or_rigidity",
        "display_name_en": "Possible repetition or rigidity pattern",
        "display_name_ar": "نمط محتمل للتكرار أو الجمود",
        "description": "A preference for repeated, controlled, or highly detailed forms that can sometimes reflect a need for structure or persistence.",
        "allowed_wording": ["possible repetition or rigidity pattern"],
        "prohibited_wording": ["the child is obsessive", "وسواسي"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "no minimum beyond speculative",
        "possible_supporting_families": ["shape_symbolism", "detail_fixation"],
        "possible_contradicting_families": [],
        "limitations": ["Geometric-shape preference and detail fixation are both weak, non-standardized heuristics."],
        "contextual_questions": ["Does your child often prefer very structured or repeated patterns in other activities too?"],
        "professional_review_recommendation": "Not typically needed on its own.",
    },
    {
        "construct_id": "high_visual_energy_or_expansiveness",
        "display_name_en": "Possible high visual energy or expansiveness pattern",
        "display_name_ar": "نمط محتمل للطاقة البصرية العالية أو التوسع",
        "description": "Bold, large, or heavily-marked drawing choices that can sometimes reflect strong energy or confident self-expression.",
        "allowed_wording": ["possible high visual energy or expansiveness pattern"],
        "prohibited_wording": ["the child is hyperactive", "مفرط النشاط"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "no minimum beyond speculative",
        "possible_supporting_families": ["size_composition", "line_quality"],
        "possible_contradicting_families": ["caution_or_low_visual_energy"],
        "limitations": ["Full-page use and heavy pressure both have simple non-clinical explanations (drawing habit, tool used, excitement about the task)."],
        "contextual_questions": ["Was your child particularly excited about this drawing?"],
        "professional_review_recommendation": "Not typically needed on its own.",
    },
    {
        "construct_id": "caution_or_low_visual_energy",
        "display_name_en": "Possible caution or low visual energy pattern",
        "display_name_ar": "نمط محتمل للحذر أو انخفاض الطاقة البصرية",
        "description": "Small, light, or minimal drawing choices that can sometimes reflect carefulness, hesitation, or lower energy.",
        "allowed_wording": ["possible caution or low visual energy pattern"],
        "prohibited_wording": ["the child lacks energy", "الطفل خامل"],
        "minimum_independent_evidence_families": 2,
        "minimum_evidence_grade_requirement": "no minimum beyond speculative",
        "possible_supporting_families": ["size_composition", "line_quality"],
        "possible_contradicting_families": ["high_visual_energy_or_expansiveness"],
        "limitations": ["Small size and light pressure both have simple non-clinical explanations (carefulness, tool used, available time)."],
        "contextual_questions": ["Did your child seem to be taking their time or being extra careful with this drawing?"],
        "professional_review_recommendation": "Not typically needed on its own.",
    },
]


def build_construct_registry() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "draft -- fixed, small, reviewable construct set; not scientifically validated",
        "construct_count": len(CONSTRUCTS),
        "constructs": CONSTRUCTS,
    }


def write_construct_registry() -> Path:
    document = build_construct_registry()
    CONSTRUCT_REGISTRY_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return CONSTRUCT_REGISTRY_PATH


if __name__ == "__main__":
    path = write_construct_registry()
    print(f"wrote {path}")
