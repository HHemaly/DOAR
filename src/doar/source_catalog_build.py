"""Builds `resources/psychology_sources/source_rule_catalog.json`
(DOAR-TRACE Phase 1.5, Section 4).

Every row from BOTH source PDFs is catalogued as its own
`source_entry_id` -- rows that clearly describe the same underlying
observable are never silently merged; instead an explicit relationship
(`exact_duplicate`/`near_duplicate`/`expands`/`contradicts`/
`related_but_distinct`) is recorded between their source_entry_ids. This
module is the single place that transcription happened; `registry_v2_build.py`
consumes its output rather than re-deriving rule content from the PDFs a
second time.

Transcription method: both PDFs were extracted via `pypdf` (no
`pdftoppm`/poppler available in this environment for page-image
rendering) and read in full. `original_text` for the Arabic source is the
already-normalized transcription in `rules_registry.json` (that
registry's own documented normalization: "spelling and OCR-like spacing
were normalized without adding missing clinical meaning" -- reused here,
not re-transcribed a second time, to avoid a second, possibly divergent
transcription of the same source). `original_text` for the English
source is copied verbatim from this session's own `pypdf` extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "source_rule_catalog.json"
AR_PDF = "التحليل النفسي للصور.pdf"
EN_PDF = "child_drawing_rules_compiled.pdf"

SCHEMA_VERSION = "source_rule_catalog_v1"

# ---------------------------------------------------------------------------
# A. التحليل النفسي للصور.pdf -- 19 rows, matching the production registry's
# own already-normalized transcription (rules_registry.json), not re-derived.
# ---------------------------------------------------------------------------
_AR_ENTRIES: list[dict[str, Any]] = [
    {"id": "SRC_AR_001", "page": 1, "section": "رسم العيون (Eye drawing)", "observable": "wide_eyes",
     "text": "رسم العيون الواسعة يدل على الشخصية المنفتحة.", "interp": "outgoing personality"},
    {"id": "SRC_AR_002", "page": 1, "section": "رسم العيون", "observable": "stern_eyes",
     "text": "رسم العيون الصارمة يدل على الشعور بالغضب.", "interp": "feeling of anger"},
    {"id": "SRC_AR_003", "page": 1, "section": "رسم العيون", "observable": "closed_eyes",
     "text": "رسم العيون المغمضة يدل على رفض الشخص النظر إلى نفسه من الداخل ورفضه الاعتراف بعيوبه.",
     "interp": "refusal to look inward / acknowledge one's own faults"},
    {"id": "SRC_AR_004", "page": 1, "section": "رسم الحيوانات (Animal drawing)", "observable": "tiger_or_wolf",
     "text": "رسم النمر أو الذئب يدل على الشعور بالغضب.", "interp": "feeling of anger"},
    {"id": "SRC_AR_005", "page": 1, "section": "رسم الحيوانات", "observable": "fox",
     "text": "رسم الثعلب يدل على أن الرسام كان يفكر في عمل أمر لئيم أو خبيث.", "interp": "thinking of doing something mean/malicious"},
    {"id": "SRC_AR_006", "page": 1, "section": "رسم الحيوانات", "observable": "squirrel",
     "text": "رسم السنجاب يشير إلى حاجة الرسام للحماية والاهتمام.", "interp": "need for protection and attention"},
    {"id": "SRC_AR_007", "page": 1, "section": "رسم الحيوانات", "observable": "lion",
     "text": "رسم الأسد يشير إلى اعتقاد الشخص بأنه متفوق على الآخرين.", "interp": "belief in one's own superiority"},
    {"id": "SRC_AR_008", "page": 1, "section": "الأشكال الهندسية (Geometric shapes)", "observable": "repeated_geometric_shapes",
     "text": "حب رسم الأشكال الهندسية يشير إلى كثرة الأهداف والخطط والعناد والإصرار.",
     "interp": "many goals/plans, stubbornness, persistence"},
    {"id": "SRC_AR_009", "page": 1, "section": "الأشكال الهندسية — النجوم", "observable": "stars",
     "text": "رسم النجوم يشير إلى أن الشخص يريد أن يكون ملفتاً لنظر الجميع.", "interp": "wanting to be the center of attention"},
    {"id": "SRC_AR_010", "page": 1, "section": "الأشكال الهندسية — الزهور والسحب والشمس", "observable": "flowers_clouds_sun",
     "text": "رسم الزهور والسحب والشمس قد يعبر عن تخيل أهداف وأفكار جديدة وشخصية إيجابية ومزاج جيد.",
     "interp": "imagining new goals/ideas, positive personality, good mood (only if drawn while distracted/absent-minded)"},
    {"id": "SRC_AR_011", "page": 1, "section": "الأشكال الهندسية — الدوائر", "observable": "circles",
     "text": "رسم الدوائر يعبر عن الشعور بالوحدة والحاجة إلى شخص قريب يقف بجانبه.", "interp": "feeling of loneliness, need for someone close"},
    {"id": "SRC_AR_012", "page": 1, "section": "رسم وسائل النقل (Transportation)", "observable": "vehicles",
     "text": "رسم وسائل النقل يشير إلى حب السفر والعطلة والاجتماعية والانفتاح والمرح والمغامرة وحب الحياة.",
     "interp": "loves travel/vacations, social, outgoing, playful, adventurous"},
    {"id": "SRC_AR_013", "page": 2, "section": "رسم القلوب (Heart drawing)", "observable": "hearts",
     "text": "رسم القلوب يدل على شخصية عاطفية أو عاشقة واجتماعية ورومانسية لطيفة وقادرة على جذب قلوب الآخرين.",
     "interp": "affectionate/loving personality, social, romantic, able to attract others"},
    {"id": "SRC_AR_014", "page": 2, "section": "حجم الرسم (Drawing size)", "observable": "coverage_about_half",
     "text": "إذا غطت الرسمة نحو 50% من الورقة فقد تدل على الانفتاح أحياناً والانطواء أحياناً أخرى.",
     "interp": "outgoing at times, introverted at other times"},
    {"id": "SRC_AR_015", "page": 2, "section": "حجم الرسم", "observable": "coverage_full",
     "text": "إذا غطت الرسمة كل الورقة فهذا يدل على احترام الشخص العالي لنفسه وتقديره لها.", "interp": "high self-respect/self-esteem"},
    {"id": "SRC_AR_016", "page": 2, "section": "حجم الرسم", "observable": "coverage_small",
     "text": "إذا لم تتجاوز الرسمة 20% من الورقة فقد يدل ذلك على عدم الاستقرار أو الخوف.", "interp": "instability or fear"},
    {"id": "SRC_AR_017", "page": 2, "section": "موقع الرسم (Drawing location)", "observable": "placement_top",
     "text": "الرسم في أعلى الصفحة قد يعبر عن شخصية حالمة أو واقع خيالي وصعوبة في التكيف.",
     "interp": "dreamy personality, fantasy world, difficulty adapting"},
    {"id": "SRC_AR_018", "page": 2, "section": "موقع الرسم", "observable": "placement_left",
     "text": "الرسم في الجانب الأيسر يدل على شخصية انطوائية.", "interp": "introverted personality"},
    {"id": "SRC_AR_019", "page": 2, "section": "موقع الرسم", "observable": "placement_right",
     "text": "الرسم في الجانب الأيمن يدل على شخصية منفتحة.", "interp": "outgoing personality"},
]

# ---------------------------------------------------------------------------
# B. child_drawing_rules_compiled.pdf -- 48 rows (4 general-principle rows +
# 43 feature/candidate-rule rows + 1 closing "Bottom line" caution row).
# `original_text` copied verbatim (paraphrase-free) from this session's own
# pypdf extraction of the actual PDF. `evidence_level_as_written` and
# `notes_as_written` are the PDF's own columns, never invented.
# ---------------------------------------------------------------------------
_EN_ENTRIES: list[dict[str, Any]] = [
    # -- General principles (not candidate rules; source_type=general_guidance) --
    {"id": "SRC_EN_G01", "page": 1, "section": "How to use the guide", "observable": None,
     "text": "Single drawing: Can suggest a possibility, but cannot diagnose depression, abuse, anxiety, or personality.",
     "interp": None, "level": "Strong caution", "notes": None, "general": True},
    {"id": "SRC_EN_G02", "page": 1, "section": "How to use the guide", "observable": None,
     "text": "Patterns across drawings: Repeated themes over time are more informative than one picture.",
     "interp": None, "level": "More useful", "notes": None, "general": True},
    {"id": "SRC_EN_G03", "page": 1, "section": "How to use the guide", "observable": None,
     "text": "Child's explanation: Always ask what the drawing means to the child.",
     "interp": None, "level": "Very useful", "notes": None, "general": True},
    {"id": "SRC_EN_G04", "page": 1, "section": "How to use the guide", "observable": None,
     "text": "Age and skill: Development, motor skill, and cultural style can change the drawing a lot.",
     "interp": None, "level": "Essential context", "notes": None, "general": True},
    # -- Eyes and face --
    {"id": "SRC_EN_001", "page": 1, "section": "Eyes and face", "observable": "wide_eyes",
     "text": "Wide eyes: Often interpreted as openness or alertness; in some emotion-drawing work, wide eyes can also fit fear/surprise.",
     "interp": "openness or alertness; sometimes fear/surprise", "level": "Weak / mixed",
     "notes": "Children's facial-expression drawings research; general projective tradition."},
    {"id": "SRC_EN_002", "page": 1, "section": "Eyes and face", "observable": "stern_eyes",
     "text": "Strict, narrowed, or intense eyes: May be read as anger, tension, suspicion, or defensiveness.",
     "interp": "anger, tension, suspicion, or defensiveness", "level": "Weak / mixed",
     "notes": "General emotion-symbol heuristics."},
    {"id": "SRC_EN_003", "page": 1, "section": "Eyes and face", "observable": "closed_eyes",
     "text": "Closed eyes: Sometimes interpreted as avoidance, shutting out the world, or refusing to look inward.",
     "interp": "avoidance, shutting out the world, or refusing to look inward", "level": "Speculative",
     "notes": "No strong standard reference found."},
    {"id": "SRC_EN_004", "page": 1, "section": "Eyes and face", "observable": "eyes_little_detail_or_missing",
     "text": "Eyes with little detail or missing eyes: May suggest withdrawal, avoidance, low interest in contact, or simply limited drawing skill.",
     "interp": "withdrawal, avoidance, low interest in contact, or limited drawing skill", "level": "Weak / mixed",
     "notes": "Projective drawing heuristics; developmental factors important."},
    {"id": "SRC_EN_005", "page": 1, "section": "Eyes and face", "observable": "face_expression",
     "text": "Face expression: Smiling can suggest positive tone; sad, tense, or frightened faces can suggest distress.",
     "interp": "positive tone (smiling) or distress (sad/tense/frightened)", "level": "Moderate for basic emotion only",
     "notes": "Children do encode emotions in faces, but not as a diagnosis test."},
    # -- Animals --
    {"id": "SRC_EN_006", "page": 1, "section": "Animals", "observable": "tiger_or_wolf",
     "text": "Tiger or wolf: Often informally linked with anger, threat, or aggression.",
     "interp": "anger, threat, or aggression", "level": "Speculative", "notes": "No fixed universal rule found."},
    {"id": "SRC_EN_007", "page": 1, "section": "Animals", "observable": "fox",
     "text": "Fox: Sometimes informally linked with sneakiness or cunning.",
     "interp": "sneakiness or cunning", "level": "Speculative", "notes": "No fixed universal rule found."},
    {"id": "SRC_EN_008", "page": 1, "section": "Animals", "observable": "squirrel",
     "text": "Squirrel: Sometimes linked with need for protection, nervous energy, or seeking care.",
     "interp": "need for protection, nervous energy, or seeking care", "level": "Speculative",
     "notes": "No fixed universal rule found."},
    {"id": "SRC_EN_009", "page": 1, "section": "Animals", "observable": "lion",
     "text": "Lion: Sometimes linked with dominance, pride, strength, or superiority.",
     "interp": "dominance, pride, strength, or superiority", "level": "Speculative", "notes": "No fixed universal rule found."},
    {"id": "SRC_EN_010", "page": 1, "section": "Animals", "observable": "animal_choice_general",
     "text": "Animal choice in general: Can reflect interests, story play, fear, identification, or a projective theme.",
     "interp": "interests, story play, fear, identification, or a projective theme", "level": "Weak / mixed",
     "notes": "Animal-drawing tests and apperception traditions."},
    # -- Geometric and symbolic shapes --
    {"id": "SRC_EN_011", "page": 1, "section": "Geometric and symbolic shapes", "observable": "repeated_geometric_shapes",
     "text": "Geometric shapes preferred: Sometimes interpreted as planning, control, persistence, or stubbornness.",
     "interp": "planning, control, persistence, or stubbornness", "level": "Speculative",
     "notes": "Projective tradition, weak support."},
    {"id": "SRC_EN_012", "page": 1, "section": "Geometric and symbolic shapes", "observable": "stars",
     "text": "Stars: Sometimes read as wanting attention, admiration, or visibility.",
     "interp": "wanting attention, admiration, or visibility", "level": "Speculative",
     "notes": "No strong standard reference found."},
    {"id": "SRC_EN_013", "page": 1, "section": "Geometric and symbolic shapes", "observable": "circles",
     "text": "Circles: Sometimes associated with unity, wholeness, or the need for closeness; in your handout, loneliness and need for support.",
     "interp": "unity, wholeness, need for closeness; (per Arabic handout) loneliness and need for support",
     "level": "Speculative", "notes": "Common heuristic, not standardized."},
    {"id": "SRC_EN_014", "page": 1, "section": "Geometric and symbolic shapes", "observable": "flowers_clouds_sun",
     "text": "Flowers, sun, clouds: May be read as optimism, imagination, or positive mood when used in a dreamy context.",
     "interp": "optimism, imagination, or positive mood, when used in a dreamy context", "level": "Speculative",
     "notes": "No fixed rule; depends heavily on context."},
    {"id": "SRC_EN_015", "page": 2, "section": "Geometric and symbolic shapes (continued) / Hearts", "observable": "hearts",
     "text": "Hearts: Usually reflect affection, love, romance, or social warmth in a cultural sense.",
     "interp": "affection, love, romance, or social warmth", "level": "Weak / common-sense",
     "notes": "Symbolic meaning is culturally obvious, not psychometric."},
    # -- Animals and objects with stories --
    {"id": "SRC_EN_016", "page": 2, "section": "Animals and objects with stories", "observable": "vehicles",
     "text": "Transportation drawings: May suggest travel interest, movement, adventure, or a social/playful orientation.",
     "interp": "travel interest, movement, adventure, or a social/playful orientation", "level": "Speculative",
     "notes": "No strong literature basis found."},
    {"id": "SRC_EN_017", "page": 2, "section": "Animals and objects with stories", "observable": "house",
     "text": "House: Sometimes discussed as family life, safety, privacy, or emotional openness, but not a reliable standalone test.",
     "interp": "family life, safety, privacy, or emotional openness", "level": "Weak / mixed",
     "notes": "Projective drawing traditions."},
    {"id": "SRC_EN_018", "page": 2, "section": "Animals and objects with stories", "observable": "tree",
     "text": "Tree: Sometimes used to think about stability, growth, roots, or vulnerability.",
     "interp": "stability, growth, roots, or vulnerability", "level": "Weak / mixed",
     "notes": "Projective tree-drawing traditions."},
    {"id": "SRC_EN_019", "page": 2, "section": "Animals and objects with stories", "observable": "repeated_monsters_danger_injury",
     "text": "Repeated monsters, danger, injury themes: May indicate fear, distress, or preoccupation with threat; needs careful follow-up.",
     "interp": "fear, distress, or preoccupation with threat", "level": "Suggestive only",
     "notes": "Trauma-informed assessment principles."},
    # -- Size and page use --
    {"id": "SRC_EN_020", "page": 2, "section": "Size and page use", "observable": "coverage_full",
     "text": "Very large drawing / fills the page: Sometimes linked with expansiveness, strong energy, or self-assertion; your handout links it to high self-esteem.",
     "interp": "expansiveness, strong energy, self-assertion; (per Arabic handout) high self-esteem", "level": "Weak / mixed",
     "notes": "Page occupation is a common projective heuristic."},
    {"id": "SRC_EN_021", "page": 2, "section": "Size and page use", "observable": "coverage_about_half",
     "text": "Moderate drawing size (~half page): Sometimes read as balance or mixed approach; your handout says open at times and introverted at times.",
     "interp": "balance or mixed approach; (per Arabic handout) outgoing at times and introverted at times",
     "level": "Speculative", "notes": "No standardized cutoff."},
    {"id": "SRC_EN_022", "page": 2, "section": "Size and page use", "observable": "very_small_drawing",
     "text": "Very small drawing: May suggest insecurity, withdrawal, fear, or low confidence; could also reflect carefulness or skill limits.",
     "interp": "insecurity, withdrawal, fear, or low confidence", "level": "Weak / mixed",
     "notes": "Size is a common but nonspecific cue."},
    {"id": "SRC_EN_023", "page": 2, "section": "Size and page use", "observable": "coverage_small",
     "text": "Uses only a small part of the page: May suggest caution, uncertainty, or low confidence.",
     "interp": "caution, uncertainty, or low confidence", "level": "Weak / mixed", "notes": "Projective interpretation."},
    # -- Placement on the page --
    {"id": "SRC_EN_024", "page": 2, "section": "Placement on the page", "observable": "placement_top",
     "text": "Top of page: Sometimes read as fantasy, dreaming, ambition, or being less grounded in current reality.",
     "interp": "fantasy, dreaming, ambition, or being less grounded in current reality", "level": "Weak / mixed",
     "notes": "Projective page-position heuristics."},
    {"id": "SRC_EN_025", "page": 2, "section": "Placement on the page", "observable": "placement_left",
     "text": "Left side: Sometimes linked to introversion, past orientation, or dependence.",
     "interp": "introversion, past orientation, or dependence", "level": "Speculative",
     "notes": "No strong standard rule found."},
    {"id": "SRC_EN_026", "page": 2, "section": "Placement on the page", "observable": "placement_right",
     "text": "Right side: Sometimes linked to extroversion, future orientation, or outward movement.",
     "interp": "extroversion, future orientation, or outward movement", "level": "Speculative",
     "notes": "No strong standard rule found."},
    {"id": "SRC_EN_027", "page": 2, "section": "Placement on the page", "observable": "placement_center",
     "text": "Centering: Often read as balance, security, or comfort.",
     "interp": "balance, security, or comfort", "level": "Speculative", "notes": "Common heuristic."},
    # -- Line quality and pressure --
    {"id": "SRC_EN_028", "page": 2, "section": "Line quality and pressure", "observable": "heavy_line_pressure_appearance",
     "text": "Heavy pressure: May suggest tension, force, anger, or high energy.",
     "interp": "tension, force, anger, or high energy", "level": "Weak / mixed",
     "notes": "Not specific to emotion; can reflect motor style."},
    {"id": "SRC_EN_029", "page": 2, "section": "Line quality and pressure", "observable": "light_line_pressure_appearance",
     "text": "Light pressure: May suggest hesitation, shyness, low energy, or delicacy.",
     "interp": "hesitation, shyness, low energy, or delicacy", "level": "Weak / mixed", "notes": "Not specific to emotion."},
    {"id": "SRC_EN_030", "page": 2, "section": "Line quality and pressure", "observable": "shaky_or_broken_lines",
     "text": "Shaky or broken lines: May suggest anxiety, uncertainty, distress, or motor difficulty.",
     "interp": "anxiety, uncertainty, distress, or motor difficulty", "level": "Weak / mixed",
     "notes": "Needs comparison with age and motor skills."},
    {"id": "SRC_EN_031", "page": 2, "section": "Line quality and pressure", "observable": "zigzag_lines",
     "text": "Zigzag lines: Sometimes read as agitation, violence, unpredictability, or conflict.",
     "interp": "agitation, violence, unpredictability, or conflict", "level": "Weak / mixed",
     "notes": "Older projective and graphological traditions."},
    {"id": "SRC_EN_032", "page": 2, "section": "Line quality and pressure", "observable": "over_erasing_appearance",
     "text": "Over-erasing / frequent corrections: May suggest perfectionism, anxiety, frustration, or uncertainty.",
     "interp": "perfectionism, anxiety, frustration, or uncertainty", "level": "Weak / mixed",
     "notes": "Also common in normal development."},
    # -- Missing or altered details --
    {"id": "SRC_EN_033", "page": 3, "section": "Missing or altered details", "observable": "missing_hands",
     "text": "Missing hands: Sometimes interpreted as difficulty acting, social difficulty, or avoidance.",
     "interp": "difficulty acting, social difficulty, or avoidance", "level": "Weak / mixed", "notes": "Can be developmental."},
    {"id": "SRC_EN_034", "page": 3, "section": "Missing or altered details", "observable": "missing_mouth",
     "text": "Missing mouth: Sometimes linked with difficulty expressing emotion or communication issues.",
     "interp": "difficulty expressing emotion or communication issues", "level": "Weak / mixed", "notes": "Not specific."},
    {"id": "SRC_EN_035", "page": 3, "section": "Missing or altered details", "observable": "missing_eyes",
     "text": "Missing eyes: May suggest avoidance or emotional withdrawal.",
     "interp": "avoidance or emotional withdrawal", "level": "Weak / mixed", "notes": "Not specific."},
    {"id": "SRC_EN_036", "page": 3, "section": "Missing or altered details", "observable": "exaggerated_body_parts",
     "text": "Exaggerated body parts: Huge hands, teeth, eyes, etc. may suggest aggression, fear, or hypervigilance.",
     "interp": "aggression, fear, or hypervigilance", "level": "Weak / mixed", "notes": "Very tentative."},
    # -- Repeated themes --
    {"id": "SRC_EN_037", "page": 3, "section": "Repeated themes", "observable": "repeated_monsters_danger_injury",
     "text": "Repeated monsters or danger scenes: May reflect a recurring fear or preoccupation.",
     "interp": "a recurring fear or preoccupation", "level": "Suggestive only", "notes": "Best used with interview."},
    {"id": "SRC_EN_038", "page": 3, "section": "Repeated themes", "observable": "repeated_family_conflict",
     "text": "Repeated family conflict scenes: May indicate tension, perceived conflict, or concern in the home.",
     "interp": "tension, perceived conflict, or concern in the home", "level": "Suggestive only", "notes": "Context is essential."},
    {"id": "SRC_EN_039", "page": 3, "section": "Repeated themes", "observable": "repeated_isolation_themes",
     "text": "Repeated isolation themes: May point to loneliness or withdrawal.",
     "interp": "loneliness or withdrawal", "level": "Suggestive only", "notes": "Not diagnostic."},
    # -- Mood or concern flags sometimes discussed --
    {"id": "SRC_EN_040", "page": 3, "section": "Mood or concern flags sometimes discussed", "observable": "dark_colors_sad_faces_isolation",
     "text": "Dark colors, sad faces, isolation: Sometimes flagged as possible sadness, low mood, or distress.",
     "interp": "possible sadness, low mood, or distress", "level": "Suggestive only", "notes": "Color evidence is mixed; not diagnostic."},
    {"id": "SRC_EN_041", "page": 3, "section": "Mood or concern flags sometimes discussed", "observable": "refusal_to_draw_figures",
     "text": "Refusal to draw figures: May suggest distress, resistance, perfectionism, or lack of task engagement.",
     "interp": "distress, resistance, perfectionism, or lack of task engagement", "level": "Suggestive only", "notes": "Can have many causes."},
    {"id": "SRC_EN_042", "page": 3, "section": "Mood or concern flags sometimes discussed", "observable": "excessive_detail_or_fixation",
     "text": "Excessive detail or fixation: May suggest anxiety, obsession with parts of the task, or strong interest.",
     "interp": "anxiety, obsession with parts of the task, or strong interest", "level": "Suggestive only", "notes": "Not specific."},
    {"id": "SRC_EN_043", "page": 3, "section": "Mood or concern flags sometimes discussed", "observable": "neglect_of_background",
     "text": "Neglect of background: May reflect focus on the main subject, limited time, or developmental style.",
     "interp": "focus on the main subject, limited time, or developmental style", "level": "Weak / mixed",
     "notes": "Not a pathology sign by itself."},
    # -- Closing caution (not a rule) --
    {"id": "SRC_EN_C01", "page": 3, "section": "Bottom line", "observable": None,
     "text": ("Use these rules only as hypotheses. The most defensible reading comes from combining the drawing with the "
              "child's age, the drawing process, repeated patterns over time, and the child's own explanation. No "
              "single item here should be treated as proof of depression, abuse, anxiety, or personality type."),
     "interp": None, "level": None, "notes": None, "general": True},
]

# ---------------------------------------------------------------------------
# Explicit relationships. Every pair here is a real, checked overlap --
# nothing is silently merged; each side keeps its own source_entry_id and
# its own (possibly different) interpretation text.
# ---------------------------------------------------------------------------
RELATIONSHIPS: list[dict[str, Any]] = [
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_001", "SRC_EN_001"],
     "note": "Same observable (wide eyes). Arabic source states 'outgoing personality' unconditionally; English source hedges with 'openness/alertness, or sometimes fear/surprise' -- broader and less certain."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_002", "SRC_EN_002"],
     "note": "Same observable (stern/narrowed eyes -> anger); English source broadens to include tension/suspicion/defensiveness."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_003", "SRC_EN_003"],
     "note": "Same observable and near-identical interpretation (closed eyes -> refusing to look inward / shutting out the world)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_004", "SRC_EN_006"],
     "note": "Same observable (tiger/wolf -> anger); English source adds threat/aggression."},
    {"type": "related_but_distinct", "source_entry_ids": ["SRC_AR_005", "SRC_EN_007"],
     "note": "Same observable (fox) but different specific interpretation: Arabic = malicious intent, English = sneakiness/cunning. Related valence, not identical claim."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_006", "SRC_EN_008"],
     "note": "Same observable and interpretation (squirrel -> need for protection); English adds nervous energy/seeking care."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_007", "SRC_EN_009"],
     "note": "Same observable and interpretation (lion -> superiority); English adds dominance/pride/strength."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_008", "SRC_EN_011"],
     "note": "Same observable (repeated geometric shapes -> stubbornness/persistence/goals/planning)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_009", "SRC_EN_012"],
     "note": "Same observable (stars -> wanting attention/admiration/visibility)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_010", "SRC_EN_014"],
     "note": "Same observable (flowers/sun/clouds -> optimism/positive mood). Arabic requires a strict behavioral precondition (drawn while distracted/absent-minded); English softens this to a vaguer 'dreamy context' -- the precondition is weakened, not identical."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_011", "SRC_EN_013"],
     "note": "Same observable; English source explicitly cites the Arabic handout's loneliness reading as part of its own broader interpretation (unity/wholeness/closeness)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_012", "SRC_EN_016"],
     "note": "Same observable and near-identical interpretation (vehicles -> travel/adventure/sociability)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_013", "SRC_EN_015"],
     "note": "Same observable and near-identical interpretation (hearts -> affection/love/social warmth)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_014", "SRC_EN_021"],
     "note": "Same observable; English source explicitly cites the Arabic handout's own wording ('open at times and introverted at times')."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_015", "SRC_EN_020"],
     "note": "Same observable; English source explicitly cites the Arabic handout's 'high self-esteem' reading."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_016", "SRC_EN_023"],
     "note": "Same observable (page-area coverage, not absolute figure size) -- both frame the measure as a fraction of the page used, matching most directly."},
    {"type": "related_but_distinct", "source_entry_ids": ["SRC_AR_016", "SRC_EN_022"],
     "note": "English 'Very small drawing' conflates absolute figure smallness with the same fear/insecurity reading as the page-coverage rule; a genuinely distinct observable (figure size vs. page-area fraction) with an overlapping interpretation."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_EN_022", "SRC_EN_023"],
     "note": "Internal to the English PDF: 'Very small drawing' and 'Uses only a small part of the page' describe closely related but not identical observables (figure size vs. page-area fraction) with overlapping interpretations (insecurity/withdrawal/fear/low confidence vs. caution/uncertainty/low confidence)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_017", "SRC_EN_024"],
     "note": "Same observable and near-identical interpretation (top placement -> dreaminess/fantasy/difficulty adapting)."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_018", "SRC_EN_025"],
     "note": "Same observable (left placement -> introversion); English adds past orientation/dependence."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_AR_019", "SRC_EN_026"],
     "note": "Same observable (right placement -> extraversion); English adds future orientation/outward movement."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_EN_019", "SRC_EN_037"],
     "note": "Internal to the English PDF: 'Repeated monsters, danger, injury themes' (page 2, Animals and objects with stories) and 'Repeated monsters or danger scenes' (page 3, Repeated themes) describe the same observable with near-identical wording -- the PDF itself restates this row across two sections."},
    {"type": "near_duplicate", "source_entry_ids": ["SRC_EN_004", "SRC_EN_035"],
     "note": "Internal to the English PDF: 'Eyes with little detail or missing eyes' (page 1, Eyes and face) and 'Missing eyes' (page 3, Missing or altered details) describe overlapping observables (absence/low-detail of eyes) with overlapping interpretations (withdrawal/avoidance)."},
    {"type": "related_but_distinct", "source_entry_ids": ["SRC_AR_011", "SRC_EN_039"],
     "note": "Both concern loneliness, but the observable differs entirely: Arabic/EN_013 is a shape-based proxy (drawing circles); EN_039 is a content/scene-based theme (repeated isolation scenes) -- not the same measurable observable, so related, not duplicate."},
    {"type": "expands", "source_entry_ids": ["SRC_EN_005"], "note": "New observable (general facial expression / smiling vs. sad/tense/frightened) not present in the Arabic source at all -- expands the 'eyes and face' evidence family."},
    {"type": "expands", "source_entry_ids": ["SRC_EN_010"], "note": "New observable (animal choice in general, independent of species) not present in the Arabic source."},
    {"type": "expands", "source_entry_ids": ["SRC_EN_017", "SRC_EN_018"], "note": "New observables (house, tree) with no Arabic counterpart."},
    {"type": "expands", "source_entry_ids": ["SRC_EN_027"], "note": "New observable (centering) with no Arabic counterpart."},
    {"type": "expands", "source_entry_ids": ["SRC_EN_028", "SRC_EN_029", "SRC_EN_030", "SRC_EN_031", "SRC_EN_032"],
     "note": "An entirely new evidence family (line quality/pressure appearance) with no Arabic counterpart at all."},
    {"type": "expands", "source_entry_ids": ["SRC_EN_033", "SRC_EN_034", "SRC_EN_036"],
     "note": "New observables (missing hands, missing mouth, exaggerated body parts) with no Arabic counterpart."},
    {"type": "expands", "source_entry_ids": ["SRC_EN_038", "SRC_EN_040", "SRC_EN_041", "SRC_EN_042", "SRC_EN_043"],
     "note": "New observables (repeated family conflict, dark-colour/sad-face/isolation combination, refusal to draw figures, excessive detail, background neglect) with no Arabic counterpart -- these are the PDF's own 'mood or concern flags', all self-labeled 'Suggestive only' or weaker."},
]


def build_source_catalog() -> dict[str, Any]:
    entries = []
    for row in _AR_ENTRIES:
        entries.append({
            "source_entry_id": row["id"], "source_document": AR_PDF, "source_page": row["page"],
            "source_section": row["section"], "original_text": row["text"], "language": "ar",
            "observable": row["observable"], "possible_interpretation": row["interp"],
            "evidence_level_as_written": None,  # the Arabic notes have no explicit evidence-level column
            "notes_as_written": None,
            "source_type": "psychologist_supplied_notes",
            "transcription_status": "normalized_transcription_reused_from_rules_registry.json",
        })
    for row in _EN_ENTRIES:
        entries.append({
            "source_entry_id": row["id"], "source_document": EN_PDF, "source_page": row["page"],
            "source_section": row["section"], "original_text": row["text"], "language": "en",
            "observable": row["observable"], "possible_interpretation": row["interp"],
            "evidence_level_as_written": row.get("level"), "notes_as_written": row.get("notes"),
            "source_type": "general_guidance" if row.get("general") else "compiled_heuristics_guide",
            "transcription_status": "verbatim_pypdf_extraction",
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            AR_PDF: {"language": "ar", "pages": 2, "source_type": "psychologist_supplied_notes",
                     "scientifically_validated": False},
            EN_PDF: {"language": "en", "pages": 3, "source_type": "compiled_heuristics_guide",
                     "scientifically_validated": False,
                     "note": "Self-describes as 'not diagnostic rules; many are speculative or weakly supported' in its own introduction."},
        },
        "entry_count": len(entries),
        "entry_count_by_document": {AR_PDF: len(_AR_ENTRIES), EN_PDF: len(_EN_ENTRIES)},
        "candidate_rule_entry_count": sum(1 for e in entries if e["source_type"] != "general_guidance"),
        "entries": entries,
        "relationships": RELATIONSHIPS,
    }


def write_source_catalog() -> Path:
    document = build_source_catalog()
    CATALOG_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return CATALOG_PATH


if __name__ == "__main__":
    path = write_source_catalog()
    print(f"wrote {path}")
