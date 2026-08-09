"""DOAR MVP: grounded Q&A extended with visual-object evidence, including
on-demand open-vocabulary search for objects not already in the case's
persisted detections.

Wraps -- never replaces -- `qa.py::answer()`: any question this module
does not recognize as visual falls straight through to the existing,
already-tested `qa.py` logic unchanged.

Design constraints (per the DOAR MVP architecture):
- Q&A first checks EXISTING case evidence (`visual_evidence.find_matching`);
  only if that is inadequate does it issue ONE live on-demand query against
  the SAME already-uploaded case image (`visual_evidence.search_visual`).
- An on-demand hit is never auto-converted into a validated psychological
  conclusion -- it is reported as "evidence supports / does not support /
  uncertain", never a diagnosis (`VisualFinding.validation_status` is
  surfaced verbatim, never upgraded here).
- If no `open_vocab_predict_fn` is injected (e.g. models unavailable in
  this environment), on-demand search degrades to an honest "unavailable"
  answer -- it never silently skips the search and answers as if absence
  were confirmed.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .qa import answer as qa_answer
from .visual_evidence import VisualFinding, find_matching, load_detections, search_visual

_NON_DIAG = {
    "en": "This is observational evidence only and is not a diagnosis.",
    "ar": "هذه أدلة رصدية فقط وليست تشخيصاً.",
}

# canonical label -> recognized question-side synonyms.
_VISUAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "person": ("person", "people", "man", "woman", "boy", "girl", "human", "figure",
               "شخص", "إنسان", "أشخاص"),
    "face": ("face", "faces", "وجه", "وجوه"),
    "hand": ("hand", "hands", "arm", "arms", "يد", "أيدي"),
    "eye": ("eye", "eyes", "عين", "عيون"),
    "mouth": ("mouth", "mouths", "فم"),
    "tree": ("tree", "trees", "شجرة", "أشجار"),
    "house": ("house", "houses", "home", "منزل", "بيت"),
    "heart": ("heart", "hearts", "قلب", "قلوب"),
    "animal": ("animal", "animals", "حيوان", "حيوانات"),
    "star": ("star", "stars", "نجمة", "نجوم"),
    "circle": ("circle", "circles", "دائرة"),
    "vehicle": ("vehicle", "vehicles", "car", "cars", "truck", "سيارة"),
    "sun": ("sun", "شمس"),
    "moon": ("moon", "قمر"),
    "cloud": ("cloud", "clouds", "سحابة"),
    "flower": ("flower", "flowers", "زهرة", "زهور"),
    "bird": ("bird", "birds", "طائر", "طيور"),
    "cat": ("cat", "cats", "قطة"),
    "dog": ("dog", "dogs", "كلب"),
    "door": ("door", "doors", "باب"),
    "window": ("window", "windows", "نافذة"),
    "road": ("road", "roads", "street", "طريق"),
    "weapon_or_weapon_like_object": ("weapon", "gun", "knife", "sword", "سلاح", "سكين"),
}

_QUESTION_NOUN_RE = re.compile(
    r"(?:is there|are there|do you see|does the drawing (?:have|show|contain)|"
    r"where is the|where are the|how many)\s+(?:an?|the)?\s*([a-z][a-z\s]{1,30}?)"
    r"(?:\?|$|\bin\b|\bthere\b)",
    re.IGNORECASE,
)
_HOW_MANY_RE = re.compile(r"\bhow many\b", re.IGNORECASE)


def extract_visual_target(question: str) -> str | None:
    """Best-effort target noun extraction. First checks the known-vocabulary
    synonym table (a canonical label always wins over the generic pattern);
    if nothing matches, falls back to a simple pattern-based noun extraction
    so the on-demand path can still attempt a live query for objects this
    module has never heard of (e.g. "is there a bicycle?"). A coarse
    heuristic, not a full NLP parser -- documented as a limitation."""
    q = question.casefold().strip()
    for canonical, synonyms in _VISUAL_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(s)}\b", q) for s in synonyms):
            return canonical
    m = _QUESTION_NOUN_RE.search(q)
    if m:
        noun = m.group(1).strip()
        if not _HOW_MANY_RE.search(q):
            noun = noun.rstrip("s") or noun
        if len(noun) > 1:
            return noun
    return None


@lru_cache(maxsize=1)
def _default_registry_v2() -> dict:
    from .registry_v2_build import build_registry_v2
    return build_registry_v2()


def _env(answer_en, answer_ar, *, language, evidence_ids, source_module,
         availability="available", limitations=None, non_diagnostic=True):
    ar = language == "ar"
    return {
        "answer": answer_ar if ar else answer_en,
        "evidence_ids": evidence_ids,
        "source_module": source_module,
        "source": source_module,
        "availability": availability,
        "limitations": limitations or [],
        "non_diagnostic_warning": (_NON_DIAG["ar"] if ar else _NON_DIAG["en"]) if non_diagnostic else None,
    }


def _describe_matches(matches: list[VisualFinding], target: str, *, arabic: bool) -> str:
    n = len(matches)
    validated = [m for m in matches if m.validation_status == "VALIDATED"]
    best = max(matches, key=lambda m: m.confidence)
    status_word = "validated" if validated else best.validation_status.lower()
    if arabic:
        return (f"تم العثور على '{target}' في الأدلة المحفوظة ({n} سجل، الحالة: {status_word}، "
                f"الثقة {best.confidence:.2f}).")
    return (f"Found '{target}' in the case's saved evidence ({n} record(s), "
            f"status: {status_word}, confidence {best.confidence:.2f}).")


def answer_with_visual_grounding(case_dir, question: str, analysis: dict, judges: dict | None = None,
                                  language: str = "en", *, registry_v2: dict | None = None,
                                  open_vocab_predict_fn=None) -> dict:
    """Wraps `qa.answer()`: if `question` names a recognizable (or
    plausible) visual target, checks the case's persisted detections first,
    then -- only if inadequate -- performs ONE live on-demand search on the
    same case image. Any question this function doesn't recognize as
    visual falls straight through to `qa.answer()` unchanged."""
    target = extract_visual_target(question)
    if target is None:
        return qa_answer(question, analysis, judges, language)

    existing = load_detections(case_dir)
    matches = find_matching(existing, target)
    if matches:
        return _env(
            _describe_matches(matches, target, arabic=False),
            _describe_matches(matches, target, arabic=True),
            language=language, evidence_ids=[f"detections:{target}"],
            source_module="visual_evidence", availability="available",
            limitations=["Detection status may be experimental/unmapped -- see Technical View for provenance."])

    if open_vocab_predict_fn is None:
        return _env(
            f"No saved evidence found for '{target}', and on-demand visual search is not "
            "available in this environment.",
            f"لا يوجد دليل محفوظ لـ '{target}'، والبحث البصري الفوري غير متاح في هذه البيئة.",
            language=language, evidence_ids=[], source_module="visual_evidence",
            availability="missing_detector",
            limitations=["Absence of a saved detection is not proof of absence in the drawing."])

    finding = search_visual(case_dir, target, registry_v2=registry_v2 or _default_registry_v2(),
                             open_vocab_predict_fn=open_vocab_predict_fn)
    if finding is not None:
        return _env(
            f"No prior evidence existed for '{target}'; an on-demand visual search found it "
            f"(confidence {finding.confidence:.2f}, status: {finding.validation_status.lower()}). "
            "This is experimental/unmapped evidence, not a validated psychological finding.",
            f"لم يكن هناك دليل سابق لـ '{target}'؛ عثر البحث البصري الفوري عليه "
            f"(الثقة {finding.confidence:.2f}، الحالة: {finding.validation_status.lower()}). "
            "هذا دليل تجريبي/غير مصنّف، وليس نتيجة نفسية موثقة.",
            language=language, evidence_ids=[f"on_demand:{target}"],
            source_module="visual_evidence_on_demand", availability="available")

    return _env(
        f"No saved evidence for '{target}', and an on-demand visual search found none either. "
        "This does not prove it is absent from the drawing.",
        f"لا يوجد دليل محفوظ لـ '{target}'، ولم يعثر البحث البصري الفوري على شيء أيضاً. "
        "هذا لا يثبت غيابه عن الرسمة.",
        language=language, evidence_ids=[], source_module="visual_evidence_on_demand",
        availability="not_found",
        limitations=["A negative on-demand search result is not proof of absence."])
