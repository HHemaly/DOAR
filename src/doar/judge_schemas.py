"""Judge architecture (DOAR-TRACE 4G, `aggregation_judge` made operational
in Phase 1.5 Section 8; `page_frame_judge` added in Phase 2A Section 8):
a shared `JudgeVerdict` schema for 9 named judges (quality, feature,
detection, relation, model, rule, aggregation, language, page_frame).

Per the task's explicit instruction, this module does **not** pretend
unimplemented judges are operational. Five of the nine (`quality_judge`,
`feature_judge`, `rule_judge`, `aggregation_judge`, `page_frame_judge`)
have real, tested logic: the first three wrap `judges.py::run_judges`'s
already-tested checks; `aggregation_judge` (Phase 1.5) independently
re-verifies every `structured_report.py` combined hypothesis against the
construct policy in `construct_registry.json`; `page_frame_judge`
(Phase 2A) independently re-verifies that every page-relative rule
(`rule_engine_v2.ALL_PAGE_GATED_RULE_IDS`) was actually gated to
`not_assessable` when the page isn't visible -- not just a rerun of
`analysis.py`'s own gating call, but an independent check against the
saved `page_frame`/`rule_evaluations` output. `model_judge` is built
directly from the same real `emotion_ran`/`emotion_status` signal
`judges.py` already computes.

The other three (`detection_judge`, `relation_judge`, `language_judge`)
have **no underlying capability to judge yet** (no object detector, no
spatial-relationship extractor, no automated bilingual-wording-quality
judge -- `language_judge` stays `not_implemented` because no LLM exists
in this phase) -- they always report `status="not_implemented"`, with an
honest reason, never a fabricated pass or fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

JUDGE_IDS = (
    "quality_judge", "feature_judge", "detection_judge", "relation_judge",
    "model_judge", "rule_judge", "aggregation_judge", "language_judge",
    "page_frame_judge",
)

STATUSES = frozenset({"pass", "fail", "requires_review", "not_implemented"})


@dataclass(frozen=True)
class JudgeVerdict:
    judge_id: str
    target: str
    status: str
    confidence: float | None
    reasons: list[str]
    limitations: list[str]
    version: str

    def __post_init__(self) -> None:
        if self.judge_id not in JUDGE_IDS:
            raise ValueError(f"judge_id {self.judge_id!r} not in {JUDGE_IDS}")
        if self.status not in STATUSES:
            raise ValueError(f"JudgeVerdict {self.judge_id!r}: status {self.status!r} not in {sorted(STATUSES)}")
        if self.status == "not_implemented" and not self.reasons:
            raise ValueError(
                f"JudgeVerdict {self.judge_id!r} is not_implemented but gives no reason -- "
                "an honest 'not implemented' verdict must still say why."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _not_implemented(judge_id: str, target: str, reasons: list[str], limitations: list[str]) -> JudgeVerdict:
    return JudgeVerdict(
        judge_id=judge_id, target=target, status="not_implemented", confidence=None,
        reasons=reasons, limitations=limitations, version="not_implemented_v1",
    )


# ---------------------------------------------------------------------------
# Real judges: wrap judges.py::run_judges's already-tested logic.
# ---------------------------------------------------------------------------

def quality_judge_v2(analysis: dict[str, Any], judges_output: dict[str, Any]) -> JudgeVerdict:
    real = judges_output["quality_judge"]
    reasons = list(real.get("reasons", [])) or [f"quality_status={real.get('quality_status')}"]
    return JudgeVerdict(
        judge_id="quality_judge", target=analysis.get("image_path", "unknown"),
        status=real["status"], confidence=None, reasons=reasons,
        limitations=["Quality-gate thresholds are engineering defaults, not clinically validated on this dataset."],
        version="judges.py_run_judges_v1",
    )


def feature_judge_v2(analysis: dict[str, Any], judges_output: dict[str, Any]) -> JudgeVerdict:
    real = judges_output["feature_judge"]
    checks = real.get("checks", {})
    reasons = [f"{name}={value}" for name, value in checks.items()]
    return JudgeVerdict(
        judge_id="feature_judge", target=analysis.get("image_path", "unknown"),
        status=real["status"], confidence=None, reasons=reasons,
        limitations=["Judges only 2 structural sanity checks (blank-has-no-centroid, bbox-evidence-present), "
                     "not per-feature correctness of all 60 objective features."],
        version="judges.py_run_judges_v1",
    )


def rule_judge_v2(analysis: dict[str, Any], judges_output: dict[str, Any]) -> JudgeVerdict:
    real = judges_output["rule_judge"]
    reasons = []
    if real.get("unsupported_evidence_references"):
        reasons.append(f"unsupported_evidence_references={real['unsupported_evidence_references']}")
    if real.get("active_concerns_from_single_symbol"):
        reasons.append("a concern was derived from a single symbol/insufficient source diversity")
    if not reasons:
        reasons.append("No rule referenced an unknown evidence_id; no single-symbol concern was active.")
    return JudgeVerdict(
        judge_id="rule_judge", target=analysis.get("image_path", "unknown"),
        status=real["status"], confidence=None, reasons=reasons, limitations=[],
        version="judges.py_run_judges_v1",
    )


_EMOTION_JUDGE_STATUS_MAP = {"pass": "pass", "suppressed": "requires_review", "unavailable": "requires_review"}


def model_judge_v2(analysis: dict[str, Any], judges_output: dict[str, Any]) -> JudgeVerdict:
    """judges.py's real emotion_judge uses a 3-value vocabulary
    (pass/suppressed/unavailable) distinct from JudgeVerdict's 4-value one
    -- mapped explicitly here (suppressed/unavailable both mean "nothing to
    judge yet", i.e. requires_review, never a fabricated pass or fail)."""
    real = judges_output["emotion_judge"]
    emotion = analysis.get("emotion", {})
    reasons = [f"emotion_model_ran={real.get('emotion_model_ran')}", f"emotion_status={real.get('emotion_status')}"]
    if emotion.get("status") == "available":
        reasons.append(f"calibration_status={emotion.get('calibration_status')}")
    return JudgeVerdict(
        judge_id="model_judge", target=analysis.get("image_path", "unknown"),
        status=_EMOTION_JUDGE_STATUS_MAP[real["status"]], confidence=emotion.get("confidence"), reasons=reasons,
        limitations=["Judges only whether the model ran and its calibration status, not predictive accuracy "
                     "(that requires held-out labeled evaluation, done separately -- see THESIS_EXPERIMENT_RESULTS.md)."],
        version="judges.py_run_judges_v1",
    )


# ---------------------------------------------------------------------------
# Interface-only judges: no underlying capability exists yet.
# ---------------------------------------------------------------------------

def detection_judge_v2(analysis: dict[str, Any]) -> JudgeVerdict:
    return _not_implemented(
        "detection_judge", analysis.get("image_path", "unknown"),
        reasons=["No object detector exists in this release (src/doar/detectors/ is schema-only scaffolding, "
                 "imported by no user-facing path). There is no detection output to judge."],
        limitations=["Cannot assess detection precision/recall/localization without a detector to evaluate."],
    )


def relation_judge_v2(analysis: dict[str, Any]) -> JudgeVerdict:
    return _not_implemented(
        "relation_judge", analysis.get("image_path", "unknown"),
        reasons=["No spatial-relationship extractor exists (it requires object detection first, which does not "
                 "exist -- see detection_judge)."],
        limitations=["Cannot assess relation correctness without any extracted relations."],
    )


def aggregation_judge_v2(
    structured_analysis: dict[str, Any] | None, construct_registry: dict[str, Any] | None = None,
) -> JudgeVerdict:
    """DOAR-TRACE Phase 1.5, Section 8: operational, not a stub. Re-verifies
    every combined_drawing_level_hypotheses entry against
    structured_report.py's own construct policy -- an independent
    per-case check, not just a rerun of structured_report.py's own unit
    tests. Checks: minimum evidence-family count, minimum evidence-ID
    count, no repeated evidence IDs (unique dependency groups), a valid
    ordinal output level, that no hypothesis is founded on a single rule
    alone, and that every real cross-theme contradiction is recorded."""
    if not structured_analysis:
        return _not_implemented(
            "aggregation_judge", "unavailable",
            reasons=["No structured_analysis.json was supplied to judge."],
            limitations=["Cannot judge aggregation without the structured analysis document."],
        )
    from .construct_registry_build import build_construct_registry
    from .structured_report import OPPOSING_CONSTRUCTS
    construct_registry = construct_registry or build_construct_registry()
    constructs_by_id = {c["construct_id"]: c for c in construct_registry["constructs"]}

    reasons: list[str] = []
    hypotheses = structured_analysis.get("combined_drawing_level_hypotheses", [])
    for hyp in hypotheses:
        construct = constructs_by_id.get(hyp["target_construct"])
        if construct is None:
            reasons.append(f"{hyp['target_construct']}: not a known construct_id")
            continue
        n_families = len(set(hyp["contributing_evidence_families"]))
        n_evidence_ids = len(set(hyp["contributing_evidence_ids"]))
        if n_families < construct["minimum_independent_evidence_families"]:
            reasons.append(f"{hyp['target_construct']}: only {n_families} evidence families, needs >={construct['minimum_independent_evidence_families']}")
        if n_evidence_ids < 2:
            reasons.append(f"{hyp['target_construct']}: only {n_evidence_ids} distinct evidence IDs, needs >=2")
        if len(hyp["contributing_evidence_ids"]) != n_evidence_ids:
            reasons.append(f"{hyp['target_construct']}: repeated evidence_id detected (dependency double-counting)")
        if hyp["ordinal_level"] not in (2, 3, 4):
            reasons.append(f"{hyp['target_construct']}: invalid ordinal_level {hyp['ordinal_level']} for a combined hypothesis")
        sole_contributor_count = len(set(hyp["contributing_rule_ids"])) + (1 if hyp["uses_expressive_model"] else 0)
        if sole_contributor_count < 2:
            reasons.append(f"{hyp['target_construct']}: promoted to combined hypothesis from a single contributor")

    by_construct = {h["target_construct"] for h in hypotheses}
    recorded_pairs = {
        tuple(sorted((c["construct_a"], c["construct_b"])))
        for c in structured_analysis.get("cross_theme_contradictions", [])
    }
    for construct_id in by_construct:
        opposite = OPPOSING_CONSTRUCTS.get(construct_id)
        if opposite and opposite in by_construct:
            pair = tuple(sorted((construct_id, opposite)))
            if pair not in recorded_pairs:
                reasons.append(f"contradiction between {pair[0]} and {pair[1]} present but not recorded")

    status = "pass" if not reasons else "fail"
    return JudgeVerdict(
        judge_id="aggregation_judge", target=f"{len(hypotheses)} combined hypothesis(es)",
        status=status, confidence=None,
        reasons=reasons or ["All combined hypotheses satisfy the construct policy; all real contradictions are recorded."],
        limitations=["Independently re-checks structured_report.py's own policy; does not re-derive whether the "
                     "underlying rule triggers themselves are scientifically correct."],
        version="aggregation_judge_v1",
    )


def page_frame_judge_v2(analysis: dict[str, Any]) -> JudgeVerdict:
    """DOAR-TRACE Phase 2A, Section 8: operational, not a stub.
    Independently re-verifies (from the saved `page_frame` and
    `rule_evaluations` output alone -- never by re-running
    `analysis.py`'s own gating call) that every page-relative rule in
    `rule_engine_v2.ALL_PAGE_GATED_RULE_IDS` is `not_assessable` whenever
    the page is not visible, and specifically that it is never
    `not_matched` in that case (Section 3's explicit "must not become
    not_matched" requirement)."""
    from .page_frame import ASSESSABLE_STATUSES
    from .rule_engine_v2 import ALL_PAGE_GATED_RULE_IDS

    page_frame = analysis.get("page_frame") or {}
    status = page_frame.get("page_frame_status")
    target = f"page_frame_status={status}"
    if status is None:
        return _not_implemented(
            "page_frame_judge", "unavailable",
            reasons=["No page_frame assessment was supplied to judge (analysis predates Phase 2A or page_frame "
                     "is missing from the saved output)."],
            limitations=["Cannot judge page-relative rule gating without a page_frame assessment."],
        )

    reasons: list[str] = []
    if status not in ASSESSABLE_STATUSES:
        for rule_eval in analysis.get("rule_evaluations", []):
            if rule_eval["rule_id"] not in ALL_PAGE_GATED_RULE_IDS:
                continue
            if rule_eval["status"] in ("weak_support", "not_matched"):
                reasons.append(
                    f"{rule_eval['rule_id']}: status={rule_eval['status']!r} while page_frame_status={status!r} "
                    f"(not in {sorted(ASSESSABLE_STATUSES)}) -- should be not_assessable"
                )

    verdict_status = "fail" if reasons else "pass"
    return JudgeVerdict(
        judge_id="page_frame_judge", target=target, status=verdict_status, confidence=page_frame.get("confidence"),
        reasons=reasons or [f"All page-relative rules correctly gated for page_frame_status={status!r}."],
        limitations=["Checks only the gating invariant (not_assessable, never not_matched, when the page isn't "
                     "visible); does not independently re-derive the page-frame heuristic's own correctness -- "
                     "see docs/PAGE_FRAME_ASSESSABILITY.md for that."],
        version="page_frame_judge_v1",
    )


def language_judge_v2(text: str | None) -> JudgeVerdict:
    return _not_implemented(
        "language_judge", text[:80] + "..." if text and len(text) > 80 else (text or "unavailable"),
        reasons=["No automated bilingual wording-quality/fluency/register judge exists yet. "
                 "Diagnostic-language safety is checked separately and IS real "
                 "(see claim_verifier.py::verify_no_diagnostic_claims, judges.py's safety_judge) -- "
                 "this judge would additionally cover tone/clarity/register, which nothing checks today."],
        limitations=["Safety (forbidden-wording) is covered elsewhere; general language quality is not."],
    )


def run_all_judges_v2(
    analysis: dict[str, Any], judges_output: dict[str, Any],
    structured_analysis: dict[str, Any] | None = None,
) -> dict[str, JudgeVerdict]:
    verdicts = {
        "quality_judge": quality_judge_v2(analysis, judges_output),
        "feature_judge": feature_judge_v2(analysis, judges_output),
        "rule_judge": rule_judge_v2(analysis, judges_output),
        "model_judge": model_judge_v2(analysis, judges_output),
        "detection_judge": detection_judge_v2(analysis),
        "relation_judge": relation_judge_v2(analysis),
        "aggregation_judge": aggregation_judge_v2(structured_analysis),
        "language_judge": language_judge_v2(None),
        "page_frame_judge": page_frame_judge_v2(analysis),
    }
    assert set(verdicts.keys()) == set(JUDGE_IDS)
    return verdicts
