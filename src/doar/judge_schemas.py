"""Judge architecture (DOAR-TRACE 4G): a shared `JudgeVerdict` schema for
8 named judges (quality, feature, detection, relation, model, rule,
aggregation, language).

Per the task's explicit instruction, this module does **not** pretend
unimplemented judges are operational. Three of the eight
(`quality_judge`, `feature_judge`, `rule_judge`) already have real,
tested logic in `judges.py::run_judges` -- this module wraps that real
output into the new shared schema, it does not reimplement the checks.
`model_judge` is new but built directly from the same real
`emotion_ran`/`emotion_status` signal `judges.py` already computes.

The other four (`detection_judge`, `relation_judge`, `aggregation_judge`,
`language_judge`) have **no underlying capability to judge yet**
(no object detector, no spatial-relationship extractor, no automated
aggregation-quality check, no automated bilingual-wording-quality check)
-- they always report `status="not_implemented"`, with an honest reason,
never a fabricated pass or fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

JUDGE_IDS = (
    "quality_judge", "feature_judge", "detection_judge", "relation_judge",
    "model_judge", "rule_judge", "aggregation_judge", "language_judge",
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
                     "not per-feature correctness of all 59 objective features."],
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


def aggregation_judge_v2(structured_analysis: dict[str, Any] | None) -> JudgeVerdict:
    target = "structured_analysis" if structured_analysis else "unavailable"
    return _not_implemented(
        "aggregation_judge", target,
        reasons=["No automated check exists yet for the correctness of structured_report.py's theme aggregation "
                 "beyond its own unit tests (tests/test_structured_report.py) -- there is no independent judge "
                 "verifying a specific case's aggregation output at analysis time."],
        limitations=["The aggregator's dependency-aware no-double-counting and escalation-threshold logic is "
                     "unit-tested, but not judged per-case here."],
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
    }
    assert set(verdicts.keys()) == set(JUDGE_IDS)
    return verdicts
