"""DOAR Visual Observer: minimal, reusable evaluation-result schema for
LATER multi-provider comparison (OpenAI vs Gemini vs other) -- NOT a
full evaluation framework or benchmark harness. This phase only proves
that one real observer connects and behaves safely on one real image
(see `scripts/doar_visual_observer_h38_check.py`'s own explicit
"NOT a benchmark" caveat); computing trustworthy precision/recall/F1
requires a properly ground-truthed evaluation set, which does not exist
yet. What DOES belong here now is the shape that future comparison will
fill in, so a later phase doesn't have to invent it under time pressure.

`ObserverEvaluationMetrics` holds one provider's counts against one
reviewed case (or a manually aggregated set of cases); a later phase
supplies real counts from real ground truth. Nothing in this codebase
currently constructs one from live data -- deliberately, until that
ground-truth process exists.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObserverEvaluationMetrics:
    """Precision/recall/F1 over labeled salient objects, plus the
    honesty-specific counts this project cares about beyond a bare P/R/F1:
    `correct_abstentions` (regions the observer honestly reported as
    low-confidence/unknown rather than guessing a specific wrong label)
    and `count_accuracy_*` (how often a reported instance count matched
    ground truth, when per-instance counting is even evaluated)."""
    provider: str
    model: str
    true_positives: int
    false_positives: int
    false_negatives: int  # missed salient objects
    correct_abstentions: int = 0
    count_accuracy_matches: int | None = None  # None until per-instance counting is evaluated
    count_accuracy_total: int | None = None

    @property
    def precision(self) -> float | None:
        denom = self.true_positives + self.false_positives
        return round(self.true_positives / denom, 4) if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.true_positives + self.false_negatives
        return round(self.true_positives / denom, 4) if denom else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r or (p + r) == 0:
            return None
        return round(2 * p * r / (p + r), 4)

    @property
    def count_accuracy(self) -> float | None:
        if not self.count_accuracy_total:
            return None
        return round(self.count_accuracy_matches / self.count_accuracy_total, 4)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider, "model": self.model,
            "true_positives": self.true_positives, "false_positives": self.false_positives,
            "false_negatives": self.false_negatives, "correct_abstentions": self.correct_abstentions,
            "count_accuracy_matches": self.count_accuracy_matches,
            "count_accuracy_total": self.count_accuracy_total,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "count_accuracy": self.count_accuracy,
        }
