"""Canonical, append-only, multi-source feature store (DOAR V2).

Multiple sources (a deterministic CV extractor, a VLM, a second VLM, a
human reviewer) may each contribute an observation for the same feature
on the same case. Nothing is ever overwritten or deleted here: every
`add_observation`/`human_confirm`/`human_correct` call appends a new,
timestamped, sequence-numbered entry. `effective()` computes a
best-current-answer *view* over that log; it never mutates it. A human
correction therefore never erases the AI observation it disagrees
with -- both remain in `history()` for audit.

This module intentionally does not implement confidence calibration,
reliability weighting, or majority voting beyond simple value equality --
the task is explicit that this is future work. The status vocabulary
below is the full extent of the aggregation logic for this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .observation import FeatureObservation


class ObservationStatus:
    """Status of a feature's accumulated evidence. Not an enum so a
    future status can be added without a migration."""

    CORROBORATED = "CORROBORATED"          # >=2 distinct sources agree
    SINGLE_SOURCE = "SINGLE_SOURCE"        # exactly one source, not an AI/CV extractor
    AI_ONLY = "AI_ONLY"                    # exactly one source, an AI/CV extractor, never human-reviewed
    CONFLICT = "CONFLICT"                  # >=2 distinct sources disagree
    NOT_ASSESSABLE = "NOT_ASSESSABLE"      # no assessable observation exists
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"    # a human confirmed the current value
    HUMAN_CORRECTED = "HUMAN_CORRECTED"    # a human overrode the current value

    ALL = (
        CORROBORATED, SINGLE_SOURCE, AI_ONLY, CONFLICT,
        NOT_ASSESSABLE, HUMAN_CONFIRMED, HUMAN_CORRECTED,
    )


@dataclass(frozen=True)
class _Entry:
    seq: int
    action: str  # "observe" | "confirm" | "correct"
    observation: FeatureObservation


@dataclass(frozen=True)
class EffectiveObservation:
    """The store's current best answer for one feature -- a computed
    view, never itself stored. This is what the rule engine reads."""

    feature_id: str
    status: str
    value: Any
    value_type: str | None
    assessable: bool
    reason: str | None
    contributing_sources: tuple[str, ...]


def _is_human_source(source: str) -> bool:
    return "human" in source.lower()


class FeatureStore:
    """Per-case canonical feature store. One instance per drawing/case."""

    def __init__(self) -> None:
        self._log: dict[str, list[_Entry]] = {}
        self._seq = 0

    def _append(self, feature_id: str, action: str, observation: FeatureObservation) -> None:
        if observation.feature_id != feature_id:
            raise ValueError(
                f"Observation feature_id {observation.feature_id!r} does not match "
                f"the feature_id it was added under ({feature_id!r})."
            )
        self._seq += 1
        self._log.setdefault(feature_id, []).append(_Entry(self._seq, action, observation))

    def add_observation(self, observation: FeatureObservation) -> None:
        """Append an ordinary (AI/CV/VLM) source observation."""
        self._append(observation.feature_id, "observe", observation)

    def human_confirm(self, observation: FeatureObservation) -> None:
        """Record that a human reviewer confirmed `observation.value` for
        this feature. Prior observations (agreeing or not) are kept."""
        self._append(observation.feature_id, "confirm", observation)

    def human_correct(self, observation: FeatureObservation) -> None:
        """Record a human override. The AI observation(s) it corrects
        remain in history() -- this call never deletes anything."""
        self._append(observation.feature_id, "correct", observation)

    def feature_ids(self) -> list[str]:
        return list(self._log.keys())

    def history(self, feature_id: str) -> list[FeatureObservation]:
        """Full, unmodified, chronological observation history for a
        feature -- every source's every claim, including ones the
        effective view no longer surfaces."""
        return [entry.observation for entry in self._log.get(feature_id, [])]

    def effective(self, feature_id: str) -> EffectiveObservation:
        entries = self._log.get(feature_id, [])
        if not entries:
            return EffectiveObservation(
                feature_id=feature_id, status=ObservationStatus.NOT_ASSESSABLE,
                value=None, value_type=None, assessable=False,
                reason="No observation has ever been recorded for this feature.",
                contributing_sources=(),
            )

        human_actions = [e for e in entries if e.action in ("confirm", "correct")]
        if human_actions:
            latest = max(human_actions, key=lambda e: e.seq)
            status = ObservationStatus.HUMAN_CORRECTED if latest.action == "correct" else ObservationStatus.HUMAN_CONFIRMED
            obs = latest.observation
            return EffectiveObservation(
                feature_id=feature_id, status=status, value=obs.value,
                value_type=obs.value_type, assessable=obs.assessable,
                reason=obs.not_assessable_reason,
                contributing_sources=(obs.source,),
            )

        assessable = [e.observation for e in entries if e.observation.assessable]
        if not assessable:
            latest = entries[-1].observation
            return EffectiveObservation(
                feature_id=feature_id, status=ObservationStatus.NOT_ASSESSABLE,
                value=None, value_type=latest.value_type, assessable=False,
                reason=latest.not_assessable_reason or "No source could assess this feature.",
                contributing_sources=tuple(dict.fromkeys(e.observation.source for e in entries)),
            )

        distinct_sources = list(dict.fromkeys(o.source for o in assessable))
        distinct_values = {repr(o.value) for o in assessable}

        if len(distinct_sources) >= 2:
            if len(distinct_values) > 1:
                return EffectiveObservation(
                    feature_id=feature_id, status=ObservationStatus.CONFLICT,
                    value=None, value_type=assessable[-1].value_type, assessable=False,
                    reason=f"{len(distinct_sources)} sources disagree: "
                           + ", ".join(f"{o.source}={o.value!r}" for o in assessable),
                    contributing_sources=tuple(distinct_sources),
                )
            agreed = assessable[-1]
            return EffectiveObservation(
                feature_id=feature_id, status=ObservationStatus.CORROBORATED,
                value=agreed.value, value_type=agreed.value_type, assessable=True,
                reason=None, contributing_sources=tuple(distinct_sources),
            )

        only = assessable[-1]
        status = ObservationStatus.AI_ONLY if not _is_human_source(only.source) else ObservationStatus.SINGLE_SOURCE
        return EffectiveObservation(
            feature_id=feature_id, status=status, value=only.value,
            value_type=only.value_type, assessable=True, reason=None,
            contributing_sources=(only.source,),
        )
