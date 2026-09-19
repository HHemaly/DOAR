"""Three-valued logic primitives.

DOAR's rule premises must never collapse "we don't know" into "false".
`TriState.UNKNOWN` is a first-class value, not an exception path, and the
AND/OR truth tables below implement standard strong Kleene logic:

    AND: FALSE dominates (FALSE AND anything = FALSE), else UNKNOWN
         dominates over TRUE.
    OR:  TRUE dominates (TRUE OR anything = TRUE), else UNKNOWN
         dominates over FALSE.

This mirrors the task's required examples exactly:
    TRUE AND TRUE = TRUE
    TRUE AND UNKNOWN = UNKNOWN
    FALSE AND UNKNOWN = FALSE
and the equivalent OR table.
"""

from __future__ import annotations

import enum


class TriState(enum.Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self) -> None:  # pragma: no cover - defensive
        # A TriState must never be silently coerced to a Python bool: that
        # coercion is exactly the "unknown becomes false" bug this type
        # exists to prevent. Force every call site to compare explicitly.
        raise TypeError(
            "TriState has no truth value -- compare explicitly against "
            "TriState.TRUE / TriState.FALSE / TriState.UNKNOWN."
        )


def tri_not(a: TriState) -> TriState:
    if a is TriState.TRUE:
        return TriState.FALSE
    if a is TriState.FALSE:
        return TriState.TRUE
    return TriState.UNKNOWN


def tri_and(*values: TriState) -> TriState:
    if not values:
        raise ValueError("tri_and() requires at least one value")
    if any(v is TriState.FALSE for v in values):
        return TriState.FALSE
    if any(v is TriState.UNKNOWN for v in values):
        return TriState.UNKNOWN
    return TriState.TRUE


def tri_or(*values: TriState) -> TriState:
    if not values:
        raise ValueError("tri_or() requires at least one value")
    if any(v is TriState.TRUE for v in values):
        return TriState.TRUE
    if any(v is TriState.UNKNOWN for v in values):
        return TriState.UNKNOWN
    return TriState.FALSE


def from_bool(value: bool | None) -> TriState:
    if value is None:
        return TriState.UNKNOWN
    return TriState.TRUE if value else TriState.FALSE
