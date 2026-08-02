"""Validation-selected deep-model winner resolution."""

from __future__ import annotations

import json
from pathlib import Path


def load_winner(comparison: str | Path) -> dict:
    path = Path(comparison)
    if not path.exists():
        raise FileNotFoundError(f"Deep comparison result not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    winner_model = payload.get("winner")
    candidates = [r for r in payload.get("runs", []) if r.get("model") == winner_model]
    candidates = [r for r in candidates if r.get("checkpoint")]
    if not winner_model or not candidates:
        raise ValueError(f"No validation-selected winner checkpoint in {path}")
    winner = max(candidates, key=lambda r: float(r.get("best_valid_macro_f1", -1)))
    checkpoint = Path(winner["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (path.parent / checkpoint).resolve()
        if not checkpoint.exists():
            checkpoint = Path(winner["checkpoint"]).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Winning checkpoint does not exist: {checkpoint}")
    return {
        "model": winner_model,
        "seed": int(winner["seed"]),
        "checkpoint": str(checkpoint),
        "best_valid_macro_f1": float(winner["best_valid_macro_f1"]),
        "selection_split": "valid",
        "comparison": str(path.resolve()),
    }


def resolve_checkpoint(checkpoint: str | None, comparison: str | None) -> str:
    if checkpoint:
        return checkpoint
    if not comparison:
        raise ValueError("Provide --checkpoint or --deep-comparison")
    return load_winner(comparison)["checkpoint"]
