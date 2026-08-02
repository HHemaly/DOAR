from __future__ import annotations

import json
from pathlib import Path

from doar.deep.selection import load_winner


def test_load_winner_uses_best_seed_of_validation_winner(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    checkpoint = runs / "mobile_seed_2" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    comparison = tmp_path / "deep_comparison.json"
    comparison.write_text(json.dumps({
        "winner": "mobile",
        "runs": [
            {"model": "mobile", "seed": 1, "best_valid_macro_f1": .6,
             "checkpoint": str(runs / "mobile_seed_1" / "best.pt")},
            {"model": "mobile", "seed": 2, "best_valid_macro_f1": .7,
             "checkpoint": str(checkpoint)},
            {"model": "resnet18", "seed": 42, "best_valid_macro_f1": .9,
             "checkpoint": "irrelevant.pt"},
        ],
    }), encoding="utf-8")
    winner = load_winner(comparison)
    assert winner["model"] == "mobile"
    assert winner["seed"] == 2
    assert winner["checkpoint"] == str(checkpoint)
