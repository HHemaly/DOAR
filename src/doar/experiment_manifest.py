"""Experiment artifact manifest contract (additive, define-only).

A small, JSON-compatible schema future Colab experiments can export so
their winning artifact can be plugged into the running app through
`research_runtime.py` WITHOUT any code change to CaseInterpretation or
its providers -- only the manifest (and the artifact it points to)
changes.

This module does NOT change any current Colab experiment and does NOT
select a winner. It only defines the loader contract, plus (see
`example_mobilenet_v3_small_development_manifest()`) one real example
built from the current MobileNet E1 development artifact already wired
into the app (research_runtime.py's emotion_provider entry).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_VALID_STATUSES = ("DEVELOPMENT", "CANDIDATE", "FINAL")


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str                  # e.g. "E1_visual_representation"
    task: str                           # e.g. "expressive_emotion_classification"
    model_name: str                     # e.g. "mobilenet_v3_small"
    artifact_version: str               # e.g. "best.pt@epoch_N" or a content hash
    checkpoint_path: str
    class_mapping: tuple[str, ...]      # e.g. ("Angry", "Fear", "Happy", "Sad")
    preprocessing: dict                 # the resolved preprocessing spec (see deep/preprocessing.py's shape)
    validation_metrics: dict            # e.g. {"val_macro_f1": 0.748}
    split_manifest_sha256: str | None   # sha256 of the train/valid manifest CSV this artifact was trained against
    training_seed: int | None
    status: str                         # DEVELOPMENT / CANDIDATE / FINAL
    created_at: str | None = None       # ISO-8601 timestamp string, if known
    notes: str = ""

    def __post_init__(self):
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unknown manifest status {self.status!r}, expected one of {_VALID_STATUSES}")

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id, "task": self.task, "model_name": self.model_name,
            "artifact_version": self.artifact_version, "checkpoint_path": self.checkpoint_path,
            "class_mapping": list(self.class_mapping), "preprocessing": self.preprocessing,
            "validation_metrics": self.validation_metrics, "split_manifest_sha256": self.split_manifest_sha256,
            "training_seed": self.training_seed, "status": self.status, "created_at": self.created_at,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExperimentManifest":
        return ExperimentManifest(
            experiment_id=d["experiment_id"], task=d["task"], model_name=d["model_name"],
            artifact_version=d["artifact_version"], checkpoint_path=d["checkpoint_path"],
            class_mapping=tuple(d["class_mapping"]), preprocessing=d["preprocessing"],
            validation_metrics=d["validation_metrics"], split_manifest_sha256=d.get("split_manifest_sha256"),
            training_seed=d.get("training_seed"), status=d["status"], created_at=d.get("created_at"),
            notes=d.get("notes", ""))


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    """Loader contract for a future Colab-exported manifest JSON file.
    Never invents a missing field -- a manifest missing a required key
    raises KeyError rather than silently defaulting."""
    return ExperimentManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def example_mobilenet_v3_small_development_manifest() -> ExperimentManifest:
    """ONE real, non-fabricated example manifest for the artifact
    ALREADY wired into the app today (see deep/e1_dev_checkpoint.py) --
    every field below is copied from that module's own documented,
    verified provenance, not invented for this example."""
    return ExperimentManifest(
        experiment_id="E1_visual_representation",
        task="expressive_emotion_classification_4class",
        model_name="mobilenet_v3_small",
        artifact_version="best.pt (best validation-epoch checkpoint)",
        checkpoint_path=(r"C:\Users\ZZ01G7865\Downloads\DOAR\DOAR-work\experiments\E1_visual_representation"
                         r"\checkpoints\mobilenet_v3_small\best.pt"),
        class_mapping=("Angry", "Fear", "Happy", "Sad"),
        preprocessing={
            "family": "torchvision", "model_name": "mobilenet_v3_small", "weights_id": "IMAGENET1K_V1",
            "revision": "torchvision_weights_transform", "preprocessing_version": "torchvision_weights_derived",
            "resize": 256, "crop": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "interpolation": "bilinear", "transform_source": "torchvision_weights.transforms()",
        },
        validation_metrics={"val_macro_f1": 0.7480453959583941},
        split_manifest_sha256=None,  # not computed for this development artifact -- honestly recorded, not guessed
        training_seed=42,
        status="DEVELOPMENT",
        created_at=None,  # not recorded in the E1 result.json for this candidate
        notes=("Highest validation macro-F1 among E1 candidates with a completed (non-smoke-only) run, "
              "as of this session. NOT a declared E1 experiment winner and NOT the final thesis model -- "
              "prototype/demo use only. See doar.deep.e1_dev_checkpoint for the read-only inference adapter "
              "and doar.research_runtime for how this is registered at runtime."),
    )
