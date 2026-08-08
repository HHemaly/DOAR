"""Phase 2C.5 annotation-app logic, split out from the thin Streamlit file
(phase2c5_part_annotation_app.py) so it is unit-testable without a
Streamlit runtime -- mirrors the project's existing precedent
(src/doar/phase2c1's Streamlit app has no dedicated test file; its own
store/schema modules carry the tested logic instead).

Nothing here ever marks a `model_proposed` instance as trusted on its
own -- accepting/editing a proposal is a user action
(`accept_proposal_instance`/`edit_proposal_instance`), always explicit,
never automatic.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .ontology import PART_TARGETS
from .schema import PartInstance


def load_proposals_csv(path: str | Path) -> dict[tuple[str, str, str], list[dict]]:
    """{(model, pilot_id, target): [{'instance_index', 'bbox_x', 'bbox_y',
    'bbox_w', 'bbox_h'}, ...]} from a scripts/phase2c5_run_feasibility_pilot.py
    -style raw-proposals CSV. Returns {} if the file does not exist yet --
    proposals are optional; manual annotation always works without them."""
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple[str, str, str], list[dict]] = {}
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["pilot_id"], row["target"])
            out.setdefault(key, []).append({
                "instance_index": int(row["instance_index"]),
                "bbox": (float(row["bbox_x"]), float(row["bbox_y"]),
                         float(row["bbox_w"]), float(row["bbox_h"])),
            })
    return out


def seed_instances_from_proposals(proposal_boxes: list[dict], *, model_name: str, checkpoint: str,
                                   prompt: str, threshold: float, timestamp: str) -> list[PartInstance]:
    """Turns raw proposal boxes into `model_proposed` PartInstances for
    initial display -- these are NOT saved until a human explicitly
    accepts/edits/rejects them via the functions below."""
    return [
        PartInstance(instance_index=i, bbox=box["bbox"], bbox_source="model_proposed",
                     proposal_model=model_name, proposal_checkpoint=checkpoint,
                     proposal_prompt=prompt, proposal_threshold=threshold, proposal_timestamp=timestamp)
        for i, box in enumerate(proposal_boxes)
    ]


def accept_proposal_instance(proposal: PartInstance) -> PartInstance:
    """A human confirmed this exact box is correct -- provenance
    (which model/checkpoint/prompt proposed it) is preserved, only the
    source label changes from 'model_proposed' to 'human_accepted'."""
    if proposal.bbox_source != "model_proposed":
        raise ValueError(f"accept_proposal_instance expects bbox_source='model_proposed', "
                          f"got {proposal.bbox_source!r}")
    d = proposal.to_dict()
    d["bbox_source"] = "human_accepted"
    return PartInstance.from_dict(d)


def edit_proposal_instance(proposal: PartInstance, new_bbox: tuple[float, float, float, float]) -> PartInstance:
    """A human adjusted a proposed box's coordinates -- provenance of the
    ORIGINAL proposal is preserved (so it's still auditable which model
    got the annotator started), but the source is 'human_edited', never
    silently 'human_accepted'."""
    if proposal.bbox_source not in ("model_proposed", "human_edited"):
        raise ValueError(f"edit_proposal_instance expects a model-derived instance, "
                          f"got bbox_source={proposal.bbox_source!r}")
    d = proposal.to_dict()
    d["bbox_source"] = "human_edited"
    d["bbox"] = new_bbox
    return PartInstance.from_dict(d)


def new_manual_instance(instance_index: int, bbox: tuple[float, float, float, float]) -> PartInstance:
    """A box the annotator drew from scratch -- no model was ever involved,
    so every proposal-provenance field stays empty (schema.py's own
    __post_init__ enforces this: bbox_source='human_drawn' with a non-empty
    proposal_model would be false provenance)."""
    return PartInstance(instance_index=instance_index, bbox=bbox, bbox_source="human_drawn")


def renumber_instances(instances: list[PartInstance]) -> tuple[PartInstance, ...]:
    """Reassigns instance_index 0..n-1 in the given order -- needed after a
    delete so schema.py's index-contiguity check keeps passing."""
    out = []
    for i, inst in enumerate(instances):
        d = inst.to_dict()
        d["instance_index"] = i
        out.append(PartInstance.from_dict(d))
    return tuple(out)


def delete_instance(instances: list[PartInstance], index_to_delete: int) -> tuple[PartInstance, ...]:
    remaining = [inst for inst in instances if inst.instance_index != index_to_delete]
    return renumber_instances(remaining)


def reviewed_proposal_summary(instances: list[PartInstance]) -> dict[str, int]:
    """Counts by bbox_source -- feeds the readiness metrics in Stage 8
    (acceptance/edit/manual/rejection-implied rates)."""
    out = {"model_proposed": 0, "human_accepted": 0, "human_edited": 0, "human_drawn": 0}
    for inst in instances:
        out[inst.bbox_source] = out.get(inst.bbox_source, 0) + 1
    return out


def validate_target_name(target_name: str) -> None:
    if target_name not in PART_TARGETS:
        raise ValueError(f"{target_name!r} not in {PART_TARGETS}")
