"""Phase 2C.6 Stage 1: graphical bounding-box coordinate conversion --
pure functions, fully testable without a Streamlit/browser runtime.

`streamlit-drawable-canvas` renders a Fabric.js canvas at a fixed pixel
size (`display_width` x `display_height`) and returns box objects in that
SAME pixel space (`left`, `top`, `width`, `height`, `scaleX`, `scaleY`).
This module is the only place that translates between that canvas pixel
space and this project's normalized (x, y, w, h) in [0, 1] bbox
convention (schema.PartInstance.bbox) -- so resizing the display never
corrupts a saved box: every conversion re-derives normalized coordinates
from the CURRENT display size, never accumulates drift across resizes.

Reconciling what the canvas returns after an edit session against what
was seeded into it (`match_canvas_boxes_to_instances`) uses IoU matching,
not object identity -- Fabric.js's own JSON round-trip does not reliably
preserve arbitrary custom fields we might attach to a seeded object, so
this module never assumes it can. This is a deliberate, documented
simplification: a box that moved a lot in one edit could in principle be
mismatched against the wrong original box if two boxes started very close
together and one moved to where the other was -- an acceptable and rare
risk for a human-reviewed annotation tool where the annotator can always
re-check before saving, not something a fully automatic pipeline could
tolerate.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..phase2c5.app_helpers import renumber_instances
from ..phase2c5.schema import PartInstance

MIN_DISPLAY_DIM = 400
MAX_DISPLAY_DIM = 800

# Visual convention: proposals are shown dashed, everything the human has
# touched (accepted/edited/drawn) is shown solid -- purely a rendering
# choice, carries no provenance meaning of its own (bbox_source is the
# actual provenance record, this is just what color/style each gets in
# the canvas seed).
BBOX_SOURCE_STYLE = {
    "model_proposed": {"stroke": "#e67e22", "dash": True},
    "human_accepted": {"stroke": "#2ecc71", "dash": False},
    "human_edited": {"stroke": "#3498db", "dash": False},
    "human_drawn": {"stroke": "#9b59b6", "dash": False},
}


def compute_display_size(image_width: int, image_height: int,
                          max_dim: int = MAX_DISPLAY_DIM, min_dim: int = MIN_DISPLAY_DIM
                          ) -> tuple[int, int]:
    """Scales (image_width, image_height) to fit within `max_dim` on the
    longer side (never distorting aspect ratio, since the canvas is given
    a pre-resized copy of the image at exactly this size -- Fabric.js does
    NOT preserve aspect ratio on its own if given a stretched background).
    Upscales small images up to `min_dim` on the longer side so tiny
    drawings are still practical to box by hand."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size ({image_width}, {image_height})")
    longer = max(image_width, image_height)
    if longer > max_dim:
        scale = max_dim / longer
    elif longer < min_dim:
        scale = min_dim / longer
    else:
        scale = 1.0
    return max(1, round(image_width * scale)), max(1, round(image_height * scale))


def normalized_to_canvas_rect(bbox: tuple[float, float, float, float], display_width: int,
                               display_height: int, *, bbox_source: str) -> dict:
    """One normalized PartInstance.bbox -> one Fabric.js rect object dict,
    for seeding `st_canvas(initial_drawing=...)`."""
    x, y, w, h = bbox
    style = BBOX_SOURCE_STYLE[bbox_source]
    return {
        "type": "rect",
        "left": x * display_width, "top": y * display_height,
        "width": w * display_width, "height": h * display_height,
        "scaleX": 1.0, "scaleY": 1.0,
        "stroke": style["stroke"], "strokeWidth": 2, "fill": "rgba(0,0,0,0)",
        "strokeDashArray": [6, 4] if style["dash"] else None,
        "selectable": True,
    }


def build_initial_drawing(instances: list[PartInstance], display_width: int, display_height: int) -> dict:
    """{'version': ..., 'objects': [...]} for `st_canvas(initial_drawing=...)`
    -- one rect per instance, in the same order as `instances` (order is
    the only thing `match_canvas_boxes_to_instances` can rely on as a
    weak prior, on top of IoU)."""
    return {
        "version": "4.4.0",
        "objects": [normalized_to_canvas_rect(inst.bbox, display_width, display_height,
                                               bbox_source=inst.bbox_source)
                    for inst in instances],
    }


def canvas_object_to_normalized_bbox(obj: dict, display_width: int, display_height: int
                                      ) -> tuple[float, float, float, float]:
    """Inverse of `normalized_to_canvas_rect` -- reads Fabric.js's actual
    returned geometry (left/top/width/height/scaleX/scaleY, since a
    resize changes scaleX/scaleY, not width/height), clamps to [0, 1] and
    to a positive minimum size (a canvas box can be dragged to
    zero/negative width mid-edit)."""
    left, top = float(obj["left"]), float(obj["top"])
    width = float(obj["width"]) * float(obj.get("scaleX", 1.0))
    height = float(obj["height"]) * float(obj.get("scaleY", 1.0))
    x = min(1.0, max(0.0, left / display_width))
    y = min(1.0, max(0.0, top / display_height))
    w = min(1.0 - x, max(1e-4, width / display_width))
    h = min(1.0 - y, max(1e-4, height / display_height))
    return (x, y, w, h)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class ReconciliationResult:
    matched: tuple[tuple[PartInstance, tuple[float, float, float, float], bool], ...]  # (seed, new_bbox, moved)
    new_boxes: tuple[tuple[float, float, float, float], ...]
    deleted: tuple[PartInstance, ...]


def match_canvas_boxes_to_instances(canvas_boxes: list[tuple[float, float, float, float]],
                                     seeded_instances: list[PartInstance],
                                     iou_threshold: float = 0.3,
                                     moved_epsilon: float = 0.01) -> ReconciliationResult:
    """Greedy best-IoU matching between what the canvas currently shows
    and what was seeded into it. A seeded instance with no matching
    canvas box was deleted by the user (dropped, never saved -- this is
    exactly how a proposal gets rejected: the annotator deletes it from
    the canvas). A canvas box with no matching seed is newly drawn
    (`human_drawn`, unless the caller decides otherwise -- see
    `resolve_bbox_source` below)."""
    remaining_canvas = list(enumerate(canvas_boxes))
    matched = []
    for seed in seeded_instances:
        best_idx, best_iou = None, 0.0
        for i, (_, box) in enumerate(remaining_canvas):
            iou = _iou(seed.bbox, box)
            if iou > best_iou:
                best_idx, best_iou = i, iou
        if best_idx is not None and best_iou >= iou_threshold:
            _, box = remaining_canvas.pop(best_idx)
            moved = any(abs(a - b) > moved_epsilon for a, b in zip(seed.bbox, box))
            matched.append((seed, box, moved))
    matched_seed_ids = {id(m[0]) for m in matched}
    deleted = tuple(s for s in seeded_instances if id(s) not in matched_seed_ids)
    new_boxes = tuple(box for _, box in remaining_canvas)
    return ReconciliationResult(matched=tuple(matched), new_boxes=new_boxes, deleted=deleted)


def resolve_bbox_source(seed: PartInstance, moved: bool) -> str:
    """What a matched (seed, moved) pair's saved bbox_source should be --
    the one place this rule lives, so the app and tests share it exactly.
    A proposal that was moved becomes edited; a proposal left untouched
    stays a proposal (NOT auto-accepted -- explicit acceptance is a
    separate action, see app_helpers.accept_proposal_instance, called
    only for proposals the annotator explicitly marks accepted without
    moving them)."""
    if seed.bbox_source == "model_proposed":
        return "human_edited" if moved else "model_proposed"
    if seed.bbox_source in ("human_accepted", "human_edited", "human_drawn"):
        return "human_edited" if moved else seed.bbox_source
    raise ValueError(f"unknown bbox_source {seed.bbox_source!r}")


def reconcile_canvas_session(canvas_boxes: list[tuple[float, float, float, float]],
                              seeded_instances: list[PartInstance]) -> tuple[PartInstance, ...]:
    """The full WORKING instance list after one canvas edit -- includes
    untouched `model_proposed` boxes (still shown, still awaiting an
    explicit accept/reject), matched boxes with `resolve_bbox_source`
    applied, and any brand-new box as `human_drawn`. A seed with no
    matching canvas box (deleted on the canvas) is dropped. Renumbered to
    a contiguous 0..n-1 range (schema.PartAnnotationRecord requires it)."""
    result = match_canvas_boxes_to_instances(canvas_boxes, seeded_instances)
    out = []
    for seed, new_bbox, moved in result.matched:
        d = seed.to_dict()
        d["bbox"] = new_bbox
        d["bbox_source"] = resolve_bbox_source(seed, moved)
        out.append(PartInstance.from_dict(d))
    for box in result.new_boxes:
        out.append(PartInstance(instance_index=0, bbox=box, bbox_source="human_drawn"))
    return renumber_instances(out)


def savable_instances(working_instances: tuple[PartInstance, ...]) -> tuple[PartInstance, ...]:
    """Drops any instance still `bbox_source == 'model_proposed'` -- an
    unreviewed proposal must never be written to the store as trusted
    annotation, even if it is still visibly displayed in the UI awaiting
    an explicit accept/reject."""
    kept = [inst for inst in working_instances if inst.bbox_source != "model_proposed"]
    return renumber_instances(kept)
