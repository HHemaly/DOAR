"""
annotate.py — annotated-image generation and per-detection crop saving.

Draws ONLY genuinely-validated detections (validator_status == "verified") on a
copy of the original image, and saves each detection's crop separately. If a
detection has no bbox it is listed in a side panel instead of being drawn, so
nothing is invented on the canvas.
"""

from __future__ import annotations
import os

_STATUS_COLOR = {
    "verified":  (39, 174, 96),    # green
    "uncertain": (243, 156, 18),   # orange
    "rejected":  (231, 76, 60),    # red
    "unavailable": (149, 165, 166),
}


def save_crops(image_path: str, detections: list, out_dir: str) -> list:
    """
    Save a crop per detection that has a bbox. Returns list of crop paths,
    and sets each detection['crop_path'] in place.
    """
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    paths = []
    for i, det in enumerate(detections):
        bbox = det.get("bbox")
        if not bbox:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = img.crop((x1, y1, x2, y2))
            label = str(det.get("label", "det")).replace(" ", "_")
            path = os.path.join(out_dir, f"crop_{i:03d}_{label}.png")
            crop.save(path)
            det["crop_path"] = path
            paths.append(path)
        except Exception:
            continue
    return paths


def annotate_image(image_path: str, detections: list, out_path: str,
                   only_verified: bool = True) -> str | None:
    """
    Draw validated detection boxes on a copy of the image. Returns out_path.
    Only detections with a bbox AND (verified, unless only_verified=False) drawn.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    drawn = 0
    for det in detections:
        bbox = det.get("bbox")
        status = det.get("validator_status", "unavailable")
        if not bbox:
            continue
        if only_verified and status != "verified":
            continue
        color = _STATUS_COLOR.get(status, (149, 165, 166))
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = det.get("label", "")
            sim = det.get("clip_similarity")
            tag = f"{label}" + (f" ({sim:.2f})" if sim is not None else "")
            draw.text((x1 + 2, max(0, y1 - 12)), tag, fill=color, font=font)
            drawn += 1
        except Exception:
            continue

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    return out_path
