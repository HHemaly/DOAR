"""Reference-mask annotation app (Streamlit). Run on your own computer:

  pip install -e ".[cv,ml,ui,phase2c6]" scikit-image
  streamlit run experiments/journal/annotation/reference_mask_app.py -- --batch <folder made by c2_reference_batch.py>

The batch folder contains neutral image names (R001.png ...) at working
resolution and NO class labels, method masks or filenames, so the annotator
is blind to everything that is being evaluated. Each save writes
  masks/<task>__<annotator>.png         (white = drawing mark)
  records/<task>__<annotator>.json      (assessability, support clicks,
                                         threshold, stroke counts, seconds spent)
Nothing is overwritten silently: re-saving a task keeps the previous record
under records/history/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c6 import canvas_compat  # noqa: E402
from doar_journal import annotation as ann  # noqa: E402

canvas_compat.apply()
from streamlit_drawable_canvas import st_canvas  # noqa: E402

DISPLAY_MAX = 900


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    return p.parse_args(sys.argv[1:])


def load_tasks(batch: Path) -> list[dict]:
    with open(batch / "batch_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    batch = Path(args().batch)
    tasks = load_tasks(batch)
    st.set_page_config(page_title="DOAR reference masks", layout="wide")
    annotator = st.sidebar.text_input("Annotator id (initials)", key="annotator").strip()
    if not annotator:
        st.info("Enter your annotator id in the sidebar to start.")
        return
    done = {p.name.split("__")[0] for p in (batch / "records").glob(f"*__{annotator}.json")}
    st.sidebar.write(f"Done: {len(done)} / {len(tasks)}")
    order = [t["task_id"] for t in tasks]
    default = next((i for i, t in enumerate(order) if t not in done), 0)
    task_id = st.sidebar.selectbox("Task", order, index=default)
    task = next(t for t in tasks if t["task_id"] == task_id)
    img = np.asarray(Image.open(batch / "images" / task["image_file"]).convert("RGB"))
    h, w = img.shape[:2]
    scale = min(1.0, DISPLAY_MAX / w)
    dw, dh = int(round(w * scale)), int(round(h * scale))
    if st.session_state.get("task_started") != (task_id, annotator):
        st.session_state["task_started"] = (task_id, annotator)
        st.session_state["t0"] = time.time()

    st.subheader(f"{task_id}  ({w}x{h}px)")
    assess = st.radio("1. Can drawing marks be separated from a support (paper) in this image?",
                      ann.ASSESSABILITY, horizontal=False, key=f"assess_{task_id}")
    note = st.text_input("Note (optional)", key=f"note_{task_id}")

    mask = np.zeros((h, w), bool)
    points, threshold, n_add, n_erase = [], None, 0, 0
    if assess == "ASSESSABLE":
        st.markdown("2. Click 2-5 **paper** points (blue), then set the threshold. "
                    "3. Brush **green** to add missed marks, **red** to remove non-marks.")
        mode = st.radio("Tool", ("paper points", "add marks (green)", "erase (red)"), horizontal=True,
                        key=f"mode_{task_id}")
        threshold = st.slider("Paper-distance threshold (higher = fewer pixels marked)", 2.0, 80.0, 18.0, 0.5,
                              key=f"thr_{task_id}")
        brush = st.slider("Brush size (display px)", 1, 30, 6, key=f"brush_{task_id}")
        pts_key = f"pts_{task_id}"
        points = st.session_state.get(pts_key, [])
        prop = ann.proposal(img, points, threshold) if points else np.zeros((h, w), bool)
        rgba_prev = st.session_state.get(f"rgba_{task_id}")
        add, erase = ann.strokes_from_canvas(rgba_prev, (h, w))
        mask = ann.combine(prop, add, erase)
        bg = Image.fromarray(ann.overlay(img, mask)).resize((dw, dh))
        stroke = {"paper points": "#1f4fff", "add marks (green)": "#00ff00", "erase (red)": "#ff0000"}[mode]
        res = st_canvas(background_image=bg, width=dw, height=dh, stroke_width=brush, stroke_color=stroke,
                        drawing_mode="point" if mode == "paper points" else "freedraw",
                        point_display_radius=4, key=f"canvas_{task_id}", update_streamlit=True)
        if res.json_data is not None:
            pts = [(int(o["left"] / scale), int(o["top"] / scale)) for o in res.json_data.get("objects", [])
                   if o.get("type") == "circle"]
            if pts != points:
                st.session_state[pts_key] = pts
                st.rerun()
        if res.image_data is not None:
            st.session_state[f"rgba_{task_id}"] = res.image_data
            add, erase = ann.strokes_from_canvas(res.image_data, (h, w))
            n_add, n_erase = int(add.sum()), int(erase.sum())
        st.caption(f"Marked: {mask.mean():.1%} of pixels; support samples: {len(points)}")

    if st.button("Save this task", type="primary"):
        (batch / "masks").mkdir(exist_ok=True)
        (batch / "records" / "history").mkdir(parents=True, exist_ok=True)
        stem = f"{task_id}__{annotator}"
        rec_path = batch / "records" / f"{stem}.json"
        if rec_path.exists():
            shutil.copy(rec_path, batch / "records" / "history" /
                        f"{stem}.{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
        if assess == "ASSESSABLE" and not points:
            st.error("Click at least one paper point first.")
            return
        Image.fromarray((mask * 255).astype(np.uint8)).save(batch / "masks" / f"{stem}.png")
        rec = {"task_id": task_id, "annotator": annotator, "assessability": assess, "note": note,
               "support_points_xy": points, "threshold": threshold, "add_pixels": n_add,
               "erase_pixels": n_erase, "mask_fraction": float(mask.mean()), "working_hw": [h, w],
               "seconds": round(time.time() - st.session_state.get("t0", time.time()), 1),
               "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        st.success(f"Saved {stem}")


main()
