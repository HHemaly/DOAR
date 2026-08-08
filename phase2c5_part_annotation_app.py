"""Phase 2C.5 rule-critical part annotation and review interface. Launch with:

    .\\.venv\\Scripts\\Activate.ps1
    python -m streamlit run phase2c5_part_annotation_app.py

Shows one target (eye/mouth/hand/face/person) at a time for one blinded
drawing, exactly like phase2c_annotation_app.py: no emotion label, source
folder, or psychological interpretation is ever displayed. Model-proposed
boxes (Grounding DINO / OWLv2, precomputed by
scripts/phase2c5_run_feasibility_pilot.py or a future bulk-proposal run --
never invoked live from inside this app) are shown as suggestions only; a
proposal only becomes a trusted annotation once you Accept, Edit, or add
your own box, and Save records exactly which of those happened
(schema.PartInstance.bbox_source).

Phase 2C.6 upgrade: box editing is now a graphical click/drag/resize
canvas (`streamlit-drawable-canvas`, MIT, see pyproject.toml's `phase2c6`
extra for why this package and how coordinates are normalized --
src/doar/phase2c6/canvas_helpers.py is the single place that converts
between canvas pixel space and this project's normalized bbox
convention). NOT interactively verified in a live browser this session
(no browser access in this environment) -- see
PHASE2C6_SCALABLE_ANNOTATION_REPORT.md's UI-verification section for what
WAS verified (coordinate math, reconciliation logic, all unit-tested).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import workspace as workspace_mod  # noqa: E402
from doar.phase2c1 import store as store2c1_mod  # noqa: E402
from doar.phase2c5 import app_helpers as helpers  # noqa: E402
from doar.phase2c5 import store as store_mod  # noqa: E402
from doar.phase2c5.ontology import (  # noqa: E402
    ATTRIBUTE_ALLOWED_VALUES, PART_TARGETS,
    TARGET_ALLOWED_ATTRIBUTE_KEYS, TARGETS_WITH_EXISTING_PRESENCE,
)
from doar.phase2c5.schema import PartAnnotationRecord, utc_now_iso  # noqa: E402
from doar.phase2c6 import canvas_helpers as ch_mod  # noqa: E402

IMAGES_DIR = Path(os.environ.get(
    "DOAR_PHASE2C1_IMAGES_DIR", str(ROOT / "outputs/phase2c1/private_images")))
STORE2C1_PATH = Path(os.environ.get(
    "DOAR_PHASE2C1_STORE_PATH", str(ROOT / "outputs/phase2c1/annotation_store.csv")))
STORE_PATH = Path(os.environ.get(
    "DOAR_PHASE2C5_STORE_PATH", str(ROOT / "outputs/phase2c5/part_annotation_store.csv")))
PROPOSALS_PATH = Path(os.environ.get(
    "DOAR_PHASE2C5_PROPOSALS_PATH",
    str(ROOT / "outputs/phase2c5/feasibility_pilot_raw_proposals_private.csv")))
EXPORT_DIR = Path(os.environ.get(
    "DOAR_PHASE2C5_EXPORT_DIR", str(ROOT / "outputs/phase2c5/exports")))
PROPOSAL_MODEL_PREFERENCE = ("grounding_dino", "owlv2")  # primary first, per Stage 4

STATUS_LABELS = {"present": "Present", "absent": "Absent",
                  "uncertain": "Uncertain", "not_assessable": "Not assessable"}
STATUS_ORDER = ["absent", "present", "uncertain", "not_assessable"]

st.set_page_config(page_title="Phase 2C.5 part annotation", layout="wide")
st.title("Phase 2C.5 -- rule-critical part annotation")
st.caption(
    "Research/annotation tool only. No emotion label, source folder, or "
    "psychological interpretation is ever displayed here. Model-proposed "
    "boxes are suggestions -- they are not trusted evidence until you "
    "accept, edit, or replace them."
)

if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
    st.error(f"No blinded images found at {IMAGES_DIR}.")
    st.stop()

pilot_ids = workspace_mod.list_workspace_images(IMAGES_DIR)
store = store_mod.load_store(STORE_PATH)
store2c1 = store2c1_mod.load_store(STORE2C1_PATH)
proposals_by_key = helpers.load_proposals_csv(PROPOSALS_PATH)

with st.sidebar:
    st.header("Annotator")
    annotator_id = st.text_input("Your annotator ID", key="annotator_id")
    st.divider()
    st.header("Target")
    if "target_idx" not in st.session_state:
        st.session_state["target_idx"] = 0
    target_nav = st.columns([1, 1])
    if target_nav[0].button("< Prev target", disabled=st.session_state["target_idx"] <= 0):
        st.session_state["target_idx"] -= 1
        st.rerun()
    if target_nav[1].button("Next target >",
                             disabled=st.session_state["target_idx"] >= len(PART_TARGETS) - 1):
        st.session_state["target_idx"] += 1
        st.rerun()
    target_name = PART_TARGETS[st.session_state["target_idx"]]
    st.write(f"**{target_name}** ({st.session_state['target_idx'] + 1}/{len(PART_TARGETS)})")
    st.divider()
    st.header("Progress")
    if annotator_id:
        prog = store_mod.progress_for_annotator(store, pilot_ids, annotator_id, PART_TARGETS)
        st.progress((prog["done"] / prog["total"]) if prog["total"] else 0.0)
        st.write(f"**{prog['done']} / {prog['total']}** images complete (all {len(PART_TARGETS)} targets)")
    st.divider()
    st.header("Export")
    if st.button("Export CSV", type="primary"):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        store_mod.save_store(EXPORT_DIR / "phase2c5_part_annotations.csv", store)
        st.success(f"Exported to {EXPORT_DIR}")

if not annotator_id:
    st.warning("Enter your annotator ID in the sidebar to begin.")
    st.stop()


def _n_targets_done(pilot_id: str) -> int:
    return len({r.target_name for r in store_mod.records_for_pilot(store, pilot_id)
                if r.annotator_id == annotator_id})


idx_key = "idx_part"
if idx_key not in st.session_state:
    st.session_state[idx_key] = next(
        (i for i, pid in enumerate(pilot_ids) if _n_targets_done(pid) < len(PART_TARGETS)), 0)

nav_cols = st.columns([1, 1, 2, 1])
if nav_cols[0].button("Previous", disabled=st.session_state[idx_key] <= 0):
    st.session_state[idx_key] -= 1
    st.rerun()
if nav_cols[1].button("Next", disabled=st.session_state[idx_key] >= len(pilot_ids) - 1):
    st.session_state[idx_key] += 1
    st.rerun()
jump = nav_cols[2].number_input("Jump to image #", min_value=1, max_value=len(pilot_ids),
                                 value=st.session_state[idx_key] + 1,
                                 key=f"jump_{st.session_state[idx_key]}")
if jump - 1 != st.session_state[idx_key]:
    st.session_state[idx_key] = jump - 1
    st.rerun()

pilot_id = pilot_ids[st.session_state[idx_key]]
image_path = workspace_mod.image_path_for_pilot_id(IMAGES_DIR, pilot_id)
st.write(f"Image **{st.session_state[idx_key] + 1} / {len(pilot_ids)}** -- target: **{target_name}**")
if image_path and image_path.exists():
    st.image(str(image_path), width=450)

if target_name in TARGETS_WITH_EXISTING_PRESENCE:
    prior = {r.class_name: r for r in store2c1_mod.records_for_pilot(store2c1, pilot_id)}.get(target_name)
    if prior is not None:
        st.info(f"Phase 2C.1 presence judgment for {target_name!r}: **{prior.status}** "
                "(shown for reference -- re-confirm below; this does not silently overwrite it).")

existing = {r.target_name: r for r in store_mod.records_for_pilot(store, pilot_id)
            if r.annotator_id == annotator_id}.get(target_name)

session_key = f"instances_{pilot_id}_{target_name}_{annotator_id}"
if session_key not in st.session_state:
    if existing is not None:
        st.session_state[session_key] = list(existing.instances)
    else:
        seeded = []
        for model in PROPOSAL_MODEL_PREFERENCE:
            boxes = proposals_by_key.get((model, pilot_id, target_name))
            if boxes:
                seeded = helpers.seed_instances_from_proposals(
                    boxes, model_name=model, checkpoint="see raw-proposals run metadata",
                    prompt="see raw-proposals run metadata", threshold=0.0, timestamp=utc_now_iso())
                break
        st.session_state[session_key] = seeded

default_status = existing.status if existing else ("present" if st.session_state[session_key] else "absent")
status = st.radio("Status", STATUS_ORDER, index=STATUS_ORDER.index(default_status),
                   format_func=lambda s: STATUS_LABELS[s], horizontal=True, key=f"status_{session_key}")

# `seeded_instances`: this session's fixed starting point (existing saved
# instances, or proposal-seeded if none) -- never mutated, used only as
# the reconciliation baseline. `working_key` holds the LIVE, editable list
# the canvas reads from and writes back to every rerun.
seeded_instances = st.session_state[session_key]
working_key = f"working_{session_key}"
if working_key not in st.session_state:
    st.session_state[working_key] = list(seeded_instances)

if status == "present":
    if not (image_path and image_path.exists()):
        st.error("Cannot open the image for box editing.")
    else:
        img = Image.open(image_path).convert("RGB")
        disp_w, disp_h = ch_mod.compute_display_size(img.width, img.height)
        img_resized = img.resize((disp_w, disp_h))

        drawing_mode = st.radio(
            "Canvas mode", ["transform", "rect"], horizontal=True, key=f"{session_key}_mode",
            format_func=lambda m: "Move / resize / select / delete (canvas toolbar)"
            if m == "transform" else "Draw new box")
        st.caption("Dashed boxes are unreviewed model proposals; solid boxes are "
                    "human-accepted/edited/drawn. The canvas's own toolbar (top-right of the "
                    "canvas) provides undo/redo/delete/clear for in-progress edits.")

        initial_drawing_key = f"{session_key}_initial_drawing"
        if initial_drawing_key not in st.session_state:
            st.session_state[initial_drawing_key] = ch_mod.build_initial_drawing(
                seeded_instances, disp_w, disp_h)

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_width=2, stroke_color="#3498db",
            background_image=img_resized, height=disp_h, width=disp_w,
            drawing_mode=drawing_mode, initial_drawing=st.session_state[initial_drawing_key],
            update_streamlit=True, display_toolbar=True, key=f"{session_key}_canvas",
        )

        if canvas_result.json_data is not None:
            canvas_boxes = [ch_mod.canvas_object_to_normalized_bbox(obj, disp_w, disp_h)
                             for obj in canvas_result.json_data.get("objects", [])]
            st.session_state[working_key] = list(
                ch_mod.reconcile_canvas_session(canvas_boxes, seeded_instances))

        working = st.session_state[working_key]
        proposals_awaiting_review = [i for i in working if i.bbox_source == "model_proposed"]
        if proposals_awaiting_review:
            st.write("**Proposals awaiting review** (not saved until accepted or edited):")
            for inst in proposals_awaiting_review:
                cols = st.columns([3, 1, 1])
                cols[0].caption(f"#{inst.instance_index} -- proposed by {inst.proposal_model}")
                if cols[1].button("Accept as-is", key=f"{session_key}_{inst.instance_index}_accept"):
                    idx = working.index(inst)
                    working[idx] = helpers.accept_proposal_instance(inst)
                    st.session_state[working_key] = working
                    st.rerun()
                if cols[2].button("Reject", key=f"{session_key}_{inst.instance_index}_reject"):
                    st.session_state[working_key] = [i for i in working if i is not inst]
                    st.rerun()

        n_savable = len(ch_mod.savable_instances(tuple(working)))
        st.caption(f"{len(working)} box(es) shown on canvas -- **{n_savable} will be saved** "
                   "(unreviewed proposals are excluded until accepted/edited).")

attributes = {}
allowed_attrs = TARGET_ALLOWED_ATTRIBUTE_KEYS.get(target_name, frozenset())
if status == "present" and allowed_attrs:
    for attr_key in sorted(allowed_attrs):
        values = sorted(ATTRIBUTE_ALLOWED_VALUES[attr_key])
        default = existing.attributes.get(attr_key, values[0]) if existing else values[0]
        attributes[attr_key] = st.selectbox(attr_key, values, index=values.index(default),
                                             key=f"{session_key}_{attr_key}")

uncertainty_reason = st.text_input(
    "Uncertainty reason", value=(existing.uncertainty_reason if existing else ""),
    key=f"{session_key}_reason", disabled=(status not in ("uncertain", "not_assessable")))
notes = st.text_input("Notes", value=(existing.notes if existing else ""), key=f"{session_key}_notes")

save_cols = st.columns([1, 1, 3])
save_clicked = save_cols[0].button("Save", type="primary", key=f"{session_key}_save")
save_next_clicked = save_cols[1].button(
    "Save & Next", key=f"{session_key}_savenext",
    disabled=st.session_state[idx_key] >= len(pilot_ids) - 1)
if save_clicked or save_next_clicked:
    rec = PartAnnotationRecord(
        pilot_id=pilot_id, target_name=target_name, status=status,
        annotator_id=annotator_id, annotation_timestamp=utc_now_iso(),
        instances=ch_mod.savable_instances(tuple(st.session_state[working_key]))
        if status == "present" else (),
        attributes=attributes, uncertainty_reason=uncertainty_reason, notes=notes,
    )
    store_mod.upsert_and_save(STORE_PATH, store, rec)
    st.session_state.pop(session_key, None)
    st.session_state.pop(working_key, None)
    st.session_state.pop(f"{session_key}_initial_drawing", None)
    st.success(f"Saved {target_name!r} for {pilot_id}.")
    if save_next_clicked:
        st.session_state[idx_key] += 1
    st.rerun()
