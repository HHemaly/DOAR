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

Box coordinates are entered as normalized x,y,w,h numbers (0-1), mirroring
phase2c_annotation_app.py's own existing bbox text-input pattern. A visual
click-and-drag canvas would be a real usability improvement here (e.g. the
`streamlit-drawable-canvas` package) -- not added this phase without
separate approval, since it is a new runtime dependency; see
PHASE2C5_RULE_CRITICAL_ANNOTATION_REPORT.md's UI-status section.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

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
    target_name = st.selectbox("Target", PART_TARGETS, key="target_name")
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

instances = st.session_state[session_key]
if status == "present":
    st.write(f"**{len(instances)} instance(s)**")
    for i, inst in enumerate(list(instances)):
        cols = st.columns([1.2, 1, 1, 1, 1, 1.3, 0.8])
        cols[0].caption(f"#{i} -- {inst.bbox_source}"
                         + (f" ({inst.proposal_model})" if inst.proposal_model else ""))
        x = cols[1].number_input("x", 0.0, 1.0, inst.bbox[0], key=f"{session_key}_{i}_x")
        y = cols[2].number_input("y", 0.0, 1.0, inst.bbox[1], key=f"{session_key}_{i}_y")
        w = cols[3].number_input("w", 0.0, 1.0, inst.bbox[2], key=f"{session_key}_{i}_w")
        h = cols[4].number_input("h", 0.0, 1.0, inst.bbox[3], key=f"{session_key}_{i}_h")
        new_bbox = (x, y, w, h)
        if inst.bbox_source == "model_proposed":
            if cols[5].button("Accept", key=f"{session_key}_{i}_accept"):
                instances[i] = helpers.accept_proposal_instance(inst)
                st.rerun()
            elif new_bbox != inst.bbox:
                instances[i] = helpers.edit_proposal_instance(inst, new_bbox)
        elif new_bbox != inst.bbox:
            instances[i] = helpers.edit_proposal_instance(inst, new_bbox) \
                if inst.proposal_model else helpers.new_manual_instance(i, new_bbox)
        if cols[6].button("Delete", key=f"{session_key}_{i}_del"):
            st.session_state[session_key] = list(helpers.delete_instance(instances, i))
            st.rerun()
    if st.button("+ Add manual box", key=f"{session_key}_add"):
        st.session_state[session_key].append(
            helpers.new_manual_instance(len(instances), (0.1, 0.1, 0.2, 0.2)))
        st.rerun()

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
        instances=tuple(instances) if status == "present" else (),
        attributes=attributes, uncertainty_reason=uncertainty_reason, notes=notes,
    )
    store_mod.upsert_and_save(STORE_PATH, store, rec)
    st.success(f"Saved {target_name!r} for {pilot_id}.")
    if save_next_clicked:
        st.session_state[idx_key] += 1
    st.rerun()
