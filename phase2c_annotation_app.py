"""Phase 2C.1 local human annotation and review interface. Launch with:

    .\\.venv\\Scripts\\Activate.ps1
    python -m streamlit run phase2c_annotation_app.py

then open the local URL Streamlit prints (default http://localhost:8501).

Shows one blinded drawing at a time from the private workspace built by
`doar.phase2c1.workspace.build_pilot_workspace` -- NEVER the emotion-class
folder, original filename, a detector prediction, or a psychological
interpretation. Every Save/Save & Next writes the whole store immediately
(atomic replace, doar.phase2c1.store.upsert_and_save), so closing the tab or
the terminal never loses progress; re-launching resumes exactly where you
left off.

Review mode asks the reviewer to record their OWN independent judgment for
each class BEFORE revealing the primary annotator's label, to reduce
anchoring bias -- same pattern as phase7b_review_app.py's "reveal hidden
details" gate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.ontology import CLASSES  # noqa: E402
from doar.phase2c1 import quality as quality_mod  # noqa: E402
from doar.phase2c1 import store as store_mod  # noqa: E402
from doar.phase2c1 import workspace as workspace_mod  # noqa: E402
from doar.phase2c1.schema import AnnotationRecord, utc_now_iso  # noqa: E402

# Overridable via environment variable ONLY for isolated testing (mirrors
# phase7b_review_app.py's own pattern) -- unset in normal use, so real usage
# always resolves to the real, private, gitignored production paths below.
IMAGES_DIR = Path(os.environ.get(
    "DOAR_PHASE2C1_IMAGES_DIR", str(ROOT / "outputs/phase2c1/private_images")))
STORE_PATH = Path(os.environ.get(
    "DOAR_PHASE2C1_STORE_PATH", str(ROOT / "outputs/phase2c1/annotation_store.csv")))
EXPORT_DIR = Path(os.environ.get(
    "DOAR_PHASE2C1_EXPORT_DIR", str(ROOT / "outputs/phase2c1/exports")))
# Read only for image_id/source_image_group (opaque hashes, already visible
# in the committed artifacts/phase2b/annotation_manifest.csv precedent) --
# `original_path` is never read into any UI-facing variable below.
MAPPING_PATH = Path(os.environ.get(
    "DOAR_PHASE2C1_MAPPING_PATH", str(ROOT / "outputs/phase2c1/private_pilot_mapping.csv")))

STATUS_LABELS = {
    "present": "Present", "absent": "Absent",
    "uncertain": "Uncertain", "not_assessable": "Not assessable",
}
STATUS_ORDER = ["absent", "present", "uncertain", "not_assessable"]

st.set_page_config(page_title="Phase 2C.1 object annotation", layout="wide")
st.title("Phase 2C.1 -- object-evidence annotation")
st.caption(
    "Research/annotation tool only. Each image is shown blind: no emotion "
    "label, source folder, detector output, or psychological interpretation "
    "is ever displayed here. Judge only what is visibly drawn."
)

if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
    st.error(
        f"No blinded images found at {IMAGES_DIR}. Build the private workspace first "
        "(doar.phase2c1.workspace.build_pilot_workspace) -- see "
        "PHASE2C1_ANNOTATION_PROVENANCE_AUDIT.md."
    )
    st.stop()

pilot_ids = workspace_mod.list_workspace_images(IMAGES_DIR)
store = store_mod.load_store(STORE_PATH)
try:
    _mapping_rows = workspace_mod.load_pilot_mapping(MAPPING_PATH)
except FileNotFoundError:
    _mapping_rows = []
PROVENANCE_BY_PILOT_ID = {
    r["pilot_id"]: (r["image_id"], r["source_image_group"]) for r in _mapping_rows
}

with st.sidebar:
    st.header("Annotator")
    annotator_id = st.text_input("Your annotator ID", key="annotator_id",
                                  help="Required. Never fabricated or defaulted for you.")
    mode = st.radio("Mode", ["Primary annotation", "Review"], key="mode")
    st.divider()
    st.header("Progress")
    if annotator_id:
        comp = quality_mod.completion_rate(store, pilot_ids, annotator_id)
        st.progress(comp["completion_rate"] or 0.0)
        st.write(f"**{comp['complete_images']} / {comp['total_images']}** images "
                 f"fully annotated by you ({comp['incomplete_images']} remaining)")
    else:
        st.info("Enter an annotator ID to see your progress.")
    st.divider()
    st.header("Export")
    st.caption("Safe to run at any point -- reflects whatever is saved so far.")
    if st.button("Export CSV + JSON", type="primary"):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        n_csv = store_mod.export_csv(store, EXPORT_DIR / "phase2c1_annotations.csv")
        n_json = store_mod.export_json(store, EXPORT_DIR / "phase2c1_annotations.json")
        st.success(f"Wrote {n_csv} rows to {EXPORT_DIR}")
    st.divider()
    st.header("Annotation quality")
    if st.button("Show quality report"):
        st.session_state["show_quality"] = True
    if st.session_state.get("show_quality"):
        st.caption("Genuine human-human inter-rater agreement (excludes legacy Phase 2B provisional labels):")
        st.json(quality_mod.compute_agreement_report(store))
        st.caption(
            "Separate reference comparison against legacy Phase 2B provisional labels "
            "(NOT inter-rater reliability -- those labels are single-annotator, unreviewed, "
            "never confirmed ground truth):"
        )
        st.json(quality_mod.compute_provisional_reference_comparison(store))
        st.json(quality_mod.review_coverage(store))

if not annotator_id:
    st.warning("Enter your annotator ID in the sidebar to begin.")
    st.stop()

def _n_classes_done(pilot_id: str) -> int:
    return len({r.class_name for r in store_mod.records_for_pilot(store, pilot_id)
                if r.annotator_id == annotator_id})


idx_key = f"idx_{mode}"
if idx_key not in st.session_state:
    st.session_state[idx_key] = next(
        (i for i, pid in enumerate(pilot_ids) if _n_classes_done(pid) < len(CLASSES)), 0,
    )

nav_cols = st.columns([1, 1, 2, 1])
if nav_cols[0].button("Previous", disabled=st.session_state[idx_key] <= 0):
    st.session_state[idx_key] -= 1
    st.rerun()
if nav_cols[1].button("Next", disabled=st.session_state[idx_key] >= len(pilot_ids) - 1):
    st.session_state[idx_key] += 1
    st.rerun()
jump = nav_cols[2].number_input(
    "Jump to image #", min_value=1, max_value=len(pilot_ids),
    value=st.session_state[idx_key] + 1,
    # Keyed on the current index so that any programmatic index change
    # (Next/Save & Next/Previous/First unannotated) creates a FRESH widget
    # instance with `value=` correctly applied -- otherwise Streamlit keeps
    # this widget's stale prior value across reruns and this code would
    # immediately snap the index back to it, undoing the navigation.
    key=f"jump_{mode}_{st.session_state[idx_key]}",
)
if jump - 1 != st.session_state[idx_key]:
    st.session_state[idx_key] = jump - 1
    st.rerun()
if nav_cols[3].button("First unannotated"):
    st.session_state[idx_key] = next(
        (i for i, pid in enumerate(pilot_ids) if _n_classes_done(pid) < len(CLASSES)), 0,
    )
    st.rerun()

pilot_id = pilot_ids[st.session_state[idx_key]]
image_path = workspace_mod.image_path_for_pilot_id(IMAGES_DIR, pilot_id)
existing_rows = {r.class_name: r for r in store_mod.records_for_pilot(store, pilot_id)
                  if r.annotator_id == annotator_id}

st.write(f"Image **{st.session_state[idx_key] + 1} / {len(pilot_ids)}** "
         f"-- {len(existing_rows)}/{len(CLASSES)} classes annotated by you")
if image_path and image_path.exists():
    st.image(str(image_path), width=400)
else:
    st.error(f"Image file missing on disk for {pilot_id!r}")


def _class_widgets(cls_name: str, key_prefix: str, existing: AnnotationRecord | None) -> dict:
    default_status = existing.status if existing else "absent"
    status = st.radio(
        "Status", STATUS_ORDER, index=STATUS_ORDER.index(default_status),
        format_func=lambda s: STATUS_LABELS[s], key=f"{key_prefix}_status", horizontal=True,
    )
    cols = st.columns([1, 1, 2])
    instance_count = cols[0].number_input(
        "Count", min_value=0, max_value=50,
        value=(existing.instance_count if existing else (1 if status == "present" else 0)),
        key=f"{key_prefix}_count", disabled=(status != "present"),
    )
    # A widget's `value=` argument only applies the FIRST time its key is
    # created -- once a rerun flips status to "present" for an
    # already-disabled, previously-0-valued widget, Streamlit keeps the
    # stale 0 from session_state regardless of `value=` here. Clamp
    # explicitly rather than relying on the widget's own default.
    if status == "present" and instance_count < 1:
        instance_count = 1
    partial = cols[1].checkbox(
        "Partial/occluded", value=(existing.partial_or_occluded if existing else False),
        key=f"{key_prefix}_partial",
    )
    uncertainty_reason = cols[2].text_input(
        "Uncertainty reason", value=(existing.uncertainty_reason if existing else ""),
        key=f"{key_prefix}_reason", disabled=(status not in ("uncertain", "not_assessable")),
    )
    existing_bbox_str = ",".join(f"{v:.3f}" for v in existing.bbox) if (existing and existing.bbox) else ""
    bbox_str = st.text_input(
        "Bounding box (optional): x,y,w,h -- normalized 0-1, e.g. 0.1,0.2,0.3,0.4",
        value=existing_bbox_str, key=f"{key_prefix}_bbox", disabled=(status != "present"),
    )
    bbox_error = None
    bbox = None
    bbox_str = bbox_str.strip()
    if status == "present" and bbox_str:
        try:
            parts = tuple(float(v) for v in bbox_str.split(","))
            if len(parts) != 4 or any(not (0.0 <= v <= 1.0) for v in parts) or parts[2] <= 0 or parts[3] <= 0:
                raise ValueError
            bbox = parts
        except ValueError:
            bbox_error = "Bounding box must be 4 comma-separated numbers in [0, 1], e.g. 0.1,0.2,0.3,0.4"
            st.warning(bbox_error)
    notes = st.text_input("Notes", value=(existing.notes if existing else ""), key=f"{key_prefix}_notes")
    return {
        "status": status,
        "instance_count": instance_count if status == "present" else 0,
        "partial_or_occluded": partial,
        "uncertainty_reason": uncertainty_reason,
        "bbox": bbox,
        "bbox_error": bbox_error,
        "notes": notes,
    }


def _build_record(pilot_id: str, cls_name: str, annotator_id: str, values: dict) -> AnnotationRecord:
    image_id, source_image_group = PROVENANCE_BY_PILOT_ID.get(pilot_id, ("", ""))
    return AnnotationRecord(
        pilot_id=pilot_id, image_id=image_id, source_image_group=source_image_group, class_name=cls_name,
        status=values["status"], annotator_id=annotator_id, annotation_timestamp=utc_now_iso(),
        instance_count=values["instance_count"], bbox=values.get("bbox"),
        partial_or_occluded=values["partial_or_occluded"],
        uncertainty_reason=values["uncertainty_reason"], notes=values["notes"],
        source_manifest_version=workspace_mod.PILOT_MANIFEST_VERSION,
    )


if mode == "Primary annotation":
    class_values = {}
    for cls in CLASSES:
        with st.expander(f"**{cls.name}** -- {cls.minimum_visible_evidence}",
                          expanded=(cls.name not in existing_rows or existing_rows[cls.name].status != "absent")):
            class_values[cls.name] = _class_widgets(cls.name, f"primary_{pilot_id}_{cls.name}",
                                                     existing_rows.get(cls.name))

    save_cols = st.columns([1, 1, 3])
    save_clicked = save_cols[0].button("Save", type="primary", key=f"save_{pilot_id}")
    save_next_clicked = save_cols[1].button(
        "Save & Next", key=f"savenext_{pilot_id}",
        disabled=st.session_state[idx_key] >= len(pilot_ids) - 1,
    )
    if save_clicked or save_next_clicked:
        for cls in CLASSES:
            rec = _build_record(pilot_id, cls.name, annotator_id, class_values[cls.name])
            store_mod.upsert(store, rec)
        store_mod.save_store(STORE_PATH, store)
        st.success(f"Saved all {len(CLASSES)} classes for {pilot_id}.")
        if save_next_clicked:
            st.session_state[idx_key] += 1
        st.rerun()

else:  # Review mode
    st.info(
        "Record your OWN independent judgment for each class first. The primary "
        "annotator's original label only becomes visible after you save your own "
        "judgment for that class -- this is deliberate, to reduce anchoring bias."
    )
    primary_rows = {
        r.class_name: r for r in store_mod.records_for_pilot(store, pilot_id)
        if r.annotator_id != annotator_id and r.review_status != "reviewed"
    }
    # Fall back to any primary row if every candidate is already reviewed by
    # someone (still viewable/re-reviewable, just not the default filter above).
    if not primary_rows:
        primary_rows = {
            r.class_name: r for r in store_mod.records_for_pilot(store, pilot_id)
            if r.annotator_id != annotator_id
        }
    if not primary_rows:
        st.warning(f"No primary annotation exists yet for {pilot_id} -- nothing to review.")
    for cls in CLASSES:
        primary = primary_rows.get(cls.name)
        if primary is None:
            continue
        key_prefix = f"review_{pilot_id}_{cls.name}_{annotator_id}"
        own_key = f"{key_prefix}_saved"
        with st.expander(f"**{cls.name}** -- {cls.minimum_visible_evidence}", expanded=True):
            values = _class_widgets(cls.name, key_prefix, existing_rows.get(cls.name))
            if st.button("Save my independent judgment", key=f"{key_prefix}_savebtn"):
                rec = _build_record(pilot_id, cls.name, annotator_id, values)
                store_mod.upsert_and_save(STORE_PATH, store, rec)
                st.session_state[own_key] = True
                st.rerun()

            my_row = existing_rows.get(cls.name)
            if my_row is not None or st.session_state.get(own_key):
                with st.expander("Reveal primary annotator's label (unlocked)"):
                    st.write({
                        "status": primary.status, "instance_count": primary.instance_count,
                        "uncertainty_reason": primary.uncertainty_reason, "notes": primary.notes,
                        "annotator_id": primary.annotator_id,
                    })
                    outcomes = [("Agreement", "agreement"),
                                ("Disagreement (unresolved)", "disagreement_unresolved"),
                                ("Disagreement (resolved)", "disagreement_resolved")]
                    outcome_cols = st.columns(len(outcomes))
                    for col, (label, status) in zip(outcome_cols, outcomes):
                        if col.button(label, key=f"{key_prefix}_{status}"):
                            store_mod.apply_review_outcome(
                                STORE_PATH, store, pilot_id=pilot_id, class_name=cls.name,
                                primary_annotator_id=primary.annotator_id, reviewer_id=annotator_id,
                                adjudication_status=status, review_timestamp=utc_now_iso(),
                            )
                            st.success(f"Recorded: {label}")
                            st.rerun()
            else:
                st.caption("Save your own judgment above to reveal the primary label.")
