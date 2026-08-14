#!/usr/bin/env python
"""DOAR development-set visual annotation UI (Streamlit).

Replaces the retired console/timed Pass-1+Pass-2 workflow with ONE page
per drawing: the image stays visible the whole time, no timer, no
separate passes. For each visible item the annotator records: label,
location, salient (yes/no), confidence (clear/ambiguous), and an
optional note -- exactly `ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md`'s
schema. No console typing is required for annotation itself (a plain
form adds items; each is then edited/deleted via its own card below the
form).

Item entry uses a plain `st.form` with individually-keyed widgets, NOT
`st.data_editor` -- `st.data_editor`'s inline checkbox column lost its
value on rerun (a real bug hit while dogfooding this tool: checking
"Salient" inside the editor did not survive the rerun the checkbox click
itself triggers). A form's own widgets keep their state across reruns by
construction, so this cannot recur the same way.

Launch ONE instance per annotator (each instance only ever reads/writes
its own `annotations/<annotator_id>/` subtree -- A1 can never see A2's
annotations or vice versa, because nothing in this script ever
constructs a path outside the launching annotator's own id):

    streamlit run scripts/annotate_ui.py -- --annotator A1
    streamlit run scripts/annotate_ui.py -- --annotator A2

Saves to `annotations/<annotator_id>/<image_id>.json` immediately after
every Add/Update/Delete, and immediately when "Mark this drawing
complete" is toggled -- no separate Save step is required to avoid
losing work, though a Save button is still provided so the annotator
gets explicit confirmation before navigating away. Re-opening always
resumes exactly what was last saved for that image, since every widget
reads its initial value from the freshly-reloaded record on each rerun.
Never invents or fabricates an item: an empty items list is recorded
exactly as an empty list, not silently skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEV_SET_PATH = ROOT / "DEVELOPMENT_SET_15.json"
# DOAR_ANNOTATIONS_DIR lets tests redirect saves to a throwaway directory
# (streamlit.testing.v1.AppTest executes this script fresh each run, so
# tests can't just monkeypatch a module attribute after import) -- unset
# in normal use, where this is exactly ROOT / "annotations".
ANNOTATIONS_DIR = Path(os.environ.get("DOAR_ANNOTATIONS_DIR", str(ROOT / "annotations")))

CONFIDENCE_OPTIONS = ("clear", "ambiguous")
BLANK_ITEM = {"label": "", "location": "", "salient": False, "confidence": "clear", "note": ""}

NON_NEGOTIABLE_REMINDER = (
    "Record ONLY what is visibly drawn -- objects, marks, shapes, people, animals, scenery. "
    "Do NOT record emotions, personality traits, mood, or any psychological interpretation. "
    "If unsure what something is, use the label 'unclear region' or 'unidentified mark'."
)


def parse_annotator_id() -> str | None:
    """Reads --annotator from sys.argv (Streamlit forwards everything
    after `--` on the command line verbatim). Returns None rather than
    raising if it's missing, so the caller can show a friendly in-page
    error instead of a crash trace."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--annotator")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.annotator


def load_development_set() -> list[dict]:
    return json.loads(DEV_SET_PATH.read_text(encoding="utf-8"))["images"]


def record_path(annotator_id: str, image_id: str) -> Path:
    out_dir = ANNOTATIONS_DIR / annotator_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_id}.json"


def load_record(annotator_id: str, image_id: str) -> dict:
    path = record_path(annotator_id, image_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"annotator_id": annotator_id, "image_id": image_id, "items": [], "complete": False}


def save_record(annotator_id: str, image_id: str, items: list[dict], complete: bool) -> None:
    record = {"annotator_id": annotator_id, "image_id": image_id, "items": items, "complete": complete}
    record_path(annotator_id, image_id).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def build_item(label: str, location: str, salient: bool, confidence: str, note: str) -> dict | None:
    """Builds one item dict in the exact `{label, location, salient,
    confidence, note}` schema, or returns None if `label` is blank
    (never records a placeholder item)."""
    clean_label = (label or "").strip()
    if not clean_label:
        return None
    if confidence not in CONFIDENCE_OPTIONS:
        confidence = "clear"
    return {
        "label": clean_label, "location": (location or "").strip(), "salient": bool(salient),
        "confidence": confidence, "note": (note or "").strip(),
    }


def widget_key(annotator_id: str, image_id: str, generation: int, name: str) -> str:
    """A brand-new generation number produces a brand-new widget key, so
    switching between "add" and "edit item i" (or resetting after a
    submit) is a fresh widget Streamlit has never seen -- its `value=`
    argument is then honestly the initial value, sidestepping both the
    "cannot set an already-instantiated widget's state" restriction and
    any ambiguity in exactly when `st.form(clear_on_submit=True)` resets
    (verified empirically to NOT reliably reset a prefilled `value=`
    field within a single simulated rerun)."""
    return f"{name}_{annotator_id}_{image_id}_{generation}"


def completion_marker(record: dict) -> str:
    if record.get("complete"):
        return "✅"  # check mark -- explicitly marked complete
    if record["items"]:
        return "\U0001F4DD"  # memo -- has a draft, not yet marked complete
    return "⬜"  # empty box -- untouched


def main() -> None:
    st.set_page_config(page_title="DOAR annotation", layout="wide")

    annotator_id = parse_annotator_id()
    if not annotator_id:
        st.error(
            "No --annotator id given. Launch with:\n\n"
            "    streamlit run scripts/annotate_ui.py -- --annotator A1\n\n"
            "(or --annotator A2)")
        st.stop()

    images = load_development_set()
    total = len(images)

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    st.sidebar.markdown(f"## Annotator: `{annotator_id}`")
    st.sidebar.caption(NON_NEGOTIABLE_REMINDER)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Drawings")
    records_by_image = {im["image_id"]: load_record(annotator_id, im["image_id"]) for im in images}
    for i, image in enumerate(images):
        record = records_by_image[image["image_id"]]
        label = f"{completion_marker(record)} {i + 1}. {image['image_id']}"
        if st.sidebar.button(label, key=f"jump_{i}", width='stretch'):
            st.session_state.current_index = i
            st.rerun()

    done_count = sum(1 for r in records_by_image.values() if r.get("complete"))
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{done_count} / {total} drawings marked complete.")

    index = st.session_state.current_index
    image = images[index]
    image_id = image["image_id"]
    image_path = ROOT / image["relative_path"]

    st.title(f"Annotator {annotator_id} -- Image {index + 1} / {total} -- {image_id}")

    record = load_record(annotator_id, image_id)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.image(str(image_path), width='stretch')

    with col_right:
        st.caption(NON_NEGOTIABLE_REMINDER)

        items = record["items"]  # mutated in place below, then explicitly saved -- never a shadow copy.

        editing_index_key = f"editing_index_{annotator_id}_{image_id}"
        form_gen_key = f"form_gen_{annotator_id}_{image_id}"
        editing_index = st.session_state.get(editing_index_key)
        generation = st.session_state.get(form_gen_key, 0)

        if editing_index is not None and 0 <= editing_index < len(items):
            prefill = items[editing_index]
        else:
            editing_index = None
            prefill = BLANK_ITEM

        st.subheader("Edit item" if editing_index is not None else "Add item")
        with st.form(key=widget_key(annotator_id, image_id, generation, "item_form")):
            label = st.text_input(
                "Label", value=prefill["label"], key=widget_key(annotator_id, image_id, generation, "label"))
            location = st.text_input(
                "Location", value=prefill["location"],
                key=widget_key(annotator_id, image_id, generation, "location"))
            salient = st.checkbox(
                "Salient", value=bool(prefill["salient"]),
                key=widget_key(annotator_id, image_id, generation, "salient"))
            confidence_index = (
                CONFIDENCE_OPTIONS.index(prefill["confidence"]) if prefill["confidence"] in CONFIDENCE_OPTIONS else 0)
            confidence = st.radio(
                "Confidence", CONFIDENCE_OPTIONS, index=confidence_index, horizontal=True,
                key=widget_key(annotator_id, image_id, generation, "confidence"))
            note = st.text_input(
                "Note (optional)", value=prefill["note"], key=widget_key(annotator_id, image_id, generation, "note"))
            submitted = st.form_submit_button("Update Item" if editing_index is not None else "Add Item")

        if editing_index is not None:
            if st.button("Cancel edit", key=f"cancel_edit_{annotator_id}_{image_id}_{generation}"):
                st.session_state[editing_index_key] = None
                st.session_state[form_gen_key] = generation + 1
                st.rerun()

        if submitted:
            new_item = build_item(label, location, salient, confidence, note)
            if new_item is None:
                st.warning("Label is required -- item not saved.")
            else:
                if editing_index is not None:
                    items[editing_index] = new_item
                else:
                    items.append(new_item)
                save_record(annotator_id, image_id, items, record.get("complete", False))
                st.session_state[editing_index_key] = None
                st.session_state[form_gen_key] = generation + 1
                st.success(f"Saved: {new_item['label']}")
                st.rerun()

        st.markdown("---")
        st.subheader(f"Items ({len(items)})")
        if not items:
            st.caption("No items recorded yet.")
        for i, item in enumerate(items):
            with st.container(border=True):
                item_cols = st.columns([5, 1, 1])
                with item_cols[0]:
                    salience_note = "salient" if item.get("salient") else "not salient"
                    note_suffix = f" | note: {item['note']}" if item.get("note") else ""
                    st.markdown(
                        f"**{item['label']}** ({salience_note}, {item.get('confidence', 'clear')})  \n"
                        f"location: {item.get('location') or '-'}{note_suffix}")
                with item_cols[1]:
                    if st.button("Edit", key=f"edit_{annotator_id}_{image_id}_{i}"):
                        st.session_state[editing_index_key] = i
                        st.session_state[form_gen_key] = generation + 1
                        st.rerun()
                with item_cols[2]:
                    if st.button("Delete", key=f"delete_{annotator_id}_{image_id}_{i}"):
                        del items[i]
                        save_record(annotator_id, image_id, items, record.get("complete", False))
                        if editing_index == i:
                            st.session_state[editing_index_key] = None
                            st.session_state[form_gen_key] = generation + 1
                        elif editing_index is not None and editing_index > i:
                            st.session_state[editing_index_key] = editing_index - 1
                        st.rerun()

        st.markdown("---")
        complete_key = f"complete_{annotator_id}_{image_id}"
        complete = st.checkbox("Mark this drawing complete", value=record.get("complete", False), key=complete_key)
        if complete != record.get("complete", False):
            save_record(annotator_id, image_id, items, complete)

        st.markdown("---")
        nav_cols = st.columns(3)
        with nav_cols[0]:
            if st.button("Previous", disabled=index == 0, width='stretch'):
                st.session_state.current_index = max(0, index - 1)
                st.rerun()
        with nav_cols[1]:
            if st.button("Save", width='stretch'):
                save_record(annotator_id, image_id, items, complete)
                st.success("Saved.")
        with nav_cols[2]:
            if st.button("Save & Next", disabled=index == total - 1, width='stretch'):
                save_record(annotator_id, image_id, items, complete)
                st.session_state.current_index = min(total - 1, index + 1)
                st.rerun()


if __name__ == "__main__":
    main()
