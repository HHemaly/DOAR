#!/usr/bin/env python
"""DOAR development-set visual annotation UI (Streamlit).

Replaces the retired console/timed Pass-1+Pass-2 workflow with ONE page
per drawing: the image stays visible the whole time, no timer, no
separate passes. For each visible item the annotator records: label,
location, salient (yes/no), confidence (clear/ambiguous), and an
optional note -- exactly `ANNOTATION_PROTOCOL_DEVELOPMENT_SET.md`'s
schema. No console typing is required for annotation itself (add/edit/
delete happen in an editable on-screen table).

Launch ONE instance per annotator (each instance only ever reads/writes
its own `annotations/<annotator_id>/` subtree -- A1 can never see A2's
annotations or vice versa, because nothing in this script ever
constructs a path outside the launching annotator's own id):

    streamlit run scripts/annotate_ui.py -- --annotator A1
    streamlit run scripts/annotate_ui.py -- --annotator A2

Saves to `annotations/<annotator_id>/<image_id>.json` on every edit
(autosave -- no separate Save step is required to avoid losing work,
though a Save button is still provided so the annotator gets explicit
confirmation). Re-opening always resumes exactly what was last saved for
that image. Never invents or fabricates an item: an empty items list is
recorded exactly as an empty list, not silently skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEV_SET_PATH = ROOT / "DEVELOPMENT_SET_15.json"
# DOAR_ANNOTATIONS_DIR lets tests redirect saves to a throwaway directory
# (streamlit.testing.v1.AppTest executes this script fresh each run, so
# tests can't just monkeypatch a module attribute after import) -- unset
# in normal use, where this is exactly ROOT / "annotations".
ANNOTATIONS_DIR = Path(os.environ.get("DOAR_ANNOTATIONS_DIR", str(ROOT / "annotations")))

ITEM_COLUMNS = ["label", "location", "salient", "confidence", "note"]

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


def items_to_dataframe(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame([], columns=ITEM_COLUMNS)
    return pd.DataFrame(
        [{
            "label": item.get("label", ""), "location": item.get("location", ""),
            "salient": bool(item.get("salient", False)), "confidence": item.get("confidence", "clear"),
            "note": item.get("note", ""),
        } for item in items],
        columns=ITEM_COLUMNS)


def dataframe_to_items(df: pd.DataFrame) -> list[dict]:
    """Drops any row with a blank label (e.g. a freshly-added, not-yet-
    filled-in row from the editor) -- never records a placeholder item."""
    items = []
    for _, row in df.iterrows():
        label = str(row.get("label", "") or "").strip()
        if not label:
            continue
        confidence = row.get("confidence", "clear")
        if confidence not in ("clear", "ambiguous"):
            confidence = "clear"
        items.append({
            "label": label, "location": str(row.get("location", "") or "").strip(),
            "salient": bool(row.get("salient", False)), "confidence": confidence,
            "note": str(row.get("note", "") or "").strip(),
        })
    return items


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

        df = items_to_dataframe(record["items"])
        edited_df = st.data_editor(
            df, num_rows="dynamic", width='stretch', key=f"editor_{annotator_id}_{image_id}",
            column_config={
                "label": st.column_config.TextColumn("Label", required=True),
                "location": st.column_config.TextColumn("Location"),
                "salient": st.column_config.CheckboxColumn("Salient", default=False),
                "confidence": st.column_config.SelectboxColumn(
                    "Confidence", options=["clear", "ambiguous"], default="clear"),
                "note": st.column_config.TextColumn("Note (optional)"),
            },
        )

        items = dataframe_to_items(edited_df)
        complete = st.checkbox(
            "Mark this drawing complete", value=record.get("complete", False),
            key=f"complete_{annotator_id}_{image_id}")

        # Autosave: every rerun (i.e. every edit) persists the current
        # state, so nothing typed is ever lost even if the annotator
        # closes the browser tab without pressing Save.
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
