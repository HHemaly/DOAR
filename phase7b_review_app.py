"""Local, blind human-review interface for Phase 7B's duplicate-pair
decisions. Launch with:

    .\\.venv\\Scripts\\Activate.ps1
    python -m streamlit run phase7b_review_app.py

then open the local URL Streamlit prints (default http://localhost:8501).

Every image pair is shown with NO emotion label, split, hash distance, or
Claude's preliminary judgment -- see doar.human_review.blind_item_for_display.
Your decisions save to outputs/phase7b/human_review/app_data/decisions.json
after every click, so closing the tab or the terminal never loses progress.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from doar.human_review import (  # noqa: E402
    HUMAN_JUDGMENT_CHOICES, HUMAN_JUDGMENT_LABELS,
    blind_item_for_display, export_all, load_decisions, load_registry,
    progress_by_category, save_decision,
)

# Overridable via environment variable ONLY for isolated testing (see
# tests/test_phase7b_review_app_smoke.py) -- unset in normal use, so real
# usage always resolves to the real production paths below. This exists so
# tests can never write a synthetic/test decision into the real
# decisions.json that real human-review progress lives in.
REGISTRY_PATH = Path(os.environ.get(
    "DOAR_PHASE7B_REGISTRY_PATH", str(ROOT / "outputs/phase7b/human_review/app_data/items_registry.json")))
DECISIONS_PATH = Path(os.environ.get(
    "DOAR_PHASE7B_DECISIONS_PATH", str(ROOT / "outputs/phase7b/human_review/app_data/decisions.json")))
EXPORT_DIR = Path(os.environ.get(
    "DOAR_PHASE7B_EXPORT_DIR", str(ROOT / "outputs/phase7b/human_review/exports")))

CATEGORY_TITLES = {
    "ambiguous": "Ambiguous pairs (Round 1: broad aHash-oriented audit, 87 pairs)",
    "threshold_boundary": "Threshold-boundary pairs (dHash distances 2-8, 71 pairs)",
    "component_17": "The 17-image cross-label component (all 47 internal near-dup edges)",
    "policy_change": "Groups that change between candidate thresholds 2 / 4 / 6 (20 sampled bridging edges)",
}
CATEGORY_HELP = {
    "ambiguous": (
        "The first blinded audit: pairs sampled across aHash Hamming distances "
        "0-5, plus hard negatives and large-group samples."
    ),
    "threshold_boundary": (
        "42 pairs sampled directly at each dHash distance 2 through 8 (Round "
        "2), plus every remaining dataset-wide distance-4 edge not already "
        "covered elsewhere (29 more) -- distance 4 is exhaustively covered "
        "here because it's the specific candidate boundary for a "
        "'selectively reviewed distance-4' conservative policy: include "
        "distance 2-3 automatically, include distance 4 only edge-by-edge "
        "once a human confirms it."
    ),
    "component_17": (
        "Every real near-dup edge (dHash <= 6) connecting two of the 17 "
        "images that dHash threshold 6 merges into one component. Reviewing "
        "these tells you whether that component is one true duplicate "
        "cluster or a chain."
    ),
    "policy_change": (
        "A sample of edges elsewhere in the dataset whose distance falls "
        "strictly between two candidate thresholds (2<d<=4 or 4<d<=6), so "
        "each one is exactly the kind of edge that changes group membership "
        "depending which threshold you adopt. Excludes the 17-image "
        "component, which is already covered exhaustively above."
    ),
}

st.set_page_config(page_title="Phase 7B duplicate-pair review", layout="wide")
st.title("Phase 7B duplicate-pair human review")
st.caption(
    "Research/dataset-engineering decision support only. Each pair is shown "
    "blind: no emotion label, split, hash distance, or Claude's preliminary "
    "judgment. Your decision is the one that matters here."
)

if not REGISTRY_PATH.exists():
    st.error(f"Item registry not found at {REGISTRY_PATH}. Build it first (see PHASE7B_DUPLICATE_POLICY.md).")
    st.stop()

registry = load_registry(REGISTRY_PATH)
decisions = load_decisions(DECISIONS_PATH)
categories_order = registry["categories_order"]
items = registry["items"]

progress = progress_by_category(categories_order, decisions)
total_items = sum(v["total"] for v in progress.values())
total_reviewed = sum(v["reviewed"] for v in progress.values())

with st.sidebar:
    st.header("Overall progress")
    st.progress(total_reviewed / total_items if total_items else 0.0)
    st.write(f"**{total_reviewed} / {total_items}** items reviewed ({total_items - total_reviewed} remaining)")
    for cat, p in progress.items():
        st.write(f"- {CATEGORY_TITLES.get(cat, cat)}: {p['reviewed']}/{p['total']}")
    st.divider()
    st.header("Export results")
    st.caption(
        "Safe to run at any point, including mid-review -- exports reflect "
        "whatever has been decided so far and can be re-run repeatedly. "
        "Does NOT select or lock a final duplicate-detection policy."
    )
    review_complete = total_reviewed >= total_items
    if not review_complete:
        st.warning(
            f"Review is INCOMPLETE: {total_items - total_reviewed} of {total_items} "
            f"items still have no decision. You can still export now for a partial "
            f"snapshot, but the dataset gate (dataset_gate.py) will not pass and no "
            f"duplicate-detection policy may be approved until every item is decided."
        )
    if st.button("Export human_pair_reviews.csv, human_group_reviews.csv, "
                  "reviewer_agreement_report.json, threshold_precision_summary.json, "
                  "unresolved_items.csv", type="primary"):
        summary = export_all(REGISTRY_PATH, DECISIONS_PATH, EXPORT_DIR)
        if review_complete:
            st.success(f"Wrote {len(summary['files_written'])} files to {EXPORT_DIR}")
        else:
            st.warning(
                f"Wrote {len(summary['files_written'])} files to {EXPORT_DIR} -- "
                f"PARTIAL export, {total_items - total_reviewed} items still undecided."
            )
        st.json(summary)

tabs = st.tabs([CATEGORY_TITLES[c] for c in categories_order])

for tab, category in zip(tabs, categories_order):
    with tab:
        st.info(CATEGORY_HELP[category])
        ids = categories_order[category]
        p = progress[category]
        st.write(f"**{p['reviewed']} / {p['total']}** reviewed in this category ({p['remaining']} remaining)")
        st.progress(p["reviewed"] / p["total"] if p["total"] else 0.0)

        idx_key = f"idx_{category}"
        if idx_key not in st.session_state:
            first_unreviewed = next((i for i, iid in enumerate(ids) if iid not in decisions), 0)
            st.session_state[idx_key] = first_unreviewed

        nav_cols = st.columns([1, 1, 2, 1])
        if nav_cols[0].button("Previous", key=f"prev_{category}", disabled=st.session_state[idx_key] <= 0):
            st.session_state[idx_key] -= 1
            st.rerun()
        if nav_cols[1].button("Next", key=f"next_{category}", disabled=st.session_state[idx_key] >= len(ids) - 1):
            st.session_state[idx_key] += 1
            st.rerun()
        jump = nav_cols[2].number_input(
            "Jump to item #", min_value=1, max_value=len(ids), value=st.session_state[idx_key] + 1,
            key=f"jump_{category}",
        )
        if jump - 1 != st.session_state[idx_key]:
            st.session_state[idx_key] = jump - 1
            st.rerun()
        if nav_cols[3].button("First unreviewed", key=f"firstunrev_{category}"):
            first_unreviewed = next((i for i, iid in enumerate(ids) if iid not in decisions), 0)
            st.session_state[idx_key] = first_unreviewed
            st.rerun()

        current_idx = st.session_state[idx_key]
        item_id = ids[current_idx]
        item = items[item_id]
        blind = blind_item_for_display(item)
        existing = decisions.get(item_id)

        st.write(f"Item **{current_idx + 1} / {len(ids)}** -- id `{item_id}`"
                 + (" (already reviewed)" if existing else " (not yet reviewed)"))
        image_path = ROOT / blind["image_rel_path"]
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.error(f"Image missing on disk: {image_path}")

        default_index = HUMAN_JUDGMENT_CHOICES.index(existing["decision"]) if existing else None
        decision = st.radio(
            "Your decision", options=HUMAN_JUDGMENT_CHOICES,
            format_func=lambda c: HUMAN_JUDGMENT_LABELS[c],
            index=default_index, key=f"decision_{category}_{item_id}",
        )
        notes = st.text_area(
            "Optional notes", value=existing["notes"] if existing else "",
            key=f"notes_{category}_{item_id}",
        )
        save_cols = st.columns([1, 1, 3])
        if save_cols[0].button("Save decision", key=f"save_{category}_{item_id}", type="primary",
                                disabled=decision is None):
            save_decision(DECISIONS_PATH, item_id, decision, notes)
            st.success("Saved.")
            st.rerun()
        if existing and save_cols[1].button("Clear decision", key=f"clear_{category}_{item_id}"):
            save_decision(DECISIONS_PATH, item_id, None)
            st.rerun()

        if existing:
            with st.expander("Show hidden details (only available because you already saved a decision for this item)"):
                st.write({
                    "class_a": item["class_a"], "class_b": item["class_b"],
                    "split_a": item["split_a"], "split_b": item["split_b"],
                    "same_label": item["same_label"],
                    "hamming_ahash": item.get("hamming_ahash"),
                    "hamming_dhash": item.get("hamming_dhash"),
                    "source_category": item.get("source_category"),
                    "ai_preliminary_judgment": item.get("ai_preliminary_judgment"),
                    "ai_notes": item.get("ai_notes"),
                })
