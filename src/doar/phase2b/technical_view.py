"""Technical View display of Phase 2B's object-evidence pilot results.

This is a standalone research-artifact display, not a per-case pipeline
integration -- Phase 2B evidence is not computed as part of `analyze_image`
for any real case in this pilot (see docs/PHASE2B_OBJECT_EVIDENCE_POLICY.md).
It shows the real, static pilot results from `artifacts/phase2b/`, always
with the same heavy, un-skippable caveats, so a viewer can never mistake a
20-image feasibility pilot for a validated per-case capability.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def render_phase2b_pilot_summary(st, root: Path) -> None:
    """Renders a Technical-View subsection summarizing the Phase 2B
    object-evidence pilot. `st` is the streamlit module (passed in, not
    imported here, so this module has no hard streamlit dependency and can
    be unit-tested without a running app)."""
    st.subheader("Phase 2B object-evidence pilot (experimental, not a working capability)")
    st.caption(
        "Zero-shot / classical-CV feasibility pilot only -- evaluated on a "
        "20-image sample, not wired into analyze_image for any real case, "
        "and no rule in rules_registry_v2.json is activated by any result "
        "shown here. See PHASE2B_AUDIT_AND_PLAN.md and "
        "docs/PHASE2B_OBJECT_EVIDENCE_POLICY.md.")

    metrics_path = root / "artifacts" / "phase2b" / "per_class_metrics.csv"
    rows = _load_csv(metrics_path)
    if not rows:
        st.write("No Phase 2B pilot results found under artifacts/phase2b/ "
                 "(expected if this checkout has not run scripts/phase2b_evaluate.py).")
        return

    st.write(f"{len(rows)} candidate object classes evaluated against real, "
             "hand-annotated ground truth on a 20-image blind sample.")
    sufficient = [r["class"] for r in rows if r["sufficient_support"] == "True"]
    insufficient = [r["class"] for r in rows if r["sufficient_support"] != "True"]
    st.write(f"Classes with enough positive examples (>=5) for even a preliminary "
             f"reliability read: {', '.join(sufficient) or 'none'}.")
    st.write(f"Classes with insufficient support at this sample size (result not "
             f"reportable): {', '.join(insufficient)}.")
    st.dataframe(rows, width="stretch")
    st.caption(
        "A missing or 'not_detected' result must never be read as 'the object is "
        "absent' -- it means only that this pilot's baseline did not detect it. "
        "No human-review step has been performed on any of these results "
        "(human_review_status is always 'not_reviewed').")
