"""A7 — Stage A readiness report (CPU). Collects every Stage A output into one
Markdown report so you can see, in one place, what was verified and what is
still open before the T1 split is approved.

  python experiments/journal/scripts/a7_stage_a_report.py --root <Drive>/DOAR_JOURNAL_2026

Missing or incomplete steps are listed as such; nothing is inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt_ci(d: dict) -> str:
    return f"{d['estimate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--t1", default=None, help="proposed T1 folder (default: <root>/T1_proposal_regroup_full)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    L = ["# Stage A readiness report", "",
         "Generated from the files under `DOAR_JOURNAL_2026/`. Missing steps are marked **OPEN**.", ""]

    s = load(root / "A1_stem_audit_T0_full" / "summary.json")
    L.append("## A1 Filename-level source audit (full T0 manifest)")
    if s:
        au = s["audit"]
        L += [f"- Images in manifest: {au['n_images']}; Roboflow-named: {au['n_roboflow_named']}; stems: {au['n_rf_stems']}",
              f"- Stem group sizes: {au['rf_group_size_distribution']}",
              f"- Stems spanning splits: {au['rf_stems_spanning_splits']} (total {au['rf_stems_spanning_splits_total']})"]
        for sp, e in au["exposure_by_split"].items():
            if e["fraction"] is not None:
                L.append(f"- {sp}: {e['share_rf_stem_with_train']} of {e['n']} share a stem with train "
                         f"({100 * e['fraction']:.1f}%); by class {e['by_class']}")
        L.append(f"- Label conflicts inside stem groups: {au['rf_stem_groups_with_class_conflict']}")
    else:
        L.append("- **OPEN**: run A1")

    L += ["", "## A2 Saved validation predictions: shared-source vs independent images (exploratory)"]
    a2 = root / "A2_rescoring_groupboot" / "rescoring_by_run.csv"
    if a2.exists():
        with open(a2, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        diffs = [float(r["diff"]) for r in rows]
        pos = sum(d > 0 for d in diffs)
        excl0 = sum(float(r["diff_ci_low"]) > 0 for r in rows)
        L += [f"- Runs re-scored: {len(rows)}; shared minus independent macro-F1 > 0 in {pos} runs; "
              f"group-bootstrap CI above 0 in {excl0} runs",
              f"- Median difference: {sorted(diffs)[len(diffs) // 2]:+.3f}",
              "- Association only: the two subsets can also differ in class mix, source family and difficulty."]
    else:
        L.append("- **OPEN**: run A2 (group bootstrap version)")

    s6 = load(root / "A6_dev_holdout_overlap" / "summary.json")
    L += ["", "## A6 Foreground holdout vs development images"]
    L.append(f"- {s6['summary']}" if s6 else "- **OPEN**: run A6")

    L += ["", "## A4 Image-level duplicate candidates"]
    a4 = load(root / "A4_image_evidence" / "DONE.json")
    if a4:
        for name in ("dihedral_edges_candidates.csv", "embedding_edges_candidates.csv"):
            p = root / "A4_image_evidence" / name
            if p.exists():
                with open(p, newline="", encoding="utf-8") as f:
                    e = list(csv.DictReader(f))
                L.append(f"- {name}: {len(e)} candidate pairs, {sum(x['cross_split'] == 'True' for x in e)} across T0 splits")
        L.append(f"- CLIP model: {a4.get('config', {}).get('clip_model', 'NOT RECORDED (old run - rerun A4)')}")
    else:
        L.append("- **OPEN**: run A4 to completion")

    L += ["", "## A5 Blinded pair review"]
    sc = load(root / "A5_pair_review" / "review_scores.json")
    if sc:
        for stratum, res in sc.items():
            for kind, v in res.items():
                r = v["rate_same"]
                L.append(f"- {stratum} / {kind}: judged same {v['judged_same']} of {v['n']} "
                         f"({100 * r['estimate']:.0f}%, 95% CI {100 * r['ci_low']:.0f}-{100 * r['ci_high']:.0f}%); "
                         f"unsure {v['unsure']}; transforms {v['transforms']}")
        L.append("- Decision rule (PREREGISTRATION C1/C1b): accept an evidence type for T1 if >= 90% of its "
                 "candidates are judged the same drawing; stems confirmed if lower CI bound >= 80% and controls <= 10%.")
    else:
        L.append("- **OPEN**: do the review in the notebook, then run the score cell")

    t1 = Path(a.t1) if a.t1 else root / "T1_proposal_regroup_full"
    st = load(t1 / "T1_STATUS.json")
    L += ["", f"## Proposed T1 split ({t1.name})"]
    if st:
        r = st["report"]
        b = r.get("balance", {})
        L += [f"- Status: {st['status']}; strategy {r.get('strategy')}; seed {r.get('seed')}",
              f"- Evidence edges used: {r['n_edges_by_evidence']}",
              f"- Cross-split detected edges: {r['cross_split_edges_by_evidence'] or 'none'}",
              f"- Split sizes: {r['split_counts']}; groups per split: {b.get('groups_per_split')}",
              f"- Largest group: {b.get('largest_group_size')} images; group sizes {b.get('group_size_distribution')}",
              f"- Worst class-share deviation: {b.get('worst_class_share_deviation', float('nan')):.3f} (limit 0.03)",
              f"- Source family vs split, Cramer's V: {b.get('source_family_cramers_v', float('nan')):.3f} (limit 0.10)",
              f"- Balance checks: {b.get('checks')} -> {'ACCEPTABLE' if b.get('acceptable') else 'NOT acceptable'}",
              "- Note: zero cross-split edges means zero DETECTED duplicates, not zero duplicates."]
    else:
        L.append("- **OPEN**: run A3 propose")
    frozen = load(root / "T1" / "FROZEN.json")
    L += ["", "## Approval", f"- T1 frozen: {'yes, by ' + frozen['approved_by'] if frozen else 'NO (needs your approval)'}"]

    out = root / "STAGE_A_READINESS.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
