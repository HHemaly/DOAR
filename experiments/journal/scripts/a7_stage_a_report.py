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


def load_manifest_rows(t1: Path) -> list[dict]:
    p = t1 / "t1_partition_manifest.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def largest_groups(t1: Path, k: int = 5) -> str:
    rows = load_manifest_rows(t1)
    by = {}
    for m in rows:
        g = by.setdefault(m["group_id"], {"n": 0, "split": m["new_split"], "classes": set(), "evidence": set()})
        g["n"] += 1
        g["classes"].add(m["class"])
        g["evidence"].update(x for x in m.get("group_evidence", "").split(";") if x)
    top = sorted(by.items(), key=lambda kv: -kv[1]["n"])[:k]
    return "; ".join(f"{gid}: {g['n']} images, {g['split']}, class {'/'.join(sorted(g['classes']))}, "
                     f"evidence {'+'.join(sorted(g['evidence'])) or '-'}" for gid, g in top) or "n/a"


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

    acc = load(root / "A5_pair_review" / "ACCEPTED_EDGES.json")
    L += ["", "## A5b Evidence accepted for T1 (applied by rule, not by eye)"]
    if acc:
        for kind, d in acc["decisions"].items():
            if "used_by_T1" in d:
                state = {True: "USED, confirmed by review", False: "USED, NOT confirmed - discuss before approving",
                         None: "USED, review not scored"}[d.get("confirmed")]
            else:
                state = "ACCEPTED" if d.get("accepted") else "not used"
            L.append(f"- {kind}: {state} - {d['reason']}")
        L.append(f"- Edge files given to A3: {[Path(x).name for x in acc['accepted_edge_files']] or 'none (filename stems + T0 groups only)'}")
    else:
        L.append("- **OPEN**: run the A5b cell after scoring the review")

    t1 = Path(a.t1) if a.t1 else root / "T1_proposal_regroup_full"
    st = load(t1 / "T1_STATUS.json")
    L += ["", f"## Proposed T1 split ({t1.name})"]
    if st:
        r = st["report"]
        b = r.get("balance", {})
        crit = b.get("criteria", {})
        L += [f"- Status: {st['status']}; strategy {r.get('strategy')}; seed {r.get('seed')}; "
              f"manifest SHA-256 {st.get('manifest_sha256', '')[:12]}...",
              f"- Evidence edges used: {r['n_edges_by_evidence']}",
              f"- Edge files used: {[Path(x).name for x in st.get('inputs', {}).get('extra_edges', {})] or 'none'}",
              "", "### Excluded records",
              f"- Images considered: {len(load_manifest_rows(t1))}; duplicate/source groups: {r.get('n_groups')}",
              f"- Excluded for label conflict (copies of one drawing with different labels): "
              f"{r.get('n_conflict_images')} images in {r.get('n_conflict_groups')} groups",
              f"- Final split counts (incl. excluded): {r['split_counts']}",
              f"- Previously exposed images placed in valid/test: {r.get('exposed_images_in_valid_or_test')} (must be 0)",
              f"- Conflict images in supervised splits: {r.get('conflict_images_in_supervised_splits')} (must be 0)",
              f"- Detected duplicate edges crossing splits: {r['cross_split_edges_by_evidence'] or 'none'} (must be none)",
              "", "### Group sizes",
              f"- Groups per split: {b.get('groups_per_split')}",
              f"- Group-size distribution (size: count): {b.get('group_size_distribution')}",
              f"- Largest groups: {largest_groups(t1)}",
              "", "### Class balance (share of each class within each split vs overall)"]
        cs, ov = b.get("class_share_by_split", {}), b.get("overall_class_share", {})
        if ov:
            L.append("| split | " + " | ".join(ov) + " |")
            L.append("|---|" + "---|" * len(ov))
            for sp, d in cs.items():
                L.append(f"| {sp} | " + " | ".join(f"{d[c]:.3f}" for c in ov) + " |")
            L.append("| overall | " + " | ".join(f"{v:.3f}" for v in ov.values()) + " |")
        L += ["", "### Source families by split (filename-based)"]
        sf = b.get("source_family_by_split", {})
        fams = sorted({f for d in sf.values() for f in d})
        if fams:
            L.append("| split | " + " | ".join(fams) + " |")
            L.append("|---|" + "---|" * len(fams))
            for sp, d in sf.items():
                L.append(f"| {sp} | " + " | ".join(str(d.get(f, 0)) for f in fams) + " |")
        frac, tgt = b.get("split_fraction", {}), b.get("target_fraction", {})
        n_sup = b.get("n_supervised") or 1
        values = {
            "split_fractions": (max((abs(frac[s] - tgt[s]) for s in frac), default=float("nan")),
                                crit.get("split_fraction_abs_dev_max"), "largest split-size deviation from target"),
            "class_shares": (b.get("worst_class_share_deviation", float("nan")),
                             crit.get("class_share_abs_dev_max"), "worst class-share deviation"),
            "largest_group": (b.get("largest_group_size", 0) / n_sup, crit.get("largest_group_share_max"),
                              "largest group / supervised images"),
            "source_family_association": (b.get("source_family_cramers_v", float("nan")),
                                          crit.get("source_family_cramers_v_max"), "Cramer's V split x source family"),
        }
        L += ["", "### Balance checks (pre-registered limits)", "| check | value | limit | result |", "|---|---|---|---|"]
        for k, ok in b.get("checks", {}).items():
            v, lim, what = values[k]
            L.append(f"| {what} | {v:.3f} | {lim} | {'PASS' if ok else 'FAIL'} |")
        reasons = [f"{values[k][2]} = {values[k][0]:.3f} exceeds the limit {values[k][1]}"
                   for k, ok in b.get("checks", {}).items() if not ok]
        if not r.get("ok"):
            reasons.append("verification failed: a detected duplicate group crosses splits, or conflict/exposed "
                           "images are in a supervised or evaluation split")
        L += ["", f"**Verdict: {'ACCEPTABLE - all checks pass' if not reasons else 'REJECTED as proposed'}**"]
        L += [f"- Reason: {x}" for x in reasons]
        if reasons:
            L.append("- Do NOT redraw with another seed. Send this report; a written deviation or a joint decision is needed.")
        L.append("- Note: zero cross-split edges means zero DETECTED duplicates, not zero duplicates.")
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
