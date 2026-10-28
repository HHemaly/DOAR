"""A5c — what each duplicate-evidence type would ADD to a proposed T1 split (CPU, descriptive).

  python experiments/journal/scripts/a5c_edge_novelty.py \
      --manifest <T0 partition_manifest.csv> --root <Drive>/DOAR_JOURNAL_2026 \
      [--t1 <root>/T1_proposal_regroup_full]

Counts only; it decides nothing and uses no model output. For every evidence
type (plain-filename `name_key`, rotation/flip dHash `dihedral`, CLIP
`embedding`) it reports, against the proposed T1 groups:

  edges_total            candidate pairs
  edges_redundant        both images already in the same T1 group (adds nothing)
  edges_new              link two different T1 groups
  new_cross_t1_split     ... whose groups currently sit in different T1 splits
                         (= leakage the current proposal would contain IF the edge is real)
  new_touching_test      ... with one end in T1 test
  new_different_class    ... linking images with different labels (would create or
                         extend label-conflict exclusions)
  new_touching_excluded  ... with one end already excluded for label conflict
  groups_merged / largest_component_after / images_that_would_be_newly_excluded
                         effect of adding ALL new edges of this type at once (a merged
                         group holding more than one label is excluded as a conflict)
  reviewable_pool        new edges between T0 train and T0 valid (the population the
                         blinded A5 review samples from; the test split is never shown)

Writes A5c_edge_novelty/summary.json and new_edges_<type>.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import load_manifest, path_of, source_key, split_of  # noqa: E402


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def name_key_edges(rows: list[dict], cross_t0_split_only: bool) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        k = source_key(path_of(r))
        if not k.is_roboflow:
            by[k.name_key].append(r)
    edges = []
    for key, members in sorted(by.items()):
        if len(members) < 2:
            continue
        if cross_t0_split_only and len({split_of(m) for m in members}) < 2:
            continue
        ids = sorted(m["image_id"] for m in members)
        edges += [{"a": ids[0], "b": o, "evidence": "manual", "detail": f"name_key={key}"} for o in ids[1:]]
    return edges


def read_edges(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyse(edges: list[dict], t1: dict[str, dict], t0_split: dict[str, str]) -> tuple[dict, list[dict]]:
    grp = {i: m["group_id"] for i, m in t1.items()}
    sp = {i: m["new_split"] for i, m in t1.items()}
    cls = {i: m["class"] for i, m in t1.items()}
    new = [e for e in edges if grp[e["a"]] != grp[e["b"]]]
    c = Counter()
    pairs = Counter()
    for e in new:
        a, b = e["a"], e["b"]
        if sp[a] != sp[b]:
            c["new_cross_t1_split"] += 1
            pairs["-".join(sorted((sp[a], sp[b])))] += 1
        c["new_touching_test"] += "test" in (sp[a], sp[b])
        c["new_different_class"] += cls[a] != cls[b]
        c["new_touching_excluded"] += "excluded_conflict" in (sp[a], sp[b])
        c["reviewable_pool"] += {t0_split[a], t0_split[b]} == {"train", "valid"}
    uf = UF()
    for i in t1:
        uf.find(grp[i])
    for e in new:
        uf.union(grp[e["a"]], grp[e["b"]])
    comp = defaultdict(list)
    for i in t1:
        comp[uf.find(grp[i])].append(i)
    merged = [m for m in comp.values() if len({grp[i] for i in m}) > 1]
    conflict_imgs = sum(len(m) for m in merged if len({cls[i] for i in m}) > 1
                        and not all(sp[i] == "excluded_conflict" for i in m))
    newly_excluded = sum(1 for m in merged if len({cls[i] for i in m}) > 1
                         for i in m if sp[i] != "excluded_conflict")
    span = sum(len(m) for m in merged if len({sp[i] for i in m if sp[i] != "excluded_conflict"}) > 1)
    out = {"edges_total": len(edges), "edges_redundant": len(edges) - len(new), "edges_new": len(new),
           **{k: c[k] for k in ("new_cross_t1_split", "new_touching_test", "new_different_class",
                                "new_touching_excluded", "reviewable_pool")},
           "new_cross_t1_split_by_pair": dict(pairs),
           "groups_merged_into": len(merged), "t1_groups_involved": sum(len({grp[i] for i in m}) for m in merged),
           "largest_component_after": max((len(m) for m in comp.values()), default=0),
           "images_in_components_spanning_t1_splits": span,
           "images_in_new_conflict_components": conflict_imgs,
           "images_that_would_be_newly_excluded": newly_excluded}
    detail = [{**{k: e[k] for k in ("a", "b", "detail")}, "group_a": grp[e["a"]], "group_b": grp[e["b"]],
               "t1_split_a": sp[e["a"]], "t1_split_b": sp[e["b"]], "class_a": cls[e["a"]], "class_b": cls[e["b"]]}
              for e in new]
    return out, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--t1", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    t1_dir = Path(a.t1) if a.t1 else root / "T1_proposal_regroup_full"
    with open(t1_dir / "t1_partition_manifest.csv", newline="", encoding="utf-8") as f:
        t1 = {m["image_id"]: m for m in csv.DictReader(f)}
    rows = load_manifest(a.manifest)
    t0_split = {r["image_id"]: split_of(r) for r in rows}
    sources = {"name_key (as A5b would add: names recurring across T0 splits)": name_key_edges(rows, True),
               "name_key (all recurring plain names)": name_key_edges(rows, False)}
    for kind in ("dihedral", "embedding"):
        p = root / "A4_image_evidence" / f"{kind}_edges_candidates.csv"
        if p.exists():
            sources[kind] = read_edges(p)
        for tp in sorted((root / "A5_pair_review").glob(f"{kind}_tight_edges_candidates.csv")):
            sources[f"{kind} (tighter threshold)"] = read_edges(tp)
    out_dir = root / "A5c_edge_novelty"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, edges in sources.items():
        s, detail = analyse(edges, t1, t0_split)
        summary[name] = s
        slug = name.split(" (")[0] + ("_all" if "all recurring" in name else "_tight" if "tighter" in name else "")
        with open(out_dir / f"new_edges_{slug}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["a", "b", "detail", "group_a", "group_b", "t1_split_a",
                                              "t1_split_b", "class_a", "class_b"])
            w.writeheader()
            w.writerows(detail)
    provenance.write_json(out_dir / "summary.json", {
        "t1_manifest_sha256": provenance.sha256_file(t1_dir / "t1_partition_manifest.csv"), "by_evidence": summary,
        "note": "Counts of CANDIDATE relationships; whether they are real duplicates is decided by the blinded "
                "review and the pre-registered rule (A5b), not by these numbers."})
    for name, s in summary.items():
        print(f"\n{name}")
        for k, v in s.items():
            print(f"  {k:42s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
