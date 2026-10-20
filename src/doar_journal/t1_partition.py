"""T1: source-grouped, leakage-controlled partition (Stage A, X2).

T1 = T0's duplicate groups (sha256 exact + dHash <= 2, reproduced from the T0
partition manifest's own `group_id` column)
   + Roboflow source-stem edges
   + optional image-evidence edges (dihedral dHash, CLIP-embedding near-duplicates),
then the UNCHANGED T0 machinery from `doar.partition`: label-conflict
exclusion, the four previously exposed images forced to train, and the
seeded per-class largest-remainder group assignment with T0's split ratios.

Protocol safeguards
* T0 files are only read. T1 is written to a new directory.
* The partition is `PROPOSED` until `freeze()` is called with an approver
  name; experiment scripts refuse to train on a non-frozen T1.
* Re-drawing the test split is legitimate here only because no model has ever
  been evaluated on the T0 test split (verified in every E1 cell output); this
  is recorded in the freeze record.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from collections import Counter, defaultdict
from pathlib import Path

from doar.partition import (_DEFAULT_SPLIT_RATIOS, _UnionFind, assess_group_label_conflicts,
                            build_group_disjoint_partition, identify_exposed_groups)

from .leakage import rf_stem_edges
from .provenance import sha256_file, write_json

# Same four disclosed smoke-test images T0 forced into train (T0 doc, Section 3).
T0_EXPOSED_IMAGE_IDS = frozenset({"e892f4f710e144bf", "b06dabe271cb4451", "5cc50fc211ecae36",
                                  "07a118a7f414a9d9"})
EVIDENCE_TYPES = ("t0_group", "rf_stem", "dihedral", "embedding", "manual")


def _rows_for_partition(t0_rows: list[dict]) -> list[dict]:
    out = []
    for r in t0_rows:
        out.append({"image_id": r["image_id"], "path": r.get("path", ""),
                    "relative_path": r.get("relative_path", ""),
                    "split": r.get("original_split") or r.get("split", ""),
                    "class": r["class"], "sha256": r.get("sha256", ""), "dhash": r.get("dhash", ""),
                    "phash": r.get("phash", ""), "t0_group_id": r.get("group_id", ""),
                    "t0_split": r.get("new_split", "")})
    return out


def load_edges(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        edges = list(csv.DictReader(f))
    bad = [e for e in edges if e.get("evidence") not in EVIDENCE_TYPES]
    if bad:
        raise ValueError(f"unknown evidence type in {path}: {bad[0]}")
    return edges


_SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}


def _minimal_change_assignment(rows, group_of, conflict_ids, exposed) -> dict:
    """Keep every image in its T0 split unless its T1 group spans splits; a
    spanning group moves as a whole to the split holding most of its members
    (ties: test > valid > train, so evaluation sets do not shrink by default).

    Rationale: T0 and T1 then share most of their validation images, so the
    T0-vs-T1 comparison isolates the effect of removing leaked groups instead
    of mixing it with a completely re-drawn validation set."""
    t0 = {r["image_id"]: r["t0_split"] for r in rows}
    members = defaultdict(list)
    for i, g in group_of.items():
        members[g].append(i)
    assignment = {}
    for g, ids in members.items():
        if g in conflict_ids:
            assignment[g] = "excluded_conflict"
            continue
        if g in exposed:
            assignment[g] = "train"
            continue
        votes = Counter(t0[i] for i in ids if t0[i] in _SPLIT_PRIORITY)
        if not votes:
            raise ValueError(f"group {g} has no T0 supervised split")
        best = max(votes.values())
        assignment[g] = min((s for s, v in votes.items() if v == best), key=_SPLIT_PRIORITY.get)
    return {"assignment": assignment, "algorithm": "minimal_change_majority_vote"}


def build_t1(t0_rows: list[dict], extra_edges: list[dict] | None = None, seed: int = 42,
             split_ratios=_DEFAULT_SPLIT_RATIOS, use_rf_stem: bool = True,
             strategy: str = "minimal_change") -> dict:
    rows = _rows_for_partition(t0_rows)
    ids = [r["image_id"] for r in rows]
    known = set(ids)
    uf = _UnionFind(ids)

    edges: list[dict] = []
    by_t0 = defaultdict(list)
    for r in rows:
        if r["t0_group_id"]:
            by_t0[r["t0_group_id"]].append(r["image_id"])
    for gid, members in sorted(by_t0.items()):
        members = sorted(members)
        edges += [{"a": members[0], "b": m, "evidence": "t0_group", "detail": gid} for m in members[1:]]
    if use_rf_stem:
        edges += rf_stem_edges(rows)
    edges += list(extra_edges or [])
    for e in edges:
        if e["a"] not in known or e["b"] not in known:
            raise ValueError(f"edge refers to unknown image id: {e}")
        uf.union(e["a"], e["b"])

    members_by_root = defaultdict(list)
    for i in ids:
        members_by_root[uf.find(i)].append(i)
    ordered = sorted(members_by_root.values(), key=lambda m: sorted(m)[0])
    group_of = {}
    for idx, members in enumerate(ordered):
        for m in members:
            group_of[m] = f"t1g_{idx:05d}"

    conflicts = assess_group_label_conflicts(rows, group_of)
    conflict_ids = set(conflicts["conflict_group_ids"])
    exposed = identify_exposed_groups(group_of, set(T0_EXPOSED_IMAGE_IDS) & known)
    if strategy == "regroup_full":
        part = build_group_disjoint_partition(rows, group_of, conflict_ids, seed=seed,
                                              split_ratios=split_ratios, exposed_group_ids=exposed)
    elif strategy == "minimal_change":
        part = _minimal_change_assignment(rows, group_of, conflict_ids, exposed)
    else:
        raise ValueError(strategy)
    size = Counter(group_of.values())
    evidence_of = defaultdict(set)
    for e in edges:
        evidence_of[e["a"]].add(e["evidence"])
        evidence_of[e["b"]].add(e["evidence"])

    manifest = []
    for r in sorted(rows, key=lambda r: r["image_id"]):
        gid = group_of[r["image_id"]]
        new_split = part["assignment"][gid]
        manifest.append({
            "image_id": r["image_id"], "path": r["path"], "relative_path": r["relative_path"],
            "class": r["class"], "original_split": r["split"], "t0_split": r["t0_split"],
            "t0_group_id": r["t0_group_id"], "group_id": gid, "group_size": size[gid],
            "group_evidence": ";".join(sorted(evidence_of[r["image_id"]])),
            "conflict_status": "unresolved_label_conflict" if gid in conflict_ids else "clean",
            "exposed_status": "previously_exposed" if gid in exposed else "not_exposed",
            "new_split": new_split,
            "include_in_supervised_training": new_split != "excluded_conflict",
        })
    report = verify_t1(manifest, edges)
    report.update({
        "strategy": strategy, "seed": seed, "split_ratios": list(split_ratios), "n_edges_by_evidence": dict(Counter(e["evidence"] for e in edges)),
        "n_groups": len(ordered), "n_conflict_groups": len(conflict_ids),
        "n_conflict_images": conflicts["n_conflict_images"],
        "split_counts": dict(Counter(m["new_split"] for m in manifest)),
        "split_class_counts": {f"{s}/{c}": n for (s, c), n in
                               sorted(Counter((m["new_split"], m["class"]) for m in manifest).items())},
        "t0_to_t1_transitions": {f"{a}->{b}": n for (a, b), n in
                                 sorted(Counter((m["t0_split"], m["new_split"]) for m in manifest).items())},
    })
    return {"manifest": manifest, "edges": edges, "report": report}


def verify_t1(manifest: list[dict], edges: list[dict]) -> dict:
    split = {m["image_id"]: m["new_split"] for m in manifest}
    cross = Counter()
    for e in edges:
        a, b = split[e["a"]], split[e["b"]]
        if a != b and "excluded_conflict" not in (a, b):
            cross[e["evidence"]] += 1
    supervised_conflict = sum(1 for m in manifest if m["conflict_status"] != "clean"
                              and m["new_split"] in ("train", "valid", "test"))
    exposed_eval = sum(1 for m in manifest if m["exposed_status"] == "previously_exposed"
                       and m["new_split"] in ("valid", "test"))
    groups = defaultdict(set)
    for m in manifest:
        groups[m["group_id"]].add(m["new_split"])
    ok = (not cross and supervised_conflict == 0 and exposed_eval == 0
          and all(len(s) == 1 for s in groups.values()))
    return {"ok": ok, "cross_split_edges_by_evidence": dict(cross),
            "conflict_images_in_supervised_splits": supervised_conflict,
            "exposed_images_in_valid_or_test": exposed_eval}


def write_t1(result: dict, out_dir: str | Path, inputs: dict[str, str]) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FROZEN.json").exists():
        raise RuntimeError(f"{out} holds a FROZEN T1; write proposals to a new directory")
    fields = list(result["manifest"][0])
    with open(out / "t1_partition_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(result["manifest"])
    with open(out / "t1_group_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "evidence", "detail"])
        w.writeheader()
        w.writerows({k: e.get(k, "") for k in ("a", "b", "evidence", "detail")} for e in result["edges"])
    status = {"status": "PROPOSED", "report": result["report"], "inputs": inputs,
              "manifest_sha256": sha256_file(out / "t1_partition_manifest.csv"),
              "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    write_json(out / "T1_STATUS.json", status)
    return status


def freeze(out_dir: str | Path, approved_by: str, note: str = "") -> dict:
    """Record explicit human approval; after this the manifest must not change."""
    out = Path(out_dir)
    status = json.loads((out / "T1_STATUS.json").read_text(encoding="utf-8"))
    if not status["report"]["ok"]:
        raise RuntimeError("T1 failed verification; it cannot be frozen")
    sha = sha256_file(out / "t1_partition_manifest.csv")
    if sha != status["manifest_sha256"]:
        raise RuntimeError("manifest changed since it was proposed")
    record = {"status": "FROZEN", "manifest_sha256": sha, "approved_by": approved_by, "note": note,
              "frozen_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
              "test_integrity_statement": ("No model had been evaluated on the T0 test split when T1 "
                                           "was drawn; T1's test split is evaluated once, after all "
                                           "model-selection and analysis decisions are frozen.")}
    write_json(out / "FROZEN.json", record)
    return record


def require_frozen(out_dir: str | Path) -> str:
    """Return the frozen manifest path or raise (used by every training script)."""
    out = Path(out_dir)
    frozen = out / "FROZEN.json"
    if not frozen.exists():
        raise RuntimeError(f"T1 at {out} is not frozen; get approval and run freeze() first")
    rec = json.loads(frozen.read_text(encoding="utf-8"))
    path = out / "t1_partition_manifest.csv"
    if sha256_file(path) != rec["manifest_sha256"]:
        raise RuntimeError("frozen T1 manifest hash mismatch")
    return str(path)
