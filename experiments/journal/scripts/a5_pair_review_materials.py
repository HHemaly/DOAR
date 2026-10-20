"""A5 / X0 — blinded pair-review materials, and scoring once the review is done.

Make materials (Colab CPU):
  python experiments/journal/scripts/a5_pair_review_materials.py make \
      --manifest <T0 partition_manifest.csv> --dataset-root <Combined_Drawing> \
      --out <Drive>/DOAR_JOURNAL_2026/A5_pair_review

This writes, per evidence stratum (rf_stem, name_key, and image-evidence
candidates if --edges is given):
  pairs_<stratum>.csv            key with the hidden pair type (do NOT open before reviewing)
  review_sheet_<stratum>.pdf     blinded side-by-side pairs, labelled only P001...
  review_<stratum>.csv           blank form: pair_id, verdict, transform_seen, notes, reviewer

Only train-valid pairs are used for the main check; the locked test split is
never shown. Score after review:
  python .../a5_pair_review_materials.py score --out <same dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import (load_manifest, make_pair_sheets, path_of,  # noqa: E402
                                  sample_review_pairs, score_review, split_of, write_review_template)


def _resolve_rel(row: dict) -> str:
    rel = row.get("relative_path")
    if rel:
        return rel
    parts = row["path"].replace("\\", "/").split("/")
    return "/".join(parts[-3:])


def edge_pairs(edges_csv: str, rows: list[dict], n: int, seed: int) -> list[dict]:
    by = {r["image_id"]: r for r in rows}
    with open(edges_csv, newline="", encoding="utf-8") as f:
        edges = [e for e in csv.DictReader(f)
                 if {split_of(by[e["a"]]), split_of(by[e["b"]])} <= {"train", "valid"}]
    random.Random(seed).shuffle(edges)
    out = []
    for i, e in enumerate(edges[:n], start=1):
        a, b = by[e["a"]], by[e["b"]]
        out.append({"pair_id": f"P{i:03d}", "pair_type_hidden": f"candidate_{e['evidence']}",
                    "image_id_a": a["image_id"], "path_a": _resolve_rel(a), "split_a": split_of(a),
                    "image_id_b": b["image_id"], "path_b": _resolve_rel(b), "split_b": split_of(b),
                    "class": a["class"], "detail": e.get("detail", "")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--manifest", required=True)
    m.add_argument("--dataset-root", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--edges", action="append", default=[], help="A4 candidate edge CSVs")
    m.add_argument("--n-pairs", type=int, default=40)
    m.add_argument("--n-controls", type=int, default=20)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.cmd == "score":
        results = {}
        for key in sorted(out.glob("pairs_*.csv")):
            stratum = key.stem.removeprefix("pairs_")
            with open(key, newline="", encoding="utf-8") as f:
                pairs = list(csv.DictReader(f))
            results[stratum] = score_review(pairs, out / f"review_{stratum}.csv")
        provenance.write_json(out / "review_scores.json", results)
        print(json.dumps(results, indent=2))
        return 0

    rows = load_manifest(args.manifest)
    for r in rows:
        r["relative_path"] = _resolve_rel(r)
    strata = {}
    for ev in ("rf_stem", "name_key"):
        strata[ev] = sample_review_pairs(rows, n_pairs=args.n_pairs, n_controls=args.n_controls, evidence=ev)
    for e in args.edges:
        name = Path(e).stem.replace("_edges_candidates", "")
        strata[name] = edge_pairs(e, rows, args.n_pairs, seed=20261022)
    out.mkdir(parents=True, exist_ok=True)
    for stratum, pairs in strata.items():
        if not pairs:
            print(f"{stratum}: no pairs available")
            continue
        with open(out / f"pairs_{stratum}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0]))
            w.writeheader()
            w.writerows(pairs)
        write_review_template(pairs, out / f"review_{stratum}.csv")
        make_pair_sheets(pairs, args.dataset_root, out / f"review_sheet_{stratum}.pdf")
        print(f"{stratum}: {len(pairs)} pairs -> review_sheet_{stratum}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
