"""A3 / X2 — build (propose) or freeze the source-grouped T1 partition (CPU).

Propose (Colab, after A1/A2 and the image-evidence scans):
  python experiments/journal/scripts/a3_build_t1.py propose \
      --t0-manifest /content/drive/MyDrive/Masters/Datasets/DOAR_FROZEN_TO/final_partition/partition_manifest.csv \
      --extra-edges <...>/A4_image_evidence/dihedral_edges_confirmed.csv \
      --extra-edges <...>/A4_image_evidence/embedding_edges_confirmed.csv \
      --out /content/drive/MyDrive/Masters/DOAR_JOURNAL_2026/T1

Freeze (only after the author has reviewed T1_STATUS.json and approves):
  python experiments/journal/scripts/a3_build_t1.py freeze --out <...>/T1 --approved-by "Hoda Elhemaly"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal.leakage import load_manifest  # noqa: E402
from doar_journal.provenance import sha256_file  # noqa: E402
from doar_journal.t1_partition import build_t1, freeze, load_edges, write_t1  # noqa: E402

T0_SHA256 = "4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--t0-manifest", required=True)
    p.add_argument("--extra-edges", action="append", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategy", choices=("minimal_change", "regroup_full"), default="minimal_change")
    p.add_argument("--allow-non-t0", action="store_true",
                   help="preview on a manifest that is not the frozen T0 file (never for training)")
    f = sub.add_parser("freeze")
    f.add_argument("--out", required=True)
    f.add_argument("--approved-by", required=True)
    f.add_argument("--note", default="")
    args = ap.parse_args(argv)

    if args.cmd == "freeze":
        print(json.dumps(freeze(args.out, args.approved_by, args.note), indent=2))
        return 0

    sha = sha256_file(args.t0_manifest)
    if sha != T0_SHA256 and not args.allow_non_t0:
        raise SystemExit(f"T0 manifest hash {sha} != frozen T0 {T0_SHA256}")
    edges = []
    for e in args.extra_edges:
        edges += load_edges(e)
    result = build_t1(load_manifest(args.t0_manifest), extra_edges=edges, seed=args.seed,
                      strategy=args.strategy)
    inputs = {"t0_manifest": args.t0_manifest, "t0_sha256": sha,
              "extra_edges": {e: sha256_file(e) for e in args.extra_edges}}
    status = write_t1(result, args.out, inputs)
    if sha != T0_SHA256:
        status["status"] = "PREVIEW_NOT_T0"
        (Path(args.out) / "T1_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status["report"], indent=2))
    return 0 if status["report"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
