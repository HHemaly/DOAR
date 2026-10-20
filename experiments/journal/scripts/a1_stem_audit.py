"""A1 — filename-level source-stem audit (CPU, no images opened).

Usage (local, committed manifests):
  python experiments/journal/scripts/a1_stem_audit.py \
      --manifest experiments/E1_visual_representation/raw/e1a_train_valid_manifest.csv \
      --manifest experiments/E1_visual_representation/raw/e1a_test_paths_LOCKED_DO_NOT_USE.csv \
      --out experiments/journal/results/A1_stem_audit_T0_included

Usage (Colab, full T0 manifest incl. excluded conflicts):
  python .../a1_stem_audit.py --manifest <Drive>/DOAR_FROZEN_TO/final_partition/partition_manifest.csv --out <...>

Reads only image_id, path/relative_path, class and split columns. The locked
test manifest is read for filenames only: no image is opened, no model output
is produced or inspected.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import load_manifest, stem_audit  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = []
    for m in args.manifest:
        rows += load_manifest(m)
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate image_id across manifests")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = stem_audit(rows)
    groups = audit.pop("group_rows")
    with open(out / "rf_stem_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rf_stem", "size", "splits", "classes", "image_ids"])
        w.writeheader()
        w.writerows(groups)
    meta = provenance.run_metadata("A1_stem_audit", {"manifests": args.manifest},
                                   inputs={f"manifest_{i}": m for i, m in enumerate(args.manifest)})
    provenance.write_json(out / "summary.json", {"audit": audit, "metadata": meta})
    provenance.mark_done(out, meta, [out / "summary.json", out / "rf_stem_groups.csv"])
    import json
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
