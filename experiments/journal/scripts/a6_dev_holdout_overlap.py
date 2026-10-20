"""A6 — do the 80 "unseen" foreground-holdout images share a source with the 84
development images? (CPU; reads two Drive CSVs and the T0 manifest.)

  python experiments/journal/scripts/a6_dev_holdout_overlap.py \
      --manifest <T0 partition_manifest.csv> \
      --dev <Drive>/DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES/E2A0_R1_PILOT_V1/tables/E2A0_R1_PILOT_MASK_RESULTS.csv \
      --holdout <Drive>/DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES/E2A0_R1_V2_UNSEEN_HOLDOUT/tables/E2A0_R1_V2_UNSEEN_HOLDOUT_MANIFEST.csv \
      --out <Drive>/DOAR_JOURNAL_2026/A6_dev_holdout_overlap

Holdout images flagged here are reported separately in the paper; they are
not "unseen" with respect to the frozen V2 development.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import load_manifest, stem_overlap  # noqa: E402


def ids_from(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "image_id" not in rows[0]:
        raise SystemExit(f"{path} has no image_id column; columns: {list(rows[0]) if rows else []}")
    return sorted({r["image_id"] for r in rows})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = load_manifest(args.manifest)
    dev, hold = ids_from(args.dev), ids_from(args.holdout)
    direct = sorted(set(dev) & set(hold))
    overlap = stem_overlap(dev, hold, rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "holdout_source_overlap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "overlap", "key"])
        w.writeheader()
        w.writerows({"key": "", **o} for o in overlap)
    summary = {"n_dev": len(dev), "n_holdout": len(hold), "identical_image_ids": len(direct),
               "holdout_sharing_rf_stem_with_dev": sum(o["overlap"] == "rf_stem" for o in overlap),
               "holdout_sharing_name_key_with_dev": sum(o["overlap"] == "name_key" for o in overlap),
               "holdout_not_in_manifest": sum(o["overlap"] == "not_in_manifest" for o in overlap)}
    meta = provenance.run_metadata("A6_dev_holdout_overlap", vars(args),
                                   inputs={"manifest": args.manifest, "dev": args.dev, "holdout": args.holdout})
    provenance.write_json(out / "summary.json", {"summary": summary, "metadata": meta})
    provenance.mark_done(out, meta, [out / "summary.json", out / "holdout_source_overlap.csv"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
