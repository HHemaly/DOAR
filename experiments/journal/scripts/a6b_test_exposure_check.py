"""A6b — was any T0 TEST image ever used by an earlier experiment? (CPU, read-only)

  python experiments/journal/scripts/a6b_test_exposure_check.py \
      --manifest <T0 partition_manifest.csv> --t1 <root>/T1_proposal_regroup_full/t1_partition_manifest.csv \
      --scan <Drive>/DOAR_GPU_RESULTS --out <Drive>/DOAR_JOURNAL_2026/A6b_test_exposure

Scans every text table (.csv .json .jsonl .txt .tsv .md, < 200 MB) under --scan
for the image ids and file names of the T0 test split, and reports which files
mention them. It opens no image and computes no metric. Two directions matter:

  * T0 test -> T1 train/valid: harmless for T1 (training images may have been
    seen before), but a hit in a prediction file would mean the T0 test lock
    was broken in the past, which must be disclosed.
  * T0 train/valid -> T1 test: these WERE used to train/select T0 models. T1
    models are retrained from public pretrained weights, so no T0 weights see
    them; the report lists the counts so the paper can state it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import load_manifest, path_of, split_of  # noqa: E402

TEXT = {".csv", ".json", ".jsonl", ".txt", ".tsv", ".md", ".log"}
SKIP_DIRS = {"embeddings", "figures", "qualitative", "panels", "contact_sheets", "masks", "images", "checkpoints"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--t1", required=True)
    ap.add_argument("--scan", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import os
    rows = load_manifest(a.manifest)
    test = [r for r in rows if split_of(r) == "test"]
    ids = {r["image_id"] for r in test}
    names = {path_of(r).replace("\\", "/").split("/")[-1] for r in test}
    # file names that also exist in another split are ambiguous evidence; count them separately
    other_names = {path_of(r).replace("\\", "/").split("/")[-1] for r in rows if split_of(r) != "test"}
    unique_names = names - other_names
    id_re = re.compile(r"\b[0-9a-f]{16}\b")
    hits = []
    n_files = 0
    for scan in a.scan:
        for dirpath, dirnames, filenames in os.walk(scan):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() not in TEXT or p.stat().st_size > 200e6:
                    continue
                n_files += 1
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                id_hits = ids & set(id_re.findall(txt))
                name_hits = {n for n in unique_names if n in txt}
                if id_hits or name_hits:
                    hits.append({"file": str(p), "t0_test_ids": len(id_hits), "t0_test_unique_filenames": len(name_hits),
                                 "looks_like_predictions": "pred" in fn.lower() or "predict" in txt[:2000].lower(),
                                 "example": sorted(id_hits or name_hits)[:3]})
    with open(a.t1, newline="", encoding="utf-8") as f:
        t1 = list(csv.DictReader(f))
    moves = Counter((m["t0_split"], m["new_split"]) for m in t1)
    summary = {"t0_test_images": len(ids), "text_files_scanned": n_files, "files_mentioning_t0_test": len(hits),
               "prediction_like_files_mentioning_t0_test": sum(h["looks_like_predictions"] for h in hits),
               "t0_test_to_t1": {k[1]: v for k, v in moves.items() if k[0] == "test"},
               "t0_trainvalid_to_t1_test": sum(v for k, v in moves.items() if k[0] in ("train", "valid") and k[1] == "test"),
               "t0_trainvalid_to_t1_valid": sum(v for k, v in moves.items() if k[0] in ("train", "valid") and k[1] == "valid"),
               "verdict": ("NO earlier result file mentions a T0 test image" if not hits else
                           "SOME files mention T0 test images - open hits.csv and check what they are before freezing")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "hits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "t0_test_ids", "t0_test_unique_filenames", "looks_like_predictions", "example"])
        w.writeheader()
        w.writerows(hits)
    provenance.write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    for h in hits[:30]:
        print(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
