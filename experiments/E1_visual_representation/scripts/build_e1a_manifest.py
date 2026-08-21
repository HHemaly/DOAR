#!/usr/bin/env python
"""E1-A manifest adapter -- reads the FROZEN T0 partition manifest verbatim
and writes a minimal, E1-scoped CSV (image_id, path, class, split,
readable) used by every E1-A script. Never modifies the T0 manifest;
never changes any image's split membership. Excludes
excluded_conflict/test rows from the working CSV (test rows are kept in a
SEPARATE, clearly-labeled file, path-verified only, never label-inspected
beyond what's already public in the frozen manifest, and never used by
any training/selection code this phase)."""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T0_MANIFEST = ROOT / "outputs" / "t0_automated" / "final_partition" / "partition_manifest.csv"
EXPECTED_SHA256 = "4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw"


def main():
    actual_sha = hashlib.sha256(T0_MANIFEST.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_SHA256, (
        f"T0 manifest SHA-256 mismatch! expected {EXPECTED_SHA256}, got {actual_sha} -- "
        f"refusing to build on a manifest that does not match the frozen checkpoint.")

    with open(T0_MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    train_valid_rows = []
    test_rows = []
    for r in rows:
        exists = os.path.exists(r["path"])
        out_row = {
            "image_id": r["image_id"], "path": r["path"], "class": r["class"],
            "split": r["new_split"], "readable": "true" if exists else "false",
        }
        if r["new_split"] in ("train", "valid"):
            train_valid_rows.append(out_row)
        elif r["new_split"] == "test":
            test_rows.append(out_row)
        # excluded_conflict rows are intentionally dropped from both files --
        # never used by any E1 code.

    n_unreadable = sum(1 for r in train_valid_rows if r["readable"] == "false")
    assert n_unreadable == 0, f"{n_unreadable} train/valid images do not resolve on this machine -- STOP."

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "e1a_train_valid_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "path", "class", "split", "readable"])
        w.writeheader()
        w.writerows(train_valid_rows)

    # Test: path-existence recorded only, for the readiness check -- no
    # training/selection code in E1-A ever reads this file.
    with open(OUT_DIR / "e1a_test_paths_LOCKED_DO_NOT_USE.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "path", "class", "split", "readable"])
        w.writeheader()
        w.writerows(test_rows)

    n_train = sum(1 for r in train_valid_rows if r["split"] == "train")
    n_valid = sum(1 for r in train_valid_rows if r["split"] == "valid")
    n_test_resolved = sum(1 for r in test_rows if r["readable"] == "true")
    print(f"train={n_train} valid={n_valid} test={len(test_rows)} (resolved={n_test_resolved})")
    print(f"T0 manifest SHA-256 verified: {actual_sha}")
    assert n_train == 2599 and n_valid == 284 and len(test_rows) == 512, "Split sizes do not match the frozen T0 spec!"
    print("OK -- matches frozen T0 spec exactly (train=2599, valid=284, test=512).")


if __name__ == "__main__":
    main()
