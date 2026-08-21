#!/usr/bin/env python
"""Portable T0 dataset verification -- NEVER edits the frozen T0 manifest.

Resolves every Train/Validation image as `dataset_root / relative_path`
(never the T0 manifest's own absolute `path` column, which is
machine-specific). Verifies the manifest SHA-256, split counts, and that
every Train/Validation file resolves on THIS machine. Test-set rows are
counted only -- never label-inspected, never opened as images, never used
by any downstream E1 code path.

Usage:
    python prepare_e1_dataset.py --dataset-root <path> --verify-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T0_MANIFEST = ROOT / "outputs" / "t0_automated" / "final_partition" / "partition_manifest.csv"
EXPECTED_SHA256 = "4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d"
EXPECTED_COUNTS = {"train": 2599, "valid": 284, "test": 512}


def load_t0_rows() -> list[dict]:
    with open(T0_MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_manifest_sha256() -> str:
    actual = hashlib.sha256(T0_MANIFEST.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"T0 manifest SHA-256 mismatch!\n  expected: {EXPECTED_SHA256}\n  actual:   {actual}\n"
            f"Refusing to proceed -- the frozen scientific split may have changed.")
    return actual


def resolve_split_rows(dataset_root: str | Path | None) -> dict[str, list[dict]]:
    """Returns {"train": [...], "valid": [...], "test": [...]}, each row
    carrying image_id/relative_path/class/split and, if dataset_root is
    given, a resolved absolute `path`. The T0 manifest's own `path` column
    is never read here -- only `relative_path`, so this function works
    identically on any machine."""
    rows = load_t0_rows()
    by_split: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for r in rows:
        split = r["new_split"]
        if split not in by_split:
            continue  # excluded_conflict rows never enter any E1 split
        out = {"image_id": r["image_id"], "relative_path": r["relative_path"],
               "class": r["class"], "split": split}
        if dataset_root is not None:
            out["path"] = str(Path(dataset_root) / r["relative_path"])
        by_split[split].append(out)
    return by_split


def verify_train_valid_paths_resolve(dataset_root: str | Path) -> dict:
    by_split = resolve_split_rows(dataset_root)
    missing = {"train": [], "valid": []}
    for split in ("train", "valid"):
        for row in by_split[split]:
            if not Path(row["path"]).exists():
                missing[split].append(row["path"])
    return {"missing_train": missing["train"], "missing_valid": missing["valid"],
            "n_train": len(by_split["train"]), "n_valid": len(by_split["valid"]),
            "n_test": len(by_split["test"])}


def run_verification(dataset_root: str | Path) -> dict:
    report = {"t0_manifest": str(T0_MANIFEST), "dataset_root": str(dataset_root)}
    report["manifest_sha256"] = verify_manifest_sha256()
    report["manifest_sha256_ok"] = True

    path_check = verify_train_valid_paths_resolve(dataset_root)
    report.update(path_check)
    report["train_count_ok"] = path_check["n_train"] == EXPECTED_COUNTS["train"]
    report["valid_count_ok"] = path_check["n_valid"] == EXPECTED_COUNTS["valid"]
    report["test_count_ok"] = path_check["n_test"] == EXPECTED_COUNTS["test"]
    report["all_train_valid_paths_resolve"] = not path_check["missing_train"] and not path_check["missing_valid"]
    report["ok"] = (
        report["manifest_sha256_ok"] and report["train_count_ok"] and report["valid_count_ok"]
        and report["test_count_ok"] and report["all_train_valid_paths_resolve"]
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", default=None, help="Optional path to write the JSON report")
    args = parser.parse_args()

    report = run_verification(args.dataset_root)
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["ok"]:
        print("\nSTOP: dataset verification FAILED. Do not proceed with training.", file=sys.stderr)
        if report["missing_train"] or report["missing_valid"]:
            n_missing = len(report["missing_train"]) + len(report["missing_valid"])
            print(f"  {n_missing} Train/Validation path(s) do not resolve under "
                  f"--dataset-root {args.dataset_root!r}. Exact missing dataset root to report: "
                  f"the --dataset-root value above does not contain the expected relative_path "
                  f"layout (train/<class>/... , valid/<class>/...).", file=sys.stderr)
        sys.exit(1)
    print("\nOK -- T0 manifest verified, all Train/Validation paths resolve, Test count matches (unopened).")


if __name__ == "__main__":
    main()
