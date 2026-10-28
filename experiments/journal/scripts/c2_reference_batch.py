"""C2 — choose the reference-mask sample and build a BLIND annotation batch (Colab CPU).

  python experiments/journal/scripts/c2_reference_batch.py \
      --manifest <T0 partition_manifest.csv> --dataset-root <Combined_Drawing> \
      --exclude <dev 84 ids csv> --exclude <holdout 80 ids csv> \
      --out <Drive>/DOAR_JOURNAL_2026/C2_reference_batch --n 120 --n-double 40

Sampling (fixed before any annotation, PREREGISTRATION.md C6):
* pool = T0 train+valid images (the locked test split is never shown), minus
  the 84 development and 80 holdout images and every image sharing a
  Roboflow stem with them
* strata = frozen-V2 support regime x A0 QC state (OK / flagged); allocation
  by largest remainder in proportion to the pool, at least 4 per non-empty
  stratum; within a stratum images are chosen by a seeded hash order
* the first --n-double tasks in a seeded order are also given to the second
  annotator

Outputs: images/R###.png (working resolution), batch_manifest.csv (task id and
file only -- what the annotator sees), key_PRIVATE.csv (task -> image id, class,
regime, QC; do not give to annotators), double_annotation_tasks.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from PIL import Image  # noqa: E402

from doar_journal import annotation as ann  # noqa: E402
from doar_journal import foreground as fg  # noqa: E402
from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import load_manifest, path_of, source_key, split_of  # noqa: E402

SEED = "doar-journal-2026-c2"


def h(x: str) -> str:
    return hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest()


def largest_remainder(counts: dict, n: int, minimum: int = 4) -> dict:
    keys = sorted(k for k, v in counts.items() if v > 0)
    quota = {k: min(minimum, counts[k]) for k in keys}
    rest = n - sum(quota.values())
    if rest < 0:
        raise ValueError("n too small for the per-stratum minimum")
    cap = {k: counts[k] - quota[k] for k in keys}
    total = sum(cap.values())
    if total == 0:
        return quota
    exact = {k: rest * cap[k] / total for k in keys}
    base = {k: min(cap[k], int(exact[k])) for k in keys}
    left = rest - sum(base.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k)):
        if left == 0:
            break
        if base[k] < cap[k]:
            base[k] += 1
            left -= 1
    return {k: quota[k] + base[k] for k in keys}


def read_ids(p: str) -> set[str]:
    with open(p, newline="", encoding="utf-8") as f:
        return {r["image_id"] for r in csv.DictReader(f)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-double", type=int, default=40)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "batch_manifest.csv").exists():
        raise SystemExit(f"{out} already holds a batch; never re-draw a batch that may have been seen")
    rows = [r for r in load_manifest(a.manifest) if split_of(r) in ("train", "valid")]
    excl = set()
    for p in a.exclude:
        excl |= read_ids(p)
    excl_stems = {source_key(path_of(r)).rf_stem for r in rows if r["image_id"] in excl} - {None}
    pool = [r for r in rows if r["image_id"] not in excl
            and source_key(path_of(r)).rf_stem not in excl_stems]
    root = Path(a.dataset_root)

    cache_p = out / "v2_regime_cache.json"
    out.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for i, r in enumerate(pool):
        if r["image_id"] in cache:
            continue
        img = fg.load_rgb(root / path_of(r).replace("\\", "/"))
        v = fg.v2(img).meta
        a0 = fg.a0(img).mask.mean()
        cache[r["image_id"]] = {"regime": v["support_regime"], "confidence": v["support_confidence"],
                                "policy": v["measurement_policy"],
                                "a0_state": "OK" if a0 >= 0.001 else "FLAGGED"}
        if i % 200 == 0:
            provenance.write_json(cache_p, cache)
            print(f"V2 regime computed for {len(cache)}/{len(pool)} pool images", flush=True)
    provenance.write_json(cache_p, cache)

    strata = defaultdict(list)
    for r in pool:
        c = cache[r["image_id"]]
        strata[(c["regime"], c["a0_state"])].append(r)
    alloc = largest_remainder({k: len(v) for k, v in strata.items()}, a.n)
    chosen = []
    for k, n in sorted(alloc.items()):
        chosen += sorted(strata[k], key=lambda r: h(r["image_id"]))[:n]
    chosen.sort(key=lambda r: h("order:" + r["image_id"]))

    (out / "images").mkdir(exist_ok=True)
    pub, key = [], []
    for i, r in enumerate(chosen, start=1):
        tid = f"R{i:03d}"
        img = ann.to_working_resolution(fg.load_rgb(root / path_of(r).replace("\\", "/")))
        Image.fromarray(img).save(out / "images" / f"{tid}.png")
        c = cache[r["image_id"]]
        pub.append({"task_id": tid, "image_file": f"{tid}.png"})
        key.append({"task_id": tid, "image_id": r["image_id"], "relative_path": path_of(r), "class": r["class"],
                    "t0_split": split_of(r), "v2_regime": c["regime"], "v2_confidence": c["confidence"],
                    "v2_policy": c["policy"], "a0_state": c["a0_state"],
                    "working_h": img.shape[0], "working_w": img.shape[1]})
    for name, data in (("batch_manifest.csv", pub), ("key_PRIVATE.csv", key)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    double = sorted(pub, key=lambda t: h("double:" + t["task_id"]))[:a.n_double]
    with open(out / "double_annotation_tasks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "image_file"])
        w.writeheader()
        w.writerows(sorted(double, key=lambda t: t["task_id"]))
    summary = {"pool": len(pool), "excluded_ids": len(excl), "excluded_stems": len(excl_stems),
               "allocation": {f"{k[0]}|{k[1]}": v for k, v in sorted(alloc.items())},
               "pool_strata": {f"{k[0]}|{k[1]}": len(v) for k, v in sorted(strata.items())},
               "chosen_by_class": dict(Counter(r["class"] for r in chosen))}
    meta = provenance.run_metadata("C2_reference_batch", vars(a), inputs={"manifest": a.manifest})
    provenance.write_json(out / "summary.json", {"summary": summary, "metadata": meta})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
