"""Stage B: rerun the existing E1 pipeline on a frozen manifest (T1).

Nothing here re-implements a model: the frozen linear probe is the committed
E1-A2 code (experiments/E1_visual_representation/scripts/e1a2_common.py at
the commit that produced the reported results), and adaptation is the
verbatim E1-B notebook code in `_frozen_e1b`. This module only (a) reads
rows from a manifest other than T0, (b) loops seeds, and (c) writes outputs
in a resumable layout.

Layout under OUT (one directory per experiment, never shared with T0 results):
  probe/embeddings/<model>/{train,valid}.npz        cached frozen embeddings
  probe/runs/<model>__probe__seed_<s>/               best_valid_predictions.csv, result.json
  adapt/runs/<model>__<mode>__seed_<s>/              (written by _frozen_e1b.run_one)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from .provenance import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
E1_SCRIPTS = ROOT / "experiments/E1_visual_representation/scripts"

PROBE_MODELS = ("mobilenet_v3_small", "resnet18", "efficientnet_b0", "densenet121", "convnext_tiny",
                "vit_b_16", "clip_vit_b32", "dinov2_vits14", "siglip2_vit_b16")
PROBE_SEEDS = (7, 19, 31, 42, 53, 67, 73, 89, 101, 131)          # E1-A2R seeds
ADAPT_MODELS = ("siglip2_vit_b16", "clip_vit_b32", "efficientnet_b0")  # E1-B shortlist rule: re-derived on T1
ADAPT_MODES = ("frozen", "partial", "full")
ADAPT_SEEDS = (19, 42, 73)                                         # E1-B seeds
CONFIRM_SEEDS = (11, 37)                                           # E1-C1 added seeds


def load_rows(manifest: str | Path, dataset_root: str | Path, expected_sha: str | None = None) -> dict:
    """Train/valid rows from a partition manifest. Test rows are COUNTED only."""
    sha = sha256_file(manifest)
    if expected_sha and sha != expected_sha:
        raise RuntimeError(f"manifest hash {sha} != expected {expected_sha}")
    out = {"train": [], "valid": [], "n_test": 0, "manifest_sha256": sha}
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = r["new_split"]
            if s == "test":
                out["n_test"] += 1
                continue
            if s not in ("train", "valid"):
                continue
            rel = (r.get("relative_path") or "").replace("\\", "/")
            if not rel:
                parts = r["path"].replace("\\", "/").split("/")
                rel = "/".join(parts[-3:])
            out[s].append({"image_id": r["image_id"], "relative_path": rel,
                           "path": str(Path(dataset_root) / rel), "class": r["class"], "split": s})
    tv = {r["image_id"] for r in out["train"]} & {r["image_id"] for r in out["valid"]}
    if tv:
        raise RuntimeError(f"train/valid overlap: {len(tv)} images")
    return out


def _ec():
    sys.path.insert(0, str(E1_SCRIPTS))
    import e1a2_common as ec  # noqa: E402
    return ec


def run_probe(rows: dict, out: str | Path, device: str, models=PROBE_MODELS, seeds=PROBE_SEEDS,
              max_epochs: int = 30, patience: int = 5) -> list[dict]:
    """E1-A2 frozen linear probe for every model x seed (embeddings extracted once)."""
    import numpy as np
    ec = _ec()
    out = Path(out) / "probe"
    results = []
    c2i = {c: i for i, c in enumerate(ec.CLASSES)}
    for model in models:
        tr = ec.extract_and_cache_embeddings(model, rows["train"], "train", rows["manifest_sha256"], out, device)
        va = ec.extract_and_cache_embeddings(model, rows["valid"], "valid", rows["manifest_sha256"], out, device)
        ytr = np.asarray([c2i[str(c)] for c in tr["labels"]])
        yva = np.asarray([c2i[str(c)] for c in va["labels"]])
        cw = ec.compute_class_weights(ytr.tolist())
        for seed in seeds:
            run_dir = out / "runs" / f"{model}__probe__seed_{seed}"
            res_path = run_dir / "result.json"
            if res_path.exists() and json.loads(res_path.read_text()).get("status") == "complete":
                results.append(json.loads(res_path.read_text()))
                continue
            ec.set_seed(seed)
            history, best_epoch, best_f1 = ec.run_linear_probe(
                tr["embeddings"], ytr, va["embeddings"], yva, va["image_ids"], device=device,
                max_epochs=max_epochs, patience=patience, head_lr=3e-4, weight_decay=1e-4,
                class_weights=cw, output_dir=run_dir, model_key=model, resume=True)
            res = {"status": "complete", "model": model, "mode": "probe", "seed": seed,
                   "best_epoch": best_epoch, "best_valid_macro_f1": best_f1,
                   "manifest_sha256": rows["manifest_sha256"], "weights_id": tr["weights_id"]}
            write_json(res_path, res)
            (run_dir / "epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            results.append(res)
            print(f"{model:20s} seed {seed:4d}  macro-F1 {best_f1:.4f}")
    return results


def run_adaptation(rows: dict, out: str | Path, dataset_root: str | Path, source_commit: str, device: str = "cuda",
                   models=ADAPT_MODELS, modes=ADAPT_MODES, seeds=ADAPT_SEEDS) -> list[dict]:
    """E1-B adaptation runs through the verbatim notebook `run_one` (resumable)."""
    from . import _frozen_e1b as e1b
    e1b.configure(train_rows=rows["train"], valid_rows=rows["valid"], out_root=Path(out) / "adapt",
                  dataset_root=dataset_root, manifest_sha=rows["manifest_sha256"],
                  source_commit=source_commit, device=device, MODELS=list(models), MODES=list(modes))
    results = []
    for model in models:
        for mode in modes:
            for seed in seeds:
                r = e1b.run_one(model, mode, seed)
                results.append({"model": model, "mode": mode, "seed": seed,
                                "best_valid_macro_f1": r.get("best_valid_macro_f1"),
                                "status": r.get("status")})
    return results
