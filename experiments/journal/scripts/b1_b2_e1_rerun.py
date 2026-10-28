"""B1-B2 — rerun E1 on the FROZEN T1 split (Colab GPU; resumable).

  python experiments/journal/scripts/b1_b2_e1_rerun.py \
      --t1-dir <Drive>/DOAR_JOURNAL_2026/T1 \
      --dataset-root /content/drive/MyDrive/Masters/Datasets/Combined_Drawing \
      --out <Drive>/DOAR_JOURNAL_2026/E1_T1 --stage all

Stages (each skips work whose outputs already exist):
  probe      9 encoders x 10 seeds frozen linear probe (E1-A2 + E1-A2R protocol)
  adapt      shortlist (top-3 probe means) x frozen/partial/full x seeds 19,42,73 (E1-B)
  confirm    the 2 finalists (ranked on seeds 19,42,73 only) x seeds 11,37 (E1-C1)
  all        the three in order
The locked T1 test split is never loaded here (rows are counted only).

Engineering smoke test (run BEFORE the full GPU job; needs no frozen T1):
  python .../b1_b2_e1_rerun.py --smoke --smoke-manifest <T0 partition_manifest.csv> \
      --dataset-root <Combined_Drawing> --out <Drive>/DOAR_JOURNAL_2026/SMOKE_ENGINEERING_ONLY
  -> 2 pretrained models (CLIP ViT-B/32, EfficientNet-B0), 64 train / 32 valid
     images, 1 epoch, probe + partial adaptation, 1 seed. It only checks that
     paths, weight downloads, GPU memory, training, checkpointing and outputs
     work. Its numbers are NOT results and must never be reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import e1_analysis as ea  # noqa: E402
from doar_journal import e1_runner as er  # noqa: E402
from doar_journal import provenance  # noqa: E402
from doar_journal.t1_partition import require_frozen  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-dir", default=None, help="frozen T1 folder (required unless --smoke)")
    ap.add_argument("--smoke", action="store_true", help="engineering check only; numbers are not results")
    ap.add_argument("--smoke-manifest", default=None, help="T0 manifest for the smoke test (train/valid rows only)")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("probe", "adapt", "confirm", "all"), default="all")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.smoke:
        return smoke(a)
    if not a.t1_dir:
        raise SystemExit("--t1-dir is required (or use --smoke)")
    manifest = require_frozen(a.t1_dir)
    rows = er.load_rows(manifest, a.dataset_root)
    missing = [r["path"] for r in rows["train"] + rows["valid"] if not Path(r["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} image paths do not resolve, e.g. {missing[:2]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = provenance.git_commit(Path(__file__).resolve().parents[3])
    provenance.write_json(out / f"run_metadata_{a.stage}.json",
                          provenance.run_metadata(f"E1_T1_{a.stage}", vars(a), inputs={"t1_manifest": manifest}))
    print(f"T1 manifest {rows['manifest_sha256'][:12]}  train={len(rows['train'])} valid={len(rows['valid'])} "
          f"test={rows['n_test']} (locked, not loaded)")

    if a.stage in ("probe", "all"):
        er.run_probe(rows, out, a.device)
    if a.stage in ("adapt", "all"):
        probe_rows = ea.config_table(ea.collect_runs(out / "probe" / "runs"), n_boot=1000)
        sl = ea.shortlist(probe_rows, 3)
        provenance.write_json(out / "SHORTLIST.json", {"shortlist": sl, "rule": "top-3 10-seed mean probe macro-F1"})
        print("shortlist:", sl)
        er.run_adaptation(rows, out, a.dataset_root, commit or "unknown", a.device, models=sl)
    if a.stage in ("confirm", "all"):
        fin = ea.finalists(ea.collect_runs(out / "adapt" / "runs"), k=2)
        provenance.write_json(out / "FINALISTS.json", {"finalists": ["/".join(f) for f in fin],
                                                         "rule": "top-2 mean macro-F1 on seeds 19,42,73"})
        print("finalists:", fin)
        for model, mode in fin:
            er.run_adaptation(rows, out, a.dataset_root, commit or "unknown", a.device,
                              models=[model], modes=[mode], seeds=er.CONFIRM_SEEDS)
    print(json.dumps({"done": a.stage}))
    return 0


def smoke(a) -> int:
    import time
    if not a.smoke_manifest:
        raise SystemExit("--smoke needs --smoke-manifest (the T0 manifest)")
    out = Path(a.out)
    if "SMOKE" not in out.name.upper():
        raise SystemExit("smoke output folder name must contain SMOKE so it can never be mistaken for results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "README_NOT_RESULTS.txt").write_text(
        "Engineering smoke test: 2 models, 64 train / 32 valid images, 1 epoch.\n"
        "These numbers are NOT experimental results and must not be reported.\n", encoding="utf-8")
    rows = er.load_rows(a.smoke_manifest, a.dataset_root)
    rows["train"] = sorted(rows["train"], key=lambda r: r["image_id"])[:64]
    rows["valid"] = sorted(rows["valid"], key=lambda r: r["image_id"])[:32]
    missing = [r["path"] for r in rows["train"] + rows["valid"] if not Path(r["path"]).exists()]
    if missing:
        raise SystemExit(f"SMOKE FAIL: {len(missing)} image paths do not resolve, e.g. {missing[:2]}")
    print(f"OK image paths ({len(rows['train'])} train / {len(rows['valid'])} valid); test rows counted only: {rows['n_test']}")
    checks = {}
    t0 = time.time()
    er.run_probe(rows, out, a.device, models=("clip_vit_b32", "efficientnet_b0"), seeds=(42,), max_epochs=1, patience=1)
    checks["probe"] = f"ok in {time.time() - t0:.0f}s"
    t0 = time.time()
    commit = provenance.git_commit(Path(__file__).resolve().parents[3]) or "unknown"
    from doar_journal import _frozen_e1b as e1b
    e1b.configure(train_rows=rows["train"], valid_rows=rows["valid"], out_root=out / "adapt",
                  dataset_root=a.dataset_root, manifest_sha=rows["manifest_sha256"], source_commit=commit,
                  device=a.device, MAX_EPOCHS=1, PATIENCE=1)
    for model in ("clip_vit_b32", "efficientnet_b0"):
        r = e1b.run_one(model, "partial", 19)
        assert r.get("status") == "complete", r
        assert (out / "adapt" / "runs" / f"{model}__partial__seed_19" / "best.pt").exists()
    checks["adaptation_partial"] = f"ok in {time.time() - t0:.0f}s"
    try:
        import torch
        checks["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        checks["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover
        checks["gpu"] = repr(exc)
    provenance.write_json(out / "SMOKE_REPORT.json", {"engineering_only": True, "checks": checks,
                                                       "metadata": provenance.run_metadata("smoke", vars(a))})
    print("SMOKE PASSED (engineering only):", json.dumps(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
