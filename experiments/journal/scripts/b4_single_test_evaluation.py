"""B4 — the ONE evaluation on the locked T1 test split (Colab GPU).

Refuses to run unless ALL of the following hold:
  * T1 is FROZEN (FROZEN.json, manifest hash unchanged)
  * B3 wrote SELECTION_FROZEN.json
  * the caller passes --confirm-single-test-run
  * no TEST_RUN_LOCK.json exists yet in --out (a second run is impossible
    without deleting that file by hand, which would be visible in the record)

  python experiments/journal/scripts/b4_single_test_evaluation.py \
      --t1-dir <...>/T1 --selection <...>/B3_E1_analysis/SELECTION_FROZEN.json \
      --probe-root <...>/E1_T1/probe --dataset-root <Combined_Drawing> \
      --out <...>/B4_TEST --confirm-single-test-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.stats import CLASSES, bootstrap_metric, mcnemar_exact, paired_bootstrap_diff  # noqa: E402
from doar_journal.t1_partition import require_frozen  # noqa: E402


def test_rows(manifest: str, dataset_root: str) -> list[dict]:
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["new_split"] == "test":
                rel = (r.get("relative_path") or "/".join(r["path"].replace("\\", "/").split("/")[-3:]))
                rows.append({"image_id": r["image_id"], "relative_path": rel,
                             "path": str(Path(dataset_root) / rel), "class": r["class"]})
    return rows


def predict_adapted(model_key, mode, ckpt, rows, device):
    import torch
    from torch.utils.data import DataLoader

    from doar_journal import _frozen_e1b as e1b
    model, _train_tf, eval_tf, _dim = e1b.build_model_and_transforms(model_key, mode)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model = model.to(device)
    loader = DataLoader(e1b.DrawingDataset(rows, eval_tf), batch_size=32, shuffle=False, num_workers=2)
    res = e1b.evaluate(model, loader, torch.nn.CrossEntropyLoss())
    return list(res["image_ids"]), np.asarray(res["probabilities"], dtype=np.float64)


def predict_probe(model_key, probe_root, rows, manifest_sha, device):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/E1_visual_representation/scripts"))
    import e1a2_common as ec
    cache = ec.extract_and_cache_embeddings(model_key, rows, "test", manifest_sha, probe_root, device)
    ckpt = torch.load(Path(probe_root) / "runs" / f"{model_key}__probe__seed_42" / "best.pt",
                      map_location=device, weights_only=False)
    head = torch.nn.Linear(cache["embedding_dimension"], len(CLASSES)).to(device)
    head.load_state_dict(ckpt["model_state"])
    with torch.no_grad():
        logits = head(torch.as_tensor(cache["embeddings"], device=device))
    return [str(i) for i in cache["image_ids"]], torch.softmax(logits, dim=1).cpu().numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--probe-root", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confirm-single-test-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.confirm_single_test_run:
        raise SystemExit("refusing: pass --confirm-single-test-run to evaluate the locked test split once")
    manifest = require_frozen(a.t1_dir)
    out = Path(a.out)
    lock = out / "TEST_RUN_LOCK.json"
    if lock.exists():
        raise SystemExit(f"refusing: the test split was already evaluated ({lock})")
    sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    provenance.write_json(lock, {"selection_sha256": provenance.sha256_file(a.selection),
                                 "t1_manifest_sha256": provenance.sha256_file(manifest),
                                 "metadata": provenance.run_metadata("B4_test", vars(a))})
    rows = test_rows(manifest, a.dataset_root)
    by_id = {r["image_id"]: r for r in rows}

    systems = {}
    for role in ("selected", "comparator"):
        cfg = sel[role]
        probs, ids0 = [], None
        for s in cfg["seeds"]:
            ckpt = sel["checkpoints"][f"{cfg['model']}/{cfg['mode']}/{s}"]
            ids, p = predict_adapted(cfg["model"], cfg["mode"], ckpt, rows, a.device)
            ids0 = ids0 or ids
            assert ids == ids0
            probs.append(p)
            systems[f"{role}_seed_{s}"] = (ids, p)
        systems[f"{role}_ensemble"] = (ids0, np.mean(probs, axis=0))
    systems["probe_baseline"] = predict_probe(sel["probe_baseline"]["model"], a.probe_root, rows,
                                              provenance.sha256_file(manifest), a.device)

    results, preds = {}, {}
    for name, (ids, p) in systems.items():
        y = np.array([by_id[i]["class"] for i in ids])
        yhat = np.array(CLASSES)[p.argmax(axis=1)]
        preds[name] = (ids, y, yhat, p)
        ci = bootstrap_metric(y, yhat)
        results[name] = {"macro_f1": ci.as_dict(), "accuracy": float((y == yhat).mean()), "n": len(ids)}
        with open(out / f"test_predictions_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "true_class", "predicted_class", *[f"prob_{c}" for c in CLASSES]])
            for i, t, h, pr in zip(ids, y, yhat, p):
                w.writerow([i, t, h, *[f"{v:.6f}" for v in pr]])
    for other in ("comparator_ensemble", "probe_baseline"):
        ida, ya, pa, _ = preds["selected_ensemble"]
        idb, yb, pb, _ = preds[other]
        order = {i: k for k, i in enumerate(idb)}
        pb_aligned = np.array([pb[order[i]] for i in ida])
        results[f"selected_vs_{other}"] = {"paired_bootstrap": paired_bootstrap_diff(ya, pa, pb_aligned),
                                           "mcnemar": mcnemar_exact(pa == ya, pb_aligned == ya)}
    provenance.write_json(out / "test_results.json", results)
    print(json.dumps({k: v.get("macro_f1", v) for k, v in results.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
