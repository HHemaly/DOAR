"""B4 — the ONE evaluation on the locked T1 test split (Colab GPU).

Protocol (PREREGISTRATION.md, C5):
  1. Pre-flight, BEFORE any test image is read, refuses unless
       * T1 is FROZEN and its manifest hash matches the selection record,
       * SELECTION_FROZEN.json exists and every listed checkpoint still has
         the SHA-256 recorded at selection time,
       * --confirm-single-test-run is given,
       * no TEST_RUN_LOCK.json exists in --out (whatever its state).
  2. Writes TEST_RUN_LOCK.json with state RUNNING (+ hashes, environment).
  3. Evaluates. On ANY exception the lock is rewritten with state FAILED and
     the traceback; the test is then considered consumed. It is never retried
     automatically; a re-run needs a written justification (--rerun-justification),
     which is stored in the lock history and must be reported in the paper.
  4. On success writes predictions, results and their SHA-256 into the lock
     (state COMPLETE) and makes the result files read-only.

  python experiments/journal/scripts/b4_single_test_evaluation.py \
      --t1-dir <...>/T1 --selection <...>/B3_E1_analysis/SELECTION_FROZEN.json \
      --probe-root <...>/E1_T1/probe --dataset-root <Combined_Drawing> \
      --out <...>/B4_TEST --confirm-single-test-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import evaluation_groups, load_manifest  # noqa: E402
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


def preflight(sel: dict, manifest: str) -> list[str]:
    problems = []
    if sel.get("t1_manifest_sha256") != provenance.sha256_file(manifest):
        problems.append("T1 manifest hash differs from the one recorded at selection")
    for key, c in sel["checkpoints"].items():
        p = Path(c["path"])
        if not p.exists():
            problems.append(f"missing checkpoint {key}: {p}")
        elif provenance.sha256_file(p) != c["sha256"]:
            problems.append(f"checkpoint changed since selection: {key}")
    return problems


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


def predict_probe(model_key, ckpt, probe_root, rows, manifest_sha, device):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/E1_visual_representation/scripts"))
    import e1a2_common as ec
    cache = ec.extract_and_cache_embeddings(model_key, rows, "test", manifest_sha, probe_root, device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    head = torch.nn.Linear(cache["embedding_dimension"], len(CLASSES)).to(device)
    head.load_state_dict(payload["model_state"])
    with torch.no_grad():
        logits = head(torch.as_tensor(cache["embeddings"], device=device))
    return [str(i) for i in cache["image_ids"]], torch.softmax(logits, dim=1).cpu().numpy()


def evaluate(a, sel, manifest, out, lock_record) -> dict:
    rows = test_rows(manifest, a.dataset_root)
    by_id = {r["image_id"]: r for r in rows}
    groups = evaluation_groups(load_manifest(manifest))
    ck = sel["checkpoints"]
    systems = {}
    for role in ("selected", "comparator"):
        cfg = sel[role]
        probs, ids0 = [], None
        for s in cfg["seeds"]:
            ids, p = predict_adapted(cfg["model"], cfg["mode"], ck[f"{cfg['model']}/{cfg['mode']}/{s}"]["path"],
                                     rows, a.device)
            if ids0 is not None and ids != ids0:
                raise RuntimeError("seed runs returned different image orders")
            ids0 = ids
            probs.append(p)
            systems[f"{role}_seed_{s}"] = (ids, p)
        systems[f"{role}_ensemble"] = (ids0, np.mean(probs, axis=0))
    pb = sel["probe_baseline"]
    systems["probe_baseline"] = predict_probe(pb["model"], ck[f"{pb['model']}/probe/42"]["path"], a.probe_root,
                                              rows, provenance.sha256_file(manifest), a.device)
    results, preds, files = {"bootstrap_unit": "duplicate/source group"}, {}, {}
    for name, (ids, p) in systems.items():
        y = np.array([by_id[i]["class"] for i in ids])
        yhat = np.array(CLASSES)[p.argmax(axis=1)]
        g = [groups.get(i, i) for i in ids]
        preds[name] = (ids, y, yhat, g)
        ci = bootstrap_metric(y, yhat, groups=g)
        per_class = {c: float(((yhat == c) & (y == c)).sum() * 2 / max(1, (yhat == c).sum() + (y == c).sum()))
                     for c in CLASSES}
        results[name] = {"macro_f1": ci.as_dict(), "accuracy": float((y == yhat).mean()), "n": len(ids),
                         "n_groups": len(set(g)), "per_class_f1": per_class}
        f = out / f"test_predictions_{name}.csv"
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image_id", "eval_group", "true_class", "predicted_class", *[f"prob_{c}" for c in CLASSES]])
            for i, gi, t, h, pr in zip(ids, g, y, yhat, p):
                w.writerow([i, gi, t, h, *[f"{v:.6f}" for v in pr]])
        files[f.name] = provenance.sha256_file(f)
    ida, ya, pa, ga = preds["selected_ensemble"]
    for other in ("comparator_ensemble", "probe_baseline"):
        idb, _yb, pbb, _gb = preds[other]
        order = {i: k for k, i in enumerate(idb)}
        pb_aligned = np.array([pbb[order[i]] for i in ida])
        results[f"selected_vs_{other}"] = {
            "paired_group_bootstrap": paired_bootstrap_diff(ya, pa, pb_aligned, groups=ga),
            "mcnemar_counts_descriptive": mcnemar_exact(pa == ya, pb_aligned == ya)}
    provenance.write_json(out / "test_results.json", results)
    files["test_results.json"] = provenance.sha256_file(out / "test_results.json")
    for name in files:
        os.chmod(out / name, 0o444)
    return {"results": results, "files_sha256": files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--probe-root", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confirm-single-test-run", action="store_true")
    ap.add_argument("--rerun-justification", default="",
                    help="only after a FAILED run; recorded permanently and must be reported")
    a = ap.parse_args(argv)
    if not a.confirm_single_test_run:
        raise SystemExit("refusing: pass --confirm-single-test-run to evaluate the locked test split once")
    manifest = require_frozen(a.t1_dir)
    sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    problems = preflight(sel, manifest)
    if problems:
        raise SystemExit("refusing, pre-flight failed:\n  " + "\n  ".join(problems))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "TEST_RUN_LOCK.json"
    history = []
    if lock.exists():
        prev = json.loads(lock.read_text(encoding="utf-8"))
        if prev.get("state") != "FAILED" or not a.rerun_justification.strip():
            raise SystemExit(f"refusing: test split already used (state {prev.get('state')}); see {lock}. "
                             "Only a FAILED run may be repeated, and only with --rerun-justification.")
        history = prev.get("history", []) + [prev]
    base = {"state": "RUNNING", "selection_sha256": provenance.sha256_file(a.selection),
            "t1_manifest_sha256": provenance.sha256_file(manifest),
            "checkpoint_sha256": {k: v["sha256"] for k, v in sel["checkpoints"].items()},
            "rerun_justification": a.rerun_justification or None, "history": history,
            "environment": provenance.run_metadata("B4_test", vars(a))}
    provenance.write_json(lock, base)
    try:
        done = evaluate(a, sel, manifest, out, base)
    except Exception:
        base.update({"state": "FAILED", "traceback": traceback.format_exc()})
        provenance.write_json(lock, base)
        raise
    base.update({"state": "COMPLETE", "outputs_sha256": done["files_sha256"]})
    provenance.write_json(lock, base)
    os.chmod(lock, 0o444)
    print(json.dumps({k: v["macro_f1"] for k, v in done["results"].items() if isinstance(v, dict) and "macro_f1" in v},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
