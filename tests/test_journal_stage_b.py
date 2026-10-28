"""CPU smoke tests for Stage B plumbing (no pretrained weights, no GPU).

The real runs happen in Colab on a GPU. These tests only check that the
verbatim E1-B code, driven through e1_runner/configure, trains, writes
resumable outputs and skips completed runs on a re-call.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doar_journal import e1_runner  # noqa: E402


@pytest.fixture()
def tiny_manifest(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        cls = ["Angry", "Fear", "Happy", "Sad"][i % 4]
        split = "valid" if i >= 16 else "train"
        rel = f"{split}/{cls}/img{i}.png"
        p = tmp_path / "data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(p)
        rows.append({"image_id": f"i{i:02d}", "relative_path": rel, "path": "", "class": cls, "new_split": split})
    rows.append({"image_id": "t0", "relative_path": "test/x.png", "path": "", "class": "Sad", "new_split": "test"})
    m = tmp_path / "manifest.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return m, tmp_path / "data"


def test_load_rows_counts_test_only(tiny_manifest):
    m, root = tiny_manifest
    rows = e1_runner.load_rows(m, root)
    assert len(rows["train"]) == 16 and len(rows["valid"]) == 8 and rows["n_test"] == 1
    assert all("test" not in r["relative_path"] for r in rows["train"] + rows["valid"])


def test_adaptation_runs_and_resumes_on_cpu(tiny_manifest, tmp_path, monkeypatch):
    from doar_journal import _frozen_e1b as e1b
    real = e1b.models.efficientnet_b0
    monkeypatch.setattr(e1b.models, "efficientnet_b0", lambda weights=None: real(weights=None))
    for name in ("empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    m, root = tiny_manifest
    rows = e1_runner.load_rows(m, root)
    e1b.configure(train_rows=rows["train"], valid_rows=rows["valid"], out_root=tmp_path / "out",
                  dataset_root=root, manifest_sha=rows["manifest_sha256"], source_commit="test",
                  device="cpu", MAX_EPOCHS=1, NUM_WORKERS=0, BATCH_SIZE=8)
    r = e1b.run_one("efficientnet_b0", "partial", 19)
    run_dir = tmp_path / "out" / "runs" / "efficientnet_b0__partial__seed_19"
    assert (run_dir / "best_valid_predictions.csv").exists()
    assert json.loads((run_dir / "result.json").read_text())["status"] == "complete"
    again = e1b.run_one("efficientnet_b0", "partial", 19)  # must SKIP, not retrain
    assert again["best_valid_macro_f1"] == r["best_valid_macro_f1"]


def _write_preds(path, truth, pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = ["Angry", "Fear", "Happy", "Sad"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "true_class", "predicted_class", *[f"prob_{c}" for c in classes]])
        for i, (t, p) in enumerate(zip(truth, pred)):
            w.writerow([f"v{i}", t, p, *[0.7 if c == p else 0.1 for c in classes]])


def test_e1_analysis_rules(tmp_path):
    from doar_journal import e1_analysis as ea
    rng = np.random.default_rng(1)
    truth = list(rng.choice(["Angry", "Fear", "Happy", "Sad"], 80))
    def noisy(acc, seed):
        r = np.random.default_rng(seed)
        return [t if r.random() < acc else r.choice(["Angry", "Fear", "Happy", "Sad"]) for t in truth]
    acc = {("clip_vit_b32", "partial"): 0.9, ("siglip2_vit_b16", "partial"): 0.8, ("efficientnet_b0", "full"): 0.6}
    for (m, md), a in acc.items():
        for s in (19, 42, 73):
            _write_preds(tmp_path / "adapt" / f"{m}__{md}__seed_{s}" / "best_valid_predictions.csv", truth, noisy(a, s))
    runs = ea.collect_runs(tmp_path / "adapt")
    fin = ea.finalists(runs, k=2)
    assert fin == [("clip_vit_b32", "partial"), ("siglip2_vit_b16", "partial")]
    for (m, md) in fin:
        for s in (11, 37):
            _write_preds(tmp_path / "adapt" / f"{m}__{md}__seed_{s}" / "best_valid_predictions.csv", truth, noisy(acc[(m, md)], s))
    runs = ea.collect_runs(tmp_path / "adapt")
    assert ea.finalists(runs, k=2) == fin  # added seeds cannot change finalists
    rows = [r for r in ea.config_table(runs, n_boot=200) if (r["model"], r["mode"]) in fin]
    assert ea.select(rows) == ("clip_vit_b32", "partial")
    cmp = ea.compare_finalists(runs, fin[0], fin[1], n_boot=200)
    assert cmp["paired_bootstrap_macro_f1"]["estimate"] > 0 and "mcnemar_counts_descriptive" in cmp
    _write_preds(tmp_path / "a2r" / "runs" / "resnet18" / "seed_7" / "best_valid_predictions.csv", truth, truth)
    assert ("resnet18", "probe") in ea.collect_runs(tmp_path / "a2r")


def _load_script(name):
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "experiments/journal/scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_b4_test_lock_protocol(tmp_path, monkeypatch):
    from doar_journal import provenance, t1_partition
    b4 = _load_script("b4_single_test_evaluation")
    rows = [{"image_id": f"i{i}", "path": f"C:\\d\\{s}\\Happy\\x{i}.jpg", "relative_path": "", "class": "Happy",
             "split": s, "new_split": s, "original_split": s, "group_id": "", "sha256": "", "dhash": ""}
            for i, s in enumerate(["train", "valid", "test", "test"])]
    t1 = tmp_path / "T1"
    t1_partition.write_t1(t1_partition.build_t1(rows, strategy="minimal_change"), t1, inputs={})
    t1_partition.freeze(t1, approved_by="tester", deviation="toy data in a unit test")
    ck = tmp_path / "best.pt"
    ck.write_bytes(b"weights")
    sel = {"t1_manifest_sha256": provenance.sha256_file(t1 / "t1_partition_manifest.csv"),
           "checkpoints": {"m/partial/19": {"path": str(ck), "sha256": provenance.sha256_file(ck)}}}
    selp = tmp_path / "SELECTION_FROZEN.json"
    selp.write_text(json.dumps(sel))
    args = ["--t1-dir", str(t1), "--selection", str(selp), "--probe-root", "x", "--dataset-root", "x",
            "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        b4.main(args)                                            # no confirmation flag
    calls = {"n": 0}
    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("GPU died")
    monkeypatch.setattr(b4, "evaluate", boom)
    with pytest.raises(RuntimeError):
        b4.main(args + ["--confirm-single-test-run"])
    lock = json.loads((tmp_path / "out" / "TEST_RUN_LOCK.json").read_text())
    assert lock["state"] == "FAILED" and "GPU died" in lock["traceback"]
    with pytest.raises(SystemExit):
        b4.main(args + ["--confirm-single-test-run"])           # failed run is not silently retried
    monkeypatch.setattr(b4, "evaluate", lambda *a, **k: {"results": {}, "files_sha256": {}})
    assert b4.main(args + ["--confirm-single-test-run", "--rerun-justification", "Colab disconnect"]) == 0
    lock = json.loads((tmp_path / "out" / "TEST_RUN_LOCK.json").read_text())
    assert lock["state"] == "COMPLETE" and lock["history"][0]["state"] == "FAILED"
    with pytest.raises(SystemExit):
        b4.main(args + ["--confirm-single-test-run", "--rerun-justification", "again"])  # COMPLETE is final
    ck.write_bytes(b"tampered")
    assert b4.preflight(sel, str(t1 / "t1_partition_manifest.csv"))   # changed checkpoint detected
