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


def _fake_dinov2():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            return x + self.fc(x)

    class FakeDino(nn.Module):              # same attribute API as DinoVisionTransformer
        embed_dim = 8

        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(3, 8)
            self.blocks = nn.ModuleList([Block() for _ in range(12)])
            self.norm = nn.LayerNorm(8)

        def forward(self, x):
            t = self.patch_embed(x.mean(dim=(2, 3)))
            for b in self.blocks:
                t = b(t)
            return self.norm(t)

    return FakeDino()


def test_dinov2_extension_modes_and_routing(monkeypatch):
    from doar_journal import _frozen_e1b as e1b
    from doar_journal import e1b_dinov2 as d
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: _fake_dinov2())
    d.install(e1b)
    d.install(e1b)                            # idempotent
    pct = {}
    for mode in ("frozen", "partial", "full"):
        m, tr, ev, dim = e1b.build_model_and_transforms("dinov2_vits14", mode)
        rep = e1b.parameter_report(m, "dinov2_vits14")
        pct[mode] = rep["visual_trainable_percent"]
        assert dim == 8 and [type(t).__name__ for t in ev.transforms] == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
        assert "RandomAffine" in [type(t).__name__ for t in tr.transforms]
        groups = e1b.build_optimizer(m, "dinov2_vits14").param_groups
        assert groups[0]["lr"] == e1b.HEAD_LR and len(groups[0]["params"]) == 2   # head weight + bias
        assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 4)
    assert pct["frozen"] == 0 and pct["full"] == 100 and 25 <= pct["partial"] < 50
    m, *_ = e1b.build_model_and_transforms("dinov2_vits14", "partial")
    assert all(p.requires_grad for p in m.backbone.blocks[-1].parameters())
    assert not any(p.requires_grad for p in m.backbone.blocks[0].parameters())
    assert all(p.requires_grad for p in m.backbone.norm.parameters())
    with pytest.raises(ValueError):
        e1b.build_model_and_transforms("unknown_model", "frozen")   # other keys -> archived builder


def test_b1c_audit_flags_good_and_bad_folders(tmp_path):
    import importlib.util
    from doar_journal import e1_runner as er
    from doar_journal.provenance import sha256_file, write_json
    t1 = tmp_path / "T1_v2"
    t1.mkdir()
    rows = [{"image_id": f"v{i}", "new_split": "valid", "class": "Sad"} for i in range(4)] + \
           [{"image_id": "t1", "new_split": "test", "class": "Sad"}]
    with open(t1 / "t1_partition_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sha = sha256_file(t1 / "t1_partition_manifest.csv")
    write_json(t1 / "FROZEN.json", {"manifest_sha256": sha})
    out = tmp_path / "E1_T1_v2"
    write_json(out / "BOUND_MANIFEST.json", {"manifest_sha256": sha})

    def preds(d, ids):
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "best_valid_predictions.csv", "w", newline="") as f:
            f.write("image_id,true_class,predicted_class,prob_Angry,prob_Fear,prob_Happy,prob_Sad\n"
                    + "".join(f"{i},Sad,Sad,0.1,0.1,0.1,0.7\n" for i in ids))

    for k, m in enumerate(er.PROBE_MODELS):
        for s in er.PROBE_SEEDS:
            d = out / "probe" / "runs" / f"{m}__probe__seed_{s}"
            preds(d, [f"v{i}" for i in range(4)])
            f1 = 0.5 + 0.01 * k
            write_json(d / "result.json", {"status": "complete", "manifest_sha256": sha, "model": m, "mode": "probe",
                                           "seed": s, "best_valid_macro_f1": f1})
    spec = importlib.util.spec_from_file_location("b1c", Path(__file__).resolve().parents[1] /
                                                  "experiments/journal/scripts/b1c_audit_runs.py")
    b1c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b1c)
    assert b1c.main(["--t1-dir", str(t1), "--out", str(out)]) == 0      # probe complete, adaptation all to run
    rep = json.loads(sorted(out.glob("AUDIT_*.json"))[-1].read_text())
    assert rep["probe"]["complete"] == 90 and len(rep["adapt"]["will_run"]) == 27
    for m in rep["shortlist_by_rule"]:
        for md in er.ADAPT_MODES:
            for s in er.ADAPT_SEEDS:
                d = out / "adapt" / "runs" / f"{m}__{md}__seed_{s}"
                preds(d, [f"v{i}" for i in range(4)])
                (d / "epoch_history.csv").write_text("epoch,train_loss\n1,0.5\n")    # archived CSV has no hash column
                write_json(d / "result.json", {"status": "complete"})
                if (m, md, s) != (rep["shortlist_by_rule"][0], "partial", 19):
                    (d / "best.pt").write_bytes(b"x")
    assert b1c.main(["--t1-dir", str(t1), "--out", str(out)]) == 0      # missing best.pt is a warning, not a stop
    rep = json.loads(sorted(out.glob("AUDIT_*.json"))[-1].read_text())
    assert rep["adapt"]["complete"] == 27 and len(rep["adapt"]["complete_but_best_pt_missing"]) == 1 and rep["warnings"]
    bad = out / "probe" / "runs" / f"{er.PROBE_MODELS[0]}__probe__seed_7"
    preds(bad, ["v0", "v1", "v2", "t1"])                                 # a test image in validation predictions
    assert b1c.main(["--t1-dir", str(t1), "--out", str(out)]) == 1
