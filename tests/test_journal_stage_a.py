"""Tests for the journal Stage A modules (leakage, T1, rescoring, stats)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doar_journal import leakage, rescoring, stats, t1_partition  # noqa: E402


def _row(i, fn, split, cls, group=""):
    return {"image_id": f"id{i:03d}", "path": f"C:\\data\\{split}\\{cls}\\{fn}", "relative_path": "",
            "class": cls, "split": split, "new_split": split, "original_split": split,
            "group_id": group, "sha256": "", "dhash": ""}


ROWS = [
    _row(1, "Happy_1_7_jpg.rf.aaaaaaaa11.jpg", "train", "Happy"),
    _row(2, "Happy_1_7_jpg.rf.bbbbbbbb22.jpg", "valid", "Happy"),
    _row(3, "Happy_1_7_jpg.rf.cccccccc33.jpg", "test", "Happy"),
    _row(4, "Sad_2_9_jpg.rf.dddddddd44.jpg", "train", "Sad"),
    _row(5, "s100.jpeg", "train", "Sad"),
    _row(6, "s100.jpeg", "valid", "Sad"),
    _row(7, "Angry_3_1_jpg.rf.eeeeeeee55.jpg", "valid", "Angry"),
    _row(8, "Fear_4_1_jpg.rf.ffffffff66.jpg", "test", "Fear"),
]


def test_source_key_roboflow_and_plain():
    k = leakage.source_key("C:\\x\\Happy_1_69_jpg.rf.feb408cd83bbe8ab41ba85041533239e.jpg")
    assert k.is_roboflow and k.rf_stem == "Happy_1_69_jpg" and k.name_key == "happy_1_69"
    p = leakage.source_key("/a/b/h12.JPG")
    assert not p.is_roboflow and p.name_key == "h12"


def test_stem_audit_counts():
    a = leakage.stem_audit(ROWS)
    assert a["n_roboflow_named"] == 6
    assert a["rf_stems_spanning_splits"] == {"test-train-valid": 1}
    assert a["exposure_by_split"]["valid"]["share_rf_stem_with_train"] == 1
    assert a["exposure_by_split"]["test"]["share_rf_stem_with_train"] == 1
    assert a["rf_stem_groups_with_class_conflict"] == 0


def test_popcount_and_dihedral_invariance(tmp_path):
    from PIL import Image
    x = np.array([0, 1, 0xFF, 2**64 - 1], dtype=np.uint64)
    assert leakage.popcount64(x).tolist() == [0, 1, 8, 64]
    rng = np.random.default_rng(0)
    img = (rng.random((64, 48)) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "a.png")
    Image.fromarray(np.ascontiguousarray(np.rot90(np.fliplr(img)))).save(tmp_path / "b.png")
    Image.fromarray((rng.random((64, 48)) * 255).astype(np.uint8)).save(tmp_path / "c.png")
    h = {k: leakage.dihedral_dhashes(tmp_path / f"{k}.png") for k in "abc"}
    pairs = leakage.dihedral_near_pairs(h, threshold=6)
    assert [(p["a"], p["b"]) for p in pairs] == [("a", "b")]


def test_review_sampling_is_blind_and_balanced():
    rows = []
    for i in range(30):
        rows.append(_row(100 + i, f"Happy_{i}_1_jpg.rf.{i:08x}aa.jpg", "train", "Happy"))
        rows.append(_row(200 + i, f"Happy_{i}_1_jpg.rf.{i:08x}bb.jpg", "valid", "Happy"))
    pairs = leakage.sample_review_pairs(rows, n_pairs=10, n_controls=5)
    kinds = [p["pair_type_hidden"] for p in pairs]
    assert kinds.count("same_rf_stem") == 10 and kinds.count("control") == 5
    for p in pairs:
        if p["pair_type_hidden"] == "control":
            assert leakage.source_key(p["path_a"]).name_key != leakage.source_key(p["path_b"]).name_key


def test_t1_groups_stems_and_verifies():
    res = t1_partition.build_t1(ROWS, strategy="minimal_change")
    split = {m["image_id"]: m["new_split"] for m in res["manifest"]}
    assert split["id001"] == split["id002"] == split["id003"] == "test"  # tie-break test > valid > train
    assert res["report"]["ok"]
    assert split["id005"] == "train" and split["id006"] == "valid"  # plain names are not merged by default
    full = t1_partition.build_t1(ROWS, strategy="regroup_full")
    assert full["report"]["ok"]


def test_t1_freeze_requires_unchanged_manifest(tmp_path):
    res = t1_partition.build_t1(ROWS)
    t1_partition.write_t1(res, tmp_path, inputs={})
    with pytest.raises(RuntimeError):
        t1_partition.require_frozen(tmp_path)
    with pytest.raises(RuntimeError):            # 8 toy images cannot meet the balance criteria
        t1_partition.freeze(tmp_path, approved_by="tester")
    t1_partition.freeze(tmp_path, approved_by="tester", deviation="toy data in a unit test")
    assert t1_partition.require_frozen(tmp_path).endswith("t1_partition_manifest.csv")
    with pytest.raises(RuntimeError):
        t1_partition.write_t1(res, tmp_path, inputs={})


def test_rescoring_subsets():
    shared, indep = rescoring.shared_source_valid_ids(ROWS)
    assert shared == {"id002"} and indep == {"id006", "id007"}
    preds = [{"image_id": "id002", "true_class": "Happy", "predicted_class": "Happy"},
             {"image_id": "id006", "true_class": "Sad", "predicted_class": "Happy"},
             {"image_id": "id007", "true_class": "Angry", "predicted_class": "Angry"}]
    r = rescoring.rescore_run(preds, shared, indep, n_boot=200)
    assert r["n_shared"] == 1 and r["n_independent"] == 2
    assert r["accuracy_shared"] == 1.0 and r["accuracy_independent"] == 0.5


def test_stats_basics():
    w = stats.wilson(8, 10)
    assert 0.44 < w.low < 0.5 and 0.94 < w.high < 0.97
    m = stats.mcnemar_exact([1, 1, 1, 0, 1], [0, 0, 1, 0, 1])
    assert m["a_right_b_wrong"] == 2 and m["a_wrong_b_right"] == 0 and m["p_exact"] == 0.5
    assert stats.kendall_tau(["a", "b", "c"], ["a", "b", "c"])["tau"] == pytest.approx(1.0)
    h = stats.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert h["x"]["p_holm"] == pytest.approx(0.03) and h["y"]["p_holm"] == pytest.approx(0.06)
    perfect = np.array([[1, 1], [2, 2], [3, 3], [4, 4.0]])
    assert stats.icc_2_1(perfect) == pytest.approx(1.0)
    y = np.array(["Happy", "Sad", "Fear", "Angry"] * 25)
    ci = stats.bootstrap_metric(y, y, n_boot=200)
    assert ci.estimate == 1.0 and ci.low == 1.0
    d = stats.paired_bootstrap_diff(y, y, np.roll(y, 1), n_boot=200)
    assert d["estimate"] == 1.0 and d["ci_low"] > 0.9


def test_vectorised_macro_f1_matches_sklearn():
    rng = np.random.default_rng(3)
    C = np.array(stats.CLASSES)
    for n in (5, 40, 284):
        y = C[rng.integers(0, 4, n)]
        p = np.where(rng.random(n) < 0.6, y, C[rng.integers(0, 4, n)])
        p[0] = "Unknown"  # a label outside the class set must never count as correct
        idx = rng.integers(0, n, size=(300, n))
        fast = stats.macro_f1_resamples(y, p, idx)
        slow = np.array([stats.macro_f1(y[i], p[i]) for i in idx])
        assert np.allclose(fast, slow, atol=1e-12)


def test_review_widget_logic_and_construction(tmp_path):
    from PIL import Image
    from doar_journal import review_widgets as rw
    pairs = [{"pair_id": "P001", "pair_type_hidden": "control", "path_a": "train/Happy/a.png",
              "path_b": "valid/Happy/b.png"},
             {"pair_id": "P002", "pair_type_hidden": "same_rf_stem", "path_a": "train/Happy/a.png",
              "path_b": "valid/Happy/b.png"}]
    for rel in ("train/Happy/a.png", "valid/Happy/b.png"):
        p = tmp_path / "data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(p)
    rd = tmp_path / "A5"
    rd.mkdir()
    import csv
    with open(rd / "pairs_rf_stem.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    leakage.write_review_template(pairs, rd / "review_rf_stem.csv")
    fields, rows = rw.read_rows(rd / "review_rf_stem.csv")
    assert rw.first_open(rows, rw.pair_done) == 0
    rw.record_pair(rows, 0, "DIFFERENT_DRAWING", "rotation", "HE")
    assert rows[0]["transform_seen"] == "not_applicable"
    rw.record_pair(rows, 1, "SAME_DRAWING", "flip", "HE")
    rw.write_rows(rd / "review_rf_stem.csv", fields, rows)
    scored = leakage.score_review(pairs, rd / "review_rf_stem.csv")
    assert scored["same_rf_stem"]["judged_same"] == 1 and scored["control"]["judged_same"] == 0
    pytest.importorskip("ipywidgets")
    rw.pair_review(rd, tmp_path / "data", "rf_stem", "HE")  # builds and displays without error


def test_cluster_bootstrap_keeps_groups_together_and_widens_ci():
    groups = np.array([f"g{i // 3}" for i in range(60)])          # 20 groups of 3 copies
    rows, flat = stats.draw_resamples(60, groups, n_boot=50, seed=1)
    for b in range(50):
        members = flat[rows == b]
        from collections import Counter
        assert all(c % 3 == 0 for c in Counter(groups[members]).values())  # whole groups only
        assert len(members) == 60                                          # equal group sizes -> fixed size
    C = np.array(stats.CLASSES)
    rng = np.random.default_rng(0)
    base_y = C[rng.integers(0, 4, 20)]
    base_ok = rng.random(20) < 0.7
    y = np.repeat(base_y, 3)
    p = np.where(np.repeat(base_ok, 3), y, np.roll(y, 1))
    img = stats.bootstrap_metric(y, p, n_boot=2000)
    grp = stats.bootstrap_metric(y, p, n_boot=2000, groups=groups)
    assert grp.estimate == img.estimate
    assert (grp.high - grp.low) > 1.3 * (img.high - img.low)  # copies are not independent evidence
    assert "group" in grp.method


def test_cluster_draws_match_slow_metric():
    C = np.array(stats.CLASSES)
    rng = np.random.default_rng(5)
    n = 50
    y = C[rng.integers(0, 4, n)]
    p = np.where(rng.random(n) < 0.6, y, C[rng.integers(0, 4, n)])
    groups = rng.integers(0, 20, n)
    rows, flat = stats.draw_resamples(n, groups, n_boot=100, seed=3)
    fast = stats.macro_f1_draws(y, p, rows, flat, 100)
    slow = np.array([stats.macro_f1(y[flat[rows == b]], p[flat[rows == b]]) for b in range(100)])
    assert np.allclose(fast, slow)


def test_regroup_full_spreads_copy_groups_across_splits():
    rows = []
    k = 0
    for src in range(300):
        cls = ["Angry", "Fear", "Happy", "Sad"][src % 4]
        for c in range(3 if src % 2 else 1):
            rows.append(_row(k, f"{cls}_{src}_1_jpg.rf.{k:08x}aa.jpg", "train", cls))
            k += 1
    res = t1_partition.build_t1(rows, strategy="regroup_full")
    m = res["manifest"]
    multi = {s: sum(1 for r in m if r["new_split"] == s and r["group_size"] > 1) for s in ("train", "valid", "test")}
    assert multi["valid"] > 0 and multi["test"] > 0       # the old largest-first routine gave 0 here
    assert res["report"]["ok"]
    b = res["report"]["balance"]
    assert b["checks"]["class_shares"] and b["checks"]["split_fractions"]
