"""Tests for M1 (transformation consistency) and the R1 controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_journal_stage_c import synthetic_drawing  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/journal/scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m1 = _load("m1_transformation_consistency")
r1 = _load("r1_representation_audit")


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    from PIL import Image
    img, _ = synthetic_drawing()
    d = tmp_path_factory.mktemp("m1")
    p = d / "ref.png"
    Image.fromarray(img).save(p)
    return d, img, m1.extract(p)


def test_roles_cover_every_extractor_feature(ref):
    _, _, (feats, _, _) = ref
    assert set(feats) == set(m1.FEATURE_ROLES)
    assert len(m1.FEATURE_ROLES) == 49


@pytest.mark.parametrize("t", m1.TRANSFORMS)
def test_registration_and_expected_values_under_each_dihedral_transform(ref, t):
    from PIL import Image
    d, img, (fa, ma, ra) = ref
    timg = m1.apply_transform(img, t)
    p = d / f"t_{'_'.join(map(str, map(int, t)))}.png"
    Image.fromarray(timg).save(p)
    fb, mb, rb = m1.extract(p)
    rec = m1.pair_record(fa, ma, ra, fb, mb, rb)
    reg = rec["registration"]
    if t == (False, False, False):
        assert reg["pair_class"] == "identical"
        return
    assert reg["pair_class"] == "geometric"
    assert tuple(reg["t"]) == t, reg
    assert rec["mask_iou"] > 0.99
    exp = m1.expected_features(fa, t)
    for f in m1.INVARIANT_GEOMETRIC + m1.EQUIVARIANT_GEOMETRIC:
        assert abs(exp[f] - fb[f]) <= 0.02 * max(1.0, abs(fb[f])), (f, exp[f], fb[f])


def test_photometric_change_goes_to_secondary_set(ref):
    _, img, (fa, ma, ra) = ref
    darker = np.clip(img.astype(int) - 25, 0, 255).astype(np.uint8)
    reg = m1.register(ra, m1.apply_transform(darker, (False, True, False)))
    assert reg["pair_class"] == "geometric+photometric" and reg["transform"] == "flip_horizontal"


def test_unrelated_image_is_unregistered(ref):
    _, img, (fa, ma, ra) = ref
    other, _ = synthetic_drawing(seed=1)
    other = np.roll(other, 90, axis=1)[:, ::-1].copy()
    other[:, :80] = 30
    assert m1.register(ra, other)["pair_class"] == "unregistered"


def test_analysis_verdicts_and_group_dependence():
    rng = np.random.default_rng(0)
    pairs = []
    for g in range(30):
        base = {f: float(rng.uniform(0.2, 0.8)) for f in m1.FEATURE_ROLES}
        for k in range(3):          # three copies of one drawing: dependent pairs, one resampling unit
            copy = dict(base)
            copy["foreground_fraction"] = base["foreground_fraction"] + rng.normal(0, 0.3)   # inconsistent
            pairs.append({"group": f"g{g}", "features_ref": base, "features_copy": copy, "mask_iou": 0.95,
                          "registration": {"pair_class": "geometric", "t": [False, False, False],
                                           "transform": "identity"}})
    rows, by_t, summ = m1.analyse(pairs, "primary")
    v = {r["feature"]: r for r in rows}
    assert v["foreground_fraction"]["verdict"] == "INCONSISTENT"
    assert v["bbox_fill_ratio"]["verdict"] == "CONSISTENT"
    assert v["bbox_fill_ratio"]["n_groups"] == 30 and v["bbox_fill_ratio"]["n_pairs"] == 90
    assert summ["n_groups"] == 30
    rows2, _, _ = m1.analyse(pairs, "secondary")
    assert rows2 == []
    few = [p for p in pairs if p["group"] in ("g0", "g1")]
    assert {r["verdict"] for r in m1.analyse(few, "primary")[0]} == {"TOO_FEW"}


def test_secondary_set_excludes_photometric_features():
    assert not set(m1.features_for_set("secondary")) & set(m1.INVARIANT_PHOTOMETRIC)
    assert set(m1.INVARIANT_PHOTOMETRIC) <= set(m1.features_for_set("primary"))


def test_r1_shuffle_keeps_group_structure_and_control_is_near_chance():
    rng = np.random.default_rng(0)
    groups = [f"g{i // 3}" for i in range(300)]
    fam = ["A" if int(g[1:]) % 2 else "B" for g in groups]
    sh = r1.shuffle_by_group(fam, groups, 1)
    for g in set(groups):
        assert len({s for s, gg in zip(sh, groups) if gg == g}) == 1
    X = np.array([[1.0 if f == "A" else -1.0] for f in fam]) + rng.normal(0, 0.1, (300, 1))
    assert r1.source_probe(X, fam, groups) > 0.95
    assert r1.shuffled_source_control(X, fam, groups)["mean"] < 0.75


def test_r1_image_stats(tmp_path):
    from PIL import Image
    img, _ = synthetic_drawing()
    p = tmp_path / "a.png"
    Image.fromarray(img).save(p)
    v = r1.image_stats(p)
    assert len(v) == len(r1.STAT_COLUMNS) and v[0] == 320 and v[1] == 240


def _fake_root(tmp_path):
    """Tiny T1 root: 12 drawings x 2-3 Roboflow copies (flips/rotations), plus test images."""
    import csv as _csv
    from PIL import Image
    root, data = tmp_path / "root", tmp_path / "data"
    (root / "T1_v2").mkdir(parents=True)
    data.mkdir()
    rows, k = [], 0
    classes = ["Angry", "Fear", "Happy", "Sad"]
    for g in range(12):
        img, _ = synthetic_drawing(seed=g)
        img = np.roll(img, 7 * g, axis=1)
        split = "test" if g >= 10 else ("valid" if g % 4 == 0 else "train")
        for j, t in enumerate([(False, False, False), (False, True, False), (True, True, False)][: 2 + g % 2]):
            base = f"{classes[g % 4].lower()}_{g}_1" if g % 2 else f"{classes[g % 4][0].lower()}{g}"
            fn = f"{base}_jpg.rf.{k:08x}abcd.png"
            Image.fromarray(m1.apply_transform(img, t)).save(data / fn)
            rows.append({"image_id": f"i{k:03d}", "class": classes[g % 4], "new_split": split,
                         "group_id": f"G{g}", "relative_path": fn, "path": fn})
            k += 1
    with open(root / "T1_v2" / "t1_partition_manifest.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root, data, rows


def test_m1_main_end_to_end_never_reads_test_split(tmp_path, monkeypatch):
    root, data, rows = _fake_root(tmp_path)
    test_files = {r["relative_path"] for r in rows if r["new_split"] == "test"}
    seen = []
    real = m1.extract
    monkeypatch.setattr(m1, "extract", lambda p: (seen.append(Path(p).name), real(p))[1])
    assert m1.main(["--root", str(root), "--dataset-root", str(data)]) == 0
    assert seen and not set(seen) & test_files
    out = root / "M1_transformation_consistency"
    import json as _json
    s = _json.loads((out / "M1_summary.json").read_text())
    assert s["pair_classes"].get("geometric", 0) >= 10 and s["inputs"]["splits_read"] == ["train", "valid"]
    assert (out / "T_m1_feature_consistency.csv").exists() and (out / "F8_transformation_consistency.png").exists()
    n_cached = len((out / "pairs.jsonl").read_text().splitlines())
    seen.clear()
    assert m1.main(["--root", str(root), "--dataset-root", str(data)]) == 0      # resume: nothing re-extracted
    assert seen == [] and len((out / "pairs.jsonl").read_text().splitlines()) == n_cached


def test_r1_simple_stats_baseline_on_fake_root(tmp_path):
    root, data, rows = _fake_root(tmp_path)
    from doar_journal.leakage import evaluation_groups, path_of
    from doar_journal.rescoring import source_family
    dev = {r["image_id"]: {"class": r["class"], "family": source_family(path_of(r))}
           for r in rows if r["new_split"] != "test"}
    out = tmp_path / "r1"
    out.mkdir()
    res = r1.simple_stats_baseline(data, {r["image_id"]: r for r in rows}, dev, evaluation_groups(rows), out)
    assert res["n_images"] == len(dev) and (out / "simple_image_stats.csv").exists()
    assert 0 <= res["emotion_probe_macro_f1"] <= 1
