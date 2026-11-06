"""M1 — transformation consistency of the explicit visual measurements (CPU, no labels, no models).

  python experiments/journal/scripts/m1_transformation_consistency.py --root <Drive>/DOAR_JOURNAL_2026 \
      --dataset-root <Drive>/.../Datasets/Combined_Drawing [--t1 T1_v2] [--limit-groups N]

Exploratory, added after B4; every rule is fixed in M1_TRANSFORMATION_CONSISTENCY_PLAN.md before the
first real run. The archived E2-A0 extractor is run UNCHANGED on pairs of digital copies of the same
drawing (same Roboflow filename stem) from T1 train+valid; the locked test split is never read.
The geometric relation of each pair is estimated among the 8 dihedral transforms; invariant
features are compared directly, equivariant ones after the exact coordinate mapping.
This measures numerical repeatability under digital transforms, NOT accuracy and NOT real
test-retest. Resumable: per-pair results are cached in pairs.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import _frozen_e2_features as fe  # noqa: E402
from doar_journal import measurement  # noqa: E402
from doar_journal.leakage import path_of, source_key  # noqa: E402
from doar_journal.provenance import run_metadata, sha256_file, write_json  # noqa: E402
from doar_journal.stats import icc_2_1  # noqa: E402

# ------------------------------------------------------------------ fixed rules (plan)
REG_MIN_CORR = 0.95
PHOTO_MAX_MEAN_DIFF = 2.0
PHOTO_STD_RATIO = (0.97, 1.03)
TAU = 0.10
VERDICT_LEVEL = 0.90
MIN_GROUPS = 10
N_BOOT = 2000
SEED = 20261106

EXCLUDED = ["canvas_width_px", "canvas_height_px", "canvas_aspect_ratio", "foreground_pixel_count"]
INVARIANT_GEOMETRIC = ["foreground_fraction", "bbox_area_ratio", "bbox_fill_ratio", "centroid_distance_from_center",
                       "occupied_grid_fraction_4x4", "connected_component_count", "largest_component_fraction",
                       "median_component_fraction", "thickness_proxy_mean_ratio", "thickness_proxy_max_ratio"]
EQUIVARIANT_GEOMETRIC = ["bbox_x_min_ratio", "bbox_y_min_ratio", "bbox_x_max_ratio", "bbox_y_max_ratio",
                         "bbox_width_ratio", "bbox_height_ratio", "left_margin_ratio", "right_margin_ratio",
                         "top_margin_ratio", "bottom_margin_ratio", "centroid_x_ratio", "centroid_y_ratio",
                         "foreground_x_std_ratio", "foreground_y_std_ratio", "ink_left_fraction",
                         "ink_right_fraction", "ink_top_fraction", "ink_bottom_fraction",
                         "quadrant_top_left_fraction", "quadrant_top_right_fraction",
                         "quadrant_bottom_left_fraction", "quadrant_bottom_right_fraction",
                         "vertical_symmetry_iou", "horizontal_symmetry_iou"]
INVARIANT_PHOTOMETRIC = ["foreground_mean_gray", "foreground_std_gray", "foreground_mean_darkness",
                         "foreground_gray_entropy", "foreground_mean_saturation", "foreground_std_saturation",
                         "foreground_hue_entropy", "foreground_colorfulness", "edge_density_canvas",
                         "edge_density_foreground_bbox", "edge_to_foreground_ratio"]
FEATURE_ROLES = {**{f: "excluded" for f in EXCLUDED}, **{f: "invariant_geometric" for f in INVARIANT_GEOMETRIC},
                 **{f: "equivariant_geometric" for f in EQUIVARIANT_GEOMETRIC},
                 **{f: "invariant_photometric" for f in INVARIANT_PHOTOMETRIC}}

SWAP_PAIRS = [("bbox_x_min_ratio", "bbox_y_min_ratio"), ("bbox_x_max_ratio", "bbox_y_max_ratio"),
              ("bbox_width_ratio", "bbox_height_ratio"), ("left_margin_ratio", "top_margin_ratio"),
              ("right_margin_ratio", "bottom_margin_ratio"), ("centroid_x_ratio", "centroid_y_ratio"),
              ("foreground_x_std_ratio", "foreground_y_std_ratio"), ("ink_left_fraction", "ink_top_fraction"),
              ("ink_right_fraction", "ink_bottom_fraction"),
              ("quadrant_top_right_fraction", "quadrant_bottom_left_fraction"),
              ("vertical_symmetry_iou", "horizontal_symmetry_iou")]
FLIP = {"x": {"min": "bbox_x_min_ratio", "max": "bbox_x_max_ratio", "m1": "left_margin_ratio",
              "m2": "right_margin_ratio", "c": "centroid_x_ratio", "i1": "ink_left_fraction",
              "i2": "ink_right_fraction",
              "q": [("quadrant_top_left_fraction", "quadrant_top_right_fraction"),
                    ("quadrant_bottom_left_fraction", "quadrant_bottom_right_fraction")]},
        "y": {"min": "bbox_y_min_ratio", "max": "bbox_y_max_ratio", "m1": "top_margin_ratio",
              "m2": "bottom_margin_ratio", "c": "centroid_y_ratio", "i1": "ink_top_fraction",
              "i2": "ink_bottom_fraction",
              "q": [("quadrant_top_left_fraction", "quadrant_bottom_left_fraction"),
                    ("quadrant_top_right_fraction", "quadrant_bottom_right_fraction")]}}
TRANSFORMS = [(s, x, y) for s in (False, True) for x in (False, True) for y in (False, True)]
TRANSFORM_NAMES = {(False, False, False): "identity", (False, True, False): "flip_horizontal",
                   (False, False, True): "flip_vertical", (False, True, True): "rotate_180",
                   (True, False, False): "transpose", (True, True, False): "rotate_90_cw",
                   (True, False, True): "rotate_90_ccw", (True, True, True): "anti_transpose"}


# ------------------------------------------------------------------ geometry
def apply_transform(img: np.ndarray, t) -> np.ndarray:
    swap, fx, fy = t
    if swap:
        img = np.swapaxes(img, 0, 1)
    if fx:
        img = img[:, ::-1]
    if fy:
        img = img[::-1]
    return np.ascontiguousarray(img)


def expected_features(fa: dict[str, float], t) -> dict[str, float]:
    """Expected feature values of the copy, from the reference, under transform t (swap, then flips)."""
    swap, fx, fy = t
    e = dict(fa)
    if swap:
        for a, b in SWAP_PAIRS:
            e[a], e[b] = fa[b], fa[a]
    for axis, on in (("x", fx), ("y", fy)):
        if not on:
            continue
        m = FLIP[axis]
        cur = dict(e)
        e[m["min"]], e[m["max"]] = 1 - cur[m["max"]], 1 - cur[m["min"]]
        e[m["m1"]], e[m["m2"]] = cur[m["m2"]], cur[m["m1"]]
        e[m["c"]] = 1 - cur[m["c"]]
        e[m["i1"]], e[m["i2"]] = cur[m["i2"]], cur[m["i1"]]
        for a, b in m["q"]:
            e[a], e[b] = cur[b], cur[a]
    return e


def _small(img: np.ndarray, size) -> np.ndarray:
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def register(a: np.ndarray, b: np.ndarray) -> dict:
    """Best dihedral transform mapping reference a onto copy b, and the pair class."""
    import cv2
    if a.shape == b.shape and np.array_equal(a, b):
        return {"transform": TRANSFORM_NAMES[(False, False, False)], "t": [False, False, False],
                "corr": 1.0, "pair_class": "identical"}

    def thumb(x):
        h, w = x.shape[:2]
        s = 256 / max(h, w)
        return cv2.resize(x, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else x
    a_s, b_s = thumb(a), thumb(b)
    gb = _small(cv2.cvtColor(b_s, cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float64).ravel()
    gb = (gb - gb.mean()) / (gb.std() + 1e-9)
    best = None
    for t in TRANSFORMS:
        ga = _small(cv2.cvtColor(apply_transform(a_s, t), cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float64).ravel()
        ga = (ga - ga.mean()) / (ga.std() + 1e-9)
        c = float(np.mean(ga * gb))
        if best is None or c > best[1]:
            best = (t, c)
    t, c = best
    pa = _small(apply_transform(a_s, t), (256, 256)).reshape(-1, 3).astype(np.float64)
    pb = _small(b_s, (256, 256)).reshape(-1, 3).astype(np.float64)
    mean_diff = float(np.max(np.abs(pa.mean(0) - pb.mean(0))))
    ratio = (pb.std(0) + 1e-9) / (pa.std(0) + 1e-9)
    photo_equal = mean_diff <= PHOTO_MAX_MEAN_DIFF and bool(np.all((ratio >= PHOTO_STD_RATIO[0]) & (ratio <= PHOTO_STD_RATIO[1])))
    cls = "unregistered" if c < REG_MIN_CORR else ("geometric" if photo_equal else "geometric+photometric")
    return {"transform": TRANSFORM_NAMES[t], "t": list(t), "corr": round(c, 5), "pair_class": cls,
            "rgb_mean_max_diff": round(mean_diff, 3), "rgb_std_ratio": [round(float(r), 4) for r in ratio],
            "size_a": list(a.shape[:2]), "size_b": list(b.shape[:2])}


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, t) -> float:
    import cv2
    ma = apply_transform(mask_a.astype(np.uint8), t)
    if ma.shape != mask_b.shape:
        ma = cv2.resize(ma, (mask_b.shape[1], mask_b.shape[0]), interpolation=cv2.INTER_NEAREST)
    a, b = ma > 0, mask_b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def extract(path: Path) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    from PIL import Image
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    result, mask = fe.extract_features({"path": str(path), "image_id": "", "split": "", "true_class": "",
                                        "relative_path": ""})
    feats = measurement.numeric_features(result)
    unknown = set(feats) - set(FEATURE_ROLES)
    if unknown:
        raise RuntimeError(f"extractor returned features without a pre-registered role: {sorted(unknown)}")
    return feats, mask, rgb


def pair_record(fa, ma, ra, fb, mb, rb) -> dict:
    reg = register(ra, rb)
    rec = {"registration": reg, "features_ref": fa, "features_copy": fb}
    if reg["pair_class"] in ("geometric", "geometric+photometric"):
        rec["mask_iou"] = mask_iou(ma, mb, tuple(reg["t"]))
    return rec


# ------------------------------------------------------------------ analysis
def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def group_ci(shares: np.ndarray, n_boot=N_BOOT, seed=SEED) -> tuple[float, float]:
    if len(shares) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(shares), (n_boot, len(shares)))
    m = shares[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def verdict(n_groups, lo, hi) -> str:
    if n_groups < MIN_GROUPS:
        return "TOO_FEW"
    if lo >= VERDICT_LEVEL:
        return "CONSISTENT"
    if hi < VERDICT_LEVEL:
        return "INCONSISTENT"
    return "INCONCLUSIVE"


def features_for_set(set_name: str) -> list[str]:
    feats = INVARIANT_GEOMETRIC + EQUIVARIANT_GEOMETRIC
    if set_name == "primary":
        feats = feats + INVARIANT_PHOTOMETRIC
    return feats


def analyse(pairs: list[dict], set_name: str) -> tuple[list[dict], list[dict], dict]:
    """pairs: dicts with group, registration, features_ref, features_copy, mask_iou."""
    classes = {"primary": ["geometric"], "secondary": ["geometric+photometric"]}[set_name]
    ps = [p for p in pairs if p["registration"]["pair_class"] in classes]
    rows, by_t = [], []
    if not ps:
        return rows, by_t, {"n_pairs": 0}
    exp = [expected_features(p["features_ref"], tuple(p["registration"]["t"])) for p in ps]
    pair_err = defaultdict(list)
    for f in features_for_set(set_name):
        ref_vals = np.array([p["features_ref"].get(f, np.nan) for p in ps], float)
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        q75, q25 = (np.percentile(ref_vals, [75, 25]) if len(ref_vals) else (np.nan, np.nan))
        scale = q75 - q25
        if not scale > 0:
            scale = float(np.std(ref_vals)) if len(ref_vals) else 0.0
        if not scale > 0:
            scale = 1.0
        per_group = defaultdict(list)
        per_group_t = defaultdict(lambda: defaultdict(list))
        abs_e, nrm_e, both_missing, xy = [], [], 0, []
        for p, e in zip(ps, exp):
            ev, ov = e.get(f), p["features_copy"].get(f)
            if _nan(ev) and _nan(ov):
                both_missing += 1
                continue
            if _nan(ev) or _nan(ov):
                ok = False
            else:
                d = abs(ov - ev)
                abs_e.append(d)
                nrm_e.append(d / scale)
                xy.append((ev, ov))
                ok = d / scale <= TAU
                if f in INVARIANT_GEOMETRIC:
                    pair_err[id(p)].append(d / scale)
            per_group[p["group"]].append(ok)
            per_group_t[p["registration"]["transform"]][p["group"]].append(ok)
        shares = np.array([np.mean(v) for v in per_group.values()], float)
        lo, hi = group_ci(shares)
        icc = icc_2_1(np.array(xy)) if len(xy) >= 3 else None
        rows.append({"set": set_name, "feature": f, "role": FEATURE_ROLES[f], "n_pairs": int(sum(map(len, per_group.values()))),
                     "n_groups": len(shares), "n_both_missing": both_missing,
                     "consistency_rate": round(float(shares.mean()), 4) if len(shares) else None,
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4), "verdict": verdict(len(shares), lo, hi),
                     "median_abs_error": round(float(np.median(abs_e)), 5) if abs_e else None,
                     "median_norm_error": round(float(np.median(nrm_e)), 4) if nrm_e else None,
                     "p90_norm_error": round(float(np.percentile(nrm_e, 90)), 4) if nrm_e else None,
                     "iqr_scale": round(float(scale), 6),
                     "icc_2_1": round(float(icc), 4) if icc is not None and np.isfinite(icc) else None})
        for tname, g in per_group_t.items():
            if len(g) >= MIN_GROUPS:
                sh = np.array([np.mean(v) for v in g.values()], float)
                by_t.append({"set": set_name, "feature": f, "transform": tname, "n_groups": len(sh),
                             "consistency_rate": round(float(sh.mean()), 4)})
    ious = defaultdict(list)
    for p in ps:
        if not _nan(p.get("mask_iou")):
            ious[p["group"]].append(p["mask_iou"])
    g_med = np.array([np.median(v) for v in ious.values()], float)
    g_hi = np.array([np.mean(np.array(v) >= 0.90) for v in ious.values()], float)
    from scipy.stats import spearmanr
    xs = [(p["mask_iou"], float(np.mean(pair_err[id(p)]))) for p in ps
          if not _nan(p.get("mask_iou")) and pair_err.get(id(p))]
    rho = spearmanr(*zip(*xs)) if len(xs) >= 10 and len({x for x, _ in xs}) > 1 and len({y for _, y in xs}) > 1 else None
    lo, hi = group_ci(g_hi)
    summary = {"n_pairs": len(ps), "n_groups": len({p["group"] for p in ps}),
               "transforms": dict(Counter(p["registration"]["transform"] for p in ps)),
               "mask_iou_group_median": round(float(np.median(g_med)), 4) if len(g_med) else None,
               "mask_iou_share_ge_0_90": round(float(g_hi.mean()), 4) if len(g_hi) else None,
               "mask_iou_share_ge_0_90_ci": [round(lo, 4), round(hi, 4)],
               "spearman_mask_iou_vs_geometric_error": ({"rho": round(float(rho[0]), 3), "p": float(rho[1]), "n": len(xs)}
                                                        if rho is not None else None),
               "verdicts": dict(Counter(r["verdict"] for r in rows)),
               "verdicts_by_role": {role: dict(Counter(r["verdict"] for r in rows if r["role"] == role))
                                    for role in ("invariant_geometric", "equivariant_geometric", "invariant_photometric")}}
    return rows, by_t, summary


# ------------------------------------------------------------------ outputs
def write_csv(p: Path, rows: list[dict]):
    if not rows:
        return
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figure(out: Path, rows: list[dict], pairs: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted([r for r in rows if r["set"] == "primary" and r["consistency_rate"] is not None],
                  key=lambda r: r["consistency_rate"])
    if not rows:
        return
    col = {"CONSISTENT": "#2a78d6", "INCONCLUSIVE": "#9a9893", "INCONSISTENT": "#d0482c", "TOO_FEW": "#c9c7c2"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 0.17 * len(rows) + 1.2),
                                 gridspec_kw={"width_ratios": [1, 1.6]})
    iou = [p["mask_iou"] for p in pairs if p["registration"]["pair_class"] == "geometric" and not _nan(p.get("mask_iou"))]
    a1.hist(iou, bins=np.linspace(0, 1, 21), color="#2a78d6", edgecolor="white")
    a1.set_xlabel("A0 mask IoU between aligned copies")
    a1.set_ylabel("pairs")
    a1.set_title("a  mask agreement", loc="left", fontsize=9)
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        a2.plot([r["ci_low"], r["ci_high"]], [i, i], color=col[r["verdict"]], lw=1.4)
        a2.scatter([r["consistency_rate"]], [i], color=col[r["verdict"]], s=14, zorder=3)
    a2.axvline(VERDICT_LEVEL, color="#52514e", lw=0.8, ls="--")
    a2.set_yticks(y)
    a2.set_yticklabels([r["feature"] for r in rows], fontsize=6)
    a2.set_xlim(0, 1.02)
    a2.set_xlabel(f"share of copy pairs within {TAU:.2f} IQR (95% CI)")
    a2.set_title("b  per-feature consistency (primary set)", loc="left", fontsize=9)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "F8_transformation_consistency.pdf", bbox_inches="tight")
    fig.savefig(out / "F8_transformation_consistency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--t1", default="T1_v2")
    ap.add_argument("--limit-groups", type=int, default=None, help="smoke test only")
    a = ap.parse_args(argv)
    root, data = Path(a.root), Path(a.dataset_root)
    manifest = root / a.t1 / "t1_partition_manifest.csv"
    with open(manifest, newline="", encoding="utf-8") as f:
        man = [r for r in csv.DictReader(f) if r["new_split"] in ("train", "valid")]
    stems = defaultdict(list)
    for r in man:
        k = source_key(path_of(r)).rf_stem
        if k:
            stems[k].append(r)
    groups = {k: sorted(v, key=lambda r: r["image_id"]) for k, v in sorted(stems.items()) if len(v) >= 2}
    if a.limit_groups:
        groups = dict(list(groups.items())[:a.limit_groups])
    out = root / ("M1_transformation_consistency" + ("_SMOKE" if a.limit_groups else ""))
    out.mkdir(parents=True, exist_ok=True)
    cache_p = out / "pairs.jsonl"
    done = {}
    if cache_p.exists():
        for line in cache_p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                done[(x["ref"], x["copy"])] = x
    n_pairs = sum(len(v) - 1 for v in groups.values())
    print(f"{len(groups)} copy groups, {n_pairs} pairs (T1 train+valid); {len(done)} already cached", flush=True)
    with open(cache_p, "a", encoding="utf-8") as fout:
        for gi, (stem, members) in enumerate(groups.items()):
            ref = members[0]
            todo = [m for m in members[1:] if (ref["image_id"], m["image_id"]) not in done]
            if not todo:
                continue
            fa, ma, ra = extract(data / path_of(ref).replace("\\", "/"))
            for m in todo:
                fb, mb, rb = extract(data / path_of(m).replace("\\", "/"))
                rec = {"ref": ref["image_id"], "copy": m["image_id"], "group": stem, **pair_record(fa, ma, ra, fb, mb, rb)}
                fout.write(json.dumps(rec) + "\n")
                done[(rec["ref"], rec["copy"])] = rec
            fout.flush()
            if gi % 50 == 0:
                print(f"  {gi + 1}/{len(groups)} groups", flush=True)
    pairs = [done[(g[0]["image_id"], m["image_id"])] for g in groups.values() for m in g[1:]]
    cls = Counter(p["registration"]["pair_class"] for p in pairs)
    print("pair classes:", dict(cls))
    write_csv(out / "T_m1_pairs.csv", [{"group": p["group"], "ref": p["ref"], "copy": p["copy"],
                                        **{k: v for k, v in p["registration"].items() if k != "t"},
                                        "mask_iou": p.get("mask_iou")} for p in pairs])
    all_rows, all_t, summ = [], [], {}
    for s in ("primary", "secondary"):
        rows, by_t, sm = analyse(pairs, s)
        all_rows += rows
        all_t += by_t
        summ[s] = sm
        print(f"{s}: {sm.get('n_pairs')} pairs / {sm.get('n_groups', 0)} groups; verdicts {sm.get('verdicts')}; "
              f"mask IoU group median {sm.get('mask_iou_group_median')}, share >= 0.90 {sm.get('mask_iou_share_ge_0_90')}")
    write_csv(out / "T_m1_feature_consistency.csv", all_rows)
    write_csv(out / "T_m1_by_transform.csv", all_t)
    for r in all_rows:
        if r["set"] == "primary":
            print(f"  {r['feature']:32s} {r['role']:22s} rate {r['consistency_rate']} "
                  f"[{r['ci_low']}, {r['ci_high']}] {r['verdict']}")
    figure(out, all_rows, pairs)
    write_json(out / "M1_summary.json", {
        "status": "exploratory, post hoc (after B4); rules fixed in M1_TRANSFORMATION_CONSISTENCY_PLAN.md",
        "claim_scope": "numerical repeatability under digital transforms; NOT accuracy, NOT real test-retest",
        "rules": {"registration_min_corr": REG_MIN_CORR, "photometric_max_mean_diff": PHOTO_MAX_MEAN_DIFF,
                  "photometric_std_ratio": PHOTO_STD_RATIO, "tau_iqr": TAU, "verdict_level": VERDICT_LEVEL,
                  "min_groups": MIN_GROUPS, "n_boot": N_BOOT, "seed": SEED},
        "feature_roles": FEATURE_ROLES, "pair_classes": dict(cls), "n_groups": len(groups), "n_pairs": len(pairs),
        "results": summ, "inputs": {"t1_manifest_sha256": sha256_file(manifest), "splits_read": ["train", "valid"]},
        "metadata": run_metadata("M1_transformation_consistency", vars(a))})
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
