"""Stage A: dataset-integrity checks for the Combined_Drawing benchmark.

Hypothesis under test (PREREGISTRATION.md, C1): files exported by Roboflow as
``<stem>.rf.<hash>.<ext>`` that share ``<stem>`` are copies (often augmented:
rotated, flipped, noised, cropped) of one source drawing. T0 grouped images
only when their dHash Hamming distance was <= 2, which misses flipped or
rotated copies, so such copies may sit in different T0 splits.

Evidence levels, kept separate everywhere in the outputs:
  * ``rf_stem``      same Roboflow source stem (filename evidence only)
  * ``name_cross``   a plain filename that matches a Roboflow stem after
                     undoing Roboflow's '.'->'_' renaming (weaker evidence)
  * ``dihedral``     dHash distance <= threshold under one of the 8
                     rotations/flips (image evidence)
  * ``embedding``    cosine similarity of frozen CLIP embeddings >= threshold
                     (image evidence; catches crops and noise)
Only image evidence or a confirmed human pair review can upgrade filename
evidence to "confirmed copy".
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np

_RF_RE = re.compile(r"^(?P<stem>.+?)\.rf\.(?P<hash>[0-9a-fA-F]{8,})\.(?P<ext>[A-Za-z0-9]+)$")
_RF_EXT_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp)$", re.IGNORECASE)
SPLITS = ("train", "valid", "test")


def basename(path: str) -> str:
    """Filename for both POSIX and Windows-style stored paths."""
    return PureWindowsPath(path).name if "\\" in path else Path(path).name


@dataclass(frozen=True)
class SourceKey:
    filename: str
    is_roboflow: bool
    rf_stem: str | None       # exact Roboflow stem, e.g. 'Happy_1_69_jpg'
    name_key: str             # case-folded original name without extension, e.g. 'happy_1_69'


def source_key(path: str) -> SourceKey:
    fn = basename(path)
    m = _RF_RE.match(fn)
    if m:
        stem = m.group("stem")
        original = _RF_EXT_SUFFIX.sub("", stem)  # Roboflow writes 'name.jpg' as 'name_jpg'
        return SourceKey(fn, True, stem, original.casefold())
    return SourceKey(fn, False, None, Path(fn).stem.casefold())


def load_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_of(row: dict) -> str:
    """T0 partition manifests use `new_split`; the committed E1 manifests use `split`."""
    return row.get("new_split") or row.get("split") or ""


def path_of(row: dict) -> str:
    return row.get("relative_path") or row.get("path") or ""


def stem_audit(rows: list[dict]) -> dict:
    """Filename-level audit. Reads only filenames, split and class columns."""
    keys = {r["image_id"]: source_key(path_of(r)) for r in rows}
    by_rf = defaultdict(list)
    for r in rows:
        k = keys[r["image_id"]]
        if k.is_roboflow:
            by_rf[k.rf_stem].append(r)

    group_sizes = Counter(len(v) for v in by_rf.values())
    span = Counter()
    class_conflicts = 0
    group_rows = []
    for stem, members in sorted(by_rf.items()):
        splits = sorted({split_of(m) for m in members})
        classes = sorted({m["class"] for m in members})
        if len(splits) > 1:
            span["-".join(splits)] += 1
        if len(classes) > 1:
            class_conflicts += 1
        group_rows.append({"rf_stem": stem, "size": len(members), "splits": ";".join(splits),
                           "classes": ";".join(classes),
                           "image_ids": ";".join(sorted(m["image_id"] for m in members))})

    # Per-split exposure: images whose rf stem also occurs in another split.
    stem_splits = {s: {split_of(m) for m in v} for s, v in by_rf.items()}
    exposure = {}
    for sp in SPLITS:
        sub = [r for r in rows if split_of(r) == sp]
        if not sub:
            continue
        shared_train = [r for r in sub if keys[r["image_id"]].is_roboflow
                        and "train" in stem_splits[keys[r["image_id"]].rf_stem] and sp != "train"]
        exposure[sp] = {"n": len(sub), "share_rf_stem_with_train": len(shared_train),
                        "fraction": (len(shared_train) / len(sub)) if sp != "train" else None,
                        "by_class": dict(Counter(r["class"] for r in shared_train))}

    # Weaker evidence: plain filenames that equal a Roboflow source name.
    rf_names = defaultdict(set)
    for r in rows:
        k = keys[r["image_id"]]
        if k.is_roboflow:
            rf_names[k.name_key].add(split_of(r))
    name_cross = [r for r in rows if not keys[r["image_id"]].is_roboflow
                  and keys[r["image_id"]].name_key in rf_names]
    name_cross_split = sum(1 for r in name_cross
                           if rf_names[keys[r["image_id"]].name_key] - {split_of(r)})

    return {
        "n_images": len(rows),
        "n_roboflow_named": sum(k.is_roboflow for k in keys.values()),
        "n_rf_stems": len(by_rf),
        "rf_group_size_distribution": dict(sorted(group_sizes.items())),
        "rf_stems_spanning_splits": dict(sorted(span.items())),
        "rf_stems_spanning_splits_total": sum(span.values()),
        "rf_stem_groups_with_class_conflict": class_conflicts,
        "exposure_by_split": exposure,
        "plain_names_matching_rf_source": len(name_cross),
        "plain_names_matching_rf_source_in_other_split": name_cross_split,
        "group_rows": group_rows,
    }


def rf_stem_edges(rows: list[dict]) -> list[dict]:
    """Star edges (min image_id -> each other member) per Roboflow stem group."""
    by_rf = defaultdict(list)
    for r in rows:
        k = source_key(path_of(r))
        if k.is_roboflow:
            by_rf[k.rf_stem].append(r["image_id"])
    edges = []
    for stem, ids in sorted(by_rf.items()):
        ids = sorted(ids)
        edges += [{"a": ids[0], "b": o, "evidence": "rf_stem", "detail": stem} for o in ids[1:]]
    return edges


# ---------------------------------------------------------------------------
# Image evidence: rotation/flip-invariant difference hash
# ---------------------------------------------------------------------------

def _dhash_bits(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    from PIL import Image
    small = np.asarray(Image.fromarray(gray).resize((hash_size + 1, hash_size), Image.BILINEAR),
                       dtype=np.float64)
    return (small[:, 1:] > small[:, :-1]).flatten()


def dihedral_dhashes(path: str | Path, hash_size: int = 8) -> list[int]:
    """dHash of the 8 dihedral transforms (4 rotations x optional mirror),
    as integers. Index 0 is the untransformed image (T0's own dHash)."""
    from PIL import Image
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"))
    out = []
    for flip in (False, True):
        base = np.fliplr(g) if flip else g
        for k in range(4):
            bits = _dhash_bits(np.ascontiguousarray(np.rot90(base, k)), hash_size)
            out.append(int("".join("1" if b else "0" for b in bits), 2))
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised popcount of uint64 arrays (numpy-version independent)."""
    b = np.ascontiguousarray(x, dtype=np.uint64).view(np.uint8).reshape(*x.shape, 8)
    return _POP8[b].sum(axis=-1)


def dihedral_near_pairs(hashes: dict[str, list[int]], threshold: int = 6) -> list[dict]:
    """Pairs whose canonical hash (index 0) is within `threshold` of ANY of the
    other image's 8 transforms. O(n^2) over ~3.7k images is ~7M comparisons x 8,
    vectorised per row; fine on Colab CPU."""
    ids = sorted(hashes)
    canon = np.array([hashes[i][0] for i in ids], dtype=np.uint64)
    allt = np.array([hashes[i] for i in ids], dtype=np.uint64)  # n x 8
    pairs = []
    for a in range(len(ids)):
        x = np.bitwise_xor(allt[a + 1:], canon[a])  # (n-a-1) x 8
        dist = popcount64(x)
        best = dist.min(axis=1)
        which = dist.argmin(axis=1)
        for j in np.nonzero(best <= threshold)[0]:
            pairs.append({"a": ids[a], "b": ids[a + 1 + j], "evidence": "dihedral",
                          "detail": f"hamming={int(best[j])};transform={int(which[j])}"})
    return pairs


def embedding_near_pairs(ids: list[str], emb: np.ndarray, threshold: float) -> list[dict]:
    """Pairs with cosine similarity >= threshold (embeddings L2-normalised here)."""
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = e @ e.T
    iu = np.triu_indices(len(ids), k=1)
    hit = sims[iu] >= threshold
    return [{"a": ids[i], "b": ids[j], "evidence": "embedding", "detail": f"cos={sims[i, j]:.4f}"}
            for i, j in zip(iu[0][hit], iu[1][hit])]


# ---------------------------------------------------------------------------
# X0: blinded pair review materials
# ---------------------------------------------------------------------------

def sample_review_pairs(rows: list[dict], n_pairs: int = 40, n_controls: int = 20,
                        split_pair: tuple[str, str] = ("train", "valid"), seed: int = 20261021,
                        evidence: str = "rf_stem") -> list[dict]:
    """Same-source pairs across `split_pair` plus random different-source
    controls from the same two splits and same class (so class cannot give the
    answer). `evidence` = "rf_stem" (Roboflow stem) or "name_key" (identical
    plain filenames, weaker evidence). Order is shuffled and the pair type is
    withheld from the review sheet."""
    if evidence not in ("rf_stem", "name_key"):
        raise ValueError(evidence)
    rng = random.Random(seed)
    by_stem = defaultdict(list)
    for r in rows:
        k = source_key(path_of(r))
        if split_of(r) not in split_pair:
            continue
        if evidence == "rf_stem" and k.is_roboflow:
            by_stem[k.rf_stem].append(r)
        elif evidence == "name_key" and not k.is_roboflow:
            by_stem[k.name_key].append(r)
    candidates = []
    for stem in sorted(by_stem):
        m = by_stem[stem]
        a = sorted([r for r in m if split_of(r) == split_pair[0]], key=lambda r: r["image_id"])
        b = sorted([r for r in m if split_of(r) == split_pair[1]], key=lambda r: r["image_id"])
        if a and b:
            candidates.append((a[0], b[0]))
    rng.shuffle(candidates)
    same = candidates[:n_pairs]

    pool_a = sorted([r for r in rows if split_of(r) == split_pair[0]], key=lambda r: r["image_id"])
    pool_b = sorted([r for r in rows if split_of(r) == split_pair[1]], key=lambda r: r["image_id"])
    controls = []
    tries = 0
    while len(controls) < n_controls and tries < 100_000:
        tries += 1
        a, b = rng.choice(pool_a), rng.choice(pool_b)
        ka, kb = source_key(path_of(a)), source_key(path_of(b))
        if a["class"] == b["class"] and ka.name_key != kb.name_key:
            controls.append((a, b))
    items = [(f"same_{evidence}", a, b) for a, b in same] + [("control", a, b) for a, b in controls]
    rng.shuffle(items)
    out = []
    for i, (kind, a, b) in enumerate(items, start=1):
        out.append({"pair_id": f"P{i:03d}", "pair_type_hidden": kind,
                    "image_id_a": a["image_id"], "path_a": path_of(a), "split_a": split_of(a),
                    "image_id_b": b["image_id"], "path_b": path_of(b), "split_b": split_of(b),
                    "class": a["class"]})
    return out


REVIEW_VERDICTS = ("SAME_DRAWING", "DIFFERENT_DRAWING", "UNSURE")
REVIEW_TRANSFORMS = ("none", "rotation", "flip", "crop_or_zoom", "noise", "brightness_or_colour",
                     "multiple", "not_applicable")


def write_review_template(pairs: list[dict], path: str | Path) -> Path:
    """Blank reviewer CSV: no pair type, no filenames, only pair ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "verdict", "transform_seen", "notes", "reviewer"])
        w.writeheader()
        for p in pairs:
            w.writerow({"pair_id": p["pair_id"], "verdict": "", "transform_seen": "", "notes": "",
                        "reviewer": ""})
    return path


def make_pair_sheets(pairs: list[dict], dataset_root: str | Path, out_pdf: str | Path,
                     per_page: int = 6, thumb: int = 360) -> Path:
    """Side-by-side pair sheets labelled only with pair ids (blinded)."""
    from PIL import Image, ImageDraw
    root = Path(dataset_root)
    pages = []
    for start in range(0, len(pairs), per_page):
        chunk = pairs[start:start + per_page]
        page = Image.new("RGB", (2 * thumb + 60, per_page * (thumb + 40) + 20), "white")
        d = ImageDraw.Draw(page)
        for row, p in enumerate(chunk):
            y = 20 + row * (thumb + 40)
            d.text((10, y - 16), p["pair_id"], fill="black")
            for col, key in enumerate(("path_a", "path_b")):
                with Image.open(root / p[key].replace("\\", "/")) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb, thumb))
                    page.paste(im, (20 + col * (thumb + 20), y))
        pages.append(page)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:])
    return out_pdf


def score_review(pairs: list[dict], review_csv: str | Path) -> dict:
    """Join completed verdicts with the hidden pair types (after review)."""
    from .stats import wilson
    with open(review_csv, newline="", encoding="utf-8") as f:
        verdicts = {r["pair_id"]: r for r in csv.DictReader(f)}
    missing = [p["pair_id"] for p in pairs if verdicts.get(p["pair_id"], {}).get("verdict") not in REVIEW_VERDICTS]
    if missing:
        raise ValueError(f"review incomplete or invalid verdicts for: {missing}")
    res = {}
    for kind in sorted({p["pair_type_hidden"] for p in pairs}):
        sub = [verdicts[p["pair_id"]] for p in pairs if p["pair_type_hidden"] == kind]
        k = sum(v["verdict"] == "SAME_DRAWING" for v in sub)
        res[kind] = {"n": len(sub), "judged_same": k, "unsure": sum(v["verdict"] == "UNSURE" for v in sub),
                     "rate_same": wilson(k, len(sub)).as_dict(),
                     "transforms": dict(Counter(v["transform_seen"] for v in sub
                                                if v["verdict"] == "SAME_DRAWING"))}
    return res


def stem_overlap(ids_a: list[str], ids_b: list[str], rows: list[dict]) -> list[dict]:
    """Images of set B whose Roboflow stem or name key also occurs in set A
    (used for the 84-image development set vs the 80-image holdout)."""
    by_id = {r["image_id"]: r for r in rows}
    ka = {source_key(path_of(by_id[i])) for i in ids_a if i in by_id}
    stems_a = {k.rf_stem for k in ka if k.rf_stem}
    names_a = {k.name_key for k in ka}
    out = []
    for i in ids_b:
        if i not in by_id:
            out.append({"image_id": i, "overlap": "not_in_manifest"})
            continue
        k = source_key(path_of(by_id[i]))
        if k.rf_stem and k.rf_stem in stems_a:
            out.append({"image_id": i, "overlap": "rf_stem", "key": k.rf_stem})
        elif k.name_key in names_a:
            out.append({"image_id": i, "overlap": "name_key", "key": k.name_key})
    return out
