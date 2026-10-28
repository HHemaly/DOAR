"""A4 — image-level near-duplicate evidence over ALL images (Colab; CPU ok, GPU faster).

Two scans, each producing CANDIDATE edges that are only used for T1 after the
human review step confirms a sample of them (thresholds below are fixed in
PREREGISTRATION.md before the scan is run):

  dihedral   dHash distance <= 6 under any of 8 rotations/flips
  embedding  cosine >= 0.95 between frozen CLIP ViT-B/32 image embeddings
             (open_clip ViT-B-32-quickgelu, pretrained OpenAI weights, no fine-tuned checkpoint is used)

The scan is label-blind: it never reads class labels or any model prediction,
and it does not use the test images for anything but duplicate detection.

  python experiments/journal/scripts/a4_image_evidence.py \
      --t0-manifest <Drive>/DOAR_FROZEN_TO/final_partition/partition_manifest.csv \
      --dataset-root /content/drive/MyDrive/Masters/Datasets/Combined_Drawing \
      --out <Drive>/DOAR_JOURNAL_2026/A4_image_evidence [--skip-embedding]
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
from doar_journal.leakage import (dihedral_dhashes, dihedral_near_pairs, embedding_near_pairs,  # noqa: E402
                                  load_manifest, path_of, split_of)

DIHEDRAL_THRESHOLD = 6
EMBEDDING_THRESHOLD = 0.95
# Same open_clip config as E1 (e1a2_common.py): OpenAI weights need QuickGELU.
CLIP_MODEL = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"


def resolve(dataset_root: Path, row: dict) -> Path:
    rel = (row.get("relative_path") or "").replace("\\", "/")
    if rel:
        return dataset_root / rel
    # committed manifests store absolute Windows paths: keep the last three parts split/class/file
    parts = row["path"].replace("\\", "/").split("/")
    return dataset_root.joinpath(*parts[-3:])


def clip_embeddings(paths: list[Path], device: str, batch: int = 64) -> np.ndarray:
    import open_clip
    import torch
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = []
            for p in paths[i:i + batch]:
                with Image.open(p) as im:
                    ims.append(preprocess(im.convert("RGB")))
            out.append(model.encode_image(torch.stack(ims).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def write_edges(path: Path, edges: list[dict], split: dict[str, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "evidence", "detail", "split_a", "split_b", "cross_split"])
        w.writeheader()
        for e in edges:
            w.writerow({**e, "split_a": split[e["a"]], "split_b": split[e["b"]],
                        "cross_split": split[e["a"]] != split[e["b"]]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-manifest", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-embedding", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    rows = load_manifest(args.t0_manifest)
    root = Path(args.dataset_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = {r["image_id"]: split_of(r) for r in rows}

    cache = out / "dihedral_hashes.json"
    hashes = json.loads(cache.read_text()) if cache.exists() else {}
    missing = []
    for r in rows:
        if r["image_id"] in hashes:
            continue
        p = resolve(root, r)
        if not p.exists():
            missing.append(str(p))
            continue
        hashes[r["image_id"]] = dihedral_dhashes(p)
        if len(hashes) % 250 == 0:
            provenance.write_json(cache, hashes)
            print(f"hashed {len(hashes)}/{len(rows)} images", flush=True)
    provenance.write_json(cache, hashes)
    if missing:
        print(f"WARNING: {len(missing)} files not found, e.g. {missing[:3]}")
    dih = dihedral_near_pairs(hashes, threshold=DIHEDRAL_THRESHOLD)
    write_edges(out / "dihedral_edges_candidates.csv", dih, split)
    print(f"dihedral candidate pairs: {len(dih)}; cross-split: {sum(split[e['a']] != split[e['b']] for e in dih)}")

    outputs = [out / "dihedral_edges_candidates.csv", cache]
    if not args.skip_embedding:
        ecache = out / f"clip_{CLIP_MODEL}_{CLIP_PRETRAINED}_embeddings.npz"  # name encodes the exact model
        ids = sorted(i for i in hashes)
        if ecache.exists():
            z = np.load(ecache, allow_pickle=False)
            assert list(z["ids"]) == ids, "embedding cache was built for a different image list"
            emb = z["emb"]
        else:
            print(f"computing CLIP embeddings for {len(ids)} images on {args.device} ...", flush=True)
            emb = clip_embeddings([resolve(root, {r["image_id"]: r for r in rows}[i]) for i in ids], args.device)
            np.savez(ecache, ids=np.array(ids), emb=emb)
        emb_pairs = embedding_near_pairs(ids, emb, EMBEDDING_THRESHOLD)
        write_edges(out / "embedding_edges_candidates.csv", emb_pairs, split)
        print(f"embedding candidate pairs: {len(emb_pairs)}; cross-split: "
              f"{sum(split[e['a']] != split[e['b']] for e in emb_pairs)}")
        outputs += [out / "embedding_edges_candidates.csv", ecache]

    meta = provenance.run_metadata("A4_image_evidence", {"dihedral_threshold": DIHEDRAL_THRESHOLD,
                                   "embedding_threshold": EMBEDDING_THRESHOLD,
                                   "skip_embedding": args.skip_embedding,
                                   "clip_model": CLIP_MODEL, "clip_pretrained": CLIP_PRETRAINED},
                                   inputs={"t0_manifest": args.t0_manifest})
    meta["n_files_missing"] = len(missing)
    provenance.mark_done(out, meta, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
