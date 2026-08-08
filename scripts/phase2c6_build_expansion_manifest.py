#!/usr/bin/env python
"""Phase 2C.6 Stage 3: builds the real ~300-image blinded development
expansion manifest. Excludes every image_id already used by the Phase
2C.1 80-image pilot, and (structurally, via Phase 2B's own frozen
group-disjoint selection) never touches the locked-test split's images
either -- select_expansion_sample draws from the FULL non-conflicted
corpus, and the 7 locked-test images were themselves drawn from that same
pool by the original Phase 2B selection, so excluding "already used"
image_ids also excludes the locked-test 7 automatically (they are among
the 80 excluded). Writes private outputs (blinded images + mapping) under
outputs/, and a public, opaque-IDs-only summary under artifacts/phase2c6/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c6 import expansion_manifest as mod  # noqa: E402

N_IMAGES = 300


def main() -> None:
    prior_mapping = ROOT / "outputs/phase2c1/private_pilot_mapping.csv"
    used = mod.load_used_image_ids(prior_mapping)
    print(f"excluding {len(used)} image_ids already used by the Phase 2C.1 80-image pilot "
          f"(this includes the 7 locked-test images, which were part of that same original "
          f"80-image selection)")

    blinded = mod.build_expansion_manifest(
        N_IMAGES,
        exclude_image_ids=used,
        corpus_manifest_path=None,  # default: real Phase 7B partition manifest
        images_output_dir=ROOT / "outputs/phase2c6/expansion_images_private",
        mapping_output_path=ROOT / "outputs/phase2c6/expansion_pilot_mapping_private.csv",
    )
    print(f"blinded {len(blinded)} images -> outputs/phase2c6/expansion_images_private/")

    out_path = mod.write_expansion_summary(
        blinded, ROOT / "artifacts/phase2c6/expansion_manifest_summary.json",
        seed=mod.EXPANSION_SEED_STAGE_B, n_excluded_prior_round=len(used))
    print(f"wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
