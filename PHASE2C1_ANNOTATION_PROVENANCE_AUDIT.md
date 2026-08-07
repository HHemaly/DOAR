# Phase 2C.1 — Annotation Provenance and Reproducibility Audit

**Status: pre-implementation audit, presented before any annotation was performed, per
instruction.** Branch `feature/doar-phase2c-annotation-expansion`, starting commit
`a0ea8e8`. This document answers the required question: *can Phase 2B's exact original
80-image blinded pilot be reconstructed from the real dataset now available on this
machine?* All checks below were run directly against the real dataset and the real
repository code — not asserted from memory or from prior-session documentation alone.

**Conclusion: YES. The exact original 80-image pilot, including the exact 20 already-
annotated `pilot_id -> image_id` assignments, was independently reconstructed and
verified byte-exact against `artifacts/phase2b/annotation_manifest.csv` and
`artifacts/phase2b/duplicate_groups.csv` (both committed, unmodified). Stage D may
proceed.**

---

## 1. What Phase 2B's selection actually depends on (read from source, not assumed)

`src/doar/phase2b/dataset.py::select_pilot_sample(n, seed=2026)` calls
`duplicate_groups.py::load_group_lookup()`, which reads
`outputs/phase7b/final_partition/partition_manifest.csv` — a **gitignored, local-only**
file. It does not exist anywhere in this repository checkout (`outputs/` currently
contains only `experiments/` and `new_device_validation/`; `outputs/phase7b/` is
absent). This file must be **regenerated**, not found, for reproduction to be possible
at all.

That manifest is itself the output of `main.py build-partition`
(`src/doar/partition.py::run_partition_design`), which:

1. Reads a base manifest built by `dataset.py::build_manifest(dataset_root, ...)`.
   Each row's `image_id = sha256(str(relative_path_within_dataset_root)).hexdigest()[:16]`
   — a hash of the **relative path only** (split/class/filename), not of file content or
   absolute path. This means `image_id` is reproducible regardless of where the dataset
   is unzipped, as long as the internal `split/class/filename` structure is unchanged.
2. Computes a `dhash` (difference-hash) column per image (Phase 7B's evidence-selected
   near-duplicate method) and unions images into duplicate groups via union-find over
   (a) exact SHA-256 equality and (b) `dhash` Hamming distance <= threshold, computed
   pairwise across the **entire** 3,688-image dataset.
3. Assigns `group_id = grp_{idx:05d}`, where `idx` is the position of each group when
   all groups are sorted by their own minimum `image_id` — deterministic given the same
   full set of `image_id`/`dhash` values, but sensitive to the *global* structure of the
   dataset (a single missing or renamed file anywhere could, in principle, shift indices).
4. Flags each group's `conflict_status` (`clean`/`conflict`) via
   `assess_group_label_conflicts` (near-dup-aware, broader than the original exact-only
   check).

Per `PHASE7B_DUPLICATE_POLICY.md` §17, this specific partition (`dhash`, threshold=6,
single-linkage) was **never formally approved or locked** (`APPROVED_POLICY.json` /
`FROZEN.json` do not exist) — Phase 2B's own code comment
(`duplicate_groups.py:16-18`) states it reused this file anyway as "the most recent real
duplicate-group computation on disk," not because it was approved. Reproducing Phase 2B
therefore means reproducing *this specific provisional artifact*, not a since-finalized
one — there is no more-final version to target instead.

## 2. Real dataset vs. expected layout

`C:\Users\ZZ01G7865\Downloads\DOAR\Combined_Drawing-20260807T152850Z-1-001\Combined_Drawing`
contains `train/`, `valid/`, `test/`, each with `{Angry, Fear, Happy, Sad}` — matches the
expected layout. Per-split/class file counts:

| Split | Angry | Fear | Happy | Sad | Total |
|---|---|---|---|---|---|
| train | 641 | 502 | 921 | 757 | 2,821 |
| valid | 77 | 45 | 110 | 78 | 310 |
| test | 123 | 130 | 168 | 136 | 557 |
| **Total** | | | | | **3,688** |

**3,688 matches exactly** the total recorded in `PHASE3A_RESULTS.md` and
`PHASE7A_DATASET_READINESS.md` (independently, before any reconstruction was attempted).
0 unreadable images, also matching `PHASE3A_RESULTS.md`'s "0 unreadable" finding,
reconfirmed directly this session by re-running `build_manifest()` against the real
files (`outputs/phase2c1_repro/manifest.csv`, gitignored, local-only, not committed).

**One anomaly, noted but not used**: the dataset also contains a 4th top-level folder,
`all/{Angry,Fear,Happy,Sad}` (3,397 files total), outside the `train/valid/test`
structure the task specifies. It is not `train + valid + test` combined (that would be
3,688) and its provenance is unclear. **It was not read, hashed, or used anywhere in
this audit or in any Phase 2C.1 tooling** — only `train/`, `valid/`, `test/` are ever
touched, per the task's explicit expected layout.

## 3. Reconstruction procedure and verification (executed, not simulated)

Three independent checks were run, each directly against the real files:

### 3.1 Filename/path identity (weakest, run first)

For each of the 20 `image_id` values in the committed
`artifacts/phase2b/annotation_manifest.csv`, the current dataset was scanned end-to-end
(3,688 files, `train/valid/test` only) computing
`sha256(relative_path).hexdigest()[:16]` exactly as `build_manifest()` does, without
touching file content. **20/20 matched** — e.g. `9c23df48180ecb10` ->
`train\Happy\Happy_2_126_jpg.rf.1a51648333e5ff9bd43a09acdbeaeb44.jpg`. A 16-hex-character
truncated SHA-256 collision is cryptographically negligible, so 20/20 matches is
effectively conclusive that the current dataset's internal file layout (split/class/
filename) is identical to whatever was used when Phase 2B's annotation manifest was
produced.

### 3.2 Duplicate-group structure (content-dependent, stronger)

`main.py build-manifest` was run against the real dataset (full SHA-256 + aHash per
image), then `main.py build-partition --hash-field dhash --near-dup-threshold 6 --seed
42` (Phase 7B's exact documented configuration) was run against that manifest,
producing a freshly regenerated `outputs/phase2c1_repro/final_partition/
partition_manifest.csv`. This step depends on actual pixel content (via `dhash`), not
just filenames.

All 20 `image_id -> group_id` / `group_size` pairs recorded in the committed
`artifacts/phase2b/duplicate_groups.csv` were checked against this regenerated
manifest: **20/20 matched exactly** (group_id string, group size, and
`conflict_status == "clean"` for every one). Example:
`5945f0106f5305b6 -> grp_01074, size 5` in both the original committed artifact and this
session's independent regeneration.

### 3.3 End-to-end pilot-selection reproduction (strongest — the actual deliverable)

`select_pilot_sample(80, seed=2026)` (the real, unmodified Phase 2B production function)
was called against the regenerated `partition_manifest.csv`, followed by the same
`pilot_id` assignment shuffle `blind_copy_sample` uses (`random.Random(seed + 1)`,
without actually copying any file). The resulting `pilot_id -> image_id` mapping for
`p2b_0000` through `p2b_0019` was compared against the 20 rows of the committed
`artifacts/phase2b/annotation_manifest.csv`:

**20/20 matched exactly, with zero mismatches.** This means the full, real production
code path — group-disjoint selection from the clean pool, deterministic shuffle order,
and opaque `pilot_id` assignment — reproduces identically on the dataset now available
on this machine.

## 4. What was and wasn't verified

- **Verified**: the 20 already-annotated images' identity, their duplicate-group
  membership, and their `pilot_id` assignment all reproduce exactly.
- **Verified**: the full 80-image selection (`select_pilot_sample(80, seed=2026)`) runs
  successfully end-to-end against the real, current dataset and returns 80 group-disjoint,
  non-conflicted images — the same function, same seed, same clean pool used originally.
- **Not independently verified**: the remaining 60 of the 80 pilot images' specific
  identities against an external ground truth (none exists — they were never annotated
  or separately recorded anywhere in Phase 2B's committed artifacts, by design). Their
  reproducibility rests on the same mechanism just verified for the other 20 (same
  function call, same input data, same seed) — there is no *additional* uncertainty
  specific to those 60, but they could not be checked against an independent record
  because no such record exists.
- **Minor, inconsequential discrepancy found**: the regenerated partition's raw
  train/valid/test **split** totals differ from `PHASE7B_DUPLICATE_POLICY.md` §8's
  table by 2 images (test 506 here vs. 504 documented; train 2,562 vs. 2,564) — both
  in Angry and Sad. This is very likely because the original run supplied
  `--exposed-image-ids` (4 specific images from Phase 6, forced into `train` to keep
  their duplicate groups out of `valid`/`test`) and this session's reconstruction did
  not, since the source documents only record 3 of the 4 original filenames precisely
  enough to recompute their `image_id` unambiguously (`Fear_1_10*` and `3-4_jpg.rf*`
  each matched two candidate files across splits). **This does not affect Phase 2C.1**:
  `select_pilot_sample` reads only `group_id`/`conflict_status`/`path`, none of which
  depend on `exposed_image_ids` — confirmed by the exact 20/20 match in §3.3 above,
  which used this same reconstructed manifest.

## 5. Answers to the specific questions posed

- **Source image identity**: verified (§3.1) — 20/20 exact match.
- **Duplicate-group identity**: verified (§3.2) — 20/20 exact match, content-dependent.
- **`pilot_id` assignment**: verified (§3.3) — 20/20 exact match, full production code
  path.
- **First 20 pilot IDs mapped back to the downloaded dataset**: yes, exactly, per
  above.
- **Remaining 60 reconstructable exactly**: yes, by the same verified mechanism — see
  the "not independently verified" caveat in §4 for what that claim does and doesn't
  cover.
- **No emotion label exposed in the annotator-facing working directory**: not yet
  applicable at audit time (no working directory existed yet) — enforced structurally in
  Stage D below (opaque `p2b_XXXX` filenames only, blind-copied via the unmodified
  `blind_copy_sample`, which writes no class/split metadata into the output directory).

## 6. Decision

**Reproducibility confirmed. Stage D (private blinded working directory) may proceed
using the exact original 80-image selection.** No different 80-image sample was created.
The 20 already-annotated images retain their original `pilot_id`s; the remaining 60
begin unannotated, exactly as `CURRENT_TO_TARGET_GAP_V7.md` §"Gaps that remain" item 4
describes.

Regenerated, gitignored, local-only artifacts backing this audit:
`outputs/phase2c1_repro/manifest.csv`, `outputs/phase2c1_repro/final_partition/*` — kept
for this session's reference, not committed (consistent with the existing
`outputs/phase7a`, `outputs/phase7b` convention), reproducible again from this document's
exact commands if needed.
