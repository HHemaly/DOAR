# Phase 7A — Leakage-Safe Dataset Readiness and Partition Design

**Status: dataset analysis and split construction only. No model was
trained or evaluated in this phase.** The original dataset was never
overwritten, cleaned, or modified — every operation reads the existing
manifest and image files and writes new, separate manifest files only
(spot-checked directly, see §7). Existing Phase 3–7 artifacts, splits, and
claims are unchanged and are treated here as preliminary development
evidence, not as final results.

---

## 1. Scope and what this phase distinguishes

This report distinguishes five things that must not be conflated:

1. **The original, contaminated train/valid/test split** used by Phase
   3A–6 (2,821 / 310 / 557 images) — unchanged, still on disk, still
   referenced by `outputs/phase3a`–`outputs/phase6`.
2. **Preliminary Phase 3–6 results** computed on that contaminated split —
   still valid as preliminary, development-time evidence; not retracted,
   not recomputed here.
3. **The new duplicate-controlled development split** produced in this
   phase (`outputs/phase7a/partition_manifest.csv`, `new_split` in
   `{train, valid}`) — recommended for future Family A–E development work.
4. **The newly locked final test split** produced in this phase
   (`new_split == "test"`, 455 images) — has been constructed and
   verified programmatically only; **not inspected, not predicted on, not
   reported on** anywhere in this document or its artifacts.
5. **The remaining subject-level leakage limitation** — structurally
   unresolvable with this dataset (§6).

---

## 2. Verifying the existing leakage findings

### 2.1 Method and thresholds, traced to source (not trusted from memory)

Read directly from `src/doar/dataset.py` and `src/doar/leakage.py` this
phase:

- **Exact-duplicate detection**: SHA-256 of the raw file bytes
  (`hashlib.sha256(raw).hexdigest()`, `dataset.py` line 53). This is a
  byte-exact hash — it catches only literal copies (identical files), and
  is broken by *any* change to the file bytes, including re-saving through
  a different encoder at identical visual quality.
- **Near-duplicate detection**: a dependency-light 64-bit average hash
  (aHash): resize to grayscale 8×8 (64 pixels), threshold each pixel
  against the image's own mean brightness, pack the 64 bits into a hex
  string (`dataset.py::_average_hash`). Distance is Hamming distance
  (population count of the XOR) between two 64-bit hashes.
  **Threshold: ≤5 bits differing (out of 64)** counts as a near-duplicate
  (`NEAR_DUP_THRESHOLD = 5`, `dataset.py` line 17). The only justification
  recorded in the source is a one-line comment ("a conservative near-dup")
  — **no empirical justification for the value 5 specifically exists
  anywhere in the codebase or prior phase documents.** §2.3 below supplies
  the empirical justification that was missing, by testing the threshold's
  actual behavior on this dataset.
- **Degenerate-hash exclusion**: images whose aHash is all-zero or
  all-`f` (flat/near-blank pages) are excluded from near-duplicate
  comparison, because aHash cannot reliably distinguish two different
  blank pages. This is a real, already-implemented false-positive guard,
  confirmed present and unchanged.
- **Label conflicts (existing implementation)**: `leakage.py::assess_leakage`
  only flags a label conflict when two images share the **same SHA-256**
  (exact duplicate) but different class labels. It does **not** check
  whether near-duplicate pairs (same content, different file bytes) carry
  conflicting labels — a real gap, addressed in §4 below.

### 2.2 Independent reproduction of the recorded counts

`assess_leakage()` was re-run this phase, directly, against
`outputs/phase5/manifest.csv` (confirmed byte-identical to Phase 3A/4's own
manifest in Phase 5), and compared to the counts already recorded in
`outputs/phase5/deep/leakage_gate/leakage_report.json`:

| Metric | Reproduced this phase | Previously recorded | Match |
|---|---|---|---|
| Exact cross-split duplicate groups | 324 | 324 | **Yes** |
| Near-duplicate cross-split pairs | 1,442 | 1,442 | **Yes** |
| Conflicting-label groups (exact-only) | 48 | 48 | **Yes** |
| Total flagged image IDs | 1,517 | 1,517 | **Yes** |

**All four counts reproduce exactly.** The previously-cited "41%" figure
(1,517/3,688) is confirmed, not merely restated. Full reproduction detail:
`outputs/phase7a/leakage_reproduction_check.json`.

**Manifest integrity spot-check**: 25 randomly sampled images (seed 42) and
the 4 test-split images already disclosed as non-blind in
`PHASE6_RESULTS.md` §2 were independently re-hashed (both SHA-256 and
aHash) directly from disk this phase. **0 mismatches, 0 missing files** —
the manifest's recorded hashes are trustworthy and current; the underlying
image files have not drifted since the manifest was built.

### 2.3 Threshold sensitivity — this is new evidence, not previously assessed

The existing codebase never assessed whether `near_dup_threshold=5` is a
good choice, beyond a one-line comment. This phase computed the full
duplicate-group structure (§3) at 8 threshold values to characterize the
actual failure mode:

| Threshold | Groups | Near-dup edges | Largest group | Groups >10 members |
|---|---|---|---|---|
| 0 (identical aHash only) | 2,481 | 1,796 | 6 | 0 |
| 1 | 2,368 | 2,071 | 8 | 0 |
| 2 | 2,324 | 2,219 | 9 | 0 |
| 3 | 2,270 | 2,372 | 22 | 4 |
| 4 | 2,206 | 2,593 | 63 | 6 |
| **5 (current project default)** | **2,106** | **3,069** | **137** | **8** |
| 6 | 1,964 | 4,076 | 265 | 13 |
| 8 | 1,597 | 8,650 | **1,104** | 5 |

**This is a clear, quantitative demonstration of single-linkage "chaining"**:
below threshold 3, the largest cluster stays under 10 images; at the
existing default of 5, one single cluster already absorbs 137 images; at 8,
one cluster absorbs 1,104 images — nearly a third of the entire dataset.
Full data: `outputs/phase7a/threshold_sensitivity.json`.

**Composition check of the 15 largest groups at threshold=5**
(`outputs/phase7a/large_group_composition.json`): **13 of the 15 largest
groups span 2–4 different emotion classes.** The single largest (137
images) spans all 4 classes. This is strong evidence that large clusters at
this threshold are predominantly **chaining artifacts** (many simple,
low-visual-entropy children's drawings — mostly white background, a
centered figure — coincidentally landing within 5 bits of each other in a
coarse 8×8 thumbnail) rather than genuine repeated content. Only 2 of the
top 15 groups are single-class (both "Happy", sizes 9 and 8) and could
plausibly be genuine near-duplicate bursts, though this was not visually
confirmed (see §2.5).

**Consequence for the false-positive/false-negative assessment below**:
because every multi-class large group is, by construction, already
excluded from supervised training as an unresolved label conflict (§4),
threshold=5's chaining behavior does **not** create a leakage risk in the
new partition (§5) — but it does cost real training data: **290 of the 674
excluded-conflict images (43.0%) come from just the 10 largest groups**,
most of which are very likely false-positive "duplicates" rather than
genuine repeats.

**Decision: this phase keeps `near_dup_threshold=5` as the primary,
project-consistent analysis** (§2.2's exact reproduction depends on it, and
changing it would not be "verifying the existing findings" but replacing
them). **Recommendation, not a silent change**: a future phase should
consider re-running this analysis at threshold=2–3, which the sensitivity
table shows would essentially eliminate chaining (largest group ≤9) at the
cost of a smaller near-duplicate recall (fewer true near-duplicates
caught). This trade-off is stated for a future decision, not resolved here.

### 2.4 Sensitivity to real-world transformations

None of resizing, recompression, cropping, rotation, or color/contrast
adjustment is handled uniformly by the existing methods:

| Transformation | Exact (SHA-256) | Near-dup (aHash, threshold 5) |
|---|---|---|
| Re-save at different quality/format | Broken (any byte change) | **Usually robust** — aHash is computed on a heavily downsampled 8×8 grayscale thumbnail, largely insensitive to compression artifacts |
| Resize | Broken | **Usually robust** — aHash resizes to 8×8 regardless of source size |
| Crop | Broken | **Not robust** — cropping shifts which content maps to which 8×8 cell; a cropped duplicate will likely NOT be caught |
| Rotation (90°/180°/mirroring) | Broken | **Not robust at all** — aHash has no rotation or flip invariance; a rotated duplicate is very unlikely to fall within 5 bits of its original |
| Brightness/contrast/color shift | Broken | **Partially robust** — aHash thresholds against the image's own mean, so uniform brightness shifts are tolerated; color changes that alter the *relative* light/dark pattern are not |

**This means: rotated, cropped, or otherwise re-composed duplicates of an
image already in the dataset would likely be MISSED by both methods** (a
false negative) — a real, stated limitation, not silently assumed away.

### 2.5 Likely false positives and false negatives (summary)

- **False positives (near-dup)**: dominant risk is **chaining** among
  visually simple, low-entropy drawings with similar overall light/dark
  layout but different actual content — quantified in §2.3. Secondary risk:
  coincidental structural similarity between two different children's
  drawings of the same generic composition (centered figure, white
  background).
- **False negatives (near-dup)**: rotated, cropped, mirrored, or
  significantly recolored duplicates of the same underlying drawing — not
  quantified (would require ground-truth pairs to measure recall, which
  does not exist for this dataset), stated as an open limitation.
- **False positives (exact)**: essentially none — SHA-256 collisions are
  cryptographically negligible.
- **False negatives (exact)**: any re-encoded copy of an original image
  (even pixel-identical after decode) — not caught by SHA-256, only
  possibly caught by the near-dup path.
- **No manual visual audit of any candidate pair was performed this
  phase** — recommended future work, not done here, given the volume
  (3,069 near-dup edges at threshold 5) and the explicit instruction that
  this phase is automated dataset analysis, not a visual review project.

---

## 3. Effective dataset size

### 3.1 Duplicate-group computation (new capability this phase)

`src/doar/leakage.py::assess_leakage()` only ever compares **cross-split**
pairs (its job is to detect violations of the *current* split assignment).
Building a **new** partition from scratch requires the full duplicate
structure across the **entire** dataset — including duplicates that
already happen to sit in the same split. `src/doar/partition.py` (new
module, this phase) implements this via union-find over the union of (a)
exact SHA-256 equality and (b) aHash Hamming distance ≤ threshold, computed
pairwise across all 3,688 images regardless of split.

| Quantity | Count |
|---|---|
| Total images | 3,688 |
| Total duplicate groups (threshold=5) | 2,106 |
| Exact-match edges | 688 |
| Near-dup edges | 3,069 |
| Singleton groups (no duplicate) | 1,376 |
| Groups with 2+ members | 730 |
| Largest group | 137 (see §2.3) |

### 3.2 Label conflicts (broader than the existing check)

`assess_group_label_conflicts()` (new this phase) flags a group as an
unresolved conflict if **any** of its members (exact- or near-duplicate
alike) carry different class labels — strictly broader than
`assess_leakage`'s exact-only check.

| | Existing exact-only check | New group-level check (this phase) |
|---|---|---|
| Conflict groups | 48 | **124** |
| Conflict images | *(not separately reported)* | **674 (18.3% of the dataset)** |

The 76 additional conflict groups (124 − 48) are all near-duplicate-based
conflicts the existing project code has never detected, because it never
checked near-duplicate pairs for label agreement.

### 3.3 Before / after, per class

| Class | Raw images (original, all splits) | Effective clean groups | Effective clean images | Conflict (excluded) images |
|---|---|---|---|---|
| Angry | 841 | 439 | 647 | 194 |
| Fear | 677 | 326 | 518 | 159 |
| Happy | 1,199 | 658 | 1,042 | 157 |
| Sad | 971 | 559 | 807 | 164 |
| **Total** | **3,688** | **1,982** | **3,014** | **674** |

Fear remains the smallest class by every measure (raw images, clean
groups, clean images) — consistent with every prior phase's finding that
Fear is the hardest class.

Full machine-readable detail: `outputs/phase7a/effective_counts.json`.

---

## 4. Duplicate-group and label-conflict policy

### 4.1 Transitive closure (how overlapping groups are prevented from leaking)

Duplicate relationships are transitively closed via union-find: if image A
is a near-duplicate of B, and B is a near-duplicate of C, then A, B, and C
are placed in **one** group even if A and C are not a direct match. This is
the same "single-linkage" logic responsible for the chaining behavior
documented in §2.3 — **it is a deliberate, conservative choice**: for
leakage-safety purposes, under-grouping (missing a real link, risking a
duplicate crossing partitions) is worse than over-grouping (merging two
images that are not truly duplicates, which only costs some data, never
correctness). The trade-off is real and is exactly what §2.3 quantifies —
it is not hidden.

### 4.2 Label-conflict policy

1. **Original files are never touched.** Every image, including every
   member of every conflict group, remains at its original path,
   unmodified.
2. **Conflicts are flagged, not resolved.** `assess_group_label_conflicts()`
   records every group with >1 class present; no label is changed, chosen,
   or silently dropped.
3. **A dedicated review manifest exists**:
   `outputs/phase7a/label_conflict_review.csv` — one row per conflicting
   image (674 rows), with `image_id`, `path`, `original_split`,
   `assigned_class`, `group_id`, `group_classes_present`, `group_size`, and
   `recommendation` (currently `exclude_pending_human_review` for every
   row — no automatic resolution is proposed here).
4. **Unresolved conflicts are structurally excluded from supervised
   experiments** — not merely flagged with a boolean a training loop could
   ignore. Every conflicting image's `new_split` is set to the distinct
   value `"excluded_conflict"` (not `"train"`/`"valid"`/`"test"` with a
   flag), so a loader that filters on split name alone cannot accidentally
   include them. `include_in_supervised_training` is additionally set to
   `False` for defense in depth.
5. **Complete audit trail**: `outputs/phase7a/duplicate_groups.csv` (every
   group, every member), `duplicate_group_edges_exact.csv` /
   `duplicate_group_edges_near.csv` (the raw pairwise evidence — which
   two images, which hash, which Hamming distance — that produced every
   union), and `label_conflicts.csv` (group-level conflict summary) provide
   full provenance for every decision.

---

## 5. New partition construction

### 5.1 Algorithm

`build_group_disjoint_partition()` (new, `src/doar/partition.py`):

1. Every duplicate group is treated as one atomic unit — it can never be
   split across train/valid/test.
2. Conflict groups are assigned `"excluded_conflict"` unconditionally.
3. Clean groups are bucketed by class, then processed **largest-group-first
   within each class** (after a seeded shuffle, for deterministic
   tie-breaking) and greedily assigned to whichever of train/valid/test has
   the largest remaining deficit against its target share for that class
   (a standard largest-remainder bin-packing approach). Target shares
   default to this dataset's own existing proportions (2,821/310/557 of
   3,688 ⇒ 76.51%/8.41%/15.11%).
4. **Determinism**: given the same manifest, seed, and threshold, this
   always produces byte-identical output — verified by running
   `main.py build-partition` twice in **separate Python processes** (so
   Python's per-process string-hash randomization is actually exercised,
   not just re-using in-memory state) and confirming SHA-256-identical
   output for every manifest file.

### 5.2 Seed and configuration used

`outputs/phase7a/partition_config.json`: `seed=42`, `near_dup_threshold=5`,
`split_ratios=[0.7651, 0.0841, 0.1511]` (this dataset's own existing
proportions), `algorithm="largest_remainder_greedy_bin_packing_per_class"`.

### 5.3 Resulting split sizes

| Split | Images | Independent groups | Angry | Fear | Happy | Sad |
|---|---|---|---|---|---|---|
| **train** | 2,305 | **1,273** | 495 | 396 | 797 | 617 |
| **valid** | 254 | 254 | 54 | 44 | 88 | 68 |
| **test (newly locked)** | 455 | 455 | 98 | 78 | 157 | 122 |
| excluded_conflict | 674 | 124 | 194 | 159 | 157 | 164 |
| **Total** | **3,688** | **2,106** | 841 | 677 | 1,199 | 971 |

**Notable property, not designed in but observed and worth stating
plainly**: valid and test ended up containing **only singleton groups**
(every one of their 254 and 455 images is its own independent group — 0
internal duplication). This is a direct, explicable consequence of the
largest-group-first greedy algorithm: train's per-class target count is
5–10× larger than valid's or test's, so train's deficit dominates whenever
a multi-member group is being placed, and by the time smaller groups are
reached, train has typically already been filled enough that ties resolve
elsewhere — in practice, every single multi-member clean group (up to size
9) landed in train. This is a **genuine improvement in evaluation
validity** (valid/test contain zero within-split duplication) at the cost
of **train's true effective sample size being smaller than its raw image
count suggests**: 2,305 raw training images collapse to **1,273
independent groups** (667 singletons + 606 multi-image groups of size 2–9).
**1,273, not 2,305 or 2,821, is the number that should inform any future
overfitting/sample-size judgment about training on this data** (see the
Phase 7 feasibility reassessment, §8).

### 5.4 Programmatic verification (all required checks, all passing)

| Check | Result |
|---|---|
| No duplicate group crosses a partition | **PASS** |
| No image assigned to more than one partition | **PASS** |
| No unresolved label conflict enters a supervised split | **PASS** |
| Split/class count tally is internally consistent | **PASS** |
| Manifest paths resolve to real, on-disk files | **PASS** (spot-checked; every `path` value is the original, unmoved file location) |
| Deterministic reproduction (separate process, same seed) | **PASS** — byte-identical SHA-256 for every output file |
| The 4 already-disclosed non-blind smoke-test images (`PHASE6_RESULTS.md` §2) do not land in the new locked test split | **PASS** (see below) |

**Cross-check against the Phase 6 disclosure**: the 4 images already
disclosed as non-blind (their labels and two checkpoints' predictions were
inspected in both Phase 3A and Phase 6) were traced through the new
partition:

| Image | New split | Group | Group size |
|---|---|---|---|
| `2-1_jpg.rf...` (Angry) | `excluded_conflict` | `grp_00022` | 137 (the large chaining-suspicious group, §2.3) |
| `Fear_1_10_jpg.rf...` (Fear) | `train` | `grp_01684` | 1 (singleton) |
| `Happy_1_19_jpg.rf...` (Happy) | `valid` | `grp_01028` | 1 (singleton) |
| `3-4_jpg.rf...` (Sad) | `train` | `grp_00114` | 2 |

**None of the 4 disclosed images land in the new test split** — this
satisfies the requirement (`SESSION_HANDOFF.md` §8 item 7) that the
eventual locked test set exclude them, though the algorithm did not target
this specifically; it is a consequence of how few images each of their
groups contains combined with where the greedy allocator had remaining
deficit when it processed them. This should be treated as **verified, not
assumed** — restated here because it was checked directly against the
actual output, not inferred.

Full verification report: `outputs/phase7a/leakage_verification_report.json`.

A genuine bug was found and fixed during this work: `compute_effective_counts()`
originally iterated a raw Python `set` of conflict-group IDs directly,
whose iteration order depends on per-process string-hash randomization —
this made `effective_counts.json`'s byte content (though never its
underlying numeric values) non-reproducible across separate process runs.
Fixed by sorting before every iteration/dict construction in that function;
a regression test (`EffectiveCountsDeterminismTests`) now asserts every
dict-valued field is key-sorted. Caught by running `main.py build-partition`
twice and diffing output hashes — exactly the kind of check this phase's
own required verification asked for.

---

## 6. Validity boundary — stated honestly

**This partition is "image-group-disjoint and duplicate-controlled." It is
NOT "subject-independent" and must never be described that way.**

- **What it controls**: no exact or near-duplicate (aHash, threshold 5)
  image — and, transitively, no chain of such duplicates — can appear in
  more than one of train/valid/test. No image with an unresolved
  cross-duplicate label conflict can enter a supervised split.
- **What it cannot control**: this dataset has **no child/subject
  identifier column** (`subject_grouping_available: false` in every
  leakage report checked, including this phase's reproduction, §2.2). Two
  *different, non-duplicate* photographs of two drawings by the **same
  child** — which would be a genuine subject-level leakage risk for any
  claim of generalizing to *new children* — cannot be detected or
  prevented by this or any other method available in this dataset. If
  subject identifiers become available in a future data-collection round,
  subject-level grouping should be layered on top of this image-level
  partition retroactively; until then, every claim made from this
  partition must be scoped to **image-level**, not **subject-level**,
  independence.

---

## 7. Original dataset preservation — verified, not assumed

- `git status` in the repository shows no change to any file outside this
  phase's own new module/tests/docs (see §9).
- The 4 already-disclosed non-blind test images
  (`PHASE6_RESULTS.md` §2) and 25 additional randomly sampled images were
  re-hashed directly from disk this phase (SHA-256 and aHash) and found to
  match the manifest's recorded values exactly, with 0 missing files (§2.2).
- No function in `src/doar/partition.py` writes, moves, deletes, or opens
  any file under the dataset root in write mode — every write goes to
  `outputs/phase7a/`. This was a design constraint of the module (see its
  module docstring), not an incidental property.
- `outputs/phase3a`–`outputs/phase6` were not touched: no file under any of
  those directories has a modification time from this session, and their
  own artifact manifests (already re-verified in Phase 6) were not
  re-checked again this phase since no operation in Phase 7A could have
  touched them (they are read nowhere in `src/doar/partition.py` or the
  driver script used this phase).

---

## 8. Phase 7 feasibility reassessment

See **`PHASE7_EXPERIMENT_MATRIX_REVISION.md`** for the full, itemized
revision (every proposed change stated with its reason).
**`PHASE7_EXPERIMENT_MATRIX.csv` itself has not been modified.** Headline:

- **DenseNet121 (A1)**: remains Recommended; caveat added for the smaller
  effective train size (1,273 groups vs. the 2,821 raw images originally
  assumed).
- **ConvNeXt-Tiny (A2)**: **downgraded from Recommended to Optional**,
  now probe-first like the compact-ViT experiment (A3) — 27.8M params
  against 1,273 independent training groups (≈21,800 params per
  independent group) is a materially worse ratio than the original
  assessment against 2,821 raw images, and no longer clears the bar for an
  automatic multi-seed campaign under this programme's own "don't assume
  bigger helps" governance rule.
- **Compact ViT / ViT-B/16 probes (A3–A5)**: unchanged in design; the
  smaller effective sample size is additional confirming evidence for their
  already-cautious, capped treatment.
- **DINOv2/OpenCLIP frozen-embedding experiments (B1–B4)**: **relatively
  strengthened** — frozen-embedding classifiers do not fine-tune a
  backbone and are far less exposed to the smaller effective training set
  than any Family A CNN.
- **Classical objective-feature baselines (C1–C5)**: unchanged — ~54
  hand-designed features and simple sklearn models are not meaningfully
  affected by 1,273 vs. 2,821 training examples.
- **Fusion (D1–D4) and fine-tuning ablations (E1–E4)**: unchanged in
  design; both must run against the new Phase 7A clean split
  (§1, item 3) rather than the original contaminated split once approved.
- **Seeds and tuning budgets**: unchanged — 3 seeds and the existing small
  hyperparameter grids remain appropriate; no class collapses to a
  critically small count in any split (smallest cell: `valid/Fear` = 44
  images, `test/Fear` = 78 images — smaller than before but still workable).

---

## 9. Verification (tests, Ruff, compileall)

- **New module**: `src/doar/partition.py` (duplicate-group computation,
  label-conflict assessment, effective-count statistics, group-disjoint
  partition construction, all required verification checks, CSV/JSON
  writers, and `run_partition_design()` — the function backing the new
  `main.py build-partition` CLI command).
- **New tests**: `tests/test_partition.py`, 20 tests covering exact/near
  duplicate grouping (including same-split near-dups, which
  `leakage.assess_leakage` never checks), transitive chaining, degenerate-
  hash exclusion, group-id determinism, exact- and near-duplicate label
  conflicts, deterministic partition reproduction, all 4 required
  verification checks, conflict images never being dropped, manifest path
  resolution, and the byte-determinism regression found in §5.4.
- **Test suite**: `pytest tests/` — **246 passed** (226 pre-existing + 20
  new), 9 pre-existing warnings, 0 failures.
- **compileall**: `python -m compileall src main.py` — exit 0.
- **Ruff**: `ruff check .` — **792 findings, exit 1**, identical to the
  established Phase 5/6/7 baseline. Two new findings (unused imports in the
  first draft of `partition.py`) were introduced and then fixed within this
  phase before finalizing — the final count is unchanged from baseline,
  confirmed by re-running `ruff check .` after the fix. No pre-existing
  finding was touched or fixed (out of scope for this phase).

---

## 10. Deliverables and artifact hashes

All under `outputs/phase7a/` (gitignored, per the `outputs/` convention —
not committed to git; hashes below are the durable record):

| File | SHA-256 |
|---|---|
| `duplicate_groups.csv` | `8adaf7d3494149a1ff8932604dbf94659f50baf28309371a21dc9a2ac261ac60` |
| `duplicate_group_edges_exact.csv` | `539e5e7ec81cd98a91ff302581295aaa0fb99ee77cff2e905e9e33b49e600987` |
| `duplicate_group_edges_near.csv` | `ee4f3879e56db40557b2b1ed603f06d1eb0d326dbcbd432dbcac6098f6d39a45` |
| `label_conflicts.csv` | `708304212a4fe260c681a10dee667ab86c5c66ecf2181cc9ebe8a8b28f1eb6c5` |
| `label_conflict_review.csv` | `e0b5b8460bb32dd99da9ff2644bba7be16b917d3185dbaca170d85b4ecf79c29` |
| `partition_manifest.csv` | `80c9a2db6ffcced745cba6bb8b544c24c625a716580cca93790aeb32224d3137` |
| `effective_counts.json` | `7311b6b8b590b561070855bb8400125664bafada8df2fbc23acae6aa5caf4a9e` |
| `partition_config.json` | `d677615d0381fa0e88e8e8672b1287cb90a4920a2fd14bc1ad2eef51e458d4af` |
| `leakage_verification_report.json` | `c79870cf89b3a6965e229a1e859332572ed50b302ef6758c1786e88ee3d761e1` |
| `leakage_reproduction_check.json` | `1ff0b8532424bd69acca3d611d5298f3841dff35ff7b7a3536f7f626da46aa3e` |
| `threshold_sensitivity.json` | `ea68d780b277753e94ea67be2f4d966b12817d7239fc274b82a3816374d82297` |
| `large_group_composition.json` | `5df52549ff68bceb17430c2ec93e230d5af14e7fd232db4e9b5b472082ab2b9e` |

(Also mirrored in `outputs/phase7a/ARTIFACT_HASHES.json`, and independently
re-verified against the on-disk files at the end of this phase — all
matched.)

**Committed** (not gitignored, like `PHASE7_RESULTS.md` before it):
this document, `PHASE7_EXPERIMENT_MATRIX_REVISION.md`,
`src/doar/partition.py`, `tests/test_partition.py`, and the small
`main.py` addition (`build-partition` CLI command).

---

## 11. Explicitly out of scope for this phase (per user instruction)

The following were **not** done and should not be inferred from this
document: training, tuning, or evaluating any model; visually inspecting
any individual test-split image; generating predictions on the new test
split; reporting any test-split metric; resolving any label conflict
automatically or by decision; overwriting or cleaning the original
dataset; modifying `PHASE7_EXPERIMENT_MATRIX.csv` in place; claiming
subject-level independence; activating any psychological interpretation;
starting Stage 0 or any other Phase 7 experiment.

**This phase stops here**, per explicit instruction, pending review of the
counts, partition quality, conflict policy, and the proposed experiment-
matrix revision.
