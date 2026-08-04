# Human-Review Export Verification

**Status: verification report. The completed Phase 7B human duplicate
review is genuinely present, internally consistent, and complete.** A
candidate (not frozen, not approved) duplicate-controlled partition has
been built directly from it. This document is the evidence trail for that
claim — every number below was computed this session from the actual
files on disk, not assumed.

---

## 1. Export files found

| File | Path | Size | Verified |
|---|---|---|---|
| Decisions | `outputs/phase7b/human_review/app_data/decisions.json` | 30,432 bytes | ✅ exists, valid JSON, 225 entries |
| Item registry | `outputs/phase7b/human_review/app_data/items_registry.json` | 166,864 bytes | ✅ exists, 225 items, 4 categories |
| Pair export | `outputs/phase7b/human_review/exports/human_pair_reviews.csv` | 46,197 bytes | ✅ 225 rows, matches decisions.json exactly |
| Group export | `outputs/phase7b/human_review/exports/human_group_reviews.csv` | 1,511 bytes | ✅ exists |
| Agreement report | `outputs/phase7b/human_review/exports/reviewer_agreement_report.json` | 1,423 bytes | ✅ exists |
| Precision summary | `outputs/phase7b/human_review/exports/threshold_precision_summary.json` | 7,915 bytes | ✅ exists, real per-distance table |
| Unresolved items | `outputs/phase7b/human_review/exports/unresolved_items.csv` | 119 bytes | ✅ 1 row, matches the single `uncertain` decision |

All 7 files exist and are internally consistent with each other (cross-checked, §3).

## 2. Number of reviewed comparisons

**225 of 225 items decided (100.0%).** Coverage by category, cross-checked
directly against `items_registry.json`'s `categories_order`:

| Category | Decided / Total |
|---|---|
| `ambiguous` | 87 / 87 |
| `threshold_boundary` | 71 / 71 |
| `component_17` | 47 / 47 |
| `policy_change` | 20 / 20 |

Zero missing decisions. Zero decisions referencing an unknown `item_id`.

## 3. Decision counts by category (overall)

| Decision | Count |
|---|---|
| `definite_duplicate` | 104 |
| `different_drawings` | 104 |
| `same_drawing_transformed` | 16 |
| `uncertain` | 1 |
| **Total** | **225** |

## 4. Missing, duplicated, or malformed records

- **Malformed records: 0.** Every decision has `decision`/`notes`/`reviewed_at`, and every `decision` value is one of the 4 valid choices.
- **Duplicate `item_id` keys: 0** (a Python dict cannot have them; also verified no `item_id` appears in more than one category's list).
- **The same underlying image pair reviewed under more than one `item_id`: 15 cases.** This happens because a few real image pairs were independently sampled into more than one review category (e.g. both `threshold_boundary` and `component_17`). Of these 15:
  - **12 are consistent** (both reviews agree, or one says `definite_duplicate` and the other says `same_drawing_transformed` — semantically the same "merge" verdict).
  - **2 are semantically consistent "soft" duplicates** (`same_drawing_transformed` vs. `definite_duplicate` on the same pair — both mean merge, not a real conflict).
  - **1 is a genuine contradiction**: image `861104569b094b1c` vs. `8b5da8797c2ae65b` (dHash distance 6) was reviewed as `definite_duplicate` under item `tb_0023` and as `different_drawings` under item `c17_0044`, with no notes on either to disambiguate. **This is flagged, not silently resolved** — see §6.

## 5. Whether all expected comparisons received decisions

**Yes — every item in the 225-item registry has a decision**, confirmed directly from the files, not inferred from the export's own claim. The registry itself (`items_registry.json`) is the same one built and documented in `PHASE7B_DUPLICATE_POLICY.md` §24 — no new sampling was needed or performed.

## 6. How pairs and duplicate groups should be handled based on these decisions

Real, now-complete data (`threshold_precision_summary.json`) supports a **conservative, evidence-grounded, three-part policy** — see `src/doar/partition.py::compute_duplicate_groups_from_review` (new this session):

**Human precision by exact dHash distance** (computed from the real 225 decisions, not the earlier AI-preliminary estimates):

| Distance | n | Precision | 95% CI |
|---|---|---|---|
| 0 | 2 | 100.0% | 34.2–100.0% (small n) |
| 2 | 7 | 100.0% | 64.6–100.0% |
| 3 | 20 | 90.0% | 69.9–97.2% |
| 4 | 46 | 76.1% | 62.1–86.1% |
| 5 | 25 | 24.0% | 11.5–43.4% |
| 6 | 26 | 19.2% | 8.5–37.9% |
| 7 | 6 | 33.3% | 9.7–70.0% |
| 8 | 6 | 33.3% | 9.7–70.0% |

Precision collapses sharply between distance 4 (76.1%) and distance 5 (24.0%). Distance ≤3 has 90–100% precision (n=27 combined) and is the only range clean enough for a **blanket automatic merge default**.

**Recommended policy** (used to build the candidate in §7, not yet approved):
1. **Exact sha256 duplicates** → always merge (unchanged, never a judgment call).
2. **dHash ≤ 3** → auto-merge by default (90–100% precision).
3. **Every one of the 225 directly-reviewed pairs** → the human's exact verdict *overrides* the default in both directions: `definite_duplicate`/`same_drawing_transformed` forces a merge even at a larger distance (114 such edges, e.g. the `same_drawing_transformed` case at distance 5 in §4); `different_drawings` forces two images to **not** merge even where distance ≤3 would otherwise auto-include them (95 such edges — **2 of them are specifically at distance ≤3**, i.e. cases the blanket threshold alone would have gotten wrong).
4. **The single genuine contradiction** (§4) is resolved conservatively **toward merging** (this project's established leakage-safety default: under-grouping risks real leakage; over-grouping only costs stratification flexibility) and is recorded, not hidden, in `flagged_contradictions.json`.
5. **Every other near-dup edge** (distance ≥4 and never directly reviewed — none exist at distance 4 specifically, since **all 43 dataset-wide distance-4 edges were reviewed**, but many do exist at distance 5–8) is **left unmerged** — the conservative choice given 19–33% precision at that range.

## 7. Whether the export can safely support a leakage-controlled split

**Yes — a candidate partition was built and verified this session.** Written to `outputs/phase7b/candidate_partition_review_based/` (a **new directory**; `outputs/phase7b/final_partition/`, the old threshold-6 partition, was **not touched**):

- 2,320 duplicate groups (vs. 2,290 near-dup edges dataset-wide feeding into them).
- Largest group: **9 images** (vs. 17 in the old, now-superseded threshold-6 policy — direct confirmation that removing blanket high-distance inclusion and honoring the explicit human overrides fixes the heterogeneous-chaining problem `PHASE7B_DUPLICATE_POLICY.md` §15–16 identified).
- 97 conflict groups (305 images) excluded from the supervised split, down from 338 under the old policy.
- **All 5 partition-internal integrity checks pass**: no group crosses a split, no image appears in more than one split, no unresolved conflict enters a supervised split, no exposed group enters valid/test, and per-split/class counts match the manifest exactly.

Split totals: `train=2590, valid=283, test=510, excluded_conflict=305` (of 3,688).

### Live 8-check gate status (`src/doar/dataset_gate.py`, re-run this session)

| # | Check | Status |
|---|---|---|
| 1 | Human review decisions recorded | ✅ **PASS** — 225/225 |
| 2 | Human review exported | ✅ **PASS** — all 5 files present |
| 3 | Duplicate-detection policy approved | ❌ **FAIL** — `outputs/phase7b/APPROVED_POLICY.json` does not exist |
| 4 | Manifest frozen | ❌ **FAIL** — `outputs/phase7b/final_partition/FROZEN.json` does not exist |
| 5 | No duplicate group crosses splits | ✅ **PASS** |
| 6 | Exposed images excluded from valid/test | ✅ **PASS** |
| 7 | Class counts and paths verified | ✅ **PASS** |
| 8 | Final test set untouched | ✅ **PASS** |

**6 of 8 pass right now.** The remaining 2 require an explicit human decision this session cannot make on your behalf (`Do not automatically approve unsupported dataset decisions`). Templates are ready:

- `outputs/phase7b/APPROVED_POLICY_TEMPLATE.json` — pre-filled with the recommended `near_dup_threshold=3, hash_field=dhash`; copy to `APPROVED_POLICY.json`, edit `approved_by`/`approved_at`/anything you disagree with.
- `outputs/phase7b/candidate_partition_review_based/FROZEN_TEMPLATE.json` — pre-filled with the candidate manifest's real SHA-256; copy to `outputs/phase7b/final_partition/FROZEN.json` (after promoting the candidate) once approved.

**Nothing was approved or frozen automatically.**

## 8. Change summary vs. the old threshold-6 partition

Full detail in `outputs/phase7b/candidate_partition_review_based/diff_vs_old_threshold6_partition.json`.

- **65 images have a genuinely different duplicate-group membership** — direct evidence of the fix (e.g. the notorious 17-image component's core cluster is now a tighter, human-confirmed 6-member group; several previously-chained images are now correctly isolated).
- **996 images have a different train/valid/test split label.** Most of this is **not** re-evaluated duplicate status — it's a downstream ripple effect of the same seeded stratified bin-packing algorithm re-running on a different (smaller, more accurate) set of group sizes. Only the 65 images above reflect an actual evidence-driven regrouping; the rest moved because the overall stratification shifted, not because their own duplicate status changed.

## 9. Provenance

Every manifest row in the candidate partition traces back to: the source manifest (`outputs/phase7b/manifest_with_dhash.csv`, SHA-256 recorded in `partition_config.json` via `provenance.py::build_manifest_and_split_provenance`), the exact review registry and decisions files used (paths recorded in `partition_config.json`), and — for every human-influenced edge — the specific `item_id`(s) that produced it (`human_forced_merge_edges.json`, `human_forced_split_edges.json`, `flagged_contradictions.json`).

## 10. What was explicitly NOT done

No image was deleted, relabeled, or overwritten. The final/locked test set was not accessed (`final_test_unlock_log.jsonl` remains empty). `outputs/phase7b/final_partition/` (the old, still-provisional partition) was not modified. No policy was approved and no manifest was frozen — those two gate checks remain open, waiting for you.
