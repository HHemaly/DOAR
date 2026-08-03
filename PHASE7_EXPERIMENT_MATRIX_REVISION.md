# Proposed Revision to PHASE7_EXPERIMENT_MATRIX.csv (Phase 7A reassessment)

**This document proposes changes only — `PHASE7_EXPERIMENT_MATRIX.csv` itself
has not been modified.** Every change below is stated explicitly with its
reason. Nothing in this revision has been approved or acted on; Phase 7's own
Stage 0 has still not started.

## Why a revision is needed

Phase 7's original feasibility numbers (dataset size, class balance, param-
to-sample-size ratios) were computed against the **original, duplicate-
contaminated** train/valid split (2,821 / 310 images). Phase 7A shows the
true, group-disjoint, duplicate-controlled numbers are meaningfully smaller
and structured differently:

| Quantity | Phase 7's original assumption | Phase 7A's measured reality |
|---|---|---|
| Train images | 2,821 (contaminated) | 2,305 (clean, group-disjoint) |
| **Train independent groups** (the real effective sample size) | *not measured* | **1,273** |
| Valid images | 310 (contaminated) | 254 (clean, all singleton groups) |
| Test images | 557 (contaminated, and non-final) | 455 (clean, all singleton groups, **newly locked**) |
| Images set aside (unresolved label conflict) | *not measured* | 674 (18.3% of the whole dataset) |

The train split's **true effective sample size for overfitting purposes is
1,273 independent groups, not 2,305 images** — 667 of those groups are
singletons, but 606 groups contain 2+ duplicate/near-duplicate images each
(up to 9), so 2,305 raw images collapse to a meaningfully smaller number of
truly independent training examples. This number, not 2,821 or 2,305,
should govern any judgment about a model's params-to-data ratio.

**Note:** valid and test contain *only* singleton groups (0 duplicate
clusters landed in either) — a side effect of the largest-remainder
partitioning algorithm processing the largest groups first and always
routing them toward whichever split has the largest absolute deficit, which
is train by a wide margin for every class. This means valid/test's 254 and
455 images are each already "effectively independent" samples with no
internal duplication — a genuine improvement in evaluation validity, not
just a partition-mechanics artifact to note in passing.

---

## Per-experiment revision

| ID | Field changed | Was | Now | Reason |
|---|---|---|---|---|
| A1 (DenseNet121) | `rank` | Recommended | **Recommended (unchanged), caveat added** | 7.0M params vs. 1,273 independent train groups (≈5,500 params/group) is a worse ratio than originally assessed against 2,821 raw images, but still the most favorable of the 3 proposed new architectures. Recommend adding early stopping sensitivity / stronger regularization given the smaller effective set — not a rank change, a tuning-note addition. |
| A2 (ConvNeXt-Tiny) | `rank` | Recommended | **Downgraded to Optional** | 27.8M params vs. 1,273 independent train groups (≈21,800 params/group) is a substantially worse ratio than the original (already-cautious) assessment against 2,821 images assumed. Given Phase 7's own governance rule against assuming larger models help, and the now-quantified overfitting risk, this no longer clears the bar for an automatic "Recommended" — it should run only as a single-seed feasibility probe first (mirroring A3's existing treatment of the compact ViT), with escalation to 3 seeds gated behind an explicit advancement rule, not run at full multi-seed scope by default. |
| A3 (compact ViT probe) | `hypothesis` note | — | **Reinforced, no rank change** | Already the most cautious entry in the original matrix (single-seed probe, capped escalation). The smaller effective train size further supports treating it as a probe, not a competitive campaign — no structural change needed, the existing design already anticipated this. |
| A4 (compact ViT multi-seed) | `advancement_rule` | Conditional on A3 | **Conditional on A3 (unchanged, note added)** | Same logic as A3; the smaller effective sample size raises the bar A3 must clear before A4 is worth running, but the conditional structure itself does not need to change. |
| A5 (ViT-B/16 probe) | — | Optional, capped at 1 seed | **Unchanged** | Already capped regardless of outcome; the smaller effective train size is additional confirming evidence for why this stays capped, not a reason to change its design. |
| B1 (DINOv2 + logistic regression) | `expected_thesis_value` note | High | **High (strengthened)** | Frozen-embedding classifiers do not fine-tune a backbone and are far less parameter-hungry relative to the training set than any Family A CNN. A smaller effective train size (1,273 groups) makes Family B *relatively more attractive* than it was under the original, larger-assumed sample size — this is a positive update, not a caveat. |
| B2 (DINOv2 + SVM) | same as B1 | High | **High (strengthened)**, same reasoning as B1 | |
| B3 (DINOv2 + MLP) | `main_confounders_limitations` | Already flagged as most overfitting-prone of the 3 classifiers | **Caveat strengthened** | 1,273 independent groups is a meaningfully smaller number to fit even a shallow MLP against; the existing overfitting caveat is now quantitatively better-supported, not new in kind. |
| B4 (OpenCLIP) | — | Optional | **Unchanged** | Same reasoning as B1/B2 applies; still offered as a bonus, not upgraded to Recommended since it remains outside the user's original explicit request. |
| C1–C5 (classical objective-feature baselines) | — | Recommended | **Unchanged** | These use ~54 hand-designed features with 6 simple sklearn model families (logistic regression, SVM, tree ensembles) — orders of magnitude fewer parameters than any deep architecture in this matrix. 1,273 independent train groups is ample for these model classes; no revision needed. |
| D1–D4 (fusion) | `controlled_variables` | Assumed Family A/B/C's original-split results as components | **Must be rerun on the Phase 7A clean split, not assumed transferable from any hypothetical original-split run** | Fusion experiments combine Family A/B/C outputs; if Family A/B/C are ever run, they must run on the new clean split (see below) for their outputs to be validly fused — no result computed against the old contaminated split may be substituted in. This is a procedural clarification, not a ranking change. |
| E1–E4 (fine-tuning ablations) | `n_seeds`, `tuning_budget` | 3 seeds, small grids, sized against 2,821 train images | **Unchanged in structure; must run against the new 2,305-image / 1,273-group clean train split** | The ablation designs themselves (freeze-depth, focal loss, augmentation on/off) do not depend on exact sample size, only on which split they run against. No budget change needed, only a split-source correction. |

## Cross-cutting procedural changes (apply to every experiment, not tied to one ID)

1. **All of Phase 7's experiments must now specify which split they run
   against**: the original Phase 3A–6 contaminated split, or the Phase 7A
   clean split. **Recommendation: use the Phase 7A clean split
   (`outputs/phase7a/partition_manifest.csv`, `new_split` column) for all
   future Family A–E development work**, since it is a strict improvement
   (image-group-disjoint, conflict-aware) at a modest cost in raw image
   count. The original contaminated split should no longer be used for new
   development-time comparisons going forward, though existing Phase 3A–6
   results computed against it remain valid as *already-completed,
   preliminary* evidence and are not retracted.
2. **The excluded_conflict subset (674 images, 18.3% of the dataset) is
   unavailable to every experiment in the matrix** unless a specific,
   documented, defensible resolution is proposed for a specific conflict
   subset (none is proposed here) — this was implicit in the original
   matrix's leakage caveats and is now a concrete, enforced exclusion.
3. **The new test split (455 images, `new_split == "test"`) is the
   candidate locked final-evaluation set** referenced in
   `PHASE7_RESULTS.md` §7 — no experiment in the revised matrix may access
   it for training, validation, or model selection, exactly as originally
   specified. It has not been unlocked, inspected, or used in this
   revision.

## What is NOT changed

- No experiment ID is removed from the matrix.
- No family's overall priority ordering (§5.2 of `PHASE7_RESULTS.md`)
  changes: Family C and Family A (A1/A2 as probes-first) remain first
  priority, Family B remains second, Family E third, Family D fourth —
  only A2's aggressiveness (multi-seed by default → probe-first) is
  actually downgraded.
- No new experiment is added in response to these findings (that would be
  exactly the "post-hoc expansion" this programme's own governance
  prohibits, applied in the opposite direction — down-scoping in response
  to new evidence is appropriate; adding speculative new experiments in
  response is not, and none are added here).

## Approval required before any of the above takes effect

This revision is a proposal. `PHASE7_EXPERIMENT_MATRIX.csv` remains the
approved reference until the user reviews and accepts (in whole or in part)
the changes listed above. No Stage 0 experiment has been started.
