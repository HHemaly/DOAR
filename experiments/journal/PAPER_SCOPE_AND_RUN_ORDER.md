# Paper scope, run order and readiness — 28 Oct 2026

Branch `journal/mva-2026`. This replaces the "Needs you" part of `STATUS.md`.

## Publication boundary (agreed)

| Component | This journal paper | Kept for the DOAR thesis |
|---|---|---|
| Duplicate/leakage audit of the benchmark | ✔ | cite |
| Source-grouped re-evaluation of 9 encoders + single test run | ✔ | cite / reuse split |
| Drawing-mark (foreground) extraction vs human references and baselines | ✔ | extend |
| Reliability of 11 focus features; when to withhold measurements | ✔ (all 49 in supplement) | broader evidence system |
| Psychological rules, emotional meaning of formal properties | ✘ | ✔ |
| Multimodal reasoning, evidence integration, DOAR architecture, LLM writer | ✘ (one sentence of future work at most) | ✔ |

Working title: *Benchmark and measurement validity in children's drawing analysis:
leakage-controlled representation learning and reliable visual features*.

## What exists today and what does not

| Evidence | Status |
|---|---|
| Filename audit of committed T0 manifests (102/284 valid, 200/512 test share a Roboflow stem with train) | **Verified here** (filenames only) |
| Archived T0 results (9-encoder probe, adaptation, finalists) | Exist; **validity in question** (leakage) |
| Frozen-V2 holdout automatic states (80 images) | Exist; human review not done |
| Full-manifest audit, A2 group-bootstrap re-scoring, A6 overlap | Code ready; **run in Colab** |
| Image-level duplicate evidence (A4) | Hashing done on your Drive; embeddings to be rerun with the fixed CLIP |
| Blinded pair review (A5) | **Needs you** (~30–60 min, click-through) |
| T1 split | Code ready; **needs your approval of the exact manifest** |
| T1 representation results, selection, test result | **Do not exist** (GPU) |
| Human reference masks, baselines vs references, feature reliability, reporting policies | **Do not exist** (annotation ~4–8 h + second annotator) |
| Dataset provenance, licences, label origin, ethics statement | **Missing — only you can supply** |

## Option A — focused paper (benchmark validity only)

- **Question:** do augmented copies across splits inflate a reused children's-drawing benchmark,
  and which representation remains best under a source-grouped split?
- **Novelty vs MJEER:** MJEER reported four-class scores on an uncontrolled split; this paper audits
  that split, shows perceptual-hash grouping misses augmented copies, and re-evaluates under control.
- **Mandatory:** A1–A7, pair review, approved T1, smoke test, B1–B3, B4 single test.
- **Missing now:** everything after A1 on real data, plus provenance/ethics.
- **Realistic timing:** about 1 week if nothing breaks (review ~1 h, GPU roughly 6–12 h — an
  estimate, the archived logs do not record wall time — analysis and writing 2–3 days).
  **Not this week.**
- **Fit:** narrower; better for *Pattern Recognition Letters* (short format) than MVA.
- **Cost:** the measurement-validity half would need its own later paper, or go to the thesis.

## Option B — full validity paper (recommended)

- **Question:** two threats to the validity of evidence in drawing analysis — benchmark leakage
  and unvalidated foreground-dependent measurements — and a protocol that controls both,
  including a test of whether unreliable measurements can be withheld.
- **Novelty vs MJEER:** a methods/validity contribution, not another classifier; no psychological claims.
- **Mandatory:** Option A + C1 holdout review, C2–C4 reference masks with a second annotator,
  baselines, feature reliability, reporting-policy comparison.
- **Realistic timing:** 2–3 weeks, bounded mainly by annotation and the second annotator.
- **Fit:** *Machine Vision and Applications* (applied vision, evaluation protocols, imaging conditions).
- **Kept for the thesis:** interpretation, rules, reasoning, integration, architecture.

**Neither option is submission-ready this week.** Submitting with missing core results would mean
assumptions in place of evidence. If there is a hard deadline, tell me what it is.

## What you run in Colab, in order (standalone notebook)

| # | Cell | Runtime | Time (est.) | You do |
|---|---|---|---|---|
| 1 | 0.1, 0.2a, 0.2 setup | CPU | 1–2 min | allow Drive |
| 2 | A1, A2, A6 | CPU | ~5 min | — |
| 3 | A4 (embeddings only now) | CPU 10–25 min, T4 ~2 min | | — |
| 4 | A5 make | CPU | ~3 min | — |
| 5 | Review cell, each `STRATUM` | CPU | 30–60 min total | **click verdicts** |
| 6 | A5 score, then A5b (applies the acceptance rule; writes ACCEPTED_EDGES.json) | CPU | seconds | if A5b says NEXT: review the `*_tight` group, re-score |
| 7 | A3 propose (reads ACCEPTED_EDGES.json — nothing to edit), A7 report | CPU | ~2 min | **send me STAGE_A_READINESS.md** |
| 7b | A5c (what each evidence type would add), A6b (do earlier result files mention T0 test images?) | CPU | A6b a few min on Drive | send summary.json of both |
| 8 | Freeze T1 | CPU | seconds | **approve (your name)** |
| 9 | B0 smoke test | GPU | 5–10 min | must print `SMOKE PASSED` |
| 10 | B1–B2 full run | A100 | ~6–12 h (est.), resumable | re-run cell after disconnects |
| 11 | B3 analysis + freeze selection | CPU | minutes | — |
| 12 | Holdout review (buttons), C1 | CPU | ~1 h | **click verdicts** |
| 13 | C2 reference batch, C2b annotation kit (zip in the batch folder) | CPU | 20–40 min | — |
| 14 | Annotation app from the kit (local PC; Streamlit 1.61.1 + drawable-canvas 0.9.3) | — | 4–8 h + second annotator 40 images | **draw masks**, upload masks/ and records/ |
| 15 | C3, C4 | CPU | ~15 min | — |
| 16 | B4 single test | GPU | minutes, once | `CONFIRM = True` |
| 17 | Export zip | CPU | 1 min | send to me |

Stop after step 7 and send me the report; nothing in Stage B starts until you approve T1.

## Manuscript outline and exhibits (Option B)

1. Introduction — two validity threats; research questions; contributions; explicit scope (no psychological claims).
2. Related work — drawing classification; formal-element measurement; binarization and segmentation QC; leakage and shortcuts; selective prediction.
3. Data — sources, licences, labels, ethics; T0 construction.
4. Methods — duplicate evidence and review; T1 construction and balance criteria; representation protocol; foreground methods; reference annotation; three-level measurement analysis; statistics (cluster bootstrap).
5. Results — 5.1 audit; 5.2 T0 vs T1 representation; 5.3 single test; 5.4 applicability and mask correctness; 5.5 feature reliability; 5.6 reporting policies.
6. Discussion — what changed and why; negative findings; lessons for benchmark builders.
7. Limitations — image- not subject-level independence; undetected duplicates; label validity; annotation anchoring; static images.

| Exhibit | Content | Source script |
|---|---|---|
| Fig. 1 | Pipeline: audit → T1 → representation; references → masks → features → reporting policy | drawn |
| Fig. 2 | Confirmed duplicate pairs (with transformation) across T0 splits | A5 sheets |
| Table 1 | Source audit and pair-review rates with Wilson CIs | A1, A5 |
| Table 2 | T1 split balance (sizes, class shares, groups, source families) | A3/A7 |
| Fig. 3 | Shared vs independent validation macro-F1 per saved run (exploratory) | A2 |
| Table 3 | T0 vs T1 macro-F1 (probe, adaptation) with group-bootstrap CIs; Kendall τ | B3 |
| Table 4 | Single test result: selected vs comparator vs probe | B4 |
| Table 5 | Applicability: human assessability × V2 policy | C4 |
| Fig. 4 | Mask F1 by method and support regime (CIs) | C4 |
| Table 6 | Focus-feature reliability (ICC, MAE) per method | C4 |
| Fig. 5 | Risk–coverage of the three reporting policies | C4 |
| Fig. 6 | Failure gallery: original / reference / A0 / V2 / best baseline | C3 |

## Remaining blockers (all must be closed before submission)

1. Dataset provenance, licences, label origin, ethics/consent statement.
2. Pair review and T1 approval.
3. Stage B results and the single test evaluation.
4. Holdout review, reference masks, second annotator.
5. Manuscript rewritten from actual results; official Springer template; VERIFY references resolved.
6. Overlap check against MJEER and the DMAR-X draft; cover-letter disclosure.

## Verified in this environment vs needs your Drive/GPU

- **Verified here:** 25 unit/integration tests; the standalone notebook's Stage A cells executed
  end to end on a synthetic dataset with a stand-in for Drive (setup, A1, A2, A6, A5 make, score
  with one group finished, A3 preview, A7 report); filename-only T1 previews on the committed
  manifests (regroup_full passes all four balance checks; minimal_change fails two).
- **Not verifiable here:** anything reading your images or Drive results; pretrained model
  downloads (blocked in this environment), so the smoke test is the first real check of Stage B.
