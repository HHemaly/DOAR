# Journal completion status — 20 Oct 2026

Branch `journal/mva-2026` (pushed). Code is **implemented and tested locally**;
**no Colab or GPU experiment from this branch has run yet** — I have no access
to your Drive or a GPU runtime.

## Done and verified

| Item | Evidence |
|---|---|
| Journal branch combining the E1/E2 research line, the archived notebook and `doar-trace-main` | `git log journal/mva-2026` |
| Filename-level source audit of T0 (3,395 images) | `experiments/journal/results/A1_stem_audit_T0_included/` — 270 Roboflow stems span splits; 102/284 validation and 200/512 test images share a stem with training; 0 stem groups with conflicting labels; 45 further plain filenames recur across splits |
| Correction to my earlier figures | Stricter rerun: 35.9% validation / 39.1% test (earlier quick count said ~39% / ~43% because it also merged plain filenames) |
| Verbatim extraction of the archived A0, R1-V1, R1-V2, 51-feature and E1-B code into importable modules | `experiments/journal/tools/extract_frozen_e2.py`; tests show the three archived copies of A0 agree and the injectable feature extractor reproduces the archive exactly |
| E1-B code runs, writes and resumes on a new manifest | CPU smoke test `test_adaptation_runs_and_resumes_on_cpu` |
| Stage A/B/C scripts wired end to end | 19 tests pass; synthetic integration run of A1, A2, A4, A5, A6, A3, C2, C3, C4 |
| Colab driver notebook | `colab/journal/DOAR_Journal_Validation_2026.ipynb` (generated, 26 cells; cells compile) |
| Draft manuscript in Springer Nature LaTeX | `manuscript/main.pdf` builds (11 pages); every missing result is a red `[PENDING]` marker; author/ethics items are blue `[AUTHOR INPUT]` |

## Implemented, awaiting execution in Colab

| Stage | Script | Runtime |
|---|---|---|
| A1 full-manifest audit, A2 re-scoring of saved predictions, A6 dev/holdout overlap | a1, a2, a6 | CPU, minutes |
| A4 rotation/flip hash + CLIP near-duplicate scan | a4 | CPU ~20 min or GPU minutes |
| A5 blinded pair-review sheets | a5 make | CPU |
| A3 T1 proposals | a3 propose | CPU |
| B1–B3 T1 rerun of the 9-encoder probe, adaptation and confirmation | b1_b2, b3 | GPU, several hours on an A100 |
| B4 single test evaluation | b4 | GPU, minutes, once |
| C1–C4 holdout scores, reference batch, evaluation, statistics | c1–c4 | CPU |

## Needs you

1. **Run Stage A** in the notebook (CPU) and send me the export zip.
2. **Blinded pair review** (A5): about 60 pairs per sheet, roughly 30 minutes.
3. **Approve one T1 proposal** after seeing its split sizes (protocol change).
4. **GPU runtime** for Stage B (A100 recommended).
5. **Holdout review**: finish the existing 80-row CSV.
6. **Reference masks**: ~120 images by you; 40 by a second annotator
   (your supervisor?) — about 4–8 hours in total.
7. **Data provenance and ethics** for the Data and Declarations sections.

## Not yet implemented (deliberately deferred)

- E1-C2 robustness perturbations on T1: will be ported from notebook cell 18
  after B3, only if the selected model changes or a reviewer is likely to ask.
- A learned (deep) binarization baseline: optional; added only if the
  classical baselines and V2 leave the comparison inconclusive.
