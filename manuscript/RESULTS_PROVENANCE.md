# Where every number in the manuscript comes from

Add a row whenever a number enters `main.tex`. A number without a row must not
be submitted.

| Manuscript location | Value(s) | Source | How verified |
|---|---|---|---|
| Abstract; Sec. 5.1; Table 1 | 2,432 Roboflow-named files; 1,300 stems; group sizes 670/148/463/18/1; spanning 76/157/15/22; valid 102/284 (35.9%), test 200/512 (39.1%); 0 conflicting stems; 45 plain names across splits | `experiments/journal/results/A1_stem_audit_T0_included/summary.json` (committed; script `a1_stem_audit.py`) and the plain-name check in the 20 Oct status report | Recomputed by script on the committed T0 manifests (filenames only) |
| Sec. 3 | 3,688 raw; 293 excluded in 93 conflict groups; 2,599/284/512 | `T0_AUTOMATED_CONSERVATIVE_PARTITION.md` §4 | Matches notebook cell 0/2 verification output |
| Table 2, Sec. 5.2 (T0 probe) | 10-seed means ± SD for 9 encoders | `colab/archive/DOAR_21_8_26_V1.ipynb` cell 3 output (E1-A2R) | Read from printed output; rounding to 3 d.p. |
| Table 2, Sec. 5.2 (T0 adaptation) | SigLIP2 partial 0.816 ± 0.015; CLIP partial 0.830 ± 0.004; EfficientNet-B0 full 0.798 ± 0.027 | same notebook, cell 9 output (3 seeds) | idem |
| Sec. 5.2 (T0 finalists) | CLIP partial 0.830 ± 0.006; SigLIP2 partial 0.820 ± 0.016 (5 seeds) | same notebook, cell 12 output | idem |
| Sec. 5.4 | A0 acceptable 2,577/2,883 (89.4%); rule ≥ 0.1% foreground | notebook cells 20–21 output; rule in cell 20 source | idem; rule read from code |
| Sec. 5.5 | 51 numeric columns; 49 drawing features; 3 mask-independent | `src/doar_journal/measurement.py` on a synthetic image; test `test_mask_dependence_detects_mask_free_features` | Executed locally 20 Oct 2026 |
