"""DOAR journal-completion package (branch journal/mva-2026).

Version-controlled experiment logic for the Machine Vision and Applications
manuscript. The Colab notebook `colab/journal/DOAR_Journal_Validation_2026.ipynb`
only configures paths, calls these modules and plots their outputs.

Modules
-------
provenance      hashing, run metadata, resumable "done" markers
stats           image-level bootstrap, Wilson, McNemar, Kendall, Holm, ICC, kappa
leakage         source-stem audit, dihedral dHash, pair sheets, overlap checks (Stage A)
t1_partition    source-grouped T1 split built on doar.partition (Stage A)
rescoring       X1: re-score saved validation predictions by subset (Stage A)
foreground      A0 / R1-V1 / R1-V2 extractors (verbatim) + classical baselines (Stage C)
seg_metrics     IoU, pixel F1, boundary F1 (Stage C)
risk_coverage   AUROC / AURC / risk-coverage curves (Stage C)
"""

__all__ = [
    "provenance",
    "stats",
    "leakage",
    "t1_partition",
    "rescoring",
    "foreground",
    "seg_metrics",
    "risk_coverage",
]
