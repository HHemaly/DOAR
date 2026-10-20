# Submission checklist — Machine Vision and Applications (Springer)

Status key: ✅ done · ⏳ waiting for an experiment · ✋ needs the authors · 🔎 verify on the journal page

I could not open MVA's "Submission guidelines" page from this environment
(the request failed), so items marked 🔎 follow Springer Nature's general
author instructions and must be checked against
https://link.springer.com/journal/138/submission-guidelines before submitting.

## Scientific content
- ⏳ No `\pending{...}` left in `manuscript/main.tex` (search the PDF for "PENDING")
- ⏳ Every number traceable in `RESULTS_PROVENANCE.md` to a script output with a hash
- ✋ Blinded pair review done (A5) and T1 approved and frozen
- ⏳ T1 rerun (B1–B3) complete; selection frozen before B4
- ⏳ Single test evaluation (B4) run once; lock file present
- ✋ Holdout visual review complete (C1); reference masks done by two annotators (C2–C3)
- ⏳ Discussion and conclusion rewritten from the actual results, including negative findings
- ✅ No psychological, clinical or diagnostic claims
- ✅ Pre-registration committed before experiments (`experiments/journal/PREREGISTRATION.md`)

## Data, ethics and prior publication
- ✋ Data section names every source dataset, its licence and label origin
- ✋ Ethics/consent statement for children's drawings
- ✋ Cover letter discloses the MJEER paper (same dataset family), states that no
  text, figures or numbers are reused, and explains the difference
- ✋ Decide whether the DMAR-X draft will be submitted; if yes, it must not overlap
  this paper's experiments while both are under review
- ✅ No child images are committed to the public repository (only manifests and file names)

## Format (🔎 confirm each)
- 🔎 Springer Nature LaTeX template: the project uses `sn-jnl.cls` **v3, Oct 2023**
  from a GitHub mirror; replace with the current official template (v3.1, Dec 2024,
  from Springer/Overleaf) and rebuild
- 🔎 Reference style: numbered (`sn-mathphys-num`) assumed
- 🔎 Abstract length (Springer default 150–250 words) and 4–6 keywords
- 🔎 Peer review type (single- vs double-blind). If single-blind, replace the
  anonymised MJEER citation with the full reference
- 🔎 Figures as vector PDF/EPS or ≥300 dpi raster; every figure cited in text
- ✋ Author names, affiliations, ORCID iDs, corresponding author email
- ✋ Declarations: funding, competing interests, ethics, consent, data availability,
  code availability, author contributions (CRediT), generative-AI use
- ✅ Builds with `latexmk -pdf main.tex` (verified 20 Oct 2026, 11 pages, draft)

## References
- ⏳ Resolve every `VERIFY` note in `references.bib` (currently: Kim et al. ×2,
  Robinson et al., PsyDraw authors, Lin et al. first names, Long et al. authors,
  Sukesh et al. venue, MJEER full citation)
- ✅ Each other entry was checked against a publisher or arXiv page during the audit

## Open access (optional)
- ✅ MVA is hybrid; subscription publication costs nothing
- ✋ Springer–EKB agreement covers OA if the corresponding **and** first author have only
  Egyptian affiliations (GUC is listed); request through the Springer workflow after acceptance

## Repository at submission
- ⏳ Tag the exact commit used for every reported number (e.g. `journal/mva-submission-1`)
- ⏳ `RUN_GUIDE.md` and `RESULTS_PROVENANCE.md` up to date
