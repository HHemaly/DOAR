# Phase 2B Evaluation Protocol

## Metrics computed

Per class (`src/doar/phase2b/evaluation.py::compute_class_metrics`):
real positive/negative/uncertain/not-assessable counts, precision,
recall, F1, and a **ranking-separation statistic** (a dependency-free
stand-in for AUC: the fraction of positive/negative example pairs where
the positive example's raw similarity score ranks higher than the
negative's; 0.5 = no better than random ranking, 1.0 = perfect
separation). `uncertain`/`not_assessable` ground-truth rows are excluded
from precision/recall (they are not usable positive/negative labels) but
their counts are always reported alongside the usable pair count, so a
class's real support is never silently inflated by treating an
`uncertain` row as if it were a clean negative.

## Threshold sensitivity

For every class with at least one usable positive and one usable
negative, precision/recall/F1 are recomputed at 7 candidate thresholds
(0.20-0.32) around the baseline's chosen operating point (0.28),
written to `artifacts/phase2b/threshold_sensitivity.csv`. This exists
specifically so a single reported metric at one threshold never hides
how unstable — or, as this pilot found, how poorly calibrated — that
threshold actually is.

## Group-safe evaluation

Every image in the 20-image pilot sample is in a distinct duplicate
group (`docs/PHASE2B_LEAKAGE_CONTROL.md`), so no additional group-safe
splitting is needed for *evaluation* — there is no train/valid split in
this pilot to leak across (zero-shot and classical-CV need no DOAR-
specific training). This section is deliberately short because Phase 2B
does not perform model selection or threshold tuning against held-out
data — a real constraint at this sample size, recorded rather than
worked around.

## Bootstrap / uncertainty intervals

**Not computed.** At n=20 images (and as few as 0-2 positive examples for
8 of the 10 classes — `artifacts/phase2b/class_frequency_audit.csv`), a
bootstrap confidence interval would be numerically present but not
meaningfully informative (e.g. resampling from 1-2 positive examples
cannot produce a defensible interval). Reporting a fabricated-looking
interval on this little data would misrepresent precision, not add
rigor — the honest choice is to report raw counts and flag
`sufficient_support = False` instead.

## Real results (n=20, single annotator, this session)

| Class | n_present | Precision @0.28 | Recall @0.28 | Ranking separation | Note |
|---|---|---|---|---|---|
| `person` | 9 | n/a (0 predicted positive) | 0.0 | 0.70 | threshold miscalibrated; ranking shows real signal above chance |
| `face` | 15 | 1.0 | 0.067 | 0.18 | **worse than random** — a genuine negative finding |
| `house` | 3 | 1.0 | 0.33 | 0.94 | promising ranking signal, but n too small to trust |
| `tree` | 1 | n/a | 0.0 | 0.93 | single positive example only |
| `star` | 1 | n/a | 0.0 | 1.00 | single positive example only, not generalizable |
| `hand`, `vehicle` | 0 | — | — | — | zero positive examples; no result possible |
| `animal`, `heart`, `circle` | 2, 2, 1 | — | 0.0 | 0.42, 0.22, 0.53 | insufficient support |
| `circle` (classical CV) | 1 | 0.059 | 1.0 | — | massive over-firing (16 false positives out of 17 negatives) |

Full detail in `artifacts/phase2b/per_class_metrics.csv`,
`artifacts/phase2b/threshold_sensitivity.csv`,
`artifacts/phase2b/error_analysis.csv`.

## Honest interpretation

This is a **negative-to-weak result for CLIP zero-shot at a single fixed
threshold**, and a genuine, master's-level-valid finding in its own
right, not a failure to hide: `face`'s similarity scores are anti-
correlated with true presence on this dataset (ranking separation 0.18
< 0.5), and `person`'s chosen threshold (0.28) never fires at all despite
the underlying similarity ranking carrying real signal (0.70 separation)
— meaning **threshold choice, not the underlying representation, is the
dominant source of error for `person`**, while **the representation
itself appears not to transfer well to `face`** on line drawings. The
classical-CV circularity baseline for `circle` has the opposite problem:
very high recall, very low precision (over-detects circular-looking
contours everywhere). None of these results are strong enough to justify
activating any detector-dependent rule, and none are claimed to be.
