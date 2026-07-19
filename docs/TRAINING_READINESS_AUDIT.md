# Training readiness audit

Baseline: branch `feature/integrated-doar-pipeline`, commit `ffa486e`.

| Capability | Current status | Relevant files | Issue | Required correction | Test required | Readiness |
|---|---|---|---|---|---|---|
| Objective-feature models | Implemented | `experiments.py` | Requires local scikit-learn execution | Run three-seed validation suite | Synthetic fit and checkpoint test | Code-ready, unmeasured |
| Deep registry | Implemented | `deep/registry.py` | Pretrained forward passes not executable in current runtime | Local PyTorch smoke suite | Forward pass per installed backbone | Code-ready, unverified locally |
| Deep training | Implemented | `deep/trainers.py` | Resume state recently corrected; local PyTorch test required | Verify optimizer/scheduler/scaler and stage restoration | One-epoch resume smoke | Requires local PyTorch |
| Deep inference | Implemented | `deep/inference.py`, `emotion.py` | Pretrained model checkpoint unavailable here | Run saved checkpoint prediction | Probability and metadata test | Requires local checkpoint |
| Embedding cache | Implemented | `deep/embeddings.py` | Pretrained weights and optional dependencies unavailable here | Validate each backbone and cache fingerprint locally | Cache hit/stale invalidation | Requires local PyTorch |
| Primary fusion training | Partially implemented | `fusion/trainer.py` | Early methods exist; late fusion/OOF stacking incomplete | Add genuinely distinct methods and bundle schema | Structural distinction and synthetic fit | Not fully ready |
| Fusion inference | Missing | `emotion.py`, `fusion/trainer.py` | Fusion joblib cannot yet reproduce features/embeddings in one-image inference | Save full bundle metadata and add inference adapter | Analyze-image fusion integration | Not ready |
| Calibration | Utilities implemented | `deep/calibration.py` | Not attached to every training/checkpoint path | Save/apply selected validation calibration | Fit/save/load/apply | Partial |
| Uncertainty | Implemented utility | `uncertainty.py` | Not attached to every inference result | Standardize prediction schema | Single/ensemble tests | Partial |
| Test lock | Implemented | `models.py`, `main.py` | Final protocol still needs frozen configuration enforcement | Require frozen config metadata | Dual-flag refusal/audit test | Partial |
| Dataset readiness | Missing | - | No dedicated structural/readability report | Implement `validate-dataset` | Synthetic folder validation | Not ready |
| Hardware readiness | Missing | - | No dependency/CUDA/disk/settings report | Implement `check-training-readiness` | CPU-only report test | Not ready |
| Experiment configurations | Minimal | `configs/experiments` | Missing deep/fusion/smoke resolved configs | Add TOML configurations | Parse/override test | Partial |
| Evaluation commands | Partial | `evaluate`, experiments | Missing image/fusion compare commands | Add validation-only orchestration | No-test-access test | Partial |
| Explainability | Objective evidence only | reports/artifacts | Grad-CAM and feature contributions missing | Add after training pipeline is frozen | Synthetic/mock explanation | Not ready |

## Misleading or duplicated interfaces

- `train` is the legacy whole-image statistical baseline; it must not be called
  an objective-feature model.
- Raw early-fusion methods are now named separately. Equal and
  validation-weighted late fusion consume probabilities only. Probability
  meta-features and OOF-fold validation exist, while full OOF base-model
  training orchestration remains incomplete.
- Fusion checkpoint files currently contain a fitted pipeline but not enough
  embedding/preprocessing metadata for direct `analyze-image` inference.
- Temperature scaling exists as a utility but is not yet consistently saved and
  applied by every model family.

## Test-split access audit

- Deep training loaders construct only `train` and `valid`.
- Objective-feature comparison loads only `train` and `valid`.
- Primary fusion loads only `train` and `valid`.
- Calibration APIs accept supplied validation arrays and do not read datasets.
- The standalone evaluation path requires two final-test flags, but the final
  protocol still needs frozen-configuration validation.
