# Remaining implementation audit

Audit baseline: branch `feature/integrated-doar-pipeline`, commit `7d5f7b8`.
The existing standard-library synthetic suite passes 10/10 tests. The Codex
runtime is Python 3.12; the supported local thesis runtime remains Python 3.11.

| Requested capability | Current status | Relevant files | Missing work | Planned implementation | Tests required | Local runtime requirements |
|---|---|---|---|---|---|---|
| Multi-strategy segmentation | Implemented, preliminary | `analysis.py`, `features.py` | Page geometry and stronger photographic cases | Preserve ensemble; add optional OpenCV refinements | Shadows, dark background, rotation, crop | NumPy, Pillow; OpenCV optional |
| Objective feature cache | Implemented, partial | `features.py`, `extract.py` | True shapes, skeleton/stroke width, feature plots | Add classical geometry and stroke module | Synthetic known geometry/strokes | NumPy, Pillow; OpenCV/scikit-image optional |
| Evidence traceability | Implemented for core features/rules | `schemas.py`, `rules.py` | Model/detection/OCR/concern evidence | Stable IDs in every new schema/output | Unknown-ID rejection | Standard library |
| Psychologist rules | Implemented safely | `rules_registry.json`, `rules.py` | Detection-backed evaluation and evidence clusters | Activate only verified/probable allowed detections | Unsupported symbols remain unevaluated | Standard library |
| Whole-image statistical baseline | Implemented | `models.py` | More complete evaluation artifacts | Keep accurately named as baseline | Train/valid filtering | scikit-learn |
| Objective-feature models | Implemented, unexecuted here | `experiments.py` | Confidence intervals and plots | Extend thesis exporter | Synthetic table smoke test | scikit-learn, joblib |
| Deep image models | Implemented, locally untrained | `deep/registry.py`, `deep/trainers.py`, `deep/inference.py` | Dataset/GPU execution and measured comparison | Run multi-seed local experiments | One-epoch PyTorch smoke test locally | PyTorch, torchvision |
| Embedding extraction | Implemented, locally unexecuted | `deep/embeddings.py` | Real cache generation and DINO/OpenCLIP downloads | Run each configured backbone locally | Cache reuse with local PyTorch | PyTorch, optional OpenCLIP |
| Multimodal fusion | Primary fusion implemented, locally untrained | `fusion/trainer.py` | Late probability inputs and measured comparison | Run train/valid multi-seed suite | Synthetic training with scikit-learn | scikit-learn, optional PyTorch |
| Emotion case integration | Implemented for joblib and PyTorch checkpoints | `emotion.py`, `deep/inference.py` | Future fusion inference adapter | Add after frozen fusion checkpoint schema | Probability and malformed-checkpoint tests | Model-dependent |
| Shape/symbol detection | Placeholder feature values | `features.py` | Geometry implementation and overlay | Classical connected-component geometry first | Circle/rectangle/triangle fixtures | Pillow/NumPy, OpenCV optional |
| Object detection | Explicitly unavailable | `detections.json` | Modular detector adapters and verification | Optional detector; never force labels | Graceful unavailable and rejection | Optional torchvision/OpenCLIP |
| OCR | Explicitly unavailable | `case_output.py` | Tesseract/EasyOCR adapter and normalization | Optional local adapter | Missing engine and normalization | Optional pytesseract/EasyOCR |
| Concern profiles | Safety-preserving empty output | `rules.py` | Multi-source convergence engine | Require source diversity and multiple evidence IDs | Single-rule non-activation | Standard library |
| Deterministic judges | Core subset implemented | `judges.py` | Shape/stroke/detection/OCR/emotion/fusion/completeness | Add uniform judge schema | Invalid probability and evidence tests | Standard library |
| Reports | Basic HTML implemented | `reports.py` | Status-specific rule wording, charts, portable paths, PDF | Improve HTML and optional PDF | Arabic/English render checks | Standard library; PDF optional |
| Grounded Q&A | Basic implemented | `qa.py` | Full topic coverage and clinician edits | Evidence-only routing | Unavailable and citation tests | Standard library |
| Psychologist UI | Basic case reviewer | `streamlit_app.py` | Upload/run/correct mask/detections/features | Expand incrementally after schemas | Review history preservation | Streamlit |
| Thesis exports | Missing | none | Leaderboards, plots, ablations, galleries | Offline exporter from saved runs | Empty and synthetic experiments | pandas/matplotlib/scikit-learn |
| Locked final test | Partial | `models.py`, CLI | Second confirmation, audit log, hashes | Require both flags and log event | Refusal without flags | Standard library |

## Confirmed placeholders and unavailable modules

- Shape values `enclosed_shape_count` and `repetition_score` are placeholders.
- `detections.json` and `emotion.json` explicitly report unavailable.
- OCR, deep embeddings, fusion, calibration, Grad-CAM, PDF output, and thesis
  plots are not implemented at this baseline.
- The UI reviews case-level output but does not yet edit masks or bounding boxes.

The implementation sequence is: integration corrections; deep model and
embedding foundation; fusion; classical shapes/strokes and optional OCR;
concerns/judges; expanded reports/UI/Q&A; thesis export and release validation.
