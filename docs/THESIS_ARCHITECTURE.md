# Thesis architecture

```mermaid
flowchart TD
  I[Drawing] --> Q[Quality and segmentation]
  Q --> F[Objective features]
  Q --> D[Deep image models and embeddings]
  F --> PF[Primary multimodal fusion]
  D --> PF
  PF --> C[Validation-fitted calibration]
  C --> U[Uncertainty]
  U --> E[Angry / Fear / Happy / Sad]
  E --> X[Explainability evidence]
  X --> R[Clinical decision support]
  R --> J[Rules, concerns, judges, review, reports and Q&A]
```

Psychologist rules and concern profiles never feed the primary classifier.
Semantic fusion must remain a separately named supplementary experiment.

The primary thesis contribution is the validation-selected fusion of objective
drawing features and deep visual representations with calibration, uncertainty,
and evidence traceability. Clinical decision support is downstream and cannot
modify classifier probabilities.
