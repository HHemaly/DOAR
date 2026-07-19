# Experiment guide

Each objective-feature run records model family, seed, feature count, training
time, complete validation metrics, confusion matrix, classification report,
class probabilities, prediction CSV, and checkpoint. The aggregate leaderboard
uses seeds 42, 123, and 2026 by default.

Never rank models using test results. Freeze feature extraction, segmentation,
model selection, fusion, and calibration before using the explicit final-test
unlock.
