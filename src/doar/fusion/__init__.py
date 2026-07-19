"""Primary classifier fusion. Clinical rules are intentionally excluded."""

PRIMARY_METHODS = ("early_scaled_concat", "pca_early_fusion", "mlp_early_fusion")
PROBABILITY_METHODS = (
    "equal_late_fusion", "validation_weighted_late_fusion",
    "oof_stacking", "logistic_probability_meta",
)
