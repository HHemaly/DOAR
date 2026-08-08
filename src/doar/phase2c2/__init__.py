"""Phase 2C.2 -- descriptive object-baseline evaluation only.

Runs Phase 2B's own, unmodified CLIP zero-shot and classical-CV baselines
(src/doar/phase2b/inference.py) against the complete 80-image Phase 2C.1
pilot, and evaluates them against genuine Phase 2C.1 human annotations
(never the legacy Phase 2B provisional labels). No detector is trained or
fine-tuned here; no threshold is tuned; no rule is activated.
"""
