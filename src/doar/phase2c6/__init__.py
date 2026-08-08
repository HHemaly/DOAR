"""Phase 2C.6 -- scalable rule-critical annotation.

Upgrades Phase 2C.5's numeric-coordinate annotation app to a graphical
bounding-box editor (canvas_helpers.py, streamlit-drawable-canvas), and
adds the machinery needed to annotate hundreds of images instead of 15:
a blinded expansion-image manifest (expansion_manifest.py, reusing Phase
2C.5's own expansion_sampling.select_expansion_sample -- never
reimplemented), a resumable/checkpointed batched proposal-generation
pipeline (proposal_batch.py), and fine-tuning-readiness criteria
(finetuning_readiness.py, design only -- no fine-tuning happens in this
package).

Reuses, never duplicates, Phase 2C.5's schema/store/ontology/proposals
modules (src/doar/phase2c5/*.py) -- a Phase 2C.6 annotation is still a
Phase 2C.5 `PartAnnotationRecord`, same schema version, same store. Never
touches src/doar/phase2c1's own store, and never modifies any frozen
Phase 2C.4/2C.4A benchmark artifact.

No rule is activated by this package. No emotion label (Angry/Fear/Happy/Sad)
is used anywhere in it for selection, proposal, or interpretation decisions.
"""
