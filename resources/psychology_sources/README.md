# Psychology sources

Place the unchanged psychologist PDF here before running source ingestion.
Structured rules must not be activated until transcription and clinician review
are complete.

- `rules_registry.json` — the 19 ACTIVE, curated rules currently used by
  `rules.py`. Verified 2026-08-02 to be thematically faithful to the source
  PDF (`../../التحليل النفسي للصور.pdf`). See `CURRENT_STATE_AUDIT.md` §5.1,
  `RULE_COVERAGE_MATRIX.csv`, and `RULE_SOURCE_REGISTER.csv` at the repo root.
- `draft_registry_ingested_2026-08-02.json` — raw `ingest-psychology-pdf`
  output: 43 auto-extracted statements, every one `blocked_pending_review`
  with `confidence_ceiling: 0.0` (per `psychology_ingest.py`'s safety policy).
  This is the full source text split into candidate statements, useful for
  checking whether `rules_registry.json` missed anything — it is NOT itself
  usable as a rule source until a clinician reviews and reconciles it.
