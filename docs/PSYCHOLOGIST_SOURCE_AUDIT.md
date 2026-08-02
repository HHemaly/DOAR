# Psychologist source audit

> **SUPERSEDED 2026-08-02** — the claim below that the PDF is unreadable is
> incorrect (it was true of an earlier sandbox, not of this one). This session
> ran `main.py ingest-psychology-pdf` against the file directly and it
> succeeded (`pypdf` installed and working), extracting 43 statements across
> 2 pages. The extracted content confirms `resources/psychology_sources/rules_registry.json`
> is thematically faithful to the source. See `CURRENT_STATE_AUDIT.md` §5.1
> and `DECISION_LOG.md` for the current, verified status. This file is kept
> as a historical record rather than rewritten or deleted.

Source expected: `التحليل النفسي للصور.pdf`.

The file was visible at the supplied Windows download path on 2026-07-19, but
the execution sandbox denied read access. No transcription, translation, or
clinical rule was fabricated. Import remains blocked until the attachment is
made readable in the writable workspace.

All future imported rules must retain page-level provenance, Arabic wording,
literal and reviewed English translations, evidence tier, confidence ceiling,
limitations, visibility policy, and clinician-review status. Presence in the
attachment does not establish scientific validation.
