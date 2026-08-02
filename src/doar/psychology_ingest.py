"""
psychology_ingest.py — provenance-preserving ingestion of a psychologist PDF
into a DRAFT rules registry.

Policy (from resources/psychology_sources/README.md and PSYCHOLOGIST_SOURCE_AUDIT):
  * The unchanged PDF is the source of record.
  * Ingestion NEVER activates rules. It emits a DRAFT with every rule marked
    blocked-pending-review, confidence_ceiling 0.0, and scientific_support
    "pending_literature_check".
  * Each draft rule retains: page provenance, literal Arabic wording, an empty
    literal+reviewed English slot, evidence tier, confidence ceiling, limitations,
    a visibility policy, and a review status.

A clinician then completes the English translation, evidence tier, and confidence
ceiling, and flips review_status to "reviewed" before any rule can be used.

`build_draft_registry` is pure (list[pages] -> draft dict) and fully testable.
`extract_pdf_pages` is a thin lazy wrapper over pypdf.
"""

from __future__ import annotations
import json
import re
from pathlib import Path


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Return [{"page": n, "arabic_text": "..."}]. Requires pypdf (lazy)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install pypdf to ingest PDFs: pip install pypdf") from exc
    reader = PdfReader(str(pdf_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": index, "arabic_text": text.strip()})
    return pages


def _split_statements(text: str) -> list[str]:
    """Split a page's Arabic text into candidate rule statements.
    Uses Arabic + Latin sentence terminators; keeps only non-trivial lines."""
    parts = re.split(r"[\.\n۔]+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def _slug(text: str, index: int) -> str:
    return f"PSY_DRAFT_P{index:03d}"


def build_draft_registry(pages: list[dict], source_meta: dict | None = None) -> dict:
    """Transform extracted pages into a DRAFT registry. Pure function.

    Every draft rule is inert (blocked, ceiling 0.0, review pending) so nothing
    can be activated before clinician review."""
    source = {
        "source_id": (source_meta or {}).get("source_id", "PSYCHOLOGIST_PDF_PENDING"),
        "source_type": "psychologist_supplied_pdf",
        "language": "ar",
        "scientifically_validated": False,
        "ingestion_note": (
            "Auto-extracted DRAFT. Presence in the PDF is NOT validation. Every "
            "rule is blocked pending clinician translation, literature check, and "
            "review before any activation."
        ),
        **(source_meta or {}),
    }
    rules = []
    counter = 0
    for page in pages:
        for statement in _split_statements(page.get("arabic_text", "")):
            counter += 1
            rules.append({
                "rule_id": _slug(statement, counter),
                "page": page.get("page"),
                "arabic": statement,                        # literal source wording
                "english_literal": "",                      # to be filled by clinician
                "english_reviewed": "",                     # to be filled by clinician
                "observable": "unassigned",                 # map to a detector later
                "evidence_tier": "unreviewed",
                "scientific_support": "pending_literature_check",
                "confidence_ceiling": 0.0,                  # inert until reviewed
                "limitations": ["Unreviewed auto-extraction; not for use."],
                "visibility": "blocked_pending_review",
                "review_status": "pending",
                "references": [],
            })
    return {
        "source": source,
        "references": {},
        "draft": True,
        "activation_blocked": True,
        "rule_count": len(rules),
        "rules": rules,
    }


def ingest_pdf(pdf_path: str | Path, output: str | Path,
               source_meta: dict | None = None) -> dict:
    """Extract a PDF and write a DRAFT registry JSON. Returns the draft dict."""
    pages = extract_pdf_pages(pdf_path)
    draft = build_draft_registry(pages, source_meta)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    return draft
