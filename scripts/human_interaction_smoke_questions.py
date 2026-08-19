#!/usr/bin/env python
"""FINAL COMPLETION PASS -- Part 7 required real smoke questions.

Runs the 5 required questions through the REAL `human_interaction`
backend against h38 (development case), using whatever providers this
environment actually resolves to (`resolve_default_*` -- degrades
honestly to the deterministic/offline default wherever GEMINI_API_KEY/
google-genai are unavailable, exactly as a real deployment without a key
configured would behave). For question 3 ("What does red mean?"), ALSO
runs a second, clearly-separated demonstration with a MOCKED external
research provider (no live network in this sandboxed environment) to
prove the external-research code path itself works end to end when a
provider is available -- this second run is NOT one of the 5 official
smoke answers, it is a supplementary proof of the Part 2 routing logic.

Never writes to any case's cache; prints only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import clinician_review_app as app  # noqa: E402
from doar import human_interaction as hi  # noqa: E402

QUESTIONS = [
    "Why did you mention stress?",
    "What is the source for that rule?",
    "What does red mean?",
    "Is my child depressed?",
    "Are there people in the car?",
]


def _print_answer(question: str, answer) -> None:
    print("=" * 90)
    print(f"Q: {question}")
    print(f"route/category: {answer.category}")
    print(f"final answer: {answer.answer}")
    evidence_ids = sorted({eid for c in answer.claims for eid in c.get("evidence_ids", [])})
    rule_ids = sorted({rid for c in answer.claims for rid in c.get("rule_ids", [])})
    source_ids = sorted({sid for c in answer.claims for sid in c.get("source_ids", [])})
    print(f"evidence IDs: {evidence_ids}")
    print(f"rule IDs: {rule_ids}")
    print(f"source IDs: {source_ids}")
    print(f"source provenance: {answer.provenance}")
    print(f"answer provider: {answer.answer_provider}")
    print(f"judge mode: {answer.judge_mode}")
    print(f"judge verdict: {answer.judge_verdict}")
    if answer.judge_reasons:
        print(f"judge reasons: {answer.judge_reasons}")
    print(f"external research used: {answer.used_external_research}")
    print(f"visual re-check used: {answer.used_visual_recheck}")


def main() -> None:
    bundle = app.load_case_bundle("h38")
    if bundle is None or bundle["rows"] is None:
        raise SystemExit("h38 bundle not available.")

    answer_provider = hi.resolve_default_answer_provider()
    research_provider = hi.resolve_default_research_provider()
    judge = hi.resolve_default_judge()
    visual_recheck_provider = hi.resolve_default_visual_recheck_provider()

    print("Resolved providers for this environment:")
    print(f"  answer_provider  = {answer_provider.__class__.__name__}")
    print(f"  research_provider = {research_provider.__class__.__name__}")
    print(f"  judge            = {judge.__class__.__name__}")
    print(f"  visual_recheck_provider = "
          f"{visual_recheck_provider.__class__.__name__ if visual_recheck_provider else None}")
    print()

    results = {}
    for q in QUESTIONS:
        answer = hi.answer_question(
            q, bundle, audience="clinician",
            answer_provider=answer_provider, external_research_provider=research_provider,
            judge=judge, visual_recheck_provider=visual_recheck_provider,
        )
        _print_answer(q, answer)
        results[q] = answer.to_dict()

    # --- Supplementary (not one of the 5): mocked external research proof ---
    print("=" * 90)
    print("SUPPLEMENTARY (not one of the 5 required questions): 'What does red mean?' "
          "with a MOCKED GeminiGroundedResearchProvider, to prove the external-research "
          "routing itself (internal-miss -> external grounded research -> labeled answer) "
          "works end to end when a provider is actually available.")
    mock_response = MagicMock()
    mock_response.text = ("Some research suggests red is one of several colours children use expressively; "
                           "evidence for one single fixed psychological meaning is weak and inconsistent.")
    mock_chunk = MagicMock()
    mock_chunk.web.title = "Colour use in children's drawings: a systematic review"
    mock_chunk.web.uri = "https://example-journal.org/colour-review"
    mock_response.candidates = [MagicMock(grounding_metadata=MagicMock(grounding_chunks=[mock_chunk]))]
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    with patch("google.genai.Client", return_value=mock_client):
        mocked_research_provider = hi.GeminiGroundedResearchProvider(api_key="sandbox-mock-key")
        mocked_answer = hi.answer_question(
            "What does red mean?", bundle, audience="parent",
            external_research_provider=mocked_research_provider)
    _print_answer("What does red mean? (mocked external research available)", mocked_answer)
    results["What does red mean? (mocked external research available)"] = mocked_answer.to_dict()

    out_dir = ROOT / "outputs" / "human_interaction_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_questions_result.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
