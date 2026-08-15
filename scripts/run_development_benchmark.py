#!/usr/bin/env python
"""DOAR 15-image DEVELOPMENT benchmark runner (rehearsal only).

Implements `BENCHMARK_SCHEMA.md` against `DEVELOPMENT_SET_15.json`:
Condition A (`direct_gemini`, raw observer candidates) and Condition B
(`doar_full_pipeline`, Observer -> Verifier -> `reasoning_chain.py`
through clinician/parent packages), both compared against the human
annotations produced by `scripts/annotate_ui.py` (one exhaustive item
list per annotator per image, each item flagged `salient` yes/no).

**This script does NOT invent annotations.** If an image has no saved
human annotation file under `annotations/<annotator>/`, its per-image
inspection folder says so plainly and its metrics are reported as
unavailable (`None`) rather than skipped silently or fabricated.

**This script does NOT call the live Gemini APIs by default** (quota
conservation). It reuses already-saved Observer/Verifier data, searched
in this order: (1) `outputs/prototype_cases/development_live_cache/`
-- this script's own persistent cache of prior `--run-live` results,
keyed by `image_id`, so a live call is NEVER repeated for an image once
made; (2) `outputs/prototype_cases/gemini_verifier_dev_check_*/` -- the
earlier, pre-existing dev-check data. Pass `--run-live` to call the real
Observer+Verifier for any image found in neither location; each
successful live result is written to the persistent cache immediately
(atomically -- write-to-temp-file then rename), before moving on to the
next image, so a later run (or a crash partway through this one) never
re-spends quota on an image already completed.

**These 15 images are development-only** (`DEVELOPMENT_SET_15.json`'s own
note) -- results here may surface implementation bugs to fix; they must
NEVER be used to tune `RULE_EVIDENCE_MATRIX.csv`, `CONCERN_DOMAIN_MAP.
json`, or any other frozen file (`BENCHMARK_SCHEMA.md` Section 6), and
this script never writes to any of them.

Usage:
    python scripts/run_development_benchmark.py
    python scripts/run_development_benchmark.py --run-live   # NOT used this phase
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar import benchmark_metrics as bm  # noqa: E402
from doar import reasoning_chain as rc  # noqa: E402
from doar.visual_entity import VisualEntity  # noqa: E402

DEV_SET_PATH = ROOT / "DEVELOPMENT_SET_15.json"
ANNOTATIONS_DIR = ROOT / "annotations"
ANNOTATOR_IDS = ("A1", "A2")

# This script's own persistent cache of `--run-live` results, keyed by
# image_id -- checked BEFORE the legacy dev-check dirs below, and the
# only location `--run-live` ever writes a fresh result to. Exists so a
# live Gemini call is made AT MOST ONCE per image across every future
# invocation of this script, not just within a single run.
LIVE_CACHE_DIR = ROOT / "outputs" / "prototype_cases" / "development_live_cache"

# Already-saved, real Observer/Verifier development data -- searched newest
# first so a later dev-check run (if one is ever added) takes precedence.
# Zero new API calls are made when data is found here.
SAVED_VERIFICATION_DIRS = sorted(
    (ROOT / "outputs" / "prototype_cases").glob("gemini_verifier_dev_check_*"),
    reverse=True,
)

RUN_STAMP = None  # set in main(), threaded through so every artifact of one run lands together


def load_development_set() -> list[dict]:
    return json.loads(DEV_SET_PATH.read_text(encoding="utf-8"))["images"]


def live_cache_path(image_id: str) -> Path:
    return LIVE_CACHE_DIR / f"{image_id}_verification.json"


def _atomic_write_json(path: Path, data) -> None:
    """Writes `data` to `path` atomically: serialize to a sibling temp
    file in the SAME directory, then `os.replace` it over the
    destination (an atomic rename on both POSIX and Windows). A crash or
    interruption mid-write can therefore never leave a half-written,
    corrupt cache file at `path` -- the file at `path` is always either
    the previous complete version or the new complete version, never a
    partial one. Used for the live cache specifically because its
    content is expensive (real Gemini API quota) to reproduce; the
    other, cheap-to-regenerate report files this script writes are not
    a correctness concern the same way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def find_saved_verification_rows(image_id: str) -> tuple[list[dict] | None, Path | None]:
    """Search order: (1) this script's own persistent live cache --
    checked FIRST so a `--run-live` result already captured (even in a
    prior invocation) is always reused and never re-fetched; (2) the
    legacy `gemini_verifier_dev_check_*` dev-check directories."""
    live_path = live_cache_path(image_id)
    if live_path.exists():
        return json.loads(live_path.read_text(encoding="utf-8")), live_path
    for d in SAVED_VERIFICATION_DIRS:
        f = d / f"{image_id}_verification.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8")), f
    return None, None


def entities_from_verification_rows(rows: list[dict]) -> list[VisualEntity]:
    """Same field conventions as the prior Observer/Verifier dev-check
    smoke test (`REASONING_CHAIN_IMPLEMENTATION_REPORT.md`) -- one
    VisualEntity per verification row, canonical_label/aliases carrying
    both the observer's and the verifier's independent labels so
    `reasoning_chain.py`'s whole-word matcher sees everything either
    model called this object."""
    entities = []
    for row in rows:
        oc = row["observer_candidate"]
        aliases = (
            tuple(oc.get("alternative_labels") or ())
            + (row.get("verifier_independent_label") or "",)
            + tuple(row.get("verifier_independent_alternative_labels") or ())
        )
        aliases = tuple(a for a in aliases if a)
        entities.append(VisualEntity(
            entity_id=row["observation_id"], entity_type=oc.get("entity_type", "unknown"),
            canonical_label=oc["label"], candidate_labels=((oc["label"], oc.get("confidence") or 0.0),),
            aliases_en=aliases, aliases_ar=(),
            broader_categories=(), possible_subtypes=(), visual_similarities=(),
            bbox=tuple(oc["bbox"]) if oc.get("bbox") else None, crop_ref=None, dominant_colors=None,
            relative_size=None, page_position=None, shape_features=None, line_features=None,
            detector=f"visual_observer:{row.get('observer_model', '')}", checkpoint="", prompt="",
            confidence=oc.get("confidence") or 0.0, model_validation_status="UNKNOWN",
            case_verification_status=row["verification_status"],
            evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
            related_rule_ids=(), source="visual_observer", query=None,
            timestamp="2026-08-13T00:00:00+00:00",
        ))
    return entities


def run_live_observer_and_verifier(image_id: str, image_path: Path) -> list[dict]:
    """NOT exercised this phase (no --run-live invocation was made). Kept
    minimal and using the exact same call pattern as
    `scripts/doar_visual_verifier_dev_check.py` so a future run behaves
    identically to that already-validated path -- not a new, untested
    integration."""
    from doar.visual_observer import GeminiVisualObserver, GeminiVisualVerifier

    observer = GeminiVisualObserver()
    observer_result = observer.observe(str(image_path))
    verifier = GeminiVisualVerifier()
    rows = []
    for i, candidate in enumerate(observer_result.candidates):
        observation_id = f"{image_id}_c{i:02d}"
        bbox = tuple(candidate.bbox) if candidate.bbox else None
        entity = VisualEntity(
            entity_id=f"ve_{observation_id}", entity_type=candidate.entity_type,
            canonical_label=candidate.label, candidate_labels=((candidate.label, candidate.confidence or 0.0),),
            aliases_en=tuple(candidate.alternative_labels or ()), aliases_ar=(),
            broader_categories=(), possible_subtypes=(), visual_similarities=(),
            bbox=bbox, crop_ref=None, dominant_colors=None, relative_size=None, page_position=None,
            shape_features=None, line_features=None, detector="visual_observer", checkpoint="", prompt="",
            confidence=candidate.confidence or 0.0, model_validation_status="UNKNOWN",
            case_verification_status="unreviewed", evidence_status="experimental_evidence_technical_view_only",
            rule_mapping_status="UNMAPPED", related_rule_ids=(), source="visual_observer", query=None,
            timestamp="2026-08-13T00:00:00+00:00",
        )
        result = verifier.verify(str(image_path), entity)
        time.sleep(3.0)
        rows.append({
            "drawing_id": image_id, "observation_id": observation_id,
            "observer_candidate": {
                "label": candidate.label, "alternative_labels": list(candidate.alternative_labels or ()),
                "bbox": candidate.bbox, "confidence": candidate.confidence, "entity_type": candidate.entity_type,
            },
            "verifier_independent_label": result.independent_label,
            "verifier_independent_alternative_labels": list(result.independent_alternative_labels),
            "verifier_confidence": result.confidence, "verification_status": result.status,
            "verifier_notes": result.notes, "observer_model": observer.model, "verifier_model": verifier.model,
            "verifier_source_note": result.source_note, "runtime_seconds": None,
        })
    return rows


def load_human_annotations(image_id: str) -> dict:
    """annotator_id -> list[HumanAnnotationItem] | None (None = that
    annotator has no saved file yet for this image). One exhaustive item
    list per (annotator, image) -- `annotations/<annotator_id>/
    <image_id>.json`, written by `scripts/annotate_ui.py`. Never merges
    annotators -- each annotator's data is kept fully separate all the
    way through, per the parent task's explicit instruction."""
    result = {}
    for annotator_id in ANNOTATOR_IDS:
        path = ANNOTATIONS_DIR / annotator_id / f"{image_id}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            result[annotator_id] = [
                bm.HumanAnnotationItem(
                    label=i["label"], location=i.get("location", ""), salient=bool(i.get("salient", False)),
                    confidence=i.get("confidence", "clear"), note=i.get("note", ""),
                )
                for i in record["items"]
            ]
        else:
            result[annotator_id] = None
    return result


def compute_visual_metrics(candidate_labels: list[str], candidates_full: list[dict], human: dict) -> dict:
    """Per-annotator visual metrics -- NEVER a merged-annotator number.
    `human` is the dict `load_human_annotations` returns for one image:
    annotator_id -> list[HumanAnnotationItem] | None. `exhaustive
    reference = all annotated items`; `salient reference = items where
    salient=True` (BENCHMARK_SCHEMA.md)."""
    per_annotator = {}
    for annotator_id, items in human.items():
        if items is None:
            per_annotator[annotator_id] = {"precision_recall_f1": None, "salient_recall": None}
            continue
        per_annotator[annotator_id] = {
            "precision_recall_f1": bm.visual_precision_recall_f1(candidate_labels, items),
            "salient_recall": bm.salient_recall(candidate_labels, bm.salient_items(items)),
        }

    all_items = [item for items in human.values() if items for item in items]
    hallucination = bm.hallucination_rate(candidate_labels, all_items) if all_items else None

    abstention = bm.abstention_rate(candidates_full)
    correction = bm.verifier_correction_rate(candidates_full)

    agreement = None
    salience = None
    a1, a2 = human.get("A1"), human.get("A2")
    if a1 is not None and a2 is not None:
        agreement = bm.inter_annotator_agreement(a1, a2)
        # Frozen salience definitions (BENCHMARK_SCHEMA.md Section 5):
        # SALIENT = union of both annotators' salient items, CORE_SALIENT
        # = intersection. Reported ALONGSIDE, never instead of, each
        # annotator's own salient_recall in `per_annotator` above.
        salience = bm.primary_and_sensitivity_salient_recall(candidate_labels, a1, a2)

    return {
        "per_annotator": per_annotator,
        "salience": salience,
        "hallucination_rate": hallucination,
        "abstention_rate": abstention,
        "verifier_correction_rate": correction,
        "inter_annotator_agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Condition registry -- deliberately just the two BENCHMARK_SCHEMA.md
# defines today. A future C2/C3 ablation adds an entry here (e.g. "doar_
# no_verifier"); nothing else in this script needs to change to support
# that later -- not built out now, per this phase's own scope limit.
# ---------------------------------------------------------------------------

CONDITIONS = ("direct_gemini", "doar_full_pipeline")


def run_conditions_for_image(image_id: str, rows: list[dict]) -> dict:
    entities = entities_from_verification_rows(rows)
    direct_gemini_labels = [row["observer_candidate"]["label"] for row in rows]

    verified_entities = [e for e in entities if e.case_verification_status == "verified"]
    doar_labels = [e.canonical_label for e in verified_entities]

    checks = rc.check_visual_preconditions(entities)
    satisfied_checks = [c for c in checks if c.status == "satisfied"]
    matches = rc.build_eligible_matches(entities)
    families = rc.deduplicate_by_evidence_family(matches)
    domains = rc.aggregate_by_concern_domain(matches)
    hypotheses = rc.build_candidate_hypotheses(matches)

    packages = []
    for h in hypotheses:
        packages.append({
            "hypothesis": h,
            "clinician_package": rc.build_clinician_package(h),
            "parent_package": rc.build_parent_package(h),
            "llm_writer_payload_clinician": rc.build_llm_writer_payload(h, audience="clinician", case_id=image_id),
            "llm_writer_payload_parent": rc.build_llm_writer_payload(h, audience="parent", case_id=image_id),
        })

    return {
        "entities": entities,
        "direct_gemini": {"raw_candidate_labels": direct_gemini_labels, "candidates_full": [row["observer_candidate"] | {"case_verification_status": row["verification_status"]} for row in rows]},
        "doar_full_pipeline": {
            "verified_entities": verified_entities, "verified_labels": doar_labels,
            "satisfied_preconditions": satisfied_checks, "eligible_matches": matches,
            "evidence_families": families, "concern_domains": domains,
            "hypotheses": hypotheses, "packages": packages,
        },
    }


def _entity_to_dict(e: VisualEntity) -> dict:
    return {"entity_id": e.entity_id, "canonical_label": e.canonical_label, "aliases_en": list(e.aliases_en),
            "entity_type": e.entity_type, "bbox": list(e.bbox) if e.bbox else None, "confidence": e.confidence,
            "case_verification_status": e.case_verification_status}


def _match_to_dict(m: rc.EligibleAtomicRuleMatch) -> dict:
    return {"rule_id": m.rule_id, "evidence_family": m.evidence_family, "concern_domain": m.concern_domain,
            "allowed_output_level": m.allowed_output_level, "source_claim": m.source_claim,
            "matched_entity_ids": list(m.matched_entity_ids), "evidence_direction": m.evidence_direction}


def _hypothesis_to_dict(h: rc.CandidateHypothesis) -> dict:
    return {"concern_domain": h.concern_domain, "hypothesis_label": h.hypothesis_label,
            "support_level": h.support_level, "evidence_families_represented": list(h.evidence_families_represented),
            "supporting_rule_ids": list(h.supporting_rule_ids), "supporting_entity_ids": list(h.supporting_entity_ids),
            "contradicting_reference_ids": list(h.contradicting_reference_ids),
            "alternative_explanations": list(h.alternative_explanations),
            "missing_clinical_information": list(h.missing_clinical_information), "disclaimer": h.disclaimer}


def _annotation_item_to_dict(i: bm.HumanAnnotationItem) -> dict:
    return {"label": i.label, "location": i.location, "salient": i.salient, "confidence": i.confidence, "note": i.note}


def write_inspection_folder(out_dir: Path, image: dict, source_path: Path | None,
                             conditions: dict | None, human: dict, metrics: dict | None,
                             reasoning_metrics: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_id = image["image_id"]

    if source_path is not None:
        (out_dir / "SOURCE_DATA.txt").write_text(
            f"Observer/Verifier data reused from: {source_path}\nOriginal image: {ROOT / image['relative_path']}\n",
            encoding="utf-8")

    human_json = {aid: ([_annotation_item_to_dict(i) for i in items] if items is not None else None)
                  for aid, items in human.items()}
    (out_dir / "human_annotations.json").write_text(json.dumps(human_json, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [f"# Development benchmark inspection -- {image_id}", "",
                f"Original image: `{image['relative_path']}`", ""]

    if conditions is None:
        md_lines += ["## Status", "",
                     "No saved Observer/Verifier data found for this image, and `--run-live` was not "
                     "requested for this run -- Conditions A/B were not prepared.", ""]
        (out_dir / "trace.json").write_text(json.dumps({"status": "no_observer_data"}, indent=2), encoding="utf-8")
    else:
        trace = {
            "image_id": image_id,
            "entities": [_entity_to_dict(e) for e in conditions["entities"]],
            "direct_gemini": {"raw_candidate_labels": conditions["direct_gemini"]["raw_candidate_labels"]},
            "doar_full_pipeline": {
                "verified_labels": conditions["doar_full_pipeline"]["verified_labels"],
                "satisfied_preconditions": [{"rule_id": c.rule_id, "reason": c.reason,
                                              "matched_entity_ids": list(c.matched_entity_ids)}
                                             for c in conditions["doar_full_pipeline"]["satisfied_preconditions"]],
                "eligible_matches": [_match_to_dict(m) for m in conditions["doar_full_pipeline"]["eligible_matches"]],
                "evidence_families": {k: [m.rule_id for m in v] for k, v in conditions["doar_full_pipeline"]["evidence_families"].items()},
                "concern_domains": {k: [m.rule_id for m in v] for k, v in conditions["doar_full_pipeline"]["concern_domains"].items()},
                "hypotheses": [_hypothesis_to_dict(h) for h in conditions["doar_full_pipeline"]["hypotheses"]],
            },
        }
        (out_dir / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

        for i, pkg in enumerate(conditions["doar_full_pipeline"]["packages"]):
            (out_dir / f"clinician_package_{i}.json").write_text(json.dumps(pkg["clinician_package"], indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / f"parent_package_{i}.json").write_text(json.dumps(pkg["parent_package"], indent=2, ensure_ascii=False), encoding="utf-8")

        md_lines += [
            "## Condition A -- Direct Gemini (raw observer candidates)", "",
            "Labels: " + (", ".join(conditions["direct_gemini"]["raw_candidate_labels"]) or "(none)"), "",
            "## Condition B -- Full DOAR pipeline", "",
            f"Verified entities ({len(conditions['doar_full_pipeline']['verified_labels'])}): " +
            (", ".join(conditions["doar_full_pipeline"]["verified_labels"]) or "(none)"), "",
            f"Eligible rule matches: {len(conditions['doar_full_pipeline']['eligible_matches'])}",
        ]
        for m in conditions["doar_full_pipeline"]["eligible_matches"]:
            md_lines.append(f"  - `{m.rule_id}` ({m.evidence_family} / {m.concern_domain}) <- {list(m.matched_entity_ids)}")
        md_lines += ["", f"Candidate hypotheses: {len(conditions['doar_full_pipeline']['hypotheses'])}"]
        for h in conditions["doar_full_pipeline"]["hypotheses"]:
            md_lines.append(f"  - **{h.concern_domain}** ({h.support_level}): {h.hypothesis_label!r}, rules={list(h.supporting_rule_ids)}")
        md_lines.append("")

    md_lines += ["## Human annotations", ""]
    any_human = False
    for annotator_id, items in human.items():
        if items is None:
            md_lines.append(f"- {annotator_id}: **not yet recorded**")
            continue
        any_human = True
        all_labels = ", ".join(i.label for i in items) or "(none)"
        salient_labels = ", ".join(i.label for i in items if i.salient) or "(none)"
        md_lines.append(f"- {annotator_id}: {len(items)} item(s) total")
        md_lines.append(f"    - all: {all_labels}")
        md_lines.append(f"    - salient: {salient_labels}")
    if not any_human:
        md_lines.append("")
        md_lines.append("(No human annotations exist yet for this image -- see repo root for how to start annotating.)")
    md_lines.append("")

    md_lines += ["## Visual metrics", ""]
    if metrics is None:
        md_lines.append("Not computed -- requires human annotations, none available for this image yet.")
    else:
        md_lines.append("```json")
        md_lines.append(json.dumps(_metrics_json_safe(metrics), indent=2))
        md_lines.append("```")
    (out_dir / "SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")

    if metrics is not None:
        (out_dir / "visual_metrics.json").write_text(json.dumps(_metrics_json_safe(metrics), indent=2), encoding="utf-8")


def _metrics_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _metrics_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_metrics_json_safe(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="DOAR 15-image DEVELOPMENT benchmark runner (rehearsal only).")
    parser.add_argument("--run-live", action="store_true",
                         help="Call the real Gemini Observer+Verifier for images with no saved data. "
                              "NOT used during this phase -- quota conservation.")
    args = parser.parse_args()

    out_root = ROOT / "outputs" / "prototype_cases" / f"development_benchmark_{int(time.time())}"
    out_root.mkdir(parents=True, exist_ok=True)

    images = load_development_set()
    rule_matrix = rc.load_rule_matrix()

    per_image_results = []
    hypotheses_by_image = {}
    all_matches = []
    all_hypotheses = []

    for i, image in enumerate(images, start=1):
        image_id = image["image_id"]
        print(f"[{i}/{len(images)}] {image_id}...", flush=True)
        image_out_dir = out_root / image_id

        rows, source_path = find_saved_verification_rows(image_id)
        if rows is None and args.run_live:
            image_path = ROOT / image["relative_path"]
            rows = run_live_observer_and_verifier(image_id, image_path)
            live_path = live_cache_path(image_id)
            _atomic_write_json(live_path, rows)
            print(f"    cached live Observer/Verifier result -> {live_path}", flush=True)
            # Also keep a copy scoped to this run's own output folder, as before.
            (out_root / f"{image_id}_verification_live.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            source_path = live_path

        # Loaded and counted unconditionally, regardless of whether
        # Observer/Verifier data exists for this image -- annotations and
        # observer data are independent facts about an image, and missing
        # one must never make the other appear missing too.
        human = load_human_annotations(image_id)
        any_annotation = any(items is not None for items in human.values())

        if rows is None:
            print(f"    no saved Observer/Verifier data for {image_id}; skipping Conditions A/B "
                  f"(pass --run-live to fetch live).")
            write_inspection_folder(image_out_dir, image, None, None, human, None, None)
            per_image_results.append({
                "image_id": image_id, "status": "no_observer_data",
                "has_human_annotations": any_annotation,
            })
            hypotheses_by_image[image_id] = []
            continue

        conditions = run_conditions_for_image(image_id, rows)
        hypotheses_by_image[image_id] = conditions["doar_full_pipeline"]["hypotheses"]
        all_matches.extend(conditions["doar_full_pipeline"]["eligible_matches"])
        all_hypotheses.extend(conditions["doar_full_pipeline"]["hypotheses"])

        metrics = None
        if any_annotation:
            metrics = compute_visual_metrics(
                conditions["direct_gemini"]["raw_candidate_labels"],
                conditions["direct_gemini"]["candidates_full"], human)

        write_inspection_folder(image_out_dir, image, source_path, conditions, human, metrics, None)
        per_image_results.append({
            "image_id": image_id, "status": "ok",
            "eligible_matches": len(conditions["doar_full_pipeline"]["eligible_matches"]),
            "hypotheses": len(conditions["doar_full_pipeline"]["hypotheses"]),
            "has_human_annotations": any_annotation,
        })

    reasoning_summary = {
        "evidence_backed_claim_rate": bm.evidence_backed_claim_rate(all_matches),
        "unsupported_claim_rate": bm.unsupported_claim_rate(all_hypotheses, rule_matrix),
        "rule_reference_traceability": bm.rule_reference_traceability(all_hypotheses, rule_matrix),
        "hypothesis_derivability": bm.hypothesis_derivability(hypotheses_by_image),
    }

    images_with_annotations = sum(1 for r in per_image_results if r.get("has_human_annotations"))
    images_with_observer_data = sum(1 for r in per_image_results if r["status"] == "ok")

    top_summary = {
        "note": ("DEVELOPMENT REHEARSAL ONLY -- not the 100-image held-out experiment. Results here "
                 "may surface implementation bugs; they must never tune RULE_EVIDENCE_MATRIX.csv, "
                 "CONCERN_DOMAIN_MAP.json, or any other frozen file (BENCHMARK_SCHEMA.md Section 6)."),
        "total_images": len(images),
        "images_with_saved_or_live_observer_data": images_with_observer_data,
        "images_with_human_annotations": images_with_annotations,
        "per_image": per_image_results,
        "reasoning_metrics": reasoning_summary,
        "output_directory": str(out_root),
    }
    (out_root / "summary.json").write_text(json.dumps(_metrics_json_safe(top_summary), indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = [
        "# DOAR 15-image development benchmark -- run summary", "",
        f"Images with Observer/Verifier data available: {images_with_observer_data}/{len(images)}",
        f"Images with at least one human annotator's data: {images_with_annotations}/{len(images)}",
        "", "## Reasoning metrics (Condition B, aggregated)", "",
        "```json", json.dumps(_metrics_json_safe(reasoning_summary), indent=2), "```", "",
        "See each `<image_id>/SUMMARY.md` for a per-drawing, code-free breakdown.",
    ]
    (out_root / "SUMMARY.md").write_text("\n".join(summary_md), encoding="utf-8")

    print(f"\nSaved run to: {out_root}")
    print(json.dumps(_metrics_json_safe(top_summary), indent=2, ensure_ascii=False))

    if images_with_annotations == 0:
        print("\nANNOTATION READY: YES (no real human annotations exist yet -- run "
              "streamlit run scripts/annotate_ui.py -- --annotator A1 (and A2) first, "
              "then re-run this benchmark).")


if __name__ == "__main__":
    main()
