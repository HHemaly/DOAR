#!/usr/bin/env python
"""E6 -- Rule / Evidence Aggregation Ablation.

Research question: which rule/evidence aggregation strategy gives the
best trade-off between useful supported coverage and resistance to
duplicate, cross-domain, or otherwise unsupported convergence?

This script is the SOLE source of every number in this experiment's
tables/figures -- re-running it against the same frozen 15-case
development-set cache reproduces every output byte-for-byte (no
randomness is used anywhere except the bootstrap resampling, which is
seeded).

SCOPE / WHAT THIS SCRIPT DOES NOT DO:
  - Makes ZERO live Gemini/API calls. Every number comes from already-
    cached Observer/Verifier JSON + the frozen RULE_EVIDENCE_MATRIX.csv /
    RULE_RELATIONSHIP_GRAPH.json / CONCERN_DOMAIN_MAP.json, read via the
    SAME loaders `scripts/clinician_review_app.py` and `src/doar/
    reasoning_chain.py` / `drawing_synthesis.py` already use in
    production -- this experiment never re-implements rule matching, it
    only re-AGGREGATES the same already-computed EligibleAtomicRuleMatch
    objects five different ways.
  - Never changes RULE_EVIDENCE_MATRIX.csv, RULE_RELATIONSHIP_GRAPH.json,
    CONCERN_DOMAIN_MAP.json, concerns.py's MIN_EVIDENCE, or any frozen
    threshold. R1-R5 are read-only re-aggregations of the SAME
    EligibleAtomicRuleMatch list `drawing_synthesis.synthesize_drawing`
    already produces for each cached case.

Five aggregation strategies (see PROTOCOL.md for the full rationale of
each design choice):
  R1  Any eligible rule            -- domain triggers if >=1 satisfied rule.
  R2  Raw rule-count convergence   -- domain triggers if >=2 satisfied
      rules, WITHOUT evidence-family deduplication (duplicate-family
      rules count as independent evidence -- the strawman this
      experiment expects to show duplicate-inflation on).
  R3  Pooled (cross-domain) evidence-family convergence -- >=2 distinct
      evidence families within one DIRECTION (positive/concern),
      POOLED ACROSS ALL DOMAINS in that direction -- reproduces the
      documented pre-acceptance-audit-fix DOAR behaviour
      (drawing_synthesis.py's own module docstring calls this "the
      original synthesis's over-claiming bug").
  R4  Same-domain independent evidence-family convergence -- CURRENT
      DOAR, `drawing_synthesis.build_overall_synthesis` unchanged.
  R5  Support ratio -- independent matched families / independent
      ASSESSABLE families, WITHIN one domain (same domain scope as R4);
      triggers at ratio >= 0.5, deliberately WITHOUT R4's own >=2
      absolute-family floor, to test whether normalizing by what could
      even be assessed changes the coverage/over-claiming trade-off.
  R6 (weighted aggregation) is NOT implemented -- see the data-readiness
      audit section of README.md: `confidence_ceiling` is
      "not_set_pending_review" for 18/41 rules (44%) and
      `evidence_strength_as_written` is free text with no defined
      numeric mapping. Inventing weights now would be exactly the
      post-hoc tuning this task explicitly forbids.

Unit of analysis: one row per (case_id, concern_domain, strategy).
R1/R2/R4/R5 are domain-local. R3 additionally computes a POOLED
direction-level family count (positive_affect_or_social_engagement is
its own direction; the five distress domains are pooled together as one
"concern" direction) and every domain row within that direction shares
that pooled trigger -- exactly reproducing what the pre-fix pooled logic
would have done to each domain's own read.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import clinician_review_app as app  # noqa: E402
from doar import reasoning_chain as rc  # noqa: E402
from doar import drawing_synthesis as ds  # noqa: E402

RNG_SEED = 20260819  # fixed -- the date this experiment was first run; never changed between runs
N_BOOTSTRAP = 10000
MIN_EVIDENCE = 2  # concerns.py's own frozen constant, reused verbatim -- never redefined here
R5_RATIO_THRESHOLD = 0.5  # documented in PROTOCOL.md: chosen BEFORE running comparisons, the
                          # only non-arbitrary "majority" cutoff; never tuned against results

POSITIVE_DOMAINS = frozenset({"positive_affect_or_social_engagement"})
DISTRESS_DOMAINS = frozenset({
    "depressive_or_low_mood_related", "anxiety_or_stress_related", "aggression_or_threat_related",
    "social_withdrawal_related", "developmental_or_attention_related",
})
NEUTRAL_DOMAINS = frozenset({"neutral_descriptive_only", "neutral_descriptive_only_no_construct_proposed"})

ASSESSABLE_STATUSES = {"satisfied", "not_satisfied"}  # structurally checkable for THIS case;
    # excludes blocked_structural / blocked_needs_unbuilt_feature / not_applicable_to_this_module
    # (semantic) and not_assessable (deterministic, page not confirmed) -- those could not even
    # be attempted for this case, so they are never counted as "assessable but unmatched".

STRATEGIES = ["R1", "R2", "R3", "R4", "R5"]


def direction_of(domain: str) -> str | None:
    if domain in POSITIVE_DOMAINS:
        return "positive"
    if domain in DISTRESS_DOMAINS:
        return "concern"
    return None  # neutral -- no direction, R3 does not apply


# ---------------------------------------------------------------------------
# 1. Load every cached case, extract the SAME EligibleAtomicRuleMatch list
#    drawing_synthesis.synthesize_drawing already computed for it (no
#    re-implementation of rule matching), plus the full (satisfied AND
#    unsatisfied) precondition-check list needed for R5's "assessable"
#    denominator.
# ---------------------------------------------------------------------------


def load_all_cases() -> dict:
    matrix = rc.load_rule_matrix()
    graph = json.load(open(ROOT / "RULE_RELATIONSHIP_GRAPH.json", encoding="utf-8"))
    context_limited_rule_ids = set()
    for edge in graph["context_limited_by_edges"]:
        context_limited_rule_ids.update(edge["rule_ids"])

    cases = {}
    for cid in app.list_case_ids():
        bundle = app.load_case_bundle(cid)
        if bundle is None or bundle["rows"] is None or bundle["synthesis"] is None:
            cases[cid] = None
            continue

        synth = bundle["synthesis"]
        assocs = synth.literature_linked_associations  # == satisfied semantic + deterministic matches

        # Full precondition status per rule (satisfied AND not) -- needed for R5's
        # "assessable" denominator. Semantic: bundle["checks"] (already computed by
        # load_case_bundle via rc.check_visual_preconditions on cached entities --
        # zero Gemini calls). Deterministic: recomputed here the SAME way
        # drawing_synthesis.synthesize_drawing does, from the cached deterministic
        # feature cache -- also zero Gemini/model calls, pure arithmetic on cached
        # numbers.
        semantic_checks = {c.rule_id: c.status for c in (bundle["checks"] or [])}
        det = bundle["deterministic_features"]
        det_checks = ds.check_deterministic_preconditions(
            det["composition"], det["objective_features"], det.get("page_reference"))
        det_status = {c.rule_id: c.status for c in det_checks}
        all_status = {**semantic_checks, **det_status}

        entities_by_id = {e.entity_id: e for e in (bundle["entities"] or [])}
        page_ref = det.get("page_reference") or {}
        page_assessable = bool(page_ref.get("page_relative_features_assessable"))

        cases[cid] = {
            "assocs": assocs, "all_status": all_status, "matrix": matrix,
            "entities_by_id": entities_by_id, "context_limited_rule_ids": context_limited_rule_ids,
            "page_assessable": page_assessable,
        }
    return cases


# ---------------------------------------------------------------------------
# 2. Raw atomic evidence table (one row per matched rule per case) --
#    exactly what Section 6's "raw/input_atomic_evidence.csv" asks for.
# ---------------------------------------------------------------------------


def build_atomic_evidence_rows(cases: dict) -> list[dict]:
    rows = []
    for cid, data in cases.items():
        if data is None:
            continue
        for a in data["assocs"]:
            matched_entity_ids = a["matched_entity_ids"]
            statuses = []
            for eid in matched_entity_ids:
                ent = data["entities_by_id"].get(eid)
                if ent is not None:
                    statuses.append(ent.case_verification_status)
            verification_status = (
                "deterministic_measurement" if not statuses
                else ("verified" if all(s == "verified" for s in statuses) else "/".join(sorted(set(statuses))))
            )
            rows.append({
                "case_id": cid, "rule_id": a["rule_id"], "evidence_family": a["evidence_family"],
                "concern_domain": a["concern_domain"], "direction_bucket": direction_of(a["concern_domain"]) or "neutral",
                "evidence_direction": a["evidence_direction"], "allowed_output_level": a["allowed_output_level"],
                "matched_entity_ids": ";".join(matched_entity_ids),
                "verification_status": verification_status,
                "is_contradiction_flagged": a["rule_id"] in data["context_limited_rule_ids"],
            })
    return rows


# ---------------------------------------------------------------------------
# 3. Per (case, domain) aggregation under all 5 strategies.
# ---------------------------------------------------------------------------


def _assessable_families_for_domain(all_status: dict, matrix: dict, domain: str) -> set[str]:
    fams = set()
    for rule_id, row in matrix.items():
        if row["concern_domain"] != domain:
            continue
        if all_status.get(rule_id) in ASSESSABLE_STATUSES:
            fams.add(row["evidence_family"])
    return fams


def build_case_domain_rows(cases: dict) -> list[dict]:
    rows = []
    for cid, data in cases.items():
        if data is None:
            continue
        assocs = data["assocs"]
        matrix = data["matrix"]
        all_status = data["all_status"]

        by_domain: dict[str, list[dict]] = {}
        for a in assocs:
            by_domain.setdefault(a["concern_domain"], []).append(a)

        # Pooled, direction-level family sets for R3 (the pre-fix "bug" behaviour).
        pooled_families = {"positive": set(), "concern": set()}
        pooled_domains_contributing = {"positive": set(), "concern": set()}
        for a in assocs:
            direction = direction_of(a["concern_domain"])
            if direction is None:
                continue
            pooled_families[direction].add(a["evidence_family"])
            pooled_domains_contributing[direction].add(a["concern_domain"])
        pooled_family_count = {d: len(f) for d, f in pooled_families.items()}
        pooled_triggered = {d: pooled_family_count[d] >= MIN_EVIDENCE for d in ("positive", "concern")}

        for domain, domain_matches in by_domain.items():
            direction = direction_of(domain)
            rule_ids = sorted({m["rule_id"] for m in domain_matches})
            fam_counts = Counter(m["evidence_family"] for m in domain_matches)
            unique_families = sorted(fam_counts)
            duplicate_family_rule_count = sum(c - 1 for c in fam_counts.values() if c > 1)
            has_duplicate_inflation = duplicate_family_rule_count > 0

            assessable_families = _assessable_families_for_domain(all_status, matrix, domain)
            support_ratio = (len(unique_families) / len(assessable_families)) if assessable_families else None

            contradictions = sorted({m["rule_id"] for m in domain_matches} & data["context_limited_rule_ids"])

            entities_by_id = data["entities_by_id"]
            unverified_used = False
            for m in domain_matches:
                for eid in m["matched_entity_ids"]:
                    ent = entities_by_id.get(eid)
                    if ent is not None and ent.case_verification_status != "verified":
                        unverified_used = True

            base = {
                "case_id": cid, "concern_domain": domain, "direction_bucket": direction or "neutral",
                "eligible_rule_ids": ";".join(rule_ids), "matched_rule_count": len(rule_ids),
                "evidence_ids": ";".join(sorted({eid for m in domain_matches for eid in m["matched_entity_ids"]})),
                "unique_evidence_families": ";".join(unique_families), "unique_family_count": len(unique_families),
                "assessable_family_count": len(assessable_families),
                "support_ratio": support_ratio,
                "verification_status": "uses_unverified_evidence" if unverified_used else "all_verified_or_deterministic",
                "contradictions": ";".join(contradictions),
                "has_contradiction": bool(contradictions),
                "page_assessability": data["page_assessable"],
            }

            for strategy in STRATEGIES:
                row = dict(base)
                row["strategy"] = strategy
                if strategy == "R1":
                    triggered = len(rule_ids) >= 1
                    reason_trigger = "at least one eligible rule matched in this domain" if triggered else None
                    duplicate_inflation = False
                    cross_domain = False
                    struct_families_used = unique_families
                elif strategy == "R2":
                    triggered = len(rule_ids) >= MIN_EVIDENCE
                    reason_trigger = (f"{len(rule_ids)} raw matched rules (no family dedup) >= "
                                       f"MIN_EVIDENCE={MIN_EVIDENCE}") if triggered else None
                    duplicate_inflation = triggered and has_duplicate_inflation
                    cross_domain = False
                    struct_families_used = unique_families
                elif strategy == "R3":
                    if direction is None:
                        # Same domain-type restriction as R4/R5 (neutral domains can never
                        # participate in convergence in any of R3/R4/R5 -- deliberately NOT
                        # falling back to an R1-like single-rule trigger here, which would
                        # silently count as spurious "coverage" for a metric that is
                        # supposed to measure genuine multi-family convergence).
                        triggered = False
                        reason_trigger = None
                        duplicate_inflation = False
                        cross_domain = False
                        struct_families_used = unique_families
                    else:
                        triggered = pooled_triggered[direction]
                        contributing_domains = pooled_domains_contributing[direction]
                        cross_domain = triggered and len(contributing_domains) > 1
                        reason_trigger = (
                            f"pooled {direction} direction reached {pooled_family_count[direction]} independent "
                            f"families across domains {sorted(contributing_domains)} (>=MIN_EVIDENCE={MIN_EVIDENCE}); "
                            f"this domain's own share attributed") if triggered else None
                        duplicate_inflation = False
                        struct_families_used = sorted(pooled_families[direction])
                elif strategy == "R4":
                    # Matches drawing_synthesis.build_overall_synthesis exactly:
                    # _domain_scoped_convergence is only ever called on positive_matches/
                    # concern_matches -- a neutral domain can NEVER reach "convergent" in
                    # real DOAR, regardless of how many families it has (it can only ever
                    # produce "descriptive_only"). Reproduced here as a hard direction gate.
                    triggered = direction is not None and len(unique_families) >= MIN_EVIDENCE
                    reason_trigger = (f"{len(unique_families)} independent evidence families WITHIN this domain "
                                       f">= MIN_EVIDENCE={MIN_EVIDENCE}") if triggered else None
                    duplicate_inflation = False
                    cross_domain = False
                    struct_families_used = unique_families
                elif strategy == "R5":
                    # Same domain-type restriction as R4 (positive/concern only) -- this
                    # keeps R4 vs R5 a single-variable ablation (count-floor vs. ratio),
                    # not a confound of also changing which domains are eligible at all.
                    triggered = (direction is not None and support_ratio is not None
                                 and support_ratio >= R5_RATIO_THRESHOLD)
                    reason_trigger = (f"support_ratio={support_ratio:.2f} >= {R5_RATIO_THRESHOLD} "
                                       f"({len(unique_families)}/{len(assessable_families)} assessable families "
                                       f"matched)") if triggered else None
                    duplicate_inflation = False
                    cross_domain = False
                    struct_families_used = unique_families
                else:  # pragma: no cover
                    raise ValueError(strategy)

                insufficient_independent = triggered and strategy in ("R1", "R5") and len(struct_families_used) < MIN_EVIDENCE
                ineligible_evidence = False  # no allowed_output_level filter exists anywhere in the frozen
                                              # pipeline (verified: check_visual_preconditions/build_eligible_
                                              # matches/build_literature_linked_associations never gate on it) --
                                              # tracked as a column for completeness, always False on this cohort.
                inappropriate_unverified_use = triggered and unverified_used
                # A page-dependent rule (PSY_AR_SIZE_*/PLACE_*) can only ever reach status
                # "satisfied" (i.e. become a matched rule at all, for ANY strategy) when
                # check_deterministic_preconditions already confirmed page_relative_features_
                # assessable -- when it is False the rule's own status is "not_assessable" and
                # it never enters `assocs` in the first place. So this is always False on this
                # cohort by construction of the frozen pipeline; tracked explicitly (not
                # hand-waved) as a genuine, reportable true-negative safety property.
                unavailable_page_dependent = False
                ignored_contradiction = triggered and bool(contradictions)

                structural_violation = any([
                    duplicate_inflation, cross_domain, insufficient_independent, ineligible_evidence,
                    inappropriate_unverified_use, ignored_contradiction,
                ])

                row.update({
                    "triggered": triggered, "abstained": not triggered,
                    "candidate_interpretation": (
                        f"{domain}: {len(unique_families)} family(ies) via {strategy}" if triggered else None),
                    "duplicate_inflation": duplicate_inflation, "cross_domain_convergence": cross_domain,
                    "insufficient_independent_evidence": insufficient_independent,
                    "ineligible_evidence_used": ineligible_evidence,
                    "inappropriate_unverified_use": inappropriate_unverified_use,
                    "unavailable_page_dependent_evidence": unavailable_page_dependent,
                    "ignored_contradiction": ignored_contradiction,
                    "structural_violation": structural_violation,
                    "reason_for_trigger": reason_trigger,
                    "reason_for_abstention": None if triggered else (
                        f"{strategy} threshold not reached ({len(unique_families)} unique families, "
                        f"{len(rule_ids)} raw rules, support_ratio="
                        f"{'n/a' if support_ratio is None else f'{support_ratio:.2f}'})"),
                })
                rows.append(row)
    return rows


def main() -> None:
    cases = load_all_cases()
    n_total = len(cases)
    n_usable = sum(1 for v in cases.values() if v is not None)
    print(f"Loaded {n_total} development-set cases, {n_usable} usable (have cached synthesis).")

    atomic_rows = build_atomic_evidence_rows(cases)
    raw_dir = EXP_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_path = raw_dir / "input_atomic_evidence.csv"
    pd.DataFrame(atomic_rows).to_csv(atomic_path, index=False)
    print(f"Wrote {atomic_path} ({len(atomic_rows)} rows)")

    case_domain_rows = build_case_domain_rows(cases)
    per_domain_path = raw_dir / "aggregation_per_domain.csv"
    pd.DataFrame(case_domain_rows).to_csv(per_domain_path, index=False)
    print(f"Wrote {per_domain_path} ({len(case_domain_rows)} rows)")

    # --- per-case rollup: for each (case, strategy), did ANY domain trigger? ---
    df = pd.DataFrame(case_domain_rows)
    per_case_rows = []
    for cid in sorted({r["case_id"] for r in case_domain_rows}) if case_domain_rows else []:
        case_df = df[df["case_id"] == cid]
        for strategy in STRATEGIES:
            sdf = case_df[case_df["strategy"] == strategy]
            triggered_domains = sdf[sdf["triggered"]]
            per_case_rows.append({
                "case_id": cid, "strategy": strategy,
                "any_domain_triggered": bool(len(triggered_domains) > 0),
                "n_domains_triggered": int(len(triggered_domains)),
                "triggered_domains": ";".join(sorted(triggered_domains["concern_domain"])),
                "any_structural_violation": bool(sdf["structural_violation"].any()),
                "any_duplicate_inflation": bool(sdf["duplicate_inflation"].any()),
                "any_cross_domain_convergence": bool(sdf["cross_domain_convergence"].any()),
                "any_ignored_contradiction": bool((sdf["triggered"] & sdf["has_contradiction"]).any()),
                "total_domains_considered": int(len(sdf)),
            })
    # cases with zero matched rules entirely never appear in case_domain_rows -- add them explicitly
    zero_rule_cases = [cid for cid, v in cases.items() if v is not None and not v["assocs"]]
    for cid in zero_rule_cases:
        for strategy in STRATEGIES:
            per_case_rows.append({
                "case_id": cid, "strategy": strategy, "any_domain_triggered": False, "n_domains_triggered": 0,
                "triggered_domains": "", "any_structural_violation": False, "any_duplicate_inflation": False,
                "any_cross_domain_convergence": False, "any_ignored_contradiction": False,
                "total_domains_considered": 0,
            })

    per_case_path = raw_dir / "aggregation_per_case.csv"
    per_case_df = pd.DataFrame(per_case_rows).sort_values(["case_id", "strategy"])
    per_case_df.to_csv(per_case_path, index=False)
    print(f"Wrote {per_case_path} ({len(per_case_df)} rows, {per_case_df['case_id'].nunique()} unique cases)")

    print("\nDone with raw extraction. Run compute_stats.py and make_figures.py next.")


if __name__ == "__main__":
    main()
