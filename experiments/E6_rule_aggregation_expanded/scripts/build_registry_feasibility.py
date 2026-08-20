"""E6X registry feasibility audit -- STATIC, no cases, no API calls.

Reads RULE_EVIDENCE_MATRIX.csv and RULE_RELATIONSHIP_GRAPH.json only.
For every concern domain, determines whether R4's >=2 independent-
evidence-family-within-one-domain threshold is theoretically reachable
given the rules CURRENTLY marked allowed_output_level ==
"individual_heuristic_only" (enabled). Writes
tables/E6X_T8_registry_feasibility.csv and does not modify the registry.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "RULE_EVIDENCE_MATRIX.csv"
GRAPH_PATH = ROOT / "RULE_RELATIONSHIP_GRAPH.json"
OUT_CSV = EXP_DIR / "tables" / "E6X_T8_registry_feasibility.csv"

ENABLED_LEVEL = "individual_heuristic_only"

# Mirrors drawing_synthesis.py's own domain partition -- R4's real
# same-domain convergence path is only ever evaluated on positive/concern
# domains, never on neutral ones.
POSITIVE_DOMAINS = {"positive_affect_or_social_engagement"}
DISTRESS_DOMAINS = {
    "depressive_or_low_mood_related",
    "anxiety_or_stress_related",
    "aggression_or_threat_related",
    "social_withdrawal_related",
    "developmental_or_attention_related",
}
NEUTRAL_DOMAINS = {
    "neutral_descriptive_only",
    "neutral_descriptive_only_no_construct_proposed",
}
R4_IN_SCOPE_DOMAINS = POSITIVE_DOMAINS | DISTRESS_DOMAINS


def load_rows():
    with open(MATRIX_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_requires():
    with open(GRAPH_PATH, encoding="utf-8") as f:
        graph = json.load(f)
    return {e["rule_id"]: e for e in graph["requires_edges"]}


def main():
    rows = load_rows()
    requires = load_requires()
    all_domains = sorted({r["concern_domain"] for r in rows})

    out_rows = []
    for domain in all_domains:
        domain_rows = [r for r in rows if r["concern_domain"] == domain]
        enabled_rows = [r for r in domain_rows if r["allowed_output_level"] == ENABLED_LEVEL]
        enabled_families = sorted({r["evidence_family"] for r in enabled_rows})
        enabled_rule_ids = sorted(r["rule_id"] for r in enabled_rows)

        prereqs = []
        for r in enabled_rows:
            req = requires.get(r["rule_id"])
            if req is not None:
                sat = "SATISFIED" if req["currently_satisfied_in_doar"] else "NOT_SATISFIED"
                prereqs.append(f"{r['rule_id']}:{req['precondition']}[{sat}]")

        in_scope = domain in R4_IN_SCOPE_DOMAINS
        max_independent_families = len(enabled_families)
        reachable = in_scope and max_independent_families >= 2

        reachable_pairs = []
        if reachable:
            # any one enabled rule from each of >=2 distinct families
            # co-occurring in a single case would satisfy R4's threshold
            by_family = {}
            for r in enabled_rows:
                by_family.setdefault(r["evidence_family"], []).append(r["rule_id"])
            fam_list = sorted(by_family.items())
            for i in range(len(fam_list)):
                for j in range(i + 1, len(fam_list)):
                    fam_a, ids_a = fam_list[i]
                    fam_b, ids_b = fam_list[j]
                    reachable_pairs.append(f"({fam_a}:{'/'.join(ids_a)}) + ({fam_b}:{'/'.join(ids_b)})")

        out_rows.append(
            {
                "concern_domain": domain,
                "r4_in_scope_domain": in_scope,
                "enabled_rule_count": len(enabled_rows),
                "enabled_rule_ids": ";".join(enabled_rule_ids),
                "enabled_evidence_family_count": len(enabled_families),
                "enabled_evidence_family_names": ";".join(enabled_families),
                "visual_deterministic_prerequisites": ";".join(prereqs) if prereqs else "",
                "max_possible_independent_family_count": max_independent_families,
                "r4_threshold_reachable": "YES" if reachable else "NO",
                "reachable_rule_pairs_if_any": ";".join(reachable_pairs) if reachable_pairs else "",
                "total_rules_in_domain": len(domain_rows),
                "disabled_rules_in_domain": len(domain_rows) - len(enabled_rows),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    reachable_domains = [r["concern_domain"] for r in out_rows if r["r4_threshold_reachable"] == "YES"]
    print(f"Wrote {OUT_CSV} ({len(out_rows)} domains)")
    print(f"R4-threshold-reachable domains: {reachable_domains or 'NONE'}")
    return out_rows, reachable_domains


if __name__ == "__main__":
    main()
