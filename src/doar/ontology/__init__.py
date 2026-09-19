"""DOAR V2 ontology foundation.

This package is additive. It does not replace, modify, or import-reshape
any existing module: `features.py`, `rules.py`, `evidence_schema.py`,
`rule_schema.py`, and `evidence_rule_engine.py` are all untouched and
remain the live, tested paths they were before this package existed.

What this package adds is a *general* layer that the task's V2 milestone
requires and that nothing in the existing codebase provides:

- a declarative feature ontology (`feature_schema.py`) that distinguishes
  a feature's type, its assessability prerequisites, and its confounds
  from any one extractor's implementation of it;
- a source-agnostic observation record (`observation.py`) that can hold
  more than one extractor's/human's opinion about the same feature on the
  same case;
- an append-only, multi-source feature store (`store.py`) that never lets
  a later correction erase an earlier observation;
- a context object (`context.py`) that represents child age/sex/task/
  narrative as genuinely optional, so a rule that needs one degrades to a
  named "missing prerequisite" outcome instead of guessing;
- a declarative rule schema (`rule.py`) with composable AND/OR premises
  over feature IDs, explicit framework tagging
  (EMPIRICAL_EXPRESSIVE vs. CLASSICAL_PROJECTIVE), and citation metadata;
- a three-valued (TRUE/FALSE/UNKNOWN) rule engine (`engine.py`) that
  produces a full machine-readable trace per evaluation.

`legacy_adapters.py` is the seam: it converts the existing, real,
already-tested evidence pipeline's outputs (`EvidenceItem`s from
`evidence_adapter.py`, `RuleV2` entries from `rule_schema.py`) into this
package's types, so the new engine can be demonstrated end-to-end on real
extractor output without re-implementing any extraction or re-deriving
the 19-rule registry's citations.
"""
