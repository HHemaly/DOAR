# Target Application Architecture

**Status: design document. Nothing in this document is itself an
implementation claim** — it specifies how the Parent and Technical views
should be assembled from what `CURRENT_CAPABILITY_AUDIT.md` found is
actually real, plus where new modules are genuinely required. The
dual-view prototype (separate deliverable) implements a subset of this,
explicitly marked in §12.

**Governing rule for every module below**: every conclusion surfaced to a
user must be traceable to exactly one of — (a) measured image evidence,
(b) actual model output, (c) an executed rule, (d) user-provided profile
information, or (e) an approved knowledge-base record. Anything else is
not shown. "Not detected," "not implemented," "not evaluated," and "low
confidence" are four different states and must never be collapsed into
one generic "unavailable."

```
┌─────────────────────────────────────────────────────────────────────┐
│                         1. Image Preprocessing                       │
│   analysis.py::_quality, _segment  (REAL, existing)                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
        ┌────────────────────────┴────────────────────────┐
        ▼                                                  ▼
┌───────────────────┐                          ┌─────────────────────────┐
│ 2. Emotion         │                          │ 3. Object & Geometry    │
│ Classification      │                          │ Detection                │
│ emotion.py (REAL)   │                          │ detectors/ (MISSING —   │
│                      │                          │ schema only, §Phase-3   │
└──────────┬───────────┘                          │ roadmap, not this task) │
           │                                       └────────────┬────────────┘
           ▼                                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  4. Objective Feature Extraction                     │
│   features.py::objective_feature_row (REAL, 59 features, 2 marked   │
│   missing) — needs wiring into the default (non-fusion) run path     │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  5. Evidence Representation                          │
│   schemas.py::Evidence (REAL, existing) — every downstream claim     │
│   must cite an evidence_id from here                                 │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              6. Validated Deterministic Rules                        │
│   rules.py::evaluate_rules + concerns.py (REAL, 6/19 executable)     │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
        ┌────────────────────────┴────────────────────────┐
        ▼                                                  ▼
┌───────────────────┐                          ┌─────────────────────────┐
│ 7. Child-Profile    │                          │ 13. Case Storage &      │
│ Context (NEW)        │                          │ Versioning               │
│ profile.py (new)     │                          │ case_output.py (REAL)   │
└──────────┬───────────┘                          └────────────┬────────────┘
           ▼                                                    │
┌─────────────────────────────────────────────────────────────────────┐
│              8. Non-Diagnostic Report Generation                     │
│   reports.py + judges.py (REAL, extended for parent rule visibility) │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
        ┌────────────────────────┴────────────────────────┐
        ▼                                                  ▼
┌───────────────────┐                          ┌─────────────────────────┐
│ 9. Parent View      │                          │ 10. Technical View       │
│ (NEW UI, real data)  │                          │ (extends streamlit_app)  │
└──────────┬───────────┘                          └─────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  11. Grounded Follow-up Chat  →  12. Claim Verification & Judging    │
│  qa.py (REAL, deterministic fallback) + new ChatProvider abstraction  │
│  See LLM_GROUNDING_AND_SAFETY_DESIGN.md for full detail               │
└─────────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────────┐
│                  14. Knowledge-Base Governance (NEW)                  │
│   candidate → trusted-source retrieval → review → approval →         │
│   versioned storage → available to chat. Never automatic.            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Image preprocessing — REAL, existing

`analysis.py::_quality` + `_segment`. No change needed structurally;
recommend adding wall-clock timing instrumentation here (§10, missing
field identified in `END_TO_END_INFERENCE_TRACE.md` §7).

## 2. Emotion classification — REAL, existing

`emotion.py::predict` → `deep/inference.py::predict_image` (deep
checkpoints) or classical/fusion branches. Already returns calibrated
confidence, full probability distribution, calibration status, and an
honest `unavailable`/`suppressed`/`failed` state machine. No change
needed for the prototype beyond surfacing all of this in both views.

## 3. Object and geometry detection — MISSING, out of this task's scope

`detectors/` is schema-only (`CURRENT_CAPABILITY_AUDIT.md` §7). Building a
real detector is a substantial, separate research effort (model
selection, annotation, validation against `PHASE3_DETECTOR_EVALUATION_PLAN.md`'s
acceptance bar) and is explicitly **not** undertaken in this task. The
architecture's job here is only to make sure every downstream module
degrades honestly when this module returns nothing — which is exactly
what `detections.json`'s `{"status": "unavailable", "detections": []}`
already does, and what both views must render as *"Detector not
implemented"*, never as "no objects in this drawing."

## 4. Objective feature extraction — REAL, needs wiring

`features.py::objective_feature_row` is complete and correct (59
features, 2 honestly missing) but is only invoked on the fusion-checkpoint
branch today. **Recommended change** (implemented in the prototype):
call it unconditionally after segmentation, independent of which
checkpoint type is used, and persist it in a dedicated
`objective_features.json` — features are useful evidence on their own,
not only as fusion-model input.

## 5. Evidence representation — REAL, existing

`schemas.py::Evidence(evidence_id, kind, value, method, confidence,
limitations)`. This is already the traceability backbone — the Technical
view's "Evidence IDs" column and the chat's claim-to-evidence mapping
(§11/§12) both key off this dataclass unchanged.

## 6. Validated deterministic rules — REAL, partial coverage

`rules.py::evaluate_rules` + `concerns.py::derive_concerns` (disabled by
design). No engine change needed; both views must render all four rule
states honestly (`RULE_AND_FEATURE_COVERAGE.md` §4) and never surface a
concern profile while `CONCERNS_ENABLED = False`.

## 7. Child-profile context — NEW, small, isolated module

**Does not exist today.** Needed fields (per your spec): age range,
gender (optional), drawing instruction/prompt, date, parent's
concern/question. Design as a plain, validated dataclass
(`profile.py::ChildProfile`) persisted alongside the case
(`<case>/profile.json`), **never fed into the emotion model or rule
evaluation** (those must stay image-only, to avoid the model learning to
key off demographic text instead of the drawing). Its only consumers are:
(a) the report/UI (context display only), and (b) the chat's
`EvidenceRetriever`, where it is clearly tagged as `source: user_provided`,
distinct from measured evidence. If a rule is ever added whose registry
tier is `tier_3_prompt_or_age_dependent` (none exist today — see
`RULE_AND_FEATURE_COVERAGE.md` §1), the profile's `drawing_instruction`
field is what would let `rules.py` resolve it out of
`not_assessable_context_unknown` — but until such a rule is added and
reviewed, the profile must never silently unlock a tier-2 rule.

## 8. Non-diagnostic report generation — REAL, needs one fix

`reports.py` + `localization.py` + `judges.py::run_judges`'s safety scan.
**One concrete, code-confirmed fix required**: `reports.py:90`'s
`rule_section = "" if parent else ...` must be replaced with a
parent-appropriate rule rendering (plain-language wording, still showing
`weak_support`/`not_matched`/`missing_detector` status, not the raw
professional table) — this is exactly what the user specified for the
Parent view and exactly what is currently missing.

## 9. Parent/User view — NEW UI, wraps 100% real data

No new inference logic — a UI layer over `analysis.json` +
`objective_features.json` (§4) + `profile.json` (§7) + `qa.py`/chat (§11).
Must render, per your spec: uploaded drawing, child context, plain-language
observations, detected objects *if supported* (currently: an honest "not
implemented" message, §3), features in plain language, rules
evaluated/triggered/unevaluable-with-reason, calibrated model output,
cautious overall interpretation, uncertainty/limitations, safe guidance,
non-diagnostic disclaimer, and the follow-up chat.

## 10. Technical/Research view — extends the existing Streamlit app's data model

Everything in `CURRENT_CAPABILITY_AUDIT.md` §13 already renders raw JSON
for most of this. The architecture change is presentational (structured
tables instead of `st.json`) plus **two new fields that don't exist yet**:
processing time (needs new timing instrumentation around
`analyze_image()`) and case/report schema version (already present,
`schema_version: "3.0.0"` — just needs surfacing consistently).

## 11. Grounded follow-up chat — NEW abstraction, real deterministic fallback exists

Full design in `LLM_GROUNDING_AND_SAFETY_DESIGN.md`. Summary: a
`ChatProvider` interface with a real, working, no-API-key
`DeterministicChatProvider` built directly on the existing
`qa.py::answer` (already grounded, bilingual, evidence-citing) as the
default/fallback, plus an optional pluggable LLM-backed provider gated
behind the full claim-verification pipeline in §12 — never enabled by
default, never required to run the base application.

## 12. Claim verification and safety judging — extends existing `judges.py`

`judges.py::run_judges`'s safety judge (diagnostic-language regex scan,
English + Arabic) is the deterministic seed of this module. It is
extended, not replaced, by an `EvidenceRetriever`/`ClaimExtractor`/
`ClaimVerifier`/`SafetyChecker`/`ResponseJudge` pipeline specifically for
chat output — see `LLM_GROUNDING_AND_SAFETY_DESIGN.md` for the exact
process graph. **A second LLM judge alone is never sufficient** — every
generated claim must first pass the deterministic grounding checks that
already exist for the rest of the pipeline.

## 13. Case storage and versioning — REAL, existing

`case_output.py::write_versioned` (archive-on-change, `history.jsonl`
audit trail). New artifacts introduced by this architecture
(`objective_features.json`, `profile.json`, chat transcripts) go through
the same `write_versioned` path, not a new persistence mechanism.

## 14. Knowledge-base governance — NEW, process + storage design

**Does not exist today** (no knowledge base exists at all — the rule
registry is the closest analog and is itself entirely static/manual,
`CURRENT_CAPABILITY_AUDIT.md` §16). Required workflow, matching your
spec exactly:

```
Candidate information
  → trusted-source retrieval (peer-reviewed papers, professional
    guidance, authoritative primary sources only)
  → source and scope verification (population, age range, study design)
  → human or qualified-expert review
  → approval
  → versioned storage (append-only, like rules_registry.json's own
    pattern — never silently edited in place)
  → availability to chat (via EvidenceRetriever, read-only)
```

Each approved record: `claim, source, publication_date, population,
age_range, supported_scope, evidence_strength, limitations, reviewer,
approval_date, version`. This reuses exactly the pattern already proven
safe for the rule registry (`psychology_ingest.py`'s inert-draft +ancient
manual-promotion discipline, `CURRENT_CAPABILITY_AUDIT.md` §16) — no
automatic web research is ever written directly to approved storage.

---

## Architecture-level invariants (must hold for every module above)

1. No module may write to the original dataset or any existing
   `outputs/phase3a`–`outputs/phase7b` artifact.
2. No module may access `split == "test"` without both
   `test_guard.py::require_test_access` flags.
3. `CONCERNS_ENABLED` stays `False` until a real taxonomy exists — no
   module in this architecture flips it.
4. `rules_registry.json` is never written by any automated path (chat,
   detector, or otherwise) — only by a human editing the file directly,
   per the existing, verified guarantee (`CURRENT_CAPABILITY_AUDIT.md`
   §16).
5. Every user-facing claim in either view carries an explicit status from
   `{measured, model_output, executed_rule, user_provided,
   approved_knowledge_base, not_available}` — never an unlabeled
   assertion.

## What the prototype (separate deliverable) actually implements

See the prototype's own section in `SESSION_HANDOFF.md` for the final,
as-built list. Planned scope, gated on this document and
`CURRENT_CAPABILITY_AUDIT.md` being accurate: modules 1, 2, 4 (wiring
fix), 5, 6, 7 (new, minimal), 8 (rule-section fix), 9, 10 — using
`qa.py` directly for module 11's deterministic fallback, with the
`ChatProvider` abstraction present in code but no real LLM wired in.
Modules 3 and 14 are **not** implemented (3 has no underlying capability
to wrap; 14 is a governance process, not a UI feature, and out of scope
for a local prototype). Module 12 is implemented only as much as
`LLM_GROUNDING_AND_SAFETY_DESIGN.md`'s deterministic layer requires for
the no-LLM fallback path.
