# LLM Grounding and Safety Design

**Status: design document, partially implemented in the prototype.** No
paid API key is required to run the base application — the prototype ships
a fully working, deterministic, evidence-grounded chat fallback
(`DeterministicChatProvider`, built directly on the existing, real
`src/doar/qa.py`). Any real LLM provider is designed behind an
interchangeable interface and is **not** enabled or called by default; no
network request to an LLM is made anywhere in this task's deliverables.

Per `CURRENT_CAPABILITY_AUDIT.md` §14: no LLM/chat SDK integration exists
anywhere in this repository today. This document designs one without
requiring one to actually run.

---

## 1. Provider choice — deferred, not decided here

OpenAI, Gemini, and Anthropic's own API are all viable *if* an operator
chooses to configure one later; none is adopted by this task. The
interface below (`ChatProvider`) is the only thing that matters
structurally — whichever provider is wired in later must implement it
without changing any calling code, evidence-grounding logic, or safety
checks.

```python
class ChatProvider(Protocol):
    def generate(self, prompt: str, *, system: str, history: list[Message]) -> str:
        """Return raw draft text. MUST NOT be shown to the user directly —
        every ChatProvider output passes through the pipeline in §4 first."""
```

`DeterministicChatProvider` (implemented, no API key, no network) is the
only concrete provider shipped in the prototype. It does not "generate"
text in the LLM sense — it dispatches the user's message to
`qa.py::answer()` and returns the grounded envelope's `answer` field
directly, skipping most of §4's pipeline (there is nothing to verify — the
answer was never generated, it was looked up). An `OpenAIChatProvider` /
`GeminiChatProvider` class stub is defined with a clear `NotImplementedError`
and a docstring describing exactly what real integration would require
(API key from environment, request/response schema, timeout/retry
policy) — deliberately not implemented, so the base application never
requires a paid key and no key material is ever handled by this codebase.

## 2. What the chat may and may not do

**Must:**
- Cite the evidence behind every substantive claim (`evidence_id`s from
  `schemas.py::Evidence`, rule IDs from `rule_evaluations`, or an explicit
  `source: user_provided` / `source: knowledge_base_record_id` tag).
- Say "unavailable" when information isn't in the case
  (`qa.py`'s existing grounded-refusal behavior, reused verbatim).
- Ask for clarification rather than guess, when a question is ambiguous.
- Avoid diagnosis (enforced by the same regex/pattern safety scan already
  in `judges.py::_diagnostic_hit`, reused for chat output too).
- Never invent an object, feature, or rule result not present in the
  case's saved evidence.
- Never present a `weak_support` rule's `parent_safe_wording` as an
  established fact — must always carry the rule's own uncertainty framing.
- Escalate: if a user message contains urgent safeguarding/health language
  (self-harm, abuse disclosure, immediate danger), the response must
  short-circuit the whole pipeline below and return a fixed,
  human-reviewed escalation message (contact a qualified professional /
  local emergency service), never an LLM-generated response to that
  specific turn.
- Never learn from or permanently store new information pulled from the
  web during a conversation — anything retrieved live is ephemeral to that
  turn's context window and is never written into the knowledge base (§6)
  without going through the full governance workflow, by a human, later.

**Must not:**
- Call any external API without an explicit, operator-supplied key (never
  required for the base app to run).
- Treat its own or another LLM's output as ground truth for anything
  clinical.

## 3. Component interfaces

```python
class EvidenceRetriever:
    """Given a case_id, return the exact set of citable evidence: Evidence
    objects (schemas.py), rule_evaluations, the child profile (tagged
    user_provided), and any approved knowledge-base records relevant to
    the conversation. Read-only; never modifies the case."""
    def retrieve(self, case_id: str, query: str) -> EvidenceBundle: ...

class ClaimExtractor:
    """Split a draft response into individual factual claims (sentence- or
    clause-level). Each claim becomes a unit the verifier can accept,
    reject, or flag independently — a response is not all-or-nothing."""
    def extract(self, draft: str) -> list[Claim]: ...

class ClaimVerifier:
    """For each claim, attempt to map it to one or more evidence_ids in
    the retrieved EvidenceBundle. A claim with zero mappable evidence is
    UNSUPPORTED. Purely deterministic string/number matching against the
    bundle — no LLM call in this step."""
    def verify(self, claim: Claim, evidence: EvidenceBundle) -> VerifiedClaim: ...

class SafetyChecker:
    """Reuses judges.py's diagnostic-language regex (English + Arabic),
    plus a safeguarding-keyword scan on the INCOMING user message (not
    just the outgoing draft) to trigger the escalation short-circuit."""
    def check(self, text: str) -> SafetyResult: ...

class ResponseJudge:
    """Optional, independent SECOND pass -- may itself be LLM-backed for a
    style/tone critique, but its approval is NEVER sufficient on its own.
    A response with unsupported claims or a failed SafetyChecker result is
    blocked/revised regardless of what ResponseJudge says."""
    def judge(self, response: str, verified_claims: list[VerifiedClaim]) -> JudgeResult: ...
```

## 4. Response process (exact pipeline, matches your spec)

```
User message
  → SafetyChecker on the INCOMING message (safeguarding/urgent-concern
    check) → if triggered: return fixed escalation message, STOP.
  → EvidenceRetriever.retrieve(case_id, message)
  → ChatProvider.generate(prompt, system=grounding_instructions, history)
      (DeterministicChatProvider: skips straight to a qa.py-style answer;
       an LLM provider would draft freely here, constrained by the system
       prompt listing exactly what evidence is available)
  → ClaimExtractor.extract(draft)
  → ClaimVerifier.verify(claim, evidence) for each claim
  → reject unsupported claims (removed or replaced with "I don't have
    evidence for that" per-claim, not a blanket refusal of the whole turn
    when only part of it is unsupported)
  → check rule consistency (a claim citing rule X's status must match
    rule_evaluations[X].status exactly — e.g. never claim a
    missing_detector rule "found" something)
  → check contradictions (no claim may contradict another claim already
    accepted in the same response, or the case's own emotion/rule output)
  → check uncertainty language (every claim derived from a `weak_support`
    rule or an emotion prediction below the calibration confidence floor
    must carry hedging language; SafetyChecker re-scans the revised draft
    for diagnostic phrasing one more time)
  → SafetyChecker on the OUTGOING revised response
  → optional ResponseJudge critique (LLM-backed, advisory only)
  → revise (loop back into claim rejection) or block (return grounded
    refusal) if still unsafe
  → display answer with evidence references attached to each surviving
    claim
```

**Deterministic-only path** (no API key, `DeterministicChatProvider`):
collapses to `SafetyChecker(incoming) → EvidenceRetriever → qa.py::answer()
→ SafetyChecker(outgoing, always passes since qa.py's own vocabulary is
pre-vetted) → display`. This is the prototype's actual, working, default
behavior — no step is skipped for safety, only for generation (there is
nothing to verify against evidence when the answer was looked up from the
evidence directly, by construction).

## 5. Why a second LLM judge is not enough

`ResponseJudge` is explicitly optional and advisory in §3/§4 — an LLM
"critiquing" another LLM's output is still two models with no access to
ground truth, and both can share the same blind spots (hallucinated
objects, overstated confidence, subtle diagnostic framing that a
pattern-based `SafetyChecker` might also miss, but which a second LLM
pass is not guaranteed to catch either, and could even rubber-stamp).
The `ClaimVerifier`'s deterministic evidence-mapping step is the actual
safety boundary — it cannot approve a claim that has no matching
`evidence_id`, regardless of how fluent or confident the claim sounds.

## 6. Knowledge-base governance (cross-reference)

Full workflow specified in `TARGET_APPLICATION_ARCHITECTURE.md` §14 —
candidate → trusted-source retrieval → verification → human/expert review
→ approval → versioned storage → available to chat. Repeating the
non-negotiable constraint here since it's chat-facing: **`EvidenceRetriever`
only ever reads already-approved records.** No component in this design
writes to the knowledge base. Any live web lookup a future LLM-backed
provider might perform is scoped to that single turn's context and
discarded — never persisted, never promoted to "approved" without the
full human-review workflow, exactly like `psychology_ingest.py`'s existing
inert-draft pattern for the rule registry (verified in
`CURRENT_CAPABILITY_AUDIT.md` §16 to have no auto-promotion path today —
the chat design must not introduce one).

## 7. What the prototype actually ships

- `src/doar/chat.py`: `ChatProvider` protocol, `DeterministicChatProvider`
  (real, working, wraps `qa.py`), `SafetyChecker` (reuses
  `judges.py::_diagnostic_hit` plus a small safeguarding-keyword list),
  `EvidenceRetriever` (reads a case's `analysis.json`/`judges.json`/
  `profile.json`), and stub classes for `ClaimExtractor`/`ClaimVerifier`/
  `ResponseJudge`/an LLM-backed `ChatProvider` — present as typed
  interfaces with `NotImplementedError` bodies and docstrings, not fake
  implementations, so the architecture is real code today even though only
  the deterministic path is wired end-to-end.
- The prototype UI's "follow-up chat" panel uses
  `DeterministicChatProvider` exclusively. No toggle to enable a real LLM
  exists in the prototype — that remains a future, separately-reviewed
  change per your instruction not to enable unsafe generation now.
