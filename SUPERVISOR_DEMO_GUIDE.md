# DOAR Supervisor Demo Guide

Branch: `feature/supervisor-demo-v2`. This is a presentation layer over the
existing, frozen DOAR pipeline -- no scientific logic, thresholds, rules,
or datasets were changed to build it.

## 1. Launch command

```
cd DOAR-psychologist-app
.\.venv\Scripts\Activate.ps1
python -m streamlit run doar_prototype_app.py
```

The app opens directly on the new **Supervisor Demo** Home page by default
(no case pre-selected). The sidebar "View" switch can flip to
**Full Research App (legacy)** at any time -- the original Parent /
Psychologist / Technical tabs are untouched and still fully work there.

## 2. Health check

```
.\.venv\Scripts\python.exe scripts\check_supervisor_demo.py
```

Expected output (also see the literal run captured in this session's
report):

```
DOAR Supervisor Demo
--------------------
App             READY
Case 1          READY
Case 2          READY
Case 3          READY
Gemini          READY / NOT CONFIGURED
Prepared Demo   READY
```

`Prepared Demo` is `READY` even when `Gemini` is `NOT CONFIGURED` --
opening a prepared case never requires GEMINI_API_KEY.

## 3. The three prepared cases

| # | Case ID | Role | Title shown in the UI |
|---|---------|------|------------------------|
| 1 | `a103_1787479142` | COMPLETE (main example) | Full Analysis Walkthrough |
| 2 | `a111_1787479358` | CONTRAST | A Different Drawing Profile |
| 3 | `h38_1786305027` | GOVERNANCE | Uncertainty and Governance |

All three live under `outputs/prototype_cases/` (gitignored by repository
convention -- see "Known limitations" below) and were selected because
each already has a real, cached `analysis.json` + image + `detections.json`
(deep visual scan already run) with a genuine applicable governed rule
match or evaluated-but-not-matched rule, so nothing needs to be
recomputed live.

### Case 1 -- Full Analysis Walkthrough (`a103_1787479142`)
The main example. Show, in order: the original drawing, the expressive
profile (Sad-dominant), "What DOAR observed" (24 detected visual elements,
several VERIFIED, plus composition/colour observations), the 7 objective
measurements, the applicable governed rules panel (3 evaluated rules, each
with "View evidence trace"), the governed synthesis card, and Ask DOAR.

### Case 2 -- A Different Drawing Profile (`a111_1787479358`)
A different image entirely: Angry-dominant expressive profile (vs. Case
1's Sad), a different set of 25 detected visual elements, and a different
composition/colour observation set. Use this to show that the pipeline's
output genuinely tracks the drawing, not a canned response.

### Case 3 -- Uncertainty and Governance (`h38_1786305027`)
The governance case. Its expressive-content model is unavailable for this
image (a real "insufficient evidence" state, not simulated), it has the
richest detection set (41 findings, several UNCERTAIN/EXPERIMENTAL rather
than VERIFIED), and one WEAK-support concern domain with a real
contradicting reference. Use Section 5 (Agreement & Uncertainty) and
Section 6's "What remains uncertain" / "What cannot be concluded" to make
the governance point concrete.

## 4. Best Ask DOAR question per case

- Case 1: *"Why did DOAR reach this conclusion, and what evidence was used?"*
- Case 2: *"What are the most important observations in this drawing?"*
- Case 3: *"Which observations were uncertain or rejected, and what can DOAR not conclude?"*

Suggested-question chips for all eight spec questions are shown above the
chat box on every case's Drawing Analysis page.

## 5. Suggested 5-7 minute sequence

- **0:00-0:40** -- Show Home. Say: *"DOAR does not simply ask generative AI
  to interpret a drawing. It combines visual models, objective
  measurements, verification, scientific evidence and governed
  reasoning."*
- **0:40-3:00** -- Open Case 1. Show the drawing, expressive profile,
  verified observations, objective measurements, applicable evidence,
  synthesis, and one evidence trace. Ask DOAR: *"Why did DOAR reach this
  conclusion, and what evidence was used?"*
- **3:00-4:10** -- Open Case 2. Show the different evidence profile. Ask:
  *"What are the most important observations in this drawing?"*
- **4:10-5:10** -- Open Case 3. Show the uncertain/rejected evidence. Say:
  *"Generative AI is not allowed to directly determine a psychological
  conclusion here -- it can only produce a candidate that DOAR's own
  governed evidence then verifies, downgrades, or rejects."* Ask: *"Which
  observations were uncertain or rejected, and what can DOAR not
  conclude?"*
- **5:10-6:00** -- Show Technical Trace. Explain the traceability table and
  that raw JSON is only ever inside "Advanced / Raw Details".
- **6:00-7:00** -- Show Research Progress. Explain E2 is ongoing, semantic
  provider comparisons are future work, and Locked Test is untouched.

## 6. What NOT to click

- Do not switch to **Full Research App (legacy)** mid-demo unless you mean
  to -- it is a different, much larger surface (upload, Run Full Analysis,
  deep visual analysis, expert review forms) and switching back does not
  restore Supervisor Demo scroll position.
- In legacy mode specifically, do not click **"Run deep visual analysis"**
  or **"Run Full Analysis"** live during a demo -- both can be slow
  (heavy vision models) and are not needed; the three prepared Supervisor
  Demo cases already have this data cached.
- Do not rely on the sidebar "reopen a previous case" list in legacy mode
  during the demo -- it lists every case ever produced in this checkout
  (development benchmarks, one-off dev-check runs, etc.), most of which
  are not curated for presentation.

## 7. Known limitations (say if asked)

- `outputs/` (including the three prepared case directories) is
  **gitignored** by long-standing repository convention -- this branch's
  code changes are pushed to `origin`, but the prepared case data itself
  only exists in this local checkout. A fresh clone needs these case
  directories copied in separately before the Supervisor Demo can open
  them (the health check will correctly report `MISSING` in that
  situation, never a crash).
- Case 3's expressive-content model is genuinely unavailable for that
  image (not a simulated gap) -- this is an authentic example of DOAR
  disclosing "insufficient evidence" rather than guessing.
- The governed rule set shown in "Applicable governed evidence/rules" is
  the original, frozen 41-rule corpus (10 enabled at baseline) --
  `MASTER_RULE_FEATURE_REGISTRY_V3.json`'s 225 source entries / 206
  canonical observables / 21 evidence families is a separate research
  organization, never presented here as 225 active rules.
- Ask DOAR was tested live with a real `GEMINI_API_KEY` in this
  environment and produced a real, evidence-grounded answer; it also
  degrades gracefully (no crash, friendly message) with no key configured
  -- both paths are covered by `tests/test_supervisor_demo_smoke.py`.
- GPT/OpenAI was **deferred** to preserve demo reliability within the
  timebox -- Gemini remains the only Ask DOAR provider in this branch.
