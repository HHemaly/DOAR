# Literature Search Log

Reproducibility record for the DOAR rule/feature literature review. Append-only
across rounds, like `DECISION_LOG.md`. This log exists so a reader can see
*exactly* what was searched, when, with what result, and why each source was
kept or rejected — not just the conclusions.

## Method (stated honestly, not idealized)

**Tool used**: `WebSearch` (a general web search backend, not a direct API to
PubMed/PsycINFO/Scopus/Web of Science/Cochrane). Results are indexed web pages
— journal publisher pages, PubMed/PMC listings, ResearchGate, arXiv, ERIC, and
similar — not a curated bibliographic database. This is a real limitation: no
formal database de-duplication, no controlled-vocabulary (MeSH-style) search,
and coverage depends on what the search backend has indexed. Where a result
pointed to a PMC (open-access) or arXiv page, `WebFetch` was used to retrieve
and read the actual text directly, and findings from that pass are marked
**VERIFIED (full text)**. Everything else is marked **SEARCH SUMMARY ONLY**
and must be treated as lower-confidence, secondary evidence — a claim about
what a study found, not independently confirmed against the study itself.

**Inclusion criteria**: peer-reviewed journal articles, systematic
reviews/meta-analyses, or official clinical/professional guidance, concerning
children's drawing, sketch, or figure-drawing research, in any language the
search surfaced in English-language results.

**Exclusion criteria**: blogs, commercial "what your child's drawing means"
pages, unattributed listicles, non-peer-reviewed opinion pieces. (No such
sources were screened in at any point in either round — none appeared as a
primary hit worth recording; general symbolism-list content, when it surfaced,
was skipped without a row in this log.)

**Duplicate handling**: a source appearing under more than one query is
recorded once, at its first appearance, with a note listing every query that
also surfaced it.

**Screening**: for each query, the top results actually returned by the tool
are listed; each is marked `INCLUDED` (recorded as a candidate — see
`LITERATURE_CANDIDATE_REGISTER.csv`) or `EXCLUDED` with a one-line reason.

---

## Round 1 (2026-08-02, first pass — see commit `20966e8`)

Queries run (informal, not pre-logged at the time — reconstructed from the
conversation transcript for completeness):

| # | Query | Included | Excluded/not pursued |
|---|---|---|---|
| R1.1 | "children's spontaneous drawing colour use emotional state empirical study peer-reviewed" | `LIT_COLOUR_EMOTION_001` (+ its contradicting study) | — |
| R1.2 | "line pressure pencil pressure children's drawing emotional indicator empirical study reliability" | `LIT_PENCIL_PRESSURE_002` | — |
| R1.3 | "human figure drawing emotional indicators systematic review validity reliability meta-analysis" | `LIT_HFD_KOPPITZ_INDICATORS_003` | Goodenough-Harris (different construct — developmental/cognitive maturity scoring, not emotional content; noted as context, not a candidate) |
| R1.4 | "spontaneous drawing versus instructed drawing task children psychology research differences" | `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006` (via its PMC page, found through this query's results) | — |
| R1.5 | "drawing size self-esteem children meta-analysis empirical validity" | `LIT_SELFESTEEM_METHOD_RELIABILITY_005` | — |
| R1.6 | "SceneDAPR dataset..." / "ChildlikeSHAPES dataset..." / "COCO dataset 80 object categories..." (dataset-verification queries, not rule-literature queries — logged in `PHASE3_DETECTOR_EVALUATION_PLAN.md` instead) | n/a | n/a |
| R1.7 | Direct fetch: PMC9204051 (`LIT_HFD_DEPRESSION_ADULT_004`) | **VERIFIED (full text)** | — |
| R1.8 | Direct fetch: PMC9697182 (`LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`) | **VERIFIED (full text)** | — |

Round 1 self-critique (per user review, 2026-08-02): 6 candidates from a
handful of queries is not exhaustive and must not be reported as if no other
useful candidates exist. `LIT_DEVELOPMENTAL_STAGE_OBJECTIVE_006`'s DOAR-
transfer applicability was overstated (corrected in
`LITERATURE_CANDIDATE_REGISTER.csv` and `RULE_SOURCE_REGISTER.csv`, see
`DECISION_LOG.md`). Round 2 below is broader and more systematically logged.

---

## Round 2 (2026-08-02, broader pass)

Pre-specified topic areas (from the user's request, listed before searching):
(a) spontaneous/free drawings; (b) emotional-expression depiction vs.
diagnosis; (c) objective geometry/colour/complexity/fragmentation/
representational content; (d) human figures and facial expressions
(supplementing round 1); (e) houses, trees, family drawings and symbols
(supplementing round 1's HTP/abuse-review citations with Kinetic Family
Drawing specifically, not yet searched); (f) developmental, cultural, motor,
and material confounders; (g) systematic reviews and negative-validity
evidence (supplementing round 1).

Queries pre-specified before running (this table's "Included/Excluded" columns
were filled in after execution, not before):

| # | Area | Query |
|---|---|---|
| R2.1 | (a) | "spontaneous free drawing children psychological assessment validity study" |
| R2.2 | (a) | "unprompted drawing children emotional state naturalistic observation research" |
| R2.3 | (b) | "children's intentional depiction of emotion in drawing accuracy encoding study" |
| R2.4 | (c) | "computational quantitative analysis children's drawing complexity fragmentation features" |
| R2.5 | (c) | "drawing complexity index measure development children age quantitative" |
| R2.6 | (e) | "Kinetic Family Drawing KFD test validity reliability systematic review" |
| R2.7 | (f) | "cross-cultural differences children's drawings developmental psychology study" |
| R2.8 | (f) | "fine motor skill development effect children's drawing characteristics confound" |
| R2.9 | (f) | "drawing material medium paper size effect children's drawing study" |
| R2.10 | (g) | "children's drawings psychological assessment systematic review evidence-based practice" |

**Results (filled in after execution, 2026-08-02):**

| # | Included (→ candidate_id) | Excluded / not pursued (reason) |
|---|---|---|
| R2.1 | Reinforces `REF_GENERAL_LIMITS_1998` (already in registry — duplicate, not re-logged); PMC9894026 fetched in full → `LIT_DRAWN_STORIES_MALTREATMENT_012` | — |
| R2.2 | Boyatzis 2000 peer-collaboration study → folded into `LIT_CULTURAL_CONFOUND_008`'s broader confounder note (social/peer context) | Frontiers editorial (2023) — a research-topic editorial, not a primary study; used only to find the maltreatment systematic review below, not logged separately |
| R2.3 | PubMed 17165414 ("Children's developing ability to depict emotions in their drawings") + related facial-expression-encoding studies → `LIT_EMOTION_FACE_ENCODING_007` | "Drawing and Memory" (Iordanou et al.) — about verbal-report vs. drawing content overlap, not emotion depiction; tangential, not logged |
| R2.4 | Fragmentation/local-processing-bias finding → `LIT_FRAGMENTATION_LOCAL_PROCESSING_013` | RGS Geography "multimodal AI" article — geography-education venue, not psychology; excluded as off-topic |
| R2.5 | Goodenough/Koppitz/Machover historical framing — background only, not a new candidate (Koppitz already `LIT_HFD_KOPPITZ_INDICATORS_003`) | "Draw-A-Man drawing age" — IQ/developmental-maturity construct, not emotion; out of DOAR's scope (DOAR does not attempt cognitive-ability assessment) |
| R2.6 | KFD validity/reliability concerns → `LIT_KFD_VALIDITY_011` | — |
| R2.7 | Cross-cultural size/action/shading/gender-pattern differences → `LIT_CULTURAL_CONFOUND_008` | — |
| R2.8 | Fine-motor-skill as independent predictor of drawing scores → `LIT_MOTOR_SKILL_CONFOUND_009` | — |
| R2.9 | Paper size / medium / material effects → `LIT_MATERIAL_MEDIUM_CONFOUND_010` | Paper-texture/art-supply blog content — commercial, excluded per method above |
| R2.10 | Maltreatment-assessment systematic review (DARE/Veltman) → folded into `LIT_DRAWN_STORIES_MALTREATMENT_012`; "HFD does not measure intellectual ability" (ScienceDirect) → noted as general negative-validity context in `IMPROVEMENT_REGISTER.md`, not logged as its own candidate (different construct — IQ, not emotion) | — |

**Full-text verification attempted this round**: PubMed 17165414 — **blocked** (page returned only a cookie-consent wall to `WebFetch`, no abstract text retrieved; recorded as SEARCH SUMMARY ONLY, not verified, despite the search engine's summary describing it in some detail). PMC9894026 — **VERIFIED (full text)**: N=1,757, ages 6–13, confirmed INSTRUCTED protocol ("draw an invented story"), and critically — the "good validity for anxiety/depression" claim traces back to a *different*, earlier study (Trombini et al. 2004, N=211) that this paper only cites; the fetched paper itself reports no diagnostic-accuracy statistics for that claim.

**Round 2 self-critique**: still not exhaustive — no formal database (PsycINFO/Scopus) access, no MeSH-term search, no forward/backward citation chasing beyond what these 10 queries' results happened to surface. Should be treated as a broadened preliminary pass, not a completed systematic review. See the final report for what a genuinely complete review would still require.
