# Insurance Claim Resolution Control Tower — Build Plan & Team Task Breakdown

> Companion to `Insurance_Claim_Resolution_System.md` (concept) and `Insurance_Claim_Intelligence_Guidelines.pdf` (guidelines).
> Purpose of this file: split the build into parallel workstreams, define every task's **purpose, inputs, deliverables and done-criteria**, and fix the **repo structure and interface contracts** so the team can work independently and the system can be extended later.

---

## Table of Contents

1. [Application Abstract](#1-application-abstract)
2. [Scope Decisions (reconciling the two reference documents)](#2-scope-decisions)
3. [Framework & Stack Choice (prototype-first)](#3-framework--stack-choice)
4. [System Architecture](#4-system-architecture)
5. [Agent Roster & State Ownership](#5-agent-roster--state-ownership)
6. [Core Contracts: `ClaimState` and friends](#6-core-contracts)
7. [Project Structure](#7-project-structure)
8. [Build Phases & Milestones](#8-build-phases--milestones)
9. [Workstreams & Task Cards](#9-workstreams--task-cards)
10. [Team Assignment Models](#10-team-assignment-models)
11. [Interface Contracts Between Workstreams](#11-interface-contracts-between-workstreams)
12. [Synthetic Data & Scenario Catalogue](#12-synthetic-data--scenario-catalogue)
13. [Evaluation Plan](#13-evaluation-plan)
14. [Demo Script](#14-demo-script)
15. [Guardrails (non-negotiables)](#15-guardrails-non-negotiables)
16. [Risks & Mitigations](#16-risks--mitigations)
17. [Open Decisions](#17-open-decisions)
18. [Extension Roadmap](#18-extension-roadmap)
19. [Appendix: Code Skeletons](#19-appendix-code-skeletons)

---

## 1. Application Abstract

**Name:** Insurance Claim Resolution Control Tower

**One-line description:** An integration-first, multi-agent AI layer that takes an incomplete or contradictory insurance claim and moves it toward a *defensible, well-evidenced resolution* — by working out what is known, what is missing, what conflicts, and what exact action would resolve the uncertainty — while leaving every material decision to a human adjuster.

**The problem.** Claims get delayed, mis-assessed, disputed or escalated because the information needed for a coverage decision is scattered across the policy, endorsements, claim documents, photographs, estimates, regulations and internal systems. Adjusters spend their time hunting, comparing and re-requesting, not exercising judgement.

**What the application does (end to end):**

1. **Registers** a claim (FNOL intake), looks up the *policy version in force on the date of loss* from a Policy Administration System (PAS), and flags duplicate candidates.
2. **Reads** every submitted document and image, shows the detected document type, and turns them into **structured facts with provenance** (document, page, exact text, extraction confidence, verification status).
3. **Understands the contract** — coverage clauses, exclusions, definitions, conditions, endorsements and their precedence — as a versioned structure, not a pile of text chunks.
4. **Finds gaps** — separates *contradictory* evidence, *missing* evidence and evidence that exists but *does not establish* a fact; then computes the **minimum additional evidence** needed to resolve each open question (the heart of the product).
5. **Reasons counterfactually** ("What would resolve this claim?") — maps how the assessment would change under scenarios A (new evidence supports sudden failure), B (supports prolonged leakage) or C (inconclusive).
6. **Builds a coverage issue matrix** with policy-wording citations; abstains where ambiguous.
7. **Calculates a settlement recommendation** with a deterministic, versioned engine (the LLM never does arithmetic).
8. **Routes** the claim to a simple / medium / complex approval tier using insurer-configurable rules, with reason codes.
9. **Plans and (under control) executes next actions** — draft information requests, create inspection tasks, update claim status — and **re-runs only the affected reasoning steps** when new evidence arrives.
10. **Sends** the approved outcome to Claims and a payment/posting instruction to a Reconciliation (Recon) system (mock in the prototype) and reads back matched/unmatched status.
11. Records a **full audit trail**: original evidence, policy wording used, agent outputs, calculation inputs, human overrides, final decision.

**What it is not:** not a claims adjudicator, not a fraud-scoring product, not a replacement for PAS / Claims / Recon, not a chatbot over PDFs.

**Heart of the project (reviewers should see this first):** the **evidence-sufficiency and next-best-investigation engine**. Policy RAG, extraction and settlement exist to feed it.

**Running example (used for demos and tests):** A property water-damage claim. Customer reports a sudden pipe burst; a plumber notes corrosion at the joint; the surveyor finds moisture in several rooms. The policy covers *sudden and accidental* water damage but excludes *gradual leakage*; an endorsement adds a sublimit. The system must say: *"coverage cannot be concluded — cause and start date of loss are unresolved — request X from Y"*, not guess.

---

## 2. Scope Decisions

The two reference documents overlap but have different emphasis. Decisions below are binding for the prototype unless the team lead changes them.

| Topic | `.md` concept doc | PDF guidelines | **Decision for prototype** |
|---|---|---|---|
| Line of business | Property, water damage | Not specified | **Property — water damage** (one pack; others added later as packs) |
| Core differentiator | Evidence sufficiency + counterfactuals | Integration across PAS/Claims/Recon, traceable dossier | **Evidence sufficiency is the hero feature; PAS/Claims/Recon integration is the delivery frame** |
| PA/PAS | — | Policy **Administration** System (not pre-authorisation) | Mock PAS adapter, read-only |
| Recon | — | Assumed claims-payment reconciliation; scope unconfirmed | Mock Recon adapter; assumption stated on screen |
| Duplicates | Not covered | Claim/FNOL, invoice, document, payment | MVP: **duplicate FNOL + duplicate invoice + payment idempotency**. Document/image-hash duplicates = stretch |
| Approval tiers | Human gate only | Simple / medium / complex with reason codes | Implement configurable tiering (sample thresholds, labelled "SAMPLE") |
| Image analysis | Multimodal vision for photos | Bounded, observable findings only | **One image task**: water-damage photo quality check + visible-damage observations + next-photo suggestion |
| Settlement formula | `min(eligible loss, limit) − deductible − adjustments` | `(loss − non-covered − deductible)` then cap | **Formula order is config, not code.** The engine reads a versioned YAML step list (see T5.3). Both examples become test cases |
| Regulatory | India (IRDAI) context | IRDAI 2024 regs, fraud framework 2025 | Design for "all required documents requested together", status information, reasoned deductions. Re-verify figures before citing |
| Benefit claims | "Don't invent percentages" | Baseline + formulas | Ship the **metrics dashboard and formulas**, no invented improvement numbers |

**Terminology fixes to keep consistent in code and UI:** `PA = PAS = Policy Administration System`. "Registration" = FNOL intake + claim creation. "Duplicate candidate" ≠ fraud. "Extraction confidence" ≠ probability of coverage.

---

## 3. Framework & Stack Choice

You asked for a high-level agentic framework for fast prototyping. Recommendation:

### Primary: **LangGraph** (with LangChain model/structured-output helpers)

Why it fits *this* project better than "role-play" crews:

| Need in this project | Why LangGraph fits |
|---|---|
| Persistent, typed, versioned `ClaimState` | Graph state is a first-class typed object; checkpointers persist/replay it |
| Supervisor + specialist agents, controlled transitions | Explicit nodes/edges; no free-for-all agent chat (a design rule in the concept doc) |
| Human approval gate | Built-in interrupt / resume pattern for human-in-the-loop |
| Incremental reassessment | Conditional edges + re-entering at a chosen node with an updated state |
| Audit / trace | Checkpoint history gives step-level replay; integrates with tracing tools |
| Speed | Prebuilt agent helpers for tool-calling nodes; you only hand-write the supervisor logic |

### Alternatives (decide in Phase 0, then freeze)

| Option | Use it if | Trade-off |
|---|---|---|
| **Microsoft Agent Framework / Semantic Kernel** | Team is Azure/.NET-leaning, hackathon judges favour the Microsoft stack | Less community example code for state-machine style flows |
| **CrewAI** | You want the fastest "role + task" demo | Weaker explicit state/approval control; harder to enforce "agent owns only these fields" |
| **OpenAI Agents SDK** | You only use OpenAI models | Vendor-tied; handoffs are simpler than a stateful graph |

> **Verify before coding:** framework APIs change quickly. Pin exact versions in `pyproject.toml` on day 0 and have one person (T0.1) confirm the checkpoint and interrupt APIs against current docs.

### Supporting stack (prototype-first, Azure-optional)

| Layer | Prototype default | Production / hackathon-Azure swap | Notes |
|---|---|---|---|
| LLM | Any chat model behind `core/llm.py` wrapper | Azure OpenAI (text + vision) | Never import a provider SDK outside the wrapper |
| Doc extraction | PyMuPDF/pdfplumber + vision-LLM fallback for scans | Azure AI Document Intelligence | Same `ExtractionResult` contract either way |
| Structured outputs | Pydantic v2 + model structured output | same | Every agent returns a Pydantic model |
| Policy store | SQLite/Postgres tables + `pgvector`/Chroma/FAISS for clause retrieval | Azure AI Search (hybrid) | Start relational; add Neo4j only when multi-hop is justified |
| App DB | SQLite → Postgres via SQLAlchemy | Azure SQL / Postgres | Alembic migrations |
| Events | In-process queue + file-watch / API call | Azure Service Bus | Behind `events/bus.py` interface |
| API | FastAPI | same | OpenAPI doc doubles as the integration contract |
| UI | **Streamlit** (fastest for the Document-AI panel) | React later | Thin; calls the API only |
| Tracing | Langfuse or LangSmith (pick one) | Application Insights | Plus our own `audit_log` table |
| Tests/Evals | pytest + custom eval runner | same | Golden scenarios in `data/scenarios` |
| Tooling | `uv`/`poetry`, `ruff`, `mypy`, `pre-commit`, GitHub Actions | same | |

---

## 4. System Architecture

```
                         ┌──────────────────────────────────────────────┐
  Entry points           │  FastAPI  (/claims, /documents, /events …)   │
  FNOL · documents ─────▶│  Streamlit UI (Document-AI panel, dossier)   │
  images · emails        └──────────────┬───────────────────────────────┘
                                        │ event: claim_created / evidence_received / human_decision
                                        ▼
                        ┌──────────────────────────────────┐
                        │  Orchestrator (LangGraph)        │◀──── Checkpointer (versioned ClaimState)
                        │  Supervisor node + routing rules │
                        └───┬──────────────────────────────┘
        ┌───────────────────┼─────────────────────────────────────────────────┐
        ▼                   ▼                    ▼                ▼            ▼
  1 Intake Agent    2 Policy Agent       3 Evidence Agent   4 Gap Analysis   5 Coverage
  (FNOL, PAS        (clause graph,       (facts +            Agent ★         Reasoning
   lookup,           endorsements,        provenance,         (contradiction, Agent
   duplicates)       snapshot@loss date)  image findings)     missing, min.
                                                              evidence,
                                                              counterfactuals)
        └───────────────────┬─────────────────────────────────────────────────┘
                            ▼
  6 Settlement Agent ──▶ deterministic Calculation Engine (YAML rules, unit-tested)
  7 Action Planning Agent ──▶ task/notification drafts, controlled API calls
  8 Quality & Audit Agent ──▶ citation check, permission check, consistency check
                            │
                            ▼
              ┌────────────────────────────┐
              │ Routing/Tier rules engine  │  simple · medium · complex + reason codes
              └─────────────┬──────────────┘
                            ▼
              ╔════════════════════════════╗
              ║  HUMAN APPROVAL GATE       ║  adjuster / supervisor / senior authority
              ╚═════════════┬══════════════╝
                            ▼
   Integration layer (adapters; idempotent; correlation-ID; audit)
     PAS adapter (read-only)  ·  Claims adapter (create/update)  ·  Recon adapter (instruct/read-back)
        [mock in prototype]            [mock]                          [mock]
```

**Key flows**

| Flow | Trigger | What runs |
|---|---|---|
| New claim | `claim_created` | Intake → Policy → Evidence → Gap → Coverage → Settlement(draft) → Action → Audit → Routing → Human gate |
| New evidence | `evidence_received` | Evidence (new doc only) → Gap (affected questions only) → Coverage (affected issues only) → Settlement (if inputs changed) → Action → Audit |
| Human decision | `human_decision` | Audit → Claims adapter commit → Recon instruction → read-back |
| Payment retry | duplicate API call | Idempotency key short-circuits; no second payment |

★ = hero component.

---

## 5. Agent Roster & State Ownership

Rule: **each agent writes only the `ClaimState` fields it owns.** The orchestrator rejects patches that touch other fields (enforced in `core/state.py`, see Appendix).

| # | Agent | Responsibility | Owns (writes) | Tools it may call | Autonomy |
|---|---|---|---|---|---|
| 1 | Intake | FNOL normalisation, classification, required-field checklist, PAS lookup, duplicate-candidate check, triage | `claim_header`, `intake_checklist`, `duplicate_candidates` | PAS adapter (read), Claims adapter (create via orchestrator), duplicate service | Autonomous (extract/flag); claim creation via controlled API |
| 2 | Policy | Decompose policy → clauses, relations, precedence; snapshot effective on loss date | `policy_snapshot` | Policy store, retrieval | Autonomous |
| 3 | Evidence | Document type detection, field/fact extraction with provenance, image observations | `documents`, `facts`, `image_findings` | Doc extraction, vision | Autonomous |
| 4 | Gap Analysis ★ | Open-question register; contradiction / missing / insufficient classification; minimum-evidence recommendation; counterfactual scenarios | `open_questions`, `contradictions`, `missing_evidence`, `counterfactuals` | Read-only on state | Autonomous |
| 5 | Coverage Reasoning | Clause × fact → issue matrix; abstain when ambiguous | `coverage_issues` | Policy retrieval | Human-reviewed output |
| 6 | Settlement | Extract calc inputs → call engine → explain | `settlement` (inputs, steps, result, reserve vs payable) | Calculation engine | Human-reviewed output |
| 7 | Action Planning | Next-best actions, drafts, task creation | `proposed_actions`, `drafts` | Task API (controlled), notification drafts | Draft-only / controlled API |
| 8 | Quality & Audit | Citation validity, permission check, consistency, abstention check | `qa_report`, `audit_events` | Read-only | Can block release to human gate |
| — | Supervisor (not an LLM agent) | Routing, tier assignment, re-entry on new evidence | `workflow`, `routing` | Rules engine | Deterministic |

---

## 6. Core Contracts

These are the **first thing built** (Phase 0) because every workstream depends on them. All live in `src/claimtower/core/schemas/` as Pydantic models and are exported to JSON Schema in `docs/contracts/`.

### 6.1 `ClaimState` (persistent, versioned)

```python
class ClaimState(BaseModel):
    claim_id: str
    state_version: int                      # increments on every committed patch
    correlation_id: str                     # ties PAS/Claims/Recon/audit together
    claim_header: ClaimHeader               # claimant, policy ref, loss date, loss type, channel
    policy_snapshot: PolicySnapshot | None  # version effective on date of loss
    documents: list[DocumentRecord]
    facts: list[Fact]                       # with provenance
    image_findings: list[ImageFinding]
    open_questions: list[OpenQuestion]      # what must be known to decide
    contradictions: list[Contradiction]
    missing_evidence: list[MissingEvidence]
    counterfactuals: list[CounterfactualScenario]
    coverage_issues: list[CoverageIssue]
    settlement: SettlementResult | None
    duplicate_candidates: list[DuplicateCandidate]
    routing: RoutingDecision | None
    proposed_actions: list[ProposedAction]
    drafts: list[Draft]
    qa_report: QAReport | None
    human_approval_required: bool
    resolution_status: Literal["registered","assessing","awaiting_evidence",
                               "awaiting_human","approved","paid","closed"]
    audit_ref: str
```

### 6.2 `Fact` (provenance is mandatory)

```python
class Fact(BaseModel):
    fact_id: str
    claim_id: str
    proposition: str                  # "Water damage originated from a pipe joint"
    source_document_id: str
    page: int | None
    region: BBox | None               # image/page region if available
    evidence_text: str                # exact excerpt
    evidence_type: Literal["supporting","contradicting","neutral"]
    related_question_ids: list[str]
    extraction_confidence: float      # NOT probability of coverage
    verification_status: Literal["unverified","human_verified","rejected"]
    extractor_version: str
```

### 6.3 Other key contracts (fields summarised; full models in code)

| Model | Essential fields |
|---|---|
| `OpenQuestion` | `question_id`, `text`, `category` (coverage-condition / cause / timing / extent / amount), `status` (unverified / possible / missing / partially_established / established / requires_validation), `linked_clause_ids`, `linked_fact_ids` |
| `Contradiction` | `question_id`, `fact_ids_a`, `fact_ids_b`, `severity`, `explanation` |
| `MissingEvidence` | `question_id`, `gap_type` (missing / contradictory / insufficient), `minimum_evidence_request` (who, what exactly, which sub-questions it must answer), `resolves_question_ids` |
| `CounterfactualScenario` | `scenario_id` (A/B/C), `assumed_finding`, `affected_issue_ids`, `implication_text` (no approve/reject prediction), `next_step` |
| `CoverageIssue` | `issue`, `status` (established / identified / unresolved / potentially_relevant / incomplete / abstain), `clause_citations[]`, `fact_ids[]` |
| `PolicySnapshot` | `policy_id`, `version`, `effective_from/to`, `clauses[]`, `endorsements[]`, `precedence_rules`, `source_system_response_ref` |
| `SettlementResult` | `reserve_forecast`, `assessed_loss`, `payable`, `steps[]` (rule_id, input refs, amount before/after), `rule_version`, `status` (calculated / cannot_calculate + missing inputs) |
| `RoutingDecision` | `tier`, `reason_codes[]`, `required_approver_role`, `rules_version` |
| `ProposedAction` | `action_type`, `responsible_party`, `execution_mode` (draft / controlled_api), `status`, `idempotency_key` |
| `AuditEvent` | `ts`, `actor` (agent / human / system), `action`, `state_version_before/after`, `inputs_ref`, `outputs_ref`, `model_version`, `prompt_version` |

---

## 7. Project Structure

Principles: **(a)** contracts first, **(b)** one folder per capability so people don't collide, **(c)** adapters behind interfaces so mock → real is a config change, **(d)** *LOB packs* so a new insurance product is a new folder, not a refactor.

```
claim-control-tower/
├── README.md
├── pyproject.toml                  # pinned deps (framework version pinned here)
├── .env.example                    # LLM keys, DB URL, feature flags (USE_MOCK_PAS=true …)
├── Makefile                        # make setup | run-api | run-ui | test | eval | demo
├── docker-compose.yml              # api + ui + postgres (+ optional qdrant/neo4j)
├── .github/workflows/ci.yml        # lint + unit tests + schema-drift check
│
├── docs/
│   ├── BUILD_PLAN.md               # this file
│   ├── reference/                  # the two source docs
│   ├── contracts/                  # exported JSON Schemas + API examples
│   ├── adr/                        # architecture decision records (0001-framework.md …)
│   ├── demo/                       # demo script, slides, screen recordings
│   └── assumptions.md              # PAS/Recon/jurisdiction assumptions shown to reviewers
│
├── configs/                        # NO business rules hard-coded in Python
│   ├── app.yaml                    # feature flags, model names, thresholds for abstention
│   ├── autonomy_matrix.yaml        # which operation is autonomous / draft / controlled / human
│   ├── routing_rules.yaml          # simple/medium/complex decision table (SAMPLE values)
│   └── rbac.yaml                   # roles: intake, adjuster, supervisor, finance
│
├── packs/                          # ← extension point: one folder per line of business
│   └── property_water_damage/
│       ├── pack.yaml               # metadata, version, supported claim types
│       ├── document_types.yaml     # types the classifier may output + required fields each
│       ├── intake_checklist.yaml   # required docs, "request all at once" list
│       ├── question_templates.yaml # canonical open questions (sudden vs gradual, start date…)
│       ├── calculation/
│       │   └── settlement_v1.yaml  # ordered step list (see T5.3)
│       ├── prompts/                # versioned prompts per agent (*.md, front-matter version)
│       └── eval_expectations/      # pack-specific golden outputs
│
├── data/
│   ├── synthetic/
│   │   ├── policies/               # base policy PDFs + endorsements (SYNTHETIC)
│   │   ├── claims/CLM-0001/ …      # fnol.pdf, plumber_report.pdf, photos/, estimate.pdf, invoice.pdf
│   │   └── pas_mock_db.json        # mock policy master
│   ├── scenarios/scenarios.yaml    # 30-50 scenarios: inputs, expected issues, expected actions
│   ├── ground_truth/               # labelled facts, contradictions, clauses, calc results
│   └── README.md                   # every file labelled "synthetic"
│
├── src/claimtower/
│   ├── core/
│   │   ├── schemas/                # ClaimState, Fact, … (Pydantic)  ★ contract
│   │   ├── state.py                # patch/ownership enforcement, versioning
│   │   ├── llm.py                  # single LLM wrapper (text + vision + structured)
│   │   ├── config.py               # loads configs/ + pack
│   │   ├── ids.py · logging.py · errors.py
│   │   └── registry.py             # agent/adapter/pack registries (plug-in loading)
│   ├── ingestion/                  # WS2
│   │   ├── loaders.py              # pdf/image/docx → pages
│   │   ├── classifier.py           # document type detection
│   │   ├── extractors/             # per-doc-type field extractors
│   │   ├── provenance.py           # build Fact with page/region/text
│   │   └── store.py                # blob + metadata store
│   ├── policy/                     # WS3
│   │   ├── decompose.py            # sections → clause objects
│   │   ├── relations.py            # modifies / excepts / defines links, precedence
│   │   ├── snapshot.py             # as-of-date resolution
│   │   └── retrieval.py            # hybrid retrieval returning page+span
│   ├── evidence/                   # WS4 (hero)
│   │   ├── questions.py            # open-question register
│   │   ├── linker.py               # fact ↔ question linking
│   │   ├── contradiction.py
│   │   ├── sufficiency.py          # missing / contradictory / insufficient classifier
│   │   ├── min_evidence.py         # minimum-evidence request generator
│   │   └── counterfactual.py       # scenarios A/B/C
│   ├── coverage/                   # WS5a
│   │   └── issue_matrix.py
│   ├── settlement/                 # WS5b — NO LLM imports allowed here (lint-enforced)
│   │   ├── engine.py               # deterministic step interpreter
│   │   ├── rules_loader.py
│   │   ├── explain.py              # step trace → human text
│   │   └── money.py                # Decimal handling, currency, rounding
│   ├── routing/                    # WS5c
│   │   └── tiering.py              # decision-table evaluator + reason codes
│   ├── duplicates/                 # WS6c
│   │   ├── exact.py · fuzzy.py · invoice.py · payment_guard.py
│   ├── vision/                     # WS2b
│   │   ├── quality.py              # blur/glare/angle/resolution checks
│   │   └── observations.py         # visible-damage findings + next-photo suggestions
│   ├── actions/                    # WS6b
│   │   ├── planner.py · drafts.py · tasks.py
│   ├── agents/                     # thin wrappers: read state → call capability → return patch
│   │   ├── base.py                 # Agent protocol + ownership declaration
│   │   ├── intake.py · policy.py · evidence.py · gap.py
│   │   ├── coverage.py · settlement.py · action.py · audit.py
│   ├── orchestration/              # WS6a
│   │   ├── graph.py                # LangGraph definition
│   │   ├── supervisor.py           # routing, re-entry, incremental invalidation
│   │   ├── dependencies.py         # which outputs depend on which inputs (for reassessment)
│   │   ├── checkpoints.py
│   │   └── events.py               # bus interface + in-proc impl
│   ├── integrations/               # WS6d
│   │   ├── base.py                 # PASClient / ClaimsClient / ReconClient protocols
│   │   ├── mock/                   # pas_mock.py · claims_mock.py · recon_mock.py
│   │   ├── idempotency.py · retry.py · deadletter.py
│   │   └── live/                   # empty until real sandboxes exist
│   ├── audit/                      # WS8
│   │   ├── trail.py · explain_pack.py (decision dossier export)
│   │   └── guards.py               # autonomy-matrix + RBAC enforcement
│   ├── api/                        # WS7
│   │   ├── main.py · routers/ · deps.py
│   └── ui/                         # WS7
│       ├── app.py
│       └── components/             # doc_viewer · fact_table · issue_matrix · gap_board · calc_trace · timeline
│
├── tests/
│   ├── unit/                       # per module; settlement/ has the heaviest coverage
│   ├── integration/                # graph-level with mocked LLM (recorded responses)
│   └── e2e/                        # scenario replay through API
│
├── evals/
│   ├── datasets/ → symlink to data/ground_truth
│   ├── metrics/                    # extraction · retrieval · contradiction · gap · calc · abstention
│   ├── runners/run_eval.py
│   └── reports/                    # generated HTML/MD scorecards (git-ignored)
│
└── scripts/
    ├── generate_synthetic_docs.py  # builds PDFs/photos from scenario YAML
    ├── seed_pas_mock.py
    ├── export_schemas.py           # JSON Schema export (CI checks drift)
    └── replay_scenario.py
```

**Enhancement hooks built into this structure**

| Future enhancement | Where it plugs in | Change needed |
|---|---|---|
| New line of business (motor, health) | `packs/<new_lob>/` | Add pack folder; no core change |
| Real PAS / Claims / Recon | `integrations/live/` + `.env` flag | Implement the 3 protocols |
| Azure AI Document Intelligence / AI Search | `ingestion/loaders`, `policy/retrieval` backends | Swap backend class via config |
| Neo4j knowledge graph | `policy/relations.py` storage backend | Add graph backend; keep interface |
| Service Bus events | `orchestration/events.py` | Add bus implementation |
| New agent (e.g. subrogation) | `agents/` + registry | Declare owned fields; add node |
| New duplicate type (image-hash) | `duplicates/` | Register detector |
| Different calc rules | `packs/*/calculation/` | New YAML version; engine unchanged |

---

## 8. Build Phases & Milestones

The six phases in the concept doc are kept, with **Phase 0 (contracts)** added in front and **Phase 7 (hardening/demo)** added at the end. Phases 1–4 run **in parallel** once Phase 0 is done, because every capability is developed against **hand-written fixture states** (JSON `ClaimState` snapshots from the synthetic scenarios) instead of waiting for upstream agents.

| Phase | Name | Maps to concept-doc phase | Workstreams | Output |
|---|---|---|---|---|
| 0 | Foundations & contracts | — | WS0, WS1 (start) | Frozen schemas, repo, LLM wrapper, framework spike, first 5 scenarios |
| 1 | Document intelligence | Phase 1 | WS2 | Classified documents → facts with provenance; image findings |
| 2 | Policy intelligence | Phase 2 | WS3 | Versioned clause store; snapshot at date of loss |
| 3 | Evidence reasoning ★ | Phase 3 | WS4 | Evidence-sufficiency report; minimum-evidence requests; counterfactuals |
| 4 | Coverage reasoning | Phase 4 | WS5a | Issue matrix with citations; abstention |
| 5 | Agentic orchestration | Phase 5 | WS6 | Supervisor graph, actions, incremental reassessment, mock integrations |
| 6 | Settlement, routing & audit | Phase 6 | WS5b/c, WS8 | Deterministic calc, tiering, human gate, audit trail |
| 7 | Hardening, eval & demo | — | All | Scorecard, rehearsed demo |

### Milestones (integration checkpoints — the team syncs here)

| Milestone | Definition | Gate owner |
|---|---|---|
| **M0 — Contracts frozen** | `ClaimState` + sub-models, agent protocol, adapter protocols exported as JSON Schema; 5 scenarios + ground truth ready | Tech lead |
| **M1 — Capabilities green on fixtures** | Each of WS2/3/4/5 passes its unit tests and golden fixtures independently | Each workstream lead |
| **M2 — Vertical slice** | Scenario S-01 (running example) flows intake → … → human gate through the real graph with real agents; UI shows it | Tech lead + UI lead |
| **M3 — Incremental & controlled** | New plumber report triggers partial reassessment; human gate works; mock PAS/Claims/Recon wired; payment retry safe | Orchestration lead |
| **M4 — Scorecard** | Eval harness runs all scenarios, publishes metrics; failures triaged | Quality lead |
| **M5 — Demo ready** | Rehearsed end-to-end script with MOCK/SYNTHETIC labels | Everyone |

> **Time scaling.** Unknown timeline → proportions: Phase 0 ≈ 10%, Phases 1–6 in parallel ≈ 55%, M2→M3 integration ≈ 15%, Phase 7 ≈ 20%. If time is very short, cut at the priority tags below (P1 = must, P2 = should, P3 = stretch).

---

## 9. Workstreams & Task Cards

**How to read a task card**

- **Priority:** P1 must-have for demo · P2 should-have · P3 stretch
- **Size:** S ≈ half a day · M ≈ 1 day · L ≈ 2+ days (one person, focused)
- **Inputs:** what you need *from other people or files* before starting
- **Deliverables:** concrete artefacts you hand over
- **Done when:** objective acceptance test

Every task's "Inputs" can be satisfied by **fixtures** from WS1 if the upstream task isn't finished — never block on another person's code.

---

### WS0 — Foundations (Tech lead)

**T0.1 — Repo, tooling and framework spike** · P1 · M
- *Purpose:* Create a working skeleton and prove the chosen agent framework supports what we need (typed state, checkpoint/resume, interrupt for human approval, re-entry at a chosen node).
- *Inputs:* This plan; framework docs; pinned Python version.
- *Deliverables:* Repo scaffold per §7; `pyproject.toml` with pinned versions; `Makefile`; pre-commit; a 30-line spike graph (two dummy nodes + interrupt + resume) under `tests/integration/test_spike.py`; ADR-0001 framework decision.
- *Done when:* `make setup && make test` passes on a clean machine; spike proves interrupt → resume → partial re-run.

**T0.2 — Core schemas and `ClaimState`** · P1 · L
- *Purpose:* The contract every workstream builds against; the single most important Phase 0 task.
- *Inputs:* §6 of this plan; concept doc §4 and §7; PDF §2.2 and §6.
- *Deliverables:* Pydantic models in `core/schemas/`; JSON Schemas in `docs/contracts/`; example valid/invalid instances; short `docs/contracts/README.md` explaining each field's owner.
- *Done when:* Peer-reviewed by one person from each workstream; `scripts/export_schemas.py` runs in CI and fails on drift. **Schema changes after M0 need Tech-lead approval.**

**T0.3 — State patching, ownership enforcement and versioning** · P1 · M
- *Purpose:* Guarantee "each agent writes only its owned fields" and that every committed patch increments `state_version` (so adjusters can see how the assessment changed).
- *Inputs:* T0.2; ownership table in §5.
- *Deliverables:* `core/state.py` (`apply_patch(state, patch, agent_id)`), ownership map, version history store, unit tests for rejection of illegal writes.
- *Done when:* Writing a foreign field raises `OwnershipViolation`; history can be replayed to reconstruct any version.

**T0.4 — LLM wrapper (text, vision, structured output)** · P1 · M
- *Purpose:* One place to swap providers (Azure OpenAI ↔ others), set temperature, retry, log token use, and record model/prompt version for audit.
- *Inputs:* API keys; model names in `configs/app.yaml`.
- *Deliverables:* `core/llm.py` with `chat()`, `structured(model_cls, prompt, …)`, `vision()`; record/replay mode (cassette files) so tests run offline; prompt loader that reads versioned prompts from the pack.
- *Done when:* Same call works against two providers via config; replay mode makes integration tests deterministic.

**T0.5 — Config and pack loader** · P1 · S
- *Purpose:* Keep business rules out of code; make a new LOB a folder.
- *Inputs:* `configs/`, `packs/property_water_damage/pack.yaml`.
- *Deliverables:* `core/config.py`, `core/registry.py` (loads agents, adapters, packs by name), validation of YAML against schemas.
- *Done when:* Changing `routing_rules.yaml` changes routing without a code change; invalid YAML fails at startup with a clear message.

**T0.6 — CI, conventions and import-boundary lint** · P2 · S
- *Purpose:* Prevent architectural drift as multiple people commit.
- *Inputs:* T0.1.
- *Deliverables:* GitHub Actions (ruff, mypy, pytest, schema-drift); import-linter rule: **`settlement/` must not import any LLM module**; branch/PR conventions; CODEOWNERS per folder.
- *Done when:* A PR importing `llm` inside `settlement/` fails CI.

---

### WS1 — Synthetic Data & Domain Content (Domain/data lead)

**T1.1 — Policy corpus (base wording + endorsements)** · P1 · L
- *Purpose:* Without realistic, *tricky* policy wording the policy and coverage agents can't be tested. Include "sudden and accidental" cover, "gradual leakage" exclusion, definitions, conditions, an endorsement that adds a water-damage sublimit, and a second endorsement with a different effective date.
- *Inputs:* Concept doc §2 and §9; PDF §8.1 document list; any real public sample policy wordings for style.
- *Deliverables:* 3–4 synthetic policy PDFs (+ DOCX source), 4–6 endorsements, each page-numbered; `data/synthetic/policies/INDEX.md` stating clause IDs and intended difficulty. **All marked SYNTHETIC.**
- *Done when:* Domain reviewer confirms wording is plausible; clause IDs are stable (used by ground truth).

**T1.2 — Scenario catalogue + ground truth** · P1 · L
- *Purpose:* The 30–50 designed scenarios drive development *and* evaluation. Ground truth is what makes the evidence engine measurable.
- *Inputs:* §12 of this plan; T1.1.
- *Deliverables:* `data/scenarios/scenarios.yaml` (inputs, documents, events in time order); `data/ground_truth/*.json` per scenario: expected facts, contradictions, missing evidence + minimum request, issue-matrix statuses, expected tier, expected calculation. First **5 scenarios by M0**, all 30+ by M1.
- *Done when:* Two people independently label 5 scenarios and agree ≥90% (disagreements resolved in a note).

**T1.3 — Synthetic claim document + photo generator** · P1 · M
- *Purpose:* Produce PDFs/images for each scenario programmatically so scenarios are reproducible and cheap to add.
- *Inputs:* `scenarios.yaml`; templates for FNOL, plumber report, surveyor report, estimate, invoice, emails.
- *Deliverables:* `scripts/generate_synthetic_docs.py`; output under `data/synthetic/claims/CLM-xxxx/`; photo set (real stock/own photos or generated, clearly licensed) including good, blurry, and unrelated ones.
- *Done when:* `make data` regenerates the corpus; documents contain realistic layout variance (tables, stamps, one scanned/rotated page).

**T1.4 — Mock PAS data, prior claims, invoices registry** · P1 · S
- *Purpose:* Back the mock PAS (policy master + versions) and duplicate-detection tests (prior claims, repeated invoice numbers).
- *Inputs:* T1.1; PDF §3 duplicate types.
- *Deliverables:* `pas_mock_db.json` (policies with versions/status/lapse dates), `prior_claims.json`, `invoice_registry.json`.
- *Done when:* Includes lapse boundary, cancelled-then-reinstated policy and a policy with two versions straddling the loss date.

**T1.5 — Sample business rules and approval matrix (labelled SAMPLE)** · P1 · S
- *Purpose:* Provide plausible values for routing and calculation without inventing insurer policy.
- *Inputs:* PDF §4 and §6.3; domain judgement.
- *Deliverables:* Draft `routing_rules.yaml` and `settlement_v1.yaml` values; a note `docs/assumptions.md` that says "SAMPLE — not insurer policy"; list of questions for a real claims domain expert.
- *Done when:* Every number in these files has an inline `# SAMPLE` comment.

---

### WS2 — Document & Image Intelligence (Phase 1)

**T2.1 — Loaders and document store** · P1 · M
- *Purpose:* Uniformly turn PDFs, images, DOCX and emails into addressable pages with stored originals (originals are evidence and must never be altered).
- *Inputs:* Synthetic claim files (T1.3).
- *Deliverables:* `ingestion/loaders.py`, `ingestion/store.py` (hash, blob path, metadata: uploader, source channel, upload time, page count); OCR path for scans (local OCR or vision-LLM fallback).
- *Done when:* Every file yields `DocumentRecord` with SHA-256, page count, text per page (or OCR flag); original byte-identical on retrieval.

**T2.2 — Document type classifier** · P1 · M
- *Purpose:* Reviewers specifically asked to see *what type of document* is being read. Output must be visible and justified.
- *Inputs:* `packs/*/document_types.yaml` (FNOL, policy schedule, wording, endorsement, plumber report, surveyor report, estimate, invoice, photo, email, other); T2.1.
- *Deliverables:* `ingestion/classifier.py` returning `{type, confidence, evidence_snippet}`; abstain → `unknown`; classification eval on the corpus.
- *Done when:* ≥ target accuracy agreed with lead on labelled set (record the baseline first); unknowns routed to human, never guessed.

**T2.3 — Field and table extractors per document type** · P1 · L
- *Purpose:* Extract the 10–20 critical fields (claimant, policy no., loss date/time, location, cause narrative, invoice no./date/vendor, line items, totals, surveyor findings, dates).
- *Inputs:* Classified docs; field specs in `document_types.yaml`; prompts in pack.
- *Deliverables:* `ingestion/extractors/*` returning `ExtractedField{name, value, page, region, confidence}`; table extraction for estimates/invoices with arithmetic self-check (line sum vs total) done in code.
- *Done when:* Field-level normalised-match score recorded on labelled set; low-confidence values flagged, not hidden.

**T2.4 — Facts with provenance** · P1 · M
- *Purpose:* Convert extracted content into `Fact` objects (proposition + exact evidence text + page/region + extraction confidence). Evidence Agent output feeds the whole hero engine.
- *Inputs:* T2.3 output; `Fact` schema; open-question templates (to tag `related_question_ids`).
- *Deliverables:* `ingestion/provenance.py`, `agents/evidence.py`; guarantee that `evidence_text` is a verbatim substring of the page text (validated in code).
- *Done when:* 100% of facts pass the substring check; citation-correctness metric computed.

**T2.5 — Image quality check** · P2 · M
- *Purpose:* Tell the customer/adjuster whether photos are usable (blur, glare, occlusion, resolution, angle, coverage of required views) before anyone reasons from them.
- *Inputs:* Photos (good/blurry/unrelated); required views from `intake_checklist.yaml`.
- *Deliverables:* `vision/quality.py` (classical metrics + optional model); quality verdict with reasons.
- *Done when:* Blur and low-res cases detected on the test set; "unrelated photo" routed to human.

**T2.6 — Visible-damage observations and next-photo suggestions** · P2 · M
- *Purpose:* Bounded vision task: describe *visible* water damage (stains, swelling, mould-like discolouration, affected component/room) and suggest the next evidence (close-up of joint, wide shot, moisture-meter reading). **Must not infer cause, date or coverage.**
- *Inputs:* Photos; prompt with the demo-safe wording from PDF §5.2.
- *Deliverables:* `vision/observations.py` → `ImageFinding{observation, region, confidence, limitation_note, is_visible_fact|is_inference}`.
- *Done when:* Zero outputs asserting cause/coverage on the test set (automated phrase + LLM-judge check); abstains on inconclusive images.

---

### WS3 — Policy Intelligence (Phase 2)

**T3.1 — Policy decomposition** · P1 · L
- *Purpose:* Turn policy PDFs into typed clause objects (coverage, exclusion, definition, condition, limit/sublimit, deductible) with page and text span.
- *Inputs:* Policy corpus (T1.1); clause-type taxonomy.
- *Deliverables:* `policy/decompose.py`, `PolicyClause` model, stored in relational tables; clause-level provenance.
- *Done when:* Clause extraction F1 against labelled clauses recorded; every clause links to a verbatim span.

**T3.2 — Relations and precedence** · P1 · L
- *Purpose:* Capture how documents modify each other (endorsement *modifies* clause X; exception *carves out* exclusion Y; definition *governs* term Z) and the precedence rule (e.g., later/specific endorsement prevails — stated per policy, never assumed silently).
- *Inputs:* T3.1 output; ground-truth relation labels.
- *Deliverables:* `policy/relations.py`, relation table (`modifies`, `excepts`, `defines`, `conditions`), precedence-rule extractor with abstain.
- *Done when:* On the "missing endorsement" and "two endorsements, different dates" scenarios the engine returns the correct effective clause set.

**T3.3 — Snapshot as of date of loss** · P1 · M
- *Purpose:* The contract in force on the incident date — base wording + endorsements with the correct effective dates + PAS status (in force / lapsed / cancelled).
- *Inputs:* T3.2; PAS mock `get_policy_version(policy_no, as_of)`.
- *Deliverables:* `policy/snapshot.py` → `PolicySnapshot` (with source-system response ref and timestamp).
- *Done when:* Lapse-boundary, reinstatement and straddling-version scenarios all produce the right snapshot or an explicit `not_in_force`.

**T3.4 — Clause retrieval with citations** · P1 · M
- *Purpose:* Hybrid retrieval (keyword + embeddings) that returns clause **plus its modifying endorsements** — the failure mode of plain vector search described in the concept doc.
- *Inputs:* T3.1–T3.3; question text.
- *Deliverables:* `policy/retrieval.py` returning `ClauseCitation{clause_id, doc, page, span, text, modified_by[]}`; retrieval eval set.
- *Done when:* Retrieval accuracy metric recorded; for every water-damage question the modifying endorsement is returned with the base clause.

**T3.5 — Policy Agent wrapper** · P1 · S
- *Purpose:* Thin agent that reads `claim_header`, calls snapshot builder, writes `policy_snapshot` only.
- *Inputs:* T3.3, T0.3.
- *Deliverables:* `agents/policy.py` implementing the Agent protocol.
- *Done when:* Passes the contract test (writes only `policy_snapshot`).

---

### WS4 — Evidence Sufficiency & Next-Best-Investigation (Phase 3) ★ HERO

> Put your strongest engineer(s) here. This workstream is what reviewers will remember.

**T4.1 — Open-question register** · P1 · M
- *Purpose:* Define *what must be known to decide* for this product and claim. Questions come from the pack's templates (sudden vs gradual? start date? extent? cost reasonableness? policy in force?) and from the clauses the snapshot makes relevant.
- *Inputs:* `question_templates.yaml`; `policy_snapshot`; claim header.
- *Deliverables:* `evidence/questions.py` generating `OpenQuestion[]` with linked clause IDs and required-fact criteria ("what would establish this").
- *Done when:* For the running example it produces the five canonical questions in the concept doc with correct links to the exclusion and the sudden-and-accidental condition.

**T4.2 — Fact ↔ question linker** · P1 · M
- *Purpose:* Attach each fact to the questions it bears on, with direction (supports / contradicts / neutral).
- *Inputs:* `facts[]` (fixtures if WS2 not ready); `open_questions[]`.
- *Deliverables:* `evidence/linker.py`; link table with rationale text.
- *Done when:* Link precision/recall measured against labelled links.

**T4.3 — Contradiction detector** · P1 · L
- *Purpose:* Identify conflicting statements across documents (customer: "sudden burst" vs plumber/surveyor: "long-term corrosion / moisture in multiple areas"), including conflicts in numbers and dates.
- *Inputs:* Linked facts; scenarios with injected contradictions.
- *Deliverables:* `evidence/contradiction.py` returning `Contradiction{fact_ids_a, fact_ids_b, severity, explanation}`; rule-based checks for dates/amounts + LLM reasoning for narrative conflicts; severity rubric.
- *Done when:* Contradiction precision/recall reported by type (date, amount, cause, extent); no contradiction without two cited facts.

**T4.4 — Sufficiency classifier (the three problem types)** · P1 · L
- *Purpose:* For each open question decide: **contradictory**, **missing**, or **exists but insufficient** — and the resulting status (established / partially established / possible / unverified / requires validation).
- *Inputs:* Links, contradictions, "what would establish this" criteria from T4.1.
- *Deliverables:* `evidence/sufficiency.py`; decision table in `packs/*/question_templates.yaml` (e.g., "start date: need dated technical evidence or dated maintenance record") so logic is auditable.
- *Done when:* ≥ agreed accuracy on the gap-type label; unit tests for each of the three types; *abstain* option used when evidence quality is unclear.

**T4.5 — Minimum-evidence request generator** · P1 · L
- *Purpose:* Instead of "upload more documents", generate the **smallest, most specific request** that would resolve the open question — who to ask, what to ask exactly, and which sub-questions it must answer (e.g., plumber: failure mechanism; whether corrosion predates incident; consistency with sudden vs prolonged leakage). Must *group* all needed items into one request (regulatory aim: don't request piecemeal).
- *Inputs:* `missing_evidence` classification; document-type catalogue; contact roles.
- *Deliverables:* `evidence/min_evidence.py` → `MissingEvidence.minimum_evidence_request`; a "does this request actually resolve the question?" self-check (each requested item must map to a criterion in T4.4).
- *Done when:* Domain reviewer rates requests "sufficient and minimal" on ≥ agreed % of scenarios; zero requests that duplicate documents already received.

**T4.6 — Counterfactual engine ("What would resolve this claim?")** · P1 · M
- *Purpose:* For each unresolved material question produce scenarios A (supports sudden), B (supports prolonged), C (inconclusive) and map which coverage issues would be re-assessed and what next step each implies. **Never predict approve/reject.**
- *Inputs:* `open_questions`, `coverage_issues` (fixture), `missing_evidence`.
- *Deliverables:* `evidence/counterfactual.py` → `CounterfactualScenario[]`; wording linter that blocks phrases like "will be approved/denied".
- *Done when:* Running example produces the table in concept doc §5; linter passes on all scenarios.

**T4.7 — Gap Agent wrapper + "affected questions" logic** · P1 · M
- *Purpose:* On new evidence, compute *which open questions the new document touches* and re-run only those (feeds incremental reassessment).
- *Inputs:* Previous state version; new facts.
- *Deliverables:* `agents/gap.py`; function `affected_questions(prev_state, new_facts) -> set[question_id]`; diff of status changes between state versions.
- *Done when:* Adding a plumber supplementary report changes only the questions linked to its facts; unaffected questions retain identical results.

---

### WS5 — Coverage, Settlement & Routing (Phases 4 & 6)

**T5.1 — Coverage issue matrix** · P1 · L
- *Purpose:* Combine clauses × facts × open questions into the issue matrix with citations and statuses (established / identified / unresolved / potentially relevant / incomplete / abstain). No binary coverage verdict.
- *Inputs:* `policy_snapshot`, `facts`, `open_questions`, `contradictions`.
- *Deliverables:* `coverage/issue_matrix.py`, `agents/coverage.py`.
- *Done when:* Every row has ≥1 clause citation and, where relevant, fact IDs; the running example yields the concept doc's matrix.

**T5.2 — Abstention policy** · P1 · S
- *Purpose:* Make "I cannot conclude" a first-class, measured behaviour.
- *Inputs:* `configs/app.yaml` abstention thresholds; ambiguity scenarios.
- *Deliverables:* Rules: abstain when a controlling question is unresolved, clause is ambiguous, or citations are missing; abstention counted in metrics.
- *Done when:* On ambiguous-clause scenarios the system abstains (target agreed after first baseline run); no output uses "covered / not covered" as a final conclusion.

**T5.3 — Deterministic calculation engine** · P1 · L
- *Purpose:* Compute settlement with Decimal arithmetic by interpreting an ordered, versioned step list from YAML (so formula order differs per product — see §2 decision and Appendix A.3).
- *Inputs:* `settlement_v1.yaml`; typed input object; rounding/currency rules.
- *Deliverables:* `settlement/engine.py`, `money.py`, `rules_loader.py`; step trace output (`rule_id`, inputs, before/after); `cannot_calculate` result listing missing inputs.
- *Done when:* No float usage; no LLM imports; returns identical output for identical inputs; unit tests include PDF §6.3 example (200,000 → 150,000) and concept-doc formula example.

**T5.4 — Calculation rule set and edge-case test suite** · P1 · M
- *Purpose:* Lock behaviour for the edge cases listed in PDF §6.4.
- *Inputs:* T5.3; T1.5 sample rules.
- *Deliverables:* Tests for zero/negative amounts, limit exhaustion, sublimit, repeated deductible, prior payments, rounding, depreciation, cancelled/reinstated policy; property-based tests (amount never exceeds remaining limit; never negative).
- *Done when:* ≥95% line coverage on `settlement/`; all edge cases pass.

**T5.5 — Settlement Agent (extract → compute → explain)** · P1 · M
- *Purpose:* The LLM only extracts and classifies inputs (eligible items, non-covered items, deductible clause, prior payments) with citations; the engine computes; explanation text is generated from the trace and **re-verified** (displayed total must equal engine output).
- *Inputs:* `facts`, `policy_snapshot`, T5.3.
- *Deliverables:* `agents/settlement.py`; explanation generator in `settlement/explain.py`; separate fields for **reserve/forecast**, **assessed loss**, **payable** (PDF §6.1).
- *Done when:* Explanation-total mismatch test fails the build; missing/contradictory input produces `cannot_calculate_final_settlement_yet` with the missing items, never a fabricated number.

**T5.6 — Routing / tiering engine** · P1 · M
- *Purpose:* Simple / medium / complex with reason codes, driven by `routing_rules.yaml`, not LLM confidence.
- *Inputs:* policy certainty, evidence completeness, contradiction severity, duplicate status, exposure, special flags.
- *Deliverables:* `routing/tiering.py`, decision-table evaluator, reason codes, required approver role; "mandatory referral" overrides.
- *Done when:* Expert-assigned tier agreement measured on scenarios; changing YAML changes routing; extraction confidence is *not* an input feature.

---

### WS6 — Orchestration, Integrations, Intake & Duplicates (Phase 5)

**T6.1 — Graph and supervisor** · P1 · L
- *Purpose:* The stateful workflow: nodes for the eight agents, supervisor transitions, retry/failure handling, human gate, termination.
- *Inputs:* Agent protocol (T0.3); agent wrappers (stubs OK first); autonomy matrix.
- *Deliverables:* `orchestration/graph.py`, `supervisor.py`, checkpointer wiring, run IDs; stub agents returning fixture patches so the graph runs on day 1 of Phase 5.
- *Done when:* M2 vertical slice passes end-to-end; failed agent → `awaiting_human` with reason, not silent loop.

**T6.2 — Incremental reassessment** · P1 · L
- *Purpose:* Avoid re-running everything when evidence arrives. Maintain a dependency map: which outputs depend on which inputs (policy unchanged → skip Policy Agent; new plumber report → Evidence (new doc only) → Gap (affected questions) → Coverage (affected issues) → Settlement if inputs changed).
- *Inputs:* `affected_questions()` (T4.7); node dependency definitions.
- *Deliverables:* `orchestration/dependencies.py`; supervisor invalidation logic; run log showing "skipped (unchanged)" vs "re-run (reason)".
- *Done when:* Scenario with late evidence shows ≥1 skipped node and correct changed statuses; state versions v1→v2 diff visible.

**T6.3 — Human approval gate** · P1 · M
- *Purpose:* Pause at the right points; accept approve / edit / reject / request-more-evidence with reason; record reviewer identity and overrides.
- *Inputs:* `routing` result; `autonomy_matrix.yaml`; RBAC.
- *Deliverables:* Interrupt/resume integration; `human_decision` event schema; override-reason capture.
- *Done when:* No path exists to payment instruction without a recorded authorised human decision (tested).

**T6.4 — Action planner, drafts and tasks** · P1 · M
- *Purpose:* Turn gaps into next steps: draft information request (customer/plumber), create inspection task, assign estimate review, draft customer status message.
- *Inputs:* `missing_evidence`, `routing`; templates in pack.
- *Deliverables:* `actions/planner.py`, `drafts.py`, `tasks.py`; each action marked `draft` or `controlled_api` per matrix; idempotency key per action.
- *Done when:* Drafts include a single consolidated request; re-running the planner creates **no duplicate tasks**.

**T6.5 — Integration protocols and mock adapters (PAS, Claims, Recon)** · P1 · L
- *Purpose:* Demonstrate the *production contract* without pretending mocks are live.
- *Inputs:* PDF §1 integration table; `pas_mock_db.json`.
- *Deliverables:* `PASClient` (read-only: `get_policy_version`, `get_status`), `ClaimsClient` (`create_claim`, `update_status`, `create_task`, `get_claim`), `ReconClient` (`submit_payment_instruction`, `get_match_status`, `list_breaks`); mock implementations with realistic latency/failure injection; OpenAPI examples; a visible "MOCK" flag in every response.
- *Done when:* Swapping mock→live is a config flag; failure injection (timeout, duplicate message, stale snapshot) is exercised in tests.

**T6.6 — Idempotency, retry and dead-letter** · P1 · M
- *Purpose:* "Duplicate payment prevention is a transaction-control problem" (PDF §3.2). A retry must never create a second claim, task or payment.
- *Inputs:* T6.5.
- *Deliverables:* `integrations/idempotency.py` (unique external reference store), bounded retry with backoff, dead-letter/manual-repair queue table.
- *Done when:* Replaying the same payment instruction 5× yields exactly one Recon record; poison message lands in dead-letter.

**T6.7 — Intake agent (FNOL, checklist, PAS lookup)** · P1 · M
- *Purpose:* Normalise FNOL, validate required fields, exact policy lookup (no guessing), consolidated required-document checklist, create claim via Claims adapter.
- *Inputs:* FNOL docs/forms, `intake_checklist.yaml`, PAS mock.
- *Deliverables:* `agents/intake.py`; registration acknowledgement draft; "policy not effective on loss date" and "no exact match → operator" branches.
- *Done when:* Registration test group from PDF §11.1 passes (complete, missing field, invalid policy ref, duplicate submission, not effective).

**T6.8 — Duplicate detection (FNOL, invoice, payment)** · P2 · L
- *Purpose:* Flag *candidates*, never conclude fraud or auto-merge. Exact → entity-resolved → fuzzy; show matched/different fields and reversible link.
- *Inputs:* `prior_claims.json`, `invoice_registry.json`; matching-attribute whitelist.
- *Deliverables:* `duplicates/exact.py`, `fuzzy.py`, `invoice.py`, `payment_guard.py`; `DuplicateCandidate{type, matched_fields, differing_fields, prior_outcome, disposition}`; UI-ready explanation text; label wording "possible duplicate" (never "fraud").
- *Done when:* Precision/recall reported per duplicate type; the "legitimate near-match" scenario is flagged for review but **not** rejected.

**T6.9 — Event bus** · P2 · S
- *Purpose:* Event-driven trigger for new evidence (`evidence_received`) and decisions.
- *Deliverables:* `orchestration/events.py` interface + in-process implementation; folder-watch demo trigger; hooks for a Service Bus implementation later.
- *Done when:* Dropping a file into a claim folder or calling the API triggers the incremental run.

---

### WS7 — API & UI (Reviewer-facing)

**T7.1 — FastAPI service** · P1 · M
- *Purpose:* Stable surface for UI, tests and later integrations.
- *Inputs:* Schemas; graph entry points.
- *Deliverables:* Endpoints: `POST /claims`, `POST /claims/{id}/documents`, `GET /claims/{id}` (state + version), `GET /claims/{id}/versions`, `GET /claims/{id}/diff?from=&to=`, `POST /claims/{id}/decision`, `GET /claims/{id}/audit`, `GET /health`; OpenAPI doc.
- *Done when:* UI uses only these endpoints; contract tests green.

**T7.2 — Streamlit shell: claim list and dossier view** · P1 · M
- *Purpose:* One screen per claim showing status, tier, open questions count, timeline.
- *Deliverables:* `ui/app.py`, navigation, MOCK/SYNTHETIC badges.
- *Done when:* Can create a scenario claim from the UI and watch status change.

**T7.3 — Document-AI panel (required by reviewers)** · P1 · L
- *Purpose:* Show **exactly what the system is reading**: original file next to detected type, page count, uploader/source, extracted fields with page/region, low-confidence markers, source excerpt beside each material finding, duplicate badge with match reasons, agent action log.
- *Inputs:* `documents`, `facts`, `duplicate_candidates`, `audit_events` from API.
- *Deliverables:* `ui/components/doc_viewer.py`, `fact_table.py`.
- *Done when:* Clicking a fact highlights/jumps to its page; no red "fraud" label appears anywhere.

**T7.4 — Gap board and counterfactual view** · P1 · M
- *Purpose:* Visualise the hero feature: open questions by status, contradictions side by side, minimum-evidence requests, scenarios A/B/C.
- *Deliverables:* `ui/components/gap_board.py`, `counterfactual_view.py`.
- *Done when:* Running example displays the three problem types distinctly and the scenario table.

**T7.5 — Issue matrix, calculation trace and approval UI** · P1 · M
- *Purpose:* Let the adjuster inspect clause citations, see reserve vs assessed vs payable, step-by-step calc trace, then Approve / Edit / Request evidence / Reject-with-reason.
- *Deliverables:* `issue_matrix.py`, `calc_trace.py`, approval form posting to `/decision`.
- *Done when:* Override reason is mandatory and appears in audit view.

**T7.6 — State-version timeline and metrics dashboard** · P2 · M
- *Purpose:* Show v1→v2 diff after new evidence (incremental reassessment) and baseline-vs-pilot metrics view.
- *Deliverables:* `timeline.py`, diff renderer, metrics page reading `evals/reports` and benefits module (T8.6).
- *Done when:* Late-evidence scenario visually shows what changed and which steps were skipped.

---

### WS8 — Quality, Audit, Governance & Evaluation

**T8.1 — Evaluation harness and metrics** · P1 · L
- *Purpose:* Turn "it seems to work" into numbers (concept doc §10, PDF §11.2).
- *Inputs:* `data/ground_truth`; agent outputs.
- *Deliverables:* `evals/metrics/*` (extraction match, classification accuracy, clause retrieval accuracy, contradiction P/R, missing-evidence recall, citation correctness, calculation exact-match, abstention rate, routing agreement, duplicate P/R by type); `evals/runners/run_eval.py`; HTML/MD scorecard.
- *Done when:* `make eval` produces a scorecard and a per-scenario failure list; baseline numbers committed.

**T8.2 — Scenario replay and regression tests** · P1 · M
- *Purpose:* Every scenario becomes an automated end-to-end test using recorded LLM responses.
- *Deliverables:* `tests/e2e/`, `scripts/replay_scenario.py`, LLM cassette recording.
- *Done when:* CI replays ≥10 scenarios deterministically.

**T8.3 — Audit trail and decision dossier export** · P1 · M
- *Purpose:* Capture original evidence refs, policy wording used, agent outputs, calc inputs, human overrides, final decision, model/prompt versions.
- *Deliverables:* `audit/trail.py` (append-only), `audit/explain_pack.py` → one-click PDF/MD "decision pack".
- *Done when:* Any final decision can be reconstructed from the audit log alone.

**T8.4 — Guards: autonomy matrix and RBAC** · P1 · M
- *Purpose:* Enforce concept doc §11 programmatically.
- *Inputs:* `autonomy_matrix.yaml`, `rbac.yaml`.
- *Deliverables:* `audit/guards.py` decorators on tool calls; denial events logged; test "agent attempts denial / payment release → blocked".
- *Done when:* Governance test group (PDF §11.1) passes: missing citation, low-confidence extraction, model unavailable, human override, action without authority.

**T8.5 — Quality & Audit Agent** · P1 · M
- *Purpose:* Pre-gate checks: every assertion has a valid citation, claims in explanations match state, no forbidden language ("fraud confirmed", "claim approved"), required approvals present.
- *Deliverables:* `agents/audit.py` → `QAReport{passed, failures[]}`; failing report blocks release to the human gate with reasons.
- *Done when:* Seeded defects (fabricated citation, mismatched total) are caught.

**T8.6 — Benefits & metrics module** · P2 · S
- *Purpose:* Implement PDF §7 formulas (handling-time reduction %, capacity value, avoided rework, gross/net benefit, first-year ROI) as functions that **refuse to produce a number without a baseline input**.
- *Deliverables:* `evals/benefits.py`, unit tests, UI hook.
- *Done when:* Calling without baseline returns an explicit "baseline required"; no hard-coded improvement percentages exist in the repo.

**T8.7 — Demo script and rehearsal** · P1 · M
- *Purpose:* One tight narrative proving the full journey with honest MOCK/SYNTHETIC labels (see §14).
- *Deliverables:* `docs/demo/script.md`, backup recorded run, fallback offline mode.
- *Done when:* Two full rehearsals complete within the time limit, one with network disabled.

---

## 10. Team Assignment Models

Assign **one owner per workstream** and one named backup. Folders map to owners (CODEOWNERS) so merge conflicts stay rare.

### 10.1 Six-person split (recommended)

| Person | Role | Owns | Tasks |
|---|---|---|---|
| **P1** | Tech lead / Orchestration | WS0, WS6a | T0.1–T0.6, T6.1–T6.3, T6.9 |
| **P2** | Document & vision engineer | WS2 | T2.1–T2.6 |
| **P3** | Policy & coverage engineer | WS3, WS5a | T3.1–T3.5, T5.1, T5.2 |
| **P4** | Evidence/gap engineer ★ (strongest reasoner) | WS4 | T4.1–T4.7 |
| **P5** | Settlement, integrations & duplicates | WS5b/c, WS6b/c/d | T5.3–T5.6, T6.4–T6.8 |
| **P6** | Data, UI & quality | WS1, WS7, WS8 | T1.1–T1.5, T7.1–T7.6, T8.1–T8.7 |

*Load note:* P6 carries the most tasks — move T8.1/T8.2 to P1 after M1 and T7.6 to P2 when WS2 completes.

### 10.2 Four-person compressed split

| Person | Owns |
|---|---|
| **A — Lead + Orchestration + Integrations** | WS0, WS6 (T6.1–T6.7), T8.4 |
| **B — Documents + Evidence ★** | WS2 (P1 tasks only), WS4 |
| **C — Policy + Coverage + Settlement + Routing** | WS3, WS5 |
| **D — Data + UI + Eval + Demo** | WS1 (smaller corpus: 15 scenarios), WS7 (P1 only), WS8 (T8.1–T8.3, T8.7) |

Drop in this order if time runs out: T6.8 duplicates → T2.5/T2.6 images → T7.6 → T6.9 → T8.6 → T5.2 depth.

### 10.3 Dependency view (what can run in parallel)

```
Phase 0:   T0.1 ─┬─ T0.2 ─ T0.3 ─ T0.5
                 ├─ T0.4
                 └─ T1.1/T1.2 (first 5 scenarios) ─ T1.3 ─ T1.4 ─ T1.5

Parallel after M0 (all develop against fixtures):
   WS2: T2.1 → T2.2 → T2.3 → T2.4   (T2.5 → T2.6 side track)
   WS3: T3.1 → T3.2 → T3.3 → T3.4 → T3.5
   WS4: T4.1 → T4.2 → T4.3 → T4.4 → T4.5 → T4.6 → T4.7
   WS5: T5.3 → T5.4 → T5.5 ; T5.1 → T5.2 ; T5.6
   WS6: T6.5 → T6.6 ; T6.1 (stubs) → T6.2 → T6.3 ; T6.7 ; T6.4 ; T6.8
   WS7: T7.1 → T7.2 → T7.3/T7.4/T7.5
   WS8: T8.1 (metrics skeleton), T8.4, T8.3

M1 → M2 (wire real agents into the graph)  → M3 → M4 → M5
```

---

## 11. Interface Contracts Between Workstreams

These are the "plugs". Implement the signatures first (returning fixture data), then fill in logic. Changing a signature after M0 requires Tech-lead sign-off.

| Producer → Consumer | Function / object | Input | Output |
|---|---|---|---|
| WS2 → WS4 | `extract_facts(document_ids) ` | `DocumentRecord[]` | `Fact[]` (provenance validated) |
| WS2 → WS4/WS5 | `analyse_images(image_ids)` | image records | `ImageFinding[]` |
| WS3 → WS4/WS5 | `get_snapshot(policy_no, loss_date)` | policy number, date | `PolicySnapshot` or `NotInForce(reason)` |
| WS3 → WS4/WS5 | `retrieve_clauses(query, snapshot)` | text query | `ClauseCitation[]` (with `modified_by`) |
| WS4 → WS5 | `ClaimState.open_questions`, `contradictions`, `missing_evidence` | — | read-only inputs |
| WS5 → WS6 | `compute_settlement(inputs, rules_version)` | `SettlementInputs` | `SettlementResult` (trace) |
| WS5 → WS6 | `route(state) -> RoutingDecision` | state | tier + reason codes |
| WS6 → all | `Agent.run(state) -> StatePatch` | state | patch (owned fields only) |
| WS6 → WS5/WS8 | `ReconClient.submit_payment_instruction(req, idempotency_key)` | approved settlement | `ReconAck` |
| WS7 ↔ WS6 | REST API | JSON per OpenAPI | state / events |
| WS8 ← all | `audit.record(event)` | `AuditEvent` | append-only log |

**Fixture rule:** every producer ships a `fixtures/` JSON of its output for each scenario on or before M0+2 days, so consumers never block.

---

## 12. Synthetic Data & Scenario Catalogue

Start with these designed scenarios; extend each into variants to reach 30–50. All data is **synthetic**.

| ID | Scenario | What it tests | Expected system behaviour |
|---|---|---|---|
| S-01 | **Running example:** sudden burst claimed; plumber notes corrosion; surveyor sees multi-area moisture | Contradiction + missing start date + sublimit endorsement | Coverage unresolved; gap request to plumber; scenarios A/B/C; Medium/Complex tier |
| S-02 | Clean, consistent sudden burst, complete docs, within limits | Happy path | Issue matrix mostly established; Simple tier; calc trace; human approval still required for payment |
| S-03 | Policy lapsed 3 days before loss | Snapshot / status | `not_in_force`; no coverage reasoning; operator/human route |
| S-04 | Two endorsements with different effective dates; loss between them | Precedence + as-of date | Correct clause set; citation of both |
| S-05 | Endorsement missing from upload but referenced in schedule | Missing contract data | Request endorsement; abstain on affected clause |
| S-06 | Gradual leakage exclusion with an exception for "resulting damage" | Ambiguous exclusion | Abstain / escalate; show both readings |
| S-07 | Multiple causes of loss (burst + earlier roof leak) | Multi-cause | Separate questions per cause; partial establishment |
| S-08 | Contractor estimate 3× surveyor-implied extent | Amount reasonableness | Estimate-validation action; calc blocked/flagged |
| S-09 | Invoice with arithmetic error + missing tax line | Incomplete invoice | Table self-check flags; cannot calculate yet |
| S-10 | Same FNOL submitted via portal and call centre | Duplicate FNOL | Candidate flagged with matched fields; not auto-merged |
| S-11 | Same invoice number appears in two claims (one revised legitimately) | Duplicate invoice / legitimate near-match | Review needed; **not** labelled fraud |
| S-12 | Payment instruction retried five times | Payment idempotency | One Recon record only |
| S-13 | Blurry photos + one unrelated photo | Image quality | Quality verdict; request retake; unrelated → human |
| S-14 | S-01 + **late plumber report confirming prolonged leakage** | Incremental reassessment (scenario B) | Only affected questions re-run; exclusion becomes "potentially applicable — human review"; no auto-denial |
| S-15 | S-01 + late report confirming sudden failure | Scenario A | Condition moves to established; settlement drafted |
| S-16 | Limit nearly exhausted by prior payment | Calc cap | Payable capped; trace shows prior payment |
| S-17 | Loss date missing/ambiguous in FNOL | Intake gap | Registered; tracked requirement, not blocked |
| S-18 | High-value claim above front-line authority | Routing | Complex/Medium by exposure reason code |

Add: low-confidence OCR page, foreign-currency invoice, handwritten note, scanned rotated page, claimant-customer email changing story.

---

## 13. Evaluation Plan

Establish **baselines first**; set target numbers only after the first run (concept doc: "don't invent expected improvements").

| Metric | Computed by | Scenarios |
|---|---|---|
| Document-type accuracy; abstention on `unknown` | T8.1 vs labels | All |
| Field extraction (exact / normalised match) | T8.1 | All with fields |
| Citation correctness (evidence text is verbatim in cited page) | automated | All |
| Clause retrieval accuracy (including modifying endorsement) | T8.1 | S-04, S-05, S-06 |
| Contradiction precision/recall (by type) | T8.1 | S-01, S-07, S-08, S-14 |
| Missing-evidence recall + "minimal request" rating (human) | T8.1 + domain reviewer | S-01, S-05, S-09, S-17 |
| Issue-matrix status agreement | T8.1 | S-01…S-07 |
| Abstention appropriateness | T8.1 | S-05, S-06 |
| Calculation exact match | T8.1 vs independent calc | S-02, S-16 + generated cases |
| Routing tier agreement; override rate | T8.1 | S-01, S-02, S-18 |
| Duplicate P/R **per type** | T8.1 | S-10, S-11, S-12 |
| Image: abstention on poor/out-of-scope; zero cause/coverage claims | T8.1 | S-13 |
| Safety: forbidden-phrase and unauthorised-action tests | T8.4/T8.5 | All |
| Incremental efficiency: nodes skipped on new evidence | run logs | S-14, S-15 |

Report separate **human-vs-system** comparison for the minimum-evidence requests: show them to a claims-savvy reviewer blind to origin.

---

## 14. Demo Script

Story: Scenario S-01, then S-14, with S-10/S-11/S-13 as inserts. Every screen carries a **SYNTHETIC / MOCK** badge.

1. **Register** — drop FNOL; show detected type, extracted fields (page/region), one low-confidence value marked; create claim in mock Claims; consolidated checklist.
2. **Policy at date of loss** — mock PAS call (shown as MOCK); snapshot with base clause + endorsement effective dates; show clause citations.
3. **Read the pack** — Document-AI panel: plumber report, surveyor report, photos, estimate, invoice, each with type + cited facts; photo-quality verdict and visible-damage notes with the "does not certify cause" wording.
4. **Duplicate insert** — near-match invoice flagged "possible duplicate" with matched/different fields; reviewer marks "related but distinct".
5. **★ Evidence sufficiency** — gap board: one contradiction, one missing start date, one insufficient extent; minimum-evidence request to plumber; counterfactual A/B/C table.
6. **Coverage issue matrix** — citations; "final coverage conclusion: requires further investigation" (abstention shown deliberately).
7. **Routing** — tier + reason codes; configurable table visible.
8. **New evidence arrives (S-14)** — drop supplementary plumber report; run log shows *which steps re-ran and which were skipped*; state v1→v2 diff.
9. **Settlement** — reserve vs assessed vs payable; calculation trace; QA agent passes; human approves with reason.
10. **Recon** — mock instruction sent with idempotency key; retry button shows no second payment; matched/unmatched status; audit decision pack export.

Close with the metrics page: formulas ready, **baseline required** (no invented savings).

---

## 15. Guardrails (non-negotiables)

| # | Rule | Enforced by |
|---|---|---|
| 1 | LLM never performs arithmetic or sets the payable amount | Import lint on `settlement/`; explanation-vs-engine reconciliation test |
| 2 | No autonomous claim denial, adverse coverage decision, settlement approval or payment release | `autonomy_matrix.yaml` + guards (T8.4) |
| 3 | Duplicate/anomaly/red-flag ≠ fraud; never auto-merge/reject on similarity | Wording linter; UI label rules; QA agent |
| 4 | Extraction confidence is not coverage probability; never used alone as a routing gate | Routing feature whitelist; code review checklist |
| 5 | Every material fact and clause assertion carries provenance; verbatim evidence text | Substring validation (T2.4); QA agent |
| 6 | Agents write only owned state fields | `apply_patch` ownership check |
| 7 | Policy source of truth is the PAS snapshot for the date of loss; no stale cached summaries | Snapshot records source response + timestamp |
| 8 | Missing document creates a tracked requirement; it never silently blocks claim registration | Intake tests |
| 9 | Ask for all required documents together, justify extra requests | Min-evidence generator self-check |
| 10 | Mocks and synthetic data are always labelled | UI badges; response flag `"mock": true` |
| 11 | No invented thresholds or benefit percentages | `# SAMPLE` comments; benefits module refuses without baseline |
| 12 | Idempotent writes: retries never create a second claim, task or payment | T6.6 tests |

---

## 16. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Schema churn after M0 | Cross-team rework | Contract review at M0; Tech-lead approval for changes; schema-drift CI |
| Fixtures diverge from real agent output | Integration surprises at M2 | Fixtures generated from the same Pydantic models; contract tests; early M2 slice using real agents for S-01 only |
| LLM nondeterminism breaks demos/tests | Flaky demo | Record/replay cassettes; low temperature; structured outputs; offline demo mode |
| Policy wording too easy → unimpressive; too hard → unusable | Weak demo | Domain review of T1.1; mix of easy/medium/hard clauses |
| Vision scope creep | Time sink | One image task, bounded outputs; P2/P3 priority |
| Overbuilding integrations | Time sink | Mocks behind protocols; no live APIs until sandbox exists |
| Hero feature under-resourced | Weak differentiation | Put strongest engineer on WS4; protect their time |
| Hallucinated citations | Credibility loss | Verbatim-substring check; QA agent blocks release |
| Market claims overreach | Reviewer pushback | Position as integration-first decision-support layer; don't claim "AI claims processing" is unique (PDF §10) |
| Unverified statistics in the pitch | Credibility | Re-verify any market/regulatory figure from the concept doc before showing |

---

## 17. Open Decisions

Resolve these in Phase 0; record each as an ADR in `docs/adr/`.

| # | Decision | Default if unanswered |
|---|---|---|
| 1 | Agent framework (LangGraph vs Microsoft Agent Framework) | LangGraph |
| 2 | LLM provider and models (Azure OpenAI or other) | Whatever is available behind `core/llm.py`; one text+vision model |
| 3 | Local OCR vs Azure Document Intelligence | PyMuPDF + vision fallback; swap later |
| 4 | Vector store (pgvector / Chroma / FAISS / Azure AI Search) | Chroma or FAISS locally |
| 5 | Tracing tool (Langfuse / LangSmith / App Insights) | Pick one; plus own audit table |
| 6 | Does "Recon" = payment/GL reconciliation? | Yes, mock; assumption shown on screen |
| 7 | Settlement step order for water-damage product | Per `settlement_v1.yaml` (SAMPLE) |
| 8 | Real approval thresholds | Sample values, labelled |
| 9 | Jurisdiction/regulator for compliance copy | India (IRDAI) as reference; confirm |
| 10 | Demo audience time limit | Prepare a 5-min and a 12-min cut |

---

## 18. Extension Roadmap (after the prototype)

| Horizon | Enhancement | Where |
|---|---|---|
| Next | Second claim type in same LOB (e.g., fire, theft) | New question templates + pack variants |
| Next | Real PAS/Claims/Recon sandbox integration | `integrations/live/` |
| Next | Azure AI Document Intelligence + AI Search backends | Backend swap behind interfaces |
| Next | Document/image-hash duplicate detection (needs labelled data and retention design) | `duplicates/` |
| Later | Motor and health LOB packs | `packs/` |
| Later | Neo4j for multi-hop clause/evidence reasoning | `policy/relations.py` backend |
| Later | Service Bus events, multi-tenant RBAC, data residency controls | `orchestration/events.py`, `core/config.py` |
| Later | Fairness monitoring by channel/location/claim type; drift monitoring; model release gates | `evals/`, `audit/` |
| Later | Customer-facing status messages (approved drafts) | `actions/drafts.py` + approval flow |

---

## 19. Appendix: Code Skeletons

Illustrative only — confirm API names against the **pinned** framework version in T0.1.

### A.1 Agent protocol with ownership enforcement

```python
# src/claimtower/agents/base.py
from typing import Protocol, ClassVar
from pydantic import BaseModel
from claimtower.core.schemas import ClaimState

class StatePatch(BaseModel):
    agent_id: str
    updates: dict          # {field_name: new_value}; only owned fields allowed
    rationale: str = ""

class Agent(Protocol):
    agent_id: ClassVar[str]
    owns: ClassVar[frozenset[str]]       # e.g. frozenset({"open_questions","contradictions", ...})
    def run(self, state: ClaimState) -> StatePatch: ...

# src/claimtower/core/state.py
class OwnershipViolation(Exception): ...

def apply_patch(state: ClaimState, patch: StatePatch, owns: frozenset[str]) -> ClaimState:
    illegal = set(patch.updates) - owns
    if illegal:
        raise OwnershipViolation(f"{patch.agent_id} tried to write {sorted(illegal)}")
    new = state.model_copy(update={**patch.updates, "state_version": state.state_version + 1})
    # persist version + audit event here
    return new
```

### A.2 Graph wiring (shape of the LangGraph definition)

```python
# src/claimtower/orchestration/graph.py  (illustrative)
from langgraph.graph import StateGraph, END
from claimtower.core.schemas import ClaimState

def build_graph(agents, supervisor, checkpointer):
    g = StateGraph(ClaimState)
    for name, agent in agents.items():
        g.add_node(name, supervisor.wrap(agent))      # wrap = ownership check + audit + skip-if-unchanged

    g.set_entry_point("supervisor")
    g.add_node("supervisor", supervisor.decide)       # chooses next node(s) from state + dependency map
    g.add_conditional_edges("supervisor", supervisor.route, {
        **{n: n for n in agents}, "human_gate": "human_gate", "done": END,
    })
    for name in agents:
        g.add_edge(name, "supervisor")                # every specialist returns to the supervisor
    g.add_node("human_gate", supervisor.human_gate)   # uses the framework's interrupt/resume
    g.add_edge("human_gate", "supervisor")
    return g.compile(checkpointer=checkpointer)
```

### A.3 Deterministic settlement engine and rule file

```yaml
# packs/property_water_damage/calculation/settlement_v1.yaml   # SAMPLE — not insurer policy
version: "sample-1.0"
currency: INR
rounding: {mode: half_up, places: 0}
steps:
  - id: start
    op: set
    from: assessed_covered_loss
  - id: less_non_covered
    op: subtract
    from: non_covered_items_total
  - id: less_deductible
    op: subtract
    from: applicable_deductible
  - id: apply_sublimit        # water-damage sublimit from endorsement, if present
    op: cap
    from: water_damage_sublimit
    optional: true
  - id: cap_remaining_limit
    op: cap
    from: remaining_policy_limit
  - id: less_prior_payments
    op: subtract
    from: prior_payments_total
  - id: floor_zero
    op: floor
    value: 0
```

```python
# src/claimtower/settlement/engine.py  — no LLM imports allowed in this package
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass

@dataclass
class Step: rule_id: str; before: Decimal; after: Decimal; input_ref: str | None

def run(rules: dict, inputs: dict[str, Decimal]) -> dict:
    missing = [s["from"] for s in rules["steps"]
               if s.get("from") and s["from"] not in inputs and not s.get("optional")]
    if missing:
        return {"status": "cannot_calculate", "missing_inputs": sorted(set(missing))}

    amount, trace = Decimal("0"), []
    for s in rules["steps"]:
        before, op, ref = amount, s["op"], s.get("from")
        val = inputs.get(ref) if ref else None
        if op == "set":        amount = val
        elif op == "subtract": amount = amount - val
        elif op == "cap":
            if val is not None: amount = min(amount, val)
        elif op == "floor":    amount = max(amount, Decimal(str(s["value"])))
        trace.append(Step(s["id"], before, amount, ref))

    q = Decimal(1).scaleb(-rules["rounding"]["places"])
    return {"status": "calculated", "payable": amount.quantize(q, ROUND_HALF_UP), "steps": trace,
            "rule_version": rules["version"]}
```

*Worked check (PDF §6.3):* 200,000 − 15,000 − 10,000 = 175,000 → capped at remaining limit 150,000 → **payable 150,000**. This must be a unit test, together with the concept-doc ordering variant provided as a second YAML version.

### A.4 Autonomy matrix (config, not code)

```yaml
# configs/autonomy_matrix.yaml
operations:
  extract_document_info:        {mode: autonomous}
  retrieve_policy_clauses:      {mode: autonomous}
  identify_missing_evidence:    {mode: autonomous}
  draft_information_request:    {mode: draft_only}
  create_investigation_task:    {mode: controlled_api}
  recommend_coverage:           {mode: human_reviewed}
  recommend_settlement:         {mode: human_reviewed}
  deny_claim:                   {mode: authorised_human_only, roles: [senior_adjuster]}
  release_payment:              {mode: authorised_human_only, roles: [supervisor, finance]}
```

### A.5 Routing rules (SAMPLE values; real values from Claims Operations)

```yaml
# configs/routing_rules.yaml   # SAMPLE — not insurer policy
rules_version: "sample-1"
mandatory_referral:
  - when: {any_of: [duplicate_status: multiple_linked, policy_match: mismatch, specialist_required: true]}
    tier: complex
tiers:
  simple:
    all_of: [policy_certainty: exact, evidence_completeness: complete,
             contradiction_severity: none, duplicate_status: clear_or_benign,
             exposure_inr: {lte: 100000}]          # SAMPLE
  medium:
    any_of: [evidence_completeness: minor_gap, contradiction_severity: low,
             duplicate_status: needs_confirmation, exposure_inr: {gt: 100000, lte: 500000}]  # SAMPLE
  complex:
    default: true
# NOTE: extraction confidence is deliberately NOT a routing feature.
```

---

*End of plan. Suggested first actions: (1) hold a 60-minute contracts review for T0.2, (2) assign owners per §10, (3) have P6 and P4 co-write the first five scenarios and their ground truth, (4) run the framework spike (T0.1) in parallel.*
