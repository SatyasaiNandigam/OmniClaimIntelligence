# NEW INSURANCE CHALLENGES: adding another line of business (for example medical insurance) and scaling the application

Status as of 2026-11-25. Prototype, SYNTHETIC data only. Today there is one pack: `packs/property_water_damage`.

## Can we add a medical pack?

**Short answer: yes, it is feasible, but it isn't just a new folder of YAML.** The workflow shape (documents, facts, questions, policy clauses, settlement, human gate) carries over.
The data, a few frozen contract fields and the runtime need work.

### Feasible as configuration
A pack folder already drives these:
- document types and extraction fields (discharge summary, hospital bill, prescriptions, lab reports);
- fact templates and the open questions with their criteria (for example "was the hospitalisation medically necessary?" or "does the waiting period apply?");
- date and amount contradiction rules (admission and discharge dates against bill dates, claimed against assessed bill);
- sufficiency table, request roles, counterfactual framing, policy-wording patterns, prompts, and the settlement formula steps.

### Needs code or contract changes
- **One pack at a time.** `active_pack` is global, and the graph, the API and the agents are built for one pack. Serving property and medical together needs a registry per pack and a pack chosen per claim (from the header's line of business).
- **Frozen contract enums are property-shaped.**
  - There are 7 clause types. Co-pay, waiting period and room-rent cap don't map cleanly to them.
  - Question categories are fixed, and contradiction kinds are fixed.
  - `SettlementInputs` has `water_damage_sublimit` hard-coded.
  - The settlement engine has only four operations (set, subtract, cap, floor). Medical co-pay is a percentage, so it needs a new one.

  This is a deliberate change to the frozen contract with re-approval, not a pack file.
- **Global routing rules.** The routing config isn't per pack yet. The routing agent (T5.6) isn't built, so build it pack-aware.
- **Prompts and examples** (pipes, corrosion) need rewriting per pack.

### The real cost
- **Data and labels.** A medical pack needs its own synthetic documents, labelled ground truth, policy corpus, mock PAS records and eval sets. Our results were only credible because of labelled dev and unseen sets, and each pack needs the same discipline. Most of the effort is here, not in code.
- **Medical-specific risks.**
  - **PHI:** health data is sensitive, and today documents and facts go to OpenAI. You'd need a privacy and compliance decision before touching real data.
  - **Clinical judgement:** this is a human task, so abstention and review matter more.
  - **Cashless vs reimbursement:** the flows differ and may need new graph branches.

## How scalable is it today

It is a **prototype, not a scalable service**:
- **Memory:** claims, checkpoints and events are in memory and are lost on restart.
- **Concurrency:** there is one global run lock, so one claim is processed at a time. Agents hold per-run attributes that aren't safe to share across threads.
- **Latency:** requests are synchronous, and the Gap agent makes about 10-15 sequential model calls per claim with the model on.
- **No auth:** the API has no authentication. The document store is local files and the policy store is SQLite.

Scaling needs:
- a persistent checkpointer (Postgres);
- stateless agents with per-claim locks;
- background workers and a queue;
- object storage;
- auth and RBAC;
- cost and latency controls.

The architecture (agents with owned fields, a versioned state, pack-driven config) supports that path without a redesign.

See also: `TEAM_HEADSUP.md` (what uses an LLM today and how to test the API), `docs/orchestration.md`, `docs/api.md`, `docs/assumptions.md`.
