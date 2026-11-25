# HOW TO CONFIGURE AZURE OPENAI (instead of OpenAI)

Status as of 2026-11-25. **Not yet tested against Azure**: everything so far was run on OpenAI (gpt-5-mini). The code has an `azure_openai` provider, so this is configuration, not new code,
but run the checks at the end before relying on it. Only `src/claimtower/core/llm.py` imports the SDK; no agent changes.

## What you need in Azure
- An Azure OpenAI resource, with its **endpoint** (`https://<resource>.openai.azure.com/`) and a **key**.
- Three **deployments** (Azure routes by deployment name, not by model id):
  1. a chat deployment for text, a reasoning model such as `gpt-5-mini` (the code sends `reasoning_effort` and `max_completion_tokens`);
  2. a vision-capable chat deployment (the same one works if it accepts images), used for OCR of scans and photos;
  3. an embeddings deployment such as `text-embedding-3-small` (it must support the `dimensions` parameter, which the policy retrieval uses).
- An **API version** your deployments support. The code defaults to `2024-10-21`, which may be too old for reasoning-model parameters: set a newer one explicitly.

## Step 1: `.env` (never commit it)
```
LLM_API_KEY=<your Azure OpenAI key>            # the same variable as for OpenAI
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=<a version your deployments support>
```
`LLM_BASE_URL` is not used for Azure.

## Step 2: `configs/app.yaml`
```yaml
llm: {provider: azure_openai, text_model: <chat deployment name>, vision_model: <vision deployment name>,
      embedding_model: <embeddings deployment name>, embedding_dimensions: 256, temperature: null, reasoning_effort: minimal}
```
`temperature: null` stays (reasoning models accept only their default). To go back to OpenAI, set `provider: openai` and the model ids again.

## Step 3: only if your chat deployment is NOT a reasoning model (for example gpt-4o)
Azure rejects `reasoning_effort` for such models. Set it to `null` in `app.yaml` and in every place that overrides it (eight places in all):
- `configs/app.yaml` (`llm.reasoning_effort`);
- `packs/property_water_damage/policy_structure.yaml` (`llm_typing`);
- `packs/property_water_damage/question_templates.yaml`: `model_questions`, `fact_links`, `contradictions`, `sufficiency`, `requests`, `counterfactuals`.

A non-reasoning model may also want a `temperature` (for example `0`); set `temperature` in `app.yaml` then.

## Step 4: check it works (one live call each)
```
python - <<'E'
import sys; sys.path.insert(0, "src")
from dotenv import load_dotenv; load_dotenv()
from claimtower.core.config import load_settings
from claimtower.core.llm import LLMClient
llm = LLMClient.from_settings(load_settings())
print(llm.chat("Reply with the single word: ready"))
print(len(llm.embed(["test"])[0]), "embedding dimensions")
E
python evals/runners/live_llm_check.py --only E1      # the document classifier tie-breaker on synthetic documents; E3 checks vision OCR, E4 the facts
```
A wrong deployment name, version or key shows up as an `LLMError` with Azure's message.

## Step 5: run the app with the model
```
python scripts/run_api.py --llm            # the Gap agent uses the model; or: python scripts/run_ui.py --start-api --llm
```
Today only the Gap agent uses the model in the API (Evidence and Policy run on rules; see `TEAM_HEADSUP.md`).

## Things to know
- **Recorded tests will miss.** Replay cassettes are keyed by the model name. After you change `text_model` to a deployment name, the replay-based eval tests (about ten files, for example `test_question_eval`, `test_sufficiency_eval`, `test_gap_agent_eval`) fail with a cassette miss until the cassettes are re-recorded (costs model calls) or the tests pin the old model name (a small test change). The running application is not affected.
- **Quotas and filters.** Rate-limit errors are retried (up to 3 times, with backoff); the timeout is 120 s per call. Azure's content filter can block a response: that surfaces as an `LLMError`, and the agent falls back to its rules or abstains.
- **Privacy.** Documents and facts are sent to the model. Everything today is SYNTHETIC. Before real claims, settle data residency and the privacy decision (see `docs/adr/0002-azure-document-intelligence.md` and `NEW_INSURANCE_CHALLENGES.md`).
- **Cost and latency.** The Gap agent makes about 10-15 sequential calls per claim; choose the deployment's capacity accordingly.

See also: `TEAM_HEADSUP.md` (what uses an LLM today), `.env.example`, `docs/api.md`.
