# agent-starter

A small, readable LangGraph agent template: **guardrails → plan → retrieve (RAG) → act (tools) → answer**, served over FastAPI (SSE + n8n webhook), with evals, Docker and CI. It runs **fully offline** with a fake LLM and local embeddings, and switches to **AWS Bedrock** (Claude + Titan v2) through `.env`.

```bash
make setup                 # uv sync + .env from .env.example
make demo OFFLINE=1        # end-to-end without credentials
make lint test             # ruff + mypy --strict + pytest (offline)
make eval OFFLINE=1        # hit-rate / faithfulness table
make dev                   # API on :8000 (uses .env → Bedrock by default)
make smoke                 # docker compose API + n8n end to end, offline (see below)
```

## Architecture (60 seconds)

```mermaid
flowchart LR
    U([User / n8n]) -->|POST /chat SSE<br/>POST /webhook/n8n| API[FastAPI]
    API --> GI{guard_input}
    GI -->|blocked| END([refusal])
    GI --> P[plan<br/>LLM]
    P --> R[retrieve<br/>FAISS top-k]
    R --> A[act<br/>LLM + tools]
    A -->|tool calls, steps < MAX_STEPS| A
    A --> ANS[answer<br/>LLM, cites S1..Sn]
    ANS --> GO{guard_output}
    GO --> OUT([answer + sources])

    A -.-> T1[(http_get_json)]
    A -.-> T2[(create_ticket → n8n webhook)]
    R -.-> IDX[(.index/ FAISS + chunks.json)]
    CP[(InMemorySaver<br/>thread_id)] -.- GI
```

| File | What it is |
|---|---|
| `src/agent_starter/graph/build.py` | The whole graph wiring, ~50 lines. Start here. |
| `graph/nodes.py` | One async method per node. Each returns a partial state update. |
| `graph/state.py` | Pydantic `AgentState`. `messages` persist per `thread_id`; the other fields reset every turn. |
| `graph/guardrails.py` | Pure-function input checks (injection, length, secrets) and output checks (citations, secret leak). |
| `prompts.py` | **Every prompt in one file.** |
| `providers.py` | `get_chat_model()` / `get_embeddings()` from env, plus the offline fakes. |
| `rag/ingest.py`, `rag/store.py` | `./data` (md/txt/pdf) → recursive chunks → FAISS. About 60 lines, no pickle. |
| `tools/` | Registry. **One file = one tool** (auto-discovered). |
| `api/main.py` | `/chat` (SSE: `node`, `token`, `final`), `/webhook/n8n`, `/health`, request-id middleware. |
| `observability.py` | JSON logs with `request_id`/`thread_id`. Langfuse callback turns on only when keys are set. |
| `evals/run.py` + `eval/evalset.yaml` | hit@k, lexical faithfulness, keyword correctness. |

**Providers.** Set `LLM_PROVIDER` to `bedrock` (default, `ChatBedrockConverse`, Claude), `anthropic`, `openai` or `fake`. Set `EMBEDDINGS_PROVIDER` to `bedrock` (Titan v2), `openai` or `local`. The `fake` and `local` options are deterministic and offline: the fake LLM copies the best-matching sentence and cites it. Tests and CI use only these. After switching embeddings providers, re-run `make ingest`.

## Adapting to a new brief in 15 minutes

1. **(2 min) Data.** Replace `data/*` with the brief's documents and run `make ingest OFFLINE=1` to check that chunking works.
2. **(3 min) Prompts.** Edit `prompts.py`: set `DOMAIN`, plus the tone and refusal rules in `ANSWER_SYSTEM`.
3. **(4 min) Tools.** Copy `tools/ticket.py` to `tools/<action>.py`. Change the args model, the function body and the description. Delete any example tools you don't need.
4. **(2 min) Guardrails.** Add domain rules to `guardrails.py` (denied topics, PII regexes, length).
5. **(2 min) Eval.** Rewrite the 5 cases in `eval/evalset.yaml` for the brief, then run `make eval OFFLINE=1`.
6. **(2 min) Real model.** Set `.env` to `LLM_PROVIDER=bedrock`, plus `AWS_PROFILE` and `AWS_REGION`. Run `make demo`, then `make dev`.
7. Only if needed: change the graph shape in `build.py`, for example a human-approval node before risky tools or a router node.

Keep `make test` green as you go. The e2e tests in `tests/test_graph_e2e.py` mirror the demo flows.

## API

```bash
curl -N localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"How many days to request a refund?","thread_id":"demo"}'
# event: node   data: {"node":"guard_input"} ... event: token ... event: final {answer, sources, guardrail_notes}

curl localhost:8000/webhook/n8n -H 'content-type: application/json' -d '{"message":"open a ticket, parcel damaged"}'
```

**n8n.** Run `docker compose up --build` (add `-f docker-compose.yml -f docker-compose.offline.yml` to use the fake LLM and local embeddings without credentials). Then import and publish the workflow; n8n only registers webhooks for published workflows:

```bash
docker compose exec n8n n8n import:workflow --input=/workflows/agent-workflow.json
docker compose exec n8n n8n publish:workflow --id=agentStarterAsk1   # n8n >= 2.x
```

It sets up two flows:
- `POST :5678/webhook/ask` → calls the agent's `/webhook/n8n` → returns the answer.
- `POST :5678/webhook/ticket` receives the agent's `create_ticket` tool calls and returns a ticket id. Replace this with Jira, Slack or email nodes.

If `N8N_WEBHOOK_TOKEN` is set (shell or `.env`), compose passes the same value to both containers: n8n sends it as `X-Webhook-Token` and the API requires it.

**Smoke test.** `make smoke` (or `scripts/smoke.sh`) does all of the above with no credentials: it brings up API + n8n with `docker-compose.offline.yml` (n8n pinned to `N8N_VERSION=2.47.4` by default), imports and publishes the workflow, then checks that (1) `POST :5678/webhook/ask` reaches the API's `/webhook/n8n` and returns 200 with a cited answer, (2) a "open a ticket" request makes the agent call `create_ticket`, which reaches n8n's `/webhook/ticket` and gets back an `N8N-…` id, and (3) the API rejects `/webhook/n8n` without the token. It uses host ports 8000 and 5678, tears everything down (volumes included) on exit, and leaves the stack running with `KEEP_UP=1`. CI runs it as the `smoke` job.

## 20-minute demo script

| Min | Section | What to show |
|---|---|---|
| 0–3 | **Problem** | Restate the brief: who the user is, what the job is, and what counts as "done". Name the risks: hallucination, unsafe actions, cost. |
| 3–7 | **Architecture** | The Mermaid diagram and `build.py`. Typed state; explicit nodes over a prebuilt ReAct agent; guardrails as nodes; max-steps stop; checkpointer memory per `thread_id`. |
| 7–13 | **Live run** | `make demo`: (1) a RAG answer with a citation, (2) the ticket path calling a tool through n8n, (3) a prompt injection that gets blocked, (4) a follow-up in the same thread that uses memory. Then `curl -N /chat` to show SSE events. Optionally show the trace in Langfuse. |
| 13–16 | **Quality** | `make eval` table, `make test` (all offline), CI. Explain what the faithfulness metric does and does not measure. |
| 16–20 | **Trade-offs & production on AWS** | See below. |

**Trade-offs taken for the prototype**
- Lexical faithfulness metric instead of an LLM-as-judge: deterministic, free and offline, but blind to paraphrase. Production would add an LLM judge (or RAGAS) on a larger golden set.
- In-process FAISS and `InMemorySaver`: no infrastructure, but single-instance, and memory is lost on restart.
- A separate `plan` LLM call adds latency. On simple briefs, merge it into `act`.
- Regex guardrails are cheap but easy to bypass. They are a first layer, not the whole defence.

**What I'd do for production on AWS**
- **Compute.** Run the API on ECS Fargate (SSE and long requests) behind an ALB. Use Lambda for the n8n or async event paths. Build ingest as a separate Step Functions or Lambda job triggered by S3 uploads.
- **Models.** Bedrock with cross-region inference profiles, Bedrock Guardrails (PII, denied topics, contextual grounding check) on input and output, and model fallback in the provider factory.
- **Data.** OpenSearch Serverless, pgvector on Aurora, or Bedrock Knowledge Bases instead of FAISS. A DynamoDB or Postgres LangGraph checkpointer instead of the in-memory one.
- **Secrets & IAM.** Secrets Manager for webhook tokens and API keys. A task role with least-privilege `bedrock:InvokeModel` on specific model ARNs. No keys in env files.
- **Observability.** OpenTelemetry (ADOT collector) to X-Ray and CloudWatch, Langfuse for LLM traces and cost. Alarms on p95 latency, guardrail block rate and tool error rate.
- **Evaluation.** Golden set in CI as a regression gate, online sampling with an LLM judge, and human feedback captured per `thread_id`.
- **Safety.** A human-in-the-loop interrupt (`langgraph` `interrupt()`) before irreversible tools, idempotency keys on ticket creation, and rate limiting per tenant.

## Development

- Python 3.12, uv, `src/` layout. `make lint` runs ruff and `mypy --strict` on `src`. `make test` runs pytest with pytest-asyncio and respx (HTTP mocked).
- Never commit secrets: only `.env.example` is tracked.
- Sample data in `data/` is a fictional company ("Northwind Logistics"). The corpus is tiny, so hit@k is an easy bar here. Re-baseline it on the real brief's data.
