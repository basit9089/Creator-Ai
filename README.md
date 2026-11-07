# Creator AI

Local-first content creation platform built on **FastAPI** and **Ollama** (Qwen3).

This repository contains the Phase 3 foundation: a provider-agnostic AI
interface, a reusable agent architecture every future agent will build on,
and a React frontend skeleton. The agents themselves (Researcher, Fact
Checker, Scriptwriter, …) and the dashboard arrive in later phases.

## Architecture

```
React / Vite  (frontend, :5173)
        │  HTTP + CORS
        ▼
FastAPI       (backend, :8000)
        │
        ▼
AgentRunner   (app/agents/runner.py)      ← execution, timing, failure
        │
        ▼
Agent         (app/agents/base.py)        ← Researcher, Fact Checker, …
        │
        ▼
AIProvider    (app/services/ai/provider.py)   ← abstraction
        │
        ▼
Ollama        (:11434)  →  Qwen3 8B
```

Agents depend on `AIProvider`, never on `OllamaProvider`. Substituting a
cloud backend means adding a provider implementation and changing
`OLLAMA_HOST` handling — not editing agent code. The agent layer does not
even import the `ollama` SDK, which a test asserts.

Later, without touching any agent:

```
AIProvider
   ├── OllamaProvider      (today)
   ├── OpenAIProvider      (later)
   └── AnthropicProvider   (later)
```

## Prerequisites

- Python 3.14+
- Node.js 20+
- [Ollama](https://ollama.com) with the configured model pulled:
  ```bash
  ollama pull qwen3:8b
  ```

## Getting started

### Backend

```bash
cd backend

# Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements-dev.txt

# Create local configuration
copy .env.example .env            # Windows
# cp .env.example .env            # macOS / Linux

# Start the development server
python main.py
```

The API is then available at <http://127.0.0.1:8000>, with interactive
docs at <http://127.0.0.1:8000/docs>.

For production-style startup, run the ASGI app directly:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at <http://localhost:5173> and talks to the backend on
port 8000. Point it elsewhere with `VITE_API_BASE_URL` in `.env.local`
(see `.env.example`).

## API endpoints

| Method | Path                | Purpose                                       |
| ------ | ------------------- | --------------------------------------------- |
| GET    | `/`                 | Service identification                        |
| GET    | `/health`           | Liveness probe (no external calls)            |
| GET    | `/api/health/ai`    | Provider reachability and model availability  |
| POST   | `/api/ai/generate`  | Generate a completion                         |
| POST   | `/api/ai/generate/stream` | Stream a completion (NDJSON chunks)     |
| GET    | `/api/agents`       | Registered agents, versions and availability  |

Agent *execution* endpoints are deliberately absent: the agents are
registered contracts, not working logic yet.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/ai/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Reply with exactly: PONG"}'
```

```json
{
  "success": true,
  "model": "qwen3:8b",
  "response": "PONG",
  "duration_ms": 926.8,
  "prompt_tokens": 14,
  "completion_tokens": 42,
  "thinking": false
}
```

`thinking` reports the reasoning mode actually used after defaults were
applied, so a caller can tell what actually happened rather than guessing.

### Error handling

Failures return an appropriate HTTP status with a consistent envelope.
Server failures are no longer reported as HTTP 200.

```json
{
  "success": false,
  "error": {
    "code": "ai_provider_unavailable",
    "message": "The AI provider is currently unavailable."
  }
}
```

| Status | Meaning                                              |
| ------ | ---------------------------------------------------- |
| 422    | Request validation failed (field details included)   |
| 502    | Provider returned an error or unusable output        |
| 503    | Provider unreachable, or the model is not installed  |
| 504    | Provider timed out                                   |

## Configuration

Backend settings come from environment variables, optionally supplied via
a local `backend/.env`. Defaults preserve the original behaviour.

| Variable            | Default                                  | Purpose                       |
| ------------------- | ---------------------------------------- | ----------------------------- |
| `APP_NAME`          | `Creator AI API`                         | Service name                  |
| `APP_ENV`           | `development`                            | `development`/`test`/`production` |
| `APP_HOST`          | `127.0.0.1`                              | Dev server bind address       |
| `APP_PORT`          | `8000`                                   | Dev server port               |
| `APP_RELOAD`        | `true`                                   | Auto-reload in development    |
| `LOG_LEVEL`         | `INFO`                                   | Log level                     |
| `LOG_JSON`          | `false`                                  | JSON log output               |
| `OLLAMA_HOST`       | `http://127.0.0.1:11434`                 | Ollama server                 |
| `OLLAMA_MODEL`      | `qwen3:8b`                               | Default model                 |
| `OLLAMA_TIMEOUT`    | `300`                                    | Generation timeout (seconds)  |
| `FRONTEND_URL`      | `http://localhost:5173,http://127.0.0.1:5173` | Allowed CORS origins   |
| `CORS_ALLOW_METHODS`| `GET,POST,PUT,PATCH,DELETE,OPTIONS`      | Allowed methods               |
| `CORS_ALLOW_HEADERS`| `*`                                      | Allowed headers               |
| `MAX_PROMPT_LENGTH` | `8000`                                   | Maximum prompt size (chars)   |
| `DEFAULT_AI_TEMPERATURE` | `0.7`                              | Default sampling temperature  |
| `DEFAULT_AI_TOP_P` | `0.9`                                     | Default nucleus sampling      |
| `DEFAULT_AI_THINKING` | `false`                               | **Default reasoning mode**    |
| `MAX_AI_OUTPUT_TOKENS` | `2048`                               | Default output cap            |
| `AI_REQUEST_TIMEOUT` | `600`                                  | Agent execution limit (s)     |

Frontend variables (must be `VITE_`-prefixed, and must never hold secrets):

| Variable               | Default                 |
| ---------------------- | ----------------------- |
| `VITE_API_BASE_URL`    | `http://127.0.0.1:8000` |
| `VITE_API_TIMEOUT_MS`  | `300000`                |

Server-side values such as the Ollama host are never sent to the browser.

## Tests

```bash
cd backend

# Unit tests — no network or Ollama required
pytest -m "not integration"

# Integration tests against a real local Ollama server
RUN_OLLAMA_INTEGRATION=1 pytest -m integration
```

The unit suite mocks the provider, including for agent execution, so the
agent architecture is tested without loading a model. Integration tests
cover the real provider, the thinking default and structured output, and
stay opt-in.

## Agents

### Contracts

| Type             | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `Agent`          | One versioned, independently retryable unit of work   |
| `AgentContext`   | Execution id, project/channel ids, config, prior outputs |
| `AgentOutput`    | What an agent returns: `success`, `data`, `error`     |
| `AgentResult`    | The runner's record: status, timing, failure reason   |
| `AgentStatus`    | `pending`/`running`/`completed`/`failed`/`cancelled` |

`AgentOutput` and `AgentResult` are separate on purpose: an agent never
fabricates a duration or a status.

### Registry

`GET /api/agents` lists the six planned agents in pipeline order. They are
registered as placeholders (`not_implemented`), so names resolve and
lookup works before the logic exists.

```json
{
  "success": true,
  "total": 6,
  "agents": [
    {
      "name": "researcher",
      "version": "1.0.0",
      "status": "not_implemented",
      "implemented": false,
      "capabilities": []
    }
  ]
}
```

Selecting an agent by name rather than by class is what lets each one be
implemented and versioned independently.

### Runner

`AgentRunner` assigns the execution id, times the run, captures
exceptions, and returns an `AgentResult`. It never raises for an agent
failure — a caller recording pipeline progress reads statuses instead of
catching exceptions.

Retry is per agent. Each execution is retained with the context and prompt
it ran under, so a failed stage can be re-run alone:

```python
result = await runner.run_named("researcher", context)
failed = await runner.run_named("scriptwriter", context)
await runner.retry(failed.execution_id)   # only the scriptwriter re-runs
```

There is no automatic retry and no queue. Both can be added later without
changing an agent, because the runner is the only thing that owns
execution.

### Structured output

`generate_structured(messages, Schema)` asks for JSON, validates it, and
returns a Pydantic model, so agents never parse prose themselves:

```python
answer = provider.generate_structured(messages, ResearchResult)
answer.key_points
```

The implementation is provider-independent: it appends the JSON Schema,
parses, and validates. A provider with native schema enforcement can
override the method without agents noticing.

### Prompts

`app/prompts/` holds `PromptTemplate`, `PromptBuilder` and a small
`PromptLibrary`. Agents describe *what* they want instead of assembling
message dicts or embedding long strings in Python:

```python
prompt = (
    PromptBuilder()
    .with_system_template(library.get("fact_checker_grounding"))
    .with_context(source_material)
    .with_reasoning(True)          # opt in per call
    .with_user("Verify each claim.")
)
messages = prompt.build()
```

## Notes on model behaviour

Qwen3 enables reasoning on its own, and on this hardware (i5-10210U,
16 GB, integrated graphics) that costs roughly **60 s versus 1–4 s** for
the same prompt. Reasoning is therefore **off by default**
(`DEFAULT_AI_THINKING=false`) and the provider states it explicitly rather
than deferring to the model.

Any call can override it — per agent, per request, without touching
configuration:

```python
GenerationOptions(think=True)     # reasoning, when it earns its cost
```

Measured on this machine: default `think=false` 1.3 s, explicit
`think=true` 60.8 s.

Agent execution is sequential by design. Two agents generating at once
would contend for one CPU and one model, making both slower.

## Project layout

```
source/
├── backend/
│   ├── app/
│   │   ├── core/            # configuration, logging
│   │   ├── api/routes/      # HTTP layer
│   │   ├── schemas/         # request/response models
│   │   ├── services/ai/     # AIProvider + OllamaProvider
│   │   ├── agents/          # base, runner, registry
│   │   ├── prompts/         # templates, builder, library
│   │   ├── exceptions/      # error handling
│   │   ├── dependencies.py  # provider injection
│   │   └── main.py          # application factory
│   ├── services/            # legacy import path (compatibility shim)
│   ├── tests/
│   ├── main.py              # development entry point
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── api/             # typed API client
│       ├── components/
│       ├── features/        # one directory per feature
│       ├── hooks/
│       ├── pages/
│       └── types/
└── .gitignore
```

## Not built yet

Deliberately absent, by phase: the six agents, web search, video
rendering, thumbnails, YouTube integration, OAuth, a database, Redis,
Docker, authentication and cloud AI providers. The architecture leaves
room for each without any of them being stubbed in prematurely.