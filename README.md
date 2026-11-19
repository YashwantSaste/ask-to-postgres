# ask-postgres: ask questions about any PostgreSQL database, from any AI agent

ask-postgres connects an AI agent to a PostgreSQL database so it can answer questions such as *"Which category
grew fastest last quarter?"* using the real data. It works with **any schema**: it discovers tables,
columns, keys and comments at runtime, so there is nothing to configure per database.

```
  Any AI agent                         ask-postgres                                PostgreSQL
 ┌──────────────────────┐   MCP     ┌──────────────────────────┐  read-only  ┌──────────┐
 │ Claude Desktop/Code  │──────────▶│ schema discovery tools   │────────────▶│          │
 │ Cursor, VS Code, ... │           │ SQL guard (SELECT only)  │  txn +      │  your    │
 ├──────────────────────┤   HTTP    │ row cap, timeout         │  timeout    │  data    │
 │ OpenAI / LangChain / │──────────▶│                          │             │          │
 │ n8n / custom app     │           │ built-in agent (any LLM) │             │          │
 └──────────────────────┘           └──────────────────────────┘             └──────────┘
```

## The tools an agent gets

| Tool | Purpose |
|---|---|
| `get_database_overview` | Database, version, limits, every schema and table with row estimates and comments |
| `list_tables` | Tables, views and materialized views, optionally filtered to one schema |
| `describe_table` | Columns, types, nullability, PK/FK/unique/check constraints, referencing tables, indexes |
| `search_schema` | Find tables and columns by keyword (matches names and comments) |
| `sample_rows` | A few example rows, to see value formats |
| `column_values` | Most frequent distinct values of a column, for exact filter spellings |
| `run_query` | Run one read-only `SELECT` and get the result as JSON |
| `explain_query` | Show the execution plan without running the query |

## Safety: three layers

1. **SQL guard** ([sql_guard.py](src/ask_postgres/sql_guard.py)): parses the SQL and accepts only a single
   `SELECT`/`WITH`/`UNION` query. It rejects DML, DDL, `SELECT INTO`, data-modifying CTEs, `FOR UPDATE`,
   and dangerous functions (`pg_sleep`, `pg_read_file`, `dblink`, `set_config`, …).
2. **Database session**: every statement runs in a `READ ONLY` transaction with a `statement_timeout`.
   Results are capped at `PGQA_MAX_ROWS`.
3. **Read-only role (you should set this up)**: connect as a role that only has `SELECT` privileges. See
   [03_readonly_role.sql](examples/seed/03_readonly_role.sql).

Use `PGQA_SCHEMAS=sales,finance` to limit what the agent can see. Set `PGQA_ALLOW_SAMPLING=false` to
turn off the raw-row tools for sensitive data. Even then, `run_query` can still return rows, so the
role's grants are the real boundary.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (or pip with Python 3.12+).

```bash
uv sync
cp .env.example .env            # then set PGQA_DATABASE_URL to your database
uv run ask-postgres check               # verify the connection and list what the agent can see
uv run ask-postgres check-llm           # (optional) verify the LLM for the built-in agent
```

To try it with sample data instead, run `docker compose up -d`. This starts an e-commerce demo database
on port 55432, and the default `.env.example` already points at it.

## Integrating with an agent

### 1. MCP agents (Claude Desktop, Claude Code, Cursor, VS Code, Windsurf, …)

**Claude Code:**

```bash
claude mcp add ask-postgres -e PGQA_DATABASE_URL=postgresql://user:pass@host:5432/db -- uv --directory /path/to/ask-postgres run ask-postgres mcp
```

**Claude Desktop / Cursor / VS Code:** add this to the client's MCP config (`claude_desktop_config.json`,
`.cursor/mcp.json`, …):

```json
{
  "mcpServers": {
    "ask-postgres": {
      "command": "uv",
      "args": ["--directory", "C:/Users/you/Projects/ask-postgres", "run", "ask-postgres", "mcp"],
      "env": { "PGQA_DATABASE_URL": "postgresql://ask_postgres_reader:reader@localhost:55432/shop" }
    }
  }
}
```

**Remote or shared agents:** run the MCP server over Streamable HTTP and point clients at
`http://host:8000/mcp`:

```bash
uv run ask-postgres mcp --transport http --host 0.0.0.0 --port 8000
```

The server also provides the resource `ask-postgres://schema` (a schema overview) and the prompt `ask_database`.
Its instructions carry the recommended analyst system prompt.

### 2. Function-calling agents (OpenAI, LangChain, LlamaIndex, Gemini, n8n, custom apps)

```bash
PGQA_API_KEY=change-me uv run ask-postgres serve --port 8080    # OpenAPI docs at http://localhost:8080/docs
```

| Endpoint | Use |
|---|---|
| `GET /tools?format=openai\|anthropic` | Tool definitions to put in your agent's `tools` array |
| `POST /tools/{name}` (JSON body = arguments) | Execute the tool the model called; errors come back as `is_error: true` with readable text |
| `GET /system-prompt` | The recommended system prompt |
| `POST /ask` `{"question": "...", "session_id": "..."}` | Complete answer from the built-in agent; reuse `session_id` for follow-up questions |

Send the key as `X-API-Key: …` or `Authorization: Bearer …`.

Your agent loop forwards each tool call it receives to `POST /tools/{name}`, then passes the `content` back
to the model as the tool result. You can also skip HTTP and use the registry in-process from Python:

```python
from ask_postgres import Settings, Database, ToolRegistry
registry = ToolRegistry(Database(Settings.from_env()))
tools = registry.openai_tools()                  # or registry.anthropic_tools()
output, is_error = registry.call_json("run_query", {"sql": "SELECT count(*) FROM sales.orders"})
```

### 3. Built-in agent: bring your own LLM

`ask-postgres ask "…"`, `ask-postgres chat` and `POST /ask` run a complete agent loop. Pick the LLM in `.env`; no code
changes are needed:

```bash
ask-postgres ask "Which product category generated the most revenue from delivered orders in Q3 2025?"
ask-postgres chat                 # multi-turn; follow-up questions keep their context
```

| You have | Settings |
|---|---|
| **Any OpenAI-compatible endpoint**: OpenAI, Azure OpenAI (v1), Ollama, LM Studio, vLLM, LiteLLM, OpenRouter, Groq, Mistral, … | `PGQA_LLM_PROVIDER=openai`, `PGQA_LLM_BASE_URL=…`, `PGQA_LLM_API_KEY=…`, `PGQA_LLM_MODEL=…` |
| **Google Gemini**: AI Studio key, Vertex AI, or a gateway that serves the native Gemini API | `PGQA_LLM_PROVIDER=gemini`, `PGQA_LLM_MODEL=gemini-…` (+ `PGQA_LLM_API_KEY`; for a gateway, `PGQA_LLM_BASE_URL=https://gateway/v1beta`) |
| **Anthropic Claude**, directly or through a gateway | `PGQA_LLM_PROVIDER=anthropic` (+ optional `PGQA_LLM_BASE_URL`, `PGQA_LLM_API_KEY`, `PGQA_LLM_MODEL`; default `claude-opus-5-5`) |
| **A local model, with no key and no data leaving the machine** | Ollama: `PGQA_LLM_PROVIDER=openai`, `PGQA_LLM_BASE_URL=http://localhost:11434/v1`, `PGQA_LLM_MODEL=qwen3:14b` |
| **An LLM gateway / proxy** (LiteLLM, Portkey, Kong, Azure APIM, Cloudflare AI Gateway, your company's gateway) | Choose the provider by the API format the gateway exposes (see below) |
| **Anything else** (Bedrock SDK, an internal API, …) | `PGQA_LLM_PROVIDER=yourpackage.module:YourBackend` (see below) |

[.env.example](.env.example) has copy-paste recipes for each case. After choosing one, test it without
touching the database:

```bash
ask-postgres check-llm          # sends one tiny request: prints the LLM in use and its reply, or the exact error
```

**LLM gateways.** A gateway usually exposes one or more standard API formats. The format decides
`PGQA_LLM_PROVIDER`, not the company that made the model. For example, Gemini models served in
OpenAI format use `openai`.

| Gateway endpoint | `PGQA_LLM_PROVIDER` | `PGQA_LLM_BASE_URL` |
|---|---|---|
| `…/v1/chat/completions` | `openai` | `https://<gateway>/v1` |
| `…/v1/messages` | `anthropic` | `https://<gateway>` (without `/v1`) |
| `…/v1beta/models/<model>:generateContent` | `gemini` | `https://<gateway>/v1beta` (keep the version) |

Use the model names from the gateway's catalog. If the gateway expects a non-standard auth header, add
`PGQA_LLM_HEADERS={"api-key": "…"}`. A `${VAR}` reference in `.env` keeps keys out of the file. On
Windows, ask-postgres falls back to the saved user/system variable when a terminal has a stale environment.
With `PGQA_LLM_PROVIDER=auto` (the default), the model name decides: names starting with `claude` use
Anthropic, names starting with `gemini` use Gemini, and any other name uses OpenAI-compatible. With no model
set, ask-postgres uses whichever key it finds: `OPENAI_API_KEY`, then `ANTHROPIC_API_KEY`, then `GEMINI_API_KEY` /
`GOOGLE_API_KEY`.

For Gemini behind a gateway, end `PGQA_LLM_BASE_URL` with the gateway's API version (`…/v1beta`). Recent
Google SDKs otherwise call `/v1`, which many gateways don't serve.

**Model choice matters.** The model must support tool/function calling. Strong models (Claude, GPT-5.x,
large Qwen/DeepSeek) explore the schema and verify results more thoroughly. Small local models work for
simple questions.

**Custom backend.** Subclass `ask_postgres.llm.LLMBackend` and implement four methods. The agent loop, tools and
guardrails stay the same:

```python
from ask_postgres.llm import LLMBackend, Turn, ToolCall, ToolResult

class MyBackend(LLMBackend):
    # self.system_prompt and self.tools ([{name, description, input_schema}]) are provided
    def add_user_message(self, text: str) -> None: ...
    def generate(self) -> Turn: ...                   # call your model -> Turn(text, [ToolCall(id, name, args)])
    def add_tool_results(self, results: list[ToolResult]) -> None: ...
    def reset(self) -> None: ...
```

## Configuration

| Variable | Default | |
|---|---|---|
| `PGQA_DATABASE_URL` (or `DATABASE_URL`) | required | libpq connection string or URL |
| `PGQA_SCHEMAS` | all non-system | Comma-separated allowlist; also becomes the `search_path` |
| `PGQA_MAX_ROWS` | 200 | Maximum rows returned per query |
| `PGQA_STATEMENT_TIMEOUT_MS` | 15000 | Per-statement timeout enforced by PostgreSQL |
| `PGQA_ALLOW_SAMPLING` | true | Enables `sample_rows` and `column_values` |
| `PGQA_API_KEY` | unset | Requires a key on the HTTP API |
| `PGQA_LLM_PROVIDER` | `auto` | `anthropic`, `openai`, `gemini`, or `package.module:Class` |
| `PGQA_LLM_MODEL` | provider default | Required for `openai`; defaults: `claude-opus-5-5` (anthropic), `gemini-3.8-flash` (gemini) |
| `PGQA_LLM_BASE_URL` | provider default | Gateway, self-hosted or local endpoint |
| `PGQA_LLM_API_KEY` | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | Not needed for local servers |
| `PGQA_LLM_HEADERS` | `{}` | Extra HTTP headers as JSON, e.g. for gateways that need a custom auth header |
| `PGQA_LLM_EXTRA_PARAMS` | `{}` | JSON passed through to the model call, e.g. `{"temperature": 0}` |
| `PGQA_EFFORT` | `high` | Anthropic only |
| `PGQA_MAX_AGENT_TURNS` | 25 | Safety stop for the agent loop |
| `PGQA_REFUSAL_FALLBACK` | true | Anthropic only; auto-disabled if the endpoint rejects it |

**Getting better answers:** add `COMMENT ON TABLE/COLUMN` to your schema. The agent reads comments and
uses them to understand business meaning, e.g. what "revenue" or an `active` flag means.

## Development

```bash
docker compose up -d     # demo database
uv run pytest            # unit tests + integration tests against the demo DB (PGQA_TEST_DATABASE_URL to override)
uv run ask-postgres call describe_table '{"table": "orders"}'   # call any tool by hand
```

```
src/ask_postgres/
  config.py      settings from env / .env
  sql_guard.py   static SQL validation
  db.py          introspection + guarded execution (psycopg 3 pool)
  tools.py       provider-neutral tool registry + system prompt
  mcp_server.py  MCP server (stdio / streamable HTTP)
  http_api.py    FastAPI REST API
  agent.py       built-in agent loop (provider-neutral)
  llm.py         bring-your-own-LLM backends (Anthropic, OpenAI-compatible, Gemini, custom)
  cli.py         `ask-postgres` command
```
