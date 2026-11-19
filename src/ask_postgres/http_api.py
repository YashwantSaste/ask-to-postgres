"""HTTP API for agents that speak function calling but not MCP (OpenAI Assistants/Responses, LangChain,
n8n, Copilot Studio, custom apps...). Interactive docs at /docs once running."""

from __future__ import annotations

import os
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ask_postgres.config import Settings
from ask_postgres.db import Database
from ask_postgres.tools import SYSTEM_PROMPT, ToolRegistry


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = Field(None, description="Reuse to ask follow-up questions in the same conversation.")


class AskResponse(BaseModel):
    session_id: str
    answer: str
    queries: list[str]
    tool_calls: int


def create_app(settings: Settings | None = None) -> FastAPI:
    state: dict[str, Any] = {}
    sessions: dict[str, Any] = {}
    sessions_lock = threading.Lock()
    api_key = os.getenv("PGQA_API_KEY")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings or Settings.from_env())
        state["registry"] = ToolRegistry(db)
        yield
        db.close()

    def registry() -> ToolRegistry:
        return state["registry"]

    def check_key(x_api_key: str | None = Header(None), authorization: str | None = Header(None)) -> None:
        if not api_key:
            return
        supplied = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied or "", api_key):
            raise HTTPException(401, "Invalid or missing API key")

    app = FastAPI(title="ask-postgres", version="0.1.0", lifespan=lifespan,
                  description="Ask questions about the data in a PostgreSQL database.",
                  dependencies=[Depends(check_key)])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "database": registry().db.server_info()["database"]}

    @app.get("/system-prompt")
    def system_prompt() -> dict:
        """Recommended system prompt for your own agent."""
        return {"system_prompt": SYSTEM_PROMPT}

    @app.get("/tools")
    def list_tools(format: Literal["openai", "anthropic"] = "openai") -> list[dict]:
        """Tool definitions ready to paste into an OpenAI-style or Anthropic-style tools array."""
        r = registry()
        return r.openai_tools() if format == "openai" else r.anthropic_tools()

    @app.post("/tools/{name}")
    def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict:
        """Execute a tool. Errors are returned (not raised) so an agent can read them and self-correct."""
        r = registry()
        if name not in r.tools:
            raise HTTPException(404, f"Unknown tool '{name}'")
        output, is_error = r.call_json(name, arguments)
        return {"tool": name, "is_error": is_error, "content": output}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        """Answer a question end-to-end with the built-in agent, using the LLM configured via PGQA_LLM_*."""
        from ask_postgres.agent import Agent

        session_id = req.session_id or uuid.uuid4().hex
        with sessions_lock:
            agent = sessions.get(session_id)
            if agent is None:
                agent = sessions[session_id] = Agent(registry())
        result = agent.ask(req.question)
        return AskResponse(session_id=session_id, answer=result.text, queries=result.queries,
                           tool_calls=result.tool_calls)

    @app.delete("/sessions/{session_id}")
    def end_session(session_id: str) -> dict:
        with sessions_lock:
            sessions.pop(session_id, None)
        return {"deleted": session_id}

    return app
