"""Built-in question-answering agent - works with any LLM backend (see llm.py).

Optional: if you already have an agent (Claude Desktop, Cursor, LangChain, an OpenAI-based app...),
connect it through the MCP server or the HTTP API instead and you don't need this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ask_postgres.llm import LLMBackend, ToolResult, create_backend
from ask_postgres.tools import SYSTEM_PROMPT, ToolRegistry

ToolEvent = Callable[[str, dict[str, Any], str, bool], None]


@dataclass
class Answer:
    text: str
    queries: list[str] = field(default_factory=list)
    tool_calls: int = 0
    stop_reason: str | None = None


class Agent:
    """Multi-turn agent: call ask() repeatedly; follow-up questions keep the conversation context."""

    def __init__(self, registry: ToolRegistry, backend: LLMBackend | None = None, max_turns: int | None = None):
        settings = registry.db.settings
        self.registry = registry
        self.backend = backend or create_backend(settings, SYSTEM_PROMPT, registry.anthropic_tools())
        self.max_turns = max_turns or settings.max_agent_turns

    def reset(self) -> None:
        self.backend.reset()

    def ask(self, question: str, on_tool: ToolEvent | None = None) -> Answer:
        self.backend.add_user_message(question)
        answer = Answer(text="")

        for _ in range(self.max_turns):
            turn = self.backend.generate()
            answer.stop_reason = turn.stop_reason
            if not turn.tool_calls:
                answer.text = turn.text
                return answer

            results = []
            for call in turn.tool_calls:
                if call.error:
                    output, is_error = call.error, True
                else:
                    output, is_error = self.registry.call_json(call.name, call.arguments)
                answer.tool_calls += 1
                if call.name == "run_query" and not is_error:
                    answer.queries.append(call.arguments.get("sql", ""))
                if on_tool:
                    on_tool(call.name, call.arguments, output, is_error)
                results.append(ToolResult(call.id, call.name, output, is_error))
            self.backend.add_tool_results(results)

        answer.text = f"Stopped after {self.max_turns} model turns without a final answer."
        return answer
