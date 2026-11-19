"""Bring-your-own-LLM backends for the built-in agent.

A backend owns the conversation in its provider's native message format and exposes a tiny,
provider-neutral interface to the agent loop (agent.py):

    add_user_message(text) -> generate() -> Turn(text, tool_calls) -> add_tool_results([...]) -> generate() ...

Built in:
  * "anthropic" - Claude via the Anthropic Messages API (direct or through a gateway)
  * "openai"    - any OpenAI-compatible Chat Completions endpoint: OpenAI, Azure OpenAI (v1 endpoint),
                  Ollama, LM Studio, vLLM, LiteLLM, OpenRouter, Groq, Mistral, Gemini's compatible
                  endpoint, corporate LLM gateways...
  * "gemini"    - Google Gemini via the native Gemini API (AI Studio key, Vertex AI, or a /v1beta gateway)
  * "package.module:ClassName" - your own LLMBackend subclass for anything else.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ask_postgres.config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Set when the model produced arguments that could not be parsed; sent back so it can retry.
    error: str | None = None


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None


class LLMBackend(ABC):
    """Subclass this to plug in any LLM. Constructor receives (settings, system_prompt, tools) where
    tools is a list of {"name", "description", "input_schema"} dicts (JSON Schema)."""

    def __init__(self, settings: Settings, system_prompt: str, tools: list[dict[str, Any]]):
        self.settings = settings
        self.system_prompt = system_prompt
        self.tools = tools

    @abstractmethod
    def add_user_message(self, text: str) -> None: ...

    @abstractmethod
    def generate(self) -> Turn:
        """Call the model once, record its reply in the history and return it."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self.settings.llm_model})"


# --------------------------------------------------------------------------------------------- Anthropic


class AnthropicBackend(LLMBackend):
    default_model = "claude-opus-5-5"

    def __init__(self, settings: Settings, system_prompt: str, tools: list[dict[str, Any]]):
        super().__init__(settings, system_prompt, tools)
        import anthropic

        self._anthropic = anthropic
        # None falls back to ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL from the environment.
        self.client = anthropic.Anthropic(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                                          default_headers=settings.llm_headers or None)
        self.model = settings.llm_model or self.default_model
        self.refusal_fallback = settings.refusal_fallback
        self.messages: list[dict[str, Any]] = []

    @property
    def description(self) -> str:
        return f"anthropic:{self.model}" + (f" @ {self.settings.llm_base_url}" if self.settings.llm_base_url else "")

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        # All results for one assistant turn go back in a single user message.
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
            for r in results
        ]})

    def reset(self) -> None:
        self.messages = []

    def _create(self):
        params: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=self.system_prompt,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            cache_control={"type": "ephemeral"},
            **self.settings.llm_extra_params,
        )
        if self.tools:
            params["tools"] = self.tools
        if self.refusal_fallback:
            # On a safety-classifier refusal, the API transparently retries on a suitable fallback model.
            try:
                return self.client.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            except self._anthropic.BadRequestError as e:
                # Some gateways/proxies don't support this beta; continue without it.
                if "anthropic-beta" not in str(e) and "fallback" not in str(e):
                    raise
                self.refusal_fallback = False
        return self.client.beta.messages.create(**params)

    def generate(self) -> Turn:
        while True:
            response = self._create()
            # Append the full content (incl. thinking blocks) so the conversation stays valid.
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "pause_turn":
                break

        if response.stop_reason == "refusal":
            return Turn("The model declined to answer this request.", stop_reason="refusal")
        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "max_tokens":
            text += "\n\n[Answer truncated: output token limit reached.]"
        calls = [ToolCall(b.id, b.name, dict(b.input or {})) for b in response.content if b.type == "tool_use"]
        return Turn(text, calls if response.stop_reason == "tool_use" else [], response.stop_reason)


# ------------------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, settings: Settings, system_prompt: str, tools: list[dict[str, Any]]):
        super().__init__(settings, system_prompt, tools)
        import openai

        if not settings.llm_model:
            raise ValueError("Set PGQA_LLM_MODEL to the model name your OpenAI-compatible endpoint serves.")
        api_key = settings.llm_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key and settings.llm_base_url:
            api_key = "not-needed"  # local servers such as Ollama / LM Studio ignore the key
        self.client = openai.OpenAI(api_key=api_key, base_url=settings.llm_base_url,
                                    default_headers=settings.llm_headers or None)
        self.model = settings.llm_model
        self.openai_tools = [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ]
        self.reset()

    @property
    def description(self) -> str:
        return f"openai-compatible:{self.model}" + (f" @ {self.settings.llm_base_url}" if self.settings.llm_base_url else "")

    def reset(self) -> None:
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})

    def generate(self) -> Turn:
        tools = {"tools": self.openai_tools} if self.openai_tools else {}
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages, **tools, **self.settings.llm_extra_params
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [tc for tc in (msg.tool_calls or []) if getattr(tc, "type", "function") == "function"]

        # Record the reply in plain dict form: portable across strict OpenAI-compatible servers.
        record: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tool_calls:
            record["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in tool_calls
            ]
        self.messages.append(record)

        if getattr(msg, "refusal", None):
            return Turn(msg.refusal, stop_reason="refusal")

        calls: list[ToolCall] = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {}))
            except json.JSONDecodeError as e:
                calls.append(ToolCall(tc.id, tc.function.name, error=f"Arguments were not valid JSON: {e}"))

        text = (msg.content or "").strip()
        if choice.finish_reason == "length":
            text += "\n\n[Answer truncated: output token limit reached.]"
        return Turn(text, calls, choice.finish_reason)


# ------------------------------------------------------------------------------------------------ Gemini


class GeminiBackend(LLMBackend):
    """Google Gemini via the native Gemini API (google-genai SDK): Google AI Studio keys, or a gateway that
    proxies /v1beta. For Vertex AI set GOOGLE_GENAI_USE_VERTEXAI=true plus GOOGLE_CLOUD_PROJECT/LOCATION."""

    default_model = "gemini-3.8-flash"

    def __init__(self, settings: Settings, system_prompt: str, tools: list[dict[str, Any]]):
        super().__init__(settings, system_prompt, tools)
        from google import genai
        from google.genai import types

        self._types = types
        options: dict[str, Any] = {"headers": settings.llm_headers} if settings.llm_headers else {}
        if settings.llm_base_url:
            # Accept the gateway's documented base path, e.g. https://gateway/v1beta: the trailing version
            # segment becomes the SDK's api_version (the SDK otherwise appends its own default, "v1").
            base, _, last = settings.llm_base_url.rstrip("/").rpartition("/")
            if re.fullmatch(r"v\d+(alpha|beta)?\d*", last):
                options.update(base_url=base + "/", api_version=last)
            else:
                options.update(base_url=settings.llm_base_url)
        http_options = types.HttpOptions(**options) if options else None
        # None falls back to GEMINI_API_KEY / GOOGLE_API_KEY (or Vertex AI settings) from the environment.
        self.client = genai.Client(api_key=settings.llm_api_key, http_options=http_options)
        self.model = settings.llm_model or self.default_model
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["input_schema"])
                for t in tools
            ])] if tools else None,
            # We run the tools ourselves (with guardrails), so the SDK must not auto-execute anything.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            **settings.llm_extra_params,
        )
        self.contents: list[Any] = []

    @property
    def description(self) -> str:
        return f"gemini:{self.model}" + (f" @ {self.settings.llm_base_url}" if self.settings.llm_base_url else "")

    def reset(self) -> None:
        self.contents = []

    def add_user_message(self, text: str) -> None:
        self.contents.append(self._types.Content(role="user", parts=[self._types.Part.from_text(text=text)]))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        parts = []
        for r in results:
            response = {"error": r.content} if r.is_error else {"result": r.content}
            parts.append(self._types.Part(function_response=self._types.FunctionResponse(
                id=r.call_id if not r.call_id.startswith("ask-postgres-") else None, name=r.name, response=response)))
        self.contents.append(self._types.Content(role="user", parts=parts))

    def generate(self) -> Turn:
        response = self.client.models.generate_content(model=self.model, contents=self.contents, config=self.config)
        candidate = (response.candidates or [None])[0]
        if candidate is None or candidate.content is None:
            reason = getattr(response.prompt_feedback, "block_reason", None) or getattr(candidate, "finish_reason", None)
            return Turn(f"The model returned no answer (reason: {reason}).", stop_reason=str(reason))

        # Keep the model's content as-is: it carries thought signatures Gemini needs on the next call.
        self.contents.append(candidate.content)
        parts = candidate.content.parts or []
        calls = [
            ToolCall(p.function_call.id or f"ask-postgres-{len(self.contents)}-{i}", p.function_call.name,
                     dict(p.function_call.args or {}))
            for i, p in enumerate(parts) if p.function_call
        ]
        text = "".join(p.text for p in parts if p.text and not p.thought).strip()
        finish = str(candidate.finish_reason.name if candidate.finish_reason else "")
        if finish == "MAX_TOKENS":
            text += "\n\n[Answer truncated: output token limit reached.]"
        return Turn(text, calls, finish)


# ------------------------------------------------------------------------------------------------ factory

_BUILTIN: dict[str, type[LLMBackend]] = {
    "anthropic": AnthropicBackend,
    "claude": AnthropicBackend,
    "openai": OpenAICompatibleBackend,
    "openai-compatible": OpenAICompatibleBackend,
    "gemini": GeminiBackend,
    "google": GeminiBackend,
}


def resolve_provider(settings: Settings) -> str:
    """Pick a provider when PGQA_LLM_PROVIDER is 'auto' (the default)."""
    provider = settings.llm_provider.strip()
    if provider and provider.lower() != "auto":
        return provider
    model = (settings.llm_model or "").lower()
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    if model or os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    raise RuntimeError(
        "No LLM configured for the built-in agent. Set PGQA_LLM_PROVIDER (anthropic | openai | gemini | module:Class), "
        "PGQA_LLM_MODEL and, if needed, PGQA_LLM_BASE_URL / PGQA_LLM_API_KEY. See README 'Bring your own LLM'. "
        "(Not needed if you connect your own agent via MCP or the HTTP tool endpoints.)"
    )


def create_backend(settings: Settings, system_prompt: str, tools: list[dict[str, Any]]) -> LLMBackend:
    if settings.llm_api_key_missing_ref:
        ref = settings.llm_api_key_missing_ref
        raise RuntimeError(
            f"PGQA_LLM_API_KEY in .env refers to ${{{ref}}}, but {ref} is not set in this terminal or in your saved "
            f"Windows environment variables. Set it (PowerShell: $env:{ref} = \"<key>\") or put the key directly "
            f"into PGQA_LLM_API_KEY."
        )
    provider = resolve_provider(settings)
    cls = _BUILTIN.get(provider.lower())
    if cls is None:
        if ":" not in provider:
            raise ValueError(f"Unknown LLM provider '{provider}'. Use anthropic, openai, gemini or package.module:ClassName.")
        module_name, _, class_name = provider.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        if not (isinstance(cls, type) and issubclass(cls, LLMBackend)):
            raise TypeError(f"{provider} must be a subclass of ask_postgres.llm.LLMBackend")
    return cls(settings, system_prompt, tools)
