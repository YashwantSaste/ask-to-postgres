"""Offline tests for the bring-your-own-LLM layer (no network, no database)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ask_postgres.agent import Agent
from ask_postgres.config import Settings
from ask_postgres.llm import (
    AnthropicBackend, LLMBackend, OpenAICompatibleBackend, ToolCall, Turn, create_backend, resolve_provider,
)

SETTINGS = Settings(database_url="postgresql://unused")


class ScriptedBackend(LLMBackend):
    """Plays back a fixed list of turns and records everything the agent sends."""

    def __init__(self, settings, system_prompt, tools, turns=()):
        super().__init__(settings, system_prompt, tools)
        self.turns = list(turns)
        self.log: list = []

    def add_user_message(self, text):
        self.log.append(("user", text))

    def generate(self):
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.log.append(("results", results))

    def reset(self):
        self.log.clear()


class FakeRegistry:
    db = SimpleNamespace(settings=SETTINGS)

    def anthropic_tools(self):
        return [{"name": "run_query", "description": "d", "input_schema": {"type": "object"}}]

    def call_json(self, name, arguments):
        return '{"rows": [{"n": 42}]}', False


def test_agent_loop_runs_tools_then_answers():
    backend = ScriptedBackend(SETTINGS, "sys", [], turns=[
        Turn("", [ToolCall("c1", "run_query", {"sql": "SELECT 42 AS n"}),
                  ToolCall("c2", "run_query", error="Arguments were not valid JSON")], "tool_calls"),
        Turn("The answer is 42.", [], "stop"),
    ])
    answer = Agent(FakeRegistry(), backend=backend).ask("question?")

    assert answer.text == "The answer is 42."
    assert answer.queries == ["SELECT 42 AS n"]
    assert answer.tool_calls == 2
    results = backend.log[1][1]
    assert [r.is_error for r in results] == [False, True]  # bad JSON goes back to the model as an error


def _completion(content=None, tool_calls=None, finish_reason="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])


def _tool_call(id_, name, arguments):
    return SimpleNamespace(id=id_, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def test_openai_compatible_backend_message_flow():
    settings = replace(SETTINGS, llm_model="some-model", llm_base_url="http://localhost:11434/v1")
    backend = OpenAICompatibleBackend(settings, "SYSTEM", FakeRegistry().anthropic_tools())
    sent = []
    replies = [
        _completion(tool_calls=[_tool_call("t1", "run_query", '{"sql": "SELECT 1"}'),
                                _tool_call("t2", "run_query", "{not json")], finish_reason="tool_calls"),
        _completion(content="Done."),
    ]

    def create(**kwargs):
        sent.append([dict(m) for m in kwargs["messages"]])
        assert kwargs["tools"][0]["function"]["name"] == "run_query"
        return replies.pop(0)

    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = Agent(FakeRegistry(), backend=backend).ask("q")

    assert answer.text == "Done."
    second_request = sent[1]
    assert second_request[0] == {"role": "system", "content": "SYSTEM"}
    assert second_request[2]["tool_calls"][0]["function"]["arguments"] == '{"sql": "SELECT 1"}'
    tool_msgs = [m for m in second_request if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["t1", "t2"]
    assert tool_msgs[1]["content"].startswith("ERROR:")


def test_openai_backend_requires_model():
    with pytest.raises(ValueError, match="PGQA_LLM_MODEL"):
        OpenAICompatibleBackend(replace(SETTINGS, llm_base_url="http://x"), "s", [])


@pytest.mark.parametrize(("overrides", "env", "expected"), [
    ({"llm_provider": "openai"}, {}, "openai"),
    ({"llm_model": "claude-sonnet-5-5"}, {}, "anthropic"),
    ({"llm_model": "llama3.3"}, {}, "openai"),
    ({}, {"OPENAI_API_KEY": "k"}, "openai"),
    ({}, {"ANTHROPIC_API_KEY": "k"}, "anthropic"),
])
def test_resolve_provider(monkeypatch, overrides, env, expected):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert resolve_provider(replace(SETTINGS, **overrides)) == expected


def test_resolve_provider_without_any_llm(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="No LLM configured"):
        resolve_provider(SETTINGS)


def test_custom_backend_by_import_path():
    backend = create_backend(replace(SETTINGS, llm_provider="tests.test_llm:ScriptedBackend"), "s", [])
    assert isinstance(backend, ScriptedBackend)
    with pytest.raises(TypeError, match="subclass"):
        create_backend(replace(SETTINGS, llm_provider="tests.test_llm:FakeRegistry"), "s", [])


def test_anthropic_backend_default_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert AnthropicBackend(SETTINGS, "s", []).model == "claude-opus-5-5"


@pytest.mark.parametrize(("base_url", "expected_base", "expected_version"), [
    ("https://gateway.example/v1beta", "https://gateway.example/", "v1beta"),
    ("https://gateway.example/v1beta/", "https://gateway.example/", "v1beta"),
    ("https://gateway.example/proxy", "https://gateway.example/proxy", None),
])
def test_gemini_base_url_version_split(base_url, expected_base, expected_version):
    from ask_postgres.llm import GeminiBackend

    backend = GeminiBackend(replace(SETTINGS, llm_base_url=base_url, llm_api_key="k"), "s", [])
    opts = backend.client._api_client._http_options
    assert opts.base_url == expected_base
    if expected_version:
        assert opts.api_version == expected_version


def test_gemini_backend_message_flow():
    from google.genai import types

    from ask_postgres.llm import GeminiBackend

    backend = GeminiBackend(replace(SETTINGS, llm_api_key="k", llm_model="gemini-x"), "SYSTEM",
                            FakeRegistry().anthropic_tools())
    assert backend.config.automatic_function_calling.disable
    sent = []

    def reply(*parts, finish="STOP"):
        content = types.Content(role="model", parts=list(parts))
        return types.GenerateContentResponse(candidates=[types.Candidate(content=content, finish_reason=finish)])

    replies = [
        reply(types.Part(function_call=types.FunctionCall(name="run_query", args={"sql": "SELECT 1"})),
              types.Part(text="thinking...", thought=True)),
        reply(types.Part(text="The answer is 42.")),
    ]

    def generate_content(model, contents, config):
        sent.append(list(contents))
        return replies.pop(0)

    backend.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    answer = Agent(FakeRegistry(), backend=backend).ask("q")

    assert answer.text == "The answer is 42."  # thought parts are not shown as answer text
    assert answer.queries == ["SELECT 1"]
    roles = [c.role for c in sent[1]]
    assert roles == ["user", "model", "user"]
    fr = sent[1][2].parts[0].function_response
    assert fr.name == "run_query" and fr.id is None and "42" in fr.response["result"]


def test_resolve_gemini(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_provider(replace(SETTINGS, llm_model="gemini-3.8-flash")) == "gemini"
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert resolve_provider(SETTINGS) == "gemini"


def test_env_reference_falls_back_to_saved_windows_variable(tmp_path, monkeypatch):
    import ask_postgres.config as config

    env = tmp_path / ".env"
    env.write_text("PGQA_DATABASE_URL=postgresql://x@localhost/x\nPGQA_LLM_API_KEY=${PGQA_TEST_SAVED_KEY}\n")
    for var in ("PGQA_TEST_SAVED_KEY", "PGQA_LLM_API_KEY", "PGQA_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(config, "_saved_windows_variable", lambda name: "saved-key" if name == "PGQA_TEST_SAVED_KEY" else None)
    assert Settings.from_env(str(env)).llm_api_key == "saved-key"

    for var in ("PGQA_TEST_SAVED_KEY", "PGQA_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "_saved_windows_variable", lambda name: None)
    s = Settings.from_env(str(env))
    assert s.llm_api_key is None and s.llm_api_key_missing_ref == "PGQA_TEST_SAVED_KEY"
    with pytest.raises(RuntimeError, match=r"\$\{PGQA_TEST_SAVED_KEY\}"):
        create_backend(s, "x", [])


def test_custom_headers_reach_every_backend(monkeypatch):
    from ask_postgres.llm import GeminiBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    s = replace(SETTINGS, llm_headers={"X-Gateway-Key": "secret"}, llm_model="m", llm_api_key="k",
                llm_base_url="http://gateway.example/v1")
    assert AnthropicBackend(s, "s", []).client.default_headers["X-Gateway-Key"] == "secret"
    assert OpenAICompatibleBackend(s, "s", []).client.default_headers["X-Gateway-Key"] == "secret"
    assert GeminiBackend(s, "s", []).client._api_client._http_options.headers["X-Gateway-Key"] == "secret"


def test_backends_omit_empty_tool_lists():
    from ask_postgres.llm import GeminiBackend

    s = replace(SETTINGS, llm_model="m", llm_api_key="k", llm_base_url="http://x/v1")
    assert GeminiBackend(s, "s", []).config.tools is None
    backend = OpenAICompatibleBackend(s, "s", [])
    seen = {}
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: seen.update(kw) or _completion(content="OK"))))
    backend.add_user_message("hi")
    assert backend.generate().text == "OK" and "tools" not in seen
