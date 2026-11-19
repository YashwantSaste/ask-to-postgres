"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from dotenv import dotenv_values, find_dotenv, load_dotenv

_REFERENCE = re.compile(r"\$\{(\w+)\}")


def _saved_windows_variable(name: str) -> str | None:
    """Read a user/system environment variable as saved in Windows (not this process's copy)."""
    if sys.platform != "win32":
        return None
    import winreg

    locations = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for root, path in locations:
        try:
            with winreg.OpenKey(root, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            return os.path.expandvars(str(value))
    return None


def _fill_missing_references(raw_values: dict[str, str | None]) -> None:
    """Terminals opened from an app that started earlier (VS Code, Windows Terminal) keep a stale copy of the
    environment. For ${VAR} references used in .env, fall back to the variable saved in Windows."""
    for raw in raw_values.values():
        for name in _REFERENCE.findall(raw or ""):
            if not os.environ.get(name) and (value := _saved_windows_variable(name)):
                os.environ[name] = value


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    # Schemas the agent may see and query. Empty = every non-system schema.
    schemas: list[str] = field(default_factory=list)
    # Hard cap on rows returned to the agent per query.
    max_rows: int = 200
    # Per-statement timeout enforced by PostgreSQL.
    statement_timeout_ms: int = 15_000
    # Rows shown by the sample_rows tool.
    sample_rows: int = 5
    # Allow the sample_rows / column_values tools (disable for sensitive data).
    allow_sampling: bool = True
    pool_min_size: int = 1
    pool_max_size: int = 5
    # Built-in agent: bring your own LLM (see llm.py).
    llm_provider: str = "auto"  # anthropic | openai | gemini | package.module:ClassName | auto
    llm_model: str | None = None  # None = provider default (Claude Opus 5.5 for anthropic)
    llm_base_url: str | None = None  # gateway / self-hosted / local endpoint
    llm_api_key: str | None = None  # None = provider's usual env var (ANTHROPIC_API_KEY, OPENAI_API_KEY)
    # Name of the variable PGQA_LLM_API_KEY refers to (${NAME}) when it could not be resolved - for errors.
    llm_api_key_missing_ref: str | None = None
    llm_extra_params: dict[str, Any] = field(default_factory=dict)  # passed through to the provider call
    llm_headers: dict[str, str] = field(default_factory=dict)  # extra HTTP headers, e.g. for gateways/proxies
    effort: str = "high"  # Anthropic only
    max_agent_turns: int = 25
    # Server-side refusal fallback (auto-disabled if the endpoint rejects it).
    refusal_fallback: bool = True

    @classmethod
    def from_env(cls, env_file: str | None = None, require_database: bool = True) -> Settings:
        env_path = env_file or find_dotenv(usecwd=True) or find_dotenv()
        raw_values = dotenv_values(env_path, interpolate=False) if env_path else {}
        _fill_missing_references(raw_values)
        load_dotenv(env_path or None)

        api_key = os.getenv("PGQA_LLM_API_KEY") or None
        missing_ref = None
        if api_key is None and (refs := _REFERENCE.findall(raw_values.get("PGQA_LLM_API_KEY") or "")):
            missing_ref = refs[0]

        url = os.getenv("PGQA_DATABASE_URL") or os.getenv("DATABASE_URL") or ""
        if not url and require_database:
            raise RuntimeError(
                "Set PGQA_DATABASE_URL (or DATABASE_URL), e.g. "
                "postgresql://readonly_user:secret@localhost:5432/mydb"
            )
        return cls(
            database_url=url,
            schemas=_csv(os.getenv("PGQA_SCHEMAS")),
            max_rows=int(os.getenv("PGQA_MAX_ROWS", "200")),
            statement_timeout_ms=int(os.getenv("PGQA_STATEMENT_TIMEOUT_MS", "15000")),
            sample_rows=int(os.getenv("PGQA_SAMPLE_ROWS", "5")),
            allow_sampling=_bool(os.getenv("PGQA_ALLOW_SAMPLING"), True),
            pool_min_size=int(os.getenv("PGQA_POOL_MIN_SIZE", "1")),
            pool_max_size=int(os.getenv("PGQA_POOL_MAX_SIZE", "5")),
            llm_provider=os.getenv("PGQA_LLM_PROVIDER", "auto"),
            llm_model=os.getenv("PGQA_LLM_MODEL") or os.getenv("PGQA_MODEL") or None,
            llm_base_url=os.getenv("PGQA_LLM_BASE_URL") or None,
            llm_api_key=api_key,
            llm_api_key_missing_ref=missing_ref,
            llm_extra_params=json.loads(os.getenv("PGQA_LLM_EXTRA_PARAMS") or "{}"),
            llm_headers=json.loads(os.getenv("PGQA_LLM_HEADERS") or "{}"),
            effort=os.getenv("PGQA_EFFORT", "high"),
            max_agent_turns=int(os.getenv("PGQA_MAX_AGENT_TURNS", "25")),
            refusal_fallback=_bool(os.getenv("PGQA_REFUSAL_FALLBACK"), True),
        )
