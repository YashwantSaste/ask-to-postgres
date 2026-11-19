"""Command line entry point: `ask-postgres <command>`."""

from __future__ import annotations

import argparse
import json
import sys


def _print_tool(name: str, args: dict, output: str, is_error: bool) -> None:
    detail = args.get("sql") or args.get("table") or args.get("keyword") or ""
    status = "ERROR" if is_error else "ok"
    print(f"  -> {name}({detail.strip()[:200]}) [{status}]", file=sys.stderr)
    if is_error:
        print(f"     {output[:300]}", file=sys.stderr)


def _llm_errors() -> tuple[type[BaseException], ...]:
    """Configuration / auth / API errors from any LLM backend - shown as a message, not a traceback."""
    import anthropic
    import openai
    from google.genai import errors as genai_errors

    return (anthropic.APIError, openai.OpenAIError, genai_errors.APIError, RuntimeError, ValueError, TypeError, ImportError)


def _registry():
    from ask_postgres.config import Settings
    from ask_postgres.db import Database
    from ask_postgres.tools import ToolRegistry

    return ToolRegistry(Database(Settings.from_env()))


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
    parser = argparse.ArgumentParser(prog="ask-postgres", description="Ask questions about a PostgreSQL database.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mcp", help="Run the MCP server (for Claude Desktop/Code, Cursor, VS Code, ...).")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("serve", help="Run the HTTP/REST API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)

    p = sub.add_parser("ask", help="Ask one question with the built-in Claude agent.")
    p.add_argument("question")
    p.add_argument("-q", "--quiet", action="store_true", help="Don't show tool calls.")

    p = sub.add_parser("chat", help="Interactive multi-turn Q&A with the built-in Claude agent.")
    p.add_argument("-q", "--quiet", action="store_true")

    p = sub.add_parser("check", help="Test the database connection and show what the agent can see.")

    p = sub.add_parser("check-llm", help="Test the configured LLM (PGQA_LLM_*) with one tiny request; no database needed.")

    p = sub.add_parser("tools", help="Print tool definitions for wiring into your own agent.")
    p.add_argument("--format", choices=["openai", "anthropic"], default="openai")

    p = sub.add_parser("call", help="Call a tool directly, e.g. ask-postgres call run_query '{\"sql\": \"select 1\"}'.")
    p.add_argument("tool")
    p.add_argument("arguments", nargs="?", default="{}")

    args = parser.parse_args(argv)

    if args.command == "mcp":
        from ask_postgres.mcp_server import main as mcp_main

        mcp_main(args.transport, args.host, args.port)

    elif args.command == "serve":
        import uvicorn

        from ask_postgres.http_api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)

    elif args.command in ("ask", "chat"):
        from ask_postgres.agent import Agent

        LLM_ERRORS = _llm_errors()
        try:
            agent = Agent(_registry())
        except LLM_ERRORS as e:
            sys.exit(f"Could not set up the LLM: {e}")
        print(f"[LLM: {agent.backend.description}]", file=sys.stderr)
        on_tool = None if args.quiet else _print_tool

        def answer(question: str) -> str | None:
            try:
                return agent.ask(question, on_tool).text
            except LLM_ERRORS as e:
                print(f"LLM error from {agent.backend.description}: {e}\n"
                      "Check PGQA_LLM_PROVIDER / PGQA_LLM_MODEL / PGQA_LLM_BASE_URL / PGQA_LLM_API_KEY.",
                      file=sys.stderr)
                return None

        if args.command == "ask":
            text = answer(args.question)
            if text is None:
                sys.exit(1)
            print(text)
            return
        print("Ask a question about your data. Commands: /reset, /exit", file=sys.stderr)
        while True:
            try:
                question = input("\n? ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question in ("/exit", "/quit"):
                break
            if question == "/reset":
                agent.reset()
                continue
            if question and (text := answer(question)) is not None:
                print("\n" + text)

    elif args.command == "check-llm":
        from ask_postgres.config import Settings
        from ask_postgres.llm import create_backend

        LLM_ERRORS = _llm_errors()
        try:
            backend = create_backend(Settings.from_env(require_database=False), "You are a connectivity test.", [])
            print(f"LLM: {backend.description}")
            backend.add_user_message("Reply with exactly: OK")
            turn = backend.generate()
        except LLM_ERRORS as e:
            sys.exit(f"LLM check FAILED: {type(e).__name__}: {e}")
        print(f"Reply: {turn.text!r}")
        print("LLM check passed.")

    elif args.command == "check":
        registry = _registry()
        overview = registry.call("get_database_overview")
        tables = overview.pop("tables")
        print(json.dumps(overview, indent=2, default=str))
        print(f"\n{len(tables)} tables/views visible:")
        for t in tables:
            rows = "rows unknown" if t["estimated_rows"] is None else f"~{t['estimated_rows']} rows"
            print(f"  {t['schema']}.{t['name']:<40} {t['type']:<18} {rows}")

    elif args.command == "tools":
        registry = _registry()
        tools = registry.openai_tools() if args.format == "openai" else registry.anthropic_tools()
        print(json.dumps(tools, indent=2))

    elif args.command == "call":
        output, is_error = _registry().call_json(args.tool, json.loads(args.arguments))
        print(json.dumps(json.loads(output), indent=2) if not is_error else output)
        sys.exit(1 if is_error else 0)


if __name__ == "__main__":
    main()
