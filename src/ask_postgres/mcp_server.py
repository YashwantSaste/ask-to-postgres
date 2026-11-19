"""MCP server: exposes the database tools to any MCP-capable agent
(Claude Desktop / Claude Code, Cursor, VS Code, Windsurf, LangChain/LlamaIndex MCP adapters, ...)."""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from ask_postgres.config import Settings
from ask_postgres.db import Database
from ask_postgres.tools import SYSTEM_PROMPT, ToolRegistry


def build_server(settings: Settings | None = None) -> MCPServer:
    settings = settings or Settings.from_env()
    registry = ToolRegistry(Database(settings))

    server = MCPServer(
        name="ask-postgres",
        title="PostgreSQL Q&A",
        description="Read-only, natural-language question answering over a PostgreSQL database.",
        instructions=SYSTEM_PROMPT,
    )

    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                                open_world_hint=False)
    for tool in registry.tools.values():
        server.add_tool(tool.func, name=tool.name, description=tool.description,
                        annotations=read_only, structured_output=False)

    @server.resource("ask-postgres://schema", name="schema", description="Overview of all accessible schemas and tables.",
                     mime_type="application/json")
    def schema_overview() -> str:
        return json.dumps(registry.call("get_database_overview"), indent=2, default=str)

    @server.prompt(name="ask_database", description="Answer a question using the data in the database.")
    def ask_database(question: str) -> str:
        return f"{SYSTEM_PROMPT}\n\nQuestion: {question}"

    return server


def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    server = build_server()
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=host, port=port)
