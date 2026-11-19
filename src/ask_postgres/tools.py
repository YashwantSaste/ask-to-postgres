"""Provider-neutral tool definitions.

Each tool is a plain typed Python function. From that single definition we derive:
  * MCP tools (see mcp_server.py),
  * Anthropic tool schemas  -> ToolRegistry.anthropic_tools()
  * OpenAI / function-calling schemas (also used by LangChain, LlamaIndex, Gemini, Ollama...)
    -> ToolRegistry.openai_tools()
  * a generic dispatcher    -> ToolRegistry.call(name, arguments)
"""


import inspect
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field, ValidationError, create_model

from ask_postgres.db import Database

SYSTEM_PROMPT = """\
You are a careful data analyst with read-only access to a PostgreSQL database through tools.
Answer the user's question using ONLY data you retrieve with the tools - never invent numbers.

How to work:
1. Orient yourself: call get_database_overview (once per conversation) to see schemas and tables,
   and search_schema when you need to find where a concept lives.
2. Call describe_table for every table you plan to query, so you use real column names, types and
   foreign keys for joins. Use column_values to learn the exact spelling of categorical values
   (statuses, countries, categories) before filtering on them.
3. Write a single PostgreSQL SELECT and call run_query. Prefer aggregation in SQL (COUNT, SUM, AVG,
   GROUP BY, window functions) over pulling raw rows. Qualify tables with their schema. Add ORDER BY
   and LIMIT for "top N" questions.
4. If a query errors or returns surprising results, read the error, inspect the schema again and
   retry. If the question is ambiguous, state the interpretation you chose.
5. If the data needed is not in the database, say so plainly.

Final answer: lead with the direct answer, then a short supporting table or bullet list if useful,
then the SQL you used in a ```sql block. Mention row truncation if a result was truncated."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    input_model: type[BaseModel]

    @property
    def input_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema["additionalProperties"] = False
        return schema


def _input_model(func: Callable[..., Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(func).parameters.items():
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (param.annotation, default)
    with warnings.catch_warnings():
        # A field called "schema" is natural here; pydantic warns because BaseModel has a deprecated .schema().
        warnings.filterwarnings("ignore", message='Field name "schema"')
        return create_model(f"{func.__name__}_input", **fields)


class ToolRegistry:
    """The tool set an agent uses to answer questions about a PostgreSQL database."""

    def __init__(self, db: Database):
        self.db = db
        self.tools: dict[str, Tool] = {}
        self._register_all()

    # -------------------------------------------------------------- tools

    def _register_all(self) -> None:
        db = self.db
        Schema = Annotated[str | None, Field(description="Schema name. Optional if the table name is unique.")]
        TableName = Annotated[str, Field(description="Table or view name, optionally schema-qualified (schema.table).")]

        def get_database_overview() -> dict:
            """Database name, PostgreSQL version, limits, plus every accessible schema and table/view with estimated
            row counts and comments. Call this first. estimated_rows is from planner statistics and may be stale;
            null means unknown (statistics never collected), NOT empty."""
            info = db.server_info()
            info["tables"] = db.list_tables()
            return info

        def list_tables(
            schema: Annotated[str | None, Field(description="Only list objects in this schema.")] = None,
        ) -> list[dict]:
            """List tables, views and materialized views with estimated row counts (null = unknown, not empty) and
            comments."""
            return db.list_tables(schema)

        def describe_table(table: TableName, schema: Schema = None) -> dict:
            """Columns (name, type, nullability, default, comment), primary/unique/foreign keys, check constraints,
            tables that reference this one, and indexes. Use before writing SQL against a table."""
            return db.describe_table(table, schema)

        def search_schema(
            keyword: Annotated[str, Field(description="Case-insensitive substring, e.g. 'customer' or 'revenue'.")],
        ) -> list[dict]:
            """Find columns whose table name, column name or comment contains the keyword."""
            return db.search_schema(keyword)

        def sample_rows(
            table: TableName,
            schema: Schema = None,
            limit: Annotated[int, Field(ge=1, le=50, description="Rows to return (max 50).")] = 5,
        ) -> dict:
            """A few example rows from a table, to understand value formats."""
            return db.sample_rows(table, schema, limit)

        def column_values(
            table: TableName,
            column: Annotated[str, Field(description="Column name.")],
            schema: Schema = None,
            limit: Annotated[int, Field(ge=1, le=500, description="Max distinct values.")] = 50,
        ) -> dict:
            """Most frequent distinct values of a column with their counts. Use it to get exact spellings of
            categorical values before filtering."""
            return db.column_values(table, column, schema, limit)

        def run_query(
            sql: Annotated[str, Field(description="A single read-only PostgreSQL SELECT (WITH/UNION allowed).")],
            max_rows: Annotated[int | None, Field(ge=1, description="Optional lower row cap.")] = None,
        ) -> dict:
            """Execute one read-only SELECT and return columns and rows as JSON. Writes, DDL and multiple statements
            are rejected; results are capped and a statement timeout applies."""
            return db.run_query(sql, max_rows)

        def explain_query(
            sql: Annotated[str, Field(description="A single read-only PostgreSQL SELECT.")],
        ) -> dict:
            """Show PostgreSQL's execution plan for a SELECT without running it. Useful for checking expensive
            queries on large tables."""
            return db.explain_query(sql)

        funcs: list[Callable[..., Any]] = [
            get_database_overview, list_tables, describe_table, search_schema,
            run_query, explain_query,
        ]
        if db.settings.allow_sampling:
            funcs[4:4] = [sample_rows, column_values]
        for f in funcs:
            self.tools[f.__name__] = Tool(
                name=f.__name__,
                description=inspect.cleandoc(f.__doc__ or ""),
                func=f,
                input_model=_input_model(f),
            )

    # ------------------------------------------------------------ dispatch

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Validate arguments and run a tool. Raises on unknown tool / invalid args / DB errors."""
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool '{name}'. Available: {', '.join(self.tools)}")
        args = tool.input_model.model_validate(arguments or {})
        return tool.func(**dict(args))

    def call_json(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[str, bool]:
        """Run a tool and return (json_text, is_error) - the shape every LLM tool-result API wants.
        Errors are returned as text so the model can read them and self-correct."""
        try:
            return json.dumps(self.call(name, arguments), ensure_ascii=False, default=str), False
        except ValidationError as e:
            return f"Invalid arguments for {name}: {e}", True
        except Exception as e:  # DB errors, guard rejections, lookups: all useful feedback for the model
            return f"{type(e).__name__}: {e}", True

    # ------------------------------------------------------------ schemas

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in self.tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in self.tools.values()]
