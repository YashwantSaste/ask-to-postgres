"""Integration tests against the demo database (docker compose up -d)."""

import json

import pytest


def test_overview_lists_tables(registry):
    overview = registry.call("get_database_overview")
    names = {f"{t['schema']}.{t['name']}" for t in overview["tables"]}
    assert {"sales.orders", "sales.customers", "sales.order_totals"} <= names


def test_describe_table_resolves_unqualified_name(registry):
    desc = registry.call("describe_table", {"table": "order_items"})
    assert desc["schema"] == "sales"
    assert {c["name"] for c in desc["columns"]} == {"order_id", "product_id", "quantity", "unit_price"}
    assert any(c["kind"] == "foreign_key" for c in desc["constraints"])


def test_run_query_aggregates(registry):
    result = registry.call("run_query", {"sql": "SELECT count(*) AS n FROM sales.orders"})
    assert result["rows"] == [{"n": 600}]


def test_run_query_truncates(registry):
    result = registry.call("run_query", {"sql": "SELECT * FROM sales.order_items", "max_rows": 10})
    assert result["row_count"] == 10 and result["truncated"]


def test_errors_are_returned_as_text(registry):
    output, is_error = registry.call_json("run_query", {"sql": "SELECT nope FROM sales.orders"})
    assert is_error and "nope" in output
    output, is_error = registry.call_json("run_query", {"sql": "DELETE FROM sales.orders"})
    assert is_error and "UnsafeSQLError" in output


def test_transaction_is_read_only_even_if_guard_is_bypassed(registry):
    import psycopg

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        registry.db._fetch("CREATE TEMP TABLE t (x int)")


def test_schemas_are_json_serializable(registry):
    json.dumps(registry.openai_tools())
    tools = {t["name"]: t for t in registry.anthropic_tools()}
    assert tools["run_query"]["input_schema"]["required"] == ["sql"]


async def test_mcp_server_in_process(settings):
    from mcp import Client

    from ask_postgres.mcp_server import build_server

    async with Client(build_server(settings)) as client:
        tools = await client.list_tools()
        assert "run_query" in {t.name for t in tools.tools}
        result = await client.call_tool("run_query", {"sql": "SELECT count(*) AS n FROM sales.products"})
        assert not result.is_error
        assert json.loads(result.content[0].text)["rows"] == [{"n": 12}]
