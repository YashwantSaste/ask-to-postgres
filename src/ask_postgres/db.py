"""PostgreSQL access: schema introspection and guarded read-only query execution."""

from __future__ import annotations

import datetime as dt
import decimal
import time
import uuid
from typing import Any

from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from ask_postgres.config import Settings
from ask_postgres.sql_guard import validate_select

_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")

# Row estimate from planner statistics. reltuples = -1 means statistics were never collected (common on
# databases where ANALYZE hasn't run): then use the live-tuple counter, 0 if the table occupies no space,
# and otherwise NULL = unknown - never a misleading 0.
_ROW_ESTIMATE = """
    CASE WHEN c.reltuples >= 0 THEN c.reltuples::bigint
         WHEN coalesce(pg_stat_get_live_tuples(c.oid), 0) > 0 THEN pg_stat_get_live_tuples(c.oid)
         WHEN c.relkind IN ('r','m') AND pg_relation_size(c.oid) = 0 THEN 0
    END"""

# describe_table counts rows exactly when the estimate is unknown and the table is small enough to be cheap.
_EXACT_COUNT_MAX_BYTES = 64 * 1024 * 1024


def _jsonable(value: Any) -> Any:
    """Convert PostgreSQL values into something JSON can carry without losing meaning."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, decimal.Decimal):
        return int(value) if value == value.to_integral_value() and abs(value) < 2**53 else float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


class Database:
    """Thin wrapper around a psycopg connection pool. Every statement runs in a READ ONLY transaction."""

    def __init__(self, settings: Settings):
        self.settings = settings
        options = f"-c statement_timeout={settings.statement_timeout_ms} -c default_transaction_read_only=on"
        if settings.schemas:
            options += " -c search_path=" + ",".join(settings.schemas)
        self.pool = ConnectionPool(
            settings.database_url,
            min_size=settings.pool_min_size,
            max_size=settings.pool_max_size,
            kwargs={"options": options, "application_name": "ask-postgres"},
            open=True,
        )
        try:
            self.pool.wait(timeout=15)
        except PoolTimeout as e:
            self.pool.close()
            raise ConnectionError(
                "Could not connect to PostgreSQL - check PGQA_DATABASE_URL, credentials and network access."
            ) from e

    def close(self) -> None:
        self.pool.close()

    # ------------------------------------------------------------------ core

    def _fetch(self, query: str | sql.Composable, params: Any = None, limit: int | None = None) -> dict:
        start = time.perf_counter()
        with self.pool.connection() as conn:
            conn.read_only = True
            try:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(query, params)
                    if cur.description is None:
                        return {"columns": [], "rows": [], "row_count": 0, "truncated": False}
                    columns = [c.name for c in cur.description]
                    rows = cur.fetchmany(limit + 1) if limit else cur.fetchall()
            finally:
                conn.rollback()
        truncated = bool(limit) and len(rows) > limit
        rows = rows[:limit] if limit else rows
        return {
            "columns": columns,
            "rows": [{k: _jsonable(v) for k, v in r.items()} for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def _schema_filter(self, column: str) -> tuple[sql.Composable, list[Any]]:
        if self.settings.schemas:
            return sql.SQL("{} = ANY(%s)").format(sql.Identifier(*column.split("."))), [self.settings.schemas]
        return sql.SQL("{} <> ALL(%s)").format(sql.Identifier(*column.split("."))), [list(_SYSTEM_SCHEMAS)]

    def _check_schema(self, schema: str) -> None:
        if self.settings.schemas and schema not in self.settings.schemas:
            raise PermissionError(f"Schema '{schema}' is not accessible. Allowed: {', '.join(self.settings.schemas)}.")
        if schema in _SYSTEM_SCHEMAS:
            raise PermissionError(f"System schema '{schema}' is not exposed.")

    def _resolve_table(self, table: str, schema: str | None) -> tuple[str, str]:
        """Accept 'schema.table' or a bare table name and return the (schema, table) that actually exists."""
        if schema is None and "." in table:
            schema, table = table.split(".", 1)
        where, params = self._schema_filter("n.nspname")
        query = sql.SQL(
            """
            SELECT n.nspname AS schema, c.relname AS name
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind IN ('r','v','m','p','f') AND {where}
              AND lower(c.relname) = lower(%s) {schema_clause}
            ORDER BY (n.nspname = 'public') DESC, n.nspname
            """
        ).format(
            where=where,
            schema_clause=sql.SQL("AND lower(n.nspname) = lower(%s)") if schema else sql.SQL(""),
        )
        params = [*params, table] + ([schema] if schema else [])
        rows = self._fetch(query, params)["rows"]
        if not rows:
            raise LookupError(f"Table '{f'{schema}.' if schema else ''}{table}' not found (or not accessible).")
        if len(rows) > 1 and not schema:
            options = ", ".join(f"{r['schema']}.{r['name']}" for r in rows)
            raise LookupError(f"Table name '{table}' is ambiguous; specify a schema: {options}.")
        self._check_schema(rows[0]["schema"])
        return rows[0]["schema"], rows[0]["name"]

    # --------------------------------------------------------- introspection

    def server_info(self) -> dict:
        info = self._fetch(
            "SELECT current_database() AS database, current_user AS user, "
            "split_part(version(), ',', 1) AS version, current_setting('TimeZone') AS timezone"
        )["rows"][0]
        info["max_rows_per_query"] = self.settings.max_rows
        info["statement_timeout_ms"] = self.settings.statement_timeout_ms
        info["schemas"] = self.list_schemas()
        return info

    def list_schemas(self) -> list[dict]:
        where, params = self._schema_filter("n.nspname")
        query = sql.SQL(
            """
            SELECT n.nspname AS schema,
                   obj_description(n.oid, 'pg_namespace') AS comment,
                   count(c.oid) FILTER (WHERE c.relkind IN ('r','p')) AS tables,
                   count(c.oid) FILTER (WHERE c.relkind IN ('v','m')) AS views
            FROM pg_namespace n
            LEFT JOIN pg_class c ON c.relnamespace = n.oid
            WHERE {where} AND n.nspname NOT LIKE 'pg\\_temp%%' AND n.nspname NOT LIKE 'pg\\_toast%%'
            GROUP BY n.oid, n.nspname
            ORDER BY n.nspname
            """
        ).format(where=where)
        return self._fetch(query, params)["rows"]

    def list_tables(self, schema: str | None = None) -> list[dict]:
        where, params = self._schema_filter("n.nspname")
        if schema:
            self._check_schema(schema)
        query = sql.SQL(
            """
            SELECT n.nspname AS schema, c.relname AS name,
                   CASE c.relkind WHEN 'r' THEN 'table' WHEN 'p' THEN 'table' WHEN 'v' THEN 'view'
                                  WHEN 'm' THEN 'materialized view' WHEN 'f' THEN 'foreign table' END AS type,
                   {row_estimate} AS estimated_rows,
                   obj_description(c.oid, 'pg_class') AS comment
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind IN ('r','v','m','p','f') AND NOT c.relispartition AND {where} {schema_clause}
            ORDER BY n.nspname, c.relname
            """
        ).format(where=where, schema_clause=sql.SQL("AND n.nspname = %s") if schema else sql.SQL(""),
                 row_estimate=sql.SQL(_ROW_ESTIMATE))
        return self._fetch(query, [*params, schema] if schema else params)["rows"]

    def describe_table(self, table: str, schema: str | None = None) -> dict:
        schema, table = self._resolve_table(table, schema)
        columns = self._fetch(
            """
            SELECT a.attname AS name,
                   format_type(a.atttypid, a.atttypmod) AS type,
                   NOT a.attnotnull AS nullable,
                   pg_get_expr(d.adbin, d.adrelid) AS default,
                   col_description(a.attrelid, a.attnum) AS comment
            FROM pg_attribute a
            LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
            WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            [self._regclass(schema, table)],
        )["rows"]
        constraints = self._fetch(
            """
            SELECT con.conname AS name, con.contype AS kind, pg_get_constraintdef(con.oid) AS definition
            FROM pg_constraint con
            WHERE con.conrelid = %s::regclass AND con.contype IN ('p','u','f','c')
            ORDER BY con.contype, con.conname
            """,
            [self._regclass(schema, table)],
        )["rows"]
        referenced_by = self._fetch(
            """
            SELECT con.conrelid::regclass::text AS from_table, pg_get_constraintdef(con.oid) AS definition
            FROM pg_constraint con
            WHERE con.confrelid = %s::regclass AND con.contype = 'f'
            ORDER BY 1
            """,
            [self._regclass(schema, table)],
        )["rows"]
        indexes = self._fetch(
            "SELECT indexname AS name, indexdef AS definition FROM pg_indexes "
            "WHERE schemaname = %s AND tablename = %s ORDER BY indexname",
            [schema, table],
        )["rows"]
        meta = self._fetch(
            sql.SQL(
                "SELECT obj_description(c.oid, 'pg_class') AS comment, {row_estimate} AS estimated_rows, "
                "c.relkind IN ('r','p','m') AND pg_total_relation_size(c.oid) <= %s AS cheap_to_count "
                "FROM pg_class c WHERE c.oid = %s::regclass"
            ).format(row_estimate=sql.SQL(_ROW_ESTIMATE)),
            [_EXACT_COUNT_MAX_BYTES, self._regclass(schema, table)],
        )["rows"][0]
        row_count_exact = None
        if meta["estimated_rows"] is None and meta["cheap_to_count"]:
            query = sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(schema, table))
            row_count_exact = self._fetch(query)["rows"][0]["n"]
        kinds = {"p": "primary_key", "u": "unique", "f": "foreign_key", "c": "check"}
        return {
            "schema": schema,
            "table": table,
            "comment": meta["comment"],
            **({"row_count": row_count_exact} if row_count_exact is not None
               else {"estimated_rows": meta["estimated_rows"]}),
            "columns": columns,
            "constraints": [{**c, "kind": kinds[c["kind"]]} for c in constraints],
            "referenced_by": referenced_by,
            "indexes": indexes,
        }

    def search_schema(self, keyword: str) -> list[dict]:
        where, params = self._schema_filter("c.table_schema")
        pattern = f"%{keyword.lower()}%"
        query = sql.SQL(
            """
            SELECT c.table_schema AS schema, c.table_name AS table, c.column_name AS column, c.data_type AS type,
                   col_description(format('%%I.%%I', c.table_schema, c.table_name)::regclass, c.ordinal_position) AS comment
            FROM information_schema.columns c
            WHERE {where}
              AND (lower(c.table_name) LIKE %s OR lower(c.column_name) LIKE %s
                   OR lower(coalesce(col_description(format('%%I.%%I', c.table_schema, c.table_name)::regclass,
                                                     c.ordinal_position), '')) LIKE %s
                   OR lower(coalesce(obj_description(format('%%I.%%I', c.table_schema, c.table_name)::regclass,
                                                     'pg_class'), '')) LIKE %s)
            ORDER BY 1, 2, c.ordinal_position
            LIMIT 200
            """
        ).format(where=where)
        return self._fetch(query, [*params, pattern, pattern, pattern, pattern])["rows"]

    def sample_rows(self, table: str, schema: str | None = None, limit: int | None = None) -> dict:
        schema, table = self._resolve_table(table, schema)
        n = max(1, min(limit or self.settings.sample_rows, 50))
        query = sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(schema, table))
        return self._fetch(query, [n])

    def column_values(self, table: str, column: str, schema: str | None = None, limit: int = 50) -> dict:
        """Most frequent distinct values of a column - ideal for learning valid filter values."""
        schema, table = self._resolve_table(table, schema)
        n = max(1, min(limit, 500))
        query = sql.SQL(
            "SELECT {col} AS value, count(*) AS frequency FROM {tbl} GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT %s"
        ).format(col=sql.Identifier(column), tbl=sql.Identifier(schema, table))
        return self._fetch(query, [n])

    # ------------------------------------------------------------- querying

    def run_query(self, query: str, max_rows: int | None = None) -> dict:
        safe = validate_select(query, self.settings.schemas)
        limit = max(1, min(max_rows or self.settings.max_rows, self.settings.max_rows))
        result = self._fetch(safe, limit=limit)
        if result["truncated"]:
            result["note"] = (
                f"Result truncated to {limit} rows. Aggregate in SQL (COUNT, SUM, GROUP BY) "
                "rather than reading raw rows when you need totals."
            )
        return result

    def explain_query(self, query: str) -> dict:
        safe = validate_select(query, self.settings.schemas)
        plan = self._fetch("EXPLAIN (FORMAT TEXT) " + safe)
        return {"plan": "\n".join(r["QUERY PLAN"] for r in plan["rows"])}

    @staticmethod
    def _regclass(schema: str, table: str) -> str:
        return sql.Identifier(schema, table).as_string(None)
