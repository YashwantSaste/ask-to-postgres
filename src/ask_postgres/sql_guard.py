"""Static validation of agent-generated SQL.

This is the first of three layers of protection (the others are a READ ONLY
transaction with a statement timeout, and - recommended - a database role that
only has SELECT privileges). It rejects anything that is not a single
read-only query before the SQL ever reaches PostgreSQL.
"""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError


class UnsafeSQLError(ValueError):
    """Raised when a query is not allowed to run."""


# Statement / clause types that write data, change schema or session state.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
        "Command", "Into", "Set", "Grant", "Revoke", "Copy", "Use", "Transaction",
        "Commit", "Rollback", "Analyze", "Pragma", "Lock",
    )
    if hasattr(exp, name)
)

# Functions that read server files, sleep, touch other sessions or change settings.
_FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_ls_logdir", "pg_ls_waldir", "pg_ls_tmpdir", "pg_ls_archive_statusdir",
    "lo_import", "lo_export", "lo_get", "lo_put", "lo_from_bytea", "lo_unlink",
    "dblink", "dblink_exec", "dblink_connect", "dblink_send_query",
    "set_config", "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "pg_rotate_logfile", "pg_switch_wal", "pg_create_restore_point",
    "pg_advisory_lock", "pg_advisory_xact_lock", "pg_try_advisory_lock",
    "query_to_xml", "query_to_xml_and_xmlschema", "cursor_to_xml",
    "nextval", "setval",
}

_ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation) if hasattr(
    exp, "SetOperation"
) else (exp.Select, exp.Union)


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return node.sql_name().lower()


def validate_select(sql: str, allowed_schemas: list[str] | None = None) -> str:
    """Return the normalized query if it is a single read-only SELECT, else raise UnsafeSQLError."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise UnsafeSQLError("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(text, read="postgres") if s is not None]
    except ParseError as e:
        raise UnsafeSQLError(f"Could not parse SQL: {e}") from e

    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed per call.")
    tree = statements[0]

    if not isinstance(tree, _ALLOWED_ROOTS):
        raise UnsafeSQLError(
            f"Only SELECT queries (optionally with WITH / UNION) are allowed, got {tree.key.upper()}."
        )

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(f"{node.key.upper()} is not allowed in a read-only query.")
        if isinstance(node, exp.Func) and _function_name(node) in _FORBIDDEN_FUNCTIONS:
            raise UnsafeSQLError(f"Function {_function_name(node)}() is not allowed.")
        if allowed_schemas and isinstance(node, exp.Table):
            schema = node.db
            if schema and schema.lower() not in {s.lower() for s in allowed_schemas}:
                raise UnsafeSQLError(
                    f"Schema '{schema}' is not accessible. Allowed: {', '.join(allowed_schemas)}."
                )

    # Return the original text: re-generating SQL could subtly alter semantics.
    return text
