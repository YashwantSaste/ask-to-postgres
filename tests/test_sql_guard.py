import pytest

from ask_postgres.sql_guard import UnsafeSQLError, validate_select

ALLOWED = [
    "SELECT 1",
    "select * from sales.orders;",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT a FROM t1 UNION ALL SELECT a FROM t2",
    "SELECT count(*) FILTER (WHERE status = 'shipped') FROM orders",
    "SELECT name, rank() OVER (ORDER BY price DESC) FROM products",
    "SELECT * FROM orders WHERE note LIKE '%drop table%'",
]

REJECTED = [
    ("", "Empty"),
    ("DELETE FROM orders", "Only SELECT"),
    ("UPDATE orders SET status = 'x'", "Only SELECT"),
    ("INSERT INTO orders VALUES (1)", "Only SELECT"),
    ("DROP TABLE orders", "Only SELECT"),
    ("TRUNCATE orders", "Only SELECT"),
    ("SELECT 1; DROP TABLE orders", "Exactly one"),
    ("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", "DELETE"),
    ("SELECT * INTO new_table FROM orders", "INTO"),
    ("SELECT pg_sleep(10)", "pg_sleep"),
    ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
    ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
    ("SELECT * FROM orders FOR UPDATE", "LOCK"),
    ("COPY orders TO '/tmp/x'", "Only SELECT"),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only_queries(sql):
    assert validate_select(sql)


@pytest.mark.parametrize(("sql", "message"), REJECTED)
def test_rejects_unsafe_queries(sql, message):
    with pytest.raises(UnsafeSQLError, match=message):
        validate_select(sql)


def test_schema_allowlist():
    assert validate_select("SELECT * FROM sales.orders", ["sales"])
    assert validate_select("SELECT * FROM orders", ["sales"])  # resolved via search_path
    with pytest.raises(UnsafeSQLError, match="not accessible"):
        validate_select("SELECT * FROM hr.salaries", ["sales"])
