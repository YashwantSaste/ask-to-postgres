import os
from dataclasses import replace

import pytest

from ask_postgres.config import Settings

# Integration tests expect the demo database (docker compose up -d), never the one configured in .env.
TEST_DATABASE_URL = os.getenv("PGQA_TEST_DATABASE_URL", "postgresql://ask_postgres_reader:reader@localhost:55432/shop")


@pytest.fixture(scope="session")
def settings():
    if os.getenv("PGQA_SKIP_DB_TESTS"):
        pytest.skip("PGQA_SKIP_DB_TESTS set")
    return replace(Settings(database_url=TEST_DATABASE_URL), statement_timeout_ms=5000)


@pytest.fixture(scope="session")
def registry(settings):
    from ask_postgres.db import Database
    from ask_postgres.tools import ToolRegistry

    try:
        db = Database(settings)
    except ConnectionError as e:
        pytest.skip(f"{e} (start the demo DB with `docker compose up -d` or set PGQA_TEST_DATABASE_URL)")
    yield ToolRegistry(db)
    db.close()
