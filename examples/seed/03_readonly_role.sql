-- Recommended: give ask-postgres a role that can only read. This is the strongest guardrail.
CREATE ROLE ask_postgres_reader LOGIN PASSWORD 'reader';
GRANT CONNECT ON DATABASE shop TO ask_postgres_reader;
GRANT USAGE ON SCHEMA sales TO ask_postgres_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA sales TO ask_postgres_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA sales GRANT SELECT ON TABLES TO ask_postgres_reader;
ALTER ROLE ask_postgres_reader SET default_transaction_read_only = on;
