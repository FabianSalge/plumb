"""Seed a local Postgres with the RAGTruth test corpus as a tenant chunk table.

The thorough-mode bench retrieves against this store: one row per distinct
source context, so retrieval must find the right document among all of
RAGTruth's — a real (if small) corpus, not a per-request oracle.

Usage: start a disposable Postgres, then from evals/:
    docker run -d --rm --name plumb-bench-pg -e POSTGRES_PASSWORD=plumb \
        -p 127.0.0.1:5433:5432 postgres:16-alpine
    uv run python -m bench.store_seed --dsn postgresql://postgres:plumb@127.0.0.1:5433/postgres
"""

import argparse

from bench.data import load_ragtruth_test

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
  id integer PRIMARY KEY,
  body text NOT NULL,
  src text NOT NULL,
  snap text NOT NULL
);
TRUNCATE chunks;
"""


def main() -> None:
    import psycopg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    args = parser.parse_args()

    examples = load_ragtruth_test()
    # RAGTruth reuses source documents across responses; the store holds each once.
    bodies: dict[str, str] = {}
    for example in examples:
        bodies.setdefault(example.context, f"ragtruth/{example.id}")

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute(TABLE_SQL)
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (id, body, src, snap) VALUES (%s, %s, %s, %s)",
                [
                    (i, body, src, "ragtruth-test-v1")
                    for i, (body, src) in enumerate(bodies.items(), start=1)
                ],
            )
    print(f"seeded {len(bodies)} distinct contexts from {len(examples)} responses")


if __name__ == "__main__":
    main()
