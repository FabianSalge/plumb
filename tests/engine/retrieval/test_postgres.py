"""Integration tests for the Postgres FTS adapter against a real server.

Marked `postgres`: excluded from `make test`, run via `make test-postgres`,
which needs Docker. The fixture starts a disposable postgres:16-alpine
container, seeds a chunk table, and tears it down.
"""

import socket
import subprocess
import time
import uuid

import pytest

from engine.retrieval import Chunk, StoreError
from engine.retrieval.postgres import PostgresStore

pytestmark = pytest.mark.postgres

ROWS = [
    (1, "The Eiffel Tower is 330 metres tall and stands in Paris.", "wiki/eiffel", "snap-1"),
    (2, "Gustave Eiffel's company built the tower for the 1889 fair.", "wiki/eiffel", "snap-1"),
    (3, "The Louvre is the world's most-visited museum.", "wiki/louvre", "snap-2"),
    (4, "Paris is the capital of France.", "wiki/paris", "snap-2"),
]


@pytest.fixture(scope="session")
def postgres_dsn():
    import psycopg

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    name = f"plumb-test-pg-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            "POSTGRES_PASSWORD=plumb",
            "-p",
            f"127.0.0.1:{port}:5432",
            "postgres:16-alpine",
        ],
        check=True,
        capture_output=True,
    )
    dsn = f"postgresql://postgres:plumb@127.0.0.1:{port}/postgres"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with psycopg.connect(dsn, connect_timeout=2):
                    break
            except psycopg.OperationalError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                "CREATE TABLE chunks ("
                "id integer PRIMARY KEY, body text NOT NULL, "
                "src text NOT NULL, snap text NOT NULL)"
            )
            conn.cursor().executemany(
                "INSERT INTO chunks (id, body, src, snap) VALUES (%s, %s, %s, %s)", ROWS
            )
        yield dsn
    finally:
        subprocess.run(["docker", "stop", name], check=False, capture_output=True)


def make_store(dsn: str, **overrides: object) -> PostgresStore:
    settings: dict[str, object] = {
        "dsn": dsn,
        "table": "chunks",
        "id_column": "id",
        "text_column": "body",
        "source_column": "src",
        "snapshot_column": "snap",
        "regconfig": "english",
    }
    settings.update(overrides)
    return PostgresStore(**settings)  # type: ignore[arg-type]


def test_recall_ranks_lexical_matches(postgres_dsn):
    store = make_store(postgres_dsn)
    chunks = store.recall("How tall is the Eiffel Tower?", k=10)
    assert chunks, "expected lexical matches for an on-corpus query"
    assert chunks[0].chunk_id == "1"
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert {chunk.chunk_id for chunk in chunks} <= {"1", "2"}


def test_recall_respects_k(postgres_dsn):
    store = make_store(postgres_dsn)
    assert len(store.recall("Eiffel tower Paris France capital museum", k=1)) == 1


def test_chunks_carry_configured_identity_columns(postgres_dsn):
    store = make_store(postgres_dsn)
    top = store.recall("most-visited museum Louvre", k=1)[0]
    assert top.source_id == "wiki/louvre"
    assert top.chunk_id == "3"
    assert top.snapshot_id == "snap-2"


def test_snapshot_identity_absent_when_unconfigured(postgres_dsn):
    store = make_store(postgres_dsn, snapshot_column=None)
    top = store.recall("most-visited museum Louvre", k=1)[0]
    assert top.snapshot_id is None


def test_source_defaults_to_table_when_unconfigured(postgres_dsn):
    store = make_store(postgres_dsn, source_column=None)
    top = store.recall("most-visited museum Louvre", k=1)[0]
    assert top.source_id == "chunks"


def test_regconfig_is_configurable(postgres_dsn):
    english = make_store(postgres_dsn, regconfig="english")
    simple = make_store(postgres_dsn, regconfig="simple")
    # 'english' stems visited/visiting to the same lexeme; 'simple' does not.
    assert english.recall("visiting museums", k=10)
    assert not simple.recall("visiting museums", k=10)


def test_session_is_read_only(postgres_dsn):
    import psycopg

    store = make_store(postgres_dsn)
    with store._connect() as conn, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        conn.execute("INSERT INTO chunks (id, body, src, snap) VALUES (99, 'x', 'y', 'z')")


def test_unreachable_store_fails_loudly():
    store = make_store("postgresql://postgres:wrong@127.0.0.1:1/postgres")
    with pytest.raises(StoreError, match="connect"):
        store.recall("anything", k=1)


def test_probe_validates_connection_and_schema(postgres_dsn):
    make_store(postgres_dsn).probe()
    with pytest.raises(StoreError, match="no_such_table"):
        make_store(postgres_dsn, table="no_such_table").probe()
    with pytest.raises(StoreError, match="no_such_column"):
        make_store(postgres_dsn, text_column="no_such_column").probe()


def test_recall_errors_name_the_store_problem(postgres_dsn):
    store = make_store(postgres_dsn, table="no_such_table")
    with pytest.raises(StoreError, match="no_such_table"):
        store.recall("anything", k=1)
