"""Postgres full-text-search adapter: the first EvidenceStore (ADR-0010).

Targets the pgvector deployment shape — chunk text in a table beside the
vectors. Recall is `ts_rank_cd` over `websearch_to_tsquery`, read-only by
session characteristic on top of whatever role the tenant granted, so a
misconfigured role still cannot write.
"""

from typing import Any

from engine.retrieval.store import Chunk, StoreError


class PostgresStore:
    def __init__(
        self,
        dsn: str,
        table: str,
        id_column: str,
        text_column: str,
        source_column: str | None = None,
        snapshot_column: str | None = None,
        regconfig: str = "simple",
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._id_column = id_column
        self._text_column = text_column
        self._source_column = source_column
        self._snapshot_column = snapshot_column
        self._regconfig = regconfig

    def _connect(self) -> Any:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover — psycopg is a core dependency
            raise StoreError("psycopg is not installed") from exc
        try:
            # Connection per call: the API serves from a threadpool and psycopg
            # connections are not safe for concurrent use. Read-only is a session
            # characteristic, not a role assumption.
            return psycopg.connect(
                self._dsn,
                autocommit=True,
                connect_timeout=5,
                options="-c default_transaction_read_only=on",
            )
        except psycopg.OperationalError as exc:
            raise StoreError(f"cannot connect to the tenant store: {exc}") from exc

    def probe(self) -> None:
        """Startup validation: the store is reachable and the configured table
        and columns exist. Fails loud so a misconfigured deployment never serves."""
        select = self._select_sql()
        try:
            with self._connect() as conn:
                conn.execute(f"{select} LIMIT 0")
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(
                f"tenant store probe failed for table {self._table!r} "
                f"(id={self._id_column!r}, text={self._text_column!r}, "
                f"source={self._source_column!r}, snapshot={self._snapshot_column!r}): {exc}"
            ) from exc

    def recall(self, query: str, k: int) -> list[Chunk]:
        import psycopg

        # websearch_to_tsquery ANDs terms, which would demand every content
        # word of an expanded multi-sentence query in one chunk — the opposite
        # of a recall stage. OR-join the terms instead: bag-of-words recall,
        # ts_rank_cd still ranks denser matches higher, and the reranker
        # restores precision (ADR-0002).
        or_query = " OR ".join(query.split())
        sql = (
            f"SELECT {self._column_sql()}, "
            f"ts_rank_cd(to_tsvector(%(reg)s::regconfig, {self._quoted(self._text_column)}), q) "
            f"FROM {self._quoted(self._table)}, "
            f"websearch_to_tsquery(%(reg)s::regconfig, %(query)s) q "
            f"WHERE to_tsvector(%(reg)s::regconfig, {self._quoted(self._text_column)}) @@ q "
            f"ORDER BY 4 DESC, {self._quoted(self._id_column)} "
            f"LIMIT %(k)s"
        )
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    sql,
                    {"reg": self._regconfig, "query": or_query, "k": k},
                ).fetchall()
        except StoreError:
            raise
        except psycopg.Error as exc:
            raise StoreError(
                f"tenant store recall failed against table {self._table!r}: {exc}"
            ) from exc
        return [self._to_chunk(row) for row in rows]

    def _column_sql(self) -> str:
        source = self._quoted(self._source_column) if self._source_column else f"'{self._table}'"
        snapshot = self._quoted(self._snapshot_column) if self._snapshot_column else "NULL"
        return (
            f"{self._quoted(self._id_column)}, {self._quoted(self._text_column)}, "
            f"{source}, {snapshot}"
        )

    def _select_sql(self) -> str:
        return f"SELECT {self._column_sql()} FROM {self._quoted(self._table)}"

    @staticmethod
    def _quoted(identifier: str) -> str:
        # Identifiers can't be bound parameters; quote and escape them instead.
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _to_chunk(self, row: tuple[Any, ...]) -> Chunk:
        chunk_id, text, source_id, snapshot_id = row[0], row[1], row[2], row[3]
        return Chunk(
            text=str(text),
            source_id=str(source_id),
            chunk_id=str(chunk_id),
            snapshot_id=None if snapshot_id is None else str(snapshot_id),
        )
