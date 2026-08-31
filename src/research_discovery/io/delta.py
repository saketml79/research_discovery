"""Delta persistence.

Writes go through ``upsert``, which is a Delta ``MERGE`` on the record's primary
key. Because every id in this project is content-derived, re-running a stage
over unchanged input is a no-op rather than a duplicate - the property that
makes each pipeline stage safely retryable.

Spark is imported lazily so the rest of the package stays unit-testable off
cluster.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from ..models import to_row

logger = logging.getLogger(__name__)

#: Primary key column per table, used as the MERGE condition.
PRIMARY_KEYS: dict[str, str] = {
    "research_source": "source_id",
    "research_source_version": "source_version_id",
    "research_chunk": "chunk_id",
    "research_claim": "claim_id",
    "research_claim_relationship": "relationship_id",
    "research_review_queue": "review_id",
    "research_taxonomy": "term_id",
    "agent_proposal": "proposal_id",
    "pipeline_run": "run_id",
}

#: Columns never overwritten by a re-run: a human decision outranks a pipeline.
PRESERVE_ON_UPDATE: dict[str, frozenset[str]] = {
    "research_claim": frozenset(
        {"review_status", "reviewed_by", "reviewed_at", "review_note", "superseded_by_claim_id"}
    ),
    "research_claim_relationship": frozenset({"review_status", "reviewed_by", "reviewed_at"}),
    "research_review_queue": frozenset({"status", "assigned_to", "resolved_at", "resolution_note"}),
    "agent_proposal": frozenset({"status", "approved_by", "approved_at"}),
}


class WriteError(RuntimeError):
    """Raised when a write cannot be performed."""


def records_to_rows(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert dataclass records to plain dicts for the Spark writer."""
    return [to_row(record) for record in records]


def upsert(
    spark: Any,
    table: str,
    records: Sequence[Any],
    *,
    dry_run: bool = False,
) -> int:
    """MERGE ``records`` into ``table`` on its primary key.

    Args:
        spark: Active ``SparkSession``.
        table: Three-level table name.
        records: Dataclass records to write.
        dry_run: Log the planned write and return the row count without writing.

    Returns:
        Number of rows merged.

    Raises:
        WriteError: The table has no registered primary key.
    """
    if not records:
        return 0

    leaf = table.rsplit(".", 1)[-1]
    key = PRIMARY_KEYS.get(leaf)
    if key is None:
        raise WriteError(f"no primary key registered for table {table!r}")

    rows = records_to_rows(records)
    if dry_run:
        logger.info("dry run: would merge %d rows into %s on %s", len(rows), table, key)
        return len(rows)

    # An all-null column (e.g. an optional field unset across the whole batch)
    # can't be type-inferred by createDataFrame; the target table's own schema
    # is always authoritative, so use it instead of letting Spark guess.
    frame = spark.createDataFrame(rows, schema=spark.table(table).schema)
    view = f"_stage_{leaf}"
    frame.createOrReplaceTempView(view)

    preserved = PRESERVE_ON_UPDATE.get(leaf, frozenset())
    update_columns = [c for c in frame.columns if c != key and c not in preserved]
    set_clause = ", ".join(f"target.{c} = source.{c}" for c in update_columns)
    insert_columns = ", ".join(frame.columns)
    insert_values = ", ".join(f"source.{c}" for c in frame.columns)

    merge_sql = f"""
        MERGE INTO {table} AS target
        USING {view} AS source
          ON target.{key} = source.{key}
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_columns}) VALUES ({insert_values})
    """
    spark.sql(merge_sql)
    logger.info("merged %d rows into %s", len(rows), table)
    return len(rows)


def append(spark: Any, table: str, records: Sequence[Any], *, dry_run: bool = False) -> int:
    """Append records without a merge. Used only for the run log."""
    if not records:
        return 0
    rows = records_to_rows(records)
    if dry_run:
        logger.info("dry run: would append %d rows to %s", len(rows), table)
        return len(rows)
    spark.createDataFrame(rows, schema=spark.table(table).schema).write.mode("append").saveAsTable(table)
    return len(rows)


def read_table(spark: Any, table: str, *, where: str | None = None) -> Any:
    """Read a table as a DataFrame, optionally filtered."""
    frame = spark.table(table)
    return frame.where(where) if where else frame
