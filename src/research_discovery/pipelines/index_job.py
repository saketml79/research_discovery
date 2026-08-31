"""Stage 7: sync the reviewed chunk corpus to an AI Search Delta Sync index.

Two rules are enforced here:

* only chunks belonging to reviewed sources are indexed, so retrieval cannot
  surface a passage the review boundary has not cleared;
* the index is not a permission boundary. AI Search endpoint ACLs govern
  endpoint access but the index does not enforce row- or column-level UC
  permissions, so nothing sensitive goes into a broadly readable index and
  filters are applied at query time as defence in depth.

The job is a no-op when no endpoint is configured; the rest of the system works
without vector retrieval, which is the correct weekend default.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import Config
from ..io import delta
from .common import RunLog, configure_logging, get_spark, parse_args, stage

logger = logging.getLogger(__name__)

INDEX_SUFFIX = "research_chunk_index"

#: Columns exposed as index metadata filters. Keep this list tight: every
#: extra column is another thing retrieval can leak.
METADATA_COLUMNS: tuple[str, ...] = (
    "chunk_id",
    "source_id",
    "source_version_id",
    "page_number",
    "section_title",
    "parser_name",
    "extraction_warning",
)

SOURCE_VIEW = "v_indexable_chunk"


def create_indexable_view(spark: Any, config: Config) -> str:
    """Create the view the index syncs from: reviewed sources only."""
    view = config.table(SOURCE_VIEW)
    spark.sql(
        f"""
        CREATE OR REPLACE VIEW {view}
        COMMENT 'Chunks from reviewed sources, the only passages eligible for retrieval.'
        AS
        SELECT c.chunk_id, c.source_id, c.source_version_id, c.page_number,
               c.section_title, c.parser_name, c.extraction_warning, c.text
        FROM {config.table('research_chunk')} AS c
        JOIN {config.table('research_source')} AS s ON c.source_id = s.source_id
        JOIN {config.table('research_source_version')} AS v
          ON c.source_version_id = v.source_version_id AND v.is_current
        WHERE c.lifecycle_state IN ('CHUNKED', 'INDEXED')
          AND s.ingestion_status IN ('REVIEWED', 'INDEXED')
        """
    )
    return view


def sync_index(config: Config, client: Any, source_view: str) -> str:
    """Create or sync the Delta Sync index over ``source_view``.

    Args:
        config: Deployment configuration.
        client: A ``VectorSearchClient``-shaped object.
        source_view: Fully qualified view the index syncs from.

    Returns:
        The index name.
    """
    index_name = config.table(INDEX_SUFFIX)
    try:
        index = client.get_index(endpoint_name=config.ai_search_endpoint, index_name=index_name)
        index.sync()
        logger.info("synced existing index %s", index_name)
    except Exception:  # noqa: BLE001 - SDK raises a typed not-found we do not import
        client.create_delta_sync_index(
            endpoint_name=config.ai_search_endpoint,
            index_name=index_name,
            source_table_name=source_view,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="text",
            embedding_model_endpoint_name="databricks-gte-large-en",
            columns_to_sync=list(METADATA_COLUMNS) + ["text"],
        )
        logger.info("created index %s", index_name)
    return index_name


def run(config: Config, *, spark: Any, client: Any | None = None) -> RunLog:
    """Run the index stage, skipping cleanly when retrieval is not configured."""
    with stage("INDEX", logger) as run_log:
        if not config.ai_search_endpoint:
            logger.info("no AI Search endpoint configured; skipping index sync")
            run_log.status = "SUCCEEDED"
            return run_log

        view = create_indexable_view(spark, config)
        run_log.records_in = spark.table(view).count()

        if client is None:
            from databricks.vector_search.client import VectorSearchClient  # noqa: PLC0415

            client = VectorSearchClient()

        if config.dry_run:
            logger.info("dry run: would sync %s from %s", INDEX_SUFFIX, view)
        else:
            sync_index(config, client, view)
        run_log.records_out = run_log.records_in
    return run_log


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)
    spark = get_spark()
    run_log = run(config, spark=spark)
    delta.append(spark, config.table("pipeline_run"), [run_log], dry_run=config.dry_run)
    return 0 if run_log.status in {"SUCCEEDED", "PARTIAL"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
