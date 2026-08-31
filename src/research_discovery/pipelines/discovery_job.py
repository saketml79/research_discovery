"""Stage 0: scheduled discovery sweep.

Runs every standing query against the metadata APIs, records new candidates, and
approves the fetchable ones for provisional ingestion. This is how the corpus
stays current between questions — the counterpart to the live, in-question
``discover_sources`` tool.

Nothing here produces knowledge. It produces *candidates*, which the ingest job
turns into unreviewed claims, which a human turns into findings.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from ..config import Config
from ..discovery.providers import (
    ArxivProvider,
    OpenAlexProvider,
    RssProvider,
    SemanticScholarProvider,
)
from ..discovery.service import DiscoveryService, IngestionSpeed, StandingQuery, sweep
from ..ids import stable_id
from ..ingest.http import UrlLibFetcher
from ..io import delta
from ..models import utcnow
from .common import RunLog, configure_logging, get_spark, parse_args, stage

logger = logging.getLogger(__name__)


class _TextTransport:
    """Adapts the project's HTTP fetcher to the providers' text interface."""

    def __init__(self, fetcher: Any) -> None:
        self._fetcher = fetcher

    def get_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        return self._fetcher.get(url).content.decode("utf-8", errors="replace")


def build_service(config: Config, transport: Any | None = None) -> DiscoveryService:
    """Construct the discovery service from configuration.

    Providers that need no credential are always on. Semantic Scholar is added
    only when a key is configured, since its unauthenticated rate limit is too
    low to be useful in a sweep.
    """
    active = transport or _TextTransport(UrlLibFetcher())
    contact = config.extra.get("contact_email", "") or "data-platform@example.org"

    providers: list[Any] = [
        OpenAlexProvider(active, contact_email=contact),
        ArxivProvider(active, contact_email=contact),
    ]
    api_key = config.extra.get("semantic_scholar_key", "")
    if api_key:
        providers.append(SemanticScholarProvider(active, api_key=api_key))
    feeds = [f for f in config.extra.get("rss_feeds", "").split(",") if f.strip()]
    if feeds:
        providers.append(RssProvider(active, feeds))

    logger.info("discovery providers: %s", ", ".join(p.name for p in providers))
    return DiscoveryService(providers)


def load_standing_queries(spark: Any, config: Config) -> list[StandingQuery]:
    """Read enabled standing queries."""
    rows = spark.sql(
        f"SELECT * FROM {config.table('research_standing_query')} WHERE enabled"
    ).collect()
    return [
        StandingQuery(
            query_text=r["query_text"],
            topic=r["topic"],
            enabled=True,
            recency_months=r["recency_months"],
            max_results=r["max_results"],
            created_by=r["created_by"],
            query_id=r["query_id"],
        )
        for r in rows
    ]


def run(
    config: Config,
    service: DiscoveryService,
    queries: Sequence[StandingQuery],
    *,
    known_urls: Sequence[str] = (),
    spark: Any = None,
    auto_approve: bool = True,
) -> RunLog:
    """Sweep every standing query and persist what it found."""
    with stage("DISCOVER", logger) as run_log:
        run_log.records_in = len(queries)
        candidate_rows: list[dict[str, Any]] = []
        run_rows: list[dict[str, Any]] = []

        def record(standing: StandingQuery, result: Any) -> None:
            for decision in result.decisions:
                hit = decision.candidate
                candidate_rows.append(
                    {
                        "candidate_id": hit.candidate_id,
                        "canonical_url": hit.canonical_url,
                        "title": hit.title,
                        "provider": hit.provider,
                        "external_id": hit.external_id,
                        "doi": hit.doi,
                        "source_type": hit.source_type.value,
                        "authors": hit.authors,
                        "venue": hit.venue,
                        "published_at": hit.published_at,
                        "abstract": (hit.abstract or "")[:4000] or None,
                        "citation_count": hit.citation_count,
                        "is_open_access": hit.is_open_access,
                        "pdf_url": hit.pdf_url,
                        "license": hit.license,
                        "fetchable": decision.fetchable,
                        "fetch_decision": decision.reason,
                        "relevance_score": hit.relevance_score,
                        "matched_query": standing.query_text,
                        "discovery_mode": "SCHEDULED_SWEEP",
                        "ingestion_speed": (
                            IngestionSpeed.PROVISIONAL.value
                            if (auto_approve and decision.fetchable)
                            else IngestionSpeed.METADATA_ONLY.value
                        ),
                        # A sweep may approve fetching, because a curator already
                        # approved the standing query. It can never approve a claim.
                        "status": "APPROVED" if (auto_approve and decision.fetchable) else "DISCOVERED",
                        "discovered_at": utcnow(),
                        "decided_by": "scheduled_sweep" if auto_approve else None,
                        "decided_at": utcnow() if auto_approve else None,
                    }
                )
            run_rows.append(
                {
                    "discovery_run_id": stable_id("drun", standing.query_id, utcnow().isoformat()),
                    "query_text": standing.query_text,
                    "query_id": standing.query_id,
                    "discovery_mode": "SCHEDULED_SWEEP",
                    "providers_searched": ",".join(
                        sorted({d.candidate.provider for d in result.decisions}) or ["none"]
                    ),
                    "provider_errors": json.dumps(result.provider_errors)
                    if result.provider_errors
                    else None,
                    "candidates_found": len(result.decisions),
                    "candidates_fetchable": len(result.fetchable),
                    "already_known": len(result.already_known),
                    "requested_by": "scheduled_sweep",
                    "started_at": result.searched_at,
                    "finished_at": utcnow(),
                }
            )
            if result.provider_errors:
                run_log.records_quarantined += len(result.provider_errors)

        sweep(service, queries, known_urls=known_urls, speed=IngestionSpeed.PROVISIONAL, on_result=record)
        run_log.records_out = len(candidate_rows)

        if spark is not None and not config.dry_run:
            if candidate_rows:
                candidate_table = config.table("research_source_candidate")
                candidate_schema = spark.table(candidate_table).schema
                spark.createDataFrame(candidate_rows, schema=candidate_schema).createOrReplaceTempView(
                    "_stage_candidates"
                )
                spark.sql(
                    f"""
                    MERGE INTO {candidate_table} AS t
                    USING _stage_candidates AS s ON t.candidate_id = s.candidate_id
                    WHEN NOT MATCHED THEN INSERT *
                    """
                )
            if run_rows:
                target_table = config.table("research_discovery_run")
                # provider_errors is frequently all-None across a batch (no
                # provider failures); createDataFrame can't infer a type for
                # an all-null column, so use the target Delta table's schema.
                target_schema = spark.table(target_table).schema
                spark.createDataFrame(run_rows, schema=target_schema).write.mode(
                    "append"
                ).saveAsTable(target_table)
            spark.sql(
                f"""
                UPDATE {config.table('research_standing_query')}
                SET last_run_at = current_timestamp()
                WHERE enabled
                """
            )
    return run_log


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)
    spark = get_spark()

    queries = load_standing_queries(spark, config)
    if not queries:
        logger.info("no enabled standing queries; nothing to sweep")
        return 0

    known = [
        r["canonical_url"]
        for r in spark.sql(
            f"SELECT canonical_url FROM {config.table('research_source')}"
        ).collect()
    ]
    run_log = run(config, build_service(config), queries, known_urls=known, spark=spark)
    delta.append(spark, config.table("pipeline_run"), [run_log], dry_run=config.dry_run)
    return 0 if run_log.status in {"SUCCEEDED", "PARTIAL"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
