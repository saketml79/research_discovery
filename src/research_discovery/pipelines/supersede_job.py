"""Stage 5b: queue reviewed knowledge for re-review after a source changes.

Runs after extraction. For every source whose current version is newer than the
version its reviewed claims were read from, it pairs the old reviewed claims
against the new version's candidates and queues each pairing for a human.

It deliberately supersedes nothing on its own. A changed source is a reason to
re-read, not a reason to retract: auto-superseding would let an edited preprint
silently withdraw a finding a human had accepted. The reviewer decides, and
``apply_supersession`` — called from the review app — is what actually retires
the old claim.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import Config
from ..io import delta
from ..models import Claim, ClaimType, ReviewItem, ReviewStatus
from ..review.supersession import build_supersession_queue, plan_supersession
from .common import RunLog, configure_logging, get_spark, parse_args, stage

logger = logging.getLogger(__name__)


def run(
    config: Config,
    pairs: Sequence[tuple[Sequence[Claim], Sequence[Claim]]],
    *,
    spark: Any = None,
) -> tuple[RunLog, list[ReviewItem]]:
    """Plan supersession for each (old reviewed claims, new candidates) pair."""
    with stage("SUPERSEDE", logger) as run_log:
        items: list[ReviewItem] = []
        for previous, replacements in pairs:
            run_log.records_in += len(previous)
            candidates = plan_supersession(previous, replacements)
            vanished = sum(1 for c in candidates if c.replacement_claim_id is None)
            if vanished:
                logger.warning(
                    "%d reviewed claim(s) have no counterpart in the new version", vanished
                )
                run_log.records_quarantined += vanished
            items.extend(build_supersession_queue(candidates))
        run_log.records_out = len(items)

        if spark is not None and items:
            delta.upsert(
                spark, config.table("research_review_queue"), items, dry_run=config.dry_run
            )
    return run_log, items


def _row_to_claim(row: Any, status: ReviewStatus) -> Claim:
    """Rebuild a claim record from a Spark row."""
    return Claim(
        source_version_id=row["source_version_id"],
        source_id=row["source_id"],
        claim_text=row["claim_text"],
        claim_type=ClaimType(row["claim_type"]),
        source_url=row["source_url"],
        extractor_name=row["extractor_name"],
        extractor_version=row["extractor_version"],
        chunk_id=row["chunk_id"],
        figure_id=row["figure_id"],
        task=row["task"],
        method=row["method"],
        metric=row["metric"],
        metric_value=row["metric_value"],
        metric_unit=row["metric_unit"],
        benchmark=row["benchmark"],
        condition_text=row["condition_text"],
        page_number=row["page_number"],
        extraction_confidence=row["extraction_confidence"],
        missing_field_reason=row["missing_field_reason"],
        review_status=status,
        reviewed_by=row["reviewed_by"],
        claim_id=row["claim_id"],
    )


def load_pairs(spark: Any, config: Config) -> list[tuple[list[Claim], list[Claim]]]:
    """Find sources with a newer version and pair their old and new claims."""
    affected = spark.sql(
        f"""
        SELECT DISTINCT c.source_id, c.source_version_id AS old_version, v.source_version_id AS new_version
        FROM {config.table('research_claim')} c
        JOIN {config.table('research_source_version')} v
          ON c.source_id = v.source_id AND v.is_current
        WHERE c.review_status = 'REVIEWED'
          AND c.superseded_by_claim_id IS NULL
          AND c.source_version_id <> v.source_version_id
        """
    ).collect()

    pairs: list[tuple[list[Claim], list[Claim]]] = []
    for row in affected:
        previous = [
            _row_to_claim(r, ReviewStatus.REVIEWED)
            for r in spark.sql(
                f"SELECT * FROM {config.table('research_claim')} "
                f"WHERE source_version_id = '{row['old_version']}' AND review_status = 'REVIEWED'"
            ).collect()
        ]
        replacements = [
            _row_to_claim(r, ReviewStatus.CANDIDATE)
            for r in spark.sql(
                f"SELECT * FROM {config.table('research_claim')} "
                f"WHERE source_version_id = '{row['new_version']}'"
            ).collect()
        ]
        if previous:
            pairs.append((previous, replacements))
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    """Job entry point."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)
    spark = get_spark()

    pairs = load_pairs(spark, config)
    if not pairs:
        logger.info("no sources changed since their claims were reviewed")
        return 0

    run_log, items = run(config, pairs, spark=spark)
    logger.info("queued %d claim(s) for re-review after source changes", len(items))
    delta.append(spark, config.table("pipeline_run"), [run_log], dry_run=config.dry_run)
    return 0 if run_log.status in {"SUCCEEDED", "PARTIAL"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
