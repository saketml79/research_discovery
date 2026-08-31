"""Auto Loader ingestion for PDFs dropped into a Unity Catalog volume.

The other intake path. Discovery finds work on the internet; this picks up files
a human put in a folder — a paper someone downloaded, a slide deck, a report with
no public URL.

Auto Loader with ``cloudFiles.format = binaryFile`` tracks discovered files in
its checkpoint, so re-running processes only what is new. The file's content hash
still decides whether a *version* is new, which means a file that is renamed or
re-uploaded unchanged produces no new version and no duplicate claims.

A volume file has no canonical URL, so it gets a stable ``file://`` identity
derived from its volume path. That keeps every downstream citation shaped the
same way while remaining honest that the source is local, not published.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..chunking import chunk_document
from ..config import Config
from ..ids import sha256_hex
from ..io import delta
from ..models import (
    Chunk,
    IngestionStatus,
    Source,
    SourceType,
    SourceVersion,
    utcnow,
)
from ..parsers.base import ParserError
from ..parsers.registry import resolve_with_fallback
from .common import RunLog, configure_logging, get_spark, parse_args, stage

logger = logging.getLogger(__name__)

CHECKPOINT_SUBDIR = "_checkpoints/volume_ingest"

#: Extensions Auto Loader picks up. Anything else in the volume is ignored
#: rather than guessed at.
INGESTIBLE_GLOB = "*.{pdf,PDF,html,htm,txt,md}"


def volume_source(path: str, *, volume_root: str, licence: str | None) -> Source:
    """Register a volume file as a corpus source.

    A local file is typed ``PRIMARY_PAPER`` only when a curator says so via the
    licence manifest; otherwise it is ``TALK_TRANSCRIPT``, the weakest tier that
    still carries provenance, because an unlabelled file on a share is not
    evidence of peer review.
    """
    relative = path.replace(volume_root, "").lstrip("/")
    return Source(
        canonical_url=f"https://local.volume/{relative}",
        source_type=SourceType.PRIMARY_PAPER if licence else SourceType.TALK_TRANSCRIPT,
        title=relative.rsplit("/", 1)[-1],
        publisher="local_volume",
        license=licence,
        storage_permitted=True,  # the file is already in the customer's own volume
        ingestion_status=IngestionStatus.DISCOVERED,
    )


def process_file(
    path: str, content: bytes, config: Config, *, licence: str | None = None
) -> tuple[Source, SourceVersion, list[Chunk]]:
    """Parse and chunk one volume file.

    Raises:
        ParserError: The file could not be parsed.
    """
    source = volume_source(path, volume_root=config.volume_path, licence=licence)
    source.advance(IngestionStatus.FETCHED)

    content_type = _content_type(path)
    version = SourceVersion(
        source_id=source.source_id,
        content_hash=sha256_hex(content),
        raw_content_uri=path,
        content_type=content_type,
        byte_size=len(content),
        retrieved_at=utcnow(),
        is_current=True,
    )

    parser, fallback_warning = resolve_with_fallback(config.parser, content_type)
    document = parser.parse(content, source_uri=path, content_type=content_type)
    if document.is_empty:
        raise ParserError(f"parser produced no text for {path}")
    source.advance(IngestionStatus.PARSED)

    chunks = chunk_document(
        document,
        source_version_id=version.source_version_id,
        source_id=source.source_id,
        config=config,
        storage_permitted=True,
        parser_warning=fallback_warning,
    )
    source.advance(IngestionStatus.CHUNKED)
    return source, version, chunks


def build_stream(spark: Any, config: Config) -> Any:
    """Build the Auto Loader stream over the raw-sources volume."""
    checkpoint = f"{config.volume_path}/{CHECKPOINT_SUBDIR}"
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.includeExistingFiles", "true")
        .option("pathGlobFilter", INGESTIBLE_GLOB)
        .option("cloudFiles.schemaLocation", f"{checkpoint}/schema")
        .load(config.volume_path)
    ), checkpoint


def run(config: Config, files: Sequence[tuple[str, bytes]], *, spark: Any = None) -> RunLog:
    """Process a batch of volume files.

    Takes an explicit list rather than reading the stream, so the same logic
    serves the Auto Loader ``foreachBatch`` and a unit test.
    """
    with stage("VOLUME_INGEST", logger) as run_log:
        run_log.records_in = len(files)
        sources: list[Source] = []
        versions: list[SourceVersion] = []
        chunks: list[Chunk] = []

        for path, content in files:
            try:
                source, version, file_chunks = process_file(path, content, config)
            except ParserError as exc:
                logger.warning("quarantining %s: %s", path, exc)
                quarantined = volume_source(path, volume_root=config.volume_path, licence=None)
                quarantined.advance(IngestionStatus.QUARANTINED)
                sources.append(quarantined)
                run_log.records_quarantined += 1
                continue
            sources.append(source)
            versions.append(version)
            chunks.extend(file_chunks)

        run_log.records_out = len(chunks)
        if spark is not None:
            delta.upsert(spark, config.table("research_source"), sources, dry_run=config.dry_run)
            delta.upsert(
                spark, config.table("research_source_version"), versions, dry_run=config.dry_run
            )
            delta.upsert(spark, config.table("research_chunk"), chunks, dry_run=config.dry_run)
    return run_log


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - streaming entry point
    """Job entry point. Runs one Auto Loader trigger-available-now pass."""
    config, args = parse_args(list(argv) if argv is not None else None)
    configure_logging(args.log_level)
    spark = get_spark()

    stream, checkpoint = build_stream(spark, config)

    def handle_batch(batch: Any, batch_id: int) -> None:
        rows = batch.select("path", "content").collect()
        logger.info("volume batch %d: %d file(s)", batch_id, len(rows))
        if not rows:
            return
        run_log = run(config, [(r["path"], bytes(r["content"])) for r in rows], spark=spark)
        delta.append(spark, config.table("pipeline_run"), [run_log], dry_run=config.dry_run)

    query = (
        stream.writeStream.foreachBatch(handle_batch)
        .option("checkpointLocation", checkpoint)
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    return 0


def _content_type(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".pdf"):
        return "application/pdf"
    if lowered.endswith((".html", ".htm")):
        return "text/html"
    return "text/plain"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
