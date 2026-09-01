"""Source acquisition and registration.

Fetching is policy-bearing, so the rules live in code rather than in a runbook:

* only allowlisted hosts are fetched;
* ``robots.txt`` is honoured;
* requests are rate-limited per host;
* a source whose licence does not permit storage is stored as metadata plus a
  short excerpt, never as a copied corpus;
* an unchanged ``content_hash`` is recognised and skipped rather than
  reprocessed.

The HTTP transport is injected, so the pipeline is testable without network
access and a workspace can substitute an approved MCP fetch tool.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol, Sequence
from urllib.parse import urlparse

from ..ids import normalize_url, sha256_hex
from ..models import IngestionStatus, Source, SourceType, SourceVersion, utcnow

logger = logging.getLogger(__name__)

#: Hosts this pipeline may fetch from. Extend deliberately, in review - not at
#: runtime from a config string a user supplied.
DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "arxiv.org",
        "www.arxiv.org",
        "aclanthology.org",
        "openreview.net",
        "github.com",
        "raw.githubusercontent.com",
        "microsoft.github.io",
        "docs.databricks.com",
    }
)

#: Minimum seconds between requests to one host.
DEFAULT_MIN_INTERVAL_SECONDS = 1.0

#: Licences under which full parsed text may be stored.
STORAGE_PERMITTED_LICENCES: frozenset[str] = frozenset(
    {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "MIT", "APACHE-2.0", "ARXIV-NONEXCLUSIVE"}
)


class FetchError(RuntimeError):
    """Raised when a source cannot be fetched or is not permitted."""


class PolicyError(FetchError):
    """Raised when a fetch is refused by allowlist, robots or licence policy."""


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Bytes plus the provenance fields a source version records."""

    content: bytes
    content_type: str
    http_status: int
    etag: str | None = None


class HttpFetcher(Protocol):
    """Minimal fetch interface. An approved MCP tool can implement this."""

    def get(self, url: str, *, etag: str | None = None) -> FetchResult:
        """Fetch ``url``, optionally conditionally on ``etag``."""


class PolicyAwareFetcher:
    """Wraps a transport with allowlist, robots and rate-limit enforcement."""

    def __init__(
        self,
        transport: HttpFetcher,
        *,
        allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
        robots_check: Callable[[str], bool] | None = None,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._allowed = frozenset(h.lower() for h in allowed_hosts)
        self._robots_check = robots_check
        self._min_interval = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_request: dict[str, float] = {}

    def get(self, url: str, *, etag: str | None = None) -> FetchResult:
        host = (urlparse(url).hostname or "").lower()
        if host not in self._allowed:
            raise PolicyError(
                f"host {host!r} is not on the fetch allowlist; add it in review, not at runtime"
            )
        if self._robots_check is not None and not self._robots_check(url):
            raise PolicyError(f"robots.txt disallows fetching {url}")
        self._throttle(host)
        return self._transport.get(url, etag=etag)

    def _throttle(self, host: str) -> None:
        last = self._last_request.get(host)
        now = self._clock()
        if last is not None:
            wait = self._min_interval - (now - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()


def licence_permits_storage(licence: str | None) -> bool:
    """True when ``licence`` allows storing full parsed text.

    Unknown or absent licences are treated as not permitting storage. The
    default is the conservative one.
    """
    return bool(licence) and licence.strip().upper() in STORAGE_PERMITTED_LICENCES


def resolve_fetch_url(canonical_url: str) -> str:
    """The URL to actually download bytes from, when it differs from the
    citation URL a human would follow.

    An arXiv "/abs/<id>" page is the citation landing page - fetching it
    returns an HTML abstract, not the paper, and every downstream parser then
    silently falls back to parsing that HTML instead of the real PDF. Its
    "/pdf/<id>" sibling returns the actual paper. Every other host's
    canonical_url already is the fetchable document (a repo page, a docs
    page), so it is returned unchanged.
    """
    parsed = urlparse(canonical_url)
    host = (parsed.hostname or "").lower()
    if host in {"arxiv.org", "www.arxiv.org"} and parsed.path.startswith("/abs/"):
        arxiv_id = parsed.path[len("/abs/"):]
        return f"{parsed.scheme}://{parsed.netloc}/pdf/{arxiv_id}"
    return canonical_url


def register_source(
    canonical_url: str,
    *,
    source_type: str | SourceType,
    title: str | None = None,
    publisher: str | None = None,
    authors: str | None = None,
    licence: str | None = None,
) -> Source:
    """Create a ``DISCOVERED`` source record from curated metadata."""
    return Source(
        canonical_url=normalize_url(canonical_url),
        source_type=SourceType(source_type),
        title=title,
        publisher=publisher,
        authors=authors,
        license=licence,
        storage_permitted=licence_permits_storage(licence),
        ingestion_status=IngestionStatus.DISCOVERED,
    )


def load_seed_sources(path: Path) -> list[Source]:
    """Load the curated seed corpus from a CSV manifest.

    Required columns: ``canonical_url``, ``source_type``, ``title``,
    ``publisher``, ``license``. A malformed row fails loudly with its line
    number - a silently skipped source is a silently missing citation.
    """
    sources: list[Source] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                sources.append(
                    register_source(
                        row["canonical_url"],
                        source_type=row["source_type"],
                        title=row.get("title") or None,
                        publisher=row.get("publisher") or None,
                        authors=row.get("authors") or None,
                        licence=row.get("license") or None,
                    )
                )
            except (KeyError, ValueError) as exc:
                raise FetchError(f"{path}:{line_number}: invalid seed row: {exc}") from exc
    return sources


def fetch_version(
    source: Source,
    fetcher: HttpFetcher,
    *,
    previous: SourceVersion | None = None,
    volume_path: str | None = None,
) -> tuple[SourceVersion | None, bytes | None]:
    """Fetch a source and build its version record.

    Args:
        source: The registered source.
        fetcher: Policy-aware transport.
        previous: The current version, when one exists. Used both for
            conditional fetching and for change detection.
        volume_path: Volume root where raw bytes are stored when the licence
            permits it.

    Returns:
        A ``(version, content)`` pair. ``version`` is ``None`` when the content
        is unchanged (304, or an identical hash), in which case ``content`` is
        also ``None`` since nothing downstream needs it. Callers must not
        re-fetch the URL themselves - one source, one fetch.

    Raises:
        FetchError: The fetch failed or was refused by policy.
    """
    fetch_url = resolve_fetch_url(source.canonical_url)
    result = fetcher.get(fetch_url, etag=previous.etag if previous else None)

    if result.http_status == 304:
        logger.info("source %s unchanged (304)", source.canonical_url)
        return None, None
    if result.http_status >= 400:
        raise FetchError(f"HTTP {result.http_status} fetching {fetch_url}")

    content_hash = sha256_hex(result.content)
    if previous is not None and previous.content_hash == content_hash:
        logger.info("source %s unchanged (identical hash)", source.canonical_url)
        return None, None

    raw_uri = None
    if source.storage_permitted and volume_path:
        raw_uri = f"{volume_path}/{source.source_id}/{content_hash[:16]}"

    version = SourceVersion(
        source_id=source.source_id,
        content_hash=content_hash,
        version_number=(previous.version_number + 1) if previous else 1,
        raw_content_uri=raw_uri,
        content_type=result.content_type,
        byte_size=len(result.content),
        http_status=result.http_status,
        etag=result.etag,
        retrieved_at=utcnow(),
        is_current=True,
    )
    return version, result.content


def supersede_versions(versions: Sequence[SourceVersion], current_id: str) -> Iterator[SourceVersion]:
    """Yield versions with ``is_current`` set only on ``current_id``."""
    for version in versions:
        version.is_current = version.source_version_id == current_id
        yield version
