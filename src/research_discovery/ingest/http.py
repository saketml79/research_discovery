"""Standard-library HTTP transport.

Deliberately minimal. It exists so the pipeline has a working default; a
workspace with an approved MCP fetch tool or a licensed content API should
implement ``HttpFetcher`` against that instead and inject it.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

from .sources import FetchError, FetchResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = "ResearchDiscoveryBot/1.0 (+governed corpus ingestion; contact: data-platform)"
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


class UrlLibFetcher:
    """``HttpFetcher`` over ``urllib`` with a size cap and conditional GET."""

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> None:
        self._timeout = timeout
        self._user_agent = user_agent
        self._max_bytes = max_bytes

    def get(self, url: str, *, etag: str | None = None) -> FetchResult:
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        if etag:
            request.add_header("If-None-Match", etag)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                # read one byte past the cap so an oversized body is detected
                # rather than silently truncated into the corpus.
                content = response.read(self._max_bytes + 1)
                if len(content) > self._max_bytes:
                    raise FetchError(f"{url} exceeds the {self._max_bytes} byte download cap")
                return FetchResult(
                    content=content,
                    content_type=response.headers.get("Content-Type", "application/octet-stream"),
                    http_status=response.status,
                    etag=response.headers.get("ETag"),
                )
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return FetchResult(b"", "", 304, etag)
            raise FetchError(f"HTTP {exc.code} fetching {url}") from exc
        except urllib.error.URLError as exc:
            raise FetchError(f"could not reach {url}: {exc.reason}") from exc


def robots_allows(url: str, *, user_agent: str = DEFAULT_USER_AGENT) -> bool:
    """Check ``robots.txt`` for ``url``.

    A robots file that cannot be read is treated as disallowing the fetch. When
    a publisher's rules are unreadable, not fetching is the correct default.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception as exc:  # noqa: BLE001 - any failure means "do not fetch"
        logger.warning("could not read %s (%s); refusing the fetch", robots_url, exc)
        return False
    return parser.can_fetch(user_agent, url)
