"""Source discovery: how the system finds research it does not already have.

**It does not crawl the web and it does not screen-scrape a search engine.** It
calls scholarly *metadata* APIs that exist for this purpose, each with published
terms, a documented rate limit and a stable identifier scheme:

* **OpenAlex** — no key, ~250M works, polite pool via a ``mailto``. Default.
* **arXiv** — the official Atom API. One request per three seconds, enforced here.
* **Semantic Scholar** — adds open-access PDF resolution and citation context.
* **Crossref** — DOI and publisher metadata, used to enrich and de-duplicate.
* **RSS/Atom** — for practitioner blogs, which have no scholarly API.

Each provider returns a ``DiscoveredSource``: metadata plus, when the API says
the work is open access, a PDF URL. Nothing is fetched at discovery time. The
fetch is a separate, allowlisted step, so "find candidates" and "download and
store content" stay independently governed — discovery can range widely while
retrieval stays narrow.

A provider is a pure adapter over an injected HTTP transport, so the whole
discovery path is testable offline.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol, Sequence
from xml.etree import ElementTree

from ..ids import normalize_url, stable_id
from ..models import SourceType, utcnow

logger = logging.getLogger(__name__)

#: Identify the caller to every API. Several providers grant a higher rate limit
#: to requests that identify themselves, and all of them ask you to.
CONTACT_EMAIL_PLACEHOLDER = "data-platform@example.org"
USER_AGENT = "ResearchDiscoveryBot/1.0 (+governed corpus ingestion; mailto:{email})"


class DiscoveryError(RuntimeError):
    """Raised when a discovery provider fails or is misconfigured."""


class HttpJsonTransport(Protocol):
    """Minimal HTTP GET used by the providers."""

    def get_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        """Return the response body as text."""


@dataclass(frozen=True, slots=True)
class DiscoveredSource:
    """A candidate source found by a discovery provider.

    This is metadata only. Nothing has been fetched, parsed or believed. The
    ``abstract`` is included because every one of these APIs distributes
    abstracts for this purpose, and it is what lets a human — or the ranking
    below — decide whether the work is worth ingesting at all.
    """

    canonical_url: str
    title: str
    provider: str
    external_id: str
    source_type: SourceType = SourceType.PRIMARY_PAPER
    authors: str | None = None
    published_at: datetime | None = None
    abstract: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    is_open_access: bool = False
    license: str | None = None
    citation_count: int | None = None
    venue: str | None = None
    discovered_at: datetime = field(default_factory=utcnow)
    relevance_score: float = 0.0
    matched_query: str = ""

    @property
    def candidate_id(self) -> str:
        """Stable id for this candidate, keyed on its canonical URL."""
        return stable_id("cand", normalize_url(self.canonical_url))

    @property
    def fetchable_url(self) -> str | None:
        """The URL worth fetching, if any.

        Only an open-access PDF is offered. A paywalled landing page is left
        alone: the metadata is still useful, but the content is not ours to take.
        """
        return self.pdf_url if (self.is_open_access and self.pdf_url) else None


class DiscoveryProvider(Protocol):
    """A source of candidate research works."""

    name: str

    def search(self, query: str, *, limit: int, since: datetime | None) -> list[DiscoveredSource]:
        """Return candidates matching ``query``."""


class _RateLimited:
    """Mixin enforcing a minimum interval between a provider's requests."""

    min_interval_seconds: float = 0.0

    def __init__(self, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = self._clock()
        if self._last is not None:
            wait = self.min_interval_seconds - (now - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()


class OpenAlexProvider(_RateLimited):
    """OpenAlex works search. No API key; identify yourself for the polite pool."""

    name = "openalex"
    min_interval_seconds = 0.15
    BASE = "https://api.openalex.org/works"

    def __init__(
        self,
        transport: HttpJsonTransport,
        *,
        contact_email: str = CONTACT_EMAIL_PLACEHOLDER,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(clock, sleep)
        self._transport = transport
        self._email = contact_email

    def search(
        self, query: str, *, limit: int = 20, since: datetime | None = None
    ) -> list[DiscoveredSource]:
        filters = ["type:article"]
        if since:
            filters.append(f"from_publication_date:{since.date().isoformat()}")
        params = {
            "search": query,
            "per-page": str(min(limit, 50)),
            "filter": ",".join(filters),
            "mailto": self._email,
        }
        url = f"{self.BASE}?{urllib.parse.urlencode(params)}"
        self._throttle()
        try:
            payload = json.loads(self._transport.get_text(url, headers=self._headers()))
        except (json.JSONDecodeError, Exception) as exc:  # noqa: B014 - any failure is a provider failure
            raise DiscoveryError(f"OpenAlex search failed: {exc}") from exc

        results: list[DiscoveredSource] = []
        for work in payload.get("results", [])[:limit]:
            location = work.get("best_oa_location") or {}
            landing = work.get("doi") or location.get("landing_page_url") or work.get("id")
            if not landing:
                continue
            results.append(
                DiscoveredSource(
                    canonical_url=str(landing),
                    title=str(work.get("title") or work.get("display_name") or "").strip(),
                    provider=self.name,
                    external_id=str(work.get("id") or ""),
                    authors=_join_authors(
                        a.get("author", {}).get("display_name")
                        for a in work.get("authorships", [])[:12]
                    ),
                    published_at=_parse_date(work.get("publication_date")),
                    abstract=_openalex_abstract(work.get("abstract_inverted_index")),
                    doi=_clean_doi(work.get("doi")),
                    pdf_url=location.get("pdf_url"),
                    is_open_access=bool((work.get("open_access") or {}).get("is_oa")),
                    license=location.get("license"),
                    citation_count=work.get("cited_by_count"),
                    venue=((work.get("primary_location") or {}).get("source") or {}).get(
                        "display_name"
                    ),
                    matched_query=query,
                )
            )
        return results

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": USER_AGENT.format(email=self._email)}


class ArxivProvider(_RateLimited):
    """The official arXiv Atom API. Terms require one request per three seconds."""

    name = "arxiv"
    min_interval_seconds = 3.0
    BASE = "https://export.arxiv.org/api/query"
    NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(
        self,
        transport: HttpJsonTransport,
        *,
        contact_email: str = CONTACT_EMAIL_PLACEHOLDER,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(clock, sleep)
        self._transport = transport
        self._email = contact_email

    def search(
        self, query: str, *, limit: int = 20, since: datetime | None = None
    ) -> list[DiscoveredSource]:
        params = {
            "search_query": f"all:{query}",
            "max_results": str(min(limit, 100)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.BASE}?{urllib.parse.urlencode(params)}"
        self._throttle()
        try:
            root = ElementTree.fromstring(
                self._transport.get_text(url, headers={"User-Agent": USER_AGENT.format(email=self._email)})
            )
        except Exception as exc:  # noqa: BLE001
            raise DiscoveryError(f"arXiv search failed: {exc}") from exc

        results: list[DiscoveredSource] = []
        for entry in root.findall("atom:entry", self.NS):
            abs_url = _text(entry, "atom:id", self.NS)
            if not abs_url:
                continue
            published = _parse_date(_text(entry, "atom:published", self.NS))
            if since and published and published < since:
                continue
            pdf_url = next(
                (
                    link.get("href")
                    for link in entry.findall("atom:link", self.NS)
                    if link.get("title") == "pdf" or link.get("type") == "application/pdf"
                ),
                None,
            )
            results.append(
                DiscoveredSource(
                    canonical_url=abs_url,
                    title=" ".join((_text(entry, "atom:title", self.NS) or "").split()),
                    provider=self.name,
                    external_id=abs_url.rsplit("/", 1)[-1],
                    authors=_join_authors(
                        _text(a, "atom:name", self.NS) for a in entry.findall("atom:author", self.NS)
                    ),
                    published_at=published,
                    abstract=" ".join((_text(entry, "atom:summary", self.NS) or "").split()) or None,
                    doi=_text(entry, "arxiv:doi", self.NS),
                    pdf_url=pdf_url,
                    is_open_access=True,  # every arXiv submission is openly readable
                    license="ARXIV-NONEXCLUSIVE",
                    venue=_text(entry, "arxiv:journal_ref", self.NS) or "arXiv",
                    matched_query=query,
                )
            )
        return results[:limit]


class SemanticScholarProvider(_RateLimited):
    """Semantic Scholar Graph API. Adds open-access PDF resolution."""

    name = "semantic_scholar"
    min_interval_seconds = 1.0
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = (
        "title,abstract,authors,year,publicationDate,externalIds,openAccessPdf,"
        "citationCount,venue,url,isOpenAccess"
    )

    def __init__(
        self,
        transport: HttpJsonTransport,
        *,
        api_key: str = "",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(clock, sleep)
        self._transport = transport
        self._api_key = api_key

    def search(
        self, query: str, *, limit: int = 20, since: datetime | None = None
    ) -> list[DiscoveredSource]:
        params = {"query": query, "limit": str(min(limit, 100)), "fields": self.FIELDS}
        if since:
            params["year"] = f"{since.year}-"
        url = f"{self.BASE}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": USER_AGENT.format(email=CONTACT_EMAIL_PLACEHOLDER)}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        self._throttle()
        try:
            payload = json.loads(self._transport.get_text(url, headers=headers))
        except Exception as exc:  # noqa: BLE001
            raise DiscoveryError(f"Semantic Scholar search failed: {exc}") from exc

        results: list[DiscoveredSource] = []
        for paper in payload.get("data", [])[:limit]:
            landing = paper.get("url")
            if not landing:
                continue
            oa = paper.get("openAccessPdf") or {}
            results.append(
                DiscoveredSource(
                    canonical_url=str(landing),
                    title=str(paper.get("title") or "").strip(),
                    provider=self.name,
                    external_id=str(paper.get("paperId") or ""),
                    authors=_join_authors(a.get("name") for a in paper.get("authors", [])[:12]),
                    published_at=_parse_date(paper.get("publicationDate")),
                    abstract=paper.get("abstract"),
                    doi=_clean_doi((paper.get("externalIds") or {}).get("DOI")),
                    pdf_url=oa.get("url"),
                    is_open_access=bool(paper.get("isOpenAccess")),
                    license=oa.get("license"),
                    citation_count=paper.get("citationCount"),
                    venue=paper.get("venue"),
                    matched_query=query,
                )
            )
        return results


class RssProvider(_RateLimited):
    """Atom/RSS feeds, for practitioner blogs that have no scholarly API.

    Everything it returns is typed ``SECONDARY_BLOG`` regardless of how the feed
    describes itself. Secondary commentary never enters the corpus wearing the
    same badge as a primary result.
    """

    name = "rss"
    min_interval_seconds = 1.0

    def __init__(
        self,
        transport: HttpJsonTransport,
        feeds: Sequence[str],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(clock, sleep)
        self._transport = transport
        self._feeds = tuple(feeds)

    def search(
        self, query: str, *, limit: int = 20, since: datetime | None = None
    ) -> list[DiscoveredSource]:
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        results: list[DiscoveredSource] = []
        for feed_url in self._feeds:
            self._throttle()
            try:
                root = ElementTree.fromstring(self._transport.get_text(feed_url))
            except Exception as exc:  # noqa: BLE001 - one bad feed must not stop the sweep
                logger.warning("feed %s failed: %s", feed_url, exc)
                continue
            for item in _feed_items(root):
                title = item.get("title", "")
                summary = item.get("summary", "")
                haystack = f"{title} {summary}".lower()
                if terms and not any(term in haystack for term in terms):
                    continue
                published = _parse_date(item.get("published"))
                if since and published and published < since:
                    continue
                link = item.get("link")
                if not link:
                    continue
                results.append(
                    DiscoveredSource(
                        canonical_url=link,
                        title=title,
                        provider=self.name,
                        external_id=link,
                        source_type=SourceType.SECONDARY_BLOG,
                        published_at=published,
                        abstract=summary or None,
                        is_open_access=True,
                        matched_query=query,
                    )
                )
        return results[:limit]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text(element: Any, path: str, ns: dict[str, str]) -> str | None:
    found = element.find(path, ns)
    return found.text.strip() if found is not None and found.text else None


def _feed_items(root: Any) -> Iterable[dict[str, str]]:
    """Yield normalized entries from either an Atom or an RSS document."""
    atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", atom_ns):
        link_el = entry.find("atom:link", atom_ns)
        yield {
            "title": _text(entry, "atom:title", atom_ns) or "",
            "summary": _text(entry, "atom:summary", atom_ns) or "",
            "published": _text(entry, "atom:published", atom_ns) or "",
            "link": (link_el.get("href") if link_el is not None else "") or "",
        }
    for item in root.iter("item"):
        yield {
            "title": (item.findtext("title") or "").strip(),
            "summary": (item.findtext("description") or "").strip(),
            "published": (item.findtext("pubDate") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
        }


def _join_authors(names: Iterable[str | None]) -> str | None:
    cleaned = [n.strip() for n in names if n and n.strip()]
    return ", ".join(cleaned) or None


def _clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip() or None


def _parse_date(value: str | None) -> datetime | None:
    """Parse the several date shapes these APIs return, without guessing."""
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _openalex_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Rebuild an abstract from OpenAlex's inverted index."""
    if not inverted:
        return None
    positions: list[tuple[int, str]] = [
        (position, word) for word, spots in inverted.items() for position in spots
    ]
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _, word in positions)
