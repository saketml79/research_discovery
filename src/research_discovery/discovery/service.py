"""Discovery service: deciding what to ingest, and how fast.

This module answers the question "a user asked something the corpus does not
cover — now what?" It defines three ingestion speeds, because the honest answer
is that they are genuinely different products and conflating them is how a
research tool starts lying:

``METADATA_ONLY`` (seconds, synchronous)
    Search the scholarly APIs and return titles, authors, dates and abstracts.
    Nothing is fetched, parsed or extracted. The agent may say "three papers
    exist that appear to address this" and cite their URLs, and may say nothing
    at all about what they found. This is what runs inside a user's question.

``PROVISIONAL`` (minutes, asynchronous)
    Fetch, parse, chunk and extract. Every resulting claim is ``CANDIDATE`` and
    every answer built on it must be labelled provisional. This is real-time
    ingestion done honestly: it makes content available quickly without
    pretending a human reviewed it.

``REVIEWED`` (hours to days, human in the loop)
    A reviewer accepts, amends or rejects each candidate claim. Only then may a
    claim support a consensus or contradiction statement.

The review gate is not bypassable by making ingestion faster. Speeding up
ingestion changes when *candidate* knowledge arrives; it does not change what
counts as established. Every tool and view in this system is built so the agent
cannot accidentally blur the three tiers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Collection, Iterable, Sequence
from urllib.parse import urlparse

from ..ids import normalize_text, normalize_url, stable_id
from ..models import Source, SourceType, utcnow
from .providers import DiscoveredSource, DiscoveryError, DiscoveryProvider

logger = logging.getLogger(__name__)


class IngestionSpeed(str, Enum):
    """How far a discovered source is taken, and therefore what may be said."""

    METADATA_ONLY = "METADATA_ONLY"
    PROVISIONAL = "PROVISIONAL"
    REVIEWED = "REVIEWED"


class EvidenceTier(str, Enum):
    """What the agent is permitted to assert from a piece of evidence.

    The agent's instructions and the answer contract both key off this, so the
    tier travels with the evidence rather than living only in a prompt.
    """

    #: Human-reviewed claim. May support a finding, a consensus or a contradiction.
    REVIEWED_CLAIM = "REVIEWED_CLAIM"
    #: Extracted but unreviewed. May be named as a provisional lead, never as a finding.
    PROVISIONAL_CLAIM = "PROVISIONAL_CLAIM"
    #: A passage from a reviewed source. Context, not a claim.
    SOURCE_PASSAGE = "SOURCE_PASSAGE"
    #: Search-API metadata for a work not in the corpus. Existence only.
    EXTERNAL_CANDIDATE = "EXTERNAL_CANDIDATE"


#: Hosts whose content may be fetched. Discovery may *find* anything the
#: metadata APIs return; only these may be downloaded and stored.
DEFAULT_FETCH_ALLOWLIST: frozenset[str] = frozenset(
    {
        "arxiv.org",
        "www.arxiv.org",
        "export.arxiv.org",
        "aclanthology.org",
        "openreview.net",
        "proceedings.neurips.cc",
        "proceedings.mlr.press",
        "dl.acm.org",
        "www.ncbi.nlm.nih.gov",
        "europepmc.org",
        "github.com",
        "raw.githubusercontent.com",
    }
)

#: Licences under which fetched full text may be stored.
STORABLE_LICENCES: frozenset[str] = frozenset(
    {"cc-by", "cc-by-sa", "cc0", "cc-by-nc", "public-domain", "mit", "apache-2.0",
     "arxiv-nonexclusive"}
)


@dataclass(frozen=True, slots=True)
class DiscoveryDecision:
    """What the service decided to do with one discovered source, and why.

    The refusal reasons are as important as the acceptances: they are what the
    agent reports when a user asks why an obviously relevant paper was not
    ingested.
    """

    candidate: DiscoveredSource
    speed: IngestionSpeed
    fetchable: bool
    reason: str

    @property
    def tier(self) -> EvidenceTier:
        """The strongest tier this decision can currently produce."""
        return (
            EvidenceTier.PROVISIONAL_CLAIM
            if self.speed is IngestionSpeed.PROVISIONAL
            else EvidenceTier.EXTERNAL_CANDIDATE
        )


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """The outcome of one discovery run."""

    query: str
    decisions: list[DiscoveryDecision] = field(default_factory=list)
    already_known: list[str] = field(default_factory=list)
    provider_errors: dict[str, str] = field(default_factory=dict)
    searched_at: datetime = field(default_factory=utcnow)

    @property
    def fetchable(self) -> list[DiscoveryDecision]:
        """Decisions whose content may actually be downloaded."""
        return [d for d in self.decisions if d.fetchable]

    def summary(self) -> str:
        """One line a human or an agent can read."""
        return (
            f"{len(self.decisions)} new candidates for {self.query!r} "
            f"({len(self.fetchable)} fetchable, {len(self.already_known)} already in corpus"
            + (f", {len(self.provider_errors)} provider errors" if self.provider_errors else "")
            + ")"
        )


class DiscoveryService:
    """Searches providers, de-duplicates against the corpus, and ranks results."""

    def __init__(
        self,
        providers: Sequence[DiscoveryProvider],
        *,
        fetch_allowlist: Collection[str] = DEFAULT_FETCH_ALLOWLIST,
        max_per_provider: int = 20,
        max_total: int = 25,
    ) -> None:
        if not providers:
            raise ValueError("at least one discovery provider is required")
        self._providers = tuple(providers)
        self._allowlist = frozenset(h.lower() for h in fetch_allowlist)
        self._max_per_provider = max_per_provider
        self._max_total = max_total

    def discover(
        self,
        query: str,
        *,
        known_urls: Collection[str] = (),
        speed: IngestionSpeed = IngestionSpeed.METADATA_ONLY,
        since: datetime | None = None,
        recency_months: int | None = 36,
    ) -> DiscoveryResult:
        """Search every provider for ``query`` and decide what to do with each hit.

        Args:
            query: Natural-language or keyword query.
            known_urls: Canonical URLs already in the corpus, so a candidate is
                never re-proposed.
            speed: How far accepted candidates should be taken.
            since: Only return works published after this date.
            recency_months: Default recency window when ``since`` is not given.

        Returns:
            The decisions, the candidates already held, and any provider errors.
            A provider failure is recorded, never raised: one API being down
            must not make the whole question unanswerable.
        """
        if since is None and recency_months:
            since = utcnow() - timedelta(days=30 * recency_months)

        known = {normalize_url(u) for u in known_urls}
        seen: dict[str, DiscoveredSource] = {}
        already: list[str] = []
        errors: dict[str, str] = {}

        for provider in self._providers:
            try:
                hits = provider.search(query, limit=self._max_per_provider, since=since)
            except DiscoveryError as exc:
                logger.warning("provider %s failed: %s", provider.name, exc)
                errors[provider.name] = str(exc)
                continue
            except Exception as exc:  # noqa: BLE001 - an adapter bug is a provider error
                logger.exception("provider %s raised", provider.name)
                errors[provider.name] = f"{type(exc).__name__}: {exc}"
                continue

            for hit in hits:
                if not hit.title.strip():
                    continue
                key = _dedupe_key(hit)
                if normalize_url(hit.canonical_url) in known:
                    already.append(hit.canonical_url)
                    continue
                existing = seen.get(key)
                if existing is None or _provider_rank(hit) < _provider_rank(existing):
                    seen[key] = hit

        ranked = sorted(
            (self._score(hit, query) for hit in seen.values()),
            key=lambda h: h.relevance_score,
            reverse=True,
        )[: self._max_total]

        decisions = [self._decide(hit, speed) for hit in ranked]
        result = DiscoveryResult(query, decisions, already, errors)
        logger.info(result.summary())
        return result

    def _score(self, hit: DiscoveredSource, query: str) -> DiscoveredSource:
        """Score relevance from term overlap, recency, citations and evidence tier.

        Deliberately simple and inspectable: a reviewer deciding whether to spend
        time on a candidate should be able to see why it was ranked where it was.
        """
        terms = {t for t in re.split(r"\W+", normalize_text(query)) if len(t) > 2}
        haystack = normalize_text(f"{hit.title} {hit.abstract or ''}")
        overlap = sum(1 for t in terms if t in haystack) / max(len(terms), 1)

        recency = 0.0
        if hit.published_at:
            age_days = (utcnow() - hit.published_at).days
            recency = max(0.0, 1.0 - age_days / (365 * 5))

        citations = min((hit.citation_count or 0) / 200.0, 1.0)
        primary = 1.0 if hit.source_type is not SourceType.SECONDARY_BLOG else 0.4
        access = 1.0 if hit.is_open_access else 0.5

        score = 0.45 * overlap + 0.20 * recency + 0.15 * citations + 0.10 * primary + 0.10 * access
        return replace(hit, relevance_score=round(score, 4))

    def _decide(self, hit: DiscoveredSource, speed: IngestionSpeed) -> DiscoveryDecision:
        """Decide whether this candidate's content may be fetched, and say why."""
        url = hit.fetchable_url
        if url is None:
            return DiscoveryDecision(
                hit,
                IngestionSpeed.METADATA_ONLY,
                False,
                "No open-access full text is available. Metadata and abstract only; "
                "the work can be cited as existing but its content cannot be ingested.",
            )

        host = (urlparse(url).hostname or "").lower()
        if host not in self._allowlist:
            return DiscoveryDecision(
                hit,
                IngestionSpeed.METADATA_ONLY,
                False,
                f"Host {host!r} is not on the fetch allowlist. Discovery may surface it, but "
                "downloading requires adding the host in review.",
            )

        licence = (hit.license or "").lower()
        if licence and not any(licence.startswith(ok) for ok in STORABLE_LICENCES):
            return DiscoveryDecision(
                hit,
                speed,
                True,
                f"Licence {hit.license!r} is not on the storable list: fetch permitted, but only "
                "metadata and short excerpts may be retained.",
            )

        return DiscoveryDecision(
            hit,
            speed,
            True,
            "Open access on an allowlisted host with a storable licence; full ingestion permitted.",
        )

    def to_source(self, decision: DiscoveryDecision) -> Source:
        """Convert an accepted decision into a registered corpus source."""
        hit = decision.candidate
        licence = hit.license or ("ARXIV-NONEXCLUSIVE" if hit.provider == "arxiv" else None)
        storable = bool(licence) and any(
            licence.lower().startswith(ok) for ok in STORABLE_LICENCES
        )
        return Source(
            canonical_url=hit.canonical_url,
            source_type=hit.source_type,
            title=hit.title,
            publisher=hit.venue or hit.provider,
            authors=hit.authors,
            published_at=hit.published_at,
            license=licence,
            storage_permitted=storable,
        )


@dataclass(frozen=True, slots=True)
class StandingQuery:
    """A saved query the scheduled discovery sweep re-runs.

    Standing queries are how the corpus stays current without anyone watching:
    the weekly job re-runs each one, and anything new lands in the review queue.
    """

    query_text: str
    topic: str
    enabled: bool = True
    recency_months: int = 36
    max_results: int = 25
    created_by: str = "curator"
    last_run_at: datetime | None = None
    query_id: str = ""

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError("query_text must not be empty")
        object.__setattr__(
            self,
            "query_id",
            self.query_id or stable_id("qry", normalize_text(self.query_text), self.topic),
        )


def sweep(
    service: DiscoveryService,
    queries: Iterable[StandingQuery],
    *,
    known_urls: Collection[str] = (),
    speed: IngestionSpeed = IngestionSpeed.PROVISIONAL,
    on_result: Callable[[StandingQuery, DiscoveryResult], None] | None = None,
) -> list[DiscoveryResult]:
    """Run every enabled standing query and collect the results."""
    results: list[DiscoveryResult] = []
    accumulated = set(known_urls)
    for standing in queries:
        if not standing.enabled:
            continue
        result = service.discover(
            standing.query_text,
            known_urls=accumulated,
            speed=speed,
            recency_months=standing.recency_months,
        )
        # Fold this query's hits into the known set so two standing queries that
        # overlap do not both propose the same paper.
        accumulated.update(d.candidate.canonical_url for d in result.decisions)
        results.append(result)
        if on_result:
            on_result(standing, result)
    return results


def _dedupe_key(hit: DiscoveredSource) -> str:
    """Identity across providers: DOI when present, else normalized title."""
    if hit.doi:
        return f"doi:{hit.doi.lower()}"
    arxiv = re.search(r"(\d{4}\.\d{4,5})", hit.canonical_url)
    if arxiv:
        return f"arxiv:{arxiv.group(1)}"
    return f"title:{normalize_text(hit.title)}"


#: Preference order when the same work comes back from several providers.
#: arXiv first because it gives an authoritative PDF and licence for preprints.
_PROVIDER_PRIORITY = {"arxiv": 0, "openalex": 1, "semantic_scholar": 2, "rss": 3}


def _provider_rank(hit: DiscoveredSource) -> int:
    return _PROVIDER_PRIORITY.get(hit.provider, 99)
