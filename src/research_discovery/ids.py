"""Deterministic identifiers and hashing.

Ids are derived from content, never from a random UUID, so that re-running any
pipeline stage over unchanged input produces the same rows. That is what makes
every stage idempotent and safe to retry.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_ID_PREFIX_MAX = 12


def sha256_hex(data: str | bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text for hashing and comparison.

    Applies NFKC normalization, collapses whitespace, strips, and lowercases.
    Used only for identity and comparison - never for stored display text.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE.sub(" ", normalized).strip().lower()


def normalize_url(url: str) -> str:
    """Canonicalize a URL for source identity.

    Lowercases the scheme and host, drops a default port, a trailing slash, and
    a trailing ``.pdf`` version suffix on arXiv-style URLs. Query strings are
    preserved because they can be semantically significant.
    """
    text = url.strip()
    if "://" in text:
        scheme, rest = text.split("://", 1)
        host, _, path = rest.partition("/")
        host = host.lower()
        for default in (":80", ":443"):
            if host.endswith(default):
                host = host[: -len(default)]
        text = f"{scheme.lower()}://{host}" + (f"/{path}" if path else "")
    return text.rstrip("/")


def stable_id(prefix: str, *parts: str) -> str:
    """Return a readable, deterministic id of the form ``prefix-<hash>``.

    Args:
        prefix: Short entity marker, e.g. ``src`` or ``clm``.
        *parts: Components hashed in order. ``None``-like parts must be passed
            as empty strings by the caller so the component count is explicit.
    """
    if not prefix:
        raise ValueError("prefix is required")
    digest = sha256_hex("\x1f".join(parts))
    return f"{prefix}-{digest[:_ID_PREFIX_MAX]}"


def source_id(canonical_url: str) -> str:
    """Identity of a logical source: its canonical URL."""
    return stable_id("src", normalize_url(canonical_url))


def source_version_id(source_id_value: str, content_hash: str) -> str:
    """Identity of one fetched revision of a source."""
    return stable_id("srcv", source_id_value, content_hash)


def chunk_id(source_version_id_value: str, chunk_index: int) -> str:
    """Identity of a chunk within a source version."""
    return stable_id("chk", source_version_id_value, str(chunk_index))


def claim_id(source_version_id_value: str, claim_text: str) -> str:
    """Identity of a claim: its normalized text within a source version."""
    return stable_id("clm", source_version_id_value, normalize_text(claim_text))


def relationship_id(from_claim: str, to_claim: str, relationship_type: str) -> str:
    """Identity of a claim edge.

    The claim pair is sorted so that the same undirected pair yields one id
    regardless of which claim the detector happened to visit first.
    """
    first, second = sorted((from_claim, to_claim))
    return stable_id("rel", first, second, relationship_type)


def review_id(target_type: str, target_id: str) -> str:
    """Identity of a review-queue item."""
    return stable_id("rev", target_type, target_id)


def taxonomy_id(dimension: str, canonical_term: str) -> str:
    """Identity of a controlled-vocabulary term."""
    return stable_id("tax", dimension.upper(), normalize_text(canonical_term))
