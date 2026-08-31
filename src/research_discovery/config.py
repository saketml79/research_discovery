"""Runtime configuration.

Every deployable knob lives here and is populated from job parameters or the
environment, never hard-coded at a call site. Fully qualified names are built in
one place so a schema rename is a one-line change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

DEFAULT_CATALOG = "main"
DEFAULT_SCHEMA = "research_discovery"
DEFAULT_VOLUME = "raw_sources"

#: Maximum characters of verbatim source text stored when a source's licence
#: does not permit full-text storage. Kept deliberately short.
RESTRICTED_EXCERPT_CHARS = 400

#: Maximum characters of an evidence excerpt attached to a claim, regardless of
#: licence. An excerpt is a pointer to the source, not a substitute for it.
MAX_EVIDENCE_EXCERPT_CHARS = 600


class ConfigError(ValueError):
    """Raised when configuration is missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable job configuration.

    Attributes:
        catalog: Unity Catalog catalog name.
        schema: Schema holding all Research Discovery tables.
        volume: UC volume holding raw source documents.
        parser: Parser adapter key (see ``parsers.registry``).
        extractor: Claim extractor adapter key (see ``extract.registry``).
        extraction_model: Serving endpoint used by the LLM extractor.
        ai_search_endpoint: Vector Search endpoint, or empty when disabled.
        max_chunk_chars: Upper bound on a chunk's character count.
        min_chunk_chars: Chunks shorter than this are merged forward.
        review_confidence_threshold: Extractions below this are queued HIGH.
        dry_run: When true, pipelines log planned writes without performing them.
    """

    catalog: str = DEFAULT_CATALOG
    schema: str = DEFAULT_SCHEMA
    volume: str = DEFAULT_VOLUME
    parser: str = "pypdf"
    extractor: str = "llm"
    extraction_model: str = "databricks-claude-sonnet-4-5"
    ai_search_endpoint: str = ""
    max_chunk_chars: int = 2400
    min_chunk_chars: int = 200
    review_confidence_threshold: float = 0.75
    dry_run: bool = False
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("catalog", "schema", "volume"):
            value = getattr(self, name)
            if not value or not _is_identifier(value):
                raise ConfigError(f"{name} must be a valid unquoted identifier, got {value!r}")
        if self.min_chunk_chars >= self.max_chunk_chars:
            raise ConfigError("min_chunk_chars must be smaller than max_chunk_chars")
        if not 0.0 <= self.review_confidence_threshold <= 1.0:
            raise ConfigError("review_confidence_threshold must be in [0, 1]")

    # -- naming -------------------------------------------------------------

    @property
    def fq_schema(self) -> str:
        """Fully qualified schema name."""
        return f"{self.catalog}.{self.schema}"

    def table(self, name: str) -> str:
        """Return the three-level name of ``name`` in this deployment."""
        if not _is_identifier(name):
            raise ConfigError(f"invalid table identifier: {name!r}")
        return f"{self.fq_schema}.{name}"

    @property
    def volume_path(self) -> str:
        """Return the ``/Volumes`` path of the raw-source volume."""
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    # -- construction -------------------------------------------------------

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        """Build a config from ``RD_*`` environment variables."""
        env = os.environ if env is None else env
        return cls(
            catalog=env.get("RD_CATALOG", DEFAULT_CATALOG),
            schema=env.get("RD_SCHEMA", DEFAULT_SCHEMA),
            volume=env.get("RD_VOLUME", DEFAULT_VOLUME),
            parser=env.get("RD_PARSER", "pypdf"),
            extractor=env.get("RD_EXTRACTOR", "llm"),
            extraction_model=env.get("RD_EXTRACTION_MODEL", "databricks-claude-sonnet-4-5"),
            ai_search_endpoint=env.get("RD_AI_SEARCH_ENDPOINT", ""),
            dry_run=_as_bool(env.get("RD_DRY_RUN", "false")),
        )

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> "Config":
        """Build a config from job task parameters, falling back to the env."""
        base = cls.from_env()
        known = {f for f in base.__dataclass_fields__ if f != "extra"}  # type: ignore[attr-defined]
        overrides: dict[str, Any] = {}
        extra: dict[str, str] = {}
        for key, value in args.items():
            if key in known:
                overrides[key] = _coerce(getattr(base, key), value)
            else:
                extra[key] = str(value)
        return replace(base, **overrides, extra=extra)


def _is_identifier(value: str) -> bool:
    """True when ``value`` is safe to interpolate into a SQL identifier."""
    return bool(value) and value.replace("_", "").isalnum() and not value[0].isdigit()


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce(reference: Any, value: Any) -> Any:
    """Coerce ``value`` to the type of ``reference``."""
    if isinstance(reference, bool):
        return _as_bool(value)
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return str(value)
