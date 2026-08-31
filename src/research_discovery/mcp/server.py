"""Custom MCP server exposing the agent's bounded non-SQL tools.

Two tools live here because neither can be a Unity Catalog SQL function:

* ``create_proposal`` performs a real INSERT. A UC SQL function cannot write, so
  a SQL "proposal tool" could only return a string claiming a proposal was
  recorded while writing nothing — the precise class of false statement this
  system exists to prevent. It is a tool endpoint instead.
* ``search_external_source`` reaches an approved external source and returns
  provenance-bearing results that are explicitly *not* corpus facts.

Both are narrow by construction. ``create_proposal`` can only insert
``PENDING_APPROVAL`` rows into one table; there is no tool here — and none
anywhere in this project — that approves or executes a proposal, changes a review
status, or writes to the corpus.

The transport is deliberately separated from the tool logic: ``ToolRegistry``
holds the behaviour and is unit-tested directly, while ``serve_stdio`` binds it
to whichever MCP runtime the workspace provides.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..config import Config
from ..models import Proposal, to_row
from ..review.proposals import ProposalValidationError, build_proposal

logger = logging.getLogger(__name__)

SERVER_NAME = "research-discovery-tools"
SERVER_VERSION = "1.0.0"


class SqlExecutor(Protocol):
    """Executes parameterised SQL against the deployment's warehouse."""

    def execute(self, statement: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        """Run ``statement`` with ``parameters`` and return any rows."""


class WarehouseSqlExecutor:
    """``SqlExecutor`` backed by the Databricks SQL Statement Execution API."""

    def __init__(self, warehouse_id: str, workspace_client: Any | None = None) -> None:
        self._warehouse_id = warehouse_id
        self._client = workspace_client

    def _resolve(self) -> Any:
        if self._client is not None:
            return self._client
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - lazy backend

        self._client = WorkspaceClient()
        return self._client

    def execute(self, statement: str, parameters: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        from databricks.sdk.service.sql import StatementParameterListItem  # noqa: PLC0415

        response = self._resolve().statement_execution.execute_statement(
            warehouse_id=self._warehouse_id,
            statement=statement,
            parameters=[
                StatementParameterListItem(name=key, value=None if value is None else str(value))
                for key, value in parameters.items()
            ],
            wait_timeout="30s",
        )
        result = getattr(response, "result", None)
        rows = getattr(result, "data_array", None) or []
        manifest = getattr(response, "manifest", None)
        schema = getattr(manifest, "schema", None)
        columns = [c.name for c in (getattr(schema, "columns", None) or [])]
        return [dict(zip(columns, row)) for row in rows]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool's structured response.

    ``ok`` is always present so the agent can distinguish a refusal from a
    failure from a success, rather than inferring it from prose.
    """

    ok: bool
    data: dict[str, Any]
    message: str

    def to_json(self) -> str:
        """Serialise for transport."""
        return json.dumps({"ok": self.ok, "message": self.message, **self.data}, default=str)


class ToolRegistry:
    """The MCP tool implementations, independent of any transport."""

    def __init__(
        self,
        config: Config,
        executor: SqlExecutor,
        *,
        principal: str = "research_discovery_agent",
        external_fetcher: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        """Args:
        config: Supplies the target table names.
        executor: SQL executor used for the proposal insert.
        principal: Recorded as ``created_by`` on every proposal.
        external_fetcher: Optional approved external-source adapter. When
            ``None``, ``search_external_source`` refuses rather than pretending
            the corpus is the whole world.
        """
        self._config = config
        self._executor = executor
        self._principal = principal
        self._external_fetcher = external_fetcher

    # -- tool: create_proposal ---------------------------------------------

    def create_proposal(
        self,
        proposal_type: str,
        payload_json: str,
        rationale: str,
        investigation_id: str | None = None,
        retrieved_claim_ids: Sequence[str] = (),
    ) -> ToolResult:
        """Insert a ``PENDING_APPROVAL`` proposal after validating its payload.

        Args:
            proposal_type: REVIEW_CLAIM, INGEST_SOURCE, RESOLVE_CONTRADICTION
                or OPEN_QUESTION.
            payload_json: JSON proposal body.
            rationale: Why this is proposed, citing the motivating evidence.
            investigation_id: Correlates the proposal with the question.
            retrieved_claim_ids: Claim ids the agent actually retrieved this
                turn. Every claim id in the payload must appear here.

        Returns:
            A ``ToolResult``. A validation failure returns ``ok=False`` with the
            reason; nothing is written in that case.
        """
        try:
            payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
        except json.JSONDecodeError as exc:
            return ToolResult(False, {"error": "INVALID_JSON"}, f"payload_json is not JSON: {exc}")
        if not isinstance(payload, dict):
            return ToolResult(False, {"error": "INVALID_PAYLOAD"}, "payload_json must be an object")

        try:
            proposal = build_proposal(
                proposal_type,
                payload,
                created_by=self._principal,
                rationale=rationale,
                investigation_id=investigation_id,
                known_claim_ids=retrieved_claim_ids,
            )
        except ProposalValidationError as exc:
            logger.warning("proposal rejected: %s", exc)
            return ToolResult(
                False,
                {"error": "VALIDATION_FAILED"},
                f"No proposal was created. {exc}",
            )

        self._insert(proposal)
        logger.info("recorded proposal %s (%s)", proposal.proposal_id, proposal.proposal_type)
        return ToolResult(
            True,
            {
                "proposal_id": proposal.proposal_id,
                "proposal_type": proposal.proposal_type,
                "status": proposal.status,
            },
            f"Proposal {proposal.proposal_id} was written with status PENDING_APPROVAL. "
            "It has not been executed and requires human approval.",
        )

    def _insert(self, proposal: Proposal) -> None:
        """Write the proposal row.

        The status is a SQL literal rather than a parameter so that no caller,
        including a future one, can insert a proposal in any other state.
        """
        row = to_row(proposal)
        statement = f"""
            INSERT INTO {self._config.table('agent_proposal')}
              (proposal_id, investigation_id, proposal_type, payload_json, rationale,
               status, created_by, created_at, approved_by, approved_at)
            VALUES
              (:proposal_id, :investigation_id, :proposal_type, :payload_json, :rationale,
               'PENDING_APPROVAL', :created_by, current_timestamp(), NULL, NULL)
        """
        self._executor.execute(
            statement,
            {
                "proposal_id": row["proposal_id"],
                "investigation_id": row["investigation_id"],
                "proposal_type": row["proposal_type"],
                "payload_json": row["payload_json"],
                "rationale": row["rationale"],
                "created_by": row["created_by"],
            },
        )

    # -- tool: search_external_source --------------------------------------

    def search_external_source(self, query: str, source_hint: str = "") -> ToolResult:
        """Query an approved external source for context.

        Results are provenance-bearing context, never corpus facts: they carry
        a URL, publisher and retrieval time, and the agent is required to label
        them as unreviewed external context rather than cite them as claims.
        """
        if self._external_fetcher is None:
            return ToolResult(
                False,
                {"error": "NO_APPROVED_EXTERNAL_SOURCE"},
                "No approved external source is configured for this deployment. Answer from "
                "the reviewed corpus, and say plainly that the corpus may not cover this.",
            )
        try:
            result = self._external_fetcher(query, source_hint)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent as a failure
            logger.warning("external source failed: %s", exc)
            return ToolResult(False, {"error": "FETCH_FAILED"}, f"External lookup failed: {exc}")

        return ToolResult(
            True,
            {
                "results": result.get("results", []),
                "publisher": result.get("publisher"),
                "retrieved_at": result.get("retrieved_at"),
                "query": query,
            },
            "External context, NOT a reviewed corpus claim. Label it as unreviewed external "
            "context, cite its URL and retrieval time, and do not use it to support a "
            "consensus or contradiction statement.",
        )


#: Tool schemas advertised over MCP. Descriptions are the agent's instructions
#: for these tools, so they state the boundary as well as the behaviour.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "create_proposal",
        "description": (
            "Record a recommended next step as a PENDING_APPROVAL proposal. Writes one row to "
            "the governed proposal table and nothing else: it cannot change the corpus, a "
            "review status, or any platform object, and no tool exists that approves or "
            "executes a proposal. Every claim id in the payload must be one you retrieved in "
            "this conversation; citing an unretrieved claim is refused and nothing is written. "
            "After calling it, tell the user the proposal is pending human approval."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["proposal_type", "payload_json", "rationale"],
            "properties": {
                "proposal_type": {
                    "type": "string",
                    "enum": [
                        "REVIEW_CLAIM",
                        "INGEST_SOURCE",
                        "RESOLVE_CONTRADICTION",
                        "OPEN_QUESTION",
                    ],
                },
                "payload_json": {"type": "string"},
                "rationale": {"type": "string"},
                "investigation_id": {"type": "string"},
                "retrieved_claim_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "search_external_source",
        "description": (
            "Look up context from an approved external source. Returns provenance-bearing "
            "context that is NOT part of the reviewed corpus: label it as unreviewed external "
            "context, cite its URL and retrieval time, and never use it to support a consensus "
            "or contradiction statement. Returns a refusal when no external source is "
            "configured for this deployment."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "source_hint": {"type": "string"},
            },
        },
    },
]


def dispatch(
    registry: ToolRegistry,
    name: str,
    arguments: Mapping[str, Any],
    discovery: Any | None = None,
) -> ToolResult:
    """Route an MCP tool call to its implementation.

    Args:
        registry: The proposal and external-source tools.
        name: Tool name requested by the agent.
        arguments: Tool arguments.
        discovery: Optional ``DiscoveryTools``. When present, the discovery
            tools are dispatchable too; when absent, asking for one returns the
            unknown-tool refusal rather than a transport error.

    An unknown tool is a refusal, not an exception: the agent should be told the
    tool does not exist rather than seeing a transport error.
    """
    if discovery is not None:
        from .discovery_tools import dispatch_discovery  # noqa: PLC0415 - avoids a cycle

        discovered = dispatch_discovery(discovery, name, arguments)
        if discovered is not None:
            return discovered

    if name == "create_proposal":
        return registry.create_proposal(
            proposal_type=str(arguments.get("proposal_type", "")),
            payload_json=arguments.get("payload_json", "{}"),
            rationale=str(arguments.get("rationale", "")),
            investigation_id=arguments.get("investigation_id"),
            retrieved_claim_ids=tuple(arguments.get("retrieved_claim_ids") or ()),
        )
    if name == "search_external_source":
        return registry.search_external_source(
            query=str(arguments.get("query", "")),
            source_hint=str(arguments.get("source_hint", "")),
        )
    return ToolResult(
        False,
        {"error": "UNKNOWN_TOOL", "available": [s["name"] for s in all_tool_schemas()]},
        f"No tool named {name!r} is exposed by this server.",
    )


def all_tool_schemas() -> list[dict[str, Any]]:
    """Every tool this server can expose, proposal and discovery alike."""
    from .discovery_tools import DISCOVERY_TOOL_SCHEMAS  # noqa: PLC0415 - avoids a cycle

    return [*TOOL_SCHEMAS, *DISCOVERY_TOOL_SCHEMAS]


def serve_stdio(config: Config, warehouse_id: str) -> None:  # pragma: no cover - transport
    """Run the server over stdio using the ``mcp`` package when available."""
    try:
        from mcp.server import Server  # noqa: PLC0415
        from mcp.server.stdio import stdio_server  # noqa: PLC0415
        import mcp.types as types  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the 'mcp' package is not installed; install research-discovery[mcp] to serve"
        ) from exc

    import asyncio

    registry = ToolRegistry(config, WarehouseSqlExecutor(warehouse_id))
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return [
            types.Tool(
                name=schema["name"],
                description=schema["description"],
                inputSchema=schema["inputSchema"],
            )
            for schema in all_tool_schemas()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        result = dispatch(registry, name, arguments)
        return [types.TextContent(type="text", text=result.to_json())]

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(main())
