"""Client for the Genie Agents conversation API.

This is the programmatic entry point Appendix A calls "a thin application or
local client that calls the Genie Agents API". It exists so the agent can be
exercised the same way from a job, a test and the benchmark harness — the
acceptance criterion being that the same prompts pass from the Genie UI and from
the API.

The client returns the tool calls and SQL Genie ran alongside the text answer,
because a graded answer needs its trace: knowing whether ``compare_claims`` was
actually called is what separates a correct answer from a lucky one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 180.0

#: Conversation states that mean the turn is finished, successfully or not.
TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"})


class GenieConversationApi(Protocol):
    """The conversation surface this client needs."""

    def start_conversation(self, space_id: str, content: str) -> dict[str, Any]:
        """Begin a conversation and return the initial message envelope."""

    def create_message(self, space_id: str, conversation_id: str, content: str) -> dict[str, Any]:
        """Post a follow-up message and return its envelope."""

    def get_message(self, space_id: str, conversation_id: str, message_id: str) -> dict[str, Any]:
        """Fetch the current state of a message."""

    def get_query_result(
        self, space_id: str, conversation_id: str, message_id: str, attachment_id: str
    ) -> dict[str, Any]:
        """Fetch the result rows for a SQL attachment."""


@dataclass(slots=True)
class AgentTurn:
    """One completed agent turn with its execution trace.

    Attributes:
        text: The agent's prose answer.
        queries: SQL statements Genie generated and ran.
        tools_called: Names of UC functions and tools invoked, inferred from the
            generated SQL and any tool attachments.
        rows: Result rows returned to the agent, per attachment.
        state: Terminal conversation state.
        conversation_id: For follow-up turns.
        message_id: The message this turn corresponds to.
        error: Populated when the turn failed.
        elapsed_seconds: Wall-clock duration, for benchmark reporting.
    """

    text: str = ""
    queries: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    state: str = "UNKNOWN"
    conversation_id: str = ""
    message_id: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """True when the turn completed and produced an answer."""
        return self.state == "COMPLETED" and self.error is None

    def called(self, tool_name: str) -> bool:
        """Whether ``tool_name`` appears in this turn's trace."""
        needle = tool_name.lower()
        if any(needle == t.lower() for t in self.tools_called):
            return True
        return any(needle in q.lower() for q in self.queries)


class WorkspaceGenieConversationApi:
    """``GenieConversationApi`` backed by the Databricks SDK's REST surface."""

    BASE = "/api/2.0/genie/spaces"

    def __init__(self, workspace_client: Any | None = None) -> None:
        self._client = workspace_client

    def _resolve(self) -> Any:
        if self._client is not None:
            return self._client
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - lazy backend

        self._client = WorkspaceClient()
        return self._client

    def start_conversation(self, space_id: str, content: str) -> dict[str, Any]:
        return self._resolve().api_client.do(
            "POST", f"{self.BASE}/{space_id}/start-conversation", body={"content": content}
        )

    def create_message(self, space_id: str, conversation_id: str, content: str) -> dict[str, Any]:
        return self._resolve().api_client.do(
            "POST",
            f"{self.BASE}/{space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
        )

    def get_message(self, space_id: str, conversation_id: str, message_id: str) -> dict[str, Any]:
        return self._resolve().api_client.do(
            "GET",
            f"{self.BASE}/{space_id}/conversations/{conversation_id}/messages/{message_id}",
        )

    def get_query_result(
        self, space_id: str, conversation_id: str, message_id: str, attachment_id: str
    ) -> dict[str, Any]:
        return self._resolve().api_client.do(
            "GET",
            f"{self.BASE}/{space_id}/conversations/{conversation_id}/messages/"
            f"{message_id}/attachments/{attachment_id}/query-result",
        )


class GenieAgentClient:
    """Asks the deployed Genie Agent a question and returns the turn with its trace."""

    def __init__(
        self,
        api: GenieConversationApi,
        space_id: str,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
    ) -> None:
        self._api = api
        self._space_id = space_id
        self._poll = poll_seconds
        self._timeout = timeout_seconds
        self._sleep = sleep
        self._clock = clock

    def ask(self, question: str, *, conversation_id: str | None = None) -> AgentTurn:
        """Ask a question and wait for the turn to finish.

        Args:
            question: The user prompt.
            conversation_id: Continue an existing conversation when supplied.

        Returns:
            The completed turn. A failure is reported on the turn rather than
            raised, so a benchmark run records it instead of aborting.
        """
        started = self._clock()
        try:
            envelope = (
                self._api.create_message(self._space_id, conversation_id, question)
                if conversation_id
                else self._api.start_conversation(self._space_id, question)
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return AgentTurn(state="FAILED", error=f"could not start turn: {exc}")

        active_conversation = str(
            envelope.get("conversation_id") or (envelope.get("conversation") or {}).get("id") or ""
        )
        message = envelope.get("message") or envelope
        message_id = str(message.get("message_id") or message.get("id") or "")
        if not active_conversation or not message_id:
            return AgentTurn(state="FAILED", error=f"unexpected API envelope: {envelope!r}")

        final = self._poll_until_done(active_conversation, message_id)
        if isinstance(final, AgentTurn):
            final.elapsed_seconds = self._clock() - started
            return final

        turn = self._build_turn(final, active_conversation, message_id)
        turn.elapsed_seconds = self._clock() - started
        return turn

    def _poll_until_done(self, conversation_id: str, message_id: str) -> dict[str, Any] | AgentTurn:
        deadline = self._clock() + self._timeout
        while True:
            try:
                message = self._api.get_message(self._space_id, conversation_id, message_id)
            except Exception as exc:  # noqa: BLE001
                return AgentTurn(
                    state="FAILED",
                    error=f"polling failed: {exc}",
                    conversation_id=conversation_id,
                    message_id=message_id,
                )
            state = str(message.get("status") or message.get("state") or "UNKNOWN")
            if state in TERMINAL_STATES:
                return message
            if self._clock() >= deadline:
                return AgentTurn(
                    state="TIMEOUT",
                    error=f"turn did not finish within {self._timeout:.0f}s (last state {state})",
                    conversation_id=conversation_id,
                    message_id=message_id,
                )
            self._sleep(self._poll)

    def _build_turn(
        self, message: dict[str, Any], conversation_id: str, message_id: str
    ) -> AgentTurn:
        """Flatten an API message into a turn, collecting its trace."""
        turn = AgentTurn(
            state=str(message.get("status") or message.get("state") or "UNKNOWN"),
            conversation_id=conversation_id,
            message_id=message_id,
            error=message.get("error") if isinstance(message.get("error"), str) else None,
        )

        texts: list[str] = []
        for attachment in message.get("attachments") or []:
            text_part = attachment.get("text") or {}
            if text_part.get("content"):
                texts.append(str(text_part["content"]))

            query_part = attachment.get("query") or {}
            statement = query_part.get("query") or query_part.get("statement")
            if statement:
                turn.queries.append(str(statement))
                turn.tools_called.extend(_functions_in(str(statement)))
                attachment_id = str(attachment.get("attachment_id") or attachment.get("id") or "")
                if attachment_id:
                    turn.rows.extend(
                        self._fetch_rows(conversation_id, message_id, attachment_id)
                    )

            tool_part = attachment.get("tool_call") or attachment.get("tool") or {}
            if tool_part.get("name"):
                turn.tools_called.append(str(tool_part["name"]))

        if not texts and message.get("content"):
            texts.append(str(message["content"]))
        turn.text = "\n\n".join(texts).strip()
        turn.tools_called = sorted(set(turn.tools_called))
        return turn

    def _fetch_rows(
        self, conversation_id: str, message_id: str, attachment_id: str
    ) -> list[list[Any]]:
        try:
            result = self._api.get_query_result(
                self._space_id, conversation_id, message_id, attachment_id
            )
        except Exception as exc:  # noqa: BLE001 - a missing result is not fatal
            logger.warning("could not fetch query result %s: %s", attachment_id, exc)
            return []
        data = (result.get("statement_response") or result).get("result") or {}
        return [list(row) for row in (data.get("data_array") or [])]


#: UC functions the agent may call. Used to read a tool trace out of generated SQL.
KNOWN_FUNCTIONS: tuple[str, ...] = (
    "search_claims",
    "compare_claims",
    "get_claim_evidence",
    "get_open_questions",
    "get_corpus_coverage",
    "search_passages",
    "get_taxonomy",
    "get_review_backlog",
    "get_figure_evidence",
)


def _functions_in(statement: str) -> list[str]:
    """Return the known agent functions referenced by a SQL statement."""
    lowered = statement.lower()
    return [name for name in KNOWN_FUNCTIONS if f"{name}(" in lowered]


def ask_all(
    client: GenieAgentClient, questions: Sequence[str], *, same_conversation: bool = False
) -> list[AgentTurn]:
    """Ask several questions, optionally within one conversation."""
    turns: list[AgentTurn] = []
    conversation_id: str | None = None
    for question in questions:
        turn = client.ask(question, conversation_id=conversation_id)
        if same_conversation and turn.conversation_id:
            conversation_id = turn.conversation_id
        turns.append(turn)
    return turns
