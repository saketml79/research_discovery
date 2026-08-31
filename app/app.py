"""Databricks App: the claim review workstation.

The review queue is the boundary between extracted candidate knowledge and what
the agent may assert. Without an interface, that boundary is unusable and the
corpus can only ever contain whatever was seeded — which is why this app exists
rather than a notebook.

A reviewer sees the claim, the passage or figure it came from, its extractor
confidence and any parser warning, and then accepts, amends or rejects it. Every
decision records who made it and when. Amendments are restricted to scope and
citation fields: a reviewer corrects the record, they do not restate the source's
finding.

Run locally with ``streamlit run app/app.py`` or deploy with the bundle.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import streamlit as st

from research_discovery.config import Config
from research_discovery.models import ReviewStatus
from research_discovery.review.queue import AMENDABLE_FIELDS, ReviewError, validate_decision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Ordered for display; the set of what may be amended comes from review.queue,
#: so the app and the pipeline cannot drift apart on the rules.
DISPLAY_FIELDS = ("task", "method", "metric", "benchmark", "condition_text", "metric_unit")

#: Colour + label for each queue priority. Centralised so a new priority value
#: only needs one new line, not a scattered set of if/else blocks.
PRIORITY_BADGE: dict[str, str] = {
    "HIGH": "HIGH",
    "NORMAL": "NORMAL",
    "LOW": "LOW",
}
PRIORITY_CSS_CLASS: dict[str, str] = {
    "HIGH": "rd-badge-high",
    "NORMAL": "rd-badge-normal",
    "LOW": "rd-badge-low",
}


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Connecting to the SQL warehouse (a cold start can take up to a minute)…")
def _connect(user_token: str) -> Any:
    """Open a SQL warehouse connection as the given user's OAuth token.

    Cache-keyed on the token itself, so each signed-in user gets their own
    connection object rather than reusing another user's session.
    """
    from databricks import sql  # noqa: PLC0415

    http_path = os.environ.get("DATABRICKS_HTTP_PATH") or (
        f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}"
    )
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=http_path,
        access_token=user_token,
    )


def get_user_token() -> str | None:
    """The signed-in user's forwarded OAuth token (on-behalf-of / OBO auth).

    Requires the app to declare the ``sql`` user-authorization scope. Every
    query then runs as the user, enforced by their own Unity Catalog grants -
    no permission ever needs to be granted to the app's own service principal.
    """
    header = st.context.headers if hasattr(st, "context") else {}
    return header.get("x-forwarded-access-token") or os.environ.get("DATABRICKS_TOKEN")


def get_connection() -> Any:
    """Open a SQL warehouse connection authenticated as the signed-in user."""
    token = get_user_token()
    if not token:
        raise RuntimeError(
            "No user OAuth token was forwarded to this app. This app requires the 'sql' "
            "user-authorization scope to be enabled (Databricks workspace -> Apps -> "
            "research-review-dev -> Authorization), and you must grant consent on first visit."
        )
    return _connect(token)


def query(statement: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run a query and return rows as dicts."""
    with get_connection().cursor() as cursor:
        cursor.execute(statement, parameters or {})
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def execute(statement: str, parameters: dict[str, Any] | None = None) -> None:
    """Run a statement that returns nothing."""
    with get_connection().cursor() as cursor:
        cursor.execute(statement, parameters or {})


def load_config() -> Config:
    """Build config from the app's environment."""
    return Config(
        catalog=os.environ.get("RD_CATALOG", "main"),
        schema=os.environ.get("RD_SCHEMA", "research_discovery"),
    )


def genie_space_url() -> str | None:
    """Direct link to the deployed Genie Agent's chat UI, if configured."""
    space_id = os.environ.get("RD_GENIE_SPACE_ID")
    host = os.environ.get("DATABRICKS_HOST")
    if not space_id or not host:
        return None
    return f"{host.rstrip('/')}/genie/rooms/{space_id}"


@st.cache_resource(show_spinner=False)
def _genie_client(user_token: str, space_id: str) -> Any:
    """Build a Genie Agent client authenticated as the signed-in user (OBO).

    Requires the app's ``genie`` user-authorization scope, in addition to ``sql``.
    """
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    from research_discovery.agent.client import GenieAgentClient, WorkspaceGenieConversationApi

    workspace_client = WorkspaceClient(host=os.environ["DATABRICKS_HOST"], token=user_token)
    api = WorkspaceGenieConversationApi(workspace_client)
    return GenieAgentClient(api, space_id)


def get_genie_client() -> Any | None:
    """The Genie client for the current user, or None if not configured."""
    space_id = os.environ.get("RD_GENIE_SPACE_ID")
    token = get_user_token()
    if not space_id or not token:
        return None
    return _genie_client(token, space_id)


def reviewer_identity() -> str:
    """Identify the reviewer for the audit trail (never shown in the UI as-is).

    Databricks Apps forward the signed-in user's identity in a header. An
    unattributed review is not a review, so the app refuses to record decisions
    when it cannot tell who is making them. The real identity is still what gets
    written to ``reviewed_by`` - only the on-screen label is pseudonymous.
    """
    header = st.context.headers if hasattr(st, "context") else {}
    return (
        header.get("X-Forwarded-Email")
        or header.get("X-Forwarded-Preferred-Username")
        or os.environ.get("RD_REVIEWER", "")
    )


def reviewer_display_name(reviewer: str) -> str:
    """A short, stable, non-identifying label shown in the UI instead of an email."""
    if not reviewer:
        return "Unknown reviewer"
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256(reviewer.encode("utf-8")).hexdigest()[:4].upper()
    return f"Reviewer {digest}"


# ---------------------------------------------------------------------------
# queue operations
# ---------------------------------------------------------------------------


def fetch_queue(config: Config, priority: str) -> list[dict[str, Any]]:
    """Load open review items with their claim context."""
    condition = "" if priority == "ALL" else "AND q.priority = :priority"
    return query(
        f"""
        SELECT q.review_id, q.priority, q.reason, q.created_at,
               c.claim_id, c.claim_text, c.claim_type, c.task, c.method, c.metric,
               c.metric_value, c.metric_unit, c.benchmark, c.condition_text,
               c.evidence_excerpt, c.page_number, c.source_url, c.figure_id,
               c.extraction_confidence, c.missing_field_reason,
               c.extractor_name, c.extractor_version,
               s.title AS source_title, s.source_type
        FROM {config.table('research_review_queue')} q
        JOIN {config.table('research_claim')} c
          ON q.target_type = 'CLAIM' AND q.target_id = c.claim_id
        JOIN {config.table('research_source')} s ON c.source_id = s.source_id
        WHERE q.status = 'OPEN' AND c.review_status IN ('CANDIDATE', 'IN_REVIEW')
        {condition}
        ORDER BY CASE q.priority WHEN 'HIGH' THEN 0 WHEN 'NORMAL' THEN 1 ELSE 2 END,
                 c.extraction_confidence ASC NULLS FIRST
        LIMIT 50
        """,
        {"priority": priority} if condition else {},
    )


def fetch_evidence(config: Config, claim_id: str) -> dict[str, Any] | None:
    """Load the passage a claim came from, with its parser provenance."""
    rows = query(
        f"""
        SELECT ch.text, ch.section_title, ch.parser_name, ch.extraction_warning
        FROM {config.table('research_claim')} c
        LEFT JOIN {config.table('research_chunk')} ch ON c.chunk_id = ch.chunk_id
        WHERE c.claim_id = :claim_id
        """,
        {"claim_id": claim_id},
    )
    return rows[0] if rows else None


def fetch_figure(config: Config, figure_id: str) -> dict[str, Any] | None:
    """Load a figure reading so a reviewer can check it against the image."""
    rows = query(
        f"""
        SELECT page_number, caption, image_uri, extracted_text, extracted_entities,
               extraction_confidence, vision_model, prompt_version
        FROM {config.table('research_figure')}
        WHERE figure_id = :figure_id
        """,
        {"figure_id": figure_id},
    )
    return rows[0] if rows else None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_coverage(_config: Config) -> list[dict[str, Any]]:
    """Corpus size, freshness and review backlog by source type."""
    return query(
        f"SELECT * FROM {_config.table('v_source_coverage')} ORDER BY source_count DESC"
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_open_questions(_config: Config) -> list[dict[str, Any]]:
    """Evidence-backed gaps the corpus has surfaced about itself."""
    return query(
        f"SELECT * FROM {_config.table('v_research_open_questions')} "
        f"ORDER BY evidence_count DESC LIMIT 25"
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_discovery_freshness(_config: Config) -> list[dict[str, Any]]:
    """When each standing query was last swept, so staleness is visible."""
    try:
        return query(
            f"SELECT * FROM {_config.table('v_discovery_freshness')} "
            f"ORDER BY days_since_sweep DESC NULLS FIRST LIMIT 25"
        )
    except Exception:  # noqa: BLE001 - the view is optional in some deployments
        return []


#: Tables/views a curator may browse directly, with a sensible default sort.
EXPLORABLE_TABLES: dict[str, str] = {
    "research_claim": "claim_id",
    "research_source": "source_id",
    "research_chunk": "chunk_id",
    "research_claim_relationship": "relationship_id",
    "research_review_queue": "review_id",
}


def fetch_table(config: Config, table_name: str, *, search: str, limit: int) -> list[dict[str, Any]]:
    """Load rows from an explorable table, optionally filtered by a text search."""
    order_by = EXPLORABLE_TABLES[table_name]
    full_table = config.table(table_name)
    if not search:
        return query(f"SELECT * FROM {full_table} ORDER BY {order_by} DESC LIMIT {int(limit)}")
    columns = [c["col_name"] for c in query(f"DESCRIBE TABLE {full_table}") if not c["col_name"].startswith("#")]
    text_columns = [
        c for c in columns
        if c not in ("metric_value", "extraction_confidence", "page_number", "chunk_index")
    ]
    predicate = " OR ".join(f"CAST({c} AS STRING) ILIKE :needle" for c in text_columns)
    return query(
        f"SELECT * FROM {full_table} WHERE {predicate} ORDER BY {order_by} DESC LIMIT {int(limit)}",
        {"needle": f"%{search}%"},
    )


def record_decision(
    config: Config,
    *,
    claim_id: str,
    review_id: str,
    decision: str,
    reviewer: str,
    note: str,
    amendments: dict[str, Any],
) -> None:
    """Apply a review decision to the claim and close its queue item.

    The two writes mirror ``review.queue.apply_claim_decision`` against SQL, and
    the rules come from the same validator that path uses.
    """
    # The same validator the pipeline uses. Repeating the rules here in different
    # words is how a UI quietly ends up with a laxer boundary than the code.
    validate_decision(
        decision, reviewer=reviewer, note=note.strip() or None, amendments=amendments
    )
    assert set(amendments) <= AMENDABLE_FIELDS  # validate_decision guarantees this

    status = ReviewStatus.REJECTED.value if decision == "REJECTED" else ReviewStatus.REVIEWED.value
    sets = ", ".join(f"{f} = :{f}" for f in amendments)
    execute(
        f"""
        UPDATE {config.table('research_claim')}
        SET review_status = :status, reviewed_by = :reviewer,
            reviewed_at = current_timestamp(), review_note = :note
            {(', ' + sets) if sets else ''}
        WHERE claim_id = :claim_id AND review_status IN ('CANDIDATE', 'IN_REVIEW')
        """,
        {
            "status": status,
            "reviewer": reviewer,
            "note": note or None,
            "claim_id": claim_id,
            **amendments,
        },
    )
    execute(
        f"""
        UPDATE {config.table('research_review_queue')}
        SET status = :decision, assigned_to = :reviewer,
            resolved_at = current_timestamp(), resolution_note = :note
        WHERE review_id = :review_id
        """,
        {
            "decision": "ACCEPTED" if decision == "ACCEPTED" else decision,
            "reviewer": reviewer,
            "note": note or None,
            "review_id": review_id,
        },
    )


# ---------------------------------------------------------------------------
# UI — shared chrome
# ---------------------------------------------------------------------------


def inject_style() -> None:
    """Plain, dense styling: no emoji, small type, minimal colour."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; max-width: 1100px;}
        div[data-testid="stMetricValue"] {font-size: 1.3rem;}
        .rd-claim-text {font-size: 1rem; font-weight: 600; line-height: 1.4;}
        .rd-caption {color: #6b7280; font-size: 0.82rem;}
        .rd-badge-high {color: #b91c1c; font-weight: 600;}
        .rd-badge-normal {color: #92400e; font-weight: 600;}
        .rd-badge-low {color: #6b7280; font-weight: 600;}
        .rd-genie-panel {
            border: 1px solid #d1d5db; border-radius: 6px; padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .rd-genie-link {
            display: inline-block; padding: 0.5rem 1rem; border-radius: 4px;
            background: #1a1a1a; color: #fff !important; text-decoration: none;
            font-weight: 600; font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: Config, reviewer_label: str) -> str:
    """Identity, filters and a compact corpus pulse. Returns the chosen priority filter."""
    with st.sidebar:
        st.markdown(f"**{reviewer_label}**")
        st.caption(f"`{config.fq_schema}`")
        st.divider()

        priority = st.selectbox(
            "Priority filter", ["ALL", "HIGH", "NORMAL", "LOW"],
            help="HIGH covers low-confidence extractions, numeric claims without an "
            "excerpt, and anything missing most of its scope.",
        )
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("**Corpus at a glance**")
        try:
            for row in fetch_coverage(config):
                reviewed = row.get("reviewed_claim_count") or 0
                pending = row.get("unreviewed_claim_count") or 0
                total = reviewed + pending
                share = reviewed / total if total else 0.0
                st.caption(f"{row['source_type']} — {reviewed} reviewed / {pending} pending")
                st.progress(share)
        except Exception as exc:  # noqa: BLE001 - sidebar must not crash the page
            logger.exception("sidebar coverage query failed")
            st.caption(f"Coverage unavailable: {exc}")

        st.divider()
        st.caption(
            "A claim is invisible to the agent until it is REVIEWED here. "
            "Amendments may only correct scope and citation fields — never the "
            "source's reported finding."
        )
    return priority


def render_connection_error(exc: Exception) -> None:
    """A calm, actionable error instead of an endless default spinner."""
    logger.exception("connection/query failed")
    st.error(
        "Couldn't reach the SQL warehouse. If it's been idle, a serverless warehouse can "
        "take up to a minute to start. If this keeps happening right after a fresh sign-in, "
        "try reloading once more — the very first load after your session starts sometimes "
        "races the authorization handshake."
    )
    with st.expander("Details"):
        st.code(str(exc))
    cols = st.columns([1, 5])
    with cols[0]:
        if st.button("Retry now"):
            st.cache_resource.clear()
            st.rerun()
    st.caption("This page checks again automatically in a few seconds…")
    time.sleep(4)
    st.rerun()


# ---------------------------------------------------------------------------
# UI — review queue
# ---------------------------------------------------------------------------


def render_claim(config: Config, item: dict[str, Any], reviewer: str) -> None:
    """Render one claim with its evidence and the decision controls."""
    badge = PRIORITY_BADGE.get(item["priority"], item["priority"])
    badge_class = PRIORITY_CSS_CLASS.get(item["priority"], "rd-badge-normal")
    header_left, header_right = st.columns([5, 2])
    with header_left:
        st.markdown(f"<div class='rd-claim-text'>{item['claim_text']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='rd-caption'>{item['source_title']} · {item['source_type']} · "
            f"page {item['page_number']} · <a href='{item['source_url']}'>source</a></span>",
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(f"<span class='{badge_class}'>{badge}</span>", unsafe_allow_html=True)
        confidence = item.get("extraction_confidence")
        st.metric("Extractor confidence", f"{confidence:.0%}" if confidence is not None else "n/a")

    st.caption(f"Queued because: {item['reason']} · extracted by {item['extractor_name']} {item['extractor_version']}")
    if item.get("missing_field_reason"):
        st.error(f"Missing field(s): {item['missing_field_reason']}")

    with st.expander("Evidence", expanded=False):
        if item.get("figure_id"):
            figure = fetch_figure(config, item["figure_id"])
            st.warning(
                "Read from a figure by a vision model. A chart reading is an interpretation, "
                "not a stated number — check it against the image before accepting."
            )
            if figure:
                st.write(
                    f"**Vision model:** {figure['vision_model']} "
                    f"(confidence {figure['extraction_confidence']:.0%})"
                )
                if figure.get("image_uri"):
                    st.image(figure["image_uri"], caption=figure.get("caption") or "")
                st.code(figure.get("extracted_text") or "", language=None)
        else:
            evidence = fetch_evidence(config, item["claim_id"])
            if evidence and evidence.get("text"):
                if evidence.get("extraction_warning"):
                    st.warning(f"Parser warning: {evidence['extraction_warning']}")
                st.markdown(f"**{evidence.get('section_title') or 'Source passage'}**")
                st.info(evidence["text"])
            elif item.get("evidence_excerpt"):
                st.caption("No stored chunk for this claim; showing its recorded evidence excerpt.")
                st.info(item["evidence_excerpt"])
            else:
                st.caption("No stored passage or excerpt for this claim.")

    st.markdown("**Scope** — the fields that decide what this claim can be compared with")
    amendments: dict[str, Any] = {}
    columns = st.columns(3)
    for index, field_name in enumerate(DISPLAY_FIELDS):
        with columns[index % 3]:
            current = item.get(field_name)
            new = st.text_input(
                field_name,
                value=current or "",
                key=f"{item['claim_id']}:{field_name}",
                placeholder="not stated",
            )
            if (new or None) != current:
                amendments[field_name] = new or None

    if item.get("metric_value") is not None:
        st.caption(
            f"Reported value: {item['metric_value']} {item.get('metric_unit') or ''} — "
            "the value is not amendable here; reject the claim if it misreads the source."
        )

    note = st.text_area("Reviewer note", key=f"note:{item['claim_id']}")

    accept, amend, reject = st.columns(3)
    decision = None
    if accept.button("Accept", key=f"a:{item['claim_id']}", use_container_width=True):
        decision = "ACCEPTED"
    if amend.button("Save amendments", key=f"m:{item['claim_id']}", use_container_width=True):
        decision = "AMENDED"
    if reject.button("Reject", key=f"r:{item['claim_id']}", use_container_width=True):
        decision = "REJECTED"

    if decision:
        try:
            record_decision(
                config,
                claim_id=item["claim_id"],
                review_id=item["review_id"],
                decision=decision,
                reviewer=reviewer,
                note=note,
                amendments=amendments if decision == "AMENDED" else {},
            )
        except (ValueError, ReviewError) as exc:
            st.error(str(exc))
        else:
            st.success(f"{decision}. The claim is now {'rejected' if decision == 'REJECTED' else 'reviewed'}.")
            st.cache_data.clear()
            st.rerun()


def render_queue_tab(config: Config, priority: str, reviewer: str) -> None:
    queue = fetch_queue(config, priority)
    if not queue:
        st.success("The review queue is empty for this filter. Every extracted claim has a decision.")
        return

    counts: dict[str, int] = {}
    for row in queue:
        counts[row["priority"]] = counts.get(row["priority"], 0) + 1
    metric_cols = st.columns(4)
    metric_cols[0].metric("Awaiting review", len(queue))
    metric_cols[1].metric("High priority", counts.get("HIGH", 0))
    metric_cols[2].metric("Normal priority", counts.get("NORMAL", 0))
    metric_cols[3].metric("Low priority", counts.get("LOW", 0))
    st.divider()

    for item in queue:
        with st.container(border=True):
            render_claim(config, item, reviewer)


# ---------------------------------------------------------------------------
# UI — corpus overview + open questions
# ---------------------------------------------------------------------------


def render_overview_tab(config: Config) -> None:
    coverage = fetch_coverage(config)
    if not coverage:
        st.info("No sources ingested yet.")
        return

    total_sources = sum(r.get("source_count") or 0 for r in coverage)
    total_reviewed = sum(r.get("reviewed_claim_count") or 0 for r in coverage)
    total_pending = sum(r.get("unreviewed_claim_count") or 0 for r in coverage)
    total_claims = total_reviewed + total_pending
    reviewed_share = total_reviewed / total_claims if total_claims else 0.0

    cols = st.columns(4)
    cols[0].metric("Sources", total_sources)
    cols[1].metric("Reviewed claims", total_reviewed)
    cols[2].metric("Pending review", total_pending)
    cols[3].metric("Reviewed share", f"{reviewed_share:.0%}")
    if reviewed_share < 0.5:
        st.warning(
            "Less than half the corpus is reviewed. Any synthesis the agent gives right now "
            "should be qualified as provisional-heavy."
        )

    st.markdown("##### Coverage by source type")
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    freshness = fetch_discovery_freshness(config)
    if freshness:
        st.markdown("##### Discovery freshness")
        st.caption("When each standing query last swept the scholarly APIs.")
        st.dataframe(freshness, use_container_width=True, hide_index=True)


def render_open_questions_tab(config: Config) -> None:
    questions = fetch_open_questions(config)
    if not questions:
        st.info("No evidence-backed open questions yet — the corpus needs more reviewed claims first.")
        return
    st.caption(
        "Derived from the corpus itself; the agent may never state a gap that isn't listed here."
    )
    for q in questions:
        with st.container(border=True):
            st.markdown(f"**{q['question_text']}**")
            st.caption(
                f"{q['question_type']} · task: {q.get('task') or '—'} · "
                f"metric: {q.get('metric') or '—'} · benchmark: {q.get('benchmark') or '—'} · "
                f"backed by {q['evidence_count']} claim(s)"
            )


# ---------------------------------------------------------------------------
# UI — data explorer
# ---------------------------------------------------------------------------


def render_data_explorer_tab(config: Config) -> None:
    """Raw table access: pick a table, optionally filter, click Load."""
    st.caption("Direct, read-only access to the corpus tables backing the agent and the queue above.")
    top = st.columns([2, 3, 1, 1])
    table_name = top[0].selectbox("Table", list(EXPLORABLE_TABLES), key="explorer_table")
    search = top[1].text_input("Filter (matches any text column)", key="explorer_search")
    limit = top[2].number_input("Rows", min_value=10, max_value=2000, value=200, step=10, key="explorer_limit")
    load = top[3].button("Load", type="primary", use_container_width=True)

    state_key = f"explorer_result:{table_name}"
    if load:
        with st.spinner("Querying..."):
            try:
                st.session_state[state_key] = fetch_table(config, table_name, search=search, limit=int(limit))
            except Exception as exc:  # noqa: BLE001 - shown inline, not a page crash
                logger.exception("data explorer query failed")
                st.session_state[state_key] = None
                st.error(f"Query failed: {exc}")

    rows = st.session_state.get(state_key)
    if rows is None:
        st.info("Choose a table and click Load.")
    elif not rows:
        st.info("No rows matched.")
    else:
        st.caption(f"{len(rows)} row(s)")
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# app entry point
# ---------------------------------------------------------------------------


#: Starting points for a curator who doesn't yet know what the corpus can answer.
SAMPLE_QUESTIONS = (
    "What reviewed claims do we have about GraphRAG's token cost?",
    "Which claims contradict each other?",
    "What open questions has the corpus surfaced?",
    "Summarize what is known about HippoRAG's retrieval performance.",
)


def render_genie_panel() -> None:
    """The primary surface: an embedded chat against the deployed Genie Agent,
    authenticated as the signed-in user (OBO) - the same Genie Space you'd reach
    through the Databricks workspace UI, run from inside this page.
    """
    st.markdown('<div class="rd-genie-panel">', unsafe_allow_html=True)
    st.markdown("#### Ask the research agent")
    st.caption(
        "Answers cite claim records and source URLs, and refuse to compare claims whose "
        "scope doesn't overlap. Backed by the same reviewed-claims corpus as the workstation below."
    )

    client = get_genie_client()
    url = genie_space_url()
    if client is None:
        st.warning(
            "Genie chat isn't available from inside this page (RD_GENIE_SPACE_ID missing, or the "
            "app's 'genie' authorization scope hasn't been granted yet)."
        )
        if url:
            st.markdown(f'<a class="rd-genie-link" href="{url}" target="_blank">Open Genie agent</a>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    history: list[dict[str, str]] = st.session_state.setdefault("genie_history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if not history:
        st.caption("Try asking:")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, sample in zip(cols, SAMPLE_QUESTIONS):
            if col.button(sample, key=f"sample:{sample}", use_container_width=True):
                st.session_state["genie_pending_question"] = sample
                st.rerun()

    question = st.chat_input("Ask a question about the reviewed corpus")
    pending = st.session_state.pop("genie_pending_question", None)
    question = question or pending
    if question:
        history.append({"role": "user", "content": question})
        with st.spinner("Genie is thinking..."):
            turn = client.ask(question, conversation_id=st.session_state.get("genie_conversation_id"))
        if turn.conversation_id:
            st.session_state["genie_conversation_id"] = turn.conversation_id
        if turn.error:
            answer = f"Genie could not answer: {turn.error}"
        else:
            answer = turn.text or "(no answer text returned)"
            if turn.queries:
                answer += "\n\n---\n" + "\n".join(f"`{q}`" for q in turn.queries)
        history.append({"role": "assistant", "content": answer})
        st.rerun()

    cols = st.columns([1, 5])
    if history and cols[0].button("New conversation"):
        st.session_state["genie_history"] = []
        st.session_state.pop("genie_conversation_id", None)
        st.rerun()
    if url:
        cols[1].markdown(f'<a class="rd-genie-link" href="{url}" target="_blank">Open in full Genie UI</a>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """App entry point."""
    st.set_page_config(page_title="Research Discovery", layout="wide")
    inject_style()
    config = load_config()
    reviewer = reviewer_identity()
    reviewer_label = reviewer_display_name(reviewer)

    st.title("Research discovery")
    render_genie_panel()

    if not reviewer:
        st.error(
            "Could not identify you. An unattributed review is not a review, so decisions are "
            "disabled. Sign in through the app, or set RD_REVIEWER when running locally."
        )
        return

    priority = render_sidebar(config, reviewer_label)

    st.markdown("---")
    st.markdown("#### Review workstation")
    st.caption(
        "Secondary to the agent above: this is where extracted candidate claims are "
        "accepted, amended or rejected before Genie may cite them."
    )
    with st.expander("What is this section?", expanded=False):
        st.markdown(
            "- The research pipeline **extracts candidate claims** from papers, benchmark docs "
            "and repos - but nobody has checked them yet.\n"
            "- **You are that check.** Every claim here needs a human decision before the Genie "
            "Agent may ever cite it as a finding.\n"
            "- **Review queue** - claims waiting on you: accept, amend the scope fields, or "
            "reject, with a note either way.\n"
            "- **Corpus overview** - how big the corpus is and how much of it is actually "
            "reviewed vs. still pending.\n"
            "- **Open questions** - evidence-backed gaps the corpus has found in itself.\n"
            "- **Data explorer** - browse the raw corpus tables directly: pick a table, "
            "optionally filter, click Load.\n"
            "- Amending never rewrites what a source claims to have found - only the scope "
            "fields (task, method, metric, benchmark, condition) that decide what it may be "
            "compared against."
        )

    try:
        tab_queue, tab_overview, tab_questions, tab_explorer = st.tabs(
            ["Review queue", "Corpus overview", "Open questions", "Data explorer"]
        )
        with tab_queue:
            render_queue_tab(config, priority, reviewer)
        with tab_overview:
            render_overview_tab(config)
        with tab_questions:
            render_open_questions_tab(config)
        with tab_explorer:
            render_data_explorer_tab(config)
    except Exception as exc:  # noqa: BLE001 - surfaced as a retry-able connection error
        render_connection_error(exc)


if __name__ == "__main__":
    main()

