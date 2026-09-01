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
    # DATABRICKS_HOST in the Apps runtime is schemeless (e.g. "dbc-x.cloud.
    # databricks.com"); an href without a scheme is resolved as a relative
    # path against the app's own origin instead of opening Databricks.
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"https://{host}"
    return f"{host.rstrip('/')}/genie/rooms/{space_id}"


@st.cache_resource(show_spinner=False)
def _genie_client(user_token: str, space_id: str) -> Any:
    """Build a Genie Agent client authenticated as the signed-in user (OBO).

    Requires the app's ``genie`` user-authorization scope, in addition to ``sql``.
    """
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    from research_discovery.agent.client import GenieAgentClient, WorkspaceGenieConversationApi

    # auth_type is forced because the app's own service-principal OAuth env vars
    # (DATABRICKS_CLIENT_ID/SECRET) are always present alongside this user token;
    # left ambiguous, Config refuses to pick between the two auth methods.
    workspace_client = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"], token=user_token, auth_type="pat"
    )
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
    """A real visual system: a consistent palette, card shadows on every
    bordered container, rounded pill tabs/buttons and chat-bubble styling -
    rather than the previous flat, unstyled defaults.
    """
    st.markdown(
        """
        <style>
        :root {
            --rd-accent: #4f46e5;
            --rd-accent-dark: #4338ca;
            --rd-bg: #f6f7fb;
            --rd-card: #ffffff;
            --rd-border: #e5e7eb;
            --rd-text: #111827;
            --rd-muted: #6b7280;
            --rd-shadow: 0 1px 2px rgba(16,24,40,.04), 0 4px 12px rgba(16,24,40,.06);
        }
        .stApp {background: var(--rd-bg);}
        .block-container {padding-top: 1rem; max-width: 1100px;}
        h1, h2, h3, h4 {color: var(--rd-text); letter-spacing: -0.01em;}

        .rd-hero {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border-radius: 20px; padding: 1.6rem 2rem; margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(79,70,229,.25);
        }
        .rd-hero h1 {color: #fff; margin: 0; font-size: 1.9rem;}
        .rd-hero p {color: #e0e7ff; margin: 0.3rem 0 0; font-size: 0.92rem;}

        div[data-testid="stMetricValue"] {font-size: 1.4rem; font-weight: 700; color: var(--rd-text);}
        div[data-testid="stMetric"] {
            background: var(--rd-card); border-radius: 14px; padding: 0.9rem 1.1rem;
            box-shadow: var(--rd-shadow); border: 1px solid var(--rd-border);
            border-top: 3px solid var(--rd-accent); transition: transform 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {transform: translateY(-2px);}

        /* Any bordered st.container - the one legal way to draw a real card,
           since Streamlit can't nest HTML written across separate calls. */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important; border: 1px solid var(--rd-border) !important;
            box-shadow: var(--rd-shadow); background: var(--rd-card);
            transition: box-shadow 0.15s ease;
        }

        /* Buttons: one consistent pill style, accent on primary/submit. */
        .stButton > button, .stFormSubmitButton > button, .stLinkButton > a {
            border-radius: 10px !important; border: 1px solid var(--rd-border) !important;
            font-weight: 600; transition: all 0.15s ease;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover, .stLinkButton > a:hover {
            border-color: var(--rd-accent) !important; color: var(--rd-accent) !important;
        }
        .stFormSubmitButton > button {
            background: var(--rd-accent) !important; color: #fff !important; border: none !important;
        }
        .stFormSubmitButton > button:hover {background: var(--rd-accent-dark) !important; color: #fff !important;}
        .stLinkButton > a {background: var(--rd-text) !important; color: #fff !important; border: none !important;}
        .stLinkButton > a:hover {color: #fff !important; opacity: 0.85;}

        /* Pill-style tabs instead of the flat default underline tabs. */
        button[data-baseweb="tab"] {
            border-radius: 999px !important; padding: 0.35rem 1.1rem !important;
            font-weight: 600; color: var(--rd-muted);
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--rd-accent) !important; color: #fff !important;
        }
        div[data-baseweb="tab-highlight"] {display: none;}
        div[data-baseweb="tab-border"] {display: none;}

        /* Chat bubbles: rounded cards with a soft shadow, right-aligned for
           the user (like a messaging app) and left-aligned for Genie. */
        div[data-testid="stChatMessage"] {
            border-radius: 16px; padding: 0.6rem 0.9rem; margin-bottom: 0.6rem;
            box-shadow: var(--rd-shadow); border: 1px solid var(--rd-border);
            max-width: 78%;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            background: var(--rd-accent); color: #fff; margin-left: auto;
            flex-direction: row-reverse; border: none;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) p {
            color: #fff;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            margin-right: auto;
        }

        div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div {
            border-radius: 10px !important;
        }

        .rd-claim-text {font-size: 1rem; font-weight: 700; line-height: 1.4; color: var(--rd-text);}
        .rd-caption {color: var(--rd-muted); font-size: 0.82rem;}
        .rd-badge {
            display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;
        }
        .rd-badge-high {color: #b91c1c; background: #fee2e2;}
        .rd-badge-normal {color: #92400e; background: #fef3c7;}
        .rd-badge-low {color: #374151; background: #f3f4f6;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_identity_bar(config: Config, reviewer_label: str) -> str:
    """Identity and priority filter, inline at the top of the review tab (no sidebar)."""
    cols = st.columns([2, 2, 1, 1])
    cols[0].caption(f"Signed in as **{reviewer_label}**")
    priority = cols[1].selectbox(
        "Priority filter", ["ALL", "HIGH", "NORMAL", "LOW"], label_visibility="collapsed",
        help="HIGH covers low-confidence extractions, numeric claims without an "
        "excerpt, and anything missing most of its scope.",
    )
    if cols[2].button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
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
        st.markdown(f"<span class='rd-badge {badge_class}'>{badge}</span>", unsafe_allow_html=True)
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
    st.caption(
        "Answers cite claim records and source URLs, and refuse to compare claims whose "
        "scope doesn't overlap. Backed by the reviewed-claims corpus (see the Review "
        "workstation tab)."
    )

    client = get_genie_client()
    url = genie_space_url()
    if client is None:
        st.warning(
            "Genie chat isn't available from inside this page (RD_GENIE_SPACE_ID missing, or the "
            "app's 'genie' authorization scope hasn't been granted yet)."
        )
        if url:
            st.link_button("Open Genie agent", url)
        return

    history: list[dict[str, Any]] = st.session_state.setdefault("genie_history", [])
    pending = st.session_state.pop("genie_pending_question", None)
    inflight = st.session_state.get("genie_inflight_question")
    busy = inflight is not None

    with st.container(border=True):
        if not history:
            st.caption("Try asking:")
            cols = st.columns(2)
            for index, sample in enumerate(SAMPLE_QUESTIONS):
                if cols[index % 2].button(
                    sample, key=f"sample:{sample}", use_container_width=True, disabled=busy
                ):
                    pending = sample
        else:
            for turn in history:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])
                    if turn.get("rows"):
                        st.dataframe(
                            turn["rows"], use_container_width=True, hide_index=True,
                            column_config=None,
                        )
                    if turn.get("queries"):
                        with st.expander("SQL Genie ran"):
                            for query in turn["queries"]:
                                st.code(query, language="sql")

        with st.form("genie_ask_form", clear_on_submit=True, border=False):
            input_cols = st.columns([5, 1])
            question = input_cols[0].text_input(
                "Ask a question", label_visibility="collapsed",
                placeholder="Ask a question about the reviewed corpus", disabled=busy,
            )
            asked = input_cols[1].form_submit_button("Ask", use_container_width=True, disabled=busy)

        # Two-phase ask: first rerun shows the question + a disabled/busy UI,
        # the *next* run does the actual (slow) Genie call and appends the
        # answer - a single-pass "set busy then immediately clear it" never
        # reaches the browser, since Streamlit only repaints between reruns.
        if not busy and (pending or (asked and question)):
            question = pending or question
            history.append({"role": "user", "content": question})
            st.session_state["genie_inflight_question"] = question
            st.rerun()

        if busy:
            with st.chat_message("assistant"):
                with st.spinner("Genie is thinking..."):
                    try:
                        turn = client.ask(
                            inflight, conversation_id=st.session_state.get("genie_conversation_id")
                        )
                    except Exception as exc:  # noqa: BLE001 - never crash the page over one turn
                        logger.exception("genie ask failed")
                        entry: dict[str, Any] = {
                            "role": "assistant",
                            "content": f"Genie could not answer: {exc}",
                        }
                    else:
                        if turn.conversation_id:
                            st.session_state["genie_conversation_id"] = turn.conversation_id
                        answer = (
                            f"Genie could not answer: {turn.error}"
                            if turn.error
                            else (turn.text or "(no answer text returned)")
                        )
                        entry = {"role": "assistant", "content": answer}
                        if not turn.error and turn.rows:
                            entry["rows"] = (
                                [dict(zip(turn.columns, row)) for row in turn.rows]
                                if turn.columns
                                else turn.rows
                            )
                        if not turn.error and turn.queries:
                            entry["queries"] = turn.queries
            history.append(entry)
            st.session_state.pop("genie_inflight_question", None)
            st.rerun()

    cols = st.columns([1, 5])
    if history and cols[0].button("New conversation", disabled=busy):
        st.session_state["genie_history"] = []
        st.session_state.pop("genie_conversation_id", None)
        st.rerun()
    if url:
        cols[1].link_button("Open in full Genie UI", url)


def render_review_workstation(config: Config, reviewer: str, reviewer_label: str) -> None:
    """Secondary surface: accept/amend/reject candidate claims before Genie can cite them."""
    st.caption(
        "Extracted candidate claims are accepted, amended or rejected here before Genie "
        "(the tab next to this one) may cite them."
    )
    with st.expander("What is this tab?", expanded=False):
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

    if not reviewer:
        st.error(
            "Could not identify you. An unattributed review is not a review, so decisions are "
            "disabled. Sign in through the app, or set RD_REVIEWER when running locally."
        )
        return

    priority = render_identity_bar(config, reviewer_label)

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


def main() -> None:
    """App entry point."""
    st.set_page_config(page_title="Research Discovery", layout="wide")
    inject_style()
    config = load_config()
    reviewer = reviewer_identity()
    reviewer_label = reviewer_display_name(reviewer)

    st.markdown(
        '<div class="rd-hero"><h1>Research discovery</h1>'
        "<p>A governed research agent over a human-reviewed claims corpus.</p></div>",
        unsafe_allow_html=True,
    )

    tab_genie, tab_review = st.tabs(["Ask Genie", "Review workstation"])
    with tab_genie:
        render_genie_panel()
    with tab_review:
        render_review_workstation(config, reviewer, reviewer_label)


if __name__ == "__main__":
    main()

