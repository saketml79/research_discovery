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

logger = logging.getLogger(__name__)

#: Ordered for display; the set of what may be amended comes from review.queue,
#: so the app and the pipeline cannot drift apart on the rules.
DISPLAY_FIELDS = ("task", "method", "metric", "benchmark", "condition_text", "metric_unit")

#: Colour + label for each queue priority. Centralised so a new priority value
#: only needs one new line, not a scattered set of if/else blocks.
PRIORITY_BADGE: dict[str, str] = {
    "HIGH": "🔴 HIGH",
    "NORMAL": "🟡 NORMAL",
    "LOW": "⚪ LOW",
}


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Connecting to the SQL warehouse (a cold start can take up to a minute)…")
def get_connection() -> Any:
    """Open a SQL warehouse connection using the app's service principal."""
    from databricks import sql  # noqa: PLC0415

    http_path = os.environ.get("DATABRICKS_HTTP_PATH") or (
        f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}"
    )
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=http_path,
        access_token=os.environ.get("DATABRICKS_TOKEN"),
    )


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


def reviewer_identity() -> str:
    """Identify the reviewer.

    Databricks Apps forward the signed-in user's identity in a header. An
    unattributed review is not a review, so the app refuses to record decisions
    when it cannot tell who is making them.
    """
    header = st.context.headers if hasattr(st, "context") else {}
    return (
        header.get("X-Forwarded-Email")
        or header.get("X-Forwarded-Preferred-Username")
        or os.environ.get("RD_REVIEWER", "")
    )


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
    """A few CSS touches so the review queue reads like a worklist, not a form dump."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1200px;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        .rd-claim-text {font-size: 1.15rem; font-weight: 600; line-height: 1.4;}
        .rd-caption {color: #6b7280; font-size: 0.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: Config, reviewer: str) -> str:
    """Identity, filters and a compact corpus pulse. Returns the chosen priority filter."""
    with st.sidebar:
        st.markdown(f"#### 👤 {reviewer or 'Unknown reviewer'}")
        st.caption(f"`{config.fq_schema}`")
        st.divider()

        priority = st.selectbox(
            "Priority filter", ["ALL", "HIGH", "NORMAL", "LOW"],
            help="HIGH covers low-confidence extractions, numeric claims without an "
            "excerpt, and anything missing most of its scope.",
        )
        if st.button("🔄 Refresh", use_container_width=True):
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
    st.error(
        "Couldn't reach the SQL warehouse yet. If it's been idle, a serverless "
        "warehouse can take up to a minute to start."
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
    header_left, header_right = st.columns([5, 2])
    with header_left:
        st.markdown(f"<div class='rd-claim-text'>{item['claim_text']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='rd-caption'>{item['source_title']} · {item['source_type']} · "
            f"page {item['page_number']} · <a href='{item['source_url']}'>source</a></span>",
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(f"**{badge}**")
        confidence = item.get("extraction_confidence")
        st.metric("Extractor confidence", f"{confidence:.0%}" if confidence is not None else "n/a")

    st.caption(f"Queued because: {item['reason']} · extracted by {item['extractor_name']} {item['extractor_version']}")
    if item.get("missing_field_reason"):
        st.error(f"Missing field(s): {item['missing_field_reason']}")

    with st.expander("📄 Evidence", expanded=False):
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
            if evidence:
                if evidence.get("extraction_warning"):
                    st.warning(f"Parser warning: {evidence['extraction_warning']}")
                st.markdown(f"**{evidence.get('section_title') or 'Source passage'}**")
                st.info(evidence["text"])
            else:
                st.caption("No stored passage for this claim.")

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
    if accept.button("✅ Accept", key=f"a:{item['claim_id']}", use_container_width=True):
        decision = "ACCEPTED"
    if amend.button("✏️ Save amendments", key=f"m:{item['claim_id']}", use_container_width=True):
        decision = "AMENDED"
    if reject.button("🚫 Reject", key=f"r:{item['claim_id']}", use_container_width=True):
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
# app entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """App entry point."""
    st.set_page_config(page_title="Research Claim Review", layout="wide", page_icon="🔬")
    inject_style()
    config = load_config()
    reviewer = reviewer_identity()

    st.title("🔬 Research claim review")
    st.caption(
        "The boundary between extracted candidate knowledge and what the Genie Agent may "
        "assert as a finding. Nothing below is visible to the agent until it is REVIEWED."
    )

    if not reviewer:
        st.error(
            "Could not identify you. An unattributed review is not a review, so decisions are "
            "disabled. Sign in through the app, or set RD_REVIEWER when running locally."
        )
        return

    priority = render_sidebar(config, reviewer)

    try:
        tab_queue, tab_overview, tab_questions = st.tabs(
            ["📋 Review queue", "📊 Corpus overview", "❓ Open questions"]
        )
        with tab_queue:
            render_queue_tab(config, priority, reviewer)
        with tab_overview:
            render_overview_tab(config)
        with tab_questions:
            render_open_questions_tab(config)
    except Exception as exc:  # noqa: BLE001 - surfaced as a retry-able connection error
        render_connection_error(exc)


if __name__ == "__main__":
    main()

