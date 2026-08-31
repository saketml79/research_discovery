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

import json
import logging
import os
from typing import Any

import streamlit as st

from research_discovery.config import Config
from research_discovery.models import ReviewStatus
from research_discovery.review.queue import AMENDABLE_FIELDS, ReviewError, validate_decision

logger = logging.getLogger(__name__)

#: Ordered for display; the set of what may be amended comes from review.queue,
#: so the app and the pipeline cannot drift apart on the rules.
DISPLAY_FIELDS = ("task", "method", "metric", "benchmark", "condition_text", "metric_unit")


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


@st.cache_resource
def get_connection() -> Any:
    """Open a SQL warehouse connection using the app's service principal."""
    from databricks import sql  # noqa: PLC0415

    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
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
# UI
# ---------------------------------------------------------------------------


def render_claim(config: Config, item: dict[str, Any], reviewer: str) -> None:
    """Render one claim with its evidence and the decision controls."""
    st.subheader(item["claim_text"])

    left, right = st.columns([2, 1])
    with left:
        st.caption(
            f"{item['source_title']} · {item['source_type']} · page {item['page_number']} · "
            f"[source]({item['source_url']})"
        )
        if item.get("figure_id"):
            figure = fetch_figure(config, item["figure_id"])
            st.warning(
                "Read from a figure by a vision model. A chart reading is an interpretation, "
                "not a stated number — check it against the image before accepting."
            )
            if figure:
                st.write(f"**Vision model:** {figure['vision_model']} "
                         f"(confidence {figure['extraction_confidence']:.0%})")
                if figure.get("image_uri"):
                    st.image(figure["image_uri"], caption=figure.get("caption") or "")
                st.code(figure.get("extracted_text") or "", language=None)
        else:
            evidence = fetch_evidence(config, item["claim_id"])
            if evidence:
                if evidence.get("extraction_warning"):
                    st.warning(f"Parser warning: {evidence['extraction_warning']}")
                st.markdown("**Source passage**")
                st.info(evidence["text"])

    with right:
        confidence = item.get("extraction_confidence")
        st.metric("Extractor confidence", f"{confidence:.0%}" if confidence is not None else "n/a")
        st.caption(f"{item['extractor_name']} · {item['extractor_version']}")
        st.caption(f"Queued: {item['reason']}")
        if item.get("missing_field_reason"):
            st.error(item["missing_field_reason"])

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


def main() -> None:
    """App entry point."""
    st.set_page_config(page_title="Research Claim Review", layout="wide")
    config = load_config()
    reviewer = reviewer_identity()

    st.title("Research claim review")
    st.caption(
        f"{config.fq_schema} — nothing here is visible to the agent until you accept it."
    )

    if not reviewer:
        st.error(
            "Could not identify you. An unattributed review is not a review, so decisions are "
            "disabled. Sign in through the app, or set RD_REVIEWER when running locally."
        )
        return

    with st.sidebar:
        st.write(f"Reviewing as **{reviewer}**")
        priority = st.selectbox("Priority", ["ALL", "HIGH", "NORMAL", "LOW"])
        st.divider()
        coverage = query(
            f"SELECT source_type, reviewed_claim_count, unreviewed_claim_count "
            f"FROM {config.table('v_source_coverage')}"
        )
        st.markdown("**Corpus**")
        for row in coverage:
            st.caption(
                f"{row['source_type']}: {row['reviewed_claim_count']} reviewed / "
                f"{row['unreviewed_claim_count']} pending"
            )

    queue = fetch_queue(config, priority)
    if not queue:
        st.success("The review queue is empty. Every extracted claim has a decision.")
        return

    st.write(f"{len(queue)} claim(s) awaiting review")
    for item in queue:
        with st.container(border=True):
            render_claim(config, item, reviewer)


if __name__ == "__main__":
    main()
