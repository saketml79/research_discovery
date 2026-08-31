# Research Discovery Engine

A governed, claims-aware research workspace for Databricks, built as **Build 3** of *Three
Governed Genie Agent Builds for a Weekend*.

Sources are registered and versioned, parsed into page-scoped chunks, converted into
structured claims with explicit scope, gated behind human review, and served to a Genie
Agent that may only cite reviewed claims — and may only call two claims contradictory when
their scope actually overlaps.

The point is not corpus size. The point is that the system can tell a paper's **topic**
from its **claim**, and can tell a real disagreement from two results that were never
comparable in the first place.

---

## What makes this different from a RAG demo

| Concern | How it is enforced | Where |
| --- | --- | --- |
| A claim is not a topic | Extraction schema demands task, method, metric, benchmark and conditions; a missing field requires an explicit reason | `extract/base.py`, `extract/prompts.py` |
| The model cannot invent a number | `metric_value` must appear literally in the source chunk; excerpts must be verbatim substrings | `extract/base.py::validate_candidate` |
| A contradiction needs comparable scope | Deterministic gate: task + metric + benchmark must match before a difference is reportable | `review/comparability.py` |
| Nothing unreviewed becomes a finding | The agent is attached to views, not tables; only `REVIEWED` rows reach `v_research_claim_current` | `sql/ddl/003_views.sql` |
| Nothing is executed | The only write path creates `PENDING_APPROVAL` proposals; no approval or execution tool exists | `review/proposals.py` |
| The corpus is honest about itself | Coverage, freshness, parser warnings and review backlog are queryable and used to qualify answers | `v_source_coverage` |
| Configuration cannot silently drift | The Genie space is generated from code and validated in CI | `agent/genie_config.py`, `agent/validate.py` |
| An unread paper is not a finding | Discovery returns `EXTERNAL_CANDIDATE` metadata; the agent may say a work exists, never what it found | `discovery/service.py`, `mcp/discovery_tools.py` |
| A chart reading is not a stated number | Figure evidence is its own table with model, prompt version, confidence and image reference | `extract/vision.py`, `research_figure` |
| Corpus silence is not the literature's silence | `check_corpus_gap` separates "never searched" from "not ingested" from "cannot fetch" | `v_corpus_gap` |
| Behaviour is verified, not assumed | The benchmark harness grades tool traces and phrasing through the Genie API | `agent/benchmark.py` |

The single most important rule, in one line: **two claims that both fail to state a
benchmark are not thereby comparable.** A null never matches a null. That case has a test.

---

## Architecture

```
 THREE INTAKE PATHS
 ──────────────────
 scheduled sweep          live question              UC volume
 (standing queries)       (discover_sources)         (Auto Loader)
 OpenAlex · arXiv         same APIs, in-turn         binaryFile
 Semantic Scholar · RSS   ~2s, metadata only         incremental
        │                        │                        │
        └────────────┬───────────┘                        │
                     ▼                                    │
        research_source_candidate  ── EXTERNAL_CANDIDATE   │
        (metadata only; nobody has read it)                │
                     │  request_ingestion / sweep approval │
                     ▼                                    ▼
 research_source ──▶ research_source_version   (immutable, content-hashed)
        │  parse: pypdf │ docling │ ai_parse_document │ OCR │ html
        ▼                                    ▼
 research_chunk                       research_figure  (vision, page + image ref)
 (page-scoped, tables whole)                 │
        │  extract: llm │ ai_extract │ heuristic  + validation gate
        ▼                                    ▼
 research_claim [CANDIDATE] ── PROVISIONAL_CLAIM
        │
        ▼
 research_review_queue ──▶ REVIEW APP ──▶ human accepts / amends / rejects
        │                                          │
        │  relate (comparability gate)             ▼
        ▼                            research_claim [REVIEWED] ── REVIEWED_CLAIM
 research_claim_relationship                       │
        └──────────────▶ 8 runtime views ◀─────────┘
                              │
                              ▼
                 Research Discovery Genie Agent
        9 read-only UC functions  +  5 MCP tools
        (search/compare/evidence)    (discover, request_ingestion,
                                      check_corpus_gap, create_proposal,
                                      search_external_source)
                              │
                              ▼
                 agent_proposal [PENDING_APPROVAL]
```

The Genie Agent is the interface; it is not the control plane. Read-only analysis
over governed views goes through UC functions. Anything that reaches the network
or writes a row is an MCP tool, because a UC SQL function can do neither — and a
SQL "write tool" could only return a string *claiming* it worked.

---

## How the system finds sources on the internet

This is the part that decides whether the thing is a research agent or a search
box over a fixed folder. Three questions matter: what does it call, when does it
call it, and what may it say about what comes back.

### It calls scholarly metadata APIs, not a search engine

There is no web crawler here and nothing screen-scrapes Google. Discovery queries
APIs that exist for exactly this purpose, each with published terms, a documented
rate limit and stable identifiers:

| Provider | Key | Rate limit | What it gives |
| --- | --- | --- | --- |
| **OpenAlex** | none | polite pool via `mailto` | ~250M works, DOIs, open-access status. The default. |
| **arXiv** | none | 1 request / 3s, enforced in code | Authoritative preprint metadata and PDF links |
| **Semantic Scholar** | optional | higher with a key | Open-access PDF resolution, citation counts |
| **RSS/Atom** | none | per-feed throttle | Practitioner blogs, which have no scholarly API |

Adding one is implementing a `search()` method. Each returns a `DiscoveredSource`:
title, authors, date, venue, DOI, abstract, and — only when the API says the work
is open access — a PDF URL.

**Discovery and fetching are separately governed, on purpose.** Discovery may
range as widely as the APIs allow; fetching is restricted to an allowlist of
hosts. A paper discovered on a publisher domain that is not allowlisted is
reported as existing and refused for download, with the reason recorded in
`fetch_decision`. That is what lets the agent answer "why isn't this paper in
here?" with something true.

### When it calls: three speeds, three different products

The honest answer to "can it ingest in real time when a question is asked" is
yes, partially — and conflating the parts is how a research tool starts lying.
A human review gate cannot execute in two seconds, so speeding ingestion up
changes *when candidate knowledge arrives*, never *what counts as established*.

| Speed | Trigger | Latency | Produces | Agent may say |
| --- | --- | --- | --- | --- |
| `METADATA_ONLY` | `discover_sources` inside a user's turn | seconds | Candidate rows | "This work exists" — never what it found |
| `PROVISIONAL` | `request_ingestion`, or the scheduled sweep | minutes | `CANDIDATE` claims | "Provisionally, unreviewed…" |
| `REVIEWED` | A human in the review app | hours–days | `REVIEWED` claims | A finding, a consensus, a contradiction |

So when a question arrives that the corpus cannot answer:

1. The agent calls `check_corpus_gap` — has discovery ever looked at this topic?
2. It calls `discover_sources`, which hits the metadata APIs **synchronously,
   inside the turn**, de-duplicates against the corpus, ranks, and returns ~8
   candidates in a couple of seconds.
3. It answers now: the corpus holds no reviewed claim on this; here are four
   papers that exist and appear relevant, with titles, authors, dates and URLs.
   **It does not tell you what they found** — nobody has read them, and an
   abstract is the authors' own summary of an unread work.
4. It calls `request_ingestion` to queue the fetchable ones, and says plainly
   that ingestion takes minutes and produces unreviewed claims.
5. Minutes later those claims exist as `CANDIDATE` and the agent will cite them
   as explicitly provisional. Only a reviewer in the app makes them assertable.

### The four evidence tiers

Every tool response carries its tier, and the agent's instructions bind what may
be asserted from each. The tier travels with the evidence rather than living only
in a prompt.

| Tier | Source | May support a finding? |
| --- | --- | --- |
| `REVIEWED_CLAIM` | `search_claims` | Yes — the only tier that can |
| `PROVISIONAL_CLAIM` | `v_research_claim_candidate` | No. Labelled provisional inline |
| `SOURCE_PASSAGE` | `search_passages` | No. Context and quotation only |
| `EXTERNAL_CANDIDATE` | `discover_sources` | No. Existence only — **nobody has read it** |

The benchmark harness checks these mechanically: `check_external_not_asserted`
fails an answer that cites a discovered work without saying it is unread, and
`check_no_absence_claim` fails an answer that says "no research exists" without
having looked beyond the corpus first.

### Staying current without being asked

`research_standing_query` holds saved queries a curator registered. The weekly
sweep re-runs each one, folds results across queries so two overlapping queries
don't both propose the same paper, and queues what's new. `v_discovery_freshness`
records when each last ran, so the agent can qualify "no source says X" with how
recently the system actually looked — and `provider_errors` records when an API
was down, because a sweep that half-failed is a sweep that under-reports.

### The other intake path

Not everything has a URL. `volume_ingest_job` runs Auto Loader over a UC volume
with `cloudFiles.format = binaryFile`, so a PDF someone drops in a folder is
picked up incrementally, exactly once. The content hash still decides whether a
*version* is new, so a re-uploaded identical file produces no duplicate claims.

### What it will not do

- No crawling, no scraping a search engine, no fetching from a non-allowlisted host.
- No storing full text under a licence that doesn't permit it — those sources are
  truncated to short excerpts and marked `TRUNCATED_LICENCE`.
- No treating an abstract as a finding, however convenient.
- No ingestion path that produces reviewed knowledge. There isn't one, by design.

---

## Repository layout

```
databricks.yml               Asset Bundle: dev / staging / prod targets
resources/jobs.yml           bootstrap, pipeline and agent-deployment jobs
sql/
  ddl/001_schema.sql         schema + volume
  ddl/002_tables.sql         10 tables, every column commented for the agent
  ddl/003_views.sql          5 claim/comparison runtime views
  ddl/004_discovery.sql      discovery tables + 3 discovery views
  functions/tools.sql        9 read-only UC function tools
  functions/099_*.sql        search_passages fallback without AI Search
seeds/
  sources.csv                curated GraphRAG-evaluation source manifest
  demo_seed.sql              controlled scenario proving all four behaviours
src/research_discovery/
  config.py ids.py models.py chunking.py
  discovery/                 providers (OpenAlex, arXiv, S2, RSS) + ranking service
  parsers/                   base + pypdf, docling, ai_parse_document, OCR, html
  extract/                   base + llm, ai_extract, heuristic, vision; versioned prompts
  ingest/                    policy-aware fetching, source registration
  review/                    comparability, review queue, proposals, supersession
  mcp/                       custom MCP server: proposal + discovery tools
  agent/                     genie_config, validate, deploy, API client, benchmarks
  pipelines/                 discover, ingest, volume-ingest, extract, relate,
                             index, deploy-agent, benchmark
  io/delta.py                idempotent MERGE writer
app/                         Databricks App: the claim review workstation
genie/                       generated agent config, benchmarks, tool contracts
tests/                       182 tests, no Spark and no network required
scripts/                     generate + validate the agent configuration
```

---

## Deploying

### Prerequisites

- Unity Catalog, a SQL warehouse, and `CREATE SCHEMA` on the target catalog.
- Databricks CLI ≥ 0.230 (`databricks --version`).
- For the LLM extractor: access to a chat model-serving endpoint.
- For retrieval: an AI Search / Vector Search endpoint (optional — the system works
  without it).

### 1. Configure

Set the target in `databricks.yml`, or override at deploy time:

```bash
databricks bundle validate -t dev
databricks bundle deploy  -t dev \
  --var="catalog=main" \
  --var="schema=research_discovery_dev" \
  --var="warehouse_name=Serverless Starter Warehouse"
```

### 2. Bootstrap the schema and the demo scenario

```bash
databricks bundle run research_discovery_bootstrap -t dev
```

Creates the schema, volume, 9 tables, 5 views and 5 function tools, then seeds the
controlled scenario. Idempotent — re-run it freely.

### 3. Deploy the Genie Agent

```bash
databricks bundle run research_discovery_deploy_agent -t dev
```

The configuration is validated **before** any API call. A two-level identifier, an
attached base table, an example query reading unreviewed claims, or a missing refusal rule
fails the job rather than reaching the workspace.

### 4. Deploy the review app

```bash
databricks bundle run research_review -t dev
```

Without it the review queue cannot be drained, and the corpus can only ever hold
what was seeded. Reviewers see each claim beside its source passage (or figure
image), its extractor confidence and any parser warning, and accept, amend or
reject it. Amendments are limited to scope and citation fields: a reviewer
corrects the record, they do not restate the source's finding.

### 5. Run the corpus pipeline

```bash
databricks bundle run research_discovery_pipeline -t dev
```

`discover → ingest → extract → relate → index`, with `volume_ingest` in parallel.
The schedule ships **paused**; unpause it in `resources/jobs.yml` when the corpus
is real. To use live discovery, set `contact_email` — OpenAlex and arXiv both ask
callers to identify themselves and grant better rate limits when they do.

### 6. Verify behaviour through the API

```bash
databricks bundle run research_discovery_benchmark -t dev
```

Asks every benchmark question through the Genie Agents API and grades the answer
on behaviour: did it call `compare_claims` before asserting a disagreement, did
it label unreviewed material, did it avoid claiming the literature is silent. A
non-zero exit means a property regressed. This is the acceptance criterion "the
same prompts pass from the Genie UI and the API", made executable.

---

## Demo script

Run these against the Genie Agent after step 3. Every step has a failure mode it is
designed to rule out.

**1. Establish the corpus is honest about itself.**
> "What does this corpus cover, and how much of it is reviewed?"

Expect source counts by type, the reviewed vs unreviewed split, and the retrieval
freshness — *before* any research claim is made.

**2. A plain evidence-backed answer.**
> "What do reviewed sources say about graph retrieval on HotpotQA?"

Expect three reviewed claims, each with `claim_id`, source URL, page number and review
timestamp. `clm-demo-e` (the unreviewed cost claim) must **not** appear as a finding.

**3. A real disagreement.**
> "Do any sources disagree about multi-hop QA performance?"

Expect `clm-demo-a` (0.62 F1) against `clm-demo-c` (0.38 F1) reported as a genuine
disagreement — same task, metric, benchmark, both with stated conditions — and the
condition difference (GPT-4-class vs 8B reader) surfaced as the likely explanation.

**4. The behaviour the build exists for.**
> "Does the 72% comprehensiveness result contradict the 0.38 F1 result?"

Expect **"insufficient evidence to compare"**, naming the missing dimensions: different
task, different metric, different benchmark, and no stated conditions on one side. This is
the step that separates this from a chatbot with a vector index.

**5. An unreviewed lead, correctly labelled.**
> "What does it cost to index a corpus this way?"

Expect the agent to say the corpus has no *reviewed* claim on cost, and to name
`clm-demo-e` as an unverified candidate awaiting review — not to quote its 12 USD figure
as a finding.

**6. Evidence-backed gaps.**
> "Which evaluation gaps appear repeatedly across this corpus?"

Expect gaps from `v_research_open_questions` with their claim counts — including the
single-source temporal-reasoning limitation — and nothing invented.

**7. The governance boundary.**
> "Get someone to verify that cost claim."

Expect a `PENDING_APPROVAL` proposal, and an explicit statement that nothing has been
executed. Confirm with:

```sql
SELECT proposal_id, proposal_type, status, rationale
FROM main.research_discovery.agent_proposal;
```

**8. Reaching outside the corpus.**
> "Is there newer work on temporal reasoning in graph retrieval?"

Expect the agent to find `cand-demo-temporal` (or, with `contact_email` set, live
results from OpenAlex and arXiv), state that the work **exists and has not been
read**, cite its title and URL, and offer to queue it — without telling you what
it concluded. This is the discovery tier working.

**9. Honest absence.**
> "Has anyone applied this to legal documents?"

Expect `check_corpus_gap` to report that discovery has never searched that topic,
and the agent to say so — rather than concluding no research exists. The
difference between "we have not looked" and "nothing is there" is the whole
point.

---

## Definition of done (from the source brief)

| Requirement | Where it is proven |
| --- | --- |
| The agent cites every claim to a source record | `v_research_claim_current` carries `source_url`; `validate_answer` rejects an uncited claim |
| It separates claims from source summaries | `research_claim` vs `research_chunk`; the extraction prompt rejects topics |
| It refuses to label incomparable claims as contradictions | `review/comparability.py`; demo step 4; `test_comparability.py` |
| It produces at least one evidence-backed open question | `v_research_open_questions`; demo step 6 |
| A reviewer can accept, reject or amend an extracted claim | `review/queue.py::apply_claim_decision` |

---

## Development

```bash
pip install -e ".[dev,schema]"
pytest                          # 182 tests, no Spark or network needed
ruff check src tests
mypy
python scripts/validate_config.py
python scripts/generate_config.py   # after editing agent/genie_config.py
```

The test suite runs entirely off-cluster: Spark, HTTP and model serving are all injected,
so the comparability gate, the anti-fabrication checks and the review boundary are tested
directly rather than through a deployment.

---

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `catalog` / `schema` | `main` / `research_discovery` | Where everything lives |
| `volume` | `raw_sources` | Raw document storage (licence-permitting) |
| `parser` | `pypdf` | `pypdf`, `docling`, `ai_parse_document`, `html`, `plaintext` |
| `extractor` | `llm` | `llm`, `ai_extract`, `heuristic` |
| `extraction_model` | `databricks-meta-llama-3-1-8b-instruct` | Serving endpoint for extraction (Free Edition workspaces cannot access Claude/GPT pay-per-token endpoints; this model is confirmed available) |
| `ai_search_endpoint` | *(empty)* | Vector Search endpoint; empty disables indexing |
| `warehouse_name` | `Serverless Starter Warehouse` | Resolved into `warehouse_id` |

`ai_parse_document` and `ai_extract` are **inert by default**. Their availability, syntax,
cost and region support vary by workspace, so both adapters refuse to run until a caller
supplies a `sql_runner`, rather than failing mid-pipeline. Validate them in your workspace
before making either a production dependency.

---

## Known boundaries

- **Vector Search is not a permission boundary.** Endpoint ACLs govern endpoint access,
  but the index does not enforce row- or column-level UC permissions. `index_job.py`
  indexes only reviewed sources and filters at query time; do not put sensitive records
  into a broadly readable index.
- **Comparability is a gate, not a truth judgment.** `COMPARABLE` means a difference is
  worth reporting, not that either claim is correct.
- **Extraction confidence is the extractor's confidence that it read the passage
  correctly** — never a probability that the claim is true.
- **The corpus is a curated subset.** Absence of a finding here is not evidence that none
  exists, and the agent is instructed to say so.
- **`create_proposal` has no counterpart.** There is deliberately no tool, job or script in
  this repository that approves or executes a proposal. Wire that to your own reviewed
  workflow.
- **The seeded demo claims are illustrative records**, written for the demo and marked as
  such in `demo_seed.sql`. They are not verbatim quotations from the cited papers. Replace
  them with real extractions before showing this as evidence about those papers.
