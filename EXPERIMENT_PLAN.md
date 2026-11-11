# RAG + Ticket Triage Experiment Plan

**Purpose:** Document the end-to-end experiment design for a support ticket + KB resolution agent across hundreds of apps on onprem and cloud.

## Context

The original controlled experiment (`rag_experiment.py`, levels a–h) tests RAG retrieval features on a **single document** (`document.txt` — NovaCluster ops manual). That experiment isolates which RAG features (hybrid, parent-child, reranking, query expansion, sparse-only) improve answer quality while holding everything else constant.

This plan extends that foundation to a **production-scale support ticket + KB resolution agent** spanning:
- hundreds of applications
- onprem and cloud environments
- Ticket triage + classification + KB-based resolution guidance

## Architecture

```
Fresh ticket
    │
    ▼
┌─────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────┐
│  CLASSIFIER PRE-STAGE   │─────►│  RAG + LLM RESOLUTION    │─────►│  Resolution      │
│  (Strands Decider       │      │  (levels a–h + GraphRAG) │      │  guidance          │
│   three-way switch)     │      │                          │      │                  │
│                         │      │                          │      │                  │
│  ROLE:                  │      │  ROLE:                   │      │                  │
│  - Route/route ticket   │      │  - Retrieve KB guidance  │      │                  │
│  - Assign app/platform  │      │  - Generate answer text  │      │                  │
│  - Set severity/priority│      │  - Cite retrieved chunks │      │                  │
│  - Decide canned vs RAG │      │                          │      │                  │
│  - Confidence score     │      │                          │      │                  │
│                         │      │                          │      │                  │
│  Outputs:               │      │  Outputs:                │      │                  │
│  - app_id / platform /  │      │  - answer text           │      │                  │
│    environment          │      │  - retrieved KB chunks   │      │                  │
│  - triage category      │      │  - context chars         │      │                  │
│  - severity             │      │  - latency               │      │                  │
│  - canned vs RAG vs ESC │      │                          │      │                  │
└─────────────────────────┘      └──────────────────────────┘      └──────────────────┘
```

### Design principle: classifier for triage, RAG for resolution

**Classifier (pre-stage)** — decides *what* to do with the ticket:
- Which app/platform/environment it belongs to
- Triage category (cluster_restart, node_provisioning, cert_expiry, security_group, crashloop, etc.)
- Severity (low / medium / high / critical)
- `needs_rag` — canned response sufficient, or KB guidance required?
- Confidence score

**RAG + LLM (resolution stage)** — decides *how* to resolve:
- Only invoked when `needs_rag=true`
- Retrieves relevant KB chunks via levels a–h
- Generates resolution guidance answer
- Can escalate if confidence is low

This split follows the AWS/Strands Agents pattern ("Amazon saves millions by replacing Sonnet 4.6 with Strands Decider"):

- **BEDROCK**: LLM + RAG handles every ticket (baseline, expensive)
- **DUAL**: Both classifier and RAG/LLM run in parallel; compare results (A/B)
- **DECIDER**: Classifier handles it alone with canned response or route (cheap, fast)

**Classifier outputs:**
- `app_id` — which of hundreds of apps
- `platform` — onprem / cloud
- `environment` — prod / dev / staging
- `triage_category` — cluster_restart, node_provisioning, cert_expiry, security_group, crashloop, etc.
- `severity` — low / medium / high / critical
- `needs_rag` — boolean: does this need KB guidance or canned response?
- `confidence` — probability score

**Experiment questions:**
1. Does a classifier pre-stage reduce cost without hurting resolution quality?
2. Large LLM vs small LLM vs encoder-only — which classifier works best for ticket triage?
3. Does fine-tuning a small encoder beat zero-shot LLM classification for infra-specific categories?
4. Which RAG level (a–h) works best per triage category?

### RAG resolution stage (existing experiment)

Levels a–h from `rag_experiment.py`:

| Level | Configuration |
|-------|---------------|
| a | Naive RAG: vector search only |
| b | Hybrid: vector + BM25 (RRF fused) |
| c | Parent-child: vector search children, return parents |
| d | Hybrid + cross-encoder reranking |
| e | Hybrid + query expansion (multi-query) |
| f | Full-context baseline: no retrieval |
| g | All advanced: hybrid + parent-child + rerank + expansion |
| h | Sparse-only: BM25 keyword search only |

**New requirement: metadata filtering**

With hundreds of apps and multiple platforms, retrieval must be filtered before vector/BM25 search:
- `app_id` filter — only search KB for this app
- `platform` filter — onprem vs cloud procedures
- `environment` filter — prod procedures may differ from dev

This was flagged as "N/A for single document" in `rag-retrieval-styles.md` but is **essential** here.

### GraphRAG (separate script: `rag_experiment_graphrag.py`)

For cross-app relationships:

- App A depends on App B (same ticket pattern)
- onprem job feeds cloud batch process
- Same error code across different apps = different root causes
- Escalation paths: app → team → on-call → runbook

GraphRAG replaces chunk-based retrieval with entity/relationship graph:
- **Index-time:** LLM extracts entities + relationships from KB documents → builds graph → detects communities → summarizes communities
- **Query-time:** LLM extracts entities from question → graph traversal → community ranking → community summaries as context

**Constants analysis (GraphRAG vs chunk-based RAG):**

| Category | Constants | Status |
|----------|-----------|--------|
| Apply (unchanged) | DOCUMENT_PATH, EMBEDDING_MODEL_NAME, LLM_BACKEND, LLM_MODEL_NAME, THINKING_LEVEL, TOP_K, MAX_CONTEXT_CHARS, OPENROUTER_*, EVAL_QUESTIONS, RESULTS_DIR, CONFIDENCE_THRESHOLD | Kept identical |
| Don't apply (removed) | CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, HYBRID_TOP_K, CHUNK_CONTEXT_OVERHEAD, CHILD_CONTEXT_CHARS, PARENT_CONTEXT_CHARS, WEIGHT_CONTEXT_BY_SCORE, USE_WEIGHTED_RRF | Removed — GraphRAG uses entity extraction, not fixed chunks |
| Change (GraphRAG-specific) | ENTITY_EXTRACTION_MODEL, ENTITY_EXTRACTION_THINKING, COMMUNITY_DETECTION_ALGORITHM, MAX_ENTITIES_PER_COMMUNITY, GRAPH_EDGE_WEIGHT_THRESHOLD, MAX_COMMUNITIES_RETURNED | New constants |

## Data

### Public datasets found via Hugging Face / Google search

| Dataset | Size | Schema (key fields) | Problem for this use case |
|---------|------|---------------------|---------------------------|
| `Console-AI/IT-helpdesk-synthetic-tickets` | 500 rows | id, subject, description, priority, category, createdAt, requesterEmail | End-user helpdesk (printers, email, Google Calendar, shared drives). Category: Network/Software/Hardware/Account/Security/etc. No resolution notes, no KB linkage, no infrastructure/platform metadata. |
| `aitrainingdatabroker/synthetic-support-tickets-with-outcomes` | ~500 rows | ticket_id, opened, closed, hours_to_resolve, category, channel, priority, product_area, messages[], resolution{outcome, action, first_contact_resolved, reopened, customer_rating} | Has resolution actions/outcomes — closest fit. But categories are customer-support/SaaS (export_error, login_failure, cancellation, data_correction). No KB linkage, no app/platform/env metadata, no infrastructure ticket types. |
| `mindweave/help-desk-tickets` | 1K–10K rows | ticket_id, created_at, resolved_at, priority, status, channel, category_id, assigned_agent_id, requester_department, affected_service, summary, description, escalated, outage_related | Full ITSM schema (multi-table: tickets, agents, categories, comments, sla_breaches). But end-user helpdesk for 500-person org (printers, webcams, Outlook, VPN for hotel Wi-Fi). No resolution notes text, no KB linkage. |
| `MeneerJohn/synthetic-it-support-tickets` | Unknown | Parquet (schema not inspected). Tags: synthetic, itsm, incident-management, support-tickets, rag, pii, redaction, privacy, benchmark | Mentions RAG and incident management. Could not inspect parquet schema. Likely still end-user focused. |
| `deepak2009thakur/It_ticket_resolution_data` | 67 rows | formatted_text only | Too small, unknown schema. |
| `tasksource/it-support-tickets` | 1K–10K rows | Parquet (schema not inspected) | Generic IT support, multi-language (en, de, pt, es). |
| `w1z4rd3k/it-support-l1-ticket-classification` | Unknown | Unknown | Unknown schema. |

**None contain:** onprem infrastructure events, cloud infrastructure events, cross-environment incidents, hundreds of app-specific procedures, KB-linked ticket→resolution pairs, or resolution notes with technical remediation steps.

**Conclusion:** Public ITSM data is consumer IT helpdesk (printers, email, laptops, VPN for hotel Wi-Fi). Your scenario is infrastructure/platform support for hundreds of apps across hybrid environments. These are fundamentally different domains.

### Synthetic data generation (recommended)

Generate labeled triage tickets from `document.txt` (NovaCluster ops manual) using an LLM:

- Read document sections (restart procedure, error codes ERR-001–004, node provisioning, security groups, CrashLoopBackOff, certificate renewal)
- LLM generates realistic tickets per section with symptoms, categories, priorities
- Each ticket: `ticket_text`, `triage_category`, `platform`, `environment`, `severity`, `needs_rag`, `kb_section`, `reference_resolution`
- Scale to hundreds of apps by generating per-app variants with different error codes and procedures

**Proposed triage schema:**
```json
{
  "ticket_id": "T-001",
  "ticket_text": "Control plane health check failing after cert rotation, pods in CrashLoopBackOff",
  "app_id": "novacloud-control-plane",
  "platform": "onprem",
  "environment": "prod",
  "triage_category": "cluster_restart",
  "severity": "critical",
  "needs_rag": true,
  "kb_section": "section_2_restart_procedure",
  "reference_resolution": "Drain nodes, stop control plane, verify pods terminated, start control plane, uncordon nodes, health-check",
  "canned_response": null
}
```

### KB documents

In addition to `document.txt`, plan for multiple KB documents:
- One per app or per domain (hundreds of apps)
- Each with metadata: `app_id`, `platform`, `environment`, `doc_type` (runbook, SOP, troubleshooting guide)
- Chunks carry metadata for filtered retrieval

## Experiment matrix

| Experiment | Variable | What it tests |
|------------|----------|---------------|
| E1 | Classifier: none vs LLM (large) vs LLM (small) vs encoder | Does classifier pre-stage help? Which model? |
| E2 | RAG level: a–h | Which RAG feature works best per triage category? |
| E3 | Metadata filtering: on vs off | Does app/platform filtering improve retrieval? |
| E4 | GraphRAG vs chunk-based RAG | Does graph-based context beat raw chunks for cross-app incidents? |
| E5 | Three-way switch: BEDROCK vs DUAL vs DECIDER | Cost vs quality trade-off for triage |

## Success metrics

| Metric | Definition |
|--------|------------|
| Classification accuracy | % tickets classified to correct app/platform/category |
| RAG quality | Answer score vs reference guidance (same as existing `analyze_results.py`) |
| Cost per ticket | LLM tokens + classifier inference cost |
| Latency per ticket | End-to-end ms from ticket arrival to guidance |
| Deflection rate | % tickets resolved by classifier alone (canned response, no RAG) |
| Escalation accuracy | % tickets routed to correct team/on-call |

## Files

| File | Purpose |
|------|---------|
| `rag_experiment.py` | Controlled RAG experiment (levels a–h) — unchanged |
| `rag_experiment_graphrag.py` | Separate GraphRAG experiment — created |
| `analyze_results.py` | Analysis script — updated for string levels |
| `run_experiment.sh` | Runner script — updated for letter levels |
| `EXPERIMENT_PLAN.md` | This document — overall design |
| `document.txt` | NovaCluster KB (single doc for controlled experiment) |
| `questions.json` | RAG eval questions (4 Q&A pairs) |
| *(planned)* `classify_experiment.py` | Classifier pre-stage experiment (not yet written) |
| *(planned)* `generate_triage_tickets.py` | Synthetic ticket generator from document.txt (not yet written) |
| *(planned)* `triage_tickets.jsonl` | Generated synthetic triage dataset (not yet created) |

## Next steps

1. **Write `generate_triage_tickets.py`** — synthetic triage data from `document.txt`
2. **Write `classify_experiment.py`** — test classifier pre-stage (large LLM vs small LLM vs encoder)
3. **Add metadata filtering** to `rag_experiment.py` — app/platform/environment filtered retrieval
4. **Run E1–E5** experiments and analyze with `analyze_results.py`
