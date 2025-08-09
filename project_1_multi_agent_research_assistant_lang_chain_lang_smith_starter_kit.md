# Project 1 — Multi‑Agent Research Assistant (LangChain + LangSmith)

**Full Project Documentation — Ideas • Instructions • Prompts • Evaluation**\
Build in **Claude Code**. Keep runtime **LLM provider‑agnostic** (Claude/OpenAI/others via env + pluggable abstraction).

---

## 0) Why this project?

A didactic, production‑leaning project that teaches **LangChain logic** (LCEL, tools, agents, routing, memory), **observable development** with **LangSmith**, and sound **RAG** practices—while producing a research assistant you can actually use (CLI + minimal API; citations; self‑critique; reproducible evals).

---

## 1) Learning Outcomes

You will be able to:

- Design **multi‑agent workflows** with explicit state and turn‑taking.
- Implement **strict tool contracts** (search, retriever, optional scraper) with Pydantic schemas.
- Build a **small, reliable RAG** stack (ingest, retriever, chunking strategy, embeddings).
- Use **LCEL** to compose, route, and parallelize runnables with type‑checked state.
- Instrument chains with **LangSmith** (traces, datasets, evals, A/B of prompts/models).
- Add **guardrails**: citations, refusal logic, domain allowlists, rate limiting.

---

## 2) Feature Set (MVP → Plus)

**MVP**

- CLI and /ask HTTP endpoint.
- Orchestrated flow: Orchestrator → Researcher → Critic → Synthesizer.
- Retrieval over local KB (Chroma) + optional web search.
- Cited, well‑structured answers with confidence + appendix.
- Full LangSmith tracing.

**Plus**

- LangSmith dataset + eval rubric, automated scoring for faithfulness/answerability.
- Experiments: chunk sizes/overlap, rerankers, different embeddings.
- Optional **Firecrawl** tool for robust extraction.
- Optional **MCP** tool exposure for editor/agent interop.

---

## 3) High‑Level Architecture

```
User → Orchestrator (planner) → Researcher (tools: WebSearch, Retriever, Firecrawl)
                                 ↓
                               Critic (gap/quality checks)
                                 ↓
                            Synthesizer (final, cited output)
```

**State** (typed) moves across steps:\
`{ question, plan, key_terms[], findings[], citations[], critique, draft, final, confidence }`

---

## 4) Requirements

- **Build environment**: Claude Code (primary IDE)
- **Runtime**: Python 3.11+
- **Core libs**: langchain, langsmith, chromadb, pydantic, fastapi, uvicorn, typer, requests
- **Embeddings**: provider of your choice (configurable)
- **LLM**: provider‑agnostic via a small adapter (see §8)

---

## 5) Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U langchain langsmith fastapi uvicorn chromadb typer pydantic requests
# Add your chosen providers (e.g., anthropic, openai, etc.)

# Minimal env
cp .env.example .env
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="multiagent-research"
# plus provider keys: e.g., ANTHROPIC_API_KEY / OPENAI_API_KEY / LANGCHAIN_API_KEY

# Ingest sample docs
python -m app.rag.ingest data/sample_docs/

# Ask via CLI
python -m app.cli ask "What are the practical impacts of policy X in 2024–2025?"

# Run API
uvicorn app.api:app --reload
```

---

## 6) Repository Layout

```
multiagent-research/
├─ app/
│  ├─ chains/
│  │  ├─ orchestrator.py
│  │  ├─ researcher.py
│  │  ├─ critic.py
│  │  └─ synthesizer.py
│  ├─ core/
│  │  ├─ llm.py              # provider-agnostic LLM adapter
│  │  ├─ config.py           # env + settings
│  │  └─ state.py            # TypedDict/Pydantic state
│  ├─ rag/
│  │  ├─ ingest.py           # load -> chunk -> embed -> persist
│  │  └─ store.py            # retriever factory
│  ├─ tools/
│  │  ├─ web_search.py
│  │  ├─ retriever.py
│  │  └─ firecrawl.py        # optional
│  ├─ eval/
│  │  ├─ datasets.jsonl
│  │  ├─ rubric.md
│  │  └─ run_eval.py
│  ├─ api.py                 # FastAPI /ask
│  ├─ cli.py                 # Typer CLI
│  └─ utils.py
├─ prompts/
│  ├─ orchestrator.claude
│  ├─ researcher.claude
│  ├─ critic.claude
│  └─ synthesizer.claude
├─ tests/
│  ├─ test_tools.py
│  └─ test_chain_contracts.py
├─ docs/
│  ├─ README.md
│  ├─ SETUP.md
│  ├─ ARCHITECTURE.md
│  ├─ EVALUATION.md
│  └─ TROUBLESHOOTING.md
├─ Makefile
├─ .env.example
└─ pyproject.toml
```

---

## 7) Environment & Configuration

**.env.example**

```
# Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multiagent-research
LANGCHAIN_API_KEY=

# LLM provider (choose one)
PROVIDER=anthropic          # or openai, other
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
MODEL_NAME=claude-3-7-sonnet-2025-XX # or gpt-4.1 or your choice

# Embeddings
EMBEDDINGS_PROVIDER=openai  # or other
EMBEDDINGS_MODEL=text-embedding-3-large

# Web search
SEARCH_API=serpapi          # or bing, searx
SEARCH_API_KEY=

# Firecrawl (optional)
FIRECRAWL_BASE_URL=
FIRECRAWL_API_KEY=
```

**config.py** centralizes env loading and exposes typed settings.\
**llm.py** maps `PROVIDER + MODEL_NAME` → a LangChain `ChatModel` instance without leaking provider specifics into chains.

---

## 8) Provider‑Agnostic LLM Adapter (Design)

**Goal**: Swap providers with **zero** changes to chain logic.

**Interface** (concept):

```python
from typing import Literal
Provider = Literal["anthropic", "openai", "other"]

def chat_model(provider: Provider, model: str, **kwargs) -> BaseChatModel:
    ... # returns a LangChain-compatible ChatModel
```

- Adapter handles system prompts, tool/function calling, and streaming flags.
- Chains import only `chat_model()` from `app.core.llm`.

---

## 9) Typed State & LCEL Composition

```python
# app/core/state.py
from typing import List, TypedDict, Optional

class Citation(TypedDict):
    marker: str
    title: str
    url: str
    date: str | None

class Finding(TypedDict):
    claim: str
    evidence: str
    source: Citation

class PipelineState(TypedDict, total=False):
    question: str
    plan: str
    key_terms: List[str]
    findings: List[Finding]
    citations: List[Citation]
    critique: dict
    draft: str
    final: str
    confidence: float
```

Compose steps with LCEL `Runnable` s, passing `PipelineState` along.

---

## 10) Tool Contracts (Pydantic + JSON Schema)

**WebSearchTool**

- Input: `{ query: str, top_k?: int }`
- Output: `{ results: [{ title, url, snippet, published_at? }] }`
- Guards: allowlist/blocklist, rate limit, dedup by domain+title hash

**RetrieverTool**

- Input: `{ query: str, top_k?: int }`
- Output: `{ contexts: [{ id, source, content, score }] }`

**FirecrawlTool (optional)**

- Input: `{ url: str, mode?: "article|full" }`
- Output: `{ text, html?, links[] }`

> Implement each as a `BaseTool` with Pydantic models so LangSmith traces are clean and comparable.

---

## 11) RAG Ingestion & Retrieval

- **Ingest**: PDF/HTML/MD → text → chunk (`512–1200` tokens, overlap `64–160`) → embed → persist (Chroma).
- **Experiments**: vary chunk size/overlap; optionally test a reranker; store runs in LangSmith.
- **Retrieval**: `SimilaritySearch` with `top_k` 4–8; return normalized contexts with `source` metadata.

CLI example:

```bash
python -m app.rag.ingest data/sample_docs/ --chunk 800 --overlap 120
```

---

## 12) Prompts — “.claude” Files (shipped)

### orchestrator.claude

```
You are the **Orchestrator**, a planning agent. Given a user question, output a minimal, actionable research plan.

OBJECTIVES
- Pick the right tools: WebSearch (fresh info), Retriever (KB), Firecrawl (extraction).
- Break work into 2–5 steps max.
- Note what evidence would *falsify* early assumptions.
- Prefer high-quality, citable sources.

OUTPUT SCHEMA (JSON)
{
  "plan": string,
  "tool_sequence": ["web_search" | "retriever" | "firecrawl"],
  "key_terms": string[]
}

HARD RULES
- Don’t fabricate sources or dates.
- If the question is unanswerable or time‑sensitive, say so and suggest clarifications.
```

### researcher.claude

```
You are the **Researcher**. Execute the plan using the available tools. Always cite.

PROCESS
1) Run tools as needed; extract relevant quotes with permalinks.
2) Build a short DRAFT with inline [#] markers.

OUTPUT SCHEMA (JSON)
{
  "findings": [
    {"claim": string, "evidence": string, "source": {"title": string, "url": string, "date?": string}}
  ],
  "draft": string,
  "citations": [
    {"marker": "#1", "url": string, "title": string, "date?": string}
  ]
}

RULES
- Prefer primary sources with dates.
- Mark low‑confidence items and propose next queries when needed.
```

### critic.claude

```
You are the **Critic**. Punch holes in the draft.

CHECKS
- Are claims supported by cited evidence?
- Are we missing *recent* developments or contrasting viewpoints?
- Any ambiguous scope or weasel words?

OUTPUT SCHEMA (JSON)
{ "issues": string[], "required_fixes": string[] }

RULES
- Be concise and specific; point to markers like [#2] when possible.
```

### synthesizer.claude

```
You are the **Synthesizer**. Produce a crisp, well‑structured, cited answer.

RESPONSE STRUCTURE
- Summary (3–5 sentences)
- Key Points (bulleted)
- Caveats / Unknowns
- Sources (with [#] markers)

OUTPUT SCHEMA (JSON)
{
  "final": string,      
  "citations": [{"marker": "#1", "url": string, "title": string, "date?": string}],
  "confidence": number
}

RULES
- No new claims beyond `findings`.
- If answerable only conditionally, state conditions.
```

---

## 13) Implementation Phases (Didactic Path)

**Phase 0 — Baseline single‑agent RAG**

- Build `RetrieverTool` and a simple `question → retrieve → answer` chain.
- Add citations; ensure LangSmith traces show contexts.

**Phase 1 — Orchestrator planning**

- `orchestrator.py` generates plan + tool sequence + key terms.
- Pass `{question, plan}` to researcher.

**Phase 2 — Multi‑agent loop**

- Researcher produces `findings, draft, citations` using tools.
- Critic returns `issues, required_fixes`; loop once if fixes exist.
- Synthesizer outputs final markdown + confidence.

**Phase 3 — Evaluation with LangSmith**

- Create `eval/datasets.jsonl` (≈15 cases): easy, tricky, ambiguous, unanswerable.
- `eval/run_eval.py` runs pipeline, computes metrics: faithfulness, answerable, citation\_coverage, latency.
- Compare prompt variants and models in LangSmith UI.

**Phase 4 — Firecrawl (optional)**

- Add `FirecrawlTool` and selection logic (discovery vs extraction).
- Implement allowlist + rate limits; log extracts as artifacts.

**Phase 5 — MCP (optional)**

- Wrap tools with JSON schemas; expose via MCP server; keep LangChain orchestrator.

---

## 14) Evaluation Details (LangSmith)

**Dataset format (**``**)**

```jsonl
{"input": {"question": "Summarize the latest WHO guidance on topic X and cite sources."}, "expected_behavior": "Recent, primary sources; 3–5 bullets; citations"}
{"input": {"question": "What changed in policy Y this year?"}, "expected_behavior": "Identify year, cite official notices; refuse if unclear"}
```

**Scoring (scripted)**

- *Faithfulness*: proportion of answer sentences that have overlapping spans in retrieved contexts.
- *Answerable*: 1 if the system answers/refuses correctly per rubric.
- *Citation coverage*: major claims have ≥1 citation; links resolvable.
- *Latency*: end‑to‑end time.

**LangSmith instrumentation**

- Name runs per phase (`run_name`), set tags (`phase:p2`, `model:claude-…`).
- Attach artifacts: retrieved snippets, cleaned extracts.

---

## 15) API & CLI

**CLI**

```bash
python -m app.cli ask "<your question>"
```

**HTTP** (FastAPI)

```
POST /ask
{ "question": "string" }
→ { "final": "md", "citations": [...], "confidence": 0.82, "trace_url": "…" }
```

---

## 16) Testing Strategy

- `tests/test_tools.py`: schema validation, retry/backoff, rate‑limit behavior.
- `tests/test_chain_contracts.py`: state carries required keys; no missing citations.
- Golden tests for prompts (snapshot against canned tool outputs).

---

## 17) Security, Guardrails, and Policy

- Domain allowlist & blocklist for web tools.
- Refusal policy for speculative/medical/financial advice without sources.
- Source dedup + soft‑voting to reduce single‑source bias.
- PII redaction hooks before logging.

---

## 18) Developer Workflow (Claude Code)

- Use Claude Code tasks to run `make watch` for API hot‑reload and `make eval` for quick evals.
- Keep prompts in `/prompts` and iterate with side‑by‑side LangSmith traces.
- Use model‑agnostic adapter; switch providers by editing `.env` only.

**Makefile (excerpt)**

```make
run:; uvicorn app.api:app --reload
cli:; python -m app.cli ask "$$Q"
eval:; python -m app.eval.run_eval --dataset basic
ingest:; python -m app.rag.ingest data/sample_docs/ --chunk 800 --overlap 120
```

---

## 19) Troubleshooting

- **No traces in LangSmith**: check `LANGCHAIN_TRACING_V2=true`, API key, project name.
- **Empty retrieval**: verify embeddings key/model; run `ingest` again; inspect vector count.
- **Long latency**: lower `top_k`, enable streaming, cache embeddings.
- **Weird citations**: ensure tool returns normalized URLs and titles; dedup per domain.

---

## 20) Roadmap

- Add reranker (e.g., cross‑encoder) for better retrieval precision.
- Per‑source credibility scoring and confidence calibration.
- UI: source cards with quote highlights.
- MCP adapter + Firecrawl enrichment at scale.

---

## 21) Definition of Done

- Clean, cited markdown answers via CLI & API.
- LangSmith shows multi‑step traces with tool calls.
- `eval/run_eval.py` produces metrics and CSV/JSON report.
- Provider can be switched via `.env` without code changes.

---

## 22) License & Contributions

- MIT by default (adjust as you prefer). PRs welcome: prompts, tools, eval datasets.

---

### Next Step

Say the word and I’ll generate the **Claude‑Code‑ready repo scaffold** with placeholder files: config, adapter, chains, tools, prompts, docs, tests, and Makefile—so you can `git init` and run in minutes.

