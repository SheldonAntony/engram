# engram

**Algorithmic memory retrieval for AI agents. Runs on your machine. No cloud, no GPU, no API keys.**

engram is a memory server that stores facts about your project — decisions, preferences, bugs — in a local SQLite database, then retrieves the most relevant ones using **8 algorithmic signals fused together**. It plugs into Claude Desktop, Cursor, OpenHands, opencode, and any [MCP](https://modelcontextprotocol.io)-compatible client.

Unlike other AI memory systems, engram requires **zero training data**, **zero cloud APIs**, and **zero LLM calls for core retrieval**. Every signal is deterministic — same query, same result, every time.

---

## Why this matters

AI agents forget everything between sessions. Without memory, every conversation starts from scratch — you repeat context, re-explain preferences, and re-state decisions you already made.

engram solves this by storing facts (embeddings + keywords + entities + temporal edges in SQLite) and retrieving the most relevant ones using a 10-stage pipeline. The agent gets exactly what it needs, with no irrelevant tokens polluting its context window.

All data stays on your machine. Nothing is sent to any cloud.

---

## How we're different

| | **engram** | **mem0** | **AgentMemory** | **LangChain Memory** |
|---|---|---|---|---|
| **Runs fully local** | ✅ | ❌ (cloud API) | ✅ | ⚠️ (varies) |
| **No GPU required** | ✅ | N/A (cloud) | ✅ | ✅ |
| **No training data** | ✅ | ❌ | ✅ | ✅ |
| **No paid API keys** | ✅ | ❌ (paid API) | ✅ | ⚠️ (varies) |
| **Deterministic retrieval** | ✅ (same result every time) | ❌ (LLM stochastic) | ❌ (LLM compression) | ❌ (varies) |
| **Multi-signal retrieval** | ✅ (8 signals) | ✅ (2026) | ✅ (3 signals) | ❌ (cosine only) |
| **Cross-encoder reranker** | ✅ (local, 80M params) | ❌ | ❌ | ❌ |
| **SM-2 spaced repetition** | ✅ (per-fact forgetting) | ❌ (generic decay) | ❌ (Ebbinghaus) | ❌ |
| **Explainable retrieval** | ✅ (per-signal scores) | ❌ (black box) | ❌ (opaque RRF) | ❌ |
| **MCP native** | ✅ | ❌ | ✅ | ❌ |

Most memory systems use **cosine similarity only** — they embed everything, compute distance, and call it done. That misses ~35% of relevant facts for real-world queries.

engram uses **8 signals** fused together:
1. **Cosine similarity** — semantic meaning matching
2. **BM25 keyword search** — exact word matching with rarity weighting
3. **Derived BM25** — WordNet hypernym expansion for vocabulary bridging
4. **Entity overlap** — named entity matching (people, places, organizations)
5. **Date/time matching** — temporal facts boosted for time-sensitive queries
6. **Lexical channels** — bigram/trigram phrase matching
7. **Conversation context BM25** — searches neighboring turns for missing context
8. **Cross-encoder reranker** — local 80M-parameter model re-reads candidates for precision

**Result:** ~77% recall@3 on LoCoMo — **without training on any dataset**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  QUERY                                                      │
│  "What database did we choose?"                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: PARALLEL RETRIEVAL (5 channels)                   │
│  • Cosine ANN — top 750 semantic matches                    │
│  • BM25 FTS5 — top 300 keyword matches                      │
│  • Derived BM25 — WordNet-expanded keywords                   │
│  • Lexical channels — phrase/bigram matches                   │
│  • Context BM25 — neighboring turn search (±3 turns)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: RRF FUSION                                        │
│  Reciprocal Rank Fusion: 0.35*cosine + 0.20*BM25 +            │
│  0.20*recency + 0.15*session_recency + 0.10*frequency         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: GUARDS & FILTERS                                  │
│  • Coverage guard — ensures diverse sources                   │
│  • CE guard — minimum relevance threshold                   │
│  • SM-2 gate — spaced repetition filter (production)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: CROSS-ENCODER RERANK                              │
│  mxbai-rerank-xsmall-v1 (80M params)                        │
│  120 candidate pairs × ~50ms = ~6s on CPU                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: DIVERSITY & BUDGET                                │
│  • MMR deduplication (λ=0.25)                                 │
│  • Greedy token budget (max_tokens=2000)                      │
│  • Graph expansion — BFS neighbors (depth=1, max 3)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Top-N facts with per-signal explainability         │
│  Fact #42: "We use Postgres 16"                             │
│    • cosine: 0.85 | bm25: 0.72 | entity: 0.30 | ce: 0.94   │
│    • Why: semantically similar; keyword match; CE verified  │
└─────────────────────────────────────────────────────────────┘
```

---

## Benchmarks

### LoCoMo (1,531 questions, 10 long conversations)

| Config | Recall@3 | Notes |
|--------|----------|-------|
| Cosine only | 65.90% | Baseline |
| + BM25 + lexical + entity | 71.35% | Multi-signal fusion |
| + cross-encoder (mxbai, pool=120) | **76.87%** | **Best without training** |
| + context BM25 (v19) | **77.86%** | Production best |
| Competitor (GBM trained on LoCoMo) | 80.99% | Requires offline training |

engram closes **~75% of the gap** to the trained competitor, with **zero training data**.

### LongMemEval-S (500 questions, ~48 sessions each)

| Split | R@1 | R@3 | MRR@5 | Time |
|-------|-----|-----|-------|------|
| Oracle (no filler) | 1.000 | 1.000 | 1.000 | 136s (0.3s/q) |
| **S split (~40 filler sessions)** | **0.764** | **0.777** | **0.784** | 39835s (~11h, 84.8s/q) |

**Category breakdown (S split):**

| Question type | N | R@1 | R@3 | MRR |
|---------------|---|-----|-----|-----|
| knowledge-update | 72 | 0.833 | 0.833 | 0.854 |
| multi-session | 121 | 0.818 | 0.826 | 0.839 |
| single-session-assistant | 56 | 1.000 | 1.000 | 1.000 |
| single-session-preference | 30 | 0.567 | 0.667 | 0.625 |
| single-session-user | 64 | 0.562 | 0.562 | 0.576 |
| temporal-reasoning | 127 | 0.717 | 0.732 | 0.739 |

**Known weakness:** single-session-user and single-session-preference queries (56-67% R@3). These require exact keyword match and our system prioritizes semantic relevance. [Fixes in progress](https://github.com/SheldonAntony/engram/issues).

---

## How it works with your LLM

engram is an **MCP server** — it speaks the Model Context Protocol that AI agents natively understand.

When you ask your agent a question, the agent calls engram's `search` tool automatically, engram returns the relevant facts, and the agent uses them as context. The agent decides what to retrieve and when — you don't need to learn a separate interface.

**Supported clients:**
- [Claude Desktop](https://claude.ai) — add `server.py` to your MCP config
- [Cursor](https://cursor.sh) — MCP server integration in settings
- [Windsurf](https://codeium.com/windsurf) — MCP server support
- [OpenHands](https://openhands.dev) — `openhands mcp add engram --transport stdio python -- /path/to/server.py`
- [opencode](https://opencode.ai) — native plugin, no config needed
- Any MCP-compatible agent

engram's memory persists across sessions, across clients — use it with Claude today, Cursor tomorrow, and the facts follow you.

---

## Quick start

### Requirements
- Python 3.10+
- An MCP-compatible AI client (Claude Desktop, Cursor, OpenHands, opencode, etc.)

### Install

```bash
git clone https://github.com/SheldonAntony/engram.git ~/.config/opencode
cd ~/.config/opencode
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
git clone https://github.com/SheldonAntony/engram.git "$env:USERPROFILE\.config\opencode"
cd "$env:USERPROFILE\.config\opencode"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Connect to Claude Desktop

Edit `claude_desktop_config.json` (Claude Desktop → Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "engram": {
      "command": "python",
      "args": ["/full/path/to/engram/server.py"]
    }
  }
}
```

Restart Claude Desktop. The agent now has `search`, `remember`, `forget`, `history`, `graph`, `consolidate`, `remember_slot`, and `get_slots` tools available.

### Connect to OpenHands

```bash
openhands mcp add engram --transport stdio \
  python -- /full/path/to/engram/server.py
```

Verify: `openhands mcp list` → should show `engram [enabled]`

### Connect to opencode

```bash
pip install opencode
# engram is a native opencode plugin — no config needed
```

### Verify

Store a fact and retrieve it:

```
You: Remember that we chose Postgres 16 for production
Agent: Stored as fact #1

You: What database are we using?
Agent: We use Postgres 16 for production (from memory)
```

---

## Design principles

1. **Your data stays yours.** Everything runs locally. No telemetry, no cloud sync, no data exfiltration.

2. **No training debt.** Signals are engineered — full quality from the first query. No "warm-up period" where accuracy improves over time.

3. **No GPU, no problem.** Cross-encoder runs on CPU (~6s for 120 candidate pairs). Embeddings via fastembed (ONNX Runtime).

4. **Deterministic.** Same query → same facts → same result. Every time. No stochastic LLM variation.

5. **Explainable.** Every retrieved fact shows per-signal scores. You know *why* the agent remembered something.

6. **Less token waste.** Multi-signal pipeline finds right facts faster, minimizing irrelevant tokens in context window.

7. **Works today, not tomorrow.** No model training in progress. Install and the agent gains memory immediately.

---

## Glossary

| Term | What it means |
|------|---------------|
| **Embedding** | Mathematical fingerprint capturing text meaning. Similar meaning = similar fingerprint. |
| **BM25** | Smart keyword search — finds exact words, weighted by rarity. |
| **Cosine similarity** | Measures embedding closeness (1.0 = identical, 0.0 = unrelated). |
| **RRF** | Voting system: if multiple searches agree a fact is relevant, it ranks higher. |
| **Cross-encoder** | Small AI model that re-reads each candidate with the query and scores relevance (0-1). |
| **MCP** | Standard protocol for AI agents to discover and call tools. Like USB for AI. |
| **SM-2** | Spaced repetition algorithm — high-value facts are reviewed more often, low-value facts decay. |
| **R@3 (Recall@3)** | Out of 100 questions, how many times does the correct fact appear in top 3? |

---

## Repository

- **engram** (this repo): https://github.com/SheldonAntony/engram
- **engram-eval** (benchmark pipeline): https://github.com/SheldonAntony/engram-eval

---

## Acknowledgments

- LoCoMo benchmark: Maharana et al., ACL 2024
- LongMemEval benchmark: Wu et al., ICLR 2025
- SM-2 algorithm: Piotr Wozniak, SuperMemo
- Cross-encoder: mixedbread-ai/mxbai-rerank-xsmall-v1
- Embeddings: BAAI/bge-small-en-v1.5 via fastembed
