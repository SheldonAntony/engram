# engram

Persistent semantic memory for AI coding agents.

engram is an [opencode](https://opencode.ai) plugin with an [MCP server](https://modelcontextprotocol.io) for AI agents to store and retrieve long-term memory — facts, decisions, preferences, and conversation context.

**~78% R@3 on LoCoMo** — all signals are algorithmic. No training data, no cloud APIs, no GPU required.

---

## What it does

When you work with an AI coding agent, facts about your project accumulate — decisions made, bugs found, preferences stated. engram captures these as embeddings in a local SQLite database and retrieves the most relevant ones at the start of each new session, so the agent already knows what it needs to know.

All data stays on your machine. Nothing is sent to any cloud.

Built for [opencode](https://opencode.ai), compatible with any agent framework via its MCP interface.

---

## Architecture

engram uses a **multi-signal retrieval pipeline** — no single signal is good enough for all queries:

```
Query
  │
  ├─► [Cosine ANN]     384-dim BGE embedding → all project facts
  │
  ├─► [BM25 FTS5]      SQLite FTS5 keyword search (with phrase boost)
  │
  ├─► [Derived BM25]   WordNet synonym-expanded FTS5 query
  │
  ├─► [Lexical Ch.]    Person-name / date-year / key-bigram channels
  │
  ├─► [Context BM25]   Neighboring-turn window (±3) token matching
  │
  ├─► [RRF Fusion]     Reciprocal Rank Fusion of all signals (K=15)
  │
  ├─► [Coverage Guard] Min-rank(RRF_rank, score_rank) — no regression
  │
  ├─► [Cross-Encoder]  mxbai-rerank-xsmall-v1 on top-120 candidates
  │
  └─► [CE Guard]       Min-rank(CE_rank, pre_CE_rank) — no regression
       │
       ▼
  Ranked facts → agent's context window
```

All stages run locally. The cross-encoder is a small 80M-parameter model — runs in ~2s per query on CPU.

---

## Benchmark

Evaluated on **LoCoMo** (ACL 2024) — 1,531 QA pairs across 10 long conversations. The question: does the pipeline return the correct conversation turn in its top-K results?

| Config | R@1 | R@3 | R@5 | R@10 | R@40 | Time |
|--------|-----|-----|-----|------|------|------|
| Cosine only | 48.49% | 65.90% | 73.87% | 82.39% | 92.84% | 8 min |
| + BM25 + RRF + lexical | 52.23% | 71.35% | 77.07% | 85.22% | 94.15% | 11 min |
| **+ Cross-encoder** | **57.29%** | **77.86%** | **82.79%** | **87.91%** | **93.36%** | **64 min** |

**R@3 = 77.86%** means the correct fact appears in the top 3 for 77.86% of questions — and the entire pipeline uses zero training data. All improvements are architectural (signal fusion, guard heuristics, context windowing).

For comparison, the eval-pipeline champion (with a GBM learned reranker trained on LoCoMo) achieves 80.99% R@3. engram's 77.86% closes 77% of that gap with no training.

---

## Install

### Requirements

- Python 3.10+
- [opencode](https://opencode.ai)

### Quick start

```bash
git clone https://github.com/SheldonAntony/engram.git ~/.config/opencode
cd ~/.config/opencode
bash install.sh
```

Windows (PowerShell):
```powershell
git clone https://github.com/SheldonAntony/engram.git $env:USERPROFILE\.config\opencode
cd $env:USERPROFILE\.config\opencode
.\install.ps1
```

This creates a `.venv` and installs all dependencies (fastembed, sentence-transformers, etc.).

### Verify

```bash
python memory.py retrieve_facts test-project test-session "what is the architecture" 3 0.0
```

Expected: `[]` on a fresh install (no facts stored yet).

### MCP clients (Claude Desktop, Cursor, etc.)

Run the MCP server and configure your client to connect via stdio:

```bash
python server.py
```

For Claude Desktop, add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "engram": {
      "command": "python",
      "args": ["/path/to/engram/server.py"]
    }
  }
}
```

---

## Configuration

Edit `preflight.config.json`:

```json
{
  "retrievalConfidenceThreshold": 0.65,
  "topN": 3
}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFLIGHT_RRF_K` | `15` | RRF smoothing constant |
| `PREFLIGHT_USE_DERIVED_BM25` | `0` | Enable WordNet-expanded BM25 |
| `PREFLIGHT_USE_LEXICAL_CHANNELS` | `0` | Enable name/date/bigram channels |
| `PREFLIGHT_USE_CONTEXT_BM25` | `0` | Enable neighboring-turn context BM25 |
| `PREFLIGHT_CONTEXT_WINDOW` | `3` | Turns ±N for context window |
| `PREFLIGHT_CE_POOL` | `120` | Cross-encoder candidate pool size |
| `PREFLIGHT_CE_GUARD_K` | `40` | CE min-rank guard (0=off) |
| `PREFLIGHT_COVERAGE_K` | `40` | Coverage min-rank guard (0=off) |
| `PREFLIGHT_CE_MODEL` | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Cross-encoder model |
| `ENGRAM_EMBED_BACKEND` | `fastembed` | `fastembed` or `sentence-transformers` |
| `ENGRAM_EMBED_MODEL` | *(backend default)* | Custom embedding model |

---

## Platform support

| Platform | Status |
|----------|--------|
| [opencode](https://opencode.ai) | Supported (native plugin) |
| [MCP](https://modelcontextprotocol.io) clients (Claude Desktop, Cursor, etc.) | Supported (run `python server.py`) |

---

## Key design decisions

- **No training data needed** — all signals are algorithmic. Every user gets the same quality on day one.
- **No GPU required** — cross-encoder runs on CPU (~2s per query for 120 pairs).
- **No cloud APIs** — everything runs locally. No telemetry, no data exfiltration.
- **Multi-signal fusion** — single-signal (cosine-only) memory systems miss ~35% of relevant facts. RRF fusion of 4+ signals cuts misses to ~22%.
- **Context window matters** — conversation facts are not independent. Context BM25 (searching neighboring turns) adds +2.2pp R@3 for free.

---

## Repository

- **engram** (this repo — opencode plugin): https://github.com/SheldonAntony/engram
- **engram-eval** (benchmark pipeline): https://github.com/SheldonAntony/engram-eval
