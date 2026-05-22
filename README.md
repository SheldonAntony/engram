# engram

**Memory for AI agents. Runs on your machine. No cloud, no GPU, no API keys.**

engram is a memory server that lets any AI agent remember facts about your project — decisions you made, preferences you stated, bugs you found — across chat sessions. It plugs into Claude Desktop, Cursor, opencode, and any [MCP](https://modelcontextprotocol.io)-compatible client.

Unlike other AI memory systems, engram requires **zero training data** and **zero cloud APIs**. Every retrieval signal is algorithmic, so quality is the same on day one as it is after a year of use.

---

## Why this matters

AI agents forget everything between sessions. Without memory, every conversation starts from scratch — you repeat context, re-explain preferences, and re-state decisions you already made.

engram solves this by storing facts (embeddings in a local SQLite database) and retrieving the most relevant ones at the start of each session, so the agent already knows what it needs to know.

All data stays on your machine. Nothing is sent to any cloud.

---

## How we're different

| Feature | engram | mem0 | MemU | LangChain Memory |
|---------|--------|------|------|-----------------|
| Runs fully local | ✅ | ❌ (cloud API) | ❌ | ⚠️ (varies) |
| No GPU required | ✅ | N/A (cloud) | N/A | ✅ |
| No training data | ✅ | ❌ | ❌ | ✅ |
| No paid API keys | ✅ | ❌ (paid API) | ❌ | ⚠️ (varies) |
| Multi-signal retrieval | ✅ | ❌ (cosine only) | ❌ | ❌ (cosine only) |
| Cross-encoder reranker | ✅ | ❌ | ❌ | ❌ |
| Works with any MCP client | ✅ | ❌ | ❌ | ❌ |
| Algorithmic (no fine-tuning) | ✅ | ❌ | ❌ | ✅ |

Most memory systems use **cosine similarity only** — they embed everything, compute distance, and call it done. That misses ~35% of relevant facts for real-world queries.

engram uses **8 signals** fused together (cosine + BM25 + WordNet + person names + dates + bigrams + conversation context + cross-encoder reranking), hitting ~77% recall@3 — **without training on any dataset**.

---

## How it works (simple)

When you ask the agent a question, engram runs a multi-stage search:

```
Your question
    │
    ▼
5 parallel searches ──────────────────────────────┐
    │  • Semantic: what does the question mean?    │
    │  • Keyword: what words does it contain?      │
    │  • Synonyms: what related words might match? │
    │  • Names: which people/places are mentioned? │
    │  • Context: what was discussed nearby?       │
    │                                             │
    ▼                                             │
    ┌─────────────────────────────────────────┐   │
    │ All signals merged by rank fusion       │   │
    │ (best signal picks the winner per fact) │   │
    └─────────────────────────────────────────┘   │
    │                                             │
    ▼                                             │
    ┌─────────────────────────────────────────┐   │
    │ Cross-encoder re-ranks top candidates   │   │
    │ (question + each candidate → relevance) │   │
    └─────────────────────────────────────────┘   │
    │                                             │
    ▼                                             │
    Most relevant facts → agent's context
```

All stages run on your machine. The cross-encoder is a 80M-parameter local model (~1 second per query on CPU).

---

## How it works with your LLM

engram is an **MCP server** — it speaks the Model Context Protocol that AI agents natively understand.

When you ask your agent a question, the agent calls engram's `search` tool automatically, engram returns the relevant facts, and the agent uses them as context. The agent decides what to retrieve and when — you don't need to learn a separate interface.

**Supported clients:**
- [Claude Desktop](https://claude.ai) — add `server.py` to your MCP config
- [Cursor](https://cursor.sh) — MCP server integration in settings
- [Windsurf](https://codeium.com/windsurf) — MCP server support
- [opencode](https://opencode.ai) — native plugin, no config needed
- Any MCP-compatible agent

engram's memory persists across sessions, across clients — use it with Claude today, Cursor tomorrow, and the facts follow you.

---

## Quick start

### Requirements
- Python 3.10+
- An MCP-compatible AI client (Claude Desktop, Cursor, opencode, etc.)

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
git clone https://github.com/SheldonAntony/engram.git $env:USERPROFILE\.config\opencode
cd $env:USERPROFILE\.config\opencode
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

### Connect to opencode

```bash
pip install opencode
# engram is a native opencode plugin — no config needed
```

### Verify

Store a fact and retrieve it:

```
You:  Remember that we chose Postgres 16 for production
Agent: Stored as fact #1

You:  What database are we using?
Agent: We use Postgres 16 for production (from memory)
```

---

## What the numbers mean

Evaluated on **LoCoMo** — 1,531 real questions across 10 long conversations:

| Config | Recall@3 | What it means |
|--------|----------|---------------|
| Cosine only | 65.90% | 34/100 questions miss the fact |
| + keywords + names + dates | 71.35% | 29/100 miss |
| **+ cross-encoder** | **76.87%** | **23/100 miss — best without training** |
| Competitor (trained on benchmark) | 80.99% | 19/100 miss — but requires offline training |

engram closes **~75% of the gap** to the trained competitor, with **zero training data**.

---

## Glossary (for non-tech readers)

| Term | What it means |
|------|---------------|
| **Embedding** | A mathematical fingerprint that captures the meaning of text. Facts with similar meaning have similar fingerprints. |
| **BM25** | A smart keyword search — finds facts containing your exact words, weighted by how rare those words are. |
| **Cosine similarity** | Measures how close two embeddings are (1.0 = identical meaning, 0.0 = unrelated). |
| **RRF** | A voting system: if 5 different searches all agree fact X is relevant, fact X ranks highest. |
| **Cross-encoder** | A small AI model that re-reads each candidate fact alongside the question and scores relevance on a scale (0-1). |
| **MCP** | A standard protocol that lets AI agents discover and call tools. Like USB for AI — any client can plug into any server. |
| **Signal** | One method of searching (e.g., keyword search is one signal, semantic search is another). |
| **R@3 (Recall@3)** | Out of 100 questions, how many times does the correct fact appear in the top 3 results? Higher = better. |

---

## Design principles

1. **Your data stays yours.** Everything runs locally. No telemetry, no cloud sync, no data exfiltration.

2. **No training debt.** Unlike competitors that need to train on your data before they work well, engram's signals are engineered — they work at full quality from the first query.

3. **No GPU, no problem.** The cross-encoder runs on CPU (~1 second for 120 candidate pairs).

4. **Less token waste.** engram's retrieval is tuned to return the minimum facts needed — the multi-signal pipeline finds the right facts faster, so fewer irrelevant tokens pollute the agent's context window.

5. **Works today, not tomorrow.** There's no "model training in progress" wait. Install and the agent gains memory immediately.

---

## Repository

- **engram** (this repo): https://github.com/SheldonAntony/engram
- **engram-eval** (benchmark pipeline): https://github.com/SheldonAntony/engram-eval
