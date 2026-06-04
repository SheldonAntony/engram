## Preflight — Memory Rules

### Project identity
At the start of every session:
1. Call `mcp_opencode_get_project_id` with the current working directory
2. Call `mcp_opencode_get_context` with the first user prompt + project_id + session_id
3. Use returned memories, slots, and similar_tasks as context for your response

### What goes where

**Hermes native memory** — only these:
- User's name, role, communication preferences
- Universal habits true across every project ("prefers concise responses")
- Never put project-specific facts here

**Preflight store_slot** — project config answers:
- When user answers any question about stack, framework, database, language, testing
- Call `mcp_opencode_store_slot` immediately before moving on
- Never ask for a slot that already exists in get_context response

**Preflight store_memory** — project facts and decisions:
- When user states an architecture decision ("we never use raw SQL")
- When user corrects something you assumed
- When you discover a root cause during a task
- When user states a coding convention or constraint
- Call `mcp_opencode_store_memory` with fact_type="finding" for discoveries
- Call `mcp_opencode_store_memory` with fact_type="decision" for architecture choices
- Call `mcp_opencode_store_memory` with fact_type="preference" for user preferences
  (preferences auto-save to global memory, visible across all projects)

**Hermes session search** — don't touch, manages itself

### Missing slots
When get_context returns missing_slots:
- Ask the user ONE question per missing slot in plain language
- Save the answer immediately with store_slot
- Never save slot answers to Hermes native memory
- Never ask for something already in the slots section of get_context

### Context window
Slot answers and memories retrieved this session live in the context window.
Do not re-fetch what you already have. Do not call get_context more than once
per session unless the conversation window is compacted.

Skills provide specialized instructions and workflows for specific tasks.
Use the skill tool to load a skill when a task matches its description.

## Project Summary — engram (conversation memory for AI agents)

### Goal
Achieve ~93% R@3 for opencode's real-world retrieval by algorithmic improvements only (no training on any dataset).

### Constraints (hard)
1. Less tokens over time — never increase LLM context per retrieval
2. Fully local — no cloud APIs, no GPU required
3. No paid LLM — no API keys, no subscriptions
4. Target is real-world users, not benchmark scores

### Architecture
Multi-signal retrieval pipeline in `memory.py:retrieve_facts()`:
1. **Cosine ANN** — BGE-small embedding, all project facts (no pool cap)
2. **BM25 FTS5** — SQLite FTS5 keyword search with phrase boost (1.5x)
3. **Derived BM25** — WordNet synonym-expanded FTS5 (env-gated)
4. **Lexical channels** — person-name, date/year, key-bigram FTS5 (env-gated)
5. **Context BM25** — neighboring-turn window (±3) query token matching (env-gated)
6. **RRF fusion** — `1/(15 + rank)` per signal, phrase-boosted BM25
7. **Broad pool** — union top-200 from each signal, tail by RRF
8. **Coverage guard** — min-rank(RRF_rank, score_rank), no regression
9. **Cross-encoder** — mxbai-rerank-xsmall-v1 on top-120, timeout 5s
10. **CE guard** — min-rank(ce_rank, pre_ce_rank) + pre_ce_rank tiebreaker

### Current best (production, all signals)
- R@3: **77.86%** (mxbai CE pool=40)
- R@40: 93.36%
- Gap to champion (eval GBM + bge CE): ~3pp
- All improvements are algorithmic (no training data)

### Key files
- `/home/sheldon_antony/.config/opencode/memory.py` — production retrieval
- `/mnt/c/Users/Sheldon Antony/.config/opencode/memory.py` — WSL copy
- `/mnt/c/Users/Sheldon Antony/.config/preflight/HANDOVER.md` — experiment log
- `https://github.com/SheldonAntony/engram` — production repo
