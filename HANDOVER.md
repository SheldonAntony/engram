
## 2026-05-25: Retrieval improvements for 93% R@3 target

### Changes made (memory.py)
1. **Composite scoring weights rebalanced** — RRF weight increased from 0.35 to 0.60. Recency/staleness/session_rec/freq reduced from combined 0.65 to 0.40. Prevents dilution of similarity signal by metadata noise.
2. **Adaptive session fallback** — When `session_idx_map` has < 3 entries, session_rec weight (0.12) shifts automatically to RRF (0.72). Benchmarks with no session metadata no longer degrade.
3. **Per-signal RRF weights** — New env var `PREFLIGHT_RRF_WEIGHTS` (format: `"vec=1.0,bm25=1.2,entity=0.8"`). Empty = equal weights (default behavior). Parsed at module load time.
4. **CE pool increased** — Default from 120 to 160 (env `PREFLIGHT_CE_POOL`).
5. **HyDE query expansion** — New env `PREFLIGHT_USE_QUERY_EXPANSION=1` enables Qwen2.5-1.5b (via Ollama) to generate hypothetical answer passages. Query embedding is interpolated: `0.6 × query_emb + 0.4 × hyde_emb`. Silently no-ops when Ollama unavailable.

### Configuration
```bash
# Per-signal RRF weights (dense=1.0, BM25=1.2, entity=0.8)
PREFLIGHT_RRF_WEIGHTS="vec=1.0,bm25=1.2,entity=0.8"

# HyDE expansion with Qwen2.5-1.5b
PREFLIGHT_USE_QUERY_EXPANSION=1

# Composite scoring weights (overrides defaults)
PREFLIGHT_W_RRF=0.65
PREFLIGHT_W_RECENCY=0.10
```

### Competitor analysis (from web research)
- **mem0**: Entity graph linking at index time (not query time). Multi-level scoping (user × agent × run).
- **Zep/Graphiti**: Temporal knowledge graph with validity windows. BFS entity expansion. Auto character-budget packing.
- **CrewAI**: LLM query analysis → sub-query generation → confidence-based routing → recursive exploration.
- **MemGPT/Letta**: Self-directed memory where the agent manages what to store/retrieve. Background dream subagents for consolidation.
- **HyDE (Qwen3-1.7B)**: Hypothetical Document Embeddings — fine-tuned query expansion model exists (`tobil/qmd-query-expansion-1.7B-gguf`, Q4_K_M ~1GB).
- **mxbai-rerank-base-v2**: 0.5B params, 55.57 BEIR NDCG@10 — best accuracy/speed tradeoff for CPU.

### Implemented from competitor analysis
- HyDE query expansion (from MemGPT/CrewAI/HyDE literature)
- Per-signal RRF weights (from RRF research — weighted RRF outperforms uniform RRF)
- Adaptive composite scoring (from CrewAI's confidence-based routing)

### Next steps for 93% R@3
1. **Tune weights** via grid search on 50-100 labeled production queries: `PREFLIGHT_RRF_WEIGHTS`, `PREFLIGHT_W_RRF`, `_CE_POOL_SIZE`
2. **BGE-base embedding** (768d vs 384d) — add `PREFLIGHT_EMBED_MODEL` env var for model selection
3. **mxbai-rerank-base-v2** as CE upgrade — ~0.5B params, 55.57 BEIR vs 43.9 for current xsmall
4. **Entity graph linking at index time** — store entities not just at query time but index time for faster entity matching
5. **Background reflection passes** — periodic re-embedding + consolidation (from Letta)
6. **FSRS-style retrievability decay** — replace simple exponential recency with DSR model

### Implemented (v22)
- **Entity extraction at index time**: turn/window facts now store capitalized-word entities (regex-based, ~0.01ms, catches speaker names). Enables entity overlap scoring for conversational facts.
- **FSRS-style retrievability decay**: Replaced `1/(1+decay*days)` with `exp(-days/stability)` where `stability = max(1.0, (rc+1)*ef)`. Facts with more retrievals and higher EF decay slower — SM-2-inspired.
- **Auto-embedding-dimension migration**: Detects when stored embedding dim differs from current model's dim (e.g. BGE-small 384d → BGE-base 768d), re-embeds all facts silently.
- **Weight tuning script**: `tune_weights.py --rrf-only --n 30` grid-searches RRF signal weights (vec/bm25/entity) on LoCoMo data by modifying module vars in-process.

### Remaining gap to 93%
Current R@3 best: 77.86% (mxbai CE pool=40). Changes in v21+v22 should close ~5-8pp. Biggest remaining levers:
1. **Run `tune_weights.py --full`** to optimize composite+RFF weights
2. **Upgrade CE** to mxbai-rerank-base-v1 or bge-reranker-v2-base via `PREFLIGHT_CE_MODEL`
3. **HyDE expansion** — enable via `PREFLIGHT_USE_QUERY_EXPANSION=1` with Ollama running
4. **Run the new eval pipeline** on production data (not LoCoMo) to measure actual R@3
