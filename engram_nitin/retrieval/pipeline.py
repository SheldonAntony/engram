"""Retrieval pipeline — the core of Engram.

Three-stage pipeline:
1. Dense retrieval (bi-encoder) — fast approximate search
2. Sparse retrieval (BM25) — keyword matching
3. Fusion (RRF) + optional cross-encoder reranking

This replaces MemPalace's ad-hoc distance reduction heuristics with
principled information retrieval techniques.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from ..backends.base import Document

_QUOTED_RE = re.compile(r"""['"]([^'"]{3,60})['"]""")
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,15}\b")

NOT_NAMES = frozenset(
    {
        "What",
        "When",
        "Where",
        "Who",
        "How",
        "Which",
        "Did",
        "Do",
        "Was",
        "Were",
        "Have",
        "Has",
        "Had",
        "Is",
        "Are",
        "The",
        "My",
        "Our",
        "Their",
        "Can",
        "Could",
        "Would",
        "Should",
        "Will",
        "Shall",
        "May",
        "Might",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "In",
        "On",
        "At",
        "For",
        "To",
        "Of",
        "With",
        "By",
        "From",
        "And",
        "But",
        "Also",
        "Just",
        "Very",
        "More",
        "Previously",
        "Recently",
        "This",
        "That",
        "These",
        "Those",
        "Not",
        "Now",
        "Then",
        "Here",
        "There",
        "Its",
        "Yes",
        "No",
    }
)

TIME_PATTERNS = [
    (r"(\d+)\s+days?\s+ago", lambda m: (int(m.group(1)), 2)),
    (r"a\s+couple\s+(?:of\s+)?days?\s+ago", lambda _: (2, 2)),
    (r"yesterday", lambda _: (1, 1)),
    (r"a\s+week\s+ago", lambda _: (7, 3)),
    (r"(\d+)\s+weeks?\s+ago", lambda m: (int(m.group(1)) * 7, 5)),
    (r"last\s+week", lambda _: (7, 3)),
    (r"a\s+month\s+ago", lambda _: (30, 7)),
    (r"(\d+)\s+months?\s+ago", lambda m: (int(m.group(1)) * 30, 10)),
    (r"last\s+month", lambda _: (30, 7)),
    (r"last\s+year", lambda _: (365, 30)),
    (r"a\s+year\s+ago", lambda _: (365, 30)),
    (r"recently", lambda _: (14, 14)),
    (r"two\s+months?\s+ago", lambda _: (60, 10)),
    (r"three\s+months?\s+ago", lambda _: (90, 14)),
    (r"six\s+months?\s+ago", lambda _: (180, 20)),
]


def extract_person_names(text: str) -> list[str]:
    """Extract likely person names from text."""
    words = _NAME_RE.findall(text)
    return list(set(w for w in words if w not in NOT_NAMES))


def extract_quoted_phrases(text: str) -> list[str]:
    """Extract quoted phrases from text."""
    return [p.strip() for p in _QUOTED_RE.findall(text) if len(p.strip()) >= 3]


def parse_temporal_offset(question: str) -> Optional[Tuple[int, int]]:
    """Extract temporal offset from question. Returns (days_back, tolerance) or None."""
    q = question.lower()
    for pattern, extractor in TIME_PATTERNS:
        m = re.search(pattern, q)
        if m:
            return extractor(m)
    return None


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse common date formats."""
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.split(" (")[0].split("T")[0].strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def reciprocal_rank_fusion(
    *rankings: List[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each ranking is a list of (doc_id, score) sorted by score descending.
    RRF score = sum over rankings of 1 / (k + rank).

    This is more principled than MemPalace's distance reduction approach
    because it's rank-based (not score-based) — different scoring scales
    don't need manual calibration.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return sorted_fused


class RetrievalPipeline:
    """Three-stage retrieval: dense + sparse + reranker.

    This is the main search interface. Callers provide a query; the pipeline
    returns ranked documents.

    Stages:
    1. Dense retrieval via bi-encoder (bge-large) — recall-oriented
    2. BM25 scoring over candidates — keyword signal
    3. RRF fusion of dense + sparse rankings
    4. Optional cross-encoder reranking — precision-oriented

    Temporal and entity boosts are applied as additional ranking signals
    fused via RRF, not as ad-hoc distance multipliers.
    """

    def __init__(
        self,
        embedder=None,
        reranker=None,
        use_reranker: bool = True,
        dense_top_k: int = 50,
    ):
        from .embedder import Embedder

        self.embedder = embedder or Embedder()
        self._reranker = reranker
        self._use_reranker = use_reranker
        self._dense_top_k = dense_top_k

        # Lazy load
        self._bm25 = None

    def _get_bm25(self):
        if self._bm25 is None:
            from .sparse import BM25

            self._bm25 = BM25()
        return self._bm25

    def _get_reranker(self):
        if self._reranker is None and self._use_reranker:
            from .reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        return self._reranker

    def search(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        question_date: Optional[str] = None,
    ) -> List[Document]:
        """Full pipeline search over a document set.

        For benchmarking: pass pre-embedded documents. The pipeline handles
        query encoding, BM25, fusion, and reranking.
        """
        if not documents:
            return []

        # Stage 1: Dense retrieval
        query_vec = self.embedder.encode_query(query)
        dense_ranking = self._dense_rank(query_vec, documents)

        # Stage 2: BM25 sparse ranking
        bm25 = self._get_bm25()
        doc_texts = [d.text for d in documents]
        bm25_scores = bm25.score_query_against_docs(query, doc_texts)
        sparse_ranking = [
            (documents[i].id, s)
            for i, s in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        ]

        # Stage 2.5: Entity and temporal boost rankings
        boost_rankings = self._compute_boost_rankings(query, documents, question_date)

        # Stage 3: RRF fusion
        all_rankings = [dense_ranking, sparse_ranking] + boost_rankings
        fused = reciprocal_rank_fusion(*all_rankings)

        # Build candidate list from fusion
        id_to_doc = {d.id: d for d in documents}
        candidates = []
        for doc_id, rrf_score in fused:
            if doc_id in id_to_doc:
                doc = id_to_doc[doc_id]
                doc.score = rrf_score
                candidates.append(doc)

        # Stage 4: Cross-encoder reranking (over top candidates)
        reranker = self._get_reranker()
        if reranker and len(candidates) > 1:
            rerank_pool = candidates[: min(20, len(candidates))]
            rerank_texts = [d.text for d in rerank_pool]
            reranked = reranker.rerank(query, rerank_texts, top_k=top_k)
            result = []
            for orig_idx, ce_score in reranked:
                doc = rerank_pool[orig_idx]
                doc.score = ce_score
                result.append(doc)
            # Append any candidates beyond the rerank pool
            reranked_ids = {rerank_pool[idx].id for idx, _ in reranked}
            for doc in candidates:
                if doc.id not in reranked_ids and len(result) < top_k:
                    result.append(doc)
            return result[:top_k]

        return candidates[:top_k]

    def _dense_rank(self, query_vec, documents: List[Document]) -> List[Tuple[str, float]]:
        """Rank documents by cosine similarity to query vector."""
        import numpy as np

        doc_embeddings = []
        for doc in documents:
            if doc.embedding is not None:
                doc_embeddings.append(doc.embedding)
            else:
                raise ValueError(f"Document {doc.id} has no embedding")

        matrix = np.array(doc_embeddings)
        q = np.array(query_vec)

        # Cosine similarity (embeddings are already normalized)
        scores = matrix @ q

        ranked = sorted(
            zip(range(len(documents)), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(documents[i].id, float(s)) for i, s in ranked]

    def _compute_boost_rankings(
        self,
        query: str,
        documents: List[Document],
        question_date: Optional[str] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Compute additional ranking signals: entity names, quoted phrases, temporal."""
        rankings = []

        # Person name boost
        names = extract_person_names(query)
        if names:
            name_scores = []
            for doc in documents:
                text_lower = doc.text.lower()
                hits = sum(1 for n in names if n.lower() in text_lower)
                score = hits / len(names) if names else 0.0
                name_scores.append((doc.id, score))
            name_scores.sort(key=lambda x: x[1], reverse=True)
            if any(s > 0 for _, s in name_scores):
                rankings.append(name_scores)

        # Quoted phrase boost
        phrases = extract_quoted_phrases(query)
        if phrases:
            phrase_scores = []
            for doc in documents:
                text_lower = doc.text.lower()
                hits = sum(1 for p in phrases if p.lower() in text_lower)
                score = hits / len(phrases) if phrases else 0.0
                phrase_scores.append((doc.id, score))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            if any(s > 0 for _, s in phrase_scores):
                rankings.append(phrase_scores)

        # Temporal proximity boost
        time_offset = parse_temporal_offset(query)
        if time_offset and question_date:
            q_date = parse_date(question_date)
            if q_date:
                days_back, tolerance = time_offset
                target = q_date - timedelta(days=days_back)
                temporal_scores = []
                for doc in documents:
                    ts = (doc.metadata or {}).get("timestamp", "")
                    doc_date = parse_date(ts) if ts else None
                    if doc_date:
                        delta = abs((doc_date - target).days)
                        if delta <= tolerance:
                            score = 1.0
                        elif delta <= tolerance * 3:
                            score = 1.0 - (delta - tolerance) / (tolerance * 2)
                        else:
                            score = 0.0
                    else:
                        score = 0.0
                    temporal_scores.append((doc.id, score))
                temporal_scores.sort(key=lambda x: x[1], reverse=True)
                if any(s > 0 for _, s in temporal_scores):
                    rankings.append(temporal_scores)

        return rankings
