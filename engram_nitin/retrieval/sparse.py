"""BM25 sparse retrieval — keyword matching that complements dense retrieval."""

import math
import re

_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)

# Common English stop words for BM25
STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "how",
        "why",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "am",
        "any",
        "as",
        "back",
        "because",
        "before",
        "between",
        "both",
        "come",
        "each",
        "few",
        "get",
        "got",
        "go",
        "here",
        "him",
        "he",
        "she",
        "her",
        "me",
        "i",
        "you",
        "we",
        "they",
        "them",
        "there",
        "up",
        "out",
        "into",
        "over",
        "own",
        "same",
        "some",
        "such",
        "only",
        "other",
        "more",
        "most",
        "make",
        "made",
        "much",
        "many",
        "well",
        "way",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase + extract tokens of length >= 2, removing stop words."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


class BM25:
    """Okapi BM25 scorer for a small corpus.

    Designed for re-ranking a candidate set from dense retrieval, not for
    full-corpus search. IDF is computed over the provided corpus.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}
        self._n_docs: int = 0

    def index(self, documents: list[str]) -> None:
        """Build BM25 index over a corpus."""
        self._corpus_tokens = [tokenize(d) for d in documents]
        self._doc_lens = [len(t) for t in self._corpus_tokens]
        self._n_docs = len(documents)
        self._avgdl = sum(self._doc_lens) / self._n_docs if self._n_docs else 1.0

        self._df = {}
        for tokens in self._corpus_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def score(self, query: str) -> list[float]:
        """Score all indexed documents against a query. Returns list of scores."""
        query_terms = set(tokenize(query))
        if not query_terms or self._n_docs == 0:
            return [0.0] * self._n_docs

        # Precompute IDF (BM25+ smoothed, always non-negative)
        idf = {}
        for term in query_terms:
            df = self._df.get(term, 0)
            idf[term] = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

        scores = []
        for tokens, dl in zip(self._corpus_tokens, self._doc_lens):
            if dl == 0:
                scores.append(0.0)
                continue
            tf: dict[str, int] = {}
            for t in tokens:
                if t in query_terms:
                    tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for term, freq in tf.items():
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += idf[term] * num / den
            scores.append(s)

        return scores

    def score_query_against_docs(self, query: str, documents: list[str]) -> list[float]:
        """One-shot: index documents and score query against them."""
        self.index(documents)
        return self.score(query)
