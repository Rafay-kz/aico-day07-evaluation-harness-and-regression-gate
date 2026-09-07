from __future__ import annotations

import math
import re
from collections import Counter

from aico.retrieval.chunker import tokenize

K1 = 1.5
B = 0.75

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "will",
        "with",
        "would",
    }
)

_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def tokenize_for_search(text: str) -> list[str]:
    """Whitespace split (same as the chunker), then fold case and strip punctuation."""
    terms: list[str] = []
    for raw in tokenize(text):
        folded = _PUNCT_RE.sub("", raw.lower())
        if folded and folded not in STOPWORDS:
            terms.append(folded)
    return terms


class BM25Index:
    """Lexical index over chunk records."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = list(chunks)
        self._term_lists = [tokenize_for_search(chunk["text"]) for chunk in self.chunks]
        self._term_freqs = [Counter(terms) for terms in self._term_lists]
        self._doc_len = [len(terms) for terms in self._term_lists]
        self._n = len(self.chunks)
        total_len = sum(self._doc_len)
        self._avgdl = total_len / self._n if self._n else 0.0
        self._df: Counter[str] = Counter()
        for freq in self._term_freqs:
            self._df.update(freq.keys())

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k chunks ranked by BM25 score."""
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        query_terms = list(dict.fromkeys(tokenize_for_search(query)))
        if not query_terms or not self.chunks:
            return []

        scored: list[tuple[float, str, int, dict]] = []
        for offset, chunk in enumerate(self.chunks):
            score = self._score(query_terms, offset)
            scored.append((score, chunk["chunk_id"], offset, chunk))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))

        hits: list[dict] = []
        for rank, (score, _chunk_id, _offset, chunk) in enumerate(scored[:top_k], start=1):
            hit = dict(chunk)
            hit["rank"] = rank
            hit["score"] = score
            hits.append(hit)
        return hits

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        tf_map = self._term_freqs[doc_index]
        dl = self._doc_len[doc_index]
        avgdl = self._avgdl if self._avgdl else 1.0
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            df = self._df[term]
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            denom = tf + K1 * (1.0 - B + B * dl / avgdl)
            score += idf * (tf * (K1 + 1.0)) / denom
        return score
