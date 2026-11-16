"""Lexical and hybrid retrieval methods for comparing against Day 6 dense search."""
import math
import json
from collections import Counter
from typing import Any

from baseline import tokenize


class BM25Index:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 k1 must be positive and b must be between 0 and 1.")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )
        self.idf = {
            term: math.log(
                1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        terms = Counter(tokenize(query))
        if not terms:
            return []

        results = []
        for chunk, counts, length in zip(
            self.chunks, self.term_counts, self.lengths
        ):
            metadata = chunk.get("metadata", {})
            if filters and any(metadata.get(key) != value for key, value in filters.items()):
                continue
            score = 0.0
            for term, query_frequency in terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b
                    + self.b * length / (self.average_length or 1.0)
                )
                score += (
                    self.idf[term]
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                    * query_frequency
                )
            if score > 0:
                results.append({**chunk, "score": score, "strategy": "bm25"})
        return sorted(
            results, key=lambda row: (-row["score"], row["id"])
        )[:top_k]


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    *,
    top_k: int = 3,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    if top_k <= 0 or rank_constant <= 0:
        raise ValueError("top_k and rank_constant must be greater than zero.")
    fused: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "score": 0.0,
                    "rank_sources": [],
                    "strategy": "hybrid_rrf",
                }
            fused[chunk_id]["score"] += 1 / (rank_constant + rank)
            fused[chunk_id]["rank_sources"].append(rank)
    return sorted(
        fused.values(), key=lambda row: (-row["score"], row["id"])
    )[:top_k]


class HybridRetriever:
    """Fuse dense Day 6 and lexical BM25 rankings without mixing score scales."""

    def __init__(self, dense_retriever: Any, lexical_index: BM25Index) -> None:
        self.dense_retriever = dense_retriever
        self.lexical_index = lexical_index

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        dense = self.dense_retriever.search(
            query, filters=filters, top_k=top_k
        )
        lexical = self.lexical_index.search(
            query, filters=filters, top_k=top_k
        )
        return reciprocal_rank_fusion([dense, lexical], top_k=top_k)


class CachingRetriever:
    """Reuse identical real retrieval calls across comparable benchmark methods."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self._cache: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        cache_key = (
            query,
            json.dumps(filters, ensure_ascii=False, sort_keys=True),
            top_k,
        )
        if cache_key not in self._cache:
            self._cache[cache_key] = self.retriever.search(
                query, filters=filters, top_k=top_k
            )
        return [dict(result) for result in self._cache[cache_key]]
