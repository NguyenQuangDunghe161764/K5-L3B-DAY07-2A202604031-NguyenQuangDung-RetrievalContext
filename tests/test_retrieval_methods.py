import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from retrieval_methods import (
    BM25Index,
    CachingRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)


def chunk(chunk_id, text, chapter):
    return {
        "id": chunk_id,
        "source_id": "doc",
        "source": "doc.pdf",
        "page": 1,
        "text": text,
        "metadata": {"chapter": chapter},
    }


class RetrievalMethodTests(unittest.TestCase):
    def test_bm25_ranks_matching_chunks_and_applies_metadata_filter(self):
        index = BM25Index(
            [
                chunk("a", "water supports living cells", 2),
                chunk("b", "membrane transport in cells", 5),
            ]
        )

        results = index.search("water cells", top_k=2)
        filtered = index.search(
            "cells", filters={"chapter": 5}, top_k=3
        )

        self.assertEqual(results[0]["id"], "a")
        self.assertEqual([row["id"] for row in filtered], ["b"])

    def test_bm25_rejects_invalid_configuration_and_inputs(self):
        with self.assertRaises(ValueError):
            BM25Index([], k1=0)
        index = BM25Index([])
        with self.assertRaises(ValueError):
            index.search("query", top_k=0)

    def test_rrf_merges_rankings_without_comparing_raw_scores(self):
        first = chunk("a", "first", 1)
        second = chunk("b", "second", 1)
        results = reciprocal_rank_fusion(
            [
                [{**first, "score": 900.0}, {**second, "score": 1.0}],
                [{**second, "score": 0.0001}, {**first, "score": 0.0}],
            ],
            top_k=2,
            rank_constant=1,
        )

        self.assertEqual([result["id"] for result in results], ["a", "b"])
        self.assertEqual(results[0]["rank_sources"], [1, 2])
        self.assertEqual(results[0]["score"], results[1]["score"])

    def test_hybrid_passes_filters_and_fuses_dense_and_bm25_results(self):
        dense = Mock()
        dense.search.return_value = [
            {**chunk("dense", "dense evidence", 2), "score": 0.9}
        ]
        lexical = BM25Index(
            [chunk("lexical", "matching biology evidence", 2)]
        )
        hybrid = HybridRetriever(dense, lexical)

        results = hybrid.search(
            "biology evidence",
            filters={"chapter": 2},
            top_k=3,
        )

        dense.search.assert_called_once_with(
            "biology evidence", filters={"chapter": 2}, top_k=3
        )
        self.assertEqual({row["id"] for row in results}, {"dense", "lexical"})

    def test_caching_retriever_reuses_identical_search(self):
        source = Mock()
        source.search.return_value = [{**chunk("a", "evidence", 1), "score": 0.9}]
        retriever = CachingRetriever(source)

        first = retriever.search("query", filters={"chapter": 1}, top_k=3)
        retriever.search("query", filters={"chapter": 1}, top_k=3)
        retriever.search("query", filters={"chapter": 2}, top_k=3)

        self.assertEqual(source.search.call_count, 2)
        first[0]["text"] = "mutated copy"
        self.assertNotEqual(
            retriever.search("query", filters={"chapter": 1}, top_k=3)[0]["text"],
            "mutated copy",
        )


if __name__ == "__main__":
    unittest.main()
