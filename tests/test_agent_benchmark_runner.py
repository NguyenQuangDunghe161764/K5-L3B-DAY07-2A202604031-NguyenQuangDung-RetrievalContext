import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from run_agent_benchmark import build_retriever, run_cases


class AgentBenchmarkRunnerTests(unittest.TestCase):
    def test_builds_bm25_without_connecting_to_vector_database(self):
        lexical = Mock()
        with patch(
            "run_agent_benchmark.load_day6_local_chunks",
            return_value=[{"id": "c1", "text": "evidence", "metadata": {}}],
        ) as load_chunks, patch(
            "run_agent_benchmark.BM25Index", return_value=lexical
        ) as index_type, patch(
            "run_agent_benchmark.connect_day6"
        ) as connect:
            retriever = build_retriever(Path("day6"), "A_fixed_500_50", "bm25")

        self.assertIsNotNone(retriever)
        load_chunks.assert_called_once_with(
            Path("day6"), strategy="A_fixed_500_50"
        )
        index_type.assert_called_once()
        connect.assert_not_called()

    def test_builds_hybrid_from_day6_dense_and_bm25(self):
        dense = Mock()
        lexical = Mock()
        chunks = [{"id": "c1", "text": "evidence", "metadata": {}}]
        with patch(
            "run_agent_benchmark.load_day6_local_chunks",
            return_value=chunks,
        ), patch(
            "run_agent_benchmark.connect_day6", return_value=dense
        ) as connect, patch(
            "run_agent_benchmark.BM25Index", return_value=lexical
        ), patch(
            "run_agent_benchmark.HybridRetriever"
        ) as hybrid_type:
            retriever = build_retriever(
                Path("day6"), "C_paragraph_filter", "hybrid"
            )

        connect.assert_called_once_with(
            Path("day6"), strategy="C_paragraph_filter"
        )
        hybrid_type.assert_called_once_with(dense, lexical)
        self.assertIs(retriever, hybrid_type.return_value)

    def test_runs_only_selected_split_and_records_model_errors(self):
        cases = [
            {"id": "dev", "split": "dev", "question": "Question?"},
            {"id": "test", "split": "test", "question": "Test?"},
        ]
        controller = Mock()
        retriever = Mock()
        controller.decide.side_effect = [
            {"action": "search", "query": "focused query"},
            RuntimeError("model unavailable"),
        ]
        retriever.search.return_value = [
            {
                "id": "c1",
                "text": "Evidence.",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 1,
                "metadata": {},
                "score": 0.9,
            }
        ]

        traces = run_cases(cases, "dev", controller, retriever)

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["query_id"], "dev")
        self.assertEqual(traces[0]["status"], "error")
        self.assertIn("model unavailable", traces[0]["error"])
        retriever.search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
