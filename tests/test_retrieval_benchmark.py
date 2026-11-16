import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from benchmark_retrieval import evaluate_retriever


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_reports_ranking_context_and_errors_with_clear_denominators(self):
        retriever = Mock()
        retriever.search.side_effect = [
            [
                {
                    "id": "c1",
                    "source_id": "doc",
                    "source": "doc.pdf",
                    "page": 1,
                    "text": "The evidence quote is present.",
                    "metadata": {},
                    "score": 0.9,
                }
            ],
            ConnectionError("vector store unavailable"),
        ]
        questions = [
            {
                "id": "q1",
                "split": "dev",
                "question": "What is supported?",
                "gold": [
                    {
                        "source_id": "doc",
                        "page": 1,
                        "evidence": "evidence quote",
                    }
                ],
            },
            {
                "id": "q2",
                "split": "dev",
                "question": "Second query?",
                "gold": [],
            },
        ]

        traces, summary = evaluate_retriever(
            retriever,
            questions,
            top_k=3,
            budget=800,
            method="dense",
        )

        self.assertTrue(traces[0]["hit_at_1"])
        self.assertTrue(traces[0]["evidence_hit_at_3"])
        self.assertTrue(traces[0]["context_evidence"])
        self.assertEqual(traces[1]["status"], "error")
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["evidence_hit_at_3"], {
            "count": 1,
            "total": 1,
            "rate": 1.0,
        })


if __name__ == "__main__":
    unittest.main()
