import sys
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from day6_retriever import Day6Retriever, connect_day6


class Day6RetrieverTests(unittest.TestCase):
    def test_search_embeds_query_and_adapts_day6_hits(self):
        embedder = Mock()
        embedder.embed_query.return_value = [0.1, 0.2]
        store = Mock()
        store.dimensions = 2
        store.search.return_value = [
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "source": "doc.pdf",
                "page": 2,
                "text": "Evidence.",
                "metadata": {"chapter": 1},
                "score": 0.9,
            }
        ]
        retriever = Day6Retriever(embedder, store, "A_fixed_500_50")

        hits = retriever.search(
            "question", filters={"chapter": 1}, top_k=5
        )

        embedder.embed_query.assert_called_once_with("question")
        store.search.assert_called_once_with(
            strategy="A_fixed_500_50",
            query_vector=[0.1, 0.2],
            top_k=5,
            filters={"chapter": 1},
        )
        self.assertEqual(
            hits,
            [
                {
                    "id": "c1",
                    "text": "Evidence.",
                    "source_id": "doc1",
                    "source": "doc.pdf",
                    "page": 2,
                    "metadata": {"chapter": 1},
                    "score": 0.9,
                }
            ],
        )

    def test_search_rejects_embedding_dimension_mismatch(self):
        embedder = Mock()
        embedder.embed_query.return_value = [0.1]
        store = Mock()
        store.dimensions = 2
        retriever = Day6Retriever(embedder, store, "A_fixed_500_50")

        with self.assertRaisesRegex(ValueError, "index uses 2"):
            retriever.search("question", filters=None, top_k=3)

        store.search.assert_not_called()

    def test_search_rejects_invalid_query_and_top_k(self):
        retriever = Day6Retriever(Mock(), Mock(), "A_fixed_500_50")

        with self.assertRaises(ValueError):
            retriever.search(" ", filters=None, top_k=3)
        with self.assertRaises(ValueError):
            retriever.search("question", filters=None, top_k=0)

    def test_connect_day6_loads_model_strategy_and_existing_dimension(self):
        embedder = Mock()
        embedding_module = SimpleNamespace(
            GeminiEmbedder=Mock(return_value=embedder),
            MyEmbedder=Mock(),
        )
        store_instance = Mock()
        store_module = SimpleNamespace(
            PgVectorStore=Mock(return_value=store_instance)
        )
        connection = Mock()
        cursor = Mock()
        connection.__enter__ = Mock(return_value=connection)
        connection.__exit__ = Mock(return_value=False)
        connection.cursor.return_value.__enter__ = Mock(return_value=cursor)
        connection.cursor.return_value.__exit__ = Mock(return_value=False)
        cursor.fetchone.return_value = (1536,)

        with tempfile.TemporaryDirectory() as temporary_directory:
            starter = Path(temporary_directory) / "starter"
            source = starter / "src"
            source.mkdir(parents=True)
            (source / "__init__.py").touch()
            with patch.dict(
                os.environ,
                {
                    "EMBEDDING_PROVIDER": "gemini",
                    "GEMINI_EMBEDDING_MODEL": "gemini-embedding-001",
                    "DATABASE_URL": "postgresql://day6-test",
                },
            ), patch("dotenv.load_dotenv") as load_dotenv, patch(
                "day6_retriever._day6_modules",
                return_value=(embedding_module, store_module),
            ), patch("psycopg.connect", return_value=connection):
                retriever = connect_day6(
                    Path(temporary_directory),
                    strategy="B_paragraph_1000",
                )

        load_dotenv.assert_called_once_with(starter / ".env", override=False)
        embedding_module.GeminiEmbedder.assert_called_once_with(
            "gemini-embedding-001"
        )
        cursor.execute.assert_called_once_with(
            "SELECT vector_dims(embedding) FROM ingestion_chunks "
            "WHERE strategy = %s LIMIT 1",
            ("B_paragraph_1000",),
        )
        store_module.PgVectorStore.assert_called_once_with(
            "postgresql://day6-test", 1536
        )
        self.assertIs(retriever.embedder, embedder)
        self.assertIs(retriever.store, store_instance)
        self.assertEqual(retriever.strategy, "B_paragraph_1000")


if __name__ == "__main__":
    unittest.main()
