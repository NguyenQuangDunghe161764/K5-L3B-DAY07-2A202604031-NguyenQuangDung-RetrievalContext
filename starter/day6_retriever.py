"""Read-only dense retrieval adapter for the Day 6 PostgreSQL/pgvector index."""
import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Protocol


class Embedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class VectorStore(Protocol):
    dimensions: int

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]: ...


def _package_name(day6_src: Path) -> str:
    return "_day6_src_" + hashlib.sha256(
        str(day6_src.resolve()).encode("utf-8")
    ).hexdigest()[:12]


class Day6Retriever:
    """Adapt Day 6 embeddings and pgvector search to the Day 7 Retriever protocol."""

    def __init__(self, embedder: Embedder, store: VectorStore, strategy: str) -> None:
        if not strategy.strip():
            raise ValueError("strategy must not be empty.")
        self.embedder = embedder
        self.store = store
        self.strategy = strategy

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary or None.")

        query_vector = self.embedder.embed_query(query)
        if len(query_vector) != self.store.dimensions:
            raise ValueError(
                f"Query embedding has {len(query_vector)} dimensions, but the "
                f"Day 6 index uses {self.store.dimensions}; use the same model "
                "and dimensions that created the index."
            )
        hits = self.store.search(
            strategy=self.strategy,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )
        return [
            {
                "id": hit["chunk_id"],
                "text": hit["text"],
                "source_id": hit["document_id"],
                "source": hit["source"],
                "page": hit["page"],
                "metadata": hit["metadata"],
                "score": hit["score"],
            }
            for hit in hits
        ]


def _day6_modules(day6_src: Path) -> tuple[Any, Any]:
    package_name = _package_name(day6_src)
    if package_name not in sys.modules:
        init_file = day6_src / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_file,
            submodule_search_locations=[str(day6_src)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Day 6 source package from {day6_src}.")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            del sys.modules[package_name]
            raise

    embeddings = importlib.import_module(f"{package_name}.embeddings")
    store = importlib.import_module(f"{package_name}.store")
    return embeddings, store


def load_day6_local_chunks(
    day6_project: Path,
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    """Rebuild exact Day 6 chunk IDs from its local corpus without API/database use."""
    project = Path(day6_project).expanduser()
    day6_starter = project if (project / "src").is_dir() else project / "starter"
    day6_src = day6_starter / "src"
    if not (day6_src / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Could not find Day 6 starter source package under {day6_starter}."
        )
    if strategy not in ("A_fixed_500_50", "B_paragraph_1000", "C_paragraph_filter"):
        raise ValueError(f"Unsupported Day 6 chunk strategy: {strategy!r}.")

    _day6_modules(day6_src)
    package_name = _package_name(day6_src)
    run_module = importlib.import_module(f"{package_name}.run")
    chunking = importlib.import_module(f"{package_name}.chunking")
    pages, _ = run_module._load_corpus(day6_starter / "data")

    chunks = []
    for page in pages:
        if strategy == "A_fixed_500_50":
            page_chunks = chunking.fixed_size(page, size=500, overlap=50)
        else:
            page_chunks = chunking.by_paragraph(page, max_size=1000)
        chunks.extend(
            {
                "id": chunk.chunk_id,
                "source_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in page_chunks
        )
    if not chunks:
        raise ValueError(f"No chunks could be rebuilt for Day 6 strategy {strategy!r}.")
    return chunks


def connect_day6(
    day6_project: Path,
    *,
    strategy: str = "A_fixed_500_50",
) -> Day6Retriever:
    """Load Day 6 config and connect to its existing, read-only vector index."""
    project = Path(day6_project).expanduser()
    day6_starter = project if (project / "src").is_dir() else project / "starter"
    day6_src = day6_starter / "src"
    if not (day6_src / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Could not find Day 6 starter source package under {day6_starter}."
        )
    if not strategy.strip():
        raise ValueError("strategy must not be empty.")

    from dotenv import load_dotenv

    load_dotenv(day6_starter / ".env", override=False)
    embeddings, store_module = _day6_modules(day6_src)

    provider = os.getenv("EMBEDDING_PROVIDER", "openrouter").strip().lower()
    if provider == "gemini":
        model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()
        embedder = embeddings.GeminiEmbedder(model_name or "gemini-embedding-001")
    elif provider == "openrouter":
        model_name = os.getenv("EMBEDDING_MODEL", "").strip()
        embedder = embeddings.MyEmbedder(
            model_name or "openai/text-embedding-3-small"
        )
    else:
        raise ValueError("EMBEDDING_PROVIDER must be 'openrouter' or 'gemini'.")

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError(f"Set DATABASE_URL in {day6_starter / '.env'}.")

    import psycopg

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT vector_dims(embedding) FROM ingestion_chunks "
                "WHERE strategy = %s LIMIT 1",
                (strategy,),
            )
            row = cursor.fetchone()
    if row is None:
        raise ValueError(
            f"No indexed chunks found for Day 6 strategy {strategy!r}."
        )

    vector_store = store_module.PgVectorStore(database_url, int(row[0]))
    return Day6Retriever(embedder, vector_store, strategy)
