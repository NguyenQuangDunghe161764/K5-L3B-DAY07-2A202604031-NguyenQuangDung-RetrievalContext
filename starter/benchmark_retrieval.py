"""Compare TF-IDF, BM25, Day 6 dense, and RRF hybrid retrieval on one split."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

from baseline import TfidfIndex, hit, pack_context
from day6_retriever import connect_day6, load_day6_local_chunks
from retrieval_methods import BM25Index, CachingRetriever, HybridRetriever


def load_indexed_chunks(database_url: str, strategy: str) -> list[dict[str, Any]]:
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, document_id, source, page, text, metadata
                FROM ingestion_chunks
                WHERE strategy = %s
                ORDER BY chunk_id
                """,
                (strategy,),
            )
            rows = cursor.fetchall()
    if not rows:
        raise ValueError(f"No chunks found for Day 6 strategy {strategy!r}.")
    return [
        {
            "id": chunk_id,
            "source_id": document_id,
            "source": source,
            "page": page,
            "text": text,
            "metadata": metadata,
        }
        for chunk_id, document_id, source, page, text, metadata in rows
    ]


def evaluate_retriever(
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces = []
    for question in questions:
        started = time.perf_counter()
        try:
            results = retriever.search(
                question["question"],
                filters=question.get("filter"),
                top_k=top_k,
            )
        except Exception as error:
            traces.append(
                {
                    "query_id": question["id"],
                    "split": question["split"],
                    "method": method,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000, 2
                    ),
                }
            )
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        context, context_ids = pack_context(results, budget=budget)
        marked = [
            {**result, "evidence_hit": hit(result, question["gold"])}
            for result in results
        ]
        top_three = marked[:3]
        context_hit = any(
            result["evidence_hit"] and result["id"] in context_ids
            for result in marked
        )
        traces.append(
            {
                "query_id": question["id"],
                "split": question["split"],
                "question": question["question"],
                "method": method,
                "top_k": top_k,
                "budget_chars": budget,
                "gold": question["gold"],
                "results": marked,
                "context": context,
                "context_ids": context_ids,
                "elapsed_ms": round(elapsed_ms, 2),
                "hit_at_1": bool(marked and marked[0]["evidence_hit"]),
                "evidence_hit_at_3": any(
                    result["evidence_hit"] for result in top_three
                ),
                "context_evidence": context_hit,
            }
        )

    successful = [trace for trace in traces if trace.get("status") != "error"]
    n = len(successful)
    summary = {
        "method": method,
        "questions": len(traces),
        "successful": n,
        "errors": sum(trace.get("status") == "error" for trace in traces),
        "hit_at_1": {
            "count": sum(trace["hit_at_1"] for trace in successful),
            "total": n,
        },
        "evidence_hit_at_3": {
            "count": sum(trace["evidence_hit_at_3"] for trace in successful),
            "total": n,
        },
        "context_evidence_rate": {
            "count": sum(trace["context_evidence"] for trace in successful),
            "total": n,
        },
        "median_ms": round(
            statistics.median(trace["elapsed_ms"] for trace in successful), 2
        )
        if successful
        else None,
    }
    for metric in ("hit_at_1", "evidence_hit_at_3", "context_evidence_rate"):
        counts = summary[metric]
        counts["rate"] = counts["count"] / counts["total"] if counts["total"] else None
    return traces, summary


class RecordedDenseRetriever:
    """Replay Day 6's saved top-3 dense hits when its database is unavailable."""

    def __init__(self, trace_path: Path, strategy: str) -> None:
        traces = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(traces, list):
            raise ValueError("Day 6 traces must be a JSON array.")
        self.traces: dict[str, dict[str, Any]] = {}
        for trace in traces:
            if trace.get("strategy") != strategy:
                continue
            query = trace.get("question")
            if query in self.traces:
                raise ValueError(f"Duplicate Day 6 trace for question: {query}")
            self.traces[query] = trace
        if not self.traces:
            raise ValueError(f"No Day 6 traces found for strategy {strategy!r}.")
        self.strategy = strategy

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or top_k > 3:
            raise ValueError("Recorded Day 6 traces support top_k from 1 to 3.")
        trace = self.traces.get(query)
        if trace is None:
            raise ValueError(f"No saved Day 6 dense results for query: {query}")
        if trace.get("filter") != filters:
            raise ValueError(
                "Day 6 saved dense trace filter differs from the requested filter."
            )
        return [
            {
                "id": hit_row["chunk_id"],
                "text": hit_row["text"],
                "source_id": hit_row["document_id"],
                "source": hit_row["source"],
                "page": hit_row["page"],
                "metadata": hit_row["metadata"],
                "score": hit_row["score"],
            }
            for hit_row in trace.get("hits", [])[:top_k]
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare retrieval methods on one split."
    )
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--day6-project", required=True, type=Path)
    parser.add_argument("--strategy", default="A_fixed_500_50")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument(
        "--offline-day6-traces",
        type=Path,
        help="Replay saved Day 6 dense top-3 results; supports matching queries only.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.top_k < 3:
        raise ValueError("top-k must be at least 3 to report Evidence Hit@3.")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("questions file must be a JSON array.")
    selected = [
        question
        for question in questions
        if question.get("split") == args.split
    ]
    if not selected:
        raise ValueError(f"No answerable questions found for {args.split}.")

    if args.offline_day6_traces:
        if args.top_k != 3:
            raise ValueError(
                "Offline Day 6 traces provide a top-3 ranking; set --top-k 3."
            )
        chunks = load_day6_local_chunks(
            args.day6_project, strategy=args.strategy
        )
        dense = RecordedDenseRetriever(
            args.offline_day6_traces, args.strategy
        )
        mode = "replayed_day6_dense_traces"
    else:
        day6_starter = (
            args.day6_project
            if (args.day6_project / "src").is_dir()
            else args.day6_project / "starter"
        )
        load_dotenv(day6_starter / ".env", override=False)
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError(f"Set DATABASE_URL in {day6_starter / '.env'}.")
        chunks = load_indexed_chunks(database_url, args.strategy)
        dense = CachingRetriever(
            connect_day6(args.day6_project, strategy=args.strategy)
        )
        mode = "live_day6_dense_retrieval"
    tfidf = TfidfIndex(chunks)
    bm25 = BM25Index(chunks)

    class TfidfRetriever:
        def search(self, query, *, filters, top_k):
            return tfidf.search(query, top_k=top_k, filters=filters)

    class BM25Retriever:
        def search(self, query, *, filters, top_k):
            return bm25.search(query, top_k=top_k, filters=filters)

    methods: list[tuple[str, Any]] = [
        ("tfidf", TfidfRetriever()),
        ("bm25", BM25Retriever()),
        ("dense_day6", dense),
        ("hybrid_rrf_dense_bm25", HybridRetriever(dense, bm25)),
    ]
    all_traces = []
    summaries = {}
    for name, retriever in methods:
        traces, summary = evaluate_retriever(
            retriever,
            selected,
            top_k=args.top_k,
            budget=args.budget,
            method=name,
        )
        all_traces.extend(traces)
        summaries[name] = summary

    if mode == "replayed_day6_dense_traces":
        for trace in all_traces:
            if trace["method"] in ("dense_day6", "hybrid_rrf_dense_bm25"):
                trace["elapsed_ms"] = None
        summaries["dense_day6"]["median_ms"] = None
        summaries["hybrid_rrf_dense_bm25"]["median_ms"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "split": args.split,
                "strategy": args.strategy,
                "corpus_chunks": len(chunks),
                "top_k": args.top_k,
                "context_budget_chars": args.budget,
                "dense_mode": mode,
                "dense_trace_source": (
                    str(args.offline_day6_traces)
                    if args.offline_day6_traces
                    else None
                ),
                "timing_note": (
                    "Dense/hybrid elapsed time is not measured in trace-replay mode; "
                    "lexical timings are local in-process timings."
                    if mode == "replayed_day6_dense_traces"
                    else "Live elapsed times include database/API query latency."
                ),
                "methods": summaries,
                "traces": all_traces,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved traces and summary to {args.output}")


if __name__ == "__main__":
    main()
