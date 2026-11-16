"""Run the real controller and Day 6 dense retriever on one benchmark split."""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentic import run_agent
from chat_controller import OpenAIController, OpenRouterController
from day6_retriever import connect_day6, load_day6_local_chunks
from retrieval_methods import BM25Index, HybridRetriever


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = BM25Index(chunks)

    def search(self, query, *, filters, top_k):
        return self.index.search(query, filters=filters, top_k=top_k)


def build_retriever(day6_project: Path, strategy: str, method: str) -> Any:
    if method == "dense":
        return connect_day6(day6_project, strategy=strategy)
    if method == "bm25":
        return BM25Retriever(
            load_day6_local_chunks(day6_project, strategy=strategy)
        )
    if method == "hybrid":
        chunks = load_day6_local_chunks(day6_project, strategy=strategy)
        return HybridRetriever(
            connect_day6(day6_project, strategy=strategy),
            BM25Index(chunks),
        )
    raise ValueError(f"Unknown retriever method: {method}")


def run_cases(
    cases: list[dict[str, Any]],
    split: str,
    controller: Any,
    retriever: Any,
    *,
    top_k: int = 3,
    budget: int = 800,
    max_searches: int = 4,
) -> list[dict[str, Any]]:
    traces = []
    for case in cases:
        if case.get("split") != split:
            continue
        query_id = case["id"]
        started = time.perf_counter()
        try:
            result = run_agent(
                case["question"],
                controller,
                retriever,
                filters=case.get("filter"),
                top_k=top_k,
                budget=budget,
                max_searches=max_searches,
            )
            trace = {
                "query_id": query_id,
                **result,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as error:
            trace = {
                "query_id": query_id,
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "error": f"{type(error).__name__}: {error}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        traces.append(trace)
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agent on one split; this calls paid external APIs."
    )
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--day6-project", required=True, type=Path)
    parser.add_argument("--strategy", default="A_fixed_500_50")
    parser.add_argument(
        "--retriever", choices=("dense", "bm25", "hybrid"), default="dense"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument(
        "--provider", choices=("openai", "openrouter"), default="openai"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit selected cases for a bounded smoke run.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    day6_starter = (
        args.day6_project
        if (args.day6_project / "src").is_dir()
        else args.day6_project / "starter"
    )
    load_dotenv(day6_starter / ".env", override=False)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be greater than zero.")
    if args.provider == "openai":
        controller = OpenAIController(args.model)
    else:
        if args.model:
            os.environ["AGENT_MODEL"] = args.model
        controller = OpenRouterController()
    retriever = build_retriever(
        args.day6_project, args.strategy, args.retriever
    )

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("cases file must be a JSON array.")
    selected = [case for case in cases if case.get("split") == args.split]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError(f"No cases found for split {args.split}.")
    traces = run_cases(
        selected,
        args.split,
        controller,
        retriever,
        top_k=args.top_k,
        budget=args.budget,
        max_searches=args.max_searches,
    )
    retriever_name = f"{args.retriever} ({args.strategy})"
    for trace in traces:
        trace["model"] = controller.model_name
        trace["provider"] = args.provider
        trace["retriever"] = retriever_name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(traces, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(traces)} traces to {args.output}")


if __name__ == "__main__":
    main()
