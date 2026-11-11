"""Run the evaluation set on each configuration and append the raw results to a CSV file."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from langchain_core.exceptions import ModelAPIError, ModelConnectionError, ModelRateLimitError, ModelTimeoutError

from evaluation.dataset import EvalItem, load_dataset
from src.llm import Provider, generate
from src.rag import RAGResult, answer_question

RESULTS_PATH = Path("results/benchmark.csv")
CONFIGS: dict[str, Provider] = {"local-rag": "local", "api-rag": "api"}
RETRY_DELAYS = [60, 180, 600]
API_PAUSE = 5
TRANSIENT_ERRORS = (ModelAPIError, ModelConnectionError, ModelRateLimitError, ModelTimeoutError)
FIELDS = [
    "config", "id", "answer", "refusal_reason", "citations", "raw_answer", "raw_citations",
    "retrieved", "latency", "input_tokens", "output_tokens", "timestamp",
]


def load_done(path: Path) -> set[tuple[str, str]]:
    """Read which (config, question id) pairs already have a result.

    Args:
        path: Results CSV file.

    Returns:
        The pairs already in the file; empty if the file does not exist yet.
    """
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as file:
        return {(row["config"], row["id"]) for row in csv.DictReader(file)}


def ask_with_retry(question: str, provider: Provider) -> RAGResult:
    """Answer a question, waiting and trying again when the model is temporarily unavailable.

    Args:
        question: Question of the evaluation set.
        provider: "local" or "api".

    Returns:
        The result of the RAG pipeline.

    Raises:
        Exception: The error of the last attempt, if every attempt failed.
    """
    for delay in RETRY_DELAYS:
        try:
            return answer_question(question, provider)
        except TRANSIENT_ERRORS as error:
            print(f"  {type(error).__name__}, new attempt in {delay} s")
            time.sleep(delay)
    return answer_question(question, provider)


def to_row(config: str, item: EvalItem, result: RAGResult) -> dict[str, str | float | int]:
    """Flatten a result into one CSV row; lists are stored as JSON text.

    Args:
        config: Configuration name, e.g. "local-rag".
        item: Question of the evaluation set.
        result: Result of the RAG pipeline.

    Returns:
        A dictionary whose keys are FIELDS.
    """
    raw = result.raw_answer
    return {
        "config": config,
        "id": item.id,
        "answer": result.answer.answer,
        "refusal_reason": result.refusal_reason or "",
        "citations": json.dumps([c.model_dump() for c in result.answer.citations]),
        "raw_answer": raw.answer if raw else "",
        "raw_citations": json.dumps([c.model_dump() for c in raw.citations] if raw else []),
        "retrieved": json.dumps(
            [{"doc_id": r.chunk.doc_id, "page": r.chunk.page, "similarity": round(r.similarity, 3)} for r in result.retrieved]
        ),
        "latency": round(result.latency, 2),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def run_benchmark(configs: list[str], path: Path = RESULTS_PATH) -> None:
    """Run every question not yet in the results file, for each configuration.

    Each result is written as soon as it is obtained, so an interrupted run resumes
    where it stopped. A question that still fails after the retries is skipped and
    will be run again next time.

    Args:
        configs: Configuration names (keys of CONFIGS).
        path: Results CSV file.
    """
    items = load_dataset()
    done = load_done(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for config in configs:
            provider = CONFIGS[config]
            todo = [item for item in items if (config, item.id) not in done]
            print(f"{config}: {len(todo)} questions to run")
            if todo and provider == "local":
                generate(provider, "Hello")
            for number, item in enumerate(todo, start=1):
                try:
                    result = ask_with_retry(item.question, provider)
                except Exception as error:
                    print(f"  [{number}/{len(todo)}] {item.id}: failed ({type(error).__name__} {error}), skipped")
                    continue
                writer.writerow(to_row(config, item, result))
                file.flush()
                print(f"  [{number}/{len(todo)}] {item.id}: {result.refusal_reason or 'answered'} ({result.latency:.1f} s)")
                if provider == "api":
                    time.sleep(API_PAUSE)