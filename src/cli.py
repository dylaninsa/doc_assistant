"""Command-line interface: python -m src.cli --help"""

import sys
from collections import Counter
from typing import Literal

import typer

from evaluation.dataset import check_dataset, load_dataset

from src.ingest import get_collection, ingest_all
from src.rag import answer_question
from src.retriever import search as search_chunks

app = typer.Typer(help="Technical documents assistant.")
PREVIEW_LENGTH = 200


@app.command()
def ingest(pdf_dir: str = "data/pdfs") -> None:
    """Extract, chunk and index all PDFs into ChromaDB (rebuilds the index)."""
    chunks = ingest_all(pdf_dir)
    counts = Counter((c.doc_id, c.type) for c in chunks)
    for doc_id in sorted({c.doc_id for c in chunks}):
        print(f"{doc_id:<15} {counts[doc_id, 'text']:>4} text  {counts[doc_id, 'table']:>4} tables")
    print(f"Total: {len(chunks)} chunks indexed")


@app.command()
def inspect(
    doc: str | None = typer.Option(None, help="Filter on a document, e.g. bme280."),
    chunk_type: Literal["text", "table"] | None = typer.Option(None, "--type", help="Filter on text or table."),
    n: int = typer.Option(5, help="Number of chunks to show."),
) -> None:
    """Show indexed chunks with their metadata, read back from ChromaDB."""
    filters = [{"doc_id": doc}] if doc else []
    if chunk_type:
        filters.append({"type": chunk_type})
    where = filters[0] if len(filters) == 1 else {"$and": filters} if filters else None

    results = get_collection().get(where=where, limit=n)
    for chunk_id, text, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
        print(f"=== {chunk_id} | p.{metadata['page']} | {metadata['type']} | {metadata['section']}")
        print(text)
        print()


@app.command()
def search(
    question: str,
    n: int = typer.Option(5, help="Number of chunks to return."),
    doc: str | None = typer.Option(None, help="Restrict to a document, e.g. bme280."),
) -> None:
    """Show the chunks closest in meaning to a question."""
    for result in search_chunks(question, k=n, doc_id=doc):
        chunk = result.chunk
        preview = chunk.text[:PREVIEW_LENGTH].replace("\n", " ")
        print(f"{result.similarity:.3f} | {chunk.id} | p.{chunk.page} | {chunk.type} | {chunk.section}")
        print(f"      {preview}")


@app.command()
def ask(
    question: str,
    config: str = typer.Option("local-rag", help="local-rag or api-rag."),
) -> None:
    """Answer a question from the documents, with citations."""
    provider, _, mode = config.partition("-")
    if provider not in ("local", "api") or mode != "rag":
        raise typer.BadParameter("Available configs: local-rag, api-rag.")

    result = answer_question(question, provider)
    answer = result.answer
    print(answer.answer)
    for citation in answer.citations:
        print(f"  [source: {citation.doc_id}, page {citation.page}]")
    print(
        f"(refusal: {result.refusal_reason} | confidence: {answer.confidence:.2f} | "
        f"latency: {result.latency:.1f} s | tokens: {result.input_tokens} in / {result.output_tokens} out)"
    )


@app.command("check-eval")
def check_eval() -> None:
    """Load the evaluation set and report annotation problems."""
    items = load_dataset()
    counts = Counter(item.type for item in items)
    print(f"{len(items)} questions: " + ", ".join(f"{count} {kind}" for kind, count in counts.items()))
    problems = check_dataset(items)
    for problem in problems:
        print(f"  - {problem}")
    print("Dataset OK" if not problems else f"{len(problems)} problem(s) found")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app()
