"""Command-line interface: python -m src.cli --help"""

import sys
from collections import Counter
from typing import Literal

import typer

from src.ingest import get_collection, ingest_all
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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    app()
