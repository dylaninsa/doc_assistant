"""Semantic search over the indexed chunks."""

from dataclasses import dataclass

from src.ingest import QUERY_PREFIX, Chunk, embeddings, get_collection


@dataclass
class RetrievedChunk:
    """A chunk returned by a search, with its similarity to the question.

    Attributes:
        chunk: The retrieved chunk.
        similarity: Cosine similarity between the question and the chunk, from -1 to 1
            (higher is closer).
    """

    chunk: Chunk
    similarity: float


def search(question: str, k: int = 5, doc_id: str | None = None) -> list[RetrievedChunk]:
    """Return the k chunks closest in meaning to a question.

    Args:
        question: Question in natural language.
        k: Number of chunks to return.
        doc_id: Restrict the search to one document (e.g. "bme280").

    Returns:
        The retrieved chunks, closest first.
    """
    query_vector = embeddings.embed_query(QUERY_PREFIX + question)
    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=k,
        where={"doc_id": doc_id} if doc_id else None,
    )
    return [
        RetrievedChunk(
            chunk=Chunk(id=chunk_id, text=text, **metadata),
            similarity=1 - distance,
        )
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
