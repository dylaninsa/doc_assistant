"""Simple RAG pipeline: retrieve the top-k chunks, then one structured generation."""

from dataclasses import dataclass
from typing import Literal

from src.llm import Provider, generate_structured
from src.retriever import RetrievedChunk, search
from src.schemas import Answer

TOP_K = 5
MIN_SIMILARITY = 0.60
REFUSAL_TEXT = "I don't know."

PROMPT_TEMPLATE = """You are a technical documentation assistant. Answer the question using ONLY the context below.

Rules:
- Every fact in your answer must come from the context. Never use outside knowledge.
- Cite every source you used with its doc_id and page, exactly as written in the [source: ...] headers.
- If the context does not contain the answer, set is_answerable to false, answer "{refusal}" and leave citations empty.
- Copy numbers and units exactly as they appear in the context.
- Answer in the language of the question.

Context:
{context}

Question: {question}"""


@dataclass
class RAGResult:
    """Answer of the RAG pipeline along with what is needed to evaluate it.

    Attributes:
        answer: Final answer, after the guardrails.
        retrieved: Chunks given to the LLM, closest first.
        refusal_reason: Why the pipeline refused, or None if it answered.
        latency: LLM call duration in seconds (0 if the LLM was not called).
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the response.
    """

    answer: Answer
    retrieved: list[RetrievedChunk]
    refusal_reason: Literal["low_similarity", "invalid_output", "no_valid_citation", "model"] | None
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


def refusal() -> Answer:
    """Build the standard "I don't know" answer."""
    return Answer(answer=REFUSAL_TEXT, citations=[], is_answerable=False, confidence=0.0)


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Write the retrieved chunks as a context block, each with a citable header.

    Args:
        retrieved: Chunks returned by the search.

    Returns:
        The chunks separated by blank lines, each starting with "[source: doc_id, page N]".
    """
    return "\n\n".join(
        f"[source: {r.chunk.doc_id}, page {r.chunk.page}]\n{r.chunk.text}" for r in retrieved
    )


def validate_citations(answer: Answer, retrieved: list[RetrievedChunk]) -> Answer:
    """Remove the citations that do not point to a retrieved chunk.

    Args:
        answer: Answer produced by the LLM.
        retrieved: Chunks actually given to the LLM.

    Returns:
        A copy of the answer keeping only the valid citations, without duplicates.
    """
    allowed = {(r.chunk.doc_id, r.chunk.page) for r in retrieved}
    valid = []
    for citation in answer.citations:
        key = (citation.doc_id, citation.page)
        if key in allowed and key not in {(c.doc_id, c.page) for c in valid}:
            valid.append(citation)
    return answer.model_copy(update={"citations": valid})


def answer_question(question: str, provider: Provider, k: int = TOP_K) -> RAGResult:
    """Answer a question from the indexed documents, or refuse.

    Guardrails: refuse without calling the LLM if no chunk is similar enough; refuse if the
    output does not match the schema; drop invented citations, and refuse if an answer is
    left without any valid citation.

    Args:
        question: Question in natural language.
        provider: "local" or "api".
        k: Number of chunks given to the LLM.

    Returns:
        The answer, the retrieved chunks, the refusal reason and the LLM measurements.
    """
    retrieved = search(question, k=k)
    if not retrieved or retrieved[0].similarity < MIN_SIMILARITY:
        return RAGResult(answer=refusal(), retrieved=retrieved, refusal_reason="low_similarity")

    prompt = PROMPT_TEMPLATE.format(context=format_context(retrieved), question=question, refusal=REFUSAL_TEXT)
    reponse = generate_structured(provider, prompt, Answer)
    measures = {"latency": reponse.latency, "input_tokens": reponse.input_tokens, "output_tokens": reponse.output_tokens}

    if reponse.parsed is None:
        return RAGResult(answer=refusal(), retrieved=retrieved, refusal_reason="invalid_output", **measures)

    answer = validate_citations(reponse.parsed, retrieved)
    if not answer.is_answerable:
        return RAGResult(answer=answer, retrieved=retrieved, refusal_reason="model", **measures)
    if not answer.citations:
        return RAGResult(answer=refusal(), retrieved=retrieved, refusal_reason="no_valid_citation", **measures)
    return RAGResult(answer=answer, retrieved=retrieved, refusal_reason=None, **measures)