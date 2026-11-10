"""Pydantic schemas of the assistant's structured outputs."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A source used in an answer: one page of one document."""

    doc_id: str = Field(description="Document identifier, exactly as written in the context header.")
    page: int = Field(description="Page number, exactly as written in the context header.")


class Answer(BaseModel):
    """Answer to a question, grounded in the retrieved context."""

    answer: str = Field(description="Answer to the question, or \"I don't know.\" if the context does not contain it.")
    citations: list[Citation] = Field(description="Sources used for the answer. Empty if the question is not answerable.")
    is_answerable: bool = Field(description="True only if the context contains the answer.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the answer, from 0 to 1.")