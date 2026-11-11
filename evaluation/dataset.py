"""Hand-annotated evaluation set: loading and consistency checks."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.ingest import get_collection
from src.schemas import Citation

DATASET_PATH = Path("evaluation/dataset.jsonl")

QuestionType = Literal["factual", "table", "calculation", "extraction", "unanswerable"]


class EvalItem(BaseModel):
    """One annotated question of the evaluation set."""

    id: str
    question: str
    expected_answer: str
    key_facts: list[str]
    gold_sources: list[Citation]
    type: QuestionType
    answerable: bool
    expected_fields: dict[str, str] | None = None


def load_dataset(path: Path = DATASET_PATH) -> list[EvalItem]:
    """Read the evaluation set, one JSON object per line.

    Args:
        path: Path of the JSONL file.

    Returns:
        The validated questions, in file order.
    """
    with path.open(encoding="utf-8") as file:
        return [EvalItem.model_validate_json(line) for line in file if line.strip()]


def check_dataset(items: list[EvalItem]) -> list[str]:
    """Look for annotation mistakes that Pydantic cannot catch alone.

    Args:
        items: Questions returned by load_dataset.

    Returns:
        One message per problem found; empty if the set is consistent.
    """
    problems = []
    seen_ids = set()
    collection = get_collection()
    for item in items:
        if item.id in seen_ids:
            problems.append(f"{item.id} is duplicated")
        seen_ids.add(item.id)

        if item.answerable != bool(item.gold_sources):
            problems.append(f"{item.id}: answerable={item.answerable} but {len(item.gold_sources)} gold sources")
        if item.answerable != bool(item.key_facts):
            problems.append(f"{item.id}: answerable={item.answerable} but {len(item.key_facts)} key facts")
        if item.answerable == (item.type == "unanswerable"):
            problems.append(f"{item.id}: type {item.type} does not match answerable={item.answerable}")
        if (item.type == "extraction") != (item.expected_fields is not None):
            problems.append(f"{item.id}: expected_fields must be set for extraction questions only")

        for source in item.gold_sources:
            where = {"$and": [{"doc_id": source.doc_id}, {"page": source.page}]}
            if not collection.get(where=where, limit=1)["ids"]:
                problems.append(f"{item.id}: no indexed chunk for {source.doc_id} page {source.page}")
    return problems