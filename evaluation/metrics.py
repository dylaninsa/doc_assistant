"""Benchmark metrics. Every function is pure: no LLM call, testable on its own."""

import math
import re
import statistics
from collections.abc import Sequence

from src.schemas import Citation

RELATIVE_TOLERANCE = 0.01

THOUSANDS_SEPARATOR = re.compile(r"(?<=\d)[,\u202f](?=\d{3}(?!\d))")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lowercase the text and unify the characters that vary between writers.

    Args:
        text: Answer or key fact.

    Returns:
        The text in lowercase, with thousands separators removed ("3,082" -> "3082"),
        the Greek mu replaced by the micro sign and the Unicode minus by "-".
    """
    text = text.lower().replace("\u03bc", "\u00b5").replace("\u2212", "-")
    return THOUSANDS_SEPARATOR.sub("", text)


def extract_numbers(text: str) -> list[float]:
    """Return every number written in an already normalized text."""
    return [float(number) for number in NUMBER.findall(text)]


def fact_matches(fact: str, answer: str) -> bool:
    """Check if one key fact appears in the answer.

    A fact can list alternatives separated by "|" ("15544|51000": metres or feet).
    A numeric alternative matches any number of the answer within RELATIVE_TOLERANCE;
    any other alternative must appear as a substring.

    Args:
        fact: Key fact, as written in the dataset.
        answer: Normalized answer.

    Returns:
        True if at least one alternative is found in the answer.
    """
    numbers = extract_numbers(answer)
    for alternative in normalize(fact).split("|"):
        if NUMBER.fullmatch(alternative):
            expected = float(alternative)
            if any(math.isclose(n, expected, rel_tol=RELATIVE_TOLERANCE) for n in numbers):
                return True
        elif alternative in answer:
            return True
    return False


def score_facts(key_facts: list[str], answer: str) -> float:
    """Share of the key facts found in the answer.

    Args:
        key_facts: Facts that a correct answer must contain.
        answer: Raw answer of the system.

    Returns:
        A score from 0 to 1 (1 = every fact found).
    """
    if not key_facts:
        return 0.0
    normalized = normalize(answer)
    return sum(fact_matches(fact, normalized) for fact in key_facts) / len(key_facts)


def is_correct(key_facts: list[str], answerable: bool, answer: str, refused: bool) -> bool:
    """Decide if the system behaved correctly on one question.

    Args:
        key_facts: Facts that a correct answer must contain.
        answerable: Whether the documents contain the answer.
        answer: Final answer of the system.
        refused: Whether the system refused to answer.

    Returns:
        For an answerable question: answered with every key fact. Otherwise: refused.
    """
    if not answerable:
        return refused
    return not refused and score_facts(key_facts, answer) == 1.0


def is_hallucination(refused: bool, correct: bool) -> bool:
    """An answer given when the system should have refused, or a wrong answer given instead of a refusal."""
    return not refused and not correct


def citation_scores(cited: list[Citation], gold: list[Citation]) -> tuple[float, bool]:
    """Compare the cited pages to the expected pages.

    Args:
        cited: Citations of the final answer.
        gold: Pages that contain at least part of the answer.

    Returns:
        The precision (share of cited pages that are expected pages) and whether at
        least one expected page is cited.
    """
    cited_pages = {(c.doc_id, c.page) for c in cited}
    gold_pages = {(c.doc_id, c.page) for c in gold}
    if not cited_pages:
        return 0.0, False
    matching = cited_pages & gold_pages
    return len(matching) / len(cited_pages), bool(matching)


def refusal_scores(should_refuse: Sequence[bool], refused: Sequence[bool]) -> dict[str, float]:
    """Score the refusals as a binary classification ("refuse" is the positive class).

    Args:
        should_refuse: For each question, True if it has no answer in the documents.
        refused: For each question, True if the system refused.

    Returns:
        The precision, recall and F1 of the refusals.
    """
    pairs = list(zip(should_refuse, refused))
    true_positives = sum(expected and actual for expected, actual in pairs)
    false_positives = sum(not expected and actual for expected, actual in pairs)
    false_negatives = sum(expected and not actual for expected, actual in pairs)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def latency_summary(latencies: Sequence[float]) -> tuple[float, float]:
    """Median and 95th percentile of the latencies, in seconds."""
    if len(latencies) < 2:
        value = latencies[0] if latencies else 0.0
        return value, value
    return statistics.median(latencies), statistics.quantiles(latencies, n=20, method="inclusive")[18]