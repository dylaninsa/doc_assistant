"""Score the raw benchmark results and write a markdown report and a chart."""

import csv
import json
from pathlib import Path

from matplotlib.figure import Figure

from evaluation.dataset import EvalItem, load_dataset
from evaluation.metrics import (
    citation_scores,
    is_correct,
    is_hallucination,
    latency_summary,
    refusal_scores,
    score_facts,
)
from evaluation.run_benchmark import CONFIGS, RESULTS_PATH
from src.schemas import Citation

REPORT_PATH = Path("results/report.md")
CHART_PATH = Path("results/benchmark.png")
MODEL_REFUSALS = {"model", "low_similarity", "invalid_output"}
PRICES = {"api-rag": (0.14, 0.40)}
COLORS = {"local-rag": "#2a78d6", "api-rag": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TABLE_ROWS = [
    ("accuracy", "Exactitude", "{:.0%}"),
    ("accuracy_before_guardrail", "Exactitude sans le garde-fou de citations", "{:.0%}"),
    ("hallucination_rate", "Taux d'hallucination", "{:.0%}"),
    ("refusal_precision", "Précision des refus", "{:.2f}"),
    ("refusal_recall", "Rappel des refus", "{:.2f}"),
    ("refusal_f1", "F1 des refus", "{:.2f}"),
    ("citation_precision", "Précision des citations", "{:.2f}"),
    ("citation_hit_rate", "Réponses citant au moins une bonne page", "{:.0%}"),
    ("extraction_score", "Extraction : part des champs trouvés", "{:.0%}"),
    ("latency_median", "Latence médiane", "{:.1f} s"),
    ("latency_p95", "Latence p95", "{:.1f} s"),
    ("input_tokens", "Tokens en entrée (total)", "{:,}"),
    ("output_tokens", "Tokens en sortie (total)", "{:,}"),
    ("cost_per_1000", "Coût estimé pour 1000 questions", "{:.2f} $"),
]


def load_results(path: Path = RESULTS_PATH) -> list[dict[str, str]]:
    """Read every row of the results CSV file."""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def average(values: list[float] | list[bool]) -> float:
    """Mean of the values, or 0 if there are none (True counts as 1)."""
    return sum(values) / len(values) if values else 0.0


def score_row(row: dict[str, str], item: EvalItem, before_guardrail: bool = False) -> dict[str, float | bool]:
    """Score the answer to one question.

    Args:
        row: One row of the results CSV file.
        item: The matching question of the evaluation set.
        before_guardrail: Score the LLM answer before the citation guardrail instead of the final answer.

    Returns:
        Whether the system refused, was correct or hallucinated, the share of key facts
        found, the citation precision and whether a right page is cited.
    """
    if before_guardrail:
        refused = row["refusal_reason"] in MODEL_REFUSALS
        answer, citations = row["raw_answer"], row["raw_citations"]
    else:
        refused = bool(row["refusal_reason"])
        answer, citations = row["answer"], row["citations"]
    correct = is_correct(item.key_facts, item.answerable, answer, refused)
    cited = [Citation(**citation) for citation in json.loads(citations)]
    precision, hit = citation_scores(cited, item.gold_sources)
    return {
        "refused": refused,
        "correct": correct,
        "hallucination": is_hallucination(refused, correct),
        "facts": score_facts(item.key_facts, answer) if not refused else 0.0,
        "citation_precision": precision,
        "citation_hit": hit,
    }


def summarize(rows: list[dict[str, str]], items: dict[str, EvalItem], before_guardrail: bool = False) -> dict:
    """Compute every metric of one configuration.

    Args:
        rows: Results of one configuration.
        items: Questions of the evaluation set, by id.
        before_guardrail: Score the answers before the citation guardrail.

    Returns:
        Metric name -> value, plus "failed": the ids of the questions not handled correctly.
    """
    scored = [(items[row["id"]], score_row(row, items[row["id"]], before_guardrail)) for row in rows]
    answered = [score for item, score in scored if item.answerable and not score["refused"]]
    refusal = refusal_scores([not item.answerable for item, _ in scored], [score["refused"] for _, score in scored])
    median, p95 = latency_summary([float(row["latency"]) for row in rows])
    return {
        "accuracy": average([score["correct"] for _, score in scored]),
        "hallucination_rate": average([score["hallucination"] for _, score in scored]),
        "refusal_precision": refusal["precision"],
        "refusal_recall": refusal["recall"],
        "refusal_f1": refusal["f1"],
        "citation_precision": average([score["citation_precision"] for score in answered]),
        "citation_hit_rate": average([score["citation_hit"] for score in answered]),
        "extraction_score": average([score["facts"] for item, score in scored if item.type == "extraction"]),
        "latency_median": median,
        "latency_p95": p95,
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "failed": [item.id for item, score in scored if not score["correct"]],
    }


def estimate_cost(config: str, input_tokens: int, output_tokens: int, questions: int) -> float | None:
    """Estimate the cost of 1000 questions at the public paid price.

    Args:
        config: Configuration name.
        input_tokens: Input tokens used for all the questions.
        output_tokens: Output tokens used for all the questions.
        questions: Number of questions.

    Returns:
        The cost in USD, or None for a local model (no price per token).
    """
    if config not in PRICES:
        return None
    input_price, output_price = PRICES[config]
    total = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return total / questions * 1000


def summarize_config(config: str, rows: list[dict[str, str]], items: dict[str, EvalItem]) -> dict:
    """Metrics of one configuration, with the accuracy before the guardrail and the cost."""
    summary = summarize(rows, items)
    before = summarize(rows, items, before_guardrail=True)
    summary["accuracy_before_guardrail"] = before["accuracy"]
    summary["failed_before_guardrail"] = before["failed"]
    summary["cost_per_1000"] = estimate_cost(config, summary["input_tokens"], summary["output_tokens"], len(rows))
    return summary


def to_markdown(summaries: dict[str, dict]) -> str:
    """Build the markdown table of the results, one column per configuration."""
    configs = list(summaries)
    lines = ["| Métrique | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for key, label, fmt in TABLE_ROWS:
        cells = ["—" if summaries[config][key] is None else fmt.format(summaries[config][key]) for config in configs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_summary(summaries: dict[str, dict], path: Path = CHART_PATH) -> None:
    """Draw accuracy, hallucination rate and median latency side by side, one bar per configuration."""
    panels = [
        ("accuracy", "Exactitude", "{:.0%}", 1.0),
        ("hallucination_rate", "Taux d'hallucination", "{:.0%}", 1.0),
        ("latency_median", "Latence médiane", "{:.1f} s", None),
    ]
    configs = list(summaries)
    figure = Figure(figsize=(10, 3.4), facecolor=SURFACE)
    axes = figure.subplots(1, len(panels))
    for ax, (key, title, fmt, top) in zip(axes, panels):
        values = [summaries[config][key] for config in configs]
        bars = ax.bar(configs, values, width=0.5, color=[COLORS[config] for config in configs])
        ax.bar_label(bars, labels=[fmt.format(value) for value in values], padding=3, color=INK, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_facecolor(SURFACE)
        ax.set_yticks([])
        ax.set_ylim(0, (top or max(values)) * 1.15)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK, length=0)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def build_report(results_path: Path = RESULTS_PATH, report_path: Path = REPORT_PATH, chart_path: Path = CHART_PATH) -> str:
    """Score the results file and write the markdown report and the chart.

    Args:
        results_path: Results CSV file written by the benchmark.
        report_path: Markdown file to write.
        chart_path: PNG file to write.

    Returns:
        The markdown table.
    """
    items = {item.id: item for item in load_dataset()}
    rows = load_results(results_path)
    summaries = {}
    for config in CONFIGS:
        config_rows = [row for row in rows if row["config"] == config]
        if config_rows:
            summaries[config] = summarize_config(config, config_rows, items)
    table = to_markdown(summaries)
    failures = [
        f"- **{config}** : {', '.join(summary['failed']) or 'aucune'} "
        f"(sans le garde-fou : {', '.join(summary['failed_before_guardrail']) or 'aucune'})"
        for config, summary in summaries.items()
    ]
    report = "\n".join(
        [
            "# Résultats du benchmark",
            "",
            f"{len(items)} questions, température 0, mêmes embeddings et même top-k pour toutes les configurations.",
            "",
            table,
            "",
            f"Coût : tarif payant public de Gemma 4 31B ({PRICES['api-rag'][0]} $ / {PRICES['api-rag'][1]} $ par million de "
            "tokens en entrée / sortie) ; le local tourne sur une RTX 2060 6 Go, sans coût par token.",
            "",
            "## Questions ratées",
            "",
            *failures,
            "",
            f"![Résultats]({chart_path.name})",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    plot_summary(summaries, chart_path)
    return table