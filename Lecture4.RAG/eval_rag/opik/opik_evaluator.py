# opik_evaluator.py

from typing import List, Dict
import opik
from opik.evaluation.metrics import (
    AnswerRelevance,
    Hallucination,
    ContextPrecision,
    ContextRecall
)
from opik.evaluation import evaluate
from opik import Opik


def create_metrics(
    model: str = "gpt-4o-mini",
    metrics_list: List[str] = None,
    threshold: float = 0.7
):
    """
    Create Opik metrics

    Args:
        model: Model name (e.g. "gpt-4o-mini")
        metrics_list: ['answer_relevance', 'hallucination', 'context_precision', 'context_recall']
        threshold: Metric threshold

    Returns:
        List of Opik metrics
    """
    if metrics_list is None:
        metrics_list = ['answer_relevance', 'hallucination']

    METRICS = {
        'answer_relevance': AnswerRelevance,
        'hallucination': Hallucination,
        'context_precision': ContextPrecision,
        'context_recall': ContextRecall
    }

    metrics = []
    for metric_name in metrics_list:
        if metric_name in METRICS:
            kwargs = {"name": metric_name}
            if model is not None:
                kwargs["model"] = model
            metric = METRICS[metric_name](**kwargs)
            metrics.append(metric)

    return metrics


def simple_rag_task(item: Dict) -> Dict:
    """
    Simple task function for experiment.
    In production, this would call your RAG system.
    """
    return {
        "input": item["input"],
        "output": item["output"],
        "context": item.get("context", []),
        "expected_output": item.get("expected_output", "")
    }
