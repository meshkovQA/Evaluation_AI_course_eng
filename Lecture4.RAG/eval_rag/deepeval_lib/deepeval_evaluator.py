from typing import List, Union, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval import evaluate


def create_metrics(
    model: Union[str],
    metrics_list: List[str] = None,
    threshold: float = 0.7
):
    if metrics_list is None:
        metrics_list = ['answer_relevancy', 'faithfulness']

    METRICS = {
        'answer_relevancy': AnswerRelevancyMetric,
        'faithfulness': FaithfulnessMetric,
        'contextual_relevancy': ContextualRelevancyMetric,
        'contextual_recall': ContextualRecallMetric,
        'contextual_precision': ContextualPrecisionMetric
    }

    metrics = []
    for metric_name in metrics_list:
        if metric_name in METRICS:
            metric = METRICS[metric_name](
                threshold=threshold,
                model=model,
                include_reason=True,
                async_mode=False
            )
            metrics.append(metric)

    return metrics


if __name__ == "__main__":

    # ============= EXAMPLE 1: Standard OpenAI =============
    print("📊 Example 1: Standard OpenAI model")

    metrics = create_metrics(
        model="gpt-4o-mini",
        metrics_list=['answer_relevancy', 'faithfulness'],
        threshold=0.7
    )

    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="You are eligible for a 30 day full refund at no extra cost.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra cost."]
    )

    evaluate(test_cases=[test_case], metrics=metrics)

    # ============= EXAMPLE 2: Direct DeepEval usage =============
    print("\n" + "="*60)
    print("📊 Example 3: Direct DeepEval usage (no wrapper)")

    metric = AnswerRelevancyMetric(
        model="gpt-4o-mini",
        include_reason=True,
        async_mode=False
    )

    metric.measure(test_case)

    print(f"\nScore: {metric.score:.3f}")
    print(f"Reason: {metric.reason}")
