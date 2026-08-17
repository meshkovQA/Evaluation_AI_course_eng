# test_batch_evaluation.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio  # noqa: E402
from eval_lib import (  # noqa: E402
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric

)
from config import model as default_model  # noqa: E402


# ==================== TEST 1: Batch Standard Metrics with OpenAI ====================

async def test_batch_standard_metrics():
    """Test batch evaluation with multiple test cases and standard metrics"""

    # Create test cases
    test_cases = [
        EvalTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            retrieval_context=["Paris is the capital of France."]
        ),
        EvalTestCase(
            input="What is photosynthesis?",
            actual_output="The weather today is sunny.",
            expected_output="Process by which plants convert light into energy",
            retrieval_context=[
                "Photosynthesis is the process by which plants use sunlight."]
        )
    ]

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(
            model=default_model,
            threshold=0.7,
            temperature=1.0,
        ),
        FaithfulnessMetric(
            model=default_model,
            threshold=0.8,
            temperature=1.0,
        ),
        ContextualRelevancyMetric(
            model=default_model,
            threshold=1.0,
        ),
    ]

    # Run batch evaluation
    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        verbose=True
    )

    return results


async def test_standard_metrics_with_custom_model():
    """Test batch evaluation with multiple test cases and standard metrics"""

    # Create test cases
    test_cases = [
        EvalTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            retrieval_context=["Paris is the capital of France."]
        ),
        EvalTestCase(
            input="What is photosynthesis?",
            actual_output="The weather today is sunny.",
            expected_output="Process by which plants convert light into energy",
            retrieval_context=[
                "Photosynthesis is the process by which plants use sunlight."]
        )
    ]

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(
            model=default_model,
            threshold=0.7,
            temperature=0.1,
        ),
        FaithfulnessMetric(
            model=default_model,
            threshold=0.8,
            temperature=1.0,
        ),
        ContextualRelevancyMetric(
            model=default_model,
            threshold=1.0,
        ),
    ]

    # Run batch evaluation
    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        verbose=True
    )

    return results


# ==================== Run All Tests ====================

async def run_all_tests():
    """Execute all batch evaluation tests"""

    # await test_batch_standard_metrics()
    await test_standard_metrics_with_custom_model()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
