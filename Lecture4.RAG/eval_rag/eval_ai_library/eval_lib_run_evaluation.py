"""
End-to-End RAG Evaluation Pipeline with eval_lib.
Loads the dataset from Excel, queries the RAG system, and runs evaluation.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time  # noqa: E402
import pandas as pd  # noqa: E402
from typing import List  # noqa: E402
from eval_lib import (  # noqa: E402
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from dataset_parser import DatasetParser  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402


async def evaluate_rag_from_excel(
    excel_path: str,
    rag_connector: RAGConnector,
    metrics_list: List[str],
    model: str = "gpt-4o-mini",
    threshold: float = 0.7,
    temperature: float = 0.5,
    sleep_time: float = 0.1,
    verbose: bool = True,
    show_dashboard: bool = False,
    session_name: str = "Evaluation Session"
):

    # ============= STEP 1: Parse Excel =============
    print(f"\n Step 1: Parsing Excel file...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Error loading the dataset")
        return

    info = parser.validate_dataset(df)
    print(f"\n Dataset info:")
    print(f"   Total rows: {info['total_rows']}")
    print(f"   Valid pairs: {info['valid_pairs']}")

    parser.preview_dataset(df, n=2)

    # ============= STEP 2: Extract questions and expected answers =============
    print(f"\n Step 2: Extracting questions from dataset...")

    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f" Extracted {len(questions)} questions")

    # ============= STEP 3: Query the RAG system =============
    print(f"\n Step 3: Getting answers from RAG system...")

    test_cases = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        rag_response = rag_connector.query(question)

        if 'error' in rag_response:
            print(f"      ⚠️  RAG error: {rag_response['error']}")
            continue

        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        print(f"      ✅ Answer: {actual_output[:60]}...")
        print(f"      📚 Contexts: {len(retrieval_context)}")

        expected = expected_responses[i-1] if i - \
            1 < len(expected_responses) else ""

        test_case = EvalTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=retrieval_context
        )

        test_cases.append(test_case)

        time.sleep(sleep_time)

    print(f"\n✅ Created {len(test_cases)} test cases")

    # ============= STEP 4: Create metrics =============
    print(f"\n Step 4: Creating evaluation metrics...")
    print(f"   Metrics: {metrics_list}")
    print(f"   Threshold: {threshold}")
    print(f"   Temperature: {temperature}")

    print(f"   Model: {model}")

    metrics = []
    metric_classes = {
        'answer_relevancy': AnswerRelevancyMetric,
        'faithfulness': FaithfulnessMetric,
        'contextual_relevancy': ContextualRelevancyMetric
    }

    for metric_name in metrics_list:
        if metric_name in metric_classes:
            metric = metric_classes[metric_name](
                model=model,
                threshold=threshold,
                temperature=temperature,
                verbose=verbose
            )
            metrics.append(metric)
            print(f"   ✅ Added metric: {metric_name}")

    print(f"\n✅ Created {len(metrics)} metrics")

    # ============= STEP 5: Run evaluation =============
    print(f"\n🧪 Step 5: Running evaluation...")
    print(f"   Test cases: {len(test_cases)}")
    print(f"   Metrics: {len(metrics)}")

    try:
        results = await evaluate(
            test_cases=test_cases,
            metrics=metrics,
            verbose=verbose,
            show_dashboard=show_dashboard,
            session_name=session_name
        )

        return results

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== TEST SCENARIOS ====================

async def scenario_1_standard_openai():
    """Scenario 1: Standard OpenAI model"""

    print("\n" + "="*70)
    print("📋 SCENARIO 1: Evaluation with standard OpenAI model")
    print("="*70)

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness',
        'contextual_relevancy'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model="gpt-4o-mini",  # Standard OpenAI
        threshold=0.7,
        temperature=0.5,
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="OpenAI Model Evaluation"

    )

    return results


async def scenario_2_custom_model():
    """Scenario 2: Same direct OpenAI model with different prompt phrasing"""

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness',
        'contextual_relevancy'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model="gpt-4o-mini",
        threshold=0.7,
        temperature=0.5,
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="OpenAI Model Evaluation"
    )

    return results


async def scenario_3_strict_evaluation():
    """Scenario 3: Strict evaluation (low temperature)"""

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model="gpt-4o-mini",
        threshold=0.8,  # High threshold
        temperature=0.1,  # STRICT: all verdicts matter
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="Strict Evaluation"
    )

    return results


async def scenario_4_lenient_evaluation():
    """Scenario 4: Lenient evaluation (high temperature)"""

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    excel_path = "data/evaluation_dataset.xlsx"

    metrics_to_use = [
        'answer_relevancy',
        'faithfulness'
    ]

    results = await evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model="gpt-4o-mini",
        threshold=0.6,  # Low threshold
        temperature=1.0,  # LENIENT: focus on positive verdicts
        sleep_time=0.1,
        verbose=True,
        show_dashboard=True,
        session_name="Lenient Evaluation"
    )

    return results


if __name__ == "__main__":

    asyncio.run(scenario_3_strict_evaluation())
