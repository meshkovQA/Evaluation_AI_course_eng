"""
Evaluation of RAG system using DeepEval metrics.
Loads the dataset from Excel, queries the RAG system, and runs evaluation.
"""

import sys
sys.path.insert(
    0, '/Users/aleksandrmeskov/Desktop/AI evaluation/AI_practice/Lecture5/eval_rag')

import time  # noqa: E402
import pandas as pd  # noqa: E402
from typing import List, Union  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from deepeval import evaluate  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from deepeval_evaluator import create_metrics  # noqa: E402


def evaluate_rag_from_excel(
    excel_path: str,
    rag_connector: RAGConnector,
    metrics_list: List[str],
    model: Union[str] = "gpt-4o-mini",
    threshold: float = 0.7,
    sleep_time: float = 0.1
):

    # ============= STEP 1: Parse Excel =============
    print(f" Step 1: Parsing Excel file...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Error loading the dataset")
        return

    parser.preview_dataset(df, n=2)

    # ============= STEP 2: Extract questions and expected answers =============
    print(f"Step 2: Extracting questions from dataset...")

    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f"Extracted {len(questions)} questions")

    # ============= STEP 3: Query the RAG system =============
    print(f"Step 3: Getting answers from RAG system...")

    test_cases = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        rag_response = rag_connector.query(question)

        if 'error' in rag_response:
            print(f"RAG error: {rag_response['error']}")
            continue

        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        print(f"Answer: {actual_output[:60]}...")
        print(f"Contexts: {len(retrieval_context)}")

        expected = expected_responses[i-1] if i - \
            1 < len(expected_responses) else ""

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=retrieval_context
        )

        test_cases.append(test_case)

        time.sleep(sleep_time)

    print(f"\n✅ Created {len(test_cases)} test cases")

    # ============= STEP 4: Create metrics =============
    print(f"\n📊 Step 4: Creating evaluation metrics...")
    print(f"   Metrics: {metrics_list}")
    print(f"   Threshold: {threshold}")

    metrics = create_metrics(
        model=model,
        metrics_list=metrics_list,
        threshold=threshold
    )

    print(f"✅ Created {len(metrics)} metrics")

    # ============= STEP 5: Run evaluation =============
    print(f"\n🧪 Step 5: Running evaluation...")

    try:
        evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        print(f"\n✅ Evaluation complete!")
        print(f"🌐 Results saved at https://app.confident-ai.com")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    # ============= CONFIGURATION =============

    # 1. Initialize RAG connector
    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    # 2. Path to Excel file
    excel_path = "data/evaluation_dataset.xlsx"

    # 3. Select evaluation metrics
    metrics_to_use = [
        'answer_relevancy',
        'faithfulness',
        'contextual_relevancy'
    ]

    # ============= RUN EVALUATION =============

    evaluate_rag_from_excel(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        model="gpt-4o-mini",
        threshold=0.7,
        sleep_time=0.1
    )
