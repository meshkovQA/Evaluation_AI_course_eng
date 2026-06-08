"""
Evaluation of RAG system using RAGAS metrics.
Loads the dataset from Excel, queries the RAG system, and runs evaluation.
"""

import sys
sys.path.insert(
    0, '/Users/aleksandrmeskov/Desktop/AI evaluation/AI_practice/Lecture5/eval_rag')

import time  # noqa: E402
import pandas as pd  # noqa: E402
from typing import List, Dict, Any  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from ragas_evaluator import RagasEvaluator  # noqa: E402


def evaluate_rag_with_ragas(
    excel_path: str,
    rag_connector: RAGConnector,
    metrics_config: Dict[str, Dict[str, Any]],
    model: str = "gpt-4o-mini",
    sleep_time: float = 0.1
):

    # ============= STEP 1: Parse Excel =============
    print(f"\n Step 1: Parsing Excel file...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Error loading the dataset")
        return

    parser.preview_dataset(df, n=2)

    # ============= STEP 2: Extract questions and expected answers =============
    print(f"\n Step 2: Extracting questions from dataset...")

    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f"Extracted {len(questions)} questions")

    # ============= STEP 3: Query the RAG system =============
    print(f"\n Step 3: Getting answers from RAG system...")

    test_cases = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        rag_response = rag_connector.query(question)

        if 'error' in rag_response:
            print(f"      ⚠️  RAG error: {rag_response['error']}")
            continue

        answer = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        contexts = [s.get('content', '') for s in sources if s.get('content')]

        print(f"      ✅ Answer: {answer[:60]}...")
        print(f"      📚 Contexts: {len(contexts)}")

        ground_truth = expected_responses[i-1] if i - \
            1 < len(expected_responses) else ""

        test_case = {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth
        }

        test_cases.append(test_case)

        time.sleep(sleep_time)

    print(f"\n✅ Created {len(test_cases)} test cases")

    # ============= STEP 4: Initialize RAGAS Evaluator =============
    print(f"\n📊 Step 4: Initializing RAGAS Evaluator...")

    evaluator = RagasEvaluator(model=model)

    evaluator.configure_metrics(metrics_config)

    print(f"✅ Configured metrics:")
    for metric_name, config in metrics_config.items():
        if config.get('enabled', True):
            print(f"   - {metric_name}")

    # ============= STEP 5: Run evaluation =============
    print(f"\n Step 5: Running RAGAS evaluation...")

    try:
        results_df = evaluator.evaluate_batch(test_cases)

        print(f"\n✅ Evaluation complete!")

        # ============= STEP 6: Show results =============

        print("\n Average metric values:")
        for metric_name in metrics_config.keys():
            if metric_name in results_df.columns:
                avg_score = results_df[metric_name].mean()
                print(f"   {metric_name}: {avg_score:.3f}")

        print("\n Detailed results:")
        print(results_df.to_string(index=False))

        output_path = "ragas_evaluation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")

        return results_df

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":

    # ============= CONFIGURATION =============

    # 1. Initialize RAG connector
    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-your-key",
        timeout=30
    )

    # 2. Path to Excel file
    excel_path = "eval_rag/data/evaluation_dataset.xlsx"

    # 3. RAGAS metrics configuration
    metrics_config = {
        'faithfulness': {
            'enabled': True,
            'threshold': 0.7
        },
        'response_relevancy': {
            'enabled': True,
            'threshold': 0.7
        },
        'context_precision': {
            'enabled': True,
            'threshold': 0.7
        }
    }

    # ============= RUN EVALUATION =============

    results = evaluate_rag_with_ragas(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_config=metrics_config,
        model="gpt-4o-mini",
        sleep_time=0.1
    )

    if results is not None:
        print("\n🎉 Evaluation successfully completed!")
        print(f"📊 Processed {len(results)} questions")
