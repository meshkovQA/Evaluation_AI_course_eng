"""
Evaluation of RAG system using Opik metrics.
Uses an existing dataset from Opik UI and runs evaluation.
"""

import sys
sys.path.insert(
    0, '/Users/aleksandrmeskov/Desktop/AI evaluation/AI_practice/Lecture5/eval_rag')

import time  # noqa: E402
from typing import List  # noqa: E402
import opik  # noqa: E402
from opik.evaluation import evaluate  # noqa: E402
from opik import Opik  # noqa: E402
from opik_evaluator import create_metrics, simple_rag_task  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402


def evaluate_rag_with_opik(
    dataset_name: str,
    rag_connector: RAGConnector,
    metrics_list: List[str],
    experiment_name: str = "RAG_Evaluation",
    project_name: str = "default",
    model: str = "gpt-4o-mini",
    sleep_time: float = 0.1
):

    opik.configure(use_local=False)
    client = Opik()

    try:
        source_dataset = client.get_dataset(name=dataset_name)
        source_items = list(source_dataset.get_items())
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    rag_results = []

    for i, item in enumerate(source_items, 1):
        question = item.get("input") or item.get("question", "")
        expected_output = item.get("expected_output", "")

        print(f"  [{i}/{len(source_items)}] {question[:60]}...")

        rag_response = rag_connector.query(question)

        if 'error' in rag_response:
            continue

        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        rag_results.append({
            "input": question,
            "output": actual_output,
            "context": retrieval_context,
            "expected_output": expected_output
        })

        time.sleep(sleep_time)

    new_dataset_name = f"{dataset_name}_results_{int(time.time())}"

    try:
        new_dataset = client.create_dataset(
            name=new_dataset_name,
            description=f"RAG evaluation results for dataset '{dataset_name}'"
        )
        new_dataset.insert(rag_results)
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return

    metrics = create_metrics(
        model=model,
        metrics_list=metrics_list
    )

    try:
        result = evaluate(
            dataset=new_dataset,
            task=simple_rag_task,
            scoring_metrics=metrics,
            experiment_name=experiment_name,
            project_name=project_name
        )

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")


if __name__ == "__main__":

    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="sk-rag-your-key",
        timeout=30
    )

    dataset_name = "simple_test_dataset_v3"
    experiment_name = "RAG_Evaluation_From_Dataset_v6"
    project_name = "Test evaluation"

    metrics_to_use = [
        'answer_relevance',
        'hallucination'
    ]

    evaluate_rag_with_opik(
        dataset_name=dataset_name,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        experiment_name=experiment_name,
        project_name=project_name,
        model="gpt-4o-mini"
    )
