"""
Integration test for agent evaluation with eval_lib
Metrics: AnswerRelevancy, ToolCorrectness, TaskSuccessRate
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio  # noqa: E402
import time  # noqa: E402
from typing import List, Optional  # noqa: E402
from eval_lib import (
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskSuccessRateMetric,
)  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from agent_connector import AgentConnector  # noqa: E402


async def evaluate_agent_with_eval_lib(
    excel_path: str,
    agent_connector: AgentConnector,
    metrics_list: List[str],
    urls: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    threshold: float = 0.7,
    temperature: float = 0.5,
    sleep_time: float = 0.5,
    verbose: bool = True,
    show_dashboard: bool = False,
    session_name: str = "Agent Evaluation"
):
    """
    Evaluate agent using eval_lib

    Args:
        excel_path: path to dataset
        agent_connector: connector to agent
        metrics_list: list of metrics for evaluation
        urls: list of URLs to pass to agent
        model: model for metrics
        threshold: metric threshold
        temperature: temperature for LLM
        sleep_time: delay between requests
        verbose: detailed output
        show_dashboard: show dashboard
        session_name: session name
    """

    print("\n" + "=" * 70)
    print(f"🚀 {session_name.upper()}")
    print("=" * 70)

    # ============= STEP 1: Loading dataset =============
    print("\n📂 Step 1: Loading dataset...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Error loading the dataset")
        return None

    # Show information
    info = parser.validate_dataset(df)
    print(f"\n📊 Dataset info:")
    print(f"   • Total rows: {info['total_rows']}")
    print(f"   • Valid pairs: {info['valid_pairs']}")
    if info['has_expected_tools_column']:
        print(f"   • Average tool count: {info['avg_tools_count']:.1f}")
        print(f"   • Unique tools: {len(info['unique_tools'])}")

    # Preview
    parser.preview_dataset(df, n=2)

    # ============= STEP 2: Extracting data =============
    print("\n📝 Step 2: Extracting data from dataset...")

    pairs = parser.get_question_response_pairs(df)
    print(f"✅ Received {len(pairs)} pairs")

    if urls:
        print(f"🔗 URLs for agent: {urls}")

    # ============= STEP 3: Queries to agent =============
    print("\n🤖 Step 3: Getting answers from agent...")

    test_cases = []

    for i, pair in enumerate(pairs, 1):
        question = pair['question']
        expected_response = pair['expected_response']
        expected_tools = pair['expected_tools']

        print(f"\n[{i}/{len(pairs)}] {question[:60]}...")

        # Query to agent with URLs
        response = agent_connector.query(question, urls=urls)

        if response.get('error'):
            print(f"   ❌ {response['error']}")
            continue

        # Extracting data
        answer = response.get('output', '')
        tools_used = response.get('tools_used', [])

        print(f"   ✅ Answer: {answer[:50]}...")
        print(f"   🔧 Tools: {tools_used}")

        # Creating EvalTestCase
        test_case = EvalTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_response,
            tools_called=tools_used,  # list of strings
            expected_tools=expected_tools  # list of strings
        )

        test_cases.append(test_case)

        if i < len(pairs):
            time.sleep(sleep_time)

    print(f"\n✅ Created {len(test_cases)} test cases")

    # ============= STEP 4: Creating metrics =============
    print("\n📊 Step 4: Creating metrics...")
    print(f"   Metrics: {metrics_list}")
    print(f"   Threshold: {threshold}")
    print(f"   Temperature: {temperature}")
    print(f"   Model: {model}")

    metrics = []
    metric_classes = {
        'answer_relevancy': AnswerRelevancyMetric,
        'tool_correctness': ToolCorrectnessMetric,
        'task_success_rate': TaskSuccessRateMetric
    }

    for metric_name in metrics_list:
        if metric_name == 'tool_correctness':
            metric = ToolCorrectnessMetric(
                threshold=threshold,
                verbose=verbose,
                exact_match=True,
                check_ordering=True
            )
            metrics.append(metric)
            print(f"   ✅ ToolCorrectnessMetric")
        elif metric_name in metric_classes:
            metric = metric_classes[metric_name](
                model=model,
                threshold=threshold,
                temperature=temperature,
                verbose=verbose
            )
            metrics.append(metric)
            print(f"   ✅ {metric_name}")

    print(f"\n✅ Created {len(metrics)} metrics")

    # ============= STEP 5: Running evaluation =============
    print("\n🧪 Step 5: Running evaluation...")
    print(f"   Test cases: {len(test_cases)}")
    print(f"   Metrics: {len(metrics)}")
    print("=" * 70)

    try:
        # Running evaluation
        results = await evaluate(
            test_cases=test_cases,
            metrics=metrics,
            verbose=verbose,
            show_dashboard=show_dashboard,
            session_name=session_name
        )

        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


async def scenario_1():
    """Scenario 1: Evaluation with OpenAI model"""

    print("\n" + "=" * 70)
    print("📋 SCENARIO 1: Evaluation with OpenAI model")
    print("=" * 70)

    # Agent
    agent = AgentConnector(
        endpoint_url="http://5.11.83.110:8004/ask",
        api_key="api-key-for-agent",
        user_id="AleksM",
        session_id="faecb783-d996-49ee-97a0-f13805f63a52"
    )

    # Dataset
    excel_path = "data/evaluation_dataset.xlsx"

    # URLs for agent
    test_urls = [
        "https://www.rentalads.com/apartments-for-rent/ny/new-york/"
    ]

    # Metrics
    metrics_to_use = [
        'answer_relevancy',
        'tool_correctness',
        'task_success_rate'
    ]

    # Running evaluation
    results = await evaluate_agent_with_eval_lib(
        excel_path=excel_path,
        agent_connector=agent,
        metrics_list=metrics_to_use,
        urls=test_urls,
        model="gpt-4o-mini",
        threshold=0.7,
        temperature=0.5,
        sleep_time=0.5,
        verbose=True,
        show_dashboard=True,
        session_name="Agent Evaluation"
    )

    return results


if __name__ == "__main__":

    asyncio.run(scenario_1())
