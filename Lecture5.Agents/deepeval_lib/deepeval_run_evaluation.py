"""
Integration test for agent evaluation
Metrics: AnswerRelevancy, ToolCorrectness, TaskCompletion
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_connector import AgentConnector  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskCompletionMetric
)  # noqa: E402
from deepeval import evaluate  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402
from typing import List, Optional   # noqa: E402
import time  # noqa: E402


def convert_tools_to_toolcalls(tools_list: List[str]) -> List[ToolCall]:
    """Converts list of tool names to ToolCall objects"""
    return [ToolCall(name=tool_name) for tool_name in tools_list]


def evaluate_agent(
    excel_path: str,
    agent_connector: AgentConnector,
    model: str = "gpt-4o-mini",
    urls: Optional[List[str]] = None,
    threshold: float = 0.7,
    sleep_time: float = 0.5
):
    """
    Evaluate agent on three metrics
    """

    print("=" * 70)
    print("🚀 AGENT INTEGRATION TEST")
    print("=" * 70)

    # ============= STEP 1: Loading dataset =============
    print("\n📂 Step 1: Loading dataset...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Error loading the dataset")
        return

    # Show preview
    parser.preview_dataset(df, n=2)

    # ============= STEP 2: Getting data =============
    print("\n📝 Step 2: Extracting data from dataset...")

    pairs = parser.get_question_response_pairs(df)
    print(f"✅ Received {len(pairs)} question-answer-tools pairs")

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

        # Query to agent
        response = agent_connector.query(question, urls=urls)

        if response.get('error'):
            print(f"   ❌ Error: {response['error']}")
            continue

        # Extracting data
        answer = response.get('output', '')
        tools_used = response.get('tools_used', [])

        print(f"   Answer: {answer}")
        print(f"   Tools: {tools_used}")

        # Creating test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_response if expected_response else None,
            tools_called=convert_tools_to_toolcalls(tools_used),
            expected_tools=convert_tools_to_toolcalls(expected_tools)
        )

        test_cases.append(test_case)

        if i < len(pairs):
            time.sleep(sleep_time)

    print(f"\n✅ Created {len(test_cases)} test cases")

    # ============= STEP 4: Creating metrics =============
    print("\n📊 Step 4: Initializing metrics...")

    metrics = [
        AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=True
        ),
        ToolCorrectnessMetric(
            threshold=threshold,
            include_reason=True,
            should_exact_match=True,  # exact match only
            should_consider_ordering=True  # consider tool call order
        ),
        TaskCompletionMetric(
            threshold=threshold,
            model=model,
            include_reason=True
        )
    ]

    print("✅ Metrics:")
    print("   • AnswerRelevancyMetric")
    print("   • ToolCorrectnessMetric")
    print("   • TaskCompletionMetric")

    # ============= STEP 5: Running evaluation =============
    print("\n🧪 Step 5: Running evaluation...")
    print("=" * 70)

    try:
        evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
        print("\n🌐 Results: https://app.confident-ai.com")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    # ============= CONFIGURATION =============

    # Agent
    agent = AgentConnector(
        endpoint_url="http://5.11.83.110:8004/ask",
        api_key="api-key-for-agent",
        user_id="AleksM",
        session_id="3bb76ef7-3c21-4644-9890-eb5d6a223017"
    )

    # Dataset
    excel_path = "data/evaluation_dataset.xlsx"

    # URLs to pass to agent
    test_urls = [
        "https://www.rentalads.com/apartments-for-rent/ny/new-york/"
    ]

    # ============= RUN =============

    evaluate_agent(
        excel_path=excel_path,
        agent_connector=agent,
        model="gpt-4o-mini",
        urls=test_urls,
        threshold=0.7,
        sleep_time=0.5
    )
