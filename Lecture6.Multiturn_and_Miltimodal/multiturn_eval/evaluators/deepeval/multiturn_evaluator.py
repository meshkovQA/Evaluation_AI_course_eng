"""
Multi-turn conversation evaluation using DeepEval metrics
"""

from typing import List, Dict, Optional
from deepeval.test_case import ConversationalTestCase, Turn, ToolCall
from deepeval.metrics import (
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric
)
from deepeval import evaluate

# Try to import TurnRelevancyMetric (may be unavailable in older versions)
try:
    from deepeval.metrics import TurnRelevancyMetric
    HAS_TURN_RELEVANCY = True
except ImportError:
    HAS_TURN_RELEVANCY = False
    print("Warning: TurnRelevancyMetric is unavailable in your version of deepeval")


# Default chatbot role (used for RoleAdherenceMetric)
DEFAULT_CHATBOT_ROLE = """AI assistant for real estate search and analysis.
Helps users find apartments, filter by various criteria
(price, neighborhood, number of rooms), convert currencies, and provide
price statistics. Responds politely, informatively, and in English."""


def create_metrics(
    model: str,
    threshold: float = 0.5,
    include_reason: bool = True
) -> List:
    """
    Creates list of metrics for multi-turn conversation evaluation
    """
    metrics = []

    # 1. TurnRelevancyMetric - relevance of each response
    if HAS_TURN_RELEVANCY:
        metrics.append(
            TurnRelevancyMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
        )

    # 2. RoleAdherenceMetric - adherence to role
    metrics.append(
        RoleAdherenceMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
    )

    # 3. KnowledgeRetentionMetric - context retention
    metrics.append(
        KnowledgeRetentionMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
    )

    # 4. ConversationCompletenessMetric - task completion coverage
    metrics.append(
        ConversationCompletenessMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
    )

    return metrics


def conversation_to_test_case(
    conv: Dict,
    chatbot_role: str = DEFAULT_CHATBOT_ROLE,
    expected_outcome: Optional[str] = None
) -> ConversationalTestCase:
    """
    Converts a saved conversation to ConversationalTestCase for DeepEval
    """
    turns = []

    for turn_data in conv.get('turns', []):
        role = turn_data.get('role')
        content = turn_data.get('content', '')

        # tools_called only for assistant
        tools_called = None
        if role == 'assistant':
            tools_list = turn_data.get('tools_called', [])
            if tools_list:
                tools_called = [ToolCall(name=tool_name)
                                for tool_name in tools_list]

        turn = Turn(
            role=role,
            content=content,
            tools_called=tools_called
        )
        turns.append(turn)

    return ConversationalTestCase(
        chatbot_role=chatbot_role,
        turns=turns,
        expected_outcome=expected_outcome
    )


def run_evaluation(
    conversations: List[Dict],
    model: str = "gpt-4o-mini",
    threshold: float = 0.5,
    chatbot_role: str = DEFAULT_CHATBOT_ROLE,
    verbose: bool = True
) -> None:
    """
    Starts evaluation of a list of conversations
    """
    if not conversations:
        print("No conversations to evaluate")
        return

    print("\n" + "=" * 70)
    print("🧪 MULTI-TURN CONVERSATION EVALUATION (DeepEval)")
    print("=" * 70)

    # Creating metrics
    print("\n📊 Initializing metrics...")
    metrics = create_metrics(model, threshold=threshold)

    metric_names = [type(m).__name__ for m in metrics]
    for name in metric_names:
        print(f"   • {name}")

    # Convert conversations to test cases
    print(f"\n📝 Converting {len(conversations)} conversations to test cases...")
    test_cases = []
    for conv in conversations:
        test_case = conversation_to_test_case(conv, chatbot_role=chatbot_role)
        test_cases.append(test_case)

        if verbose:
            conv_id = conv.get('id', 'N/A')
            turns_count = len(conv.get('turns', []))
            print(f"   [{conv_id}] {turns_count} turns")

    # Running evaluation
    print(f"\n🚀 Running evaluation...")
    print("-" * 70)

    try:
        results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
        print("\n🌐 Detailed results: https://app.confident-ai.com")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
