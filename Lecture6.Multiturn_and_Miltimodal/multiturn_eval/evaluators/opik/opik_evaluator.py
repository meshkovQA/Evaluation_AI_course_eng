"""
Multi-turn conversation evaluation using Opik metrics

Conversation LLM as a Judge Metrics:
- ConversationalCoherenceMetric - conversation coherence
- SessionCompletenessQuality - fulfillment of user goals
- UserFrustrationMetric - user frustration level
"""

import os
from typing import List, Dict
import opik

# Import conversation metrics
from opik.evaluation.metrics import (
    ConversationalCoherenceMetric,
    SessionCompletenessQuality,
    UserFrustrationMetric
)


# Default chatbot role
DEFAULT_CHATBOT_ROLE = """AI assistant for real estate search and analysis.
Helps users find apartments, filter by various criteria
(price, neighborhood, number of rooms), convert currencies, and provide
price statistics. Responds politely, informatively, and in English."""

# Default project name for Opik
DEFAULT_PROJECT_NAME = "multiturn-evaluation"


def create_metrics(
    model: str = "gpt-4o-mini",
    include_reason: bool = True,
    window_size: int = 5
) -> List:
    """
    Creates list of metrics for multi-turn conversation evaluation
    """
    metrics = []

    # 1. ConversationalCoherenceMetric - conversation coherence (LLM)
    metrics.append(
        ConversationalCoherenceMetric(
            model=model,
            window_size=window_size,
            include_reason=include_reason
        )
    )

    # 2. SessionCompletenessQuality - fulfillment of user goals (LLM)
    metrics.append(
        SessionCompletenessQuality(
            model=model,
            include_reason=include_reason
        )
    )

    # 3. UserFrustrationMetric - frustration level (LLM)
    metrics.append(
        UserFrustrationMetric(
            model=model,
            include_reason=include_reason
        )
    )

    return metrics


def conversation_to_opik_format(conv: Dict) -> List[Dict[str, str]]:
    """
    Converts a saved conversation to Opik format
    """
    conversation = []

    for turn_data in conv.get('turns', []):
        conversation.append({
            "role": turn_data.get('role'),
            "content": turn_data.get('content', '')
        })

    return conversation


def run_evaluation(
    conversations: List[Dict],
    model: str = "gpt-4o-mini",
    verbose: bool = True,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict:
    """
    Starts evaluation of a list of conversations with Opik metrics
    """
    if not conversations:
        print("No conversations to evaluate")
        return {}

    # Initialize Opik with project name
    opik.configure(use_local=False)

    # Set default project for traces
    os.environ["OPIK_PROJECT_NAME"] = project_name

    print("\n" + "=" * 70)
    print("🧪 MULTI-TURN CONVERSATION EVALUATION (Opik)")
    print(f"   📁 Project: {project_name}")
    print(f"   🤖 Model: {model}")
    print("=" * 70)

    # Create metrics with model (string model name)
    print("\n📊 Initializing metrics...")
    metrics = create_metrics(model=model)

    metric_names = [type(m).__name__ for m in metrics]
    for name in metric_names:
        print(f"   • {name}")

    # Convert conversations
    print(f"\n📝 Processing {len(conversations)} conversations...")

    all_results = []

    for conv in conversations:
        conv_id = conv.get('id', 'N/A')
        turns_count = len(conv.get('turns', []))

        if verbose:
            print(f"\n   [{conv_id}] {turns_count} turns")

        # Convert to Opik format
        conversation = conversation_to_opik_format(conv)

        # Evaluate with each metric
        conv_results = {
            "id": conv_id,
            "turns_count": turns_count,
            "scores": {}
        }

        for metric in metrics:
            try:
                result = metric.score(conversation=conversation)
                conv_results["scores"][type(metric).__name__] = {
                    "value": result.value,
                    "reason": getattr(result, 'reason', None)
                }

                if verbose:
                    print(
                        f"      • {type(metric).__name__}: {result.value:.3f}")

            except Exception as e:
                conv_results["scores"][type(metric).__name__] = {
                    "value": None,
                    "error": str(e)
                }
                if verbose:
                    print(f"      • {type(metric).__name__}: ❌ {e}")

        all_results.append(conv_results)

    # Compute average scores
    print("\n" + "=" * 70)
    print("📈 AVERAGE SCORES:")
    print("-" * 70)

    avg_scores = {}
    for metric_name in metric_names:
        values = [
            r["scores"].get(metric_name, {}).get("value")
            for r in all_results
            if r["scores"].get(metric_name, {}).get("value") is not None
        ]
        if values:
            avg = sum(values) / len(values)
            avg_scores[metric_name] = avg
            print(f"   {metric_name}: {avg:.3f}")
        else:
            print(f"   {metric_name}: N/A")

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    print("\n🌐 Detailed results: https://www.comet.com/opik")

    return {
        "conversations": all_results,
        "averages": avg_scores
    }
