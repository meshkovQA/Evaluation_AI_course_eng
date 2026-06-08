"""
Multi-turn conversation evaluation module
Supports DeepEval and Opik frameworks
"""

# Import from DeepEval
from .deepeval import run_evaluation as run_deepeval_evaluation
from .deepeval import DEFAULT_CHATBOT_ROLE

# Try to import Opik
try:
    from .opik import run_evaluation as run_opik_evaluation
    HAS_OPIK = True
except ImportError:
    HAS_OPIK = False
    run_opik_evaluation = None


def run_evaluation(
    conversations,
    framework: str = "deepeval",
    **kwargs
):
    """
    Universal evaluation function with framework selection

    Args:
        conversations: list of conversations
        framework: "deepeval" or "opik"
        **kwargs: additional parameters for the selected framework

    Returns:
        Evaluation results
    """
    if framework == "deepeval":
        return run_deepeval_evaluation(conversations, **kwargs)
    elif framework == "opik":
        if not HAS_OPIK:
            raise ImportError("Opik is not installed. Run: pip install opik")
        return run_opik_evaluation(conversations, **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}. Use 'deepeval' or 'opik'")
