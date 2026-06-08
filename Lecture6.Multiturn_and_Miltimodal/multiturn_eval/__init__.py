"""
Multiturn Evaluation Module for DeepEval
Interactive CLI chat with multi-turn conversation evaluation

Standalone module — no external project file dependencies.
"""

from .conversation_storage import ConversationStorage
from .evaluators.deepeval.multiturn_evaluator import (
    create_metrics,
    conversation_to_test_case,
    run_evaluation,
    DEFAULT_CHATBOT_ROLE
)
from .agent_connector import AgentConnector

__all__ = [
    'ConversationStorage',
    'create_metrics',
    'conversation_to_test_case',
    'run_evaluation',
    'DEFAULT_CHATBOT_ROLE',
    'AgentConnector',
]
