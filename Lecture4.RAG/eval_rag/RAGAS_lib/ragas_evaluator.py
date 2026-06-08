"""
RAGAS Evaluator (uses OpenAI directly via LangChain)
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
    Faithfulness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class RagasEvaluator:
    """RAGAS Evaluator using OpenAI directly"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialization

        Args:
            model: Model name
            temperature: Temperature
        """
        self.model = model
        self.temperature = temperature
        self.metric_configs = {}

        llm = ChatOpenAI(model=model, temperature=temperature)
        self.evaluator_llm = LangchainLLMWrapper(llm)

        embeddings = OpenAIEmbeddings()
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

        print(f"✅ RAGAS configured: {model}")

    def configure_metrics(self, metrics_config: Dict[str, Dict[str, Any]]):
        """Configure metrics"""
        self.metric_configs = metrics_config

    def _create_metric(self, metric_name: str):
        """Create a metric"""
        if metric_name == 'context_precision':
            return LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        elif metric_name == 'context_recall':
            return LLMContextRecall(llm=self.evaluator_llm)
        elif metric_name == 'response_relevancy':
            return ResponseRelevancy(
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )
        elif metric_name == 'faithfulness':
            return Faithfulness(llm=self.evaluator_llm)

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a single case

        Args:
            question: User question
            answer: RAG system answer
            contexts: List of retrieved contexts
            ground_truth: Correct answer

        Returns:
            Dict with metric scores
        """
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=ground_truth,
            retrieved_contexts=contexts
        )

        results = {}

        for metric_name, config in self.metric_configs.items():
            if not config.get('enabled', True):
                continue

            print(f"   Evaluating: {metric_name}")
            metric = self._create_metric(metric_name)
            score = await metric.single_turn_ascore(sample)
            results[metric_name] = float(score)

        return results

    def evaluate_batch(self, test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Batch evaluation

        Args:
            test_cases: List of dicts with keys:
                - question: str
                - answer: str
                - contexts: List[str]
                - ground_truth: str

        Returns:
            DataFrame with results
        """
        async def evaluate_all():
            all_results = []
            for i, case in enumerate(test_cases, 1):
                print(f"\n📝 {i}/{len(test_cases)}: {case['question'][:50]}...")

                scores = await self.evaluate_single(
                    question=case['question'],
                    answer=case['answer'],
                    contexts=case['contexts'],
                    ground_truth=case['ground_truth']
                )

                scores['question'] = case['question']
                all_results.append(scores)

            return all_results

        results = asyncio.run(evaluate_all())
        return pd.DataFrame(results)
