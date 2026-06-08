# embedding_metrics_demo.py

from ai_client import openai_chat_v2
from typing import List, Dict
from embedding_metrics import (
    calculate_bertscore,
    calculate_semantic_similarity,
    calculate_coherence_sbert
)


def print_metric_per_text(metric_name: str, results: Dict, texts: List[str]):
    print(f"\n--- {metric_name.upper()} ---")

    if metric_name == "BERTScore":
        print(f"Overall scores:")
        print(f"  Precision: {results['overall']['precision']:.3f}")
        print(f"  Recall: {results['overall']['recall']:.3f}")
        print(f"  F1: {results['overall']['f1']:.3f}")

        print(f"Pairwise comparisons:")
        for i, (ref, cand, f1) in enumerate(results['individual']):
            print(f"  Pair {i+1}: F1 = {f1:.3f}")

    elif metric_name == "Semantic Similarity":
        print(
            f"Mean semantic similarity: {results['overall_similarity_mean']:.3f}")
        print(f"Pairwise similarities:")
        n_texts = len(texts)
        pair_idx = 0
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                similarity = results['pairwise'][pair_idx]
                print(f"  Text {i+1} <-> Text {j+1}: {similarity:.3f}")
                pair_idx += 1

    elif metric_name == "Coherence SBERT":
        print(f"Overall coherence: {results['overall']:.3f}")
        print(f"Per-text:")
        for i, score in enumerate(results['individual']):
            if score is not None:
                print(f"  Text {i+1}: {score:.3f}")
            else:
                print(f"  Text {i+1}: N/A (not enough sentences)")

    print("-" * 50)


if __name__ == "__main__":
    print("=== EMBEDDING METRICS TESTING ===")

    # Prompts
    prompts = [
        "Write a short text about artificial intelligence",
        "Write a text about artificial intelligence",
        # "Explain in simple terms what neural networks are"
    ]

    print("Generating texts...")
    texts = [openai_chat_v2(prompt) for prompt in prompts]

    print("\nGenerated texts:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}\n")

    print("=" * 60)

    # BERTSCORE
    print("\nComputing BERTSCORE...")
    bertscore_results = calculate_bertscore(texts, lang="en")
    print_metric_per_text("BERTScore", bertscore_results, texts)

    # SEMANTIC SIMILARITY
    print("\nComputing SEMANTIC SIMILARITY...")
    similarity_results = calculate_semantic_similarity(texts)
    print_metric_per_text("Semantic Similarity", similarity_results, texts)

    # COHERENCE SBERT
    print("\nComputing COHERENCE SBERT...")
    coherence_results = calculate_coherence_sbert(texts)
    print_metric_per_text("Coherence SBERT", coherence_results, texts)
