# metric_eval_2.py

from reference_free import (
    calculate_distinct,
    calculate_perplexity,
    calculate_readability,
    calculate_self_bleu,
    calculate_coherence_with_transformers,
    calculate_grammar_errors
)
from typing import List, Dict
from ai_client import openai_chat_v2

"""
    How to read the metrics:

    PERPLEXITY - perplexity shows how "natural" and grammatically normal the text looks from the language model's point of view.
    - The lower the value, the more natural and "understandable" the text.
    - High values mean the words look unusual to the model, or the structure is strange.

    READABILITY (Flesch reading-ease score). The metric gauges how easy it is for a human to read the text.
    Values usually range from 0 to 100:
        • 90-100 - very easy to read (children's texts)
        • 60-70 - regular level (news, articles)
        • 30-50 - difficult texts (academic, technical)
        • <30 - very hard to read

    GRAMMAR ERRORS. Number of grammatical mistakes detected by a tool (often LanguageTool / a GPT-based detector).
    - Lower is better. 0 is perfect (no errors).

    COHERENCE. How logical and connected the text is.
    - Values from 0 to 1, where 1 is a perfectly coherent text.
    Typically:
        • 0.6-0.7 - moderate coherence
        • 0.8+ - excellent logic and flow

    SELF-BLEU. How diverse the text is compared to itself.
    - Values from 0 to 1, where 0 is maximally diverse and 1 is fully repetitive.
    Typically:
    • 0.3-0.5 - a good level of diversity
    • 0.6+ - the text may be too repetitive

    DISTINCT (diversity). How diverse the text is in terms of unique n-grams.
    Shows the fraction of unique words and phrases:
        • distinct_1 - unique words (unigrams)
        • distinct_2 - unique word pairs (bigrams)

    DISTINCT-1 - word-level diversity (unigrams)
    Higher means more diverse and "lively" text.
    - Low values 0.1-0.3 - the text may be monotonous and repetitive
    - Medium 0.3-0.6 - balanced diversity. Normal for news, technical, or FAQ texts
    - High 0.6-0.8 - diverse text, natural variability
    - Very high 0.8+ - often in creative or long texts

    DISTINCT-2 - phrase-level diversity (bigrams)
    Higher means more varied phrases and constructions are used.
    - Low values 0.2-0.8 - the same connectors repeat ("in this article", "you can use")
    - Medium 0.8-1.5 - normal phrase variety. Acceptable for news and scientific articles
    - High 1.5-2.5 - text with a good variety of phrases and constructions, natural language
    - Very high 2.5+ - very varied, creative style


    """


def print_metric_per_text(metric_name: str, results: List, texts: List[str]):
    print(f"\n--- {metric_name.upper()} ---")
    for i, (text, score) in enumerate(zip(texts, results)):
        print(f"Text {i+1}: {text[:50]}...")
        if isinstance(score, dict):
            for key, val in score.items():
                print(f"  {key}: {val:.3f}")
        else:
            print(f"  Score: {score:.3f}")
    print("-" * 50)


if __name__ == "__main__":
    print("=== REFERENCE-FREE METRICS TESTING ===")

    # Prompts
    prompts = [
        "Write a short text about artificial intelligence",
        "Describe the benefits of machine learning",
        "Explain in simple terms what neural networks are"
    ]

    print("Generating texts...")
    texts = [openai_chat_v2(prompt) for prompt in prompts]

    print("\nGenerated texts:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}\n")

    print("=" * 60)

    # PERPLEXITY
    print("\nComputing PERPLEXITY...")
    perplexities = [calculate_perplexity(text) for text in texts]
    print_metric_per_text("Perplexity", perplexities, texts)

    # READABILITY
    print("\nComputing READABILITY...")
    readability_scores = [calculate_readability(text) for text in texts]
    print_metric_per_text("Readability", readability_scores, texts)

    # GRAMMAR ERRORS
    print("\nComputing GRAMMAR ERRORS...")
    grammar_errors = [calculate_grammar_errors(text) for text in texts]
    print_metric_per_text("Grammar Errors", grammar_errors, texts)

    # COHERENCE
    print("\nComputing COHERENCE...")
    coherence_scores = [
        calculate_coherence_with_transformers(text) for text in texts]
    print_metric_per_text("Coherence", coherence_scores, texts)

    # SELF-BLEU
    print("\nComputing SELF-BLEU...")
    self_bleu_scores = [calculate_self_bleu(
        i, texts, n=2) for i in range(len(texts))]
    print_metric_per_text("Self-BLEU (2-gram)", self_bleu_scores, texts)

    # DISTINCT
    print("\nComputing DISTINCT...")
    distinct_result = calculate_distinct(texts)
    print(f"\n--- DISTINCT ---")
    for key, val in distinct_result.items():
        print(f"{key}: {val:.3f}")
    print("-" * 50)
