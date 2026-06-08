from ai_client import openai_chat_v2
from typing import List, Dict, Any
from reference_based import calculate_rouge, calculate_meteor, calculate_sacrebleu, calculate_exact_match


def print_detailed_results(metric_name: str, results: Dict, prompts: List[str], predictions: List[str], references: List[str]):
    """Print detailed metric results"""
    print(f"\n--- {metric_name} ---")
    if isinstance(results["overall"], dict):
        # For ROUGE
        for key, value in results["overall"].items():
            print(f"Overall {key.upper()}: {value:.3f}")
        print("\nPer query:")
        for i, (prompt, pred, ref) in enumerate(zip(prompts, predictions, references)):
            print(f"Query:")
            print(f"  {prompt}")
            print(f"LLM answer:")
            print(f"  {pred}")
            print(f"Reference:")
            print(f"  {ref}")
            for key in results["individual"]:
                print(f"  {key.upper()}: {results['individual'][key][i]:.3f}")
            print()
    else:
        # For the other metrics
        print(f"Overall: {results['overall']:.3f}")
        print("\nPer query:")
        for i, (prompt, pred, ref, score) in enumerate(zip(prompts, predictions, references, results["individual"])):
            print(f"Query:")
            print(f"  {prompt}")
            print(f"LLM answer:")
            print(f"  {pred}")
            print(f"Reference:")
            print(f"  {ref}")
            print(f"  Score: {score:.3f}")
            print()


if __name__ == "__main__":
    # ---------  Translation testing example ----------

    print("=== TRANSLATION TESTING ===")

    translation_prompts = [
        "Translate to French: Good morning",
        "Translate to French: Beautiful weather",
        "Translate to French: I love books"
    ]

    translation_references = [
        "Bonjour",
        "Beau temps",
        "J'aime les livres"
    ]

    print("Generating translations...")
    translation_predictions = [openai_chat_v2(
        prompt) for prompt in translation_prompts]

    print("Generated translations:")
    for i, pred in enumerate(translation_predictions):
        print(f"  {i+1}. {pred}")

    meteor_results = calculate_meteor(
        translation_predictions, translation_references)

    print_detailed_results("METEOR", meteor_results, translation_prompts,
                           translation_predictions, translation_references)

    # ---------  Summarization testing example ----------

    print("\n\n=== SUMMARIZATION TESTING ===")

    summarization_prompts = [
        "Summarize in one sentence: Artificial intelligence is rapidly transforming industries through machine learning algorithms that can process vast amounts of data and identify patterns that humans might miss.",
        "Summarize in one sentence: Climate change represents one of the most pressing challenges of our time, requiring immediate global action to reduce greenhouse gas emissions."
    ]

    summarization_references = [
        "artificial intelligence transforms industries through machine learning algorithms for data analysis and pattern detection",
        "climate change requires urgent global measures to reduce greenhouse gas emissions"
    ]

    print("Generating summaries...")
    summarization_predictions = [openai_chat_v2(
        prompt) for prompt in summarization_prompts]

    print("Generated summaries:")
    for i, pred in enumerate(summarization_predictions):
        print(f"  {i+1}. {pred}")

    rouge_results = calculate_rouge(
        summarization_predictions, summarization_references)
    print_detailed_results("ROUGE", rouge_results, summarization_prompts,
                           summarization_predictions, summarization_references)

    # ---------  Exact-translation testing example ----------

    print("\n\n=== EXACT TRANSLATION TESTING ===")

    precise_translation_prompts = [
        "Translate exactly to French: Hello world",
        "Translate exactly to French: Thank you"
    ]

    precise_translation_references = [
        "Bonjour le monde",
        "Merci"
    ]

    print("Generating exact translations...")
    precise_predictions = [openai_chat_v2(
        prompt) for prompt in precise_translation_prompts]

    print("Generated translations:")
    for i, pred in enumerate(precise_predictions):
        print(f"  {i+1}. {pred}")

    sacrebleu_results = calculate_sacrebleu(
        precise_predictions, precise_translation_references)
    print_detailed_results("SacreBLEU", sacrebleu_results, precise_translation_prompts,
                           precise_predictions, precise_translation_references)

    # ---------  Exact-answer (QA) testing example ----------

    print("\n\n=== EXACT ANSWER (QA) TESTING ===")

    qa_prompts = [
        "What is the capital of France? Answer in one word.",
        "What is 2+2? Answer with a single digit.",
        "What color do you get by mixing red and blue? Answer in one word."
    ]

    qa_references = [
        "Paris",
        "4",
        "Purple"
    ]

    print("Generating answers...")
    qa_predictions = [openai_chat_v2(prompt) for prompt in qa_prompts]

    print("Generated answers:")
    for i, pred in enumerate(qa_predictions):
        print(f"  {i+1}. {pred}")

    exact_match_results = calculate_exact_match(qa_predictions, qa_references)
    print_detailed_results("Exact Match", exact_match_results,
                           qa_prompts, qa_predictions, qa_references)
