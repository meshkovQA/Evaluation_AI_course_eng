# reference_based.py
import evaluate
from sacrebleu import corpus_bleu, sentence_bleu
from typing import List, Dict
import re

# Load metrics once
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
exact_match_metric = evaluate.load("exact_match")


def preprocess_text(text: str) -> str:
    """Preprocess text for a fairer comparison"""
    # Strip extra whitespace and lowercase
    text = text.strip().lower()
    # Drop trailing punctuation
    text = re.sub(r'[.,!?;:]+$', '', text)
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict:
    """ROUGE scores for summarization"""
    # Minimal preprocessing - only strip trailing punctuation
    clean_predictions = [re.sub(r'[.,!?;:]+$', '', pred.strip())
                         for pred in predictions]
    clean_references = [re.sub(r'[.,!?;:]+$', '', ref.strip())
                        for ref in references]

    # Overall ROUGE
    overall_result = rouge_metric.compute(
        predictions=clean_predictions,
        references=clean_references,
        use_stemmer=True,
        use_aggregator=True  # Aggregate the results
    )

    # Per-example ROUGE
    individual_rouge1 = []
    individual_rouge2 = []
    individual_rougeL = []

    for pred, ref in zip(clean_predictions, clean_references):
        individual_result = rouge_metric.compute(
            predictions=[pred],
            references=[ref],
            use_stemmer=True
        )
        individual_rouge1.append(individual_result["rouge1"])
        individual_rouge2.append(individual_result["rouge2"])
        individual_rougeL.append(individual_result["rougeL"])

    return {
        "overall": {
            "rouge1": overall_result["rouge1"],
            "rouge2": overall_result["rouge2"],
            "rougeL": overall_result["rougeL"]
        },
        "individual": {
            "rouge1": individual_rouge1,
            "rouge2": individual_rouge2,
            "rougeL": individual_rougeL
        }
    }


def calculate_meteor(predictions: List[str], references: List[str]) -> Dict:
    """METEOR score that accounts for synonyms"""
    # METEOR handles text on its own, but we can help with normalization.
    # Only strip trailing punctuation for a better match
    clean_predictions = [re.sub(r'[.,!?;:]+$', '', pred.strip())
                         for pred in predictions]
    clean_references = [re.sub(r'[.,!?;:]+$', '', ref.strip())
                        for ref in references]

    # Overall METEOR
    overall_result = meteor_metric.compute(
        predictions=clean_predictions,
        references=clean_references
    )

    # Per-example METEOR
    individual_scores = []
    for pred, ref in zip(clean_predictions, clean_references):
        individual_result = meteor_metric.compute(
            predictions=[pred],
            references=[ref]
        )
        individual_scores.append(individual_result["meteor"])

    return {
        "overall": overall_result["meteor"],
        "individual": individual_scores
    }


def calculate_sacrebleu(predictions: List[str], references: List[str]) -> Dict:
    """SacreBLEU for a more precise evaluation"""
    # SacreBLEU prefers the original text without preprocessing.
    # Only strip trailing punctuation for a better match
    clean_predictions = [re.sub(r'[.,!?;:]+$', '', pred.strip())
                         for pred in predictions]
    clean_references = [re.sub(r'[.,!?;:]+$', '', ref.strip())
                        for ref in references]

    # Overall SacreBLEU
    overall_score = corpus_bleu(clean_predictions, [clean_references])

    # Per-example SacreBLEU
    individual_scores = []
    for pred, ref in zip(clean_predictions, clean_references):
        individual_score = sentence_bleu(pred, [ref])
        individual_scores.append(individual_score.score / 100)

    return {
        "overall": overall_score.score / 100,
        "individual": individual_scores
    }


def calculate_exact_match(predictions: List[str], references: List[str]) -> Dict:
    """Exact Match for answer equality"""
    # Apply full preprocessing for exact match
    clean_predictions = [preprocess_text(pred) for pred in predictions]
    clean_references = [preprocess_text(ref) for ref in references]

    # Overall Exact Match
    overall_result = exact_match_metric.compute(
        predictions=clean_predictions,
        references=clean_references
    )

    # Per-example Exact Match
    individual_scores = []
    for pred, ref in zip(clean_predictions, clean_references):
        individual_result = exact_match_metric.compute(
            predictions=[pred],
            references=[ref]
        )
        individual_scores.append(individual_result["exact_match"])

    return {
        "overall": overall_result["exact_match"],
        "individual": individual_scores
    }
