"""
DeepEval multimodal metrics for image evaluation

Metrics based on GPT-4o (or other MLLM):
- ImageCoherence: coherence of images with text
- ImageHelpfulness: helpfulness of images for understanding
- TextToImageMetric: quality of image generation from prompt
- MultimodalAnswerRelevancy: relevance of RAG answer with images

All metrics return a score 0-1 (higher is better)
"""

from pathlib import Path
from typing import Union, List

# DeepEval imports
from deepeval.test_case import MLLMTestCase, MLLMImage
from deepeval.metrics import (
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    TextToImageMetric,
    MultimodalAnswerRelevancyMetric
)
from deepeval import evaluate


def _create_mllm_image(path: Union[str, Path]) -> MLLMImage:
    """
    Creates MLLMImage object from file path.

    Args:
        path: Path to image (local or URL)

    Returns:
        MLLMImage object for DeepEval
    """
    path_str = str(path)

    if path_str.startswith(("http://", "https://")):
        return MLLMImage(url=path_str, local=False)
    else:
        abs_path = str(Path(path).resolve())
        return MLLMImage(url=abs_path, local=True)


def _build_actual_output(items: List) -> List:
    """
    Converts list of elements to actual_output for DeepEval.
    File path strings are converted to MLLMImage.
    """
    actual_output = []
    for item in items:
        if isinstance(item, str):
            if Path(item).exists():
                actual_output.append(_create_mllm_image(item))
            else:
                actual_output.append(item)
        elif isinstance(item, MLLMImage):
            actual_output.append(item)
        else:
            actual_output.append(item)
    return actual_output


def evaluate_image_coherence(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[input_prompt],
        actual_output=_build_actual_output(actual_output)
    )

    metric = ImageCoherenceMetric(
        threshold=threshold,
        model=model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])

    return res


def evaluate_image_helpfulness(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[input_prompt],
        actual_output=_build_actual_output(actual_output)
    )

    metric = ImageHelpfulnessMetric(
        threshold=threshold,
        model=model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])

    return res


def evaluate_text_to_image(
    prompt: str,
    generated_image_path: Union[str, Path],
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[prompt],
        actual_output=[_create_mllm_image(generated_image_path)]
    )

    metric = TextToImageMetric(
        threshold=threshold,
        model=model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])
    return res


def evaluate_multimodal_relevancy(
    question: str,
    answer_with_images: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini"
) -> dict:

    test_case = MLLMTestCase(
        input=[question],
        actual_output=_build_actual_output(answer_with_images),

    )

    metric = MultimodalAnswerRelevancyMetric(
        threshold=threshold,
        model=model
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])

    return res


# Testing when started directly
if __name__ == "__main__":
    print("=" * 50)
    print("Testing DeepEval multimodal metrics")
    print("=" * 50)

    # Use ready-made sample_images
    sample_dir = Path(__file__).parent.parent / "sample_images"

    original = sample_dir / "original.png"
    edited = sample_dir / "edited.png"
    different = sample_dir / "different.png"

    if not original.exists():
        print(f"\n⚠️  Test images not found in {sample_dir}")
        print("   Add images to the sample_images/ folder")
        exit(1)

    print(f"\n📁 Using images from {sample_dir}")

    # ---------------------------------------------------
    # 1. TextToImageMetric
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("1. TextToImageMetric")
    print("-" * 40)

    prompt = "a man with glasses working on a laptop with code on monitors in an office"

    result = evaluate_text_to_image(
        prompt=prompt,
        generated_image_path=original
    )

    # ---------------------------------------------------
    # 2. ImageCoherence
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("2. ImageCoherence")
    print("-" * 40)

    input_prompt = "Show me a software developer at work"
    actual_output = [
        str(original)
    ]

    result = evaluate_image_coherence(
        input_prompt=input_prompt,
        actual_output=actual_output
    )

    # ---------------------------------------------------
    # 3. ImageHelpfulness
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("3. ImageHelpfulness")
    print("-" * 40)

    input_prompt = "Show me what a programmer's workspace looks like"
    actual_output = [
        str(edited)
    ]

    result = evaluate_image_helpfulness(
        input_prompt=input_prompt,
        actual_output=actual_output
    )

    # ---------------------------------------------------
    # 4. MultimodalAnswerRelevancy
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("4. MultimodalAnswerRelevancy")
    print("-" * 40)

    question = "What does a software developer's workspace look like?"
    answer_with_images = [
        "A software developer's workspace typically includes a computer or laptop, multiple monitors displaying code, and other tech accessories.",
        str(original),
        "Here's an edited image showing a more organized and modern workspace.",
        str(edited)
    ]

    result = evaluate_multimodal_relevancy(
        question=question,
        answer_with_images=answer_with_images
    )
