"""
DeepEval multimodal metrics for image evaluation (DeepEval 4.x API).

Metrics based on an MLLM judge (GPT-4o / gpt-4o-mini by default):
- ImageCoherence:   how well images fit the surrounding text
- ImageHelpfulness: how much images help the user understand the text
- ImageReference:   whether the text properly references the images
- TextToImage:      quality of an image generated from a prompt

Since DeepEval 4.x the multimodal API changed:
- `MLLMTestCase` was removed; use the regular `LLMTestCase` instead.
- `input` is a plain string, `actual_output` is a single string in which images
  are embedded as `MLLMImage` placeholders (metrics parse them back out and
  analyse the text surrounding each image).
- `MLLMImage` auto-detects local vs. remote and loads the image on creation.

All metrics return a score in the range 0-1 (higher is better).
"""

from pathlib import Path
from typing import List, Union

# DeepEval imports (4.x)
from deepeval.test_case import LLMTestCase, MLLMImage
from deepeval.metrics import (
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
    TextToImageMetric,
)


def _is_image_source(item) -> bool:
    """True if `item` points to an image (existing local file or http(s) URL)."""
    if isinstance(item, MLLMImage):
        return True
    source = str(item)
    if source.startswith(("http://", "https://")):
        return True
    try:
        return Path(source).is_file()
    except OSError:
        return False


def _to_mllm_image(source: Union[str, Path, MLLMImage]) -> MLLMImage:
    """
    Build an MLLMImage from a path/URL/MLLMImage.

    In DeepEval 4.x `local` is auto-detected, but we pass it explicitly for
    clarity. Local paths are resolved to absolute so evaluation works regardless
    of the current working directory.
    """
    if isinstance(source, MLLMImage):
        return source

    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        return MLLMImage(url=source_str, local=False)

    abs_path = str(Path(source_str).resolve())
    return MLLMImage(url=abs_path, local=True)


def _build_multimodal_output(items: List) -> str:
    """
    Interleave text and images into a single `actual_output` string.

    Accepts a list mixing plain text (str) and images (file paths, URLs or
    MLLMImage objects). Each image is inserted as its DeepEval placeholder
    (via `str(MLLMImage(...))`), so the judge can read the text above/below it.

    Example:
        ["Here is the workspace:", "img.png", "That is the setup."]
        -> "Here is the workspace:\\n[DEEPEVAL:IMAGE:...]\\nThat is the setup."
    """
    parts: List[str] = []
    for item in items:
        if _is_image_source(item):
            parts.append(str(_to_mllm_image(item)))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _metric_result(metric) -> dict:
    """Extract a clean, serialisable result from a measured metric."""
    return {
        "metric": metric.__name__,
        "score": metric.score,
        "reason": metric.reason,
        "threshold": metric.threshold,
        "success": metric.is_successful(),
    }


def _measure_single_turn(metric, input_prompt: str, actual_output: List) -> dict:
    """Shared flow for image-in-context metrics (coherence/helpfulness/reference)."""
    test_case = LLMTestCase(
        input=input_prompt,
        actual_output=_build_multimodal_output(actual_output),
    )
    metric.measure(test_case)
    return _metric_result(metric)


def evaluate_image_coherence(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """Evaluate how coherently each image fits its surrounding text."""
    metric = ImageCoherenceMetric(model=model, threshold=threshold)
    return _measure_single_turn(metric, input_prompt, actual_output)


def evaluate_image_helpfulness(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """Evaluate how much each image helps the user understand the text."""
    metric = ImageHelpfulnessMetric(model=model, threshold=threshold)
    return _measure_single_turn(metric, input_prompt, actual_output)


def evaluate_image_reference(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """Evaluate whether the text correctly references the accompanying images."""
    metric = ImageReferenceMetric(model=model, threshold=threshold)
    return _measure_single_turn(metric, input_prompt, actual_output)


def evaluate_text_to_image(
    prompt: str,
    generated_image_path: Union[str, Path, MLLMImage],
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Evaluate a generated image against its prompt.

    TextToImageMetric requires exactly 0 images in `input` (plain-text prompt)
    and exactly 1 image in `actual_output`.
    """
    image = _to_mllm_image(generated_image_path)
    test_case = LLMTestCase(input=prompt, actual_output=str(image))

    metric = TextToImageMetric(model=model, threshold=threshold)
    metric.measure(test_case)
    return _metric_result(metric)


def _print_result(result: dict) -> None:
    """Pretty-print a metric result."""
    status = "✅ PASS" if result["success"] else "❌ FAIL"
    print(f"   {result['metric']}: {result['score']:.2f} "
          f"(threshold {result['threshold']}) {status}")
    if result.get("reason"):
        print(f"   Reason: {result['reason']}")


# Testing when started directly
if __name__ == "__main__":
    print("=" * 50)
    print("Testing DeepEval multimodal metrics (DeepEval 4.x)")
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
    # 1. TextToImage
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("1. TextToImageMetric")
    print("-" * 40)

    prompt = "a man with glasses working on a laptop with code on monitors in an office"
    result = evaluate_text_to_image(prompt=prompt, generated_image_path=original)
    _print_result(result)

    # ---------------------------------------------------
    # 2. ImageCoherence
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("2. ImageCoherenceMetric")
    print("-" * 40)

    result = evaluate_image_coherence(
        input_prompt="Show me a software developer at work",
        actual_output=[
            "Here is a software developer at work:",
            str(original),
        ],
    )
    _print_result(result)

    # ---------------------------------------------------
    # 3. ImageHelpfulness
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("3. ImageHelpfulnessMetric")
    print("-" * 40)

    result = evaluate_image_helpfulness(
        input_prompt="Show me what a programmer's workspace looks like",
        actual_output=[
            "A typical programmer's workspace looks like this:",
            str(edited),
        ],
    )
    _print_result(result)

    # ---------------------------------------------------
    # 4. ImageReference
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("4. ImageReferenceMetric")
    print("-" * 40)

    result = evaluate_image_reference(
        input_prompt="What does a software developer's workspace look like?",
        actual_output=[
            "As shown in the image below, a developer's workspace has multiple monitors:",
            str(original),
            "The edited version below shows a cleaner, more modern setup:",
            str(edited),
        ],
    )
    _print_result(result)

    print("\n✅ Testing completed!")
