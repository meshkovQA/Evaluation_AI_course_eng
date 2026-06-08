"""
CLIP Score metrics for image evaluation

CLIP Score measures semantic similarity between:
- Image and text (image-text similarity)
- Two images (image-image similarity)

Value range: 0-100 (higher is better)
"""
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_loader import preprocess_for_clip, load_image  # noqa: E402


import torch  # noqa: E402
from torchmetrics.multimodal.clip_score import CLIPScore  # noqa: E402
from typing import Union, Optional  # noqa: E402


# Add parent directory to path (for direct execution)


# Global cache for model (lazy initialization)
_clip_model: Optional[CLIPScore] = None


def _get_clip_model(model_name: str = "openai/clip-vit-base-patch16") -> CLIPScore:

    global _clip_model

    if _clip_model is None:
        print(f"📥 Loading CLIP model: {model_name}")
        _clip_model = CLIPScore(model_name_or_path=model_name)
        # Explicitly set to CPU
        _clip_model = _clip_model.to("cpu")
        print("✅ CLIP model loaded")

    return _clip_model


def compute_clip_score_text(
    image_path: Union[str, Path],
    text: str,
    model_name: str = "openai/clip-vit-base-patch16"
) -> float:

    # Getting model
    metric = _get_clip_model(model_name)

    # Preprocess image
    image_tensor = preprocess_for_clip(image_path)

    # Compute score
    with torch.no_grad():
        score = metric(image_tensor, text)

    return score.detach().item()


def compute_clip_score_images(
    image1_path: Union[str, Path],
    image2_path: Union[str, Path],
    model_name: str = "openai/clip-vit-base-patch16"
) -> float:

    metric = _get_clip_model(model_name)

    # Preprocess images
    image1_tensor = preprocess_for_clip(image1_path)
    image2_tensor = preprocess_for_clip(image2_path)

    # Compute score between images
    with torch.no_grad():
        score = metric(image1_tensor, image2_tensor)

    return score.detach().item()


def compute_clip_score_batch(
    image_paths: list,
    texts: list,
    model_name: str = "openai/clip-vit-base-patch16"
) -> list:

    if len(image_paths) != len(texts):
        raise ValueError("Number of images and texts must match")

    scores = []
    for img_path, text in zip(image_paths, texts):
        score = compute_clip_score_text(img_path, text, model_name)
        scores.append(score)

    return scores


def interpret_clip_score(score: float) -> str:

    if score >= 30:
        return "Excellent match"
    elif score >= 25:
        return "Good match"
    elif score >= 20:
        return "Moderate match"
    elif score >= 15:
        return "Weak match"
    else:
        return "Very weak match"


# Testing when started directly
if __name__ == "__main__":
    print("=" * 50)
    print("Testing CLIP Score metrics")
    print("=" * 50)

    # Use ready-made sample_images
    sample_dir = Path(__file__).parent.parent / "sample_images"

    original = sample_dir / "original.png"
    edited = sample_dir / "edited.png"
    different = sample_dir / "different.png"

    if not original.exists():
        print(f"\n⚠️  Test images not found in {sample_dir}")
        print("   Run: python create_sample_images.py")
        exit(1)

    print(f"\n📁 Using images from {sample_dir}")

    # ---------------------------------------------------
    # 1. CLIP Score (image-text)
    # ---------------------------------------------------
    print("\n1. CLIP Score (image-text):")
    print("-" * 40)

    # Test: correct description
    prompt_correct = "a man with glasses working on a laptop with code on monitors"
    score_match = compute_clip_score_text(original, prompt_correct)
    print(f"   Original + '{prompt_correct}'")
    print(f"   Score: {score_match:.2f} - {interpret_clip_score(score_match)}")

    # Test: incorrect description
    prompt_wrong = "a cat sleeping on a sofa"
    score_wrong = compute_clip_score_text(original, prompt_wrong)
    print(f"\n   Original + '{prompt_wrong}'")
    print(f"   Score: {score_wrong:.2f} - {interpret_clip_score(score_wrong)}")

    # ---------------------------------------------------
    # 2. CLIP Score (image-image)
    # ---------------------------------------------------
    print("\n2. CLIP Score (image-image):")
    print("-" * 40)

    # Similar images (same person, slightly different angle)
    score_similar = compute_clip_score_images(original, edited)
    print(f"   Original vs Edited (same person): {score_similar:.2f}")

    # Different images (different people)
    score_different = compute_clip_score_images(original, different)
    print(f"   Original vs Different (different person): {score_different:.2f}")

    # Identical
    score_same = compute_clip_score_images(original, original)
    print(f"   Identical images: {score_same:.2f}")

    print("\n✅ Testing completed!")
    print("\n📝 Note:")
    print("   CLIP Score 0-100, higher is better match")
