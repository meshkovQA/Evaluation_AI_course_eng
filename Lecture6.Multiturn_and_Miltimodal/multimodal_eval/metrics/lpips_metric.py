"""
LPIPS (Learned Perceptual Image Patch Similarity) metric

LPIPS measures perceptual similarity between two images,
using deep features from pretrained networks.

Value range: 0-1 (LOWER values mean more similar images)
- 0.0 = identical images
- ~0.5 = moderately different images
- >0.7 = very different images
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
# Add parent directory to path (for direct execution)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_loader import preprocess_for_lpips  # noqa: E402

from typing import Union, Optional  # noqa: E402
import lpips  # noqa: E402
import torch  # noqa: E402


# Global cache for model (lazy initialization)
_lpips_model: Optional[lpips.LPIPS] = None
_current_net: Optional[str] = None


def _get_lpips_model(net: str = "squeeze") -> lpips.LPIPS:

    global _lpips_model, _current_net

    if _lpips_model is None or _current_net != net:
        print(f"📥 Loading LPIPS model (network: {net})")
        _lpips_model = lpips.LPIPS(net=net)
        _lpips_model.eval()  # Inference mode
        _current_net = net
        print("✅ LPIPS model loaded")

    return _lpips_model


def compute_lpips(
    image1_path: Union[str, Path],
    image2_path: Union[str, Path],
    net: str = "squeeze"
) -> float:

    # Getting model
    loss_fn = _get_lpips_model(net)

    # Preprocess images
    img1_tensor = preprocess_for_lpips(image1_path)
    img2_tensor = preprocess_for_lpips(image2_path)

    # Compute distance
    with torch.no_grad():
        distance = loss_fn(img1_tensor, img2_tensor)

    return distance.item()


def compute_lpips_batch(
    reference_paths: list,
    comparison_paths: list,
    net: str = "squeeze"
) -> list:

    if len(reference_paths) != len(comparison_paths):
        raise ValueError(
            "Number of reference and comparison images must match")

    distances = []
    for ref_path, cmp_path in zip(reference_paths, comparison_paths):
        dist = compute_lpips(ref_path, cmp_path, net)
        distances.append(dist)

    return distances


def interpret_lpips(distance: float) -> str:

    if distance < 0.05:
        return "Nearly identical images"
    elif distance < 0.1:
        return "Very similar images"
    elif distance < 0.2:
        return "Similar images with minor differences"
    elif distance < 0.4:
        return "Noticeable differences between images"
    elif distance < 0.6:
        return "Significant differences"
    else:
        return "Very different images"


def lpips_to_similarity(distance: float) -> float:

    return 1.0 - min(distance, 1.0)


# Testing when started directly
if __name__ == "__main__":
    print("=" * 50)
    print("Testing LPIPS metrics")
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
    # 1. Identical images
    # ---------------------------------------------------
    print("\n1. Comparing identical images:")
    print("-" * 40)
    dist_identical = compute_lpips(original, original)
    print(f"   LPIPS Distance: {dist_identical:.4f}")
    print(f"   Interpretation: {interpret_lpips(dist_identical)}")
    print(f"   Similarity: {lpips_to_similarity(dist_identical):.2%}")

    # ---------------------------------------------------
    # 2. Similar images (same person, different angle)
    # ---------------------------------------------------
    print("\n2. Similar images (original vs edited — same person):")
    print("-" * 40)
    dist_similar = compute_lpips(original, edited)
    print(f"   LPIPS Distance: {dist_similar:.4f}")
    print(f"   Interpretation: {interpret_lpips(dist_similar)}")
    print(f"   Similarity: {lpips_to_similarity(dist_similar):.2%}")

    # ---------------------------------------------------
    # 3. Different images (different people)
    # ---------------------------------------------------
    print("\n3. Different images (original vs different — different person):")
    print("-" * 40)
    dist_different = compute_lpips(original, different)
    print(f"   LPIPS Distance: {dist_different:.4f}")
    print(f"   Interpretation: {interpret_lpips(dist_different)}")
    print(f"   Similarity: {lpips_to_similarity(dist_different):.2%}")

    print("\n✅ Testing completed!")
    print("\n📝 Note:")
    print("   LPIPS 0-1, LOWER values mean more similar images")
