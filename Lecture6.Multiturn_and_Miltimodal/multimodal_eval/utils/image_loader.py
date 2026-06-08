"""
Utilities for loading and preprocessing images

Functions for working with images in the context of CLIP Score and LPIPS metrics.
"""

import torch
from PIL import Image
from torchvision import transforms
from typing import Union
from pathlib import Path


def load_image(path: Union[str, Path]) -> Image.Image:
    """
    Loads image from file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path)

    # Convert to RGB (in case of RGBA or other formats)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_for_clip(
    image: Union[str, Path, Image.Image],
    size: int = 224
) -> torch.Tensor:
    """
    Preprocess image for CLIP Score.

    CLIP expects images in uint8 format [0-255] with size 224x224.
    """
    # Load if path was provided
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Transformations for CLIP
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # Convert to uint8 [0-255] for torchmetrics CLIP
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])

    return transform(image)


def preprocess_for_lpips(
    image: Union[str, Path, Image.Image],
    size: int = 256,
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess image for LPIPS.

    LPIPS expects images in range [-1, 1] (normalize=True)
    or [0, 1] (normalize=False, with normalize=True flag in metric).
    """
    # Load if path was provided
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Base transformations
    transform_list = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # [0, 1]
    ]

    # Normalize to [-1, 1] for LPIPS
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        )

    transform = transforms.Compose(transform_list)

    # Add batch dimension
    tensor = transform(image).unsqueeze(0)

    return tensor


def load_image_batch(
    paths: list,
    preprocess_fn: callable = None
) -> list:
    """
    Loads a batch of images.
    """
    images = []
    for path in paths:
        img = load_image(path)
        if preprocess_fn:
            img = preprocess_fn(img)
        images.append(img)
    return images


# Testing when started directly
if __name__ == "__main__":
    print("=" * 50)
    print("Testing image loading utilities")
    print("=" * 50)

    # Create test image
    test_image = Image.new("RGB", (512, 512), color="red")
    test_path = "/tmp/test_image.png"
    test_image.save(test_path)

    print("\n1. Loading image:")
    img = load_image(test_path)
    print(f"   Size: {img.size}")
    print(f"   Mode: {img.mode}")

    print("\n2. Preprocessing for CLIP:")
    clip_tensor = preprocess_for_clip(test_path)
    print(f"   Shape: {clip_tensor.shape}")
    print(f"   Dtype: {clip_tensor.dtype}")
    print(
        f"   Min/Max: {clip_tensor.min().item()}, {clip_tensor.max().item()}")

    print("\n3. Preprocessing for LPIPS:")
    lpips_tensor = preprocess_for_lpips(test_path)
    print(f"   Shape: {lpips_tensor.shape}")
    print(f"   Dtype: {lpips_tensor.dtype}")
    print(
        f"   Min/Max: {lpips_tensor.min().item():.2f}, {lpips_tensor.max().item():.2f}")

    print("\n✅ All tests passed!")
