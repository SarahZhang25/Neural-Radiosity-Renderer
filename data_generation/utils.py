"""
Utility functions for radiosity rendering.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import os


def tone_map_reinhard(image: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """
    Apply Reinhard tone mapping to HDR image.

    Args:
        image: HDR image array (any shape, values can exceed 1)
        exposure: Exposure multiplier

    Returns:
        Tone-mapped image in [0, 1] range
    """
    scaled = image * exposure
    return scaled / (1.0 + scaled)


def tone_map_gamma(image: np.ndarray, gamma: float = 2.2, exposure: float = 1.0) -> np.ndarray:
    """
    Apply simple gamma correction tone mapping.

    Args:
        image: HDR image array
        gamma: Gamma value (typically 2.2 for sRGB)
        exposure: Exposure multiplier

    Returns:
        Tone-mapped image in [0, 1] range
    """
    scaled = np.clip(image * exposure, 0, None)
    return np.power(scaled, 1.0 / gamma)


def tone_map(image: np.ndarray, method: str = "reinhard", **kwargs) -> np.ndarray:
    """
    Apply tone mapping with specified method.

    Args:
        image: HDR image array
        method: "reinhard" or "gamma"
        **kwargs: Additional arguments for the specific method

    Returns:
        Tone-mapped image in [0, 1] range
    """
    if method == "reinhard":
        return tone_map_reinhard(image, **kwargs)
    elif method == "gamma":
        return tone_map_gamma(image, **kwargs)
    else:
        raise ValueError(f"Unknown tone mapping method: {method}")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert float image [0, 1] to uint8 [0, 255].

    Args:
        image: Float image in [0, 1] range

    Returns:
        uint8 image
    """
    # TODO: remove these debug messages after fixing this...
    nan_count = np.isnan(image).sum()
    posinf_count = np.isposinf(image).sum()
    neginf_count = np.isneginf(image).sum()
    
    if nan_count > 0 or posinf_count > 0 or neginf_count > 0:
        print(f"[to_uint8] Cleaning image: {nan_count} NaNs, {posinf_count} +Infs, {neginf_count} -Infs")
        

    clean_image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    clipped = np.clip(clean_image, 0, 1)
    return (clipped * 255).astype(np.uint8)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1/2.4) - a)


def save_image(image: np.ndarray, path: str, tone_mapping: str = "reinhard", exposure: float = 1.0):
    """
    Save image to file with tone mapping.

    Args:
        image: HDR image array (height, width, 3)
        path: Output file path
        tone_mapping: Tone mapping method ("reinhard", "gamma", or "none")
        exposure: Exposure value for tone mapping
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Apply tone mapping
    if tone_mapping == "none":
        processed = np.clip(image * exposure, 0, 1)  # or just image if already [0,1]
    else:
        processed = tone_map(image, method=tone_mapping, exposure=exposure)

    # Clamp to [0, 1]
    processed = np.clip(processed, 0, 1)
    
    processed = linear_to_srgb(processed)

    # # Apply sRGB gamma correction (linear -> display)
    # processed = np.power(image, 1.0 / 2.2)

    # Convert to uint8
    img_uint8 = to_uint8(processed)

    # Save using available library
    try:
        from PIL import Image
        pil_image = Image.fromarray(img_uint8)
        pil_image.save(path)
    except ImportError:
        try:
            import imageio
            imageio.imwrite(path, img_uint8)
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(path, img_uint8)


def save_contributions(
    contributions: Dict[Tuple[int, int], int],
    path: str,
    image_shape: Tuple[int, int] = None
):
    """
    Save per-pixel contribution data to NPZ file.

    The NPZ file contains:
    - pixel_coords: (N, 2) array of (x, y) coordinates
    - patch_ids: (N,) array of patch IDs

    Args:
        contributions: Dict mapping (x, y) -> patch_id
        path: Output NPZ file path
        image_shape: Optional (height, width) for full contribution map
    """
    if not contributions:
        np.savez(path, pixel_coords=np.array([]), patch_ids=np.array([]))
        return

    pixel_coords = np.array(list(contributions.keys()))
    patch_ids = np.array(list(contributions.values()))

    # Optionally create a full contribution map
    if image_shape is not None:
        contribution_map = np.full(image_shape, -1, dtype=np.int32)
        for (x, y), patch_id in contributions.items():
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                contribution_map[y, x] = patch_id
        np.savez(path,
                 pixel_coords=pixel_coords,
                 patch_ids=patch_ids,
                 contribution_map=contribution_map)
    else:
        np.savez(path,
                 pixel_coords=pixel_coords,
                 patch_ids=patch_ids)


def load_contributions(path: str) -> Dict[Tuple[int, int], int]:
    """
    Load per-pixel contribution data from NPZ file.

    Args:
        path: NPZ file path

    Returns:
        Dict mapping (x, y) -> patch_id
    """
    data = np.load(path)
    pixel_coords = data['pixel_coords']
    patch_ids = data['patch_ids']

    contributions = {}
    for i in range(len(pixel_coords)):
        x, y = pixel_coords[i]
        contributions[(x, y)] = patch_ids[i]

    return contributions


def visualize_patch_contributions(
    contributions: Dict[Tuple[int, int], int],
    n_patches: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Create a visualization of which patches contribute to each pixel.

    Args:
        contributions: Dict mapping (x, y) -> patch_id
        n_patches: Total number of patches (for color mapping)
        width: Image width
        height: Image height

    Returns:
        RGB image where each patch has a unique color
    """
    # Generate unique colors for each patch
    np.random.seed(42)
    colors = np.random.rand(n_patches + 1, 3)
    colors[0] = [0, 0, 0]  # Background/no contribution

    image = np.zeros((height, width, 3))
    for (x, y), patch_id in contributions.items():
        if 0 <= y < height and 0 <= x < width:
            color_idx = patch_id + 1 if patch_id >= 0 else 0
            image[y, x] = colors[color_idx % len(colors)]

    return image


def print_progress(iteration: int, total_energy: float, prefix: str = "Radiosity"):
    """
    Print progress information.

    Args:
        iteration: Current iteration number
        total_energy: Remaining unshot energy
        prefix: Prefix string for the output
    """
    print(f"\r{prefix}: iteration {iteration:4d}, unshot energy: {total_energy:.6f}", end="", flush=True)
