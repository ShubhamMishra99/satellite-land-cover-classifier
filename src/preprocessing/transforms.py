"""Preprocessing and data augmentation pipelines for EuroSAT RGB imagery.

This module provides torchvision transformation pipelines tailored specifically
for satellite remote-sensing imagery:
1. Training transforms: Conservative geometric augmentations (horizontal flip,
   vertical flip, orthogonal rotations) and mild color jitter to prevent overfitting
   while preserving land-cover semantic integrity.
2. Validation & Test transforms: Completely deterministic pipelines (Resize, ToTensor,
   and dataset normalization).
3. Inference transforms: Identical to evaluation transforms for single/batch prediction.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torchvision.transforms as T

from src.constants import (
    EUROSAT_MEAN,
    EUROSAT_STD,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class RandomDiscreteRotation:
    """Rotate image by one of the specified discrete angles (e.g., 0, 90, 180, 270 degrees).

    In overhead Earth observation and satellite imagery, there is no canonical 'up'
    orientation, making orthogonal 90-degree rotations physically meaningful and
    distortion-free.
    """

    def __init__(self, angles: Sequence[int] = (0, 90, 180, 270)):
        self.angles = tuple(angles)

    def __call__(self, img: Any) -> Any:
        idx = int(torch.randint(0, len(self.angles), (1,)).item())
        angle = self.angles[idx]
        if angle == 0:
            return img
        return T.functional.rotate(img, angle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles})"


def get_normalization_stats(
    norm_type: str = "eurosat",
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Retrieve mean and standard deviation for normalization.

    Args:
        norm_type: One of 'eurosat' (empirical dataset statistics) or 'imagenet'
                   (standard ImageNet statistics for transfer learning).

    Returns:
        Tuple of (mean, std), each being a 3-element tuple of floats.

    Raises:
        ValueError: If norm_type is unrecognized.
    """
    normalized_type = norm_type.lower().strip()
    if normalized_type == "eurosat":
        return EUROSAT_MEAN, EUROSAT_STD
    elif normalized_type == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    else:
        raise ValueError(
            f"Unknown normalization type '{norm_type}'. Expected 'eurosat' or 'imagenet'."
        )


def get_train_transforms(
    image_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    norm_type: str = "eurosat",
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotation_angles: Optional[Sequence[int]] = (0, 90, 180, 270),
    color_jitter: Optional[Dict[str, float]] = None,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> T.Compose:
    """Build the data augmentation transform pipeline for training.

    Conservative augmentations are applied to respect land-cover semantics.

    Args:
        image_size: Target image dimensions (height, width).
        norm_type: Normalization preset ('eurosat' or 'imagenet').
        horizontal_flip_prob: Probability of random horizontal flip.
        vertical_flip_prob: Probability of random vertical flip.
        rotation_angles: Sequence of discrete rotation angles (e.g. [0, 90, 180, 270]).
        color_jitter: Optional dict with keys 'brightness', 'contrast', 'saturation', 'hue'.
        mean: Optional explicit mean tuple (overrides norm_type if provided).
        std: Optional explicit std tuple (overrides norm_type if provided).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list = [
        T.Resize(image_size, antialias=True),
    ]

    if horizontal_flip_prob > 0:
        transform_list.append(T.RandomHorizontalFlip(p=horizontal_flip_prob))

    if vertical_flip_prob > 0:
        transform_list.append(T.RandomVerticalFlip(p=vertical_flip_prob))

    if rotation_angles and len(rotation_angles) > 1:
        transform_list.append(RandomDiscreteRotation(angles=rotation_angles))

    if color_jitter:
        transform_list.append(
            T.ColorJitter(
                brightness=color_jitter.get("brightness", 0.0),
                contrast=color_jitter.get("contrast", 0.0),
                saturation=color_jitter.get("saturation", 0.0),
                hue=color_jitter.get("hue", 0.0),
            )
        )

    transform_list.append(T.ToTensor())

    if mean is None or std is None:
        calc_mean, calc_std = get_normalization_stats(norm_type)
        mean = mean or calc_mean
        std = std or calc_std

    transform_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transform_list)


def get_eval_transforms(
    image_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    norm_type: str = "eurosat",
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> T.Compose:
    """Build the deterministic transform pipeline for validation and evaluation.

    Contains no stochastic data augmentation.

    Args:
        image_size: Target image dimensions (height, width).
        norm_type: Normalization preset ('eurosat' or 'imagenet').
        mean: Optional explicit mean tuple (overrides norm_type if provided).
        std: Optional explicit std tuple (overrides norm_type if provided).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if mean is None or std is None:
        calc_mean, calc_std = get_normalization_stats(norm_type)
        mean = mean or calc_mean
        std = std or calc_std

    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
    norm_type: str = "eurosat",
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> T.Compose:
    """Build deterministic transform pipeline for inference and API serving.

    Equivalent to evaluation transforms.
    """
    return get_eval_transforms(image_size=image_size, norm_type=norm_type, mean=mean, std=std)


def get_transforms(
    split: str,
    config: Optional[Dict[str, Any]] = None,
    norm_type: Optional[str] = None,
) -> T.Compose:
    """Unified transform factory supporting train, val, test, and inference splits.

    Args:
        split: One of 'train', 'val', 'validation', 'test', 'inference', 'eval'.
        config: Optional project configuration dictionary.
        norm_type: Optional normalization type ('eurosat' or 'imagenet'). If not provided,
                   read from config or defaults to 'eurosat'.

    Returns:
        torchvision.transforms.Compose instance.
    """
    split_norm = split.lower().strip()
    if split_norm not in {"train", "val", "validation", "test", "inference", "eval"}:
        raise ValueError(
            f"Invalid split name '{split}'. Supported: 'train', 'val', 'validation', 'test', 'inference', 'eval'."
        )

    # Extract image size from config if present
    image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    if config and "dataset" in config and "image_size" in config["dataset"]:
        cfg_size = config["dataset"]["image_size"]
        if isinstance(cfg_size, (list, tuple)) and len(cfg_size) == 2:
            image_size = (int(cfg_size[0]), int(cfg_size[1]))

    # Determine normalization type
    if norm_type is None:
        norm_type = "eurosat"
        if config and "normalization" in config:
            # If custom mean/std is provided in config, use them
            norm_cfg = config["normalization"]
            if isinstance(norm_cfg, dict) and "mean" in norm_cfg and "std" in norm_cfg:
                custom_mean = tuple(float(x) for x in norm_cfg["mean"])
                custom_std = tuple(float(x) for x in norm_cfg["std"])
                if split_norm == "train":
                    return _get_train_transforms_from_config(config, image_size, custom_mean, custom_std)
                else:
                    return get_eval_transforms(image_size=image_size, mean=custom_mean, std=custom_std)

    if split_norm == "train":
        if config and "augmentation" in config and "train" in config["augmentation"]:
            return _get_train_transforms_from_config(config, image_size, norm_type=norm_type)
        return get_train_transforms(image_size=image_size, norm_type=norm_type)
    else:
        return get_eval_transforms(image_size=image_size, norm_type=norm_type)


def _get_train_transforms_from_config(
    config: Dict[str, Any],
    image_size: Tuple[int, int],
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    norm_type: str = "eurosat",
) -> T.Compose:
    """Helper to parse augmentation parameters from config dictionary."""
    aug_cfg = config.get("augmentation", {}).get("train", {})
    h_flip = aug_cfg.get("random_horizontal_flip_prob", 0.5)
    v_flip = aug_cfg.get("random_vertical_flip_prob", 0.5)
    rot_angles = aug_cfg.get("random_rotation_degrees", [0, 90, 180, 270])
    jitter = aug_cfg.get("color_jitter", None)

    return get_train_transforms(
        image_size=image_size,
        norm_type=norm_type,
        horizontal_flip_prob=h_flip,
        vertical_flip_prob=v_flip,
        rotation_angles=rot_angles,
        color_jitter=jitter,
        mean=mean,
        std=std,
    )
