"""Unit tests for EuroSAT preprocessing transform pipelines and data augmentation."""

import numpy as np
from PIL import Image
import pytest
import torch

from src.constants import (
    EUROSAT_MEAN,
    EUROSAT_STD,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.preprocessing.transforms import (
    RandomDiscreteRotation,
    get_eval_transforms,
    get_inference_transforms,
    get_normalization_stats,
    get_train_transforms,
    get_transforms,
)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Generate a deterministic synthetic 64x64 RGB PIL Image."""
    np.random.seed(42)
    arr = np.random.randint(0, 256, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_normalization_stats_retrieval():
    """Verify normalization statistics for EuroSAT and ImageNet."""
    eurosat_mean, eurosat_std = get_normalization_stats("eurosat")
    assert eurosat_mean == EUROSAT_MEAN
    assert eurosat_std == EUROSAT_STD

    imagenet_mean, imagenet_std = get_normalization_stats("imagenet")
    assert imagenet_mean == IMAGENET_MEAN
    assert imagenet_std == IMAGENET_STD

    with pytest.raises(ValueError, match="Unknown normalization type"):
        get_normalization_stats("invalid_norm")


def test_eval_transforms_output(sample_pil_image: Image.Image):
    """Verify evaluation transforms output shape, type, and value range."""
    transform = get_eval_transforms(norm_type="eurosat")
    tensor = transform(sample_pil_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert tensor.dtype == torch.float32


def test_eval_transforms_deterministic(sample_pil_image: Image.Image):
    """Verify that evaluation/test transforms are strictly deterministic across multiple runs."""
    transform = get_eval_transforms(norm_type="eurosat")
    t1 = transform(sample_pil_image)
    t2 = transform(sample_pil_image)

    assert torch.equal(t1, t2), "Evaluation transforms produced non-deterministic results!"


def test_inference_transforms_matches_eval(sample_pil_image: Image.Image):
    """Verify inference transforms are identical to evaluation transforms."""
    eval_tf = get_eval_transforms(norm_type="eurosat")
    inf_tf = get_inference_transforms(norm_type="eurosat")

    t_eval = eval_tf(sample_pil_image)
    t_inf = inf_tf(sample_pil_image)

    assert torch.equal(t_eval, t_inf)


def test_train_transforms_output(sample_pil_image: Image.Image):
    """Verify training transform pipeline output shape and type."""
    transform = get_train_transforms(
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        rotation_angles=(0, 90, 180, 270),
        color_jitter={"brightness": 0.1, "contrast": 0.1, "saturation": 0.1},
    )
    tensor = transform(sample_pil_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert tensor.dtype == torch.float32


def test_random_discrete_rotation(sample_pil_image: Image.Image):
    """Verify RandomDiscreteRotation transforms image without altering spatial dimensions."""
    rot = RandomDiscreteRotation(angles=(0, 90, 180, 270))
    for _ in range(10):
        rotated = rot(sample_pil_image)
        assert rotated.size == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_get_transforms_factory(sample_pil_image: Image.Image):
    """Verify the unified get_transforms factory for all partition keys."""
    for split_name in ["train", "val", "validation", "test", "inference", "eval"]:
        tf = get_transforms(split=split_name)
        out = tf(sample_pil_image)
        assert out.shape == (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

    with pytest.raises(ValueError, match="Invalid split name"):
        get_transforms("non_existent_split")


def test_get_transforms_with_config(sample_pil_image: Image.Image):
    """Verify get_transforms correctly interprets project configuration dictionary."""
    mock_config = {
        "dataset": {"image_size": [64, 64]},
        "normalization": {
            "mean": [0.3444, 0.3803, 0.4078],
            "std": [0.2026, 0.1368, 0.1156],
        },
        "augmentation": {
            "train": {
                "random_horizontal_flip_prob": 0.5,
                "random_vertical_flip_prob": 0.5,
                "random_rotation_degrees": [0, 90, 180, 270],
                "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1},
            }
        },
    }

    train_tf = get_transforms("train", config=mock_config)
    val_tf = get_transforms("val", config=mock_config)

    t_train = train_tf(sample_pil_image)
    t_val = val_tf(sample_pil_image)

    assert t_train.shape == (3, 64, 64)
    assert t_val.shape == (3, 64, 64)
