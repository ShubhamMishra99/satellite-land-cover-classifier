"""Inference engine, model loading, and single/batch prediction pipelines for EuroSAT."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn

from src.config import get_base_config, get_custom_cnn_config, get_resnet18_config
from src.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    EUROSAT_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MODELS_DIR,
    NUM_CLASSES,
)
from src.models import build_model
from src.preprocessing.transforms import get_inference_transforms
from src.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DOMAIN_WARNING_MESSAGE = (
    "Note: This model was trained on EuroSAT RGB imagery. Predictions on external "
    "satellite images may be unreliable if their visual characteristics differ substantially from EuroSAT."
)


def validate_image_path(image_path: Union[str, Path]) -> Path:
    """Validate image file existence and supported extension.

    Args:
        image_path: Path to the target image file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file extension is unsupported or file is not a regular file.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found at: {path}")
    if not path.is_file():
        raise ValueError(f"Specified path is a directory, not an image file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. Supported formats: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    return path


def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image],
    transform: Optional[Callable] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Open an image, convert to RGB, apply inference transform, and return input tensor.

    Args:
        image_input: Path to image file or existing PIL Image object.
        transform: Optional torchvision transform pipeline (defaults to standard inference transforms).

    Returns:
        Tuple of (preprocessed_tensor of shape [1, 3, 64, 64], original_dimensions (width, height)).

    Raises:
        ValueError: If image cannot be decoded or converted.
    """
    if isinstance(image_input, (str, Path)):
        validated_path = validate_image_path(image_input)
        try:
            pil_img = Image.open(validated_path)
            # Fully load image data to catch truncation/corruption
            pil_img.load()
        except Exception as exc:
            raise ValueError(f"Failed to open image file '{validated_path}': {exc}") from exc
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}. Expected Path or PIL.Image.")

    original_size = (pil_img.width, pil_img.height)

    # Convert to standard 3-channel RGB (handles RGBA, grayscale, palette, etc.)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Apply deterministic inference transformation
    pipeline = transform if transform is not None else get_inference_transforms()
    tensor = pipeline(pil_img)  # Shape: [3, 64, 64]

    # Add batch dimension -> [1, 3, 64, 64]
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor, original_size


def load_trained_model(
    model_name: str = "resnet18",
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Instantiate model architecture and restore trained weights from checkpoint.

    Args:
        model_name: Model type ('custom_cnn' or 'resnet18').
        checkpoint_path: Optional explicit path to .pt checkpoint.
                         Defaults to 'models/resnet18_best.pt' or 'models/custom_cnn_best.pt'.
        device: PyTorch device (CPU or CUDA).
        config: Optional configuration dictionary.

    Returns:
        Loaded PyTorch model in eval() mode on the specified device.

    Raises:
        FileNotFoundError: If checkpoint file is missing.
        ValueError: If model name is unknown.
    """
    target_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    normalized_name = model_name.lower().strip().replace("-", "_")

    # Resolve default checkpoint path if not provided
    if checkpoint_path is None:
        if normalized_name in ("custom_cnn", "customcnn", "baseline", "cnn"):
            checkpoint_path = MODELS_DIR / "custom_cnn_best.pt"
            merged_cfg = {**get_base_config(), **get_custom_cnn_config(), **(config or {})}
        elif normalized_name in ("resnet18", "resnet_18", "resnet18_transfer", "resnet"):
            checkpoint_path = MODELS_DIR / "resnet18_best.pt"
            merged_cfg = {**get_base_config(), **get_resnet18_config(), **(config or {})}
        else:
            raise ValueError(f"Unknown model name '{model_name}'. Supported: 'custom_cnn', 'resnet18'.")
    else:
        merged_cfg = config or {}

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found for model '{model_name}' at: {ckpt_path}. "
            f"Please ensure the model training milestone has produced this checkpoint."
        )

    # Build model without downloading weights since we load checkpoint state dict
    if normalized_name in ("resnet18", "resnet_18", "resnet18_transfer", "resnet"):
        model = build_model("resnet18", config=merged_cfg, pretrained=False)
    else:
        model = build_model("custom_cnn", config=merged_cfg)

    # Load weights
    load_checkpoint(ckpt_path, model, device=target_device)
    model.to(target_device)
    model.eval()

    logger.info(f"Successfully loaded '{model_name}' model from {ckpt_path} onto {target_device}")
    return model


def predict_single_tensor(
    tensor: torch.Tensor,
    model: nn.Module,
    device: Optional[torch.device] = None,
    top_k: int = 3,
    classes: Sequence[str] = EUROSAT_CLASSES,
    confidence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute forward inference on a preprocessed tensor and return structured predictions.

    Args:
        tensor: Preprocessed tensor of shape [1, 3, 64, 64] or [3, 64, 64].
        model: PyTorch model in eval mode.
        device: Target execution device.
        top_k: Number of top candidate classes to rank (default: 3).
        classes: Sequence of target class names.
        confidence_threshold: Minimum maximum-softmax-probability required to accept prediction.
                              Defaults to DEFAULT_CONFIDENCE_THRESHOLD (0.60).

    Returns:
        Structured prediction dictionary with status, is_confident, threshold_used,
        uncertainty_reason, predicted_class, confidence, top_k, and all probabilities.
    """
    if device is not None:
        target_device = device
    else:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    model.eval()

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(target_device, non_blocking=True)

    with torch.inference_mode():
        logits = model(tensor)
        # Softmax applied exclusively during inference to obtain probabilities
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())

    # Evaluate uncertainty / out-of-distribution threshold
    thresh = DEFAULT_CONFIDENCE_THRESHOLD if confidence_threshold is None else float(confidence_threshold)
    is_confident = bool(confidence >= thresh)
    status = "accepted" if is_confident else "potentially_unsupported"
    uncertainty_reason = (
        None
        if is_confident
        else (
            f"Prediction confidence ({confidence * 100:.2f}%) is below the uncertainty threshold "
            f"({thresh * 100:.2f}%). The image may be visually ambiguous, degraded, or outside "
            "the supported EuroSAT domain."
        )
    )

    # Calculate Top-K
    k = min(top_k, len(classes))
    top_k_probs, top_k_indices = torch.topk(probs, k=k)

    top_k_list = []
    for rank, (p, idx) in enumerate(zip(top_k_probs.tolist(), top_k_indices.tolist()), start=1):
        cls_name = classes[idx]
        top_k_list.append({
            "rank": rank,
            "class": cls_name,
            "probability": round(float(p), 4),
            "percentage": f"{p * 100:.2f}%",
        })

    all_probabilities = {
        classes[i]: round(float(probs[i].item()), 4) for i in range(len(classes))
    }

    return {
        "status": status,
        "is_confident": is_confident,
        "threshold_used": thresh,
        "uncertainty_reason": uncertainty_reason,
        "predicted_class": classes[pred_idx],
        "predicted_index": pred_idx,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "top3": top_k_list,
        "all_probabilities": all_probabilities,
    }


def predict_image(
    image_input: Union[str, Path, Image.Image],
    model: nn.Module,
    device: Optional[torch.device] = None,
    top_k: int = 3,
    classes: Sequence[str] = EUROSAT_CLASSES,
    transform: Optional[Callable] = None,
    confidence_threshold: Optional[float] = None,
) -> Tuple[Dict[str, Any], Tuple[int, int]]:
    """End-to-end single image prediction wrapper.

    Args:
        image_input: Path to image file or PIL Image object.
        model: Loaded PyTorch model.
        device: Target device.
        top_k: Number of top candidate classes to rank.
        classes: Sequence of target class names.
        transform: Optional transform pipeline.
        confidence_threshold: Optional uncertainty threshold.

    Returns:
        Tuple of (prediction_dict, original_dimensions).
    """
    tensor, original_size = load_and_preprocess_image(image_input, transform=transform)
    results = predict_single_tensor(
        tensor=tensor,
        model=model,
        device=device,
        top_k=top_k,
        classes=classes,
        confidence_threshold=confidence_threshold,
    )
    return results, original_size
