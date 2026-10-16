"""Batch model evaluation engine and inference latency benchmarking for EuroSAT classifiers.

This module provides reproducible, non-leaking evaluation across the test partition:
- Strict evaluation mode (`model.eval()`).
- Disabled gradient computation (`torch.inference_mode()`).
- Verification that model weights are immutable and remain untouched.
- Batch processing with collection of ground truth, predictions, logits, probabilities, and file paths.
- Accurate inference latency measurement:
  - Cache warm-up passes before timing.
  - CUDA synchronization when executing on GPU.
  - Precise exclusion of checkpoint-loading and dataset initialization overhead.
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.constants import EUROSAT_CLASSES, MODELS_DIR, NUM_CLASSES
from src.inference.predictor import load_trained_model

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured container for model evaluation predictions, timing, and metadata."""

    model_name: str
    checkpoint_path: str
    device: str
    num_samples: int
    batch_size: int
    total_parameters: int
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    image_paths: List[str]
    relative_paths: List[str]
    inference_time_total_sec: float
    inference_time_per_image_ms: float
    timestamp: str

    def get_confidence_scores(self) -> np.ndarray:
        """Return maximum predicted probability (confidence) for each sample."""
        return np.max(self.probabilities, axis=1)

    def get_is_correct(self) -> np.ndarray:
        """Return boolean array indicating whether each sample was correctly classified."""
        return self.y_true == self.y_pred


def compute_model_parameter_hash(model: nn.Module) -> str:
    """Compute SHA-256 hash of all model parameter bytes to verify weight immutability."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for param in model.parameters():
            hasher.update(param.cpu().numpy().tobytes())
    return hasher.hexdigest()


class ModelEvaluator:
    """Evaluation runner for EuroSAT classifiers on batched PyTorch DataLoaders."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize ModelEvaluator.

        Args:
            model: PyTorch model to evaluate.
            model_name: Identifier for model (e.g. 'custom_cnn' or 'resnet18').
            device: Target execution device (defaults to current model device or CPU).
            checkpoint_path: Optional path to checkpoint file for metadata logging.
        """
        self.model = model
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else "in_memory"

        if device is not None:
            self.device = device
            self.model.to(self.device)
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        # Explicitly ensure eval mode
        self.model.eval()

        # Count total parameters
        self.total_parameters = sum(p.numel() for p in self.model.parameters())

    def _warmup(self, warmup_batches: int = 5, input_shape: Tuple[int, ...] = (16, 3, 64, 64)) -> None:
        """Execute dummy forward passes to warm up hardware caches, memory allocators, and JIT kernels."""
        if warmup_batches <= 0:
            return

        dummy_tensor = torch.zeros(input_shape, device=self.device)
        self.model.eval()

        with torch.inference_mode():
            for _ in range(warmup_batches):
                _ = self.model(dummy_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def evaluate(
        self,
        dataloader: DataLoader,
        warmup_batches: int = 5,
    ) -> EvaluationResult:
        """Evaluate the model over the full DataLoader in batches.

        Guarantees:
        - `model.eval()` is enforced throughout.
        - `torch.inference_mode()` disables gradient tracking and autograd bookkeeping.
        - Model weights are verified to be immutable (hash before == hash after).
        - Latency timing is measured accurately for pure inference.

        Args:
            dataloader: PyTorch DataLoader yielding batches of test data.
            warmup_batches: Number of warm-up forward passes before timing.

        Returns:
            EvaluationResult dataclass holding predictions, timing, and metadata.
        """
        # Ensure eval mode
        self.model.eval()
        if self.model.training:
            raise RuntimeError(f"Model {self.model_name} is still in training mode! Must be in eval() mode.")

        # Record parameter hash before evaluation to prove zero weight modification
        pre_eval_hash = compute_model_parameter_hash(self.model)

        # Hardware warm-up pass
        batch_size = getattr(dataloader, "batch_size", 64) or 64
        self._warmup(warmup_batches=warmup_batches, input_shape=(min(batch_size, 16), 3, 64, 64))

        all_y_true: List[np.ndarray] = []
        all_y_pred: List[np.ndarray] = []
        all_logits: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        all_paths: List[str] = []
        all_rel_paths: List[str] = []

        total_forward_time_sec = 0.0

        with torch.inference_mode():
            for batch in dataloader:
                # Handle both (images, labels, metadata) and (images, labels)
                if len(batch) == 3:
                    images, labels, metadata = batch
                    if isinstance(metadata, dict):
                        if "path" in metadata:
                            paths = metadata["path"]
                            all_paths.extend(paths if isinstance(paths, list) else list(paths))
                        if "relative_path" in metadata:
                            rel_paths = metadata["relative_path"]
                            all_rel_paths.extend(rel_paths if isinstance(rel_paths, list) else list(rel_paths))
                elif len(batch) == 2:
                    images, labels = batch
                else:
                    raise ValueError(f"Unexpected batch format with length {len(batch)}")

                images = images.to(self.device, non_blocking=True)

                # Synchronize CUDA before starting timing for this batch
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                batch_start = time.perf_counter()
                logits = self.model(images)

                # Synchronize CUDA after forward pass for exact timing
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                batch_elapsed = time.perf_counter() - batch_start
                total_forward_time_sec += batch_elapsed

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Accumulate on CPU
                all_y_true.append(labels.detach().cpu().numpy())
                all_y_pred.append(preds.detach().cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())

        # Concatenate arrays
        y_true_arr = np.concatenate(all_y_true, axis=0).astype(np.int64)
        y_pred_arr = np.concatenate(all_y_pred, axis=0).astype(np.int64)
        logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
        probs_arr = np.concatenate(all_probs, axis=0).astype(np.float32)

        num_samples = len(y_true_arr)
        if num_samples == 0:
            raise ValueError(f"Evaluation encountered an empty DataLoader (0 samples evaluated).")

        # Fallback for paths if not provided by metadata
        if len(all_paths) != num_samples:
            all_paths = [f"sample_{i}" for i in range(num_samples)]
        if len(all_rel_paths) != num_samples:
            all_rel_paths = [f"sample_{i}" for i in range(num_samples)]

        # Verify model immutability
        post_eval_hash = compute_model_parameter_hash(self.model)
        if pre_eval_hash != post_eval_hash:
            raise AssertionError(
                f"CRITICAL: Model weights were modified during evaluation! Pre-hash: {pre_eval_hash}, Post-hash: {post_eval_hash}"
            )

        # Calculate timing metrics
        time_per_image_ms = (total_forward_time_sec / num_samples) * 1000.0

        logger.info(
            f"Evaluation finished for {self.model_name}: {num_samples} samples evaluated. "
            f"Forward latency: {total_forward_time_sec:.3f} s total ({time_per_image_ms:.2f} ms/image) on {self.device}."
        )

        return EvaluationResult(
            model_name=self.model_name,
            checkpoint_path=self.checkpoint_path,
            device=str(self.device),
            num_samples=num_samples,
            batch_size=batch_size,
            total_parameters=self.total_parameters,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            probabilities=probs_arr,
            logits=logits_arr,
            image_paths=all_paths,
            relative_paths=all_rel_paths,
            inference_time_total_sec=round(total_forward_time_sec, 4),
            inference_time_per_image_ms=round(time_per_image_ms, 3),
            timestamp=datetime.now().isoformat(),
        )


def load_model_for_evaluation(
    model_name: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Path]:
    """Load trained model and checkpoint for evaluation.

    Args:
        model_name: 'custom_cnn' or 'resnet18'.
        checkpoint_path: Optional explicit path to checkpoint.
        device: PyTorch device.
        config: Optional configuration dictionary.

    Returns:
        Tuple of (model in eval mode, resolved checkpoint Path).
    """
    resolved_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    normalized_name = model_name.lower().strip().replace("-", "_")
    if checkpoint_path is None:
        if normalized_name in ("custom_cnn", "customcnn", "baseline", "cnn"):
            resolved_ckpt = MODELS_DIR / "custom_cnn_best.pt"
        elif normalized_name in ("resnet18", "resnet_18", "resnet18_transfer", "resnet"):
            resolved_ckpt = MODELS_DIR / "resnet18_best.pt"
        else:
            raise ValueError(f"Unknown model name '{model_name}'. Expected 'custom_cnn' or 'resnet18'.")
    else:
        resolved_ckpt = Path(checkpoint_path)

    if not resolved_ckpt.exists():
        raise FileNotFoundError(
            f"Model checkpoint for '{model_name}' not found at: {resolved_ckpt}. "
            "Please verify that training completed and saved the model checkpoint."
        )

    model = load_trained_model(
        model_name=model_name,
        checkpoint_path=resolved_ckpt,
        device=resolved_device,
        config=config,
    )
    model.eval()

    return model, resolved_ckpt
