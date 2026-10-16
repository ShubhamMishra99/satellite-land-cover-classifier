"""Model checkpointing and serialization utilities for EuroSAT classifiers."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.constants import MODELS_DIR

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    best_val_macro_f1: float,
    config: Dict[str, Any],
    filepath: Union[str, Path],
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    history: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model checkpoint and training state to disk.

    Args:
        model: PyTorch model instance.
        epoch: Current training epoch index.
        best_val_macro_f1: Best validation Macro F1 score achieved so far.
        config: Configuration dictionary used for training.
        filepath: Destination path (e.g., 'models/custom_cnn_best.pt').
        optimizer: Optional optimizer to save state dict for resumption.
        scheduler: Optional learning rate scheduler to save state dict.
        history: Optional training metrics history dictionary.
        extra_metadata: Optional dictionary of additional metadata tags.

    Returns:
        Resolved Path to saved checkpoint file.
    """
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap model if wrapped in DataParallel / DistributedDataParallel
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model_to_save.state_dict(),
        "model_architecture": model_to_save.__class__.__name__,
        "epoch": epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "config": config,
        "history": history or {},
        "timestamp": datetime.now().isoformat(),
        "extra_metadata": extra_metadata or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint (Epoch {epoch}, Best Val Macro F1: {best_val_macro_f1:.4f}) -> {save_path}")
    return save_path


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load model checkpoint weights and training state from disk.

    Args:
        filepath: Path to checkpoint file (.pt or .pth).
        model: PyTorch model instance into which weights will be loaded.
        optimizer: Optional optimizer to restore state dict into.
        scheduler: Optional scheduler to restore state dict into.
        device: Device to map tensors onto (default: current device or CPU).

    Returns:
        Full checkpoint dictionary with metadata.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If weight shape mismatch occurs.
    """
    load_path = Path(filepath)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {load_path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(load_path, map_location=map_location, weights_only=False)

    # Unwrap model if needed
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model weights from: {load_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Restored optimizer state from checkpoint.")

    if scheduler is not None and "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored scheduler state from checkpoint.")

    return checkpoint
