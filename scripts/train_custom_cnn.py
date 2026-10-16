"""CLI training script for Custom CNN Baseline on EuroSAT RGB dataset."""

import argparse
import logging
from pathlib import Path
import random
import sys
import numpy as np
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_base_config, get_custom_cnn_config
from src.constants import MODELS_DIR
from src.data.dataloader import create_dataloaders
from src.models import build_model
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_custom_cnn")


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and PyTorch for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Custom CNN Baseline on EuroSAT RGB.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Path to save best model checkpoint")
    parser.add_argument("--history-path", type=Path, default=None, help="Path to save training history JSON")

    args = parser.parse_args()

    print("=" * 70)
    print("   SATELLITE LAND-USE CLASSIFIER - BASELINE MODEL TRAINING")
    print("                    Custom CNN Architecture")
    print("=" * 70)

    # 1. Load Configurations
    base_cfg = get_base_config()
    cnn_cfg = get_custom_cnn_config()

    # Merge configuration
    merged_cfg = {**base_cfg, **cnn_cfg}
    seed = args.seed if args.seed is not None else merged_cfg.get("seed", 42)
    set_seed(seed)

    # Apply overrides
    if args.epochs is not None:
        merged_cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        merged_cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        merged_cfg.setdefault("training", {})["learning_rate"] = args.lr
    if args.checkpoint_path is not None:
        merged_cfg.setdefault("artifacts", {})["best_model_path"] = str(args.checkpoint_path)
    if args.history_path is not None:
        merged_cfg.setdefault("artifacts", {})["metrics_path"] = str(args.history_path)

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = merged_cfg.get("training", {}).get("batch_size", 64)
    epochs = merged_cfg.get("training", {}).get("epochs", 30)
    lr = merged_cfg.get("training", {}).get("learning_rate", 0.001)

    print(f"\n[CONFIG] Training Parameters:")
    print(f"  Device           : {device}")
    print(f"  Random Seed      : {seed}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Learning Rate    : {lr}")
    print(f"  Optimizer        : AdamW (Weight Decay: {merged_cfg.get('training', {}).get('weight_decay', 0.0001)})")
    print(f"  LR Scheduler     : ReduceLROnPlateau (Mode: max, Monitor: Val Macro F1)")
    print(f"  Early Stopping   : Patience {merged_cfg.get('training', {}).get('early_stopping', {}).get('patience', 6)}")

    # 2. Instantiate DataLoaders
    print(f"\n[DATA] Creating PyTorch DataLoaders (EuroSAT RGB)...")
    loaders = create_dataloaders(
        config=merged_cfg,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=seed,
    )
    print(f"  Train Samples    : {len(loaders.train.dataset):,d} ({len(loaders.train)} batches)")
    print(f"  Val Samples      : {len(loaders.val.dataset):,d} ({len(loaders.val)} batches)")
    print(f"  Test Samples     : {len(loaders.test.dataset):,d} (UNTOUCHED - Final evaluation only)")

    # 3. Instantiate Custom CNN Model
    print(f"\n[MODEL] Building CustomCNN Baseline Model...")
    model = build_model("custom_cnn", config=merged_cfg)
    total_p, train_p = model.count_parameters()
    print(f"  Total Parameters    : {total_p:,d}")
    print(f"  Trainable Parameters: {train_p:,d}")

    # 4. Instantiate Trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        config=merged_cfg,
        device=device,
    )

    # 5. Execute Training Loop
    print(f"\n[TRAINING] Starting Optimization Loop across {epochs} Epochs:\n")
    results = trainer.train(epochs=epochs)

    # 6. Summary Output
    print("\n" + "=" * 70)
    print("   TRAINING RUN COMPLETE - SUMMARY REPORT")
    print("=" * 70)
    print(f"  Total Epochs Run       : {results['total_epochs_run']}")
    print(f"  Best Epoch             : {results['best_epoch']}")
    print(f"  Best Val Macro F1      : {results['best_val_macro_f1']:.4f}")
    print(f"  Total Training Time    : {results['total_duration_sec']:.2f} s")
    print(f"  Best Checkpoint Saved  : {results['best_checkpoint_path']}")
    print(f"  Training History Saved : {results['history_path']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
