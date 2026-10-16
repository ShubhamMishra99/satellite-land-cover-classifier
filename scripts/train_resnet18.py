"""CLI training script for ResNet-18 Transfer Learning & Staged Fine-Tuning on EuroSAT RGB dataset."""

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

from src.config import get_base_config, get_resnet18_config
from src.constants import MODELS_DIR
from src.data.dataloader import create_dataloaders
from src.models import build_model
from src.training.trainer import StagedTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_resnet18")


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
    parser = argparse.ArgumentParser(description="Train ResNet-18 Transfer Learning with Staged Fine-Tuning on EuroSAT RGB.")
    parser.add_argument("--phase1-epochs", type=int, default=None, help="Phase 1 (Feature Extraction) epochs")
    parser.add_argument("--phase2-epochs", type=int, default=None, help="Phase 2 (Fine-Tuning) epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--phase1-lr", type=float, default=None, help="Phase 1 learning rate (overrides config)")
    parser.add_argument("--phase2-lr", type=float, default=None, help="Phase 2 learning rate (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Path to save final best model checkpoint")
    parser.add_argument("--phase1-checkpoint-path", type=Path, default=None, help="Path to save Phase 1 best model checkpoint")
    parser.add_argument("--history-path", type=Path, default=None, help="Path to save training history JSON")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretraining (train from random init)")

    args = parser.parse_args()

    print("=" * 75)
    print("   SATELLITE LAND-USE CLASSIFIER - TRANSFER LEARNING TRAINING")
    print("               ResNet-18 Staged Fine-Tuning Pipeline")
    print("=" * 75)

    # 1. Load Configurations
    base_cfg = get_base_config()
    resnet_cfg = get_resnet18_config()

    # Merge configuration
    merged_cfg = {**base_cfg, **resnet_cfg}
    seed = args.seed if args.seed is not None else merged_cfg.get("seed", 42)
    set_seed(seed)

    # Apply overrides
    if args.phase1_epochs is not None:
        merged_cfg.setdefault("training", {})["phase1_epochs"] = args.phase1_epochs
    if args.phase2_epochs is not None:
        merged_cfg.setdefault("training", {})["phase2_epochs"] = args.phase2_epochs
    if args.batch_size is not None:
        merged_cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.phase1_lr is not None:
        merged_cfg.setdefault("training", {})["phase1_learning_rate"] = args.phase1_lr
    if args.phase2_lr is not None:
        merged_cfg.setdefault("training", {})["phase2_learning_rate"] = args.phase2_lr
    if args.checkpoint_path is not None:
        merged_cfg.setdefault("checkpoint", {})["path"] = str(args.checkpoint_path)
    if args.phase1_checkpoint_path is not None:
        merged_cfg.setdefault("checkpoint", {})["phase1_path"] = str(args.phase1_checkpoint_path)
    if args.history_path is not None:
        merged_cfg.setdefault("history", {})["path"] = str(args.history_path)
    if args.no_pretrained:
        merged_cfg.setdefault("model", {})["pretrained"] = False

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = merged_cfg.get("training", {}).get("batch_size", 64)
    phase1_epochs = merged_cfg.get("training", {}).get("phase1_epochs", 10)
    phase2_epochs = merged_cfg.get("training", {}).get("phase2_epochs", 15)
    phase1_lr = merged_cfg.get("training", {}).get("phase1_learning_rate", 0.001)
    phase2_lr = merged_cfg.get("training", {}).get("phase2_learning_rate", 0.00005)
    use_pretrained = merged_cfg.get("model", {}).get("pretrained", True)

    print(f"\n[DEVICE & ENVIRONMENT]")
    print(f"  Device              : {device}")
    print(f"  Random Seed         : {seed}")

    # 2. Instantiate DataLoaders
    print(f"\n[DATA] Creating PyTorch DataLoaders (EuroSAT RGB)...")
    loaders = create_dataloaders(
        config=merged_cfg,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=seed,
    )
    print(f"  Train Samples       : {len(loaders.train.dataset):,d} ({len(loaders.train)} batches)")
    print(f"  Val Samples         : {len(loaders.val.dataset):,d} ({len(loaders.val)} batches)")
    print(f"  Test Samples        : {len(loaders.test.dataset):,d} (UNTOUCHED - Final evaluation only)")

    # 3. Instantiate ResNet-18 Model
    print(f"\n[MODEL] Building ResNet-18 Transfer Learning Model...")
    model = build_model("resnet18", config=merged_cfg)
    total_params, _ = model.count_parameters()

    # Inspect parameter counts for Phase 1 and Phase 2
    model.freeze_for_phase1()
    _, p1_trainable = model.count_parameters()

    model.unfreeze_for_phase2(["layer3", "layer4", "fc"])
    _, p2_trainable = model.count_parameters()

    print(f"  Architecture        : ResNet-18")
    print(f"  Pretrained Weights  : {'ImageNet-1K (Loaded)' if model.pretrained_loaded else 'Random Initialization'}")
    print(f"  Total Parameters    : {total_params:,d}")
    print(f"\n  Phase 1 (Feature Extraction):")
    print(f"    Trainable Params  : {p1_trainable:,d} (Classification Head: fc)")
    print(f"    Epochs            : {phase1_epochs}")
    print(f"    Learning Rate     : {phase1_lr}")
    print(f"\n  Phase 2 (Fine-Tuning):")
    print(f"    Trainable Params  : {p2_trainable:,d} (layer3, layer4, fc)")
    print(f"    Epochs            : {phase2_epochs}")
    print(f"    Learning Rate     : {phase2_lr}")

    # 4. Instantiate StagedTrainer
    trainer = StagedTrainer(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        config=merged_cfg,
        device=device,
    )

    # 5. Execute Staged Training Loop
    print(f"\n[TRAINING] Starting Staged Fine-Tuning Execution:\n")
    results = trainer.train_staged(
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
    )

    # 6. Summary Output
    print("\n" + "=" * 75)
    print("   RESNET-18 TRAINING COMPLETE - SUMMARY REPORT")
    print("=" * 75)
    print(f"  Best Validation Macro F1 : {results['best_val_macro_f1']:.4f}")
    print(f"  Best Global Epoch        : {results['best_epoch']}")
    print(f"  Best Phase               : {results['best_phase']}")
    print(f"  Phase 1 Best Val Macro F1: {results['phase1_best_val_macro_f1']:.4f} (Epoch {results['phase1_best_epoch']})")
    print(f"  Total Epochs Executed    : {results['total_epochs_run']}")
    print(f"  Total Training Time      : {results['total_duration_sec']:.2f} s")
    print(f"  Phase 1 Checkpoint       : {results['phase1_checkpoint_path']}")
    print(f"  Best Checkpoint Saved    : {results['best_checkpoint_path']}")
    print(f"  Training History Saved   : {results['history_path']}")
    print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
