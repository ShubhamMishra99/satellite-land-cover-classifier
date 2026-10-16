"""Training engine, validation loops, early stopping, and training history tracking."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.constants import MODELS_DIR
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import get_loss_fn
from src.training.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-End Trainer for Satellite Land-Use Classifiers.

    Manages the optimization loop, epoch evaluations, learning rate scheduling on
    validation Macro F1, early stopping, checkpointing, and metrics history persistence.

    Attributes:
        model: PyTorch neural network module.
        train_loader: DataLoader for the training partition.
        val_loader: DataLoader for the validation partition.
        config: Configuration dictionary.
        device: Active PyTorch computation device (CUDA or CPU).
        optimizer: PyTorch optimizer (AdamW).
        scheduler: Learning rate scheduler.
        loss_fn: Criterion (CrossEntropyLoss).
        best_val_macro_f1: Highest validation Macro F1 achieved during training.
        history: Epoch-by-epoch training and validation metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        """Initialize Trainer.

        Args:
            model: Neural network model to train.
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            config: Full project or model training configuration dictionary.
            device: Optional device (defaults to CUDA if available, else CPU).
            test_loader: Optional test DataLoader (NOT used during training or model selection).
        """
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Parse training hyperparameters from config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.lr = float(train_cfg.get("learning_rate", 0.001))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0001))

        # Loss function
        self.loss_fn = get_loss_fn("cross_entropy").to(self.device)

        # Optimizer: AdamW on trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer: Optimizer = AdamW(
            trainable_params if trainable_params else self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning Rate Scheduler: ReduceLROnPlateau monitoring val_macro_f1
        sched_cfg = train_cfg.get("scheduler", config.get("scheduler", {}))
        sched_type = sched_cfg.get("type", "ReduceLROnPlateau")
        if sched_type == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=int(sched_cfg.get("T_max", self.epochs)),
                eta_min=float(sched_cfg.get("eta_min", 1e-6)),
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=sched_cfg.get("mode", "max"),
                factor=float(sched_cfg.get("factor", 0.5)),
                patience=int(sched_cfg.get("patience", 3)),
                min_lr=float(sched_cfg.get("min_lr", 1e-5)),
            )

        # Early Stopping configuration
        es_cfg = train_cfg.get("early_stopping", config.get("early_stopping", {}))
        self.early_stopping_patience = int(es_cfg.get("patience", 6))
        self.early_stopping_min_delta = float(es_cfg.get("min_delta", 0.001))
        self.early_stopping_counter = 0
        self.best_val_macro_f1 = -1.0
        self.best_epoch = 0

        # Output artifact destinations
        artifacts_cfg = config.get("artifacts", {})
        checkpoint_cfg = config.get("checkpoint", {})
        best_path = checkpoint_cfg.get("path") or artifacts_cfg.get("best_model_path", MODELS_DIR / "custom_cnn_best.pt")
        history_cfg = config.get("history", {})
        metrics_path = history_cfg.get("path") or artifacts_cfg.get("metrics_path", MODELS_DIR / "custom_cnn_history.json")

        self.best_checkpoint_path = Path(best_path)
        self.history_path = Path(metrics_path)

        # History tracker
        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_macro_precision": [],
            "train_macro_recall": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_precision": [],
            "val_macro_recall": [],
            "val_macro_f1": [],
            "learning_rate": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """Execute one complete training epoch across train_loader."""
        self.model.train()
        running_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

        total_samples = len(all_targets)
        epoch_loss = running_loss / max(total_samples, 1)
        metrics = calculate_metrics(y_true=all_targets, y_pred=all_preds)

        return {
            "loss": round(epoch_loss, 4),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Execute one complete evaluation epoch across val_loader without gradient computation."""
        self.model.eval()
        running_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())

        total_samples = len(all_targets)
        epoch_loss = running_loss / max(total_samples, 1)
        metrics = calculate_metrics(y_true=all_targets, y_pred=all_preds)

        return {
            "loss": round(epoch_loss, 4),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }

    def _step_scheduler(self, val_f1: float) -> None:
        """Step learning rate scheduler based on scheduler type."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_f1)
        elif self.scheduler is not None and hasattr(self.scheduler, "step"):
            self.scheduler.step()

    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Execute the full training pipeline with validation, scheduling, and early stopping.

        Args:
            epochs: Optional epoch count override (defaults to config value).

        Returns:
            Dictionary containing overall training summary, best metrics, and history.
        """
        total_epochs = epochs if epochs is not None else self.epochs
        logger.info(f"Starting training run: {total_epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()

            # 1. Train epoch
            train_res = self.train_epoch()

            # 2. Validate epoch
            val_res = self.validate_epoch()

            # 3. Current learning rate
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            # 4. Step LR scheduler based on validation Macro F1
            self._step_scheduler(val_res["macro_f1"])

            # 5. Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_res["loss"])
            self.history["train_accuracy"].append(train_res["accuracy"])
            self.history["train_macro_precision"].append(train_res["macro_precision"])
            self.history["train_macro_recall"].append(train_res["macro_recall"])
            self.history["train_macro_f1"].append(train_res["macro_f1"])
            self.history["val_loss"].append(val_res["loss"])
            self.history["val_accuracy"].append(val_res["accuracy"])
            self.history["val_macro_precision"].append(val_res["macro_precision"])
            self.history["val_macro_recall"].append(val_res["macro_recall"])
            self.history["val_macro_f1"].append(val_res["macro_f1"])
            self.history["learning_rate"].append(current_lr)

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch [{epoch:02d}/{total_epochs:02d}] ({epoch_time:.1f}s) | "
                f"Train Loss: {train_res['loss']:.4f}, Train F1: {train_res['macro_f1']:.4f} | "
                f"Val Loss: {val_res['loss']:.4f}, Val Acc: {val_res['accuracy']:.4f}, Val F1: {val_res['macro_f1']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # 6. Check for improvement in validation Macro F1
            val_f1 = val_res["macro_f1"]
            if val_f1 > self.best_val_macro_f1 + self.early_stopping_min_delta:
                self.best_val_macro_f1 = val_f1
                self.best_epoch = epoch
                self.early_stopping_counter = 0

                # Save best model checkpoint
                save_checkpoint(
                    model=self.model,
                    epoch=epoch,
                    best_val_macro_f1=self.best_val_macro_f1,
                    config=self.config,
                    filepath=self.best_checkpoint_path,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    history=self.history,
                )
            else:
                self.early_stopping_counter += 1
                logger.info(
                    f"EarlyStopping counter: {self.early_stopping_counter}/{self.early_stopping_patience} "
                    f"(Best Val Macro F1: {self.best_val_macro_f1:.4f} at Epoch {self.best_epoch})"
                )

            # 7. Check early stopping trigger
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs without improvement in Val Macro F1."
                )
                break

        total_duration = time.time() - start_time
        logger.info(
            f"Training finished in {total_duration:.1f}s. "
            f"Best Val Macro F1: {self.best_val_macro_f1:.4f} (Epoch {self.best_epoch})"
        )

        # 8. Persist training history JSON
        self._save_history()

        # 9. Reload best weights into model
        if self.best_checkpoint_path.exists():
            load_checkpoint(self.best_checkpoint_path, self.model, device=self.device)
            logger.info("Restored best validation checkpoint into model.")

        return {
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "total_epochs_run": len(self.history["epoch"]),
            "total_duration_sec": round(total_duration, 2),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "history_path": str(self.history_path),
            "history": self.history,
        }

    def _save_history(self) -> None:
        """Persist structured training history dictionary to JSON."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history metrics to: {self.history_path}")


class StagedTrainer(Trainer):
    """Trainer specialized for Staged Fine-Tuning workflows (e.g. ResNet-18).

    Orchestrates two-stage transfer learning:
    - Phase 1 (Feature Extraction): Backbone frozen, classification head trained.
    - Phase 2 (Fine-Tuning): High-level semantic layers (layer3, layer4, fc) unfrozen with
      lower learning rate and cosine annealing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            test_loader=test_loader,
        )

        # Staged hyperparameters parsing
        staged_cfg = config.get("staged_training", {})
        p1_cfg = staged_cfg.get("phase1", {})
        p2_cfg = staged_cfg.get("phase2", {})
        train_cfg = config.get("training", {})
        opt_cfg = config.get("optimizer", {})

        self.phase1_epochs = int(train_cfg.get("phase1_epochs", p1_cfg.get("epochs", 10)))
        self.phase2_epochs = int(train_cfg.get("phase2_epochs", p2_cfg.get("epochs", 15)))

        self.phase1_lr = float(opt_cfg.get("phase1_learning_rate", train_cfg.get("phase1_learning_rate", p1_cfg.get("learning_rate", 0.001))))
        self.phase2_lr = float(opt_cfg.get("phase2_learning_rate", train_cfg.get("phase2_learning_rate", p2_cfg.get("learning_rate", 0.00005))))

        # Checkpoint destinations
        artifacts_cfg = config.get("artifacts", {})
        checkpoint_cfg = config.get("checkpoint", {})
        history_cfg = config.get("history", {})

        p1_path = checkpoint_cfg.get("phase1_path") or artifacts_cfg.get("phase1_model_path", MODELS_DIR / "resnet18_phase1_best.pt")
        best_path = checkpoint_cfg.get("path") or artifacts_cfg.get("best_model_path", MODELS_DIR / "resnet18_best.pt")
        hist_path = history_cfg.get("path") or artifacts_cfg.get("metrics_path", MODELS_DIR / "resnet18_history.json")

        self.phase1_best_checkpoint_path = Path(p1_path)
        self.best_checkpoint_path = Path(best_path)
        self.history_path = Path(hist_path)

        # Extended history tracker including phase tracking
        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "phase": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_macro_precision": [],
            "train_macro_recall": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_precision": [],
            "val_macro_recall": [],
            "val_macro_f1": [],
            "learning_rate": [],
        }

        self.phase1_best_val_macro_f1 = -1.0
        self.phase1_best_epoch = 0
        self.best_phase = "none"

    def train_staged(
        self,
        phase1_epochs: Optional[int] = None,
        phase2_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute two-phase staged fine-tuning.

        Phase 1: Feature Extraction (Backbone frozen, train classification head).
        Phase 2: Fine-Tuning (Unfreeze layer3, layer4, classification head at lower LR).

        Args:
            phase1_epochs: Optional override for Phase 1 epoch count.
            phase2_epochs: Optional override for Phase 2 epoch count.

        Returns:
            Dictionary with training summary, best metrics, and full history.
        """
        p1_epochs = phase1_epochs if phase1_epochs is not None else self.phase1_epochs
        p2_epochs = phase2_epochs if phase2_epochs is not None else self.phase2_epochs

        start_time = time.time()
        global_epoch = 0

        # ==========================================
        # PHASE 1: Feature Extraction
        # ==========================================
        if p1_epochs > 0:
            logger.info("=" * 60)
            logger.info("   STARTING PHASE 1: FEATURE EXTRACTION")
            logger.info("   (Backbone Frozen | Training Classification Head)")
            logger.info("=" * 60)

            # 1. Freeze backbone
            if hasattr(self.model, "freeze_for_phase1"):
                self.model.freeze_for_phase1()
            else:
                for name, p in self.model.named_parameters():
                    p.requires_grad = ("fc" in name or "classifier" in name)

            # 2. Setup Phase 1 Optimizer (only trainable parameters)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(
                trainable_params,
                lr=self.phase1_lr,
                weight_decay=self.weight_decay,
            )

            # 3. Phase 1 Scheduler
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
            )

            self.early_stopping_counter = 0

            for ep in range(1, p1_epochs + 1):
                global_epoch += 1
                ep_start = time.time()

                train_res = self.train_epoch()
                val_res = self.validate_epoch()
                current_lr = float(self.optimizer.param_groups[0]["lr"])

                self._step_scheduler(val_res["macro_f1"])

                # Record history
                self.history["epoch"].append(global_epoch)
                self.history["phase"].append("phase1")
                self.history["train_loss"].append(train_res["loss"])
                self.history["train_accuracy"].append(train_res["accuracy"])
                self.history["train_macro_precision"].append(train_res["macro_precision"])
                self.history["train_macro_recall"].append(train_res["macro_recall"])
                self.history["train_macro_f1"].append(train_res["macro_f1"])
                self.history["val_loss"].append(val_res["loss"])
                self.history["val_accuracy"].append(val_res["accuracy"])
                self.history["val_macro_precision"].append(val_res["macro_precision"])
                self.history["val_macro_recall"].append(val_res["macro_recall"])
                self.history["val_macro_f1"].append(val_res["macro_f1"])
                self.history["learning_rate"].append(current_lr)

                ep_time = time.time() - ep_start
                logger.info(
                    f"[Phase 1] Epoch [{ep:02d}/{p1_epochs:02d}] (Total Ep {global_epoch:02d}) ({ep_time:.1f}s) | "
                    f"Train Loss: {train_res['loss']:.4f}, Train F1: {train_res['macro_f1']:.4f} | "
                    f"Val Loss: {val_res['loss']:.4f}, Val Acc: {val_res['accuracy']:.4f}, Val F1: {val_res['macro_f1']:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

                val_f1 = val_res["macro_f1"]
                if val_f1 > self.phase1_best_val_macro_f1 + self.early_stopping_min_delta:
                    self.phase1_best_val_macro_f1 = val_f1
                    self.phase1_best_epoch = ep
                    self.early_stopping_counter = 0

                    save_checkpoint(
                        model=self.model,
                        epoch=global_epoch,
                        best_val_macro_f1=self.phase1_best_val_macro_f1,
                        config=self.config,
                        filepath=self.phase1_best_checkpoint_path,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        history=self.history,
                        extra_metadata={"phase": "phase1", "phase_epoch": ep},
                    )

                if val_f1 > self.best_val_macro_f1:
                    self.best_val_macro_f1 = val_f1
                    self.best_epoch = global_epoch
                    self.best_phase = "phase1"

                # Check early stopping in Phase 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Phase 1 Early stopping triggered after {ep} epochs.")
                    break

            # Restore best Phase 1 checkpoint before moving to Phase 2
            if self.phase1_best_checkpoint_path.exists():
                load_checkpoint(self.phase1_best_checkpoint_path, self.model, device=self.device)
                logger.info(
                    f"Restored best Phase 1 checkpoint (Val Macro F1: {self.phase1_best_val_macro_f1:.4f}) before Phase 2."
                )

        # ==========================================
        # PHASE 2: Staged Fine-Tuning
        # ==========================================
        if p2_epochs > 0:
            logger.info("=" * 60)
            logger.info("   STARTING PHASE 2: FINE-TUNING")
            logger.info("   (Unfreezing layer3, layer4, and classifier head)")
            logger.info("=" * 60)

            # 1. Unfreeze selective layers
            if hasattr(self.model, "unfreeze_for_phase2"):
                self.model.unfreeze_for_phase2(unfreeze_layers=("layer3", "layer4", "fc"))
            else:
                for name, p in self.model.named_parameters():
                    p.requires_grad = any(k in name for k in ("layer3", "layer4", "fc", "classifier"))

            # 2. Setup Phase 2 Optimizer
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(
                trainable_params,
                lr=self.phase2_lr,
                weight_decay=self.weight_decay,
            )

            # 3. Phase 2 Scheduler: CosineAnnealingLR
            sched_cfg = self.config.get("scheduler", {})
            sched_type = sched_cfg.get("type", "CosineAnnealingLR")
            if sched_type == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=p2_epochs,
                    eta_min=float(sched_cfg.get("eta_min", 1e-6)),
                )
            else:
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode="max",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                )

            self.early_stopping_counter = 0

            for ep in range(1, p2_epochs + 1):
                global_epoch += 1
                ep_start = time.time()

                train_res = self.train_epoch()
                val_res = self.validate_epoch()
                current_lr = float(self.optimizer.param_groups[0]["lr"])

                self._step_scheduler(val_res["macro_f1"])

                # Record history
                self.history["epoch"].append(global_epoch)
                self.history["phase"].append("phase2")
                self.history["train_loss"].append(train_res["loss"])
                self.history["train_accuracy"].append(train_res["accuracy"])
                self.history["train_macro_precision"].append(train_res["macro_precision"])
                self.history["train_macro_recall"].append(train_res["macro_recall"])
                self.history["train_macro_f1"].append(train_res["macro_f1"])
                self.history["val_loss"].append(val_res["loss"])
                self.history["val_accuracy"].append(val_res["accuracy"])
                self.history["val_macro_precision"].append(val_res["macro_precision"])
                self.history["val_macro_recall"].append(val_res["macro_recall"])
                self.history["val_macro_f1"].append(val_res["macro_f1"])
                self.history["learning_rate"].append(current_lr)

                ep_time = time.time() - ep_start
                logger.info(
                    f"[Phase 2] Epoch [{ep:02d}/{p2_epochs:02d}] (Total Ep {global_epoch:02d}) ({ep_time:.1f}s) | "
                    f"Train Loss: {train_res['loss']:.4f}, Train F1: {train_res['macro_f1']:.4f} | "
                    f"Val Loss: {val_res['loss']:.4f}, Val Acc: {val_res['accuracy']:.4f}, Val F1: {val_res['macro_f1']:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

                val_f1 = val_res["macro_f1"]
                if val_f1 > self.best_val_macro_f1 + self.early_stopping_min_delta:
                    self.best_val_macro_f1 = val_f1
                    self.best_epoch = global_epoch
                    self.best_phase = "phase2"
                    self.early_stopping_counter = 0

                    save_checkpoint(
                        model=self.model,
                        epoch=global_epoch,
                        best_val_macro_f1=self.best_val_macro_f1,
                        config=self.config,
                        filepath=self.best_checkpoint_path,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        history=self.history,
                        extra_metadata={"phase": "phase2", "phase_epoch": ep},
                    )
                else:
                    self.early_stopping_counter += 1
                    logger.info(
                        f"EarlyStopping counter: {self.early_stopping_counter}/{self.early_stopping_patience} "
                        f"(Best Val Macro F1: {self.best_val_macro_f1:.4f} at Epoch {self.best_epoch})"
                    )

                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Phase 2 Early stopping triggered after {ep} epochs.")
                    break

        total_duration = time.time() - start_time

        # Save training history JSON
        self._save_history()

        # Restore overall best model checkpoint
        if self.best_checkpoint_path.exists():
            load_checkpoint(self.best_checkpoint_path, self.model, device=self.device)
            logger.info(f"Restored best overall checkpoint ({self.best_checkpoint_path}) into model.")

        return {
            "best_epoch": self.best_epoch,
            "best_phase": self.best_phase,
            "best_val_macro_f1": self.best_val_macro_f1,
            "phase1_best_val_macro_f1": self.phase1_best_val_macro_f1,
            "phase1_best_epoch": self.phase1_best_epoch,
            "total_epochs_run": len(self.history["epoch"]),
            "total_duration_sec": round(total_duration, 2),
            "phase1_checkpoint_path": str(self.phase1_best_checkpoint_path),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "history_path": str(self.history_path),
            "history": self.history,
        }

    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Delegate train call to train_staged for compatibility."""
        return self.train_staged()
