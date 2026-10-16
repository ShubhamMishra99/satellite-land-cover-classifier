"""Lightweight smoke test for Custom CNN baseline architecture and training primitives.

Validates forward pass, loss calculation, backward gradient flow, optimizer step,
checkpoint round-trip serialization, and metrics computation without a full training run.
"""

import logging
from pathlib import Path
import sys
import tempfile
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_base_config, get_custom_cnn_config
from src.constants import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES
from src.models.custom_cnn import CustomCNN
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import get_loss_fn
from src.training.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test_custom_cnn")


def run_smoke_test() -> int:
    print("=" * 70)
    print("   SATELLITE LAND-USE CLASSIFIER - MILESTONE 3 SMOKE TEST")
    print("           Custom CNN Baseline & Training Primitives")
    print("=" * 70)

    # 1. Configuration loading
    print("\n[STEP 1] Loading Configuration:")
    try:
        base_cfg = get_base_config()
        cnn_cfg = get_custom_cnn_config()
        print(f"  [+] base_config.yaml loaded (Seed: {base_cfg.get('seed')})")
        print(f"  [+] custom_cnn.yaml loaded (Model: {cnn_cfg.get('model', {}).get('name')})")
    except Exception as exc:
        print(f"  [-] FAIL: Failed to load config: {exc}")
        return 1

    # 2. Device determination
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[STEP 2] Execution Device: {device}")

    # 3. Model instantiation
    print("\n[STEP 3] Instantiating CustomCNN Baseline:")
    try:
        model = CustomCNN(
            in_channels=IMAGE_CHANNELS,
            num_classes=NUM_CLASSES,
            conv_channels=(32, 64, 128),
            dropout_rate=0.3,
        ).to(device)
        total_p, train_p = model.count_parameters()
        print(f"  [+] Model instantiated successfully.")
        print(f"  [+] Total Parameters    : {total_p:,}")
        print(f"  [+] Trainable Parameters: {train_p:,}")
    except Exception as exc:
        print(f"  [-] FAIL: Model instantiation failed: {exc}")
        return 1

    # 4. Forward Pass Verification
    print("\n[STEP 4] Forward Pass with Batch [4, 3, 64, 64]:")
    torch.manual_seed(42)
    dummy_x = torch.randn(4, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, device=device)
    dummy_y = torch.tensor([0, 3, 7, 9], device=device, dtype=torch.long)

    try:
        logits = model(dummy_x)
        print(f"  [+] Output Logits Shape: {tuple(logits.shape)} (Expected: (4, 10))")
        if tuple(logits.shape) != (4, 10):
            print(f"  [-] FAIL: Output shape mismatch: {logits.shape}")
            return 1
        if not torch.isfinite(logits).all():
            print("  [-] FAIL: Output contains NaN or Inf values!")
            return 1
    except Exception as exc:
        print(f"  [-] FAIL: Forward pass failed: {exc}")
        return 1

    # 5. Loss computation
    print("\n[STEP 5] Loss Computation (CrossEntropyLoss):")
    try:
        criterion = get_loss_fn("cross_entropy").to(device)
        loss = criterion(logits, dummy_y)
        print(f"  [+] Loss Value: {loss.item():.4f}")
        if not torch.isfinite(loss):
            print("  [-] FAIL: Loss is not finite!")
            return 1
    except Exception as exc:
        print(f"  [-] FAIL: Loss computation failed: {exc}")
        return 1

    # 6. Backward Pass & Gradient Verification
    print("\n[STEP 6] Backward Pass & Gradient Flow:")
    try:
        loss.backward()
        grads_ok = True
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"  [-] Missing gradient for parameter: {name}")
                    grads_ok = False
                elif not torch.isfinite(param.grad).all():
                    print(f"  [-] Non-finite gradient for parameter: {name}")
                    grads_ok = False

        if not grads_ok:
            return 1
        print("  [+] PASS: Gradients computed and finite for all trainable layers.")
    except Exception as exc:
        print(f"  [-] FAIL: Backward pass failed: {exc}")
        return 1

    # 7. Optimizer Step Verification
    print("\n[STEP 7] Optimizer Step (AdamW):")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer.step()
        optimizer.zero_grad()
        print("  [+] PASS: Optimizer step executed successfully.")
    except Exception as exc:
        print(f"  [-] FAIL: Optimizer step failed: {exc}")
        return 1

    # 8. Metrics Computation Verification
    print("\n[STEP 8] Metrics Computation:")
    try:
        preds = torch.argmax(logits, dim=1)
        metrics = calculate_metrics(y_true=dummy_y, y_pred=preds)
        print(f"  [+] Accuracy        : {metrics['accuracy']:.4f}")
        print(f"  [+] Macro Precision : {metrics['macro_precision']:.4f}")
        print(f"  [+] Macro Recall    : {metrics['macro_recall']:.4f}")
        print(f"  [+] Macro F1        : {metrics['macro_f1']:.4f}")
    except Exception as exc:
        print(f"  [-] FAIL: Metrics computation failed: {exc}")
        return 1

    # 9. Checkpoint Round-Trip Serialization
    print("\n[STEP 9] Checkpoint Save & Load Round-Trip:")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_path = Path(tmp_dir) / "test_model.pt"
            save_checkpoint(
                model=model,
                epoch=1,
                best_val_macro_f1=0.85,
                config=cnn_cfg,
                filepath=chk_path,
                optimizer=optimizer,
            )

            new_model = CustomCNN().to(device)
            chk_data = load_checkpoint(chk_path, new_model, device=device)

            # Test weight equivalence
            p1 = next(model.parameters())
            p2 = next(new_model.parameters())
            if not torch.allclose(p1, p2):
                print("  [-] FAIL: Loaded weights do not match saved weights!")
                return 1
            print("  [+] PASS: Model checkpoint serialized and restored identically.")
    except Exception as exc:
        print(f"  [-] FAIL: Checkpoint test failed: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("ALL MILESTONE 3 CUSTOM CNN SMOKE TESTS PASSED!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
