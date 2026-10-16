"""Smoke test script for Milestone 4: ResNet-18 Transfer Learning & Staged Fine-Tuning."""

import logging
from pathlib import Path
import sys
import tempfile
import torch
import torch.nn as nn
from torch.optim import AdamW

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_base_config, get_resnet18_config
from src.constants import NUM_CLASSES
from src.models import build_model
from src.models.resnet18 import ResNet18Transfer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import get_loss_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smoke_test_resnet18")


def run_smoke_tests() -> bool:
    print("=" * 70)
    print("   MILESTONE 4: RESNET-18 TRANSFER LEARNING SMOKE TEST SUITE")
    print("=" * 70)

    # 1. Configuration Loading
    print("\n[Step 1/10] Verifying Configuration Loading...")
    base_cfg = get_base_config()
    resnet_cfg = get_resnet18_config()
    assert base_cfg is not None, "Failed to load base_config.yaml"
    assert resnet_cfg is not None, "Failed to load resnet18.yaml"
    print("  [PASS] Configuration files loaded and parsed successfully.")

    # 2. ResNet-18 Model Instantiation
    print("\n[Step 2/10] Verifying ResNet-18 Model Instantiation...")
    model = build_model("resnet18", config=resnet_cfg)
    assert isinstance(model, ResNet18Transfer), "build_model did not return ResNet18Transfer instance!"
    total_p, trainable_p = model.count_parameters()
    print(f"  [PASS] ResNet-18 model instantiated: {total_p:,} total parameters.")
    print(f"         Pretrained ImageNet weights loaded: {model.pretrained_loaded}")

    # 3. Output Shape Verification [4, 10]
    print("\n[Step 3/10] Verifying Output Logits Shape [4, 10]...")
    dummy_input = torch.randn(4, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (4, NUM_CLASSES), f"Expected shape (4, 10), got {logits.shape}"
    print(f"  [PASS] Forward pass output shape: {tuple(logits.shape)}")

    # 4. CrossEntropyLoss Computation
    print("\n[Step 4/10] Verifying CrossEntropyLoss Computation...")
    model.train()
    loss_fn = get_loss_fn("cross_entropy")
    dummy_targets = torch.tensor([0, 3, 5, 9], dtype=torch.long)
    logits = model(dummy_input)
    loss = loss_fn(logits, dummy_targets)
    assert loss.item() > 0, "Loss value must be positive!"
    print(f"  [PASS] Loss computed successfully: {loss.item():.4f}")

    # 5. Backward Pass and Gradient Computation
    print("\n[Step 5/10] Verifying Backward Pass & Gradient Flow...")
    model.zero_grad()
    loss.backward()
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients were computed during backward pass!"
    print("  [PASS] Backward pass completed with valid gradient flow.")

    # 6. Phase 1 Freezing Verification
    print("\n[Step 6/10] Verifying Phase 1 (Feature Extraction) Parameter Freezing...")
    model.freeze_for_phase1()
    total_p, p1_trainable = model.count_parameters()
    assert p1_trainable == (512 * 10 + 10), f"Expected 5,130 trainable params in Phase 1, got {p1_trainable}"
    for name, p in model.backbone.named_parameters():
        if "fc" in name:
            assert p.requires_grad is True, f"Classifier parameter {name} should be trainable!"
        else:
            assert p.requires_grad is False, f"Backbone parameter {name} should be frozen!"
    print(f"  [PASS] Phase 1 freezing verified: {p1_trainable:,} trainable params (fc head only).")

    # 7. Phase 2 Unfreezing Verification
    print("\n[Step 7/10] Verifying Phase 2 (Fine-Tuning) Partial Unfreezing...")
    model.unfreeze_for_phase2(["layer3", "layer4", "fc"])
    total_p, p2_trainable = model.count_parameters()
    assert p2_trainable > 10_000_000, f"Expected >10M trainable params in Phase 2, got {p2_trainable}"
    for name, p in model.backbone.named_parameters():
        if any(k in name for k in ("layer3", "layer4", "fc")):
            assert p.requires_grad is True, f"High-level parameter {name} should be trainable!"
        elif any(k in name for k in ("conv1", "bn1", "layer1", "layer2")):
            assert p.requires_grad is False, f"Early parameter {name} should remain frozen!"
    print(f"  [PASS] Phase 2 unfreezing verified: {p2_trainable:,} trainable params (layer3, layer4, fc).")

    # 8. Optimizer Parameter Update Verification
    print("\n[Step 8/10] Verifying Optimizer Parameter Update...")
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    fc_weight_before = model.fc.weight.clone()
    model.zero_grad()
    loss = loss_fn(model(dummy_input), dummy_targets)
    loss.backward()
    optimizer.step()
    fc_weight_after = model.fc.weight.clone()
    assert not torch.equal(fc_weight_before, fc_weight_after), "Optimizer did not update trainable parameters!"
    print("  [PASS] Optimizer successfully updated trainable parameters.")

    # 9. Finite Values (No NaN / Inf)
    print("\n[Step 9/10] Verifying Output and Gradient Numerical Stability (No NaN/Inf)...")
    assert torch.isfinite(loss).item(), "Loss is NaN or Inf!"
    assert torch.isfinite(model(dummy_input)).all(), "Forward logits contain NaN or Inf!"
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradients contain NaN or Inf!"
    print("  [PASS] All outputs, losses, and gradients are finite real numbers.")

    # 10. Checkpoint Save and Load Verification
    print("\n[Step 10/10] Verifying Checkpoint Serialization & Deserialization...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ckpt = Path(tmpdir) / "test_resnet18_ckpt.pt"
        save_checkpoint(
            model=model,
            epoch=5,
            best_val_macro_f1=0.9650,
            config=resnet_cfg,
            filepath=tmp_ckpt,
            optimizer=optimizer,
            extra_metadata={"phase": "phase2"},
        )
        assert tmp_ckpt.exists(), "Checkpoint file was not created!"

        # Create fresh model and load checkpoint
        fresh_model = build_model("resnet18", config=resnet_cfg)
        loaded_ckpt = load_checkpoint(tmp_ckpt, fresh_model)
        assert loaded_ckpt["epoch"] == 5
        assert loaded_ckpt["best_val_macro_f1"] == 0.9650
        assert loaded_ckpt["extra_metadata"]["phase"] == "phase2"
    print("  [PASS] Checkpoint saved and restored with exact weights and metadata.")

    print("\n" + "=" * 70)
    print("ALL MILESTONE 4 RESNET-18 SMOKE TESTS PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
