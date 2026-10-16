"""Unit and regression tests for uncertainty detection, OOD awareness, and confidence thresholding."""

from pathlib import Path
import pytest
from PIL import Image
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_service import ModelService
from src.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    EUROSAT_CLASSES,
    MODEL_CONFIDENCE_THRESHOLDS,
    RAW_DATA_DIR,
)
from src.inference.predictor import (
    load_trained_model,
    predict_image,
    predict_single_tensor,
)
from src.models.custom_cnn import CustomCNN
from src.models.resnet18 import ResNet18Transfer


class MockLogitsModel(nn.Module):
    """Deterministic mock model that outputs fixed logits regardless of input."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return self.fixed_logits.unsqueeze(0).repeat(batch_size, 1)


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# -------------------------------------------------------------------------
# Step 5.1: Deterministic Synthetic Logits Testing
# -------------------------------------------------------------------------

def test_deterministic_synthetic_logits_accepted():
    """Verify predict_single_tensor marks high-confidence predictions as accepted."""
    # Create logits where class 1 (Forest) has strong probability ~0.85
    logits = torch.tensor([0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    model = MockLogitsModel(logits)
    tensor = torch.zeros(1, 3, 64, 64)

    results = predict_single_tensor(tensor, model=model, confidence_threshold=0.60)

    assert results["predicted_class"] == "Forest"
    assert results["confidence"] > 0.60
    assert results["status"] == "accepted"
    assert results["is_confident"] is True
    assert results["threshold_used"] == 0.60
    assert results["uncertainty_reason"] is None


def test_deterministic_synthetic_logits_uncertain():
    """Verify predict_single_tensor marks low-confidence predictions as potentially_unsupported."""
    # Create near-uniform logits where max prob is ~0.15 (below 0.60)
    logits = torch.tensor([1.1, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0])
    model = MockLogitsModel(logits)
    tensor = torch.zeros(1, 3, 64, 64)

    results = predict_single_tensor(tensor, model=model, confidence_threshold=0.60)

    assert results["status"] == "potentially_unsupported"
    assert results["is_confident"] is False
    assert results["threshold_used"] == 0.60
    assert results["uncertainty_reason"] is not None
    assert "below the uncertainty threshold" in results["uncertainty_reason"]
    # Verify backward compatibility: predicted_class and probabilities are still populated
    assert results["predicted_class"] in EUROSAT_CLASSES
    assert len(results["top3"]) == 3
    assert len(results["all_probabilities"]) == 10


def test_threshold_exact_boundary_behavior():
    """Verify strict boundary behavior at confidence >= threshold."""
    # Construct logits where softmax max is exactly 0.50
    # Two classes at 2.0, eight at 0.0 gives equal top probability
    logits = torch.zeros(10)
    logits[0] = 2.0
    logits[1] = 2.0
    probs = torch.softmax(logits, dim=0)
    top_p = float(probs[0].item())

    model = MockLogitsModel(logits)
    tensor = torch.zeros(1, 3, 64, 64)

    # When threshold equals top_p exactly: accepted (>=)
    res_at_boundary = predict_single_tensor(tensor, model=model, confidence_threshold=top_p)
    assert res_at_boundary["is_confident"] is True
    assert res_at_boundary["status"] == "accepted"

    # When threshold is slightly above top_p: rejected
    res_above = predict_single_tensor(tensor, model=model, confidence_threshold=top_p + 0.001)
    assert res_above["is_confident"] is False
    assert res_above["status"] == "potentially_unsupported"


def test_threshold_edge_values():
    """Verify threshold at 0.0 accepts everything and 1.0 rejects non-certain."""
    logits = torch.randn(10)
    model = MockLogitsModel(logits)
    tensor = torch.zeros(1, 3, 64, 64)

    # Threshold 0.0 must accept any valid probability
    res_zero = predict_single_tensor(tensor, model=model, confidence_threshold=0.0)
    assert res_zero["status"] == "accepted"
    assert res_zero["is_confident"] is True

    # Threshold 1.0 requires 100% confidence
    res_one = predict_single_tensor(tensor, model=model, confidence_threshold=1.0)
    if res_one["confidence"] < 1.0:
        assert res_one["status"] == "potentially_unsupported"
        assert res_one["is_confident"] is False


# -------------------------------------------------------------------------
# Step 5.2: Model-Specific Threshold Verification
# -------------------------------------------------------------------------

def test_model_service_default_thresholds():
    """Verify ModelService assigns empirical validation thresholds."""
    service = ModelService.get_instance()
    assert service.get_default_threshold("resnet18") == 0.60
    assert service.get_default_threshold("custom_cnn") == 0.40
    assert service.get_default_threshold("resnet_18") == 0.60
    assert MODEL_CONFIDENCE_THRESHOLDS["resnet18"] == 0.60
    assert MODEL_CONFIDENCE_THRESHOLDS["custom_cnn"] == 0.40


# -------------------------------------------------------------------------
# Step 5.3: Valid EuroSAT Sample Evaluation
# -------------------------------------------------------------------------

def test_valid_eurosat_sample_accepted(client: TestClient):
    """Verify real EuroSAT satellite tile passes uncertainty filter as accepted."""
    forest_tile = RAW_DATA_DIR / "EuroSAT_RGB" / "Forest" / "Forest_1.jpg"
    if not forest_tile.exists():
        pytest.skip("EuroSAT raw data directory not available.")

    with open(forest_tile, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("Forest_1.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, data={"model": "resnet18"})

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "accepted"
    assert res["is_confident"] is True
    assert res["predicted_class"] == "Forest"
    assert res["confidence"] >= 0.60
    assert res["uncertainty_reason"] is None


# -------------------------------------------------------------------------
# Step 5.4: Real OOD Image Fixtures Testing
# -------------------------------------------------------------------------

def test_ood_fixtures_behavior(client: TestClient):
    """Evaluate real non-satellite photos against suitability and uncertainty detection."""
    ood_dir = Path("tests/fixtures/ood")
    if not ood_dir.exists():
        pytest.skip("OOD test fixtures directory not found.")

    food_path = ood_dir / "food_pasta_plate.jpg"
    if food_path.exists():
        with open(food_path, "rb") as f:
            food_bytes = f.read()

        response = client.post(
            "/predict",
            files={"file": ("food.jpg", food_bytes, "image/jpeg")},
            data={"model": "resnet18"},
        )
        assert response.status_code == 200
        res = response.json()
        assert res["status"] in ("unsupported", "potentially_unsupported")
        assert res["is_confident"] is False
        assert "This image does not appear suitable" in res["uncertainty_reason"] or "uncertainty threshold" in res["uncertainty_reason"]

    indoor_path = ood_dir / "indoor_living_room.jpg"
    if indoor_path.exists():
        with open(indoor_path, "rb") as f:
            indoor_bytes = f.read()

        response = client.post(
            "/predict",
            files={"file": ("indoor.jpg", indoor_bytes, "image/jpeg")},
            data={"model": "resnet18"},
        )
        assert response.status_code == 200
        res = response.json()
        assert res["status"] in ("unsupported", "potentially_unsupported")
        assert res["is_confident"] is False


def test_dog_image_rejected_on_custom_cnn(client: TestClient):
    """Regression test: Verify ordinary dog photograph is rejected on Custom CNN."""
    dog_path = Path("tests/fixtures/ood/dog_golden_retriever.jpg")
    if not dog_path.exists():
        pytest.skip("Dog fixture not found.")

    with open(dog_path, "rb") as f:
        img_bytes = f.read()

    # Even though Custom CNN without suitability filter assigns >67% confidence,
    # the pre-classifier suitability stage must intercept and flag it as unsupported.
    response = client.post(
        "/predict",
        files={"file": ("dog.jpg", img_bytes, "image/jpeg")},
        data={"model": "custom_cnn"},
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "unsupported"
    assert res["is_confident"] is False
    assert res["is_suitable"] is False
    assert "This image does not appear suitable for satellite land-use classification" in res["uncertainty_reason"]
    assert "Please upload a satellite image containing one of the supported EuroSAT land-use classes" in res["uncertainty_reason"]


def test_dog_image_rejected_on_resnet18(client: TestClient):
    """Regression test: Verify ordinary dog photograph is rejected on ResNet-18."""
    dog_path = Path("tests/fixtures/ood/dog_golden_retriever.jpg")
    if not dog_path.exists():
        pytest.skip("Dog fixture not found.")

    with open(dog_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": ("dog.jpg", img_bytes, "image/jpeg")},
        data={"model": "resnet18"},
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "unsupported"
    assert res["is_confident"] is False
    assert res["is_suitable"] is False


def test_suitability_detector_direct_evaluation():
    """Unit test for ImageSuitabilityDetector stages and metrics."""
    from src.inference.suitability import ImageSuitabilityDetector
    detector = ImageSuitabilityDetector.get_instance()

    ood_dir = Path("tests/fixtures/ood")
    if not ood_dir.exists():
        pytest.skip("OOD test fixtures directory not found.")

    # 1. Dog photograph must fail domain distance
    dog_res = detector.evaluate(ood_dir / "dog_golden_retriever.jpg")
    assert dog_res.is_suitable is False
    assert dog_res.status == "unsupported"
    assert dog_res.stage_failed == "domain_distance"
    assert dog_res.min_distance > detector.distance_threshold
    assert "This image does not appear suitable" in dog_res.message

    # 2. Portrait, food, indoor room must fail domain distance
    for name in ["human_portrait.jpg", "food_pasta_plate.jpg", "indoor_living_room.jpg"]:
        p = ood_dir / name
        if p.exists():
            res = detector.evaluate(p)
            assert res.is_suitable is False
            assert res.status == "unsupported"
            assert res.stage_failed == "domain_distance"

    # 3. Degenerate blank white and uniform noise must fail structural sanity
    for name in ["blank_white.jpg", "random_uniform_noise.jpg"]:
        p = ood_dir / name
        if p.exists():
            res = detector.evaluate(p)
            assert res.is_suitable is False
            assert res.status == "unsupported"
            assert res.stage_failed == "structural_sanity"

    # 4. Valid EuroSAT satellite tile must pass
    valid_tile = RAW_DATA_DIR / "EuroSAT_RGB" / "Forest" / "Forest_1.jpg"
    if valid_tile.exists():
        res_valid = detector.evaluate(valid_tile)
        assert res_valid.is_suitable is True
        assert res_valid.status == "accepted"
        assert res_valid.stage_failed is None
        assert res_valid.min_distance <= detector.distance_threshold


# -------------------------------------------------------------------------
# Step 5.5: API Threshold Customization & Validation
# -------------------------------------------------------------------------

def test_api_predict_with_custom_threshold_override(client: TestClient):
    """Verify caller can pass custom confidence_threshold to override default."""
    forest_tile = RAW_DATA_DIR / "EuroSAT_RGB" / "Forest" / "Forest_1.jpg"
    if not forest_tile.exists():
        pytest.skip("EuroSAT raw data directory not available.")

    with open(forest_tile, "rb") as f:
        img_bytes = f.read()

    # Very high threshold (0.9999) forces even valid images to be flagged as uncertain
    files = {"file": ("Forest_1.jpg", img_bytes, "image/jpeg")}
    response = client.post(
        "/predict",
        files=files,
        data={"model": "resnet18", "confidence_threshold": 0.9999},
    )
    assert response.status_code == 200
    res = response.json()
    assert res["threshold_used"] == 0.9999

    # Invalid threshold (> 1.0) must return 400 Bad Request
    response_bad = client.post(
        "/predict",
        files=files,
        data={"model": "resnet18", "confidence_threshold": 1.5},
    )
    assert response_bad.status_code == 400
    assert "between 0.0 and 1.0" in response_bad.json()["detail"].lower()
