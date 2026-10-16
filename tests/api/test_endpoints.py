"""API integration tests for FastAPI backend endpoints."""

import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app
from src.constants import EUROSAT_CLASSES, RAW_DATA_DIR


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def valid_image_bytes() -> bytes:
    """Provide real EuroSAT satellite tile bytes for valid prediction testing."""
    sample_path = RAW_DATA_DIR / "EuroSAT_RGB" / "Forest" / "Forest_1.jpg"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            return f.read()
    # Fallback to realistic non-degenerate image
    im = Image.new("RGB", (64, 64), color=(34, 139, 34))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint(client: TestClient):
    """Test GET /health returns expected schema and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "models" in data
    assert "resnet18" in data["models"]
    assert "custom_cnn" in data["models"]
    assert data["num_classes"] == 10


def test_classes_endpoint(client: TestClient):
    """Test GET /classes returns all 10 canonical EuroSAT classes."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 10
    classes = [c["name"] for c in data["classes"]]
    assert classes == list(EUROSAT_CLASSES)


def test_predict_endpoint_resnet18(client: TestClient, valid_image_bytes: bytes):
    """Test POST /predict with default model (ResNet-18) returns structured predictions."""
    files = {"file": ("test_tile.jpg", valid_image_bytes, "image/jpeg")}
    data = {"model": "resnet18"}
    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["predicted_class"] in EUROSAT_CLASSES
    assert 0 <= res["predicted_index"] < 10
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top3"]) == 3
    assert len(res["all_probabilities"]) == 10
    assert res["model_name"] == "resnet18"
    assert "inference_time_ms" in res
    assert "disclaimer" in res
    assert res["status"] == "accepted"
    assert res["is_confident"] is True
    assert res["threshold_used"] == 0.60


def test_predict_endpoint_custom_cnn(client: TestClient, valid_image_bytes: bytes):
    """Test POST /predict with Custom CNN model option."""
    files = {"file": ("test_tile.jpg", valid_image_bytes, "image/jpeg")}
    data = {"model": "custom_cnn"}
    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["predicted_class"] in EUROSAT_CLASSES
    assert res["model_name"] == "custom_cnn"
    assert len(res["top3"]) == 3
    assert res["status"] in ("accepted", "potentially_unsupported")
    assert res["threshold_used"] == 0.40


def test_predict_endpoint_empty_file(client: TestClient):
    """Test POST /predict with 0-byte file rejects with 400 Bad Request."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_endpoint_corrupt_file(client: TestClient):
    """Test POST /predict with invalid non-image payload rejects with 400 Bad Request."""
    files = {"file": ("bad.jpg", b"not-an-image-data-at-all", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "failed" in response.json()["detail"].lower()


def test_predict_endpoint_invalid_model(client: TestClient, valid_image_bytes: bytes):
    """Test POST /predict with unsupported model name returns 400 Bad Request."""
    files = {"file": ("test_tile.jpg", valid_image_bytes, "image/jpeg")}
    data = {"model": "non_existent_model"}
    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 400
    assert "invalid model" in response.json()["detail"].lower()


def test_samples_endpoint(client: TestClient):
    """Test GET /samples returns demo sample metadata list."""
    response = client.get("/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    if len(samples) > 0:
        s = samples[0]
        assert "class_name" in s
        assert "url" in s
        assert "filename" in s


def test_real_eurosat_image_end_to_end(client: TestClient):
    """Test end-to-end prediction using an actual EuroSAT RGB image from disk."""
    sample_path = RAW_DATA_DIR / "EuroSAT_RGB" / "Forest" / "Forest_1.jpg"
    if not sample_path.exists():
        pytest.skip("EuroSAT raw dataset not found on disk")

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("Forest_1.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, data={"model": "resnet18"})
    assert response.status_code == 200
    res = response.json()

    # The actual model predicts Forest with high confidence
    assert res["predicted_class"] == "Forest"
    assert res["confidence"] > 0.8
    assert res["top3"][0]["class"] == "Forest"
    assert res["status"] == "accepted"
    assert res["is_confident"] is True
    assert res["is_suitable"] is True


def test_predict_endpoint_unsupported_dog_image_custom_cnn(client: TestClient):
    """Verify dog photo uploaded to Custom CNN is flagged as unsupported and NOT accepted."""
    dog_path = Path("tests/fixtures/ood/dog_golden_retriever.jpg")
    if not dog_path.exists():
        pytest.skip("Dog test fixture not found")

    with open(dog_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("dog.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, data={"model": "custom_cnn"})
    assert response.status_code == 200
    res = response.json()

    assert res["status"] == "unsupported"
    assert res["is_confident"] is False
    assert res["is_suitable"] is False
    assert "This image does not appear suitable for satellite land-use classification" in res["uncertainty_reason"]
    assert "Please upload a satellite image containing one of the supported EuroSAT land-use classes" in res["uncertainty_reason"]


def test_predict_endpoint_unsupported_dog_image_resnet18(client: TestClient):
    """Verify dog photo uploaded to ResNet-18 is flagged as unsupported and NOT accepted."""
    dog_path = Path("tests/fixtures/ood/dog_golden_retriever.jpg")
    if not dog_path.exists():
        pytest.skip("Dog test fixture not found")

    with open(dog_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("dog.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, data={"model": "resnet18"})
    assert response.status_code == 200
    res = response.json()

    assert res["status"] == "unsupported"
    assert res["is_confident"] is False
    assert res["is_suitable"] is False
    assert "This image does not appear suitable for satellite land-use classification" in res["uncertainty_reason"]


def test_predict_endpoint_unsupported_portraits_food_indoor(client: TestClient):
    """Verify non-satellite photographs (portrait, food, indoor) are flagged as unsupported."""
    ood_fixtures = [
        "human_portrait.jpg",
        "food_pasta_plate.jpg",
        "indoor_living_room.jpg",
    ]
    for fixture_name in ood_fixtures:
        fixture_path = Path("tests/fixtures/ood") / fixture_name
        if not fixture_path.exists():
            continue

        with open(fixture_path, "rb") as f:
            img_bytes = f.read()

        for model_choice in ("resnet18", "custom_cnn"):
            files = {"file": (fixture_name, img_bytes, "image/jpeg")}
            response = client.post("/predict", files=files, data={"model": model_choice})
            assert response.status_code == 200
            res = response.json()
            assert res["status"] == "unsupported", f"Expected unsupported for {fixture_name} on {model_choice}"
            assert res["is_confident"] is False
            assert res["is_suitable"] is False
            assert "This image does not appear suitable" in res["uncertainty_reason"]
