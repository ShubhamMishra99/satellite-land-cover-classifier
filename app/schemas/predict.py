"""Pydantic schemas for FastAPI endpoints."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class Top3Item(BaseModel):
    """Single candidate prediction item in Top-3 ranking."""

    rank: int = Field(..., description="Rank (1, 2, or 3)")
    class_name: str = Field(..., alias="class", description="Class name")
    probability: float = Field(..., description="Softmax probability in range [0, 1]")
    percentage: str = Field(..., description="Formatted probability percentage string")

    model_config = {
        "populate_by_name": True,
    }


class PredictionResponse(BaseModel):
    """Complete prediction output schema."""

    status: str = Field("accepted", description="Prediction status: 'accepted', 'unsupported', or 'potentially_unsupported'")
    is_confident: bool = Field(True, description="True if confidence is at or above uncertainty threshold")
    is_suitable: bool = Field(True, description="True if image passed image suitability / OOD detection stage")
    threshold_used: float = Field(..., description="Operating confidence threshold applied")
    uncertainty_reason: Optional[str] = Field(None, description="Explanation when prediction is uncertain or unsupported")
    predicted_class: str = Field(..., description="Canonical EuroSAT predicted class name")
    predicted_index: int = Field(..., description="Class integer index (0-9)")
    confidence: float = Field(..., description="Confidence score in range [0, 1]")
    confidence_percentage: str = Field(..., description="Formatted percentage string")
    top3: List[Top3Item] = Field(..., description="Top 3 ranked predictions")
    all_probabilities: Dict[str, float] = Field(
        ..., description="Probabilities across all 10 canonical classes"
    )
    model_name: str = Field(..., description="Model identifier used ('resnet18' or 'custom_cnn')")
    model_display_name: str = Field(..., description="Human-readable model name")
    original_size: Tuple[int, int] = Field(..., description="Original image dimensions (width, height)")
    inference_time_ms: float = Field(..., description="Model forward pass latency in milliseconds")
    class_color: str = Field(..., description="Hex color assigned to the predicted class")
    suitability_stage_failed: Optional[str] = Field(None, description="Suitability stage failed if unsupported")
    suitability_distance: Optional[float] = Field(None, description="Domain feature distance to EuroSAT prototypes")
    suitability_threshold: Optional[float] = Field(None, description="Suitability distance threshold")
    disclaimer: str = Field(
        "Confidence represents the model's predicted probability and does not guarantee that the classification is correct. Low-confidence predictions are flagged as potentially unsupported.",
        description="Scientific interpretation disclaimer",
    )



class HealthResponse(BaseModel):
    """Backend service health and readiness status."""

    status: str = Field("healthy", description="Service status ('healthy' or 'degraded')")
    models: Dict[str, bool] = Field(..., description="Availability status for trained model checkpoints")
    default_model: str = Field("resnet18", description="Default model used for inference")
    device: str = Field(..., description="Active compute device (e.g. 'cpu' or 'cuda')")
    version: str = Field("0.1.0", description="Application version")
    num_classes: int = Field(10, description="Total supported land-use classes")


class ClassItem(BaseModel):
    """Metadata for an individual EuroSAT class."""

    index: int
    name: str
    color: str
    description: str


class ClassesResponse(BaseModel):
    """List of all 10 canonical EuroSAT land-use classes with metadata."""

    classes: List[ClassItem]
    total_classes: int = 10
    dataset: str = "EuroSAT RGB (Sentinel-2 optical)"


class SampleImageItem(BaseModel):
    """Curated demo satellite tile available for one-click testing."""

    id: str
    class_name: str
    filename: str
    display_name: str
    color: str
    url: str
