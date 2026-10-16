"""Image prediction and sample tile management API routes."""

import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.predict import PredictionResponse, SampleImageItem, Top3Item
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Classification"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "application/octet-stream"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/predict", response_model=PredictionResponse)
async def predict_satellite_image(
    file: UploadFile = File(..., description="Satellite tile image file (.jpg, .jpeg, .png)"),
    model: str = Form("resnet18", description="Target model: 'resnet18' or 'custom_cnn'"),
    confidence_threshold: Optional[float] = Form(
        None,
        description="Optional confidence threshold in range [0.0, 1.0]. If omitted, uses model's validated operating threshold.",
    ),
) -> PredictionResponse:
    """Classify an uploaded satellite image into one of 10 EuroSAT land-use classes.

    Processes real uploaded image using trained PyTorch checkpoints with deterministic
    preprocessing, Softmax probability ranking, and configurable uncertainty detection.
    """
    # Validate extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext and file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{file_ext}'. Allowed formats: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Read bytes safely
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file contents: {exc}")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    # Validate model selection
    norm_model = model.lower().strip()
    if norm_model not in ("resnet18", "custom_cnn", "resnet_18", "cnn"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model selection '{model}'. Choose 'resnet18' or 'custom_cnn'.",
        )

    # Validate confidence threshold if provided
    if confidence_threshold is not None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid confidence_threshold '{confidence_threshold}'. Must be a float between 0.0 and 1.0.",
            )

    service = ModelService.get_instance()
    try:
        result = service.predict(
            contents,
            model_name=norm_model,
            confidence_threshold=confidence_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {exc}")
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Internal inference failure: {exc}")

    top3_items = [
        Top3Item(
            rank=item["rank"],
            class_name=item["class"],
            probability=item["probability"],
            percentage=item["percentage"],
        )
        for item in result["top3"]
    ]

    return PredictionResponse(
        status=result["status"],
        is_confident=result["is_confident"],
        is_suitable=result.get("is_suitable", True),
        threshold_used=result["threshold_used"],
        uncertainty_reason=result["uncertainty_reason"],
        predicted_class=result["predicted_class"],
        predicted_index=result["predicted_index"],
        confidence=result["confidence"],
        confidence_percentage=result["confidence_percentage"],
        top3=top3_items,
        all_probabilities=result["all_probabilities"],
        model_name=result["model_name"],
        model_display_name=result["model_display_name"],
        original_size=result["original_size"],
        inference_time_ms=result["inference_time_ms"],
        class_color=result["class_color"],
        suitability_stage_failed=result.get("suitability_stage_failed"),
        suitability_distance=result.get("suitability_distance"),
        suitability_threshold=result.get("suitability_threshold"),
    )


@router.get("/samples", response_model=List[SampleImageItem])
def get_sample_images() -> List[SampleImageItem]:
    """Retrieve list of curated sample satellite tiles available locally for one-click testing."""
    service = ModelService.get_instance()
    samples = service.get_sample_images()
    return [SampleImageItem(**s) for s in samples]


@router.get("/samples/{class_name}/{filename}")
def serve_sample_image(class_name: str, filename: str) -> FileResponse:
    """Serve sample image file bytes directly for preview and testing."""
    service = ModelService.get_instance()
    path = service.get_sample_image_path(class_name, filename)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")

    media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return FileResponse(path, media_type=media_type)
