"""Health, metadata, and evaluation metrics API routes."""

import json
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
import pandas as pd

from app.schemas.predict import ClassesResponse, ClassItem, HealthResponse
from app.services.model_service import ModelService
from src.constants import PROJECT_ROOT

router = APIRouter(tags=["System & Metadata"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check backend operational health and model checkpoint availability."""
    service = ModelService.get_instance()
    resnet_ok = service.is_model_available("resnet18")
    cnn_ok = service.is_model_available("custom_cnn")

    status = "healthy" if (resnet_ok or cnn_ok) else "degraded"

    return HealthResponse(
        status=status,
        models={
            "resnet18": resnet_ok,
            "custom_cnn": cnn_ok,
        },
        default_model="resnet18",
        device=str(service.device),
        version="0.1.0",
        num_classes=10,
    )


@router.get("/classes", response_model=ClassesResponse)
def get_classes() -> ClassesResponse:
    """Retrieve all 10 canonical EuroSAT land-use classes with color palettes."""
    service = ModelService.get_instance()
    classes_meta = service.get_classes_metadata()
    items = [ClassItem(**c) for c in classes_meta]
    return ClassesResponse(classes=items, total_classes=len(items))


@router.get("/model-metrics")
def get_model_metrics() -> Dict[str, Any]:
    """Retrieve actual Milestone 5 test evaluation metrics for both models."""
    eval_dir = PROJECT_ROOT / "reports" / "evaluation"
    comp_json_path = eval_dir / "model_comparison.json"
    per_class_csv_path = eval_dir / "per_class_metrics.csv"

    if not comp_json_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Please ensure Milestone 5 evaluation has run.",
        )

    with open(comp_json_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    per_class_data = []
    if per_class_csv_path.exists():
        df = pd.read_csv(per_class_csv_path)
        per_class_data = df.to_dict(orient="records")

    return {
        "comparison": comp_data,
        "per_class": per_class_data,
        "evaluated_samples": 4050,
        "split": "test (untouched)",
    }
