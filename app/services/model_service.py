"""Model execution service and singleton manager for inference."""

import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.constants import (
    CLASS_COLORS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EUROSAT_CLASSES,
    MODEL_CONFIDENCE_THRESHOLDS,
    MODELS_DIR,
    NUM_CLASSES,
    RAW_DATA_DIR,
)
from src.inference.predictor import (
    load_and_preprocess_image,
    load_trained_model,
    predict_single_tensor,
)
from src.inference.suitability import ImageSuitabilityDetector

logger = logging.getLogger(__name__)

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "AnnualCrop": "Regular agricultural fields with rotating annual crops (cereals, grains, vegetables).",
    "Forest": "Dense woodland, coniferous and broadleaf canopies with high biomass density.",
    "HerbaceousVegetation": "Natural unmanaged meadows, grasslands, steppes, and wild foliage.",
    "Highway": "Paved asphalt motorways, arterial transit routes, and road infrastructures.",
    "Industrial": "Commercial complexes, manufacturing facilities, warehouses, and storage depots.",
    "Pasture": "Managed grazing pastures, enclosed agricultural grasslands, and livestock ranges.",
    "PermanentCrop": "Perennial agroforestry, fruit orchards, vineyards, and olive groves.",
    "Residential": "High- and low-density urban housing, suburban subdivisions, and residential blocks.",
    "River": "Inland natural waterways, active river channels, and riparian streams.",
    "SeaLake": "Open water bodies, deep marine coasts, estuaries, reservoirs, and natural lakes.",
}


class ModelService:
    """Singleton service for cached PyTorch model execution and sample management."""

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: Dict[str, nn.Module] = {}
        self._checkpoints: Dict[str, Path] = {
            "resnet18": MODELS_DIR / "resnet18_best.pt",
            "custom_cnn": MODELS_DIR / "custom_cnn_best.pt",
        }
        self.suitability_detector = ImageSuitabilityDetector.get_instance()
        logger.info(f"Initialized ModelService on device: {self.device}")

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_model_available(self, model_name: str) -> bool:
        norm_name = "resnet18" if "resnet" in model_name.lower() else "custom_cnn"
        ckpt_path = self._checkpoints.get(norm_name)
        return ckpt_path is not None and ckpt_path.exists()

    def get_model(self, model_name: str = "resnet18") -> nn.Module:
        norm_name = "resnet18" if "resnet" in model_name.lower() else "custom_cnn"

        if norm_name not in self._models:
            ckpt_path = self._checkpoints[norm_name]
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Model checkpoint for '{norm_name}' not found at: {ckpt_path}")

            logger.info(f"Loading '{norm_name}' into memory on {self.device}...")
            model = load_trained_model(
                model_name=norm_name,
                checkpoint_path=ckpt_path,
                device=self.device,
            )
            model.eval()
            self._models[norm_name] = model

        return self._models[norm_name]

    def get_default_threshold(self, model_name: str) -> float:
        """Return the validated uncertainty threshold for the specified model."""
        norm_name = "resnet18" if "resnet" in model_name.lower() else "custom_cnn"
        return MODEL_CONFIDENCE_THRESHOLDS.get(norm_name, DEFAULT_CONFIDENCE_THRESHOLD)

    def predict(
        self,
        image_bytes: bytes,
        model_name: str = "resnet18",
        confidence_threshold: Optional[float] = None,
        check_suitability: bool = True,
    ) -> Dict[str, Any]:
        """Process image bytes and execute inference with uncertainty and suitability awareness."""
        norm_name = "resnet18" if "resnet" in model_name.lower() else "custom_cnn"
        display_name = "ResNet-18 Transfer Learning" if norm_name == "resnet18" else "Custom CNN Baseline"

        # Determine effective confidence threshold
        operating_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.get_default_threshold(norm_name)
        )

        # Validate and open image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.load()
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image format: {exc}") from exc

        original_size = (pil_img.width, pil_img.height)

        # Stage 0: Pre-inference Image Suitability & Out-of-Distribution Pre-Filter
        suitability_res = None
        if check_suitability:
            suitability_res = self.suitability_detector.evaluate(pil_img)

        # Preprocess image deterministically using standard EuroSAT transform
        tensor, _ = load_and_preprocess_image(pil_img)

        # Get loaded model
        model = self.get_model(norm_name)

        # Timed forward pass
        start_time = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        pred_dict = predict_single_tensor(
            tensor=tensor,
            model=model,
            device=self.device,
            top_k=3,
            classes=EUROSAT_CLASSES,
            confidence_threshold=operating_threshold,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # If flagged by the Image Suitability Detector as unsupported:
        if suitability_res is not None and not suitability_res.is_suitable:
            return {
                "status": "unsupported",
                "is_confident": False,
                "is_suitable": False,
                "threshold_used": operating_threshold,
                "uncertainty_reason": suitability_res.message,
                "predicted_class": pred_dict["predicted_class"],
                "predicted_index": pred_dict["predicted_index"],
                "confidence": pred_dict["confidence"],
                "confidence_percentage": pred_dict["confidence_percentage"],
                "top3": pred_dict["top3"],
                "all_probabilities": pred_dict["all_probabilities"],
                "model_name": norm_name,
                "model_display_name": display_name,
                "original_size": original_size,
                "inference_time_ms": round(elapsed_ms, 2),
                "class_color": "#F59E0B",
                "suitability_stage_failed": suitability_res.stage_failed,
                "suitability_distance": suitability_res.min_distance,
                "suitability_threshold": suitability_res.threshold,
                "suitability_title": suitability_res.title,
            }

        predicted_class = pred_dict["predicted_class"]
        class_color = CLASS_COLORS.get(predicted_class, "#38BDF8")

        return {
            "status": pred_dict["status"],
            "is_confident": pred_dict["is_confident"],
            "is_suitable": True,
            "threshold_used": pred_dict["threshold_used"],
            "uncertainty_reason": pred_dict["uncertainty_reason"],
            "predicted_class": predicted_class,
            "predicted_index": pred_dict["predicted_index"],
            "confidence": pred_dict["confidence"],
            "confidence_percentage": pred_dict["confidence_percentage"],
            "top3": pred_dict["top3"],
            "all_probabilities": pred_dict["all_probabilities"],
            "model_name": norm_name,
            "model_display_name": display_name,
            "original_size": original_size,
            "inference_time_ms": round(elapsed_ms, 2),
            "class_color": class_color,
            "suitability_stage_failed": None,
            "suitability_distance": suitability_res.min_distance if suitability_res else None,
            "suitability_threshold": suitability_res.threshold if suitability_res else None,
            "suitability_title": None,
        }

    def get_classes_metadata(self) -> List[Dict[str, Any]]:
        """Return rich metadata for all 10 canonical EuroSAT classes."""
        return [
            {
                "index": idx,
                "name": cls_name,
                "color": CLASS_COLORS.get(cls_name, "#38BDF8"),
                "description": CLASS_DESCRIPTIONS.get(cls_name, ""),
            }
            for idx, cls_name in enumerate(EUROSAT_CLASSES)
        ]

    def get_sample_images(self) -> List[Dict[str, Any]]:
        """Return curated sample satellite tiles available locally for quick testing."""
        samples = []
        raw_dir = RAW_DATA_DIR / "EuroSAT_RGB"

        if raw_dir.exists():
            for cls_name in EUROSAT_CLASSES:
                cls_dir = raw_dir / cls_name
                if cls_dir.exists():
                    # Pick sample 1 or first available image
                    sample_file = cls_dir / f"{cls_name}_1.jpg"
                    if not sample_file.exists():
                        jpgs = list(cls_dir.glob("*.jpg"))
                        sample_file = jpgs[0] if jpgs else None

                    if sample_file and sample_file.exists():
                        samples.append({
                            "id": f"sample_{cls_name.lower()}",
                            "class_name": cls_name,
                            "filename": sample_file.name,
                            "display_name": f"{cls_name} Tile",
                            "color": CLASS_COLORS.get(cls_name, "#38BDF8"),
                            "url": f"/api/samples/{cls_name}/{sample_file.name}",
                        })
        return samples

    def get_sample_image_path(self, class_name: str, filename: str) -> Optional[Path]:
        """Resolve full filesystem path for a sample tile securely."""
        if class_name not in EUROSAT_CLASSES:
            return None
        safe_filename = Path(filename).name
        target = RAW_DATA_DIR / "EuroSAT_RGB" / class_name / safe_filename
        if target.exists() and target.is_file():
            return target
        return None
