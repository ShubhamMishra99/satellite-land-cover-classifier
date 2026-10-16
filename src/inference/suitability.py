"""Image suitability and out-of-distribution (OOD) detection for satellite land-use classification.

Evaluates whether an incoming image is suitable EuroSAT-like satellite imagery
before passing it to downstream classifiers (ResNet-18 or Custom CNN).

Two-Stage Suitability Architecture:
-----------------------------------
Stage 1: Low-Level Structural Sanity Filter
  - Rejects degenerate inputs such as blank/solid fields, zero-variance frames,
    extreme single-color saturations, and pure unphysical high-frequency noise.
Stage 2: Deep Semantic Domain Distance Filter
  - Computes embedding representations using a visual feature extractor and
    measures minimum cosine distance to empirical EuroSAT class prototypes
    derived from the training dataset.
  - Images whose representations exceed the calibrated suitability threshold
    (e.g., ordinary dog photographs, human portraits, food, indoor scenes) are
    flagged as unsupported before classification.
"""

from dataclasses import asdict, dataclass
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms

from src.constants import (
    EUROSAT_CLASSES,
    MODELS_DIR,
    SUITABILITY_DISTANCE_THRESHOLD,
    SUITABILITY_MAX_LAPLACIAN_VAR,
    SUITABILITY_MIN_ENTROPY,
    SUITABILITY_MIN_VARIANCE,
    SUITABILITY_PROTOTYPES_PATH,
    SUITABILITY_UNSUPPORTED_MESSAGE,
    SUITABILITY_UNSUPPORTED_TITLE,
)

logger = logging.getLogger(__name__)


@dataclass
class SuitabilityResult:
    """Structured result from the Image Suitability stage."""

    is_suitable: bool
    status: str  # "accepted" or "unsupported"
    title: Optional[str]
    message: Optional[str]
    min_distance: float
    threshold: float
    nearest_class: Optional[str]
    stage_failed: Optional[str]  # "structural_sanity", "domain_distance", or None
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImageSuitabilityDetector:
    """Evaluates image suitability and detects out-of-domain imagery.

    Executes a two-stage evaluation:
      1. Structural Sanity: verifies pixel variance, information entropy, and texture gradient bounds.
      2. Semantic Domain Distance: measures feature cosine distance to EuroSAT class prototypes.
    """

    _instance: Optional["ImageSuitabilityDetector"] = None

    def __init__(
        self,
        prototypes_path: Optional[Union[str, Path]] = None,
        distance_threshold: float = SUITABILITY_DISTANCE_THRESHOLD,
        min_variance: float = SUITABILITY_MIN_VARIANCE,
        min_entropy: float = SUITABILITY_MIN_ENTROPY,
        max_laplacian_var: float = SUITABILITY_MAX_LAPLACIAN_VAR,
        device: Optional[torch.device] = None,
        backbone: Optional[nn.Module] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distance_threshold = float(distance_threshold)
        self.min_variance = float(min_variance)
        self.min_entropy = float(min_entropy)
        self.max_laplacian_var = float(max_laplacian_var)

        self.prototypes_path = Path(prototypes_path or SUITABILITY_PROTOTYPES_PATH)
        self.class_prototypes: Optional[np.ndarray] = None
        self.cluster_prototypes: Optional[np.ndarray] = None

        self._load_prototypes()

        # Feature extractor backbone
        if backbone is not None:
            self.backbone = backbone.to(self.device)
        else:
            try:
                base_model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
                self.backbone = nn.Sequential(*(list(base_model.children())[:-1])).to(self.device)
            except Exception as exc:
                logger.warning(f"Could not load default ResNet18 weights ({exc}), initializing architecture: {exc}")
                base_model = tv_models.resnet18(weights=None)
                self.backbone = nn.Sequential(*(list(base_model.children())[:-1])).to(self.device)

        self.backbone.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info(
            f"ImageSuitabilityDetector initialized on {self.device} (threshold={self.distance_threshold})"
        )

    @classmethod
    def get_instance(cls) -> "ImageSuitabilityDetector":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_prototypes(self) -> None:
        """Load precomputed EuroSAT domain prototypes from disk."""
        if not self.prototypes_path.exists():
            logger.warning(
                f"Suitability prototypes file not found at {self.prototypes_path}. "
                "Semantic domain distance check will be disabled until prototypes are computed."
            )
            return

        try:
            data = torch.save if False else torch.load(self.prototypes_path, map_location="cpu")
            if "class_prototypes" in data:
                cp = data["class_prototypes"]
                self.class_prototypes = cp.numpy() if isinstance(cp, torch.Tensor) else np.array(cp)
            if "cluster_prototypes" in data:
                clp = data["cluster_prototypes"]
                self.cluster_prototypes = clp.numpy() if isinstance(clp, torch.Tensor) else np.array(clp)
            logger.info(
                f"Loaded suitability prototypes from {self.prototypes_path}: "
                f"{len(self.class_prototypes) if self.class_prototypes is not None else 0} classes, "
                f"{len(self.cluster_prototypes) if self.cluster_prototypes is not None else 0} clusters"
            )
        except Exception as exc:
            logger.error(f"Failed to load prototypes from {self.prototypes_path}: {exc}")

    def check_structural_sanity(
        self, pil_image: Image.Image
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Evaluate low-level image statistics for degenerate conditions.

        Args:
            pil_image: PIL Image object.

        Returns:
            Tuple of (is_pass, failure_reason_if_any, diagnostics_dict).
        """
        # Convert and resize to 64x64 for statistical analysis
        img_rgb = pil_image.convert("RGB").resize((64, 64))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0  # shape: (64, 64, 3)

        # 1. Pixel variance across all channels
        var_total = float(np.var(arr))
        if var_total < self.min_variance:
            return (
                False,
                f"Image pixel variance ({var_total:.6f}) is below minimum threshold ({self.min_variance}). Flat or blank image detected.",
                {"variance": var_total, "entropy": 0.0, "laplacian_variance": 0.0},
            )

        # 2. Grayscale conversion for texture and entropy
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Shannon Entropy (8-bit grayscale)
        gray_uint8 = (gray * 255).astype(np.uint8)
        hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        if entropy < self.min_entropy:
            return (
                False,
                f"Image entropy ({entropy:.3f}) is below minimum threshold ({self.min_entropy}). Monochromatic or uniform field detected.",
                {"variance": var_total, "entropy": entropy, "laplacian_variance": 0.0},
            )

        # 3. Laplacian Variance (edge frequency and unphysical noise check)
        lap = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        lap_var = float(np.var(lap))

        if lap_var > self.max_laplacian_var:
            return (
                False,
                f"Image high-frequency gradient variance ({lap_var:.4f}) exceeds threshold ({self.max_laplacian_var}). Unphysical noise pattern detected.",
                {"variance": var_total, "entropy": entropy, "laplacian_variance": lap_var},
            )

        return True, None, {"variance": var_total, "entropy": entropy, "laplacian_variance": lap_var}

    def compute_embedding(self, pil_image: Image.Image) -> np.ndarray:
        """Extract L2-normalized feature representation from the backbone."""
        img_rgb = pil_image.convert("RGB")
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.backbone(tensor).squeeze().cpu().numpy()

        if feat.ndim > 1:
            feat = feat.flatten()

        norm = float(np.linalg.norm(feat))
        if norm > 1e-8:
            feat = feat / norm
        return feat

    def compute_distance(self, embedding: np.ndarray) -> Tuple[float, Optional[str], Optional[float]]:
        """Calculate cosine distance to EuroSAT prototypes.

        Returns:
            Tuple of (class_min_distance, nearest_class_name, cluster_min_distance).
        """
        if self.class_prototypes is None:
            return 0.0, None, None

        # Cosine similarity = dot product of L2-normalized vectors
        sims = embedding @ self.class_prototypes.T  # shape (10,)
        nearest_idx = int(np.argmax(sims))
        nearest_class = EUROSAT_CLASSES[nearest_idx] if nearest_idx < len(EUROSAT_CLASSES) else None
        class_min_dist = float(1.0 - sims[nearest_idx])

        cluster_min_dist = None
        if self.cluster_prototypes is not None:
            cluster_sims = embedding @ self.cluster_prototypes.T
            cluster_min_dist = float(1.0 - cluster_sims.max())

        return class_min_dist, nearest_class, cluster_min_dist

    def evaluate(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        distance_threshold: Optional[float] = None,
    ) -> SuitabilityResult:
        """Run two-stage suitability evaluation on target image.

        Args:
            image_input: File path, bytes, or PIL Image.
            distance_threshold: Optional threshold override.

        Returns:
            SuitabilityResult indicating whether image is suitable for satellite classification.
        """
        # Resolve PIL Image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
            pil_img.load()
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
            pil_img.load()
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        thresh = self.distance_threshold if distance_threshold is None else float(distance_threshold)

        # Stage 1: Structural Sanity Check
        is_pass, reason, stats = self.check_structural_sanity(pil_img)
        if not is_pass:
            return SuitabilityResult(
                is_suitable=False,
                status="unsupported",
                title=SUITABILITY_UNSUPPORTED_TITLE,
                message=SUITABILITY_UNSUPPORTED_MESSAGE,
                min_distance=1.0,
                threshold=thresh,
                nearest_class=None,
                stage_failed="structural_sanity",
                diagnostics={**stats, "reason_internal": reason},
            )

        # Stage 2: Deep Semantic Domain Distance
        if self.class_prototypes is not None:
            embedding = self.compute_embedding(pil_img)
            min_dist, nearest_class, cluster_dist = self.compute_distance(embedding)

            diagnostics = {
                **stats,
                "domain_distance": round(min_dist, 4),
                "cluster_distance": round(cluster_dist, 4) if cluster_dist is not None else None,
                "nearest_class": nearest_class,
            }

            if min_dist > thresh:
                return SuitabilityResult(
                    is_suitable=False,
                    status="unsupported",
                    title=SUITABILITY_UNSUPPORTED_TITLE,
                    message=SUITABILITY_UNSUPPORTED_MESSAGE,
                    min_distance=round(min_dist, 4),
                    threshold=thresh,
                    nearest_class=nearest_class,
                    stage_failed="domain_distance",
                    diagnostics=diagnostics,
                )
        else:
            min_dist = 0.0
            nearest_class = None
            diagnostics = stats

        return SuitabilityResult(
            is_suitable=True,
            status="accepted",
            title=None,
            message=None,
            min_distance=round(min_dist, 4),
            threshold=thresh,
            nearest_class=nearest_class,
            stage_failed=None,
            diagnostics=diagnostics,
        )
