"""App routes package."""

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router

__all__ = ["health_router", "predict_router"]
