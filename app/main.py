"""FastAPI Application Factory for Satellite Land-Use Classifier."""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.services.model_service import ModelService
from src.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    app = FastAPI(
        title="Satellite Land-Use Classifier API",
        description=(
            "Deep learning API for EuroSAT RGB land-use / land-cover classification. "
            "Supports Custom CNN baseline and fine-tuned ResNet-18 transfer learning."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register root routes
    app.include_router(health_router)
    app.include_router(predict_router)

    # Register /api prefixed routes for proxy compatibility
    app.include_router(health_router, prefix="/api")
    app.include_router(predict_router, prefix="/api")

    # Mount frontend static distribution if built
    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
        logger.info(f"Mounted frontend static files from: {frontend_dist}")

    @app.on_event("startup")
    async def startup_event():
        logger.info("Satellite Land-Use Classifier API starting up...")
        service = ModelService.get_instance()
        logger.info(f"Target compute device: {service.device}")

    return app


app = create_app()
