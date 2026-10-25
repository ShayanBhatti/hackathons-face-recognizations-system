"""
Face Attendance API Main Application

This module initializes and configures the FastAPI application for the
Face Attendance System, including middleware, exception handling,
and API routing.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .api.routes import get_face_detector_service, router
from .api.records import router as records_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    This handles application startup and shutdown, including initialization
    of services and cleanup of resources.
    """
    logger.info("Starting Face Attendance API")

    try:
        detector = get_face_detector_service()
        if detector.is_model_available():
            logger.info("Face detector initialized successfully")
        else:
            logger.warning("Face detector failed to initialize")

    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")

    yield

    logger.info("Shutting down Face Attendance API")


app = FastAPI(
    title="Face Attendance API",
    version="0.1.0",
    description="API for face detection and attendance management",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(records_router, prefix="/api")


@app.get("/")
async def root():
    """Serve the HTML attendance dashboard."""
    template = Path(__file__).parent / "templates" / "dashboard.html"
    return FileResponse(template, media_type="text/html")



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions.

    This provides consistent error responses without exposing
    sensitive server information.
    """
    logger.error(f"Unhandled exception: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred",
        },
    )