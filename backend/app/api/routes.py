"""
API Routes for Face Detection

This module provides the FastAPI routes for the face detection API,
including the /api/detection endpoint and health checks.
"""

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from ..services.face_detector import FaceDetectorService, FaceDetectionConfig

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()


def get_face_detector_service() -> FaceDetectorService:
    """
    Dependency to get or create the face detector service instance.

    This uses FastAPI's dependency injection to ensure the service
    is loaded once at startup and reused across requests.
    """
    config = FaceDetectionConfig()

    model_path = config.get_model_path()
    device = config.get_device()
    confidence_threshold = config.get_confidence_threshold()
    iou_threshold = config.get_iou_threshold()

    detector = FaceDetectorService(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )

    return detector


@router.get("/health")
async def health_check(detector: FaceDetectorService = Depends(get_face_detector_service)):
    """
    Health check endpoint.

    Returns the status of the face detector service and system information.
    """
    is_model_available = detector.is_model_available()

    health_status = {
        "status": "healthy" if is_model_available else "unhealthy",
        "model_available": is_model_available,
        "device": detector.device,
        "confidence_threshold": detector.confidence_threshold,
        "model_path": str(detector.model_path) if is_model_available else None,
    }

    status_code = status.HTTP_200_OK if is_model_available else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=health_status)


@router.post("/detection")
async def detect_faces(
    file: UploadFile = File(..., description="Image file to analyze for faces"),
    detector: FaceDetectorService = Depends(get_face_detector_service),
):
    """
    Detect faces in an uploaded image.

    This endpoint accepts image uploads, validates them, runs face detection
    using the YOLOv8 model, and returns the detected face bounding boxes
    and confidence scores.

    Args:
        file: Uploaded image file (PNG, JPG, JPEG)
        detector: Face detector service instance

    Returns:
        JSON response with detection results
    """
    if not detector.is_model_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face detection service is not available",
        )

    try:
        content_type = file.content_type
        file_name = file.filename

        logger.info(f"Received face detection request for file: {file_name}")
        logger.info(f"Content-Type: {content_type}")

        if not file_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided in upload",
            )

        allowed_extensions = os.environ.get("ALLOWED_EXTENSIONS", "png,jpg,jpeg").split(",")
        if not any(file_name.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
            )

        max_size = int(os.environ.get("MAX_FILE_SIZE", "10485760"))
        file_size = 0
        image_bytes = b""

        chunks = []
        for chunk in iter(lambda: file.file.read(8192), b""):
            if not chunk:
                break
            chunks.append(chunk)
            file_size += len(chunk)

        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {max_size // 1024 // 1024}MB limit",
            )

        image_bytes = b"".join(chunks)

        try:
            import cv2
            import numpy as np

            nparr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image file or corrupted image",
                )

            image_height, image_width, _ = img.shape
            logger.info(f"Image dimensions: {image_width}x{image_height}")

        except Exception as e:
            logger.error(f"Image decoding failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode image file",
            )

        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / file_name

        with open(temp_file_path, "wb") as f:
            f.write(image_bytes)

        try:
            success, response, error_message = detector.detect_faces(
                image_path=temp_file_path, image_bytes=image_bytes
            )

            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_message or "Face detection failed",
                )

            return response.model_dump()

        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in face detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during face detection",
        )




