"""
YOLOv8 Face Detection Service

This module provides a face detection service using YOLOv8 models.
It handles model loading, inference, and result processing with error handling
and logging.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from ultralytics import YOLO

load_dotenv()

logger = logging.getLogger(__name__)


class DetectionResult(BaseModel):
    """Pydantic model for detection results."""

    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")
    confidence: float = Field(..., description="Detection confidence score (0-1)")
    class_name: str = Field(..., description="Detected class name")
    class_id: int = Field(..., description="Detected class ID")


class DetectionResponse(BaseModel):
    """Pydantic model for the API response."""

    success: bool = Field(..., description="Whether the detection was successful")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    detection_count: int = Field(..., description="Number of faces detected")
    confidence_threshold: float = Field(..., description="Confidence threshold used")
    faces: Optional[List[DetectionResult]] = Field(None, description="Detected faces")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class FaceDetectorService:
    """
    Face Detection Service using YOLOv8.

    This service loads a YOLOv8 model once at startup and provides
    face detection capabilities with configurable device selection
    (CPU or CUDA) and inference parameters.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.7,
    ):
        """
        Initialize the face detection service.

        Args:
            model_path: Path to the YOLOv8 model checkpoint
            device: Device to run inference on (None for auto, 'cuda', 'cpu')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        logger.info(f"Initializing Face Detector Service")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"IoU threshold: {self.iou_threshold}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Device: {self.device}")

        try:
            logger.info("Loading YOLO model...")
            self.model = YOLO(str(self.model_path), task="detect")
            self.model.to(self.device)

            logger.info("Model loaded successfully")
            logger.info(f"Model device: {next(self.model.model.parameters()).device}")

            self.is_initialized = True
            self.model_available = True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_initialized = False
            self.model_available = False

    def is_model_available(self) -> bool:
        """Check if the model is available and loaded."""
        return self.model_available and self.is_initialized

    def detect_faces(
        self, image_path: Path, image_bytes: Optional[bytes] = None
    ) -> Tuple[bool, DetectionResponse, Optional[str]]:
        """
        Detect faces in an image.

        Args:
            image_path: Path to the image file
            image_bytes: Optional image bytes for memory efficiency

        Returns:
            Tuple of (success, detection_response, error_message)
        """
        if not self.is_model_available():
            error_msg = "Model not available - initialization failed"
            logger.error(error_msg)
            return False, DetectionResponse(
                success=False,
                image_width=0,
                image_height=0,
                detection_count=0,
                confidence_threshold=self.confidence_threshold,
                faces=None,
                processing_time_ms=None,
            ), error_msg

        try:
            import cv2

            logger.info(f"Running face detection on: {image_path}")

            import time
            use_cuda_timing = torch.cuda.is_available()
            if use_cuda_timing:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
            else:
                start_time = time.perf_counter()

            results = self.model(
                str(image_path),
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
            )

            if use_cuda_timing:
                end_time.record()
                torch.cuda.synchronize()
                processing_time_ms = start_time.elapsed_time(end_time)
            else:
                processing_time_ms = (time.perf_counter() - start_time) * 1000

            result = results[0]
            faces = []

            if result.boxes is not None:
                boxes = result.boxes
                class_ids = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                logger.info(f"Detected {len(xyxy)} faces")

                for i, (class_id, confidence, bbox) in enumerate(
                    zip(class_ids, confidences, xyxy)
                ):
                    x1, y1, x2, y2 = bbox

                    logger.info(
                        f"Face {i+1}: bbox=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}), "
                        f"conf={confidence:.4f}, class_id={int(class_id)}"
                    )

                    face = DetectionResult(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(confidence),
                        class_name="face",
                        class_id=int(class_id),
                    )
                    faces.append(face)

            response = DetectionResponse(
                success=True,
                image_width=int(result.orig_shape[1]),
                image_height=int(result.orig_shape[0]),
                detection_count=len(faces),
                confidence_threshold=self.confidence_threshold,
                faces=faces,
                processing_time_ms=processing_time_ms,
            )

            return True, response, None

        except Exception as e:
            error_msg = f"Face detection failed: {str(e)}"
            logger.error(error_msg)
            return False, DetectionResponse(
                success=False,
                image_width=0,
                image_height=0,
                detection_count=0,
                confidence_threshold=self.confidence_threshold,
                faces=None,
                processing_time_ms=None,
            ), error_msg


class FaceDetectionConfig:
    """Configuration class for face detection service."""

    @staticmethod
    def get_model_path() -> str:
        """Get the model path from environment or use default."""
        model_path = os.environ.get("MODEL_PATH", "C:\\Users\\shaya\\Desktop\\Hackathon\\face-attendance\\ml\\best.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please ensure the YOLOv8 face detection model is available."
            )

        logger.info(f"Using model path: {model_path}")
        return model_path

    @staticmethod
    def get_device() -> Optional[str]:
        """Get the device to use for inference."""
        device = os.environ.get("MODEL_DEVICE")
        if device:
            logger.info(f"Using device from environment: {device}")
            return device

        if torch.cuda.is_available():
            logger.info("CUDA available, using 'cuda'")
            return "cuda"
        else:
            logger.info("CUDA not available, using 'cpu'")
            return "cpu"

    @staticmethod
    def get_confidence_threshold() -> float:
        """Get confidence threshold from environment."""
        return float(os.environ.get("MODEL_CONFIDENCE", "0.5"))

    @staticmethod
    def get_iou_threshold() -> float:
        """Get IoU threshold from environment."""
        return float(os.environ.get("MODEL_IOU", "0.7"))
