from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings
from detector import CameraSource, choose_backend


class WasteDetector:
    """Compat layer utilisee par app.py avec backend unifie PT/ONNX."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        db_path: Optional[str] = None,
    ):
        self._backend_name = (backend or settings.backend).lower()
        self._confidence = confidence_threshold or settings.confidence_threshold
        self._db_path = Path(db_path) if db_path else settings.db_path
        self._class_map = dict(settings.waste_classes)
        self._last_error = ""

        pt_candidates = list(settings.pt_candidates)
        if model_path:
            custom_path = Path(model_path)
            if not custom_path.is_absolute():
                custom_path = settings.base_dir / custom_path
            pt_candidates = [custom_path] + pt_candidates

        self._backend, self._last_error = choose_backend(
            requested_backend=self._backend_name,
            confidence=self._confidence,
            class_map=self._class_map,
            onnx_model=settings.onnx_model,
            pt_candidates=pt_candidates,
        )

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return "unavailable"
        return self._backend.name

    @property
    def last_error(self) -> str:
        return self._last_error

    def is_ready(self) -> bool:
        return self._backend is not None

    def detect_from_image(self, image_path: str):
        if not self._backend:
            return None, self._last_error or "Aucun backend disponible"

        image_file = Path(image_path)
        if not image_file.exists():
            return None, f"Image introuvable: {image_file}"

        image = cv2.imread(str(image_file))
        if image is None:
            return None, f"Impossible de lire l'image: {image_file}"

        try:
            detections = self._backend.detect(image)
        except Exception as exc:
            return None, f"Erreur inference ({self.backend_name}): {exc}"

        formatted = []
        for det in detections:
            formatted.append(
                {
                    "waste_type": det.label,
                    "confidence": det.confidence,
                    "box": np.array(det.box_xyxy, dtype=np.int32),
                }
            )
        return formatted, {"backend": self.backend_name}

    def detect_from_frame(self, frame: np.ndarray):
        if not self._backend:
            return frame, {}

        try:
            detections = self._backend.detect(frame)
        except Exception:
            return frame, {}

        summary: Dict[str, int] = {}
        for det in detections:
            summary[det.label] = summary.get(det.label, 0) + 1
            x1, y1, x2, y2 = det.box_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.putText(
                frame,
                f"{det.label} {det.confidence:.2f}",
                (x1, max(12, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 230, 0),
                2,
            )
        return frame, summary

    def detect_from_webcam(self, user_id: int, duration: int = 10):
        if not self._backend:
            return {}

        camera = CameraSource(mode=settings.camera_mode, camera_index=settings.camera_index)
        if not camera.open():
            return {}

        end_at = time.time() + max(1, int(duration))
        summary: Dict[str, int] = {}
        try:
            while time.time() < end_at:
                ok, frame = camera.read()
                if not ok or frame is None:
                    continue
                _, frame_summary = self.detect_from_frame(frame)
                for key, value in frame_summary.items():
                    summary[key] = summary.get(key, 0) + value
        finally:
            camera.release()
        return summary

    def save_detections_to_db(self, user_id: int, detections_dict: Dict[str, int]):
        if not detections_dict:
            return True
        conn = sqlite3.connect(str(self._db_path))
        try:
            c = conn.cursor()
            for waste_type, quantity in detections_dict.items():
                c.execute(
                    """
                    INSERT INTO waste_detection (user_id, waste_type, quantity, detection_date)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, waste_type, int(quantity), datetime.now()),
                )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
