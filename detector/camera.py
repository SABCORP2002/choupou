from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise RuntimeError(
        "OpenCV (cv2) est requis pour la camera. Installez opencv-python-headless."
    ) from exc


class CameraSource:
    """Abstraction camera: OpenCV d'abord, Picamera2 en option."""

    def __init__(self, mode: str = "auto", camera_index: int = 0):
        self.mode = mode
        self.camera_index = camera_index
        self._capture = None
        self._picam2 = None
        self._active_mode = None

    def open(self) -> bool:
        if self.mode in {"auto", "picamera2"}:
            if self._open_picamera2():
                return True
            if self.mode == "picamera2":
                return False
        return self._open_opencv()

    def _open_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2
        except Exception:
            return False

        try:
            self._picam2 = Picamera2()
            config = self._picam2.create_preview_configuration(main={"size": (640, 480)})
            self._picam2.configure(config)
            self._picam2.start()
            self._active_mode = "picamera2"
            return True
        except Exception:
            self._picam2 = None
            return False

    def _open_opencv(self) -> bool:
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            return False
        self._active_mode = "opencv"
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._active_mode == "picamera2" and self._picam2 is not None:
            try:
                frame = self._picam2.capture_array()
                if frame is None:
                    return False, None
                # Picamera2 retourne RGB: conversion BGR pour OpenCV/annot.
                return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                return False, None

        if self._capture is not None:
            return self._capture.read()
        return False, None

    def is_opened(self) -> bool:
        if self._active_mode == "picamera2":
            return self._picam2 is not None
        if self._capture is not None:
            return self._capture.isOpened()
        return False

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
        self._active_mode = None

    @property
    def active_mode(self) -> Optional[str]:
        return self._active_mode
