from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DB_PATH = BASE_DIR / "waste.db"
UPLOAD_DIR = STATIC_DIR / "uploads" / "profiles"

DEFAULT_WASTE_CLASSES: Dict[int, str] = {
    0: "Papier",
    1: "Plastique",
    2: "Metal",
    3: "Verre",
    4: "Carton",
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    models_dir: Path
    db_path: Path
    templates_dir: Path
    static_dir: Path
    upload_dir: Path
    backend: str
    confidence_threshold: float
    camera_mode: str
    camera_index: int
    flask_host: str
    flask_port: int
    flask_debug: bool
    secret_key: str
    waste_classes: Dict[int, str]
    pt_candidates: List[Path]
    onnx_model: Path

    @property
    def is_raspberry_pi(self) -> bool:
        machine = platform.machine().lower()
        if "arm" in machine or "aarch64" in machine:
            model_file = Path("/proc/device-tree/model")
            if model_file.exists():
                try:
                    return "raspberry pi" in model_file.read_text().lower()
                except OSError:
                    return False
        return False


def build_settings() -> Settings:
    backend = os.getenv("WASTEAI_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "pt", "onnx"}:
        backend = "auto"

    camera_mode = os.getenv("WASTEAI_CAMERA_MODE", "auto").strip().lower()
    if camera_mode not in {"auto", "opencv", "picamera2"}:
        camera_mode = "auto"

    pt_custom = MODELS_DIR / "my_model.pt"
    pt_fallback = MODELS_DIR / "yolov8n.pt"
    onnx_default = MODELS_DIR / "my_model.onnx"

    return Settings(
        base_dir=BASE_DIR,
        models_dir=MODELS_DIR,
        db_path=DB_PATH,
        templates_dir=TEMPLATES_DIR,
        static_dir=STATIC_DIR,
        upload_dir=UPLOAD_DIR,
        backend=backend,
        confidence_threshold=_env_float("WASTEAI_CONFIDENCE", 0.5),
        camera_mode=camera_mode,
        camera_index=_env_int("WASTEAI_CAMERA_INDEX", 0),
        flask_host=os.getenv("FLASK_HOST", "0.0.0.0"),
        flask_port=_env_int("FLASK_PORT", 5000),
        flask_debug=os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes"},
        secret_key=os.getenv("FLASK_SECRET_KEY", "wasteai-dev-secret"),
        waste_classes=DEFAULT_WASTE_CLASSES,
        pt_candidates=[pt_custom, pt_fallback],
        onnx_model=onnx_default,
    )


settings = build_settings()
