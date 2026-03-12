from __future__ import annotations

import importlib
import platform
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def check_module(name: str):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:
        return False, str(exc)


def check_camera() -> str:
    try:
        import cv2
    except Exception as exc:
        return f"cv2 manquant: {exc}"

    cam = cv2.VideoCapture(settings.camera_index)
    try:
        if not cam.isOpened():
            return f"camera index {settings.camera_index} non ouverte"
        ok, _ = cam.read()
        if not ok:
            return "camera ouverte mais lecture frame echouee"
        return "camera OK (opencv)"
    finally:
        cam.release()


def check_db() -> str:
    if not settings.db_path.exists():
        return f"DB absente: {settings.db_path}"
    try:
        conn = sqlite3.connect(str(settings.db_path))
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in c.fetchall()]
        conn.close()
        return "tables=" + ", ".join(tables) if tables else "aucune table"
    except Exception as exc:
        return f"DB erreur: {exc}"


def main() -> int:
    print("=== ENVIRONNEMENT ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"os: {platform.platform()}")
    print(f"arch: {platform.machine()}")
    print(f"raspberry_pi: {settings.is_raspberry_pi}")
    print(f"cwd: {Path.cwd()}")
    print()

    print("=== CONFIG PROJET ===")
    print(f"base_dir: {settings.base_dir}")
    print(f"db_path: {settings.db_path} (exists={settings.db_path.exists()})")
    print(f"backend demande: {settings.backend}")
    print(f"camera_mode: {settings.camera_mode} camera_index: {settings.camera_index}")
    print(f"pt_candidates:")
    for candidate in settings.pt_candidates:
        print(f"  - {candidate} (exists={candidate.exists()})")
    print(f"onnx_model: {settings.onnx_model} (exists={settings.onnx_model.exists()})")
    print()

    print("=== MODULES ===")
    for module_name in ["flask", "cv2", "numpy", "ultralytics", "torch", "onnxruntime", "reportlab"]:
        ok, info = check_module(module_name)
        status = "OK" if ok else "KO"
        print(f"{module_name:12} {status:2} {info}")
    print()

    print("=== CAMERA ===")
    print(check_camera())
    print()

    print("=== SQLITE ===")
    print(check_db())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
