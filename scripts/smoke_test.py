from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def check_files() -> list[str]:
    required = [
        settings.base_dir / "app.py",
        settings.templates_dir / "login.html",
        settings.templates_dir / "dashboard.html",
        settings.static_dir / "css" / "style.css",
        settings.db_path,
    ]
    failures = []
    for path in required:
        if not path.exists():
            failures.append(f"Fichier requis manquant: {path}")
    return failures


def check_db() -> list[str]:
    failures = []
    if not settings.db_path.exists():
        return [f"Base SQLite absente: {settings.db_path}"]
    conn = sqlite3.connect(str(settings.db_path))
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in c.fetchall()}
        for needed in {"users", "waste_detection", "robots", "notifications"}:
            if needed not in tables:
                failures.append(f"Table SQLite manquante: {needed}")
    finally:
        conn.close()
    return failures


def check_flask_routes() -> list[str]:
    failures = []
    try:
        from app import app as flask_app
    except Exception as exc:
        return [f"Import app Flask echoue: {exc}"]

    client = flask_app.test_client()
    login_resp = client.get("/login")
    if login_resp.status_code >= 500:
        failures.append(f"/login retourne {login_resp.status_code}")

    health_resp = client.get("/")
    if health_resp.status_code >= 500:
        failures.append(f"/ retourne {health_resp.status_code}")
    return failures


def check_detector() -> list[str]:
    failures = []
    try:
        from yolo_detector import WasteDetector
    except Exception as exc:
        return [f"Import detecteur echoue: {exc}"]

    detector = WasteDetector(backend=settings.backend, db_path=str(settings.db_path))
    if not detector.is_ready():
        failures.append(f"Detecteur non pret: {detector.last_error}")
        return failures

    image_candidates = [settings.base_dir / "test.jpg", settings.static_dir / "img" / "dechet1.jpeg"]
    image_path = next((path for path in image_candidates if path.exists()), None)
    if image_path is None:
        failures.append("Image de test manquante (test.jpg ou static/img/dechet1.jpeg)")
        return failures

    detections, details = detector.detect_from_image(str(image_path))
    if detections is None:
        failures.append(f"Inference echouee: {details}")
    return failures


def main() -> int:
    checks = [
        ("fichiers", check_files),
        ("sqlite", check_db),
        ("flask", check_flask_routes),
        ("detecteur", check_detector),
    ]
    all_failures: list[str] = []

    for name, fn in checks:
        failures = fn()
        if failures:
            print(f"[KO] {name}")
            for item in failures:
                print(f"  - {item}")
            all_failures.extend(failures)
        else:
            print(f"[OK] {name}")

    if all_failures:
        print(f"\nSmoke test en echec ({len(all_failures)} probleme(s)).")
        return 1

    print("\nSmoke test valide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
