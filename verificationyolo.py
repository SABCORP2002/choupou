from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import settings


def find_default_image() -> Path:
    candidates = [
        settings.base_dir / "test.jpg",
        settings.static_dir / "img" / "dechet1.jpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Aucune image de test trouvee. Ajoutez test.jpg a la racine du projet."
    )


def main() -> int:
    try:
        from yolo_detector import WasteDetector
    except Exception as exc:
        print(f"[ERREUR] Impossible d'importer le detecteur: {exc}")
        return 2

    parser = argparse.ArgumentParser(description="Verification rapide de la detection.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Chemin image a tester (defaut: test.jpg ou static/img/dechet1.jpeg).",
    )
    parser.add_argument(
        "--backend",
        default=settings.backend,
        choices=["auto", "onnx", "pt"],
        help="Backend d'inference.",
    )
    args = parser.parse_args()

    image_path = args.image or find_default_image()
    if not image_path.is_absolute():
        image_path = (settings.base_dir / image_path).resolve()

    detector = WasteDetector(backend=args.backend, db_path=str(settings.db_path))
    if not detector.is_ready():
        print(f"[ERREUR] Detecteur indisponible: {detector.last_error}")
        return 3

    detections, details = detector.detect_from_image(str(image_path))
    if detections is None:
        print(f"[ERREUR] Inference impossible: {details}")
        return 4

    print(f"[OK] Backend: {detector.backend_name}")
    print(f"[OK] Image: {image_path}")
    print(f"[OK] Nombre detections: {len(detections)}")
    for idx, det in enumerate(detections, start=1):
        print(
            f"  {idx}. {det['waste_type']} | conf={det['confidence']:.3f} | box={det['box'].tolist()}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"[ERREUR] Verification echouee: {exc}")
        raise SystemExit(1)
