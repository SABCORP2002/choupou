from __future__ import annotations

import argparse
from pathlib import Path

from config import settings


def default_image() -> Path:
    img = settings.base_dir / "test.jpg"
    if img.exists():
        return img
    raise FileNotFoundError("Image de test introuvable: ajoutez test.jpg a la racine.")


def main() -> int:
    try:
        from detector.detector_backend import choose_backend
    except Exception as exc:
        print(f"[ERREUR] Import backend ONNX impossible: {exc}")
        return 1

    parser = argparse.ArgumentParser(description="Test d'inference ONNX CPU.")
    parser.add_argument("--image", type=Path, help="Image a tester.")
    parser.add_argument(
        "--model",
        type=Path,
        default=settings.onnx_model,
        help="Chemin du modele ONNX (defaut: models/my_model.onnx).",
    )
    args = parser.parse_args()

    model_path = args.model if args.model.is_absolute() else (settings.base_dir / args.model)
    image_path = args.image or default_image()
    if not image_path.is_absolute():
        image_path = (settings.base_dir / image_path).resolve()

    backend, error = choose_backend(
        requested_backend="onnx",
        confidence=settings.confidence_threshold,
        class_map=settings.waste_classes,
        onnx_model=model_path,
        pt_candidates=settings.pt_candidates,
    )
    if backend is None:
        print(f"[ERREUR] Backend ONNX indisponible: {error}")
        return 2

    try:
        import cv2
    except ImportError as exc:
        print(f"[ERREUR] OpenCV manquant: {exc}")
        return 4

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERREUR] Impossible de lire l'image: {image_path}")
        return 3

    detections = backend.detect(frame)
    print(f"[OK] Modele ONNX: {model_path}")
    print(f"[OK] Image: {image_path}")
    print(f"[OK] Detections: {len(detections)}")
    for idx, det in enumerate(detections, start=1):
        print(f"  {idx}. {det.label} | conf={det.confidence:.3f} | box={det.box_xyxy}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"[ERREUR] Test ONNX echoue: {exc}")
        raise SystemExit(1)
