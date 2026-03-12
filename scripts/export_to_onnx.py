from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def pick_pt_model(explicit: Path | None) -> Path:
    if explicit:
        return explicit if explicit.is_absolute() else (settings.base_dir / explicit)
    for candidate in settings.pt_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Aucun modele .pt trouve. Placez my_model.pt ou yolov8n.pt dans models/."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exporter un modele YOLO .pt vers ONNX.")
    parser.add_argument("--pt", type=Path, help="Modele source .pt")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=settings.onnx_model,
        help="Chemin sortie ONNX (defaut: models/my_model.onnx)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Taille image export.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[ERREUR] ultralytics/torch manquant: {exc}")
        print("Installez requirements/requirements-dev.txt puis relancez.")
        return 1

    pt_path = pick_pt_model(args.pt)
    if not pt_path.exists():
        print(f"[ERREUR] Modele PT introuvable: {pt_path}")
        return 2

    onnx_path = args.onnx if args.onnx.is_absolute() else (settings.base_dir / args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Export PT -> ONNX")
    print(f"  source: {pt_path}")
    print(f"  output: {onnx_path}")
    print(f"  imgsz : {args.imgsz}")

    model = YOLO(str(pt_path))
    generated = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        simplify=True,
        opset=12,
    )
    generated_path = Path(generated)

    if generated_path.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(generated_path.read_bytes())
    print(f"[OK] Modele ONNX genere: {onnx_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"[ERREUR] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"[ERREUR] Export ONNX echoue: {exc}")
        raise SystemExit(1)
