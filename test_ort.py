from __future__ import annotations

import platform
import sys


def main() -> int:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        print(f"[ERREUR] onnxruntime indisponible: {exc}")
        return 1

    providers = ort.get_available_providers()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Arch: {platform.machine()}")
    print(f"onnxruntime: {ort.__version__}")
    print(f"Providers: {providers}")

    if "CPUExecutionProvider" not in providers:
        print("[ERREUR] CPUExecutionProvider absent. Inference CPU non disponible.")
        return 2

    print("[OK] ONNX Runtime CPU disponible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
