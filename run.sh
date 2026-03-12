#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "[ERREUR] .venv absent. Lancez ./bootstrap.sh d'abord."
  exit 1
fi

source .venv/bin/activate

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
fi

python app.py
