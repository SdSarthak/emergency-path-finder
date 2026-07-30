#!/usr/bin/env bash
# Quick start for Emergency Path Finder.
#
#   ./setup.sh            core install + tests
#   ./setup.sh --train    also install the training extras (torch, ultralytics)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WITH_TRAINING=0
for arg in "$@"; do
  case "$arg" in
    --train) WITH_TRAINING=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "Python 3.9+ is required but was not found." >&2
  exit 1
}

echo "==> Python: $($PYTHON --version)"

echo "==> Installing core dependencies"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt -r requirements-dev.txt

if [ "$WITH_TRAINING" -eq 1 ]; then
  echo "==> Installing training extras (this downloads several GB)"
  "$PYTHON" -m pip install -r training/requirements.txt
fi

echo "==> Creating working directories"
mkdir -p datasets ml_models flutter_app/assets/models

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Wrote .env from .env.example - add your ROBOFLOW_API_KEY to it"
fi

echo "==> Running the test suite"
"$PYTHON" -m pytest -q

if command -v flutter >/dev/null 2>&1; then
  echo "==> Flutter: $(flutter --version | head -1)"
else
  echo "==> Flutter not found - skip this unless you are building the app"
fi

cat <<'NEXT'

Setup complete.

Try it right now, no model or dataset needed:
  python -m emergency_path_finder --image <photo.jpg>
  python -m emergency_path_finder --camera

Then, to train your own detector:
  ./setup.sh --train
  python training/download_datasets.py        # needs ROBOFLOW_API_KEY
  python training/train_exit_detector.py

And for the mobile app:
  cd flutter_app
  flutter create --platforms=android,ios --project-name emergency_path_finder .
  flutter pub get && flutter run

Docs: README.md, QUICK_REFERENCE.md, docs/SETUP.md, docs/ARCHITECTURE.md
NEXT
