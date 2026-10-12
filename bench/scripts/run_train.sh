#!/usr/bin/env bash
# Local / interactive run helper.  Assumes the venv is already created & active,
# or pass the venv path as $1.  Runs the smoke test first, then training.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BENCH_DIR"

if [[ "${1:-}" != "" ]]; then
  # shellcheck disable=SC1091
  source "$1/bin/activate"
fi

CONFIG="${CONFIG:-configs/a0.yaml}"

echo "== smoke test =="
python scripts/smoke_test.py --config "$CONFIG" --device "${SMOKE_DEVICE:-cpu}"

echo "== training ($CONFIG) =="
python -m engine.train --config "$CONFIG"
