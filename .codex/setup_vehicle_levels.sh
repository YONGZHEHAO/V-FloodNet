#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python || ! -x .codex/tools/bin/uv ]]; then
  echo 'Install the base CPU environment first: bash .codex/setup.sh' >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.codex/cache/uv}"
export TMPDIR="${TMPDIR:-$ROOT/.codex/tmp}"
mkdir -p "$TMPDIR" .codex/vendor/vehicle-levels
# Overlay only: preserve the existing torch, numpy and headless OpenCV.
.codex/tools/bin/uv pip install --python .venv/bin/python --target .codex/vendor/vehicle-levels --no-deps --require-hashes -r .codex/requirements-vehicle-levels.lock
bash .codex/run.sh scripts/vehicle_levels.py --check
if [[ "${VFLOOD_DOWNLOAD_WEIGHTS:-1}" == 1 ]]; then
  bash .codex/run.sh scripts/vehicle_levels.py --download
fi
bash .codex/run.sh -m pip check
