#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo 'Environment missing. Run bash .codex/setup.sh first.' >&2
  exit 1
fi
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.codex/cache/torch}"
exec "$ROOT/.venv/bin/python" "$@"
