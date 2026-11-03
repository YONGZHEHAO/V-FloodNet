#!/usr/bin/env bash
# Codex Cloud install script. No activation or exported PATH is needed later.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This setup targets Linux x86_64 (Codex Cloud or WSL2).' >&2
  exit 1
fi
PROFILE="${VFLOOD_PROFILE:-cpu}"
if [[ "$PROFILE" != cpu && "$PROFILE" != image ]]; then
  echo 'VFLOOD_PROFILE must be cpu (image + stop sign) or image.' >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.codex/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.codex/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.codex/cache/pip}"
export TMPDIR="${TMPDIR:-$ROOT/.codex/tmp}"
export MPLBACKEND=Agg
export MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
UV="$ROOT/.codex/tools/bin/uv"
if [[ ! -x "$UV" ]]; then
  python3 -m pip install --disable-pip-version-check --target "$ROOT/.codex/tools" 'uv==0.9.7'
fi
"$UV" python install 3.10.16
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV" venv --python 3.10.16 "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Remove/recreate .venv with Python 3.10"'
"$UV" pip install --python "$ROOT/.venv/bin/python" -r .codex/requirements-cpu.lock

if [[ "$PROFILE" == cpu ]]; then
  for tool in git g++; do
    if ! command -v "$tool" >/dev/null; then
      echo "Missing $tool. Install git and build-essential in the cloud setup session." >&2
      exit 1
    fi
  done
  D2_SHA=247691a1c292654867deb35cabb27f18d75a1363
  D2_DIR="$ROOT/.codex/vendor/detectron2"
  if [[ ! -d "$D2_DIR/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/detectron2.git "$D2_DIR"
  fi
  git -C "$D2_DIR" checkout --detach "$D2_SHA"
  # Build against the installed CPU torch, rather than an isolated CUDA build.
  "$UV" pip install --python "$ROOT/.venv/bin/python" --no-deps --no-build-isolation -e "$D2_DIR"
fi

if [[ "${VFLOOD_DOWNLOAD_WEIGHTS:-1}" == 1 ]]; then
  "$ROOT/.venv/bin/python" scripts/download_cloud_weights.py --profile "$PROFILE"
  CHECK_ARGS=(--require-weights)
else
  CHECK_ARGS=()
fi
"$ROOT/.venv/bin/python" -m pip check
bash .codex/run.sh .codex/check_environment.py --profile "$PROFILE" "${CHECK_ARGS[@]}"
echo 'Ready. Run Python with: bash .codex/run.sh <script.py> [arguments]'
