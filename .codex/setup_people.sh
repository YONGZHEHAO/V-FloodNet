#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python || ! -x .codex/tools/bin/uv ]]; then
  echo 'Install the base environment first: bash .codex/setup.sh' >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.codex/cache/uv}"
export TMPDIR="${TMPDIR:-$ROOT/.codex/tmp}"
mkdir -p "$TMPDIR"
VENDOR="$ROOT/.codex/vendor/MeshTransformer"
if [[ ! -d "$VENDOR/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/microsoft/MeshTransformer.git "$VENDOR"
fi
git -C "$VENDOR" checkout --detach 412b30142f39e4ac4480a54943993e232697c909
git -C "$VENDOR" submodule update --init --depth 1 --filter=blob:none transformers
.codex/tools/bin/uv pip install --python .venv/bin/python --no-build-isolation -r .codex/requirements-people.lock
bash .codex/run.sh .codex/patch_metro.py
if [[ "${VFLOOD_DOWNLOAD_WEIGHTS:-1}" == 1 ]]; then
  bash .codex/run.sh scripts/download_people_weights.py
fi
bash .codex/run.sh -m pip check
bash .codex/run.sh scripts/people_depth.py --check
