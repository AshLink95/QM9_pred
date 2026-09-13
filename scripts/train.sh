#!/bin/bash
# Run training. Call this from your sbatch (put #SBATCH specs + module loads there).
# Runs the venv's Python directly — NO `uv` at runtime (compute nodes have no internet, so
# `uv run` would hang syncing the venv). Run `uv sync` on the login node first.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # don't grab all GPU memory up front

.venv/bin/python -u -m scripts.train \
    --config configs/default.yaml \
    --dataset qm9.pkl \
    --out params.msgpack
