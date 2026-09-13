#!/bin/bash
# GPU training. Call from your sbatch (put #SBATCH specs / --gres=gpu:1 there).
# Prereq ON THE LOGIN NODE (has internet): `uv sync --group gpu`  (installs jax[cuda12],
# which bundles the CUDA libs — do NOT `module load cuda`, the wheels bring their own).
# Runs the venv's Python directly — no `uv` at runtime (compute nodes have no internet).
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # don't grab all GPU memory up front

# Fail fast if JAX can't see a GPU (else it would silently train on CPU for hours).
.venv/bin/python -c "import jax,sys; \
d=jax.devices(); print('devices:',d); \
sys.exit(0 if d[0].platform=='gpu' else 1)" || {
    echo "ERROR: no GPU visible to JAX." >&2
    echo "  - run 'uv sync --group gpu' on the login node" >&2
    echo "  - check 'nvidia-smi' works on this node (driver present)" >&2
    exit 1
}

.venv/bin/python -u -m scripts.train \
    --config configs/gpu.yaml \
    --dataset qm9.pkl \
    --out params.msgpack
