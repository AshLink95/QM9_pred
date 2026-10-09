#!/bin/bash
# Sweep worker: submit this from MANY jobs (or a job array) at once. Each job claims runs from
# the queue defined in the sweep file, trains them one after another, and exits when nothing is
# left (docs/sweeps.md). Jobs never interfere: claims are atomic file renames, every run writes
# only to its own folder, and a run whose job dies is resumed by another job.
#
#   bash scripts/train-bunch.sh                      # uses configs/sweep.yaml
#   bash scripts/train-bunch.sh configs/my_sweep.yaml
#
# Prereq ON THE LOGIN NODE (has internet): uv sync --group gpu --group eval
# Runs the venv's Python directly — no `uv` at runtime (compute nodes have no internet).
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # don't grab all GPU memory up front
# shared on-disk cache of compiled programs: the slow first-epoch compile is paid once and
# reused by every later job (set JAX_COMPILATION_CACHE_DIR= to disable)
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR-$PWD/.jax_cache}"

# Fail fast if JAX can't see a GPU (set ALLOW_CPU=1 to run on CPU anyway).
if [ "${ALLOW_CPU:-0}" != "1" ]; then
    .venv/bin/python -c "import jax,sys; d=jax.devices(); print('devices:',d); \
sys.exit(0 if d[0].platform=='gpu' else 1)" || {
        echo "ERROR: no GPU visible to JAX (run 'uv sync --group gpu --group eval' on the login" \
             "node; check nvidia-smi; or set ALLOW_CPU=1)." >&2
        exit 1
    }
fi

# exec: Slurm's SIGTERM/SIGUSR1 reach Python directly -> it saves state and re-queues its run
exec .venv/bin/python -u -m scripts.sweep work "${1:-configs/sweep.yaml}"
