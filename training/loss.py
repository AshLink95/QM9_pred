"""Example -> JAX-array conversion for eval helpers (CLAUDE.md §6).

Training uses the padded-batch path in `training/train.py`; this converter is for the
per-molecule eval utilities (`training/metrics.py`, `scripts/evaluate.py`).
"""

from __future__ import annotations

import jax.numpy as jnp


def example_to_jax(ex: dict) -> dict:
    """numpy example -> jax arrays (leaves the per-spin wannier list structure intact)."""
    return {
        "z": jnp.asarray(ex["z"]),
        "pos": jnp.asarray(ex["pos"]),
        "energy": jnp.asarray(ex["energy"]),
        "wannier": [{"centers": jnp.asarray(w["centers"]),
                     "radii": jnp.asarray(w["radii"])} for w in ex["wannier"]],
    }
