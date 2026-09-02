"""Combine energy + Wannier (+ optional dipole later) into one weighted loss (CLAUDE.md §6).

    loss = lambda_E * energy_loss + lambda_W * wannier_cloud_loss

Weights live in config (units differ); no magic numbers here.
"""

from __future__ import annotations

import jax.numpy as jnp

from losses.energy import energy_loss
from losses.wannier_cloud import wannier_loss


def example_to_jax(ex: dict) -> dict:
    """numpy example -> jax arrays (leaves the per-spin wannier list structure intact)."""
    return {
        "z": jnp.asarray(ex["z"]),
        "pos": jnp.asarray(ex["pos"]),
        "energy": jnp.asarray(ex["energy"]),
        "wannier": [{"centers": jnp.asarray(w["centers"]),
                     "radii": jnp.asarray(w["radii"])} for w in ex["wannier"]],
    }


def molecule_loss(params, apply_fn, ex, lambda_e, lambda_w, width_scale):
    """Scalar total loss for one molecule. `apply_fn(params, z, pos) -> out dict`."""
    out = apply_fn(params, ex["z"], ex["pos"])
    e = energy_loss(out["energy"], ex["energy"])
    w = wannier_loss(out["wannier"], ex["wannier"], width_scale) if lambda_w > 0 else 0.0
    total = lambda_e * e + lambda_w * w
    return total, {"energy": e, "wannier": w}
