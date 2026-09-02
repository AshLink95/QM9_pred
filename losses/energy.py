"""Energy regression loss (CLAUDE.md §5) — plain squared error on the invariant scalar."""

from __future__ import annotations

import jax.numpy as jnp


def energy_loss(pred_energy: jnp.ndarray, true_energy: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred_energy - true_energy) ** 2)
