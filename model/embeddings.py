"""Element (Z) embedding -> invariant scalar track `h` (CLAUDE.md §3, §4)."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class ElementEmbedding(nn.Module):
    n_elements: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """z[N] int atomic numbers -> h[N, hidden_dim]. Position is NOT a node feature (§3)."""
        return nn.Embed(self.n_elements, self.hidden_dim)(z)
