"""Assemble embedding + EGNN backbone + heads into one callable (CLAUDE.md §11).

Operates on a single molecule: z[N] (atomic numbers), pos[N,3] (Angstrom). The invariant
track h is seeded from the element embedding; the equivariant track x is seeded from
positions (§4). Batching is done by vmap in training.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .egnn import EGNNBackbone
from .embeddings import ElementEmbedding
from .heads import EnergyHead, WannierHead


class QM9Model(nn.Module):
    hidden_dim: int
    n_layers: int
    n_elements: int
    max_centers: int
    n_spins: int
    rbf_enabled: bool = True
    n_basis: int = 16
    cutoff: float = 10.0

    @nn.compact
    def __call__(self, z: jnp.ndarray, pos: jnp.ndarray) -> dict:
        h = ElementEmbedding(self.n_elements, self.hidden_dim)(z)
        x = pos
        h, x = EGNNBackbone(self.hidden_dim, self.n_layers, self.rbf_enabled,
                            self.n_basis, self.cutoff)(h, x)
        energy = EnergyHead(self.hidden_dim)(h)
        wannier = WannierHead(self.hidden_dim, self.max_centers, self.n_spins)(h, x)
        return {"energy": energy, "wannier": wannier}


def model_from_config(cfg: dict) -> QM9Model:
    """Build QM9Model from a parsed config dict (configs/*.yaml)."""
    m = cfg["model"]
    return QM9Model(
        hidden_dim=m["hidden_dim"],
        n_layers=m["n_layers"],
        n_elements=m["n_elements"],
        max_centers=cfg["wannier"]["max_centers"],
        n_spins=len(cfg["wannier"]["spins"]),
        rbf_enabled=m["rbf"]["enabled"],
        n_basis=m["rbf"]["n_basis"],
        cutoff=m["rbf"]["cutoff"],
    )
