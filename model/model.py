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
    atom_ref: tuple | None = None   # per-element reference energies (eV), init only; see below

    @nn.compact
    def __call__(self, z: jnp.ndarray, pos: jnp.ndarray) -> dict:
        h = ElementEmbedding(self.n_elements, self.hidden_dim)(z)
        x = pos
        h, x = EGNNBackbone(self.hidden_dim, self.n_layers, self.rbf_enabled,
                            self.n_basis, self.cutoff)(h, x)
        # E = sum_i ref[Z_i] + sum_i MLP(h_i). The per-element offset absorbs the huge
        # (~1e3 eV) scale of SIESTA total energies so the network only learns the eV-scale
        # residual — raw targets made the loss ~1e12 and training diverged. Still a sum over
        # atoms: invariant, permutation-invariant, size-extensive (CLAUDE.md §5). It is a
        # normal (trainable) param, so it is saved in checkpoints; `atom_ref` only sets its
        # initial value at training start (inference loads the trained value).
        if self.atom_ref is None:
            ref_init = nn.initializers.zeros
        else:
            ref_init = lambda key, shape, dtype=jnp.float32: \
                jnp.asarray(self.atom_ref, dtype).reshape(shape)
        e_ref = jnp.sum(nn.Embed(self.n_elements, 1, embedding_init=ref_init,
                                 name="atom_ref")(z))
        energy = EnergyHead(self.hidden_dim)(h) + e_ref
        wannier = WannierHead(self.hidden_dim, self.max_centers, self.n_spins)(h, x)
        return {"energy": energy, "wannier": wannier}


def model_from_config(cfg: dict, atom_ref: tuple | None = None) -> QM9Model:
    """Build QM9Model from a parsed config dict (configs/*.yaml). `atom_ref` (training only)
    initializes the per-element energy offsets from a fit to the training set."""
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
        atom_ref=atom_ref,
    )
