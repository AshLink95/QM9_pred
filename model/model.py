"""Assemble embedding + EGNN backbone + Wannier head into one callable (CLAUDE.md §11).

Operates on a single molecule: z[N] (atomic numbers), pos[N,3] (Angstrom). The invariant
track h is seeded from the element embedding; the equivariant track x is seeded from
positions (§4). Batching is done by vmap in training. The model predicts Wannier targets only
(CLAUDE.md §1); the output stays a dict keyed by target so another head can sit beside it.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .egnn import EGNNBackbone
from .embeddings import ElementEmbedding
from .heads import WannierHead


class QM9Model(nn.Module):
    hidden_dim: int
    n_layers: int
    n_elements: int
    n_spins: int
    singles_per_atom: int = 2       # Wannier candidates per atom and spin (model/heads.py)
    pairs_per_atom: int = 1         # ± candidate pairs per atom and spin
    update_coords: bool = True      # EGNN coordinate updates (False = EGNN's QM9 setting)
    edge_gate: bool = False         # EGNN soft edge gate ("attention")
    rbf_enabled: bool = True
    n_basis: int = 16
    cutoff: float = 10.0
    radius_init: float | None = None  # dataset mean Wannier radius (Å), init only

    @nn.compact
    def __call__(self, z: jnp.ndarray, pos: jnp.ndarray) -> dict:
        h = ElementEmbedding(self.n_elements, self.hidden_dim)(z)
        x = pos
        h, x = EGNNBackbone(self.hidden_dim, self.n_layers, self.rbf_enabled,
                            self.n_basis, self.cutoff, self.update_coords,
                            self.edge_gate)(h, x)
        wannier = WannierHead(self.hidden_dim, self.n_spins, self.singles_per_atom,
                              self.pairs_per_atom, self.rbf_enabled, self.n_basis, self.cutoff,
                              self.radius_init)(h, x)
        return {"wannier": wannier}


def model_from_config(cfg: dict, radius_init: float | None = None) -> QM9Model:
    """Build QM9Model from a parsed config dict (configs/*.yaml). `radius_init` (training only)
    sets the radius output's starting value from the training set; inference loads the trained
    value from the checkpoint."""
    m, w = cfg["model"], cfg["wannier"]
    return QM9Model(
        hidden_dim=m["hidden_dim"],
        n_layers=m["n_layers"],
        n_elements=m["n_elements"],
        n_spins=len(w["spins"]),
        singles_per_atom=w.get("singles_per_atom", 2),
        pairs_per_atom=w.get("pairs_per_atom", 1),
        update_coords=m.get("update_coords", True),
        edge_gate=m.get("edge_gate", False),
        rbf_enabled=m["rbf"]["enabled"],
        n_basis=m["rbf"]["n_basis"],
        cutoff=m["rbf"]["cutoff"],
        radius_init=radius_init,
    )
