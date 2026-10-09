"""Output heads (CLAUDE.md §5).

EnergyHead    -- invariant scalar, sum of per-atom readouts (size-extensive).
WannierHead   -- equivariant point set per spin channel: fixed max_centers slots, each with
                 a center (equivariant) and a radius (invariant, >0). How many slots are real
                 is NOT predicted: it's Σ valence/2 per spin (data/electrons.py), and the first
                 k slots of each spin are the predicted centers.

Equivariance of the centers (§5: must come from the equivariant track, never a raw MLP):

    centroid   = mean_i x_i                                    # equivariant
    center_m   = centroid + sum_i A[i,m] * (x_i - centroid)    # A invariant weights

Under translation x->x+t: centroid->centroid+t, (x_i-centroid) unchanged -> center_m shifts
by t. Under rotation x->Rx: everything rotates by R. Correct by construction.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class EnergyHead(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        """h[N,H] -> scalar energy = sum_i MLP(h_i) (invariant, size-extensive)."""
        per_atom = nn.Sequential([
            nn.Dense(self.hidden_dim, name="e0"), nn.silu,
            nn.Dense(1, name="e1"),
        ])(h)                                       # [N,1]
        return jnp.sum(per_atom)                     # scalar


class WannierHead(nn.Module):
    hidden_dim: int
    max_centers: int
    n_spins: int                                     # len(config spins), e.g. up/down -> 2
    radius_init: float | None = None                 # data mean radius (Å), init only

    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """h[N,H], x[N,3] -> dict of [n_spins, max_centers, ...] targets."""
        M, S = self.max_centers, self.n_spins
        centroid = jnp.mean(x, axis=0)               # [3] equivariant

        # invariant per-atom weights that place each slot relative to the centroid
        A = nn.Sequential([
            nn.Dense(self.hidden_dim, name="w0"), nn.silu,
            nn.Dense(S * M, name="w1"),
        ])(h)                                         # [N, S*M]
        A = A.reshape(h.shape[0], S, M)               # [N,S,M]
        offsets = jnp.einsum("nsm,nd->smd", A, x - centroid)   # [S,M,3]
        centers = centroid[None, None, :] + offsets   # [S,M,3] equivariant

        # invariant per-slot radius (>0). MEAN over atoms (a center's size doesn't grow with the
        # molecule, unlike energy), plus a learned offset that starts at the dataset's mean
        # radius so training begins at the "constant guess" error instead of far from it.
        rp = jnp.mean(nn.Sequential([
            nn.Dense(self.hidden_dim, name="rp0"), nn.silu,
            nn.Dense(S * M, name="rp1"),
        ])(h), axis=0).reshape(S, M)                  # [S,M]
        r0 = 0.0 if self.radius_init is None else \
            float(np.log(np.expm1(self.radius_init)))      # softplus^-1(mean radius)
        offset = self.param("radius_offset", lambda key: jnp.asarray(r0, jnp.float32))
        radii = nn.softplus(rp + offset)              # [S,M] >0

        return {"centers": centers, "radii": radii}
