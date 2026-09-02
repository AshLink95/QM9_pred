"""Output heads (CLAUDE.md §5).

EnergyHead    -- invariant scalar, sum of per-atom readouts (size-extensive).
WannierHead   -- equivariant point set per spin channel: fixed max_centers slots, each with
                 a center (equivariant), radius (invariant, >0) and presence (invariant, 0..1).

Equivariance of the centers (§5: must come from the equivariant track, never a raw MLP):

    centroid   = mean_i x_i                                    # equivariant
    center_m   = centroid + sum_i A[i,m] * (x_i - centroid)    # A invariant weights

Under translation x->x+t: centroid->centroid+t, (x_i-centroid) unchanged -> center_m shifts
by t. Under rotation x->Rx: everything rotates by R. Correct by construction.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


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

        # invariant per-slot radius (>0) and presence (0..1), size-extensive sum over atoms
        rp = jnp.sum(nn.Sequential([
            nn.Dense(self.hidden_dim, name="rp0"), nn.silu,
            nn.Dense(S * M * 2, name="rp1"),
        ])(h), axis=0).reshape(S, M, 2)               # [S,M,2]
        radii = nn.softplus(rp[..., 0])               # [S,M] >0
        presence = nn.sigmoid(rp[..., 1])             # [S,M] in (0,1)

        return {"centers": centers, "radii": radii, "presence": presence}
