"""EGNN layer + backbone (Satorras, Hoogeboom, Welling 2021) — CLAUDE.md §4.

Two tracks per node: invariant scalars `h` and equivariant coords `x`. Operates densely on
one molecule's N atoms (all ordered pairs i!=j); batching is done by vmap with masks in
training. Equivariance mechanism (do NOT alter, §4):

    m_ij = phi_e(h_i, h_j, radial(||x_i - x_j||))        # invariant message
    h_i  = h_i + phi_h(h_i, sum_j m_ij)                  # invariant scalar update
    x_i  = x_i + sum_j (x_i - x_j) * phi_x(m_ij)         # equivariant coord update

phi_x outputs an INVARIANT scalar weight, so the coord update rotates/translates correctly.
Vectors are only ever scaled by invariant weights, never mixed frame-dependently.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def gaussian_rbf(dist: jnp.ndarray, n_basis: int, cutoff: float) -> jnp.ndarray:
    """Smooth radial expansion of a distance (§4, optional but on by default).
    dist[...] -> [..., n_basis]. Purely a function of the invariant distance."""
    centers = jnp.linspace(0.0, cutoff, n_basis)
    width = cutoff / n_basis
    return jnp.exp(-((dist[..., None] - centers) ** 2) / (2.0 * width**2))


def _mlp(dims, name):
    layers = []
    for k, d in enumerate(dims):
        layers.append(nn.Dense(d, name=f"{name}_dense{k}"))
        if k < len(dims) - 1:
            layers.append(nn.silu)
    return nn.Sequential(layers)


class EGNNLayer(nn.Module):
    hidden_dim: int
    rbf_enabled: bool = True
    n_basis: int = 16
    cutoff: float = 10.0

    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        n = h.shape[0]
        H = self.hidden_dim

        diff = x[:, None, :] - x[None, :, :]            # [N,N,3] equivariant
        dist2 = jnp.sum(diff**2, axis=-1)               # [N,N] invariant
        dist = jnp.sqrt(dist2 + 1e-12)
        radial = gaussian_rbf(dist, self.n_basis, self.cutoff) if self.rbf_enabled \
            else dist2[..., None]

        # message: depends ONLY on invariants (h_i, h_j, radial)
        hi = jnp.broadcast_to(h[:, None, :], (n, n, H))
        hj = jnp.broadcast_to(h[None, :, :], (n, n, H))
        edge_in = jnp.concatenate([hi, hj, radial], axis=-1)
        m_ij = _mlp([H, H], "phi_e")(edge_in)           # [N,N,H]

        # zero out self-edges (i==j) so they contribute to neither aggregate
        off_diag = (1.0 - jnp.eye(n))[..., None]
        m_ij = m_ij * off_diag

        # invariant scalar update (residual)
        m_i = jnp.sum(m_ij, axis=1)                     # [N,H]
        h = h + _mlp([H, H], "phi_h")(jnp.concatenate([h, m_i], axis=-1))

        # equivariant coordinate update; phi_x -> invariant scalar weight per pair
        w = _mlp([H, 1], "phi_x")(m_ij)                 # [N,N,1] invariant
        x = x + jnp.sum(diff * w, axis=1) / (n - 1 + 1e-9)
        return h, x


class EGNNBackbone(nn.Module):
    hidden_dim: int
    n_layers: int
    rbf_enabled: bool = True
    n_basis: int = 16
    cutoff: float = 10.0

    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        for _ in range(self.n_layers):
            h, x = EGNNLayer(self.hidden_dim, self.rbf_enabled, self.n_basis,
                             self.cutoff)(h, x)
        return h, x
