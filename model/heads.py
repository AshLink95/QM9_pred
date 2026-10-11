"""Output heads (CLAUDE.md §5).

EnergyHead    -- invariant scalar, sum of per-atom readouts (size-extensive).
WannierHead   -- equivariant point SET per spin channel, built from per-atom candidates. Each
                 candidate has a center (equivariant), a radius (invariant, >0) and a selection
                 logit (invariant). How many centers are real is NOT predicted: it's Σ valence/2
                 per spin (data/electrons.py); `select_centers` orders the candidates by logit
                 and the first k are the predicted centers.

Why candidates belong to atoms (not to global slots m = 1..M):
A global slot is one output that every atom contributes to by the same rule. If a molecule maps
onto itself under a rotation/mirror g (methane under a 120° turn, a planar molecule under its
own mirror), an equivariant model must give f(X) = g·f(X) slot by slot, so every slot has to lie
on every symmetry element: in methane all slots sat exactly on the carbon (any weights), in
planar molecules all slots stayed in the plane. Physics only requires the SET of centers to map
onto itself; centers may swap. Candidates indexed by atoms swap exactly when the atoms swap, so
the output is a set that transforms with the molecule, deterministically.

Construction for atom i and spin s (r_ij = (x_j - x_i)/(d_ij + 1), bounded as in the EGNN
coordinate update; all weights are invariant functions of h_i, h_j, d_ij):

    single  c = x_i + Σ_j α_ij r_ij                         # equivariant point
    pair    c± = b_i ± s_i (u_i × v_i)                      # b_i like a single;
            u_i = Σ_j a_ij r_ij,  v_i = Σ_j β_ij r_ij       # two different weight sets

The pair handles centers that sit off a mirror plane: for an atom in the plane, u and v lie in
it, so u × v is normal to it. u × v is a pseudovector (it flips sign under a reflection), so a
reflection swaps c+ and c-: the PAIR transforms exactly as a set, and no chirality is introduced.
Both members share one radius and one logit (a symmetry may swap them), so training uses a pair
whole or not at all (losses/wannier_match.py).

Known ceiling: an atom lying on TWO mirror planes (e.g. the carbons of ethylene or benzene) can't
build any nonzero pseudovector — the perpendicular mirror forces u × v = 0 there — so its pairs
stay collapsed; neighboring atoms with a single mirror (the H's) can still host the off-plane
pair. Lifting this needs a sign-free axis (a rank-2 tensor), which CLAUDE.md §7 defers. Most QM9
molecules have at most the molecular plane as a mirror.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from .egnn import _mlp, gaussian_rbf


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
    n_spins: int                                     # len(config spins), e.g. up/down -> 2
    singles: int = 2                                 # single candidates per atom and spin
    pairs: int = 1                                   # ± pairs per atom and spin
    rbf_enabled: bool = True
    n_basis: int = 16
    cutoff: float = 10.0
    radius_init: float | None = None                 # data mean radius (Å), init only

    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """h[N,H], x[N,3] -> {centers[S,C,3], radii[S,C], logits[S,C]}, C = N·(singles+2·pairs).
        Candidate c = i·(singles+2·pairs) + q belongs to atom i (q: singles, then +, then -)."""
        n, H = h.shape
        S, K, P = self.n_spins, self.singles, self.pairs

        diff = x[None, :, :] - x[:, None, :]               # [N,N,3] x_j - x_i (equivariant)
        dist2 = jnp.sum(diff ** 2, axis=-1)
        dist = jnp.sqrt(dist2 + 1e-12)
        off = (1.0 - jnp.eye(n))[..., None]                # drop self-edges
        r = diff / (dist[..., None] + 1.0) * off           # [N,N,3] bounded directions
        radial = gaussian_rbf(dist, self.n_basis, self.cutoff) if self.rbf_enabled \
            else dist2[..., None]
        edge = _mlp([H, H], "edge")(jnp.concatenate(
            [jnp.broadcast_to(h[:, None, :], (n, n, H)),
             jnp.broadcast_to(h[None, :, :], (n, n, H)), radial], axis=-1))   # [N,N,H] invariant

        # invariant edge weights -> equivariant vectors. Point offsets start at zero (every
        # candidate begins on its own atom). The u/v frame weights and the pair spread start
        # random: u × v has zero gradient at u = v = 0, and a pair that starts collapsed is never
        # nearer to an off-plane target than an in-plane single, so it would never get matched.
        a_pt = nn.Dense(S * (K + P), kernel_init=nn.initializers.zeros, name="point_w")(edge)
        a_fr = nn.Dense(S * 2 * P, name="frame_w")(edge)
        vec = lambda a, q: jnp.einsum("ijsq,ijd->isqd", a.reshape(n, n, S, q), r)
        pts = x[:, None, None, :] + vec(a_pt, K + P)       # [N,S,K+P,3]
        uv = vec(a_fr, 2 * P)                              # [N,S,2P,3]
        normal = jnp.cross(uv[:, :, :P], uv[:, :, P:])     # [N,S,P,3] pseudovector

        node = _mlp([H, H], "node")(jnp.concatenate([h, jnp.sum(edge * off, axis=1)], -1))
        scale = nn.Dense(S * P, name="pair_scale")(node)
        shift = scale.reshape(n, S, P)[..., None] * normal
        base = pts[:, :, K:]
        centers = jnp.concatenate([pts[:, :, :K], base + shift, base - shift], axis=2)

        # invariant per-candidate radius (>0) and selection logit; a pair shares both. The
        # radius offset starts at the dataset's mean radius ("constant guess" error at step 0).
        rl = nn.Dense(S * 2 * (K + P), name="radius_logit")(node).reshape(n, S, 2, K + P)
        dup = lambda a: jnp.concatenate([a[..., :K], a[..., K:], a[..., K:]], axis=-1)
        r0 = 0.0 if self.radius_init is None else \
            float(np.log(np.expm1(self.radius_init)))      # softplus^-1(mean radius)
        offset = self.param("radius_offset", lambda key: jnp.asarray(r0, jnp.float32))
        radii = nn.softplus(dup(rl[:, :, 0]) + offset)     # [N,S,K+2P]
        logits = dup(rl[:, :, 1])

        C = n * (K + 2 * P)
        flat = lambda a: jnp.moveaxis(a, 1, 0).reshape((S, C) + a.shape[3:])
        return {"centers": flat(centers), "radii": flat(radii), "logits": flat(logits)}


def select_centers(wannier):
    """Order each spin's candidates by selection logit, highest first (host-side, NumPy).

    wannier: {centers[...,S,C,3], radii[...,S,C], logits[...,S,C]} (one molecule or a batch).
    Returns {centers, radii} in that order: the first k = Σ valence/2 of a spin are its predicted
    centers. Every consumer (validation, report, main.py) goes through this, so predictions are
    always the selected candidates, never an oracle match over all of them.
    # ponytail: candidates tied at the k-th place (equal logits, e.g. the two members of a pair)
    # are cut by index order; a trained model rarely leaves a tie exactly at the boundary.
    """
    order = np.argsort(-np.asarray(wannier["logits"]), axis=-1, kind="stable")
    return {"centers": np.take_along_axis(np.asarray(wannier["centers"]), order[..., None], -2),
            "radii": np.take_along_axis(np.asarray(wannier["radii"]), order, -1)}
