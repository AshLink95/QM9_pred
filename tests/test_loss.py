"""Wannier cloud-loss properties (CLAUDE.md §6): zero on identical sets, order-invariant,
spin-split."""

import jax.numpy as jnp
import numpy as np

from losses.wannier_cloud import cloud_l2, wannier_loss


def _cloud(seed, k=4):
    rng = np.random.default_rng(seed)
    c = jnp.array(rng.standard_normal((k, 3)))
    w = jnp.ones(k)
    s = jnp.array(rng.uniform(0.3, 0.8, size=k) ** 2)
    return c, w, s


def test_zero_on_identical():
    c, w, s = _cloud(0)
    assert float(cloud_l2(c, w, s, c, w, s)) < 1e-5


def test_order_invariant():
    c, w, s = _cloud(1)
    base = cloud_l2(c, w, s, c, w, s)
    perm = np.random.default_rng(9).permutation(len(w))
    cp, wp, sp = c[perm], w[perm], s[perm]
    # true set reordered -> same loss (unordered set)
    assert abs(float(cloud_l2(c, w, s, cp, wp, sp)) - float(base)) < 1e-5


def test_positive_when_different():
    c, w, s = _cloud(2)
    c2, w2, s2 = _cloud(3)
    assert float(cloud_l2(c, w, s, c2, w2, s2)) > 1e-3


def test_spin_split():
    # pred/true each: spin0 cloud at A, spin1 cloud at B (far apart).
    A = jnp.array([[0.0, 0.0, 0.0]])
    B = jnp.array([[10.0, 0.0, 0.0]])
    r = jnp.array([0.5])
    m = jnp.array([1.0])
    pred = {"centers": jnp.stack([A, B]), "radii": jnp.stack([r, r]),
            "presence": jnp.stack([m, m])}
    true_aligned = [{"centers": A, "radii": r}, {"centers": B, "radii": r}]
    true_swapped = [{"centers": B, "radii": r}, {"centers": A, "radii": r}]
    aligned = float(wannier_loss(pred, true_aligned))
    swapped = float(wannier_loss(pred, true_swapped))
    assert aligned < 1e-6          # up<->up, down<->down match
    assert swapped > 1e-3          # swapping spins must NOT cancel (no cross-spin match)
