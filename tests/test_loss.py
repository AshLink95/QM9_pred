"""Gaussian-cloud overlap (the fixed-width guide term): zero on identical sets, order-free.
Spin-split behaviour of the full Wannier loss is tested in test_wannier_match.py."""

import jax.numpy as jnp
import numpy as np

from losses.wannier_cloud import cloud_l2


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
