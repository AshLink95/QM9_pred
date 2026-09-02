"""Equivariance gate (CLAUDE.md §8) — MUST pass on the UNTRAINED net before any training.

Random weights still satisfy equivariance if the architecture is right (§10 step 4). If any
of these fail, the model is wrong regardless of its loss value.
"""

import jax

jax.config.update("jax_enable_x64", True)   # tight tolerances for the symmetry checks

import jax.numpy as jnp
import numpy as np
import pytest

from model.model import QM9Model
from symmetry import transforms as T


def _model():
    # small + few layers: fast, still exercises the full equivariance machinery
    return QM9Model(hidden_dim=16, n_layers=3, n_elements=100,
                    max_centers=8, n_spins=2, rbf_enabled=True, n_basis=8, cutoff=10.0)


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(42)
    n = 6
    z = jnp.array(rng.integers(1, 10, size=n))
    pos = jnp.array(rng.standard_normal((n, 3)) * 1.5)
    model = _model()
    params = model.init(jax.random.PRNGKey(0), z, pos)
    apply = jax.jit(lambda p, pos_: model.apply(p, z, pos_))
    return apply, params, z, pos


def test_translation(fixture):
    apply, params, z, pos = fixture
    t = jnp.array([1.3, -2.1, 0.7])
    base = apply(params, pos)
    out = apply(params, jnp.array(T.translate(np.asarray(pos), np.asarray(t))))
    np.testing.assert_allclose(out["energy"], base["energy"], atol=1e-8)
    np.testing.assert_allclose(out["wannier"]["centers"],
                               base["wannier"]["centers"] + t, atol=1e-8)


def test_rotation(fixture):
    apply, params, z, pos = fixture
    R = T.random_rotation(seed=1)
    base = apply(params, pos)
    out = apply(params, jnp.array(T.rotate(np.asarray(pos), R)))
    np.testing.assert_allclose(out["energy"], base["energy"], atol=1e-8)
    rotated = base["wannier"]["centers"] @ jnp.array(R).T
    np.testing.assert_allclose(out["wannier"]["centers"], rotated, atol=1e-8)
    # invariant per-slot quantities unchanged
    np.testing.assert_allclose(out["wannier"]["radii"], base["wannier"]["radii"], atol=1e-8)


def test_permutation(fixture):
    apply, params, z, pos = fixture
    model = _model()
    perm = T.permutation(len(z), seed=2)
    base = apply(params, pos)
    out = model.apply(params, jnp.array(np.asarray(z)[perm]),
                      jnp.array(np.asarray(pos)[perm]))
    np.testing.assert_allclose(out["energy"], base["energy"], atol=1e-8)
    # unordered center set: centroid+symmetric-sum construction is permutation-invariant
    np.testing.assert_allclose(np.sort(np.asarray(out["wannier"]["centers"]), axis=None),
                               np.sort(np.asarray(base["wannier"]["centers"]), axis=None),
                               atol=1e-8)


def test_mirror(fixture):
    apply, params, z, pos = fixture
    base = apply(params, pos)
    out = apply(params, jnp.array(T.mirror(np.asarray(pos), axis=1)))
    Mx = jnp.eye(3).at[1, 1].set(-1.0)
    np.testing.assert_allclose(out["energy"], base["energy"], atol=1e-8)
    # architecture has no chiral features -> centers reflect exactly (§8)
    np.testing.assert_allclose(out["wannier"]["centers"],
                               base["wannier"]["centers"] @ Mx.T, atol=1e-8)
