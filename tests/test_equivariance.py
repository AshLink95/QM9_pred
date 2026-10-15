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


K, P = 2, 1                                  # singles / ± pairs per atom -> 4 candidates


def _model():
    # small + few layers: fast, still exercises the full equivariance machinery
    return QM9Model(hidden_dim=16, n_layers=3, n_elements=100, n_spins=2,
                    singles_per_atom=K, pairs_per_atom=P, rbf_enabled=True, n_basis=8,
                    cutoff=10.0)


def _per_atom(a, n):
    """[S, C, ...] candidates -> [S, N, K+2P, ...] (candidate c belongs to atom c // (K+2P))."""
    a = np.asarray(a)
    return a.reshape((a.shape[0], n, K + 2 * P) + a.shape[2:])


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(42)
    n = 6
    z = jnp.array(rng.integers(1, 10, size=n))
    pos = jnp.array(rng.standard_normal((n, 3)) * 1.5)
    model = _model()
    params = model.init(jax.random.PRNGKey(0), z, pos)
    # zero-initialized output layers (candidate offsets, pair spread) would make some checks
    # trivially pass; give every weight a random value
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    params = jax.tree_util.tree_unflatten(
        tree, [l + 0.3 * jax.random.normal(k, l.shape, l.dtype) for l, k in zip(leaves, keys)])
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
    # candidates belong to atoms: permuting the atoms permutes their candidates, nothing else
    n = len(z)
    for key in ("centers", "radii", "logits"):
        np.testing.assert_allclose(_per_atom(out["wannier"][key], n),
                                   _per_atom(base["wannier"][key], n)[:, perm], atol=1e-8)


def test_mirror(fixture):
    apply, params, z, pos = fixture
    base = apply(params, pos)
    out = apply(params, jnp.array(T.mirror(np.asarray(pos), axis=1)))
    Mx = jnp.eye(3).at[1, 1].set(-1.0)
    np.testing.assert_allclose(out["energy"], base["energy"], atol=1e-8)
    # no chiral features -> centers reflect exactly (§8). Singles map one to one; the two members
    # of a ± pair swap (their offset u × v is a pseudovector), so each pair reflects AS A SET.
    n = len(z)
    ref = _per_atom(np.asarray(base["wannier"]["centers"]) @ np.asarray(Mx).T, n)
    got = _per_atom(out["wannier"]["centers"], n)
    np.testing.assert_allclose(got[:, :, :K], ref[:, :, :K], atol=1e-8)
    np.testing.assert_allclose(got[:, :, K:K + P], ref[:, :, K + P:], atol=1e-8)
    np.testing.assert_allclose(got[:, :, K + P:], ref[:, :, K:K + P], atol=1e-8)
    assert np.abs(got[:, :, K:K + P] - got[:, :, K + P:]).max() > 1e-3     # pairs really split
    np.testing.assert_allclose(out["wannier"]["radii"], base["wannier"]["radii"], atol=1e-8)
    np.testing.assert_allclose(out["wannier"]["logits"], base["wannier"]["logits"], atol=1e-8)


def test_selected_set_same_in_every_frame(fixture):
    """What leaves the model (k top-scoring candidates) is the same set, however the molecule is
    presented: rotated, shifted far away, and listed in reverse order."""
    from model.heads import select_centers
    apply, params, z, pos = fixture
    model = _model()
    R = T.random_rotation(seed=3)
    t = np.array([100.0, -50.0, 25.0])
    rev = np.arange(len(z))[::-1]
    base = select_centers(apply(params, pos)["wannier"])
    out = select_centers(model.apply(params, jnp.array(np.asarray(z)[rev]),
                                     jnp.array(np.asarray(pos)[rev] @ R.T + t))["wannier"])
    k = 5
    for s in range(2):
        np.testing.assert_allclose(np.sort(out["centers"][s, :k], axis=0),
                                   np.sort(base["centers"][s, :k] @ R.T + t, axis=0), atol=1e-7)


def test_selected_set_with_nms_same_in_every_frame(fixture):
    """With NMS on, the selected set is still the same set in every frame and atom order."""
    from model.heads import select_centers
    apply, params, z, pos = fixture
    model = _model()
    R = T.random_rotation(seed=4)
    t = np.array([-30.0, 80.0, 5.0])
    rev = np.arange(len(z))[::-1]
    k = np.array([5, 5])                       # any count works; fixture atoms aren't real
    base = select_centers(apply(params, pos)["wannier"], k, 0.3)
    out = select_centers(model.apply(params, jnp.array(np.asarray(z)[rev]),
                                     jnp.array(np.asarray(pos)[rev] @ R.T + t))["wannier"], k, 0.3)
    for s in range(2):
        np.testing.assert_allclose(np.sort(out["centers"][s, :5], axis=0),
                                   np.sort(base["centers"][s, :5] @ R.T + t, axis=0), atol=1e-7)


@pytest.mark.parametrize("opts", [{"edge_gate": True}, {"update_coords": False}])
def test_egnn_options_stay_equivariant(opts):
    """EGNN's QM9 options (soft edge gate; fixed coordinates) keep every symmetry."""
    rng = np.random.default_rng(5)
    z = jnp.array(rng.integers(1, 10, size=6))
    pos = jnp.array(rng.standard_normal((6, 3)) * 1.5)
    model = QM9Model(hidden_dim=16, n_layers=3, n_elements=100, n_spins=2, n_basis=8,
                     cutoff=10.0, **opts)
    params = model.init(jax.random.PRNGKey(1), z, pos)
    base = model.apply(params, z, pos)
    R = np.asarray(T.random_rotation(seed=6))
    t = np.array([0.4, -1.0, 2.0])
    out = model.apply(params, z, jnp.array(np.asarray(pos) @ R.T + t))
    np.testing.assert_allclose(out["energy"], base["energy"], atol=1e-8)
    np.testing.assert_allclose(out["wannier"]["centers"],
                               np.asarray(base["wannier"]["centers"]) @ R.T + t, atol=1e-8)
    perm = T.permutation(6, seed=7)
    outp = model.apply(params, z[perm], pos[perm])
    np.testing.assert_allclose(outp["energy"], base["energy"], atol=1e-8)
