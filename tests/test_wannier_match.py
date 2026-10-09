"""Matched Wannier loss: exact on a reordered truth, ignores unused slots, never mixes spins,
known offsets give known values, and it differentiates under jit."""

import jax
import jax.numpy as jnp
import numpy as np

from losses.wannier_match import assign, wannier_match_loss

B, S, M = 2, 2, 6
K = np.array([[4, 4], [3, 3]], dtype=np.int32)          # real centers per (molecule, spin)
KW = dict(r_std=0.07, guide_sigma=0.3, radius_weight=1.0, guide_weight=0.1)


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    c = np.zeros((B, S, M, 3)); r = np.zeros((B, S, M))
    for b in range(B):
        for s in range(S):
            k = K[b, s]
            c[b, s, :k] = rng.standard_normal((k, 3)) * 1.5
            r[b, s, :k] = rng.uniform(0.75, 0.95, k)
    return c, r


def _loss(pc, pr, tc, tr):
    perm = assign(np.asarray(pc), np.asarray(tc), K)       # host pairing, as in training
    return wannier_match_loss(jnp.asarray(pc), jnp.asarray(pr), jnp.asarray(tc),
                              jnp.asarray(tr), jnp.asarray(K), jnp.asarray(perm), **KW)


def _shuffled_prediction(tc, tr, seed=1):
    """Prediction = the truth reordered inside the first k slots, junk in slots >= k."""
    rng = np.random.default_rng(seed)
    pc, pr = tc.copy(), tr.copy()
    for b in range(B):
        for s in range(S):
            k = K[b, s]
            p = rng.permutation(k)
            pc[b, s, :k], pr[b, s, :k] = tc[b, s, p], tr[b, s, p]
            pc[b, s, k:] = rng.standard_normal((M - k, 3)) * 10      # unused slots: garbage
            pr[b, s, k:] = 5.0
    return pc, pr


def test_reordered_truth_gives_zero():
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    total, parts = _loss(pc, pr, tc, tr)
    assert float(parts["pos"]) < 1e-10 and float(parts["radius"]) < 1e-10
    assert abs(float(parts["guide"])) < 1e-5 and float(total) < 1e-5


def test_known_offset():
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    _, parts = _loss(pc + np.array([0.1, 0, 0]), pr, tc, tr)
    assert abs(float(parts["pos"]) - 0.01) < 1e-6                   # (0.1 Å)^2


def test_spins_never_mix():
    tc, tr = _truth()
    swapped_c, swapped_r = tc[:, ::-1].copy(), tr[:, ::-1].copy()   # up <-> down predictions
    _, parts = _loss(swapped_c, swapped_r, tc, tr)
    assert float(parts["pos"]) > 0.1                                # an up center can't fill a down slot


def test_gradient_under_jit():
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    perm = jnp.asarray(assign(pc, tc, K))
    loss = lambda c, r: wannier_match_loss(c, r, jnp.asarray(tc), jnp.asarray(tr),
                                           jnp.asarray(K), perm, **KW)[0]
    f = jax.jit(jax.grad(loss, argnums=(0, 1)))
    gc, gr = f(jnp.asarray(pc + 0.05), jnp.asarray(pr + 0.02))
    assert np.all(np.isfinite(gc)) and np.all(np.isfinite(gr))
    assert np.abs(np.asarray(gc)[:, :, :3]).sum() > 0                # real slots get gradient
    assert np.abs(np.asarray(gc)[1, :, 3:]).sum() == 0               # slots >= k get none


def test_assign_with_thread_pool_matches_serial():
    from concurrent.futures import ThreadPoolExecutor
    tc, _ = _truth()
    pc, _ = _shuffled_prediction(tc, _truth()[1])
    with ThreadPoolExecutor(4) as pool:
        assert np.array_equal(assign(pc, tc, K, pool), assign(pc, tc, K))
