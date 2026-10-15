"""Matched Wannier loss: exact on a reordered truth, ignores unused candidates, never mixes spins,
known offsets give known values, the selection target follows the pairing, more candidates than
centers work, and it differentiates under jit."""

import jax
import jax.numpy as jnp
import numpy as np

from losses.wannier_match import assign, wannier_match_loss

B, S, M = 2, 2, 6
K = np.array([[4, 4], [3, 3]], dtype=np.int32)          # real centers per (molecule, spin)
KW = dict(r_std=0.07, guide_sigma=0.3, radius_weight=1.0, guide_weight=0.1, select_weight=1.0)


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    c = np.zeros((B, S, M, 3)); r = np.zeros((B, S, M))
    for b in range(B):
        for s in range(S):
            k = K[b, s]
            c[b, s, :k] = rng.standard_normal((k, 3)) * 1.5
            r[b, s, :k] = rng.uniform(0.75, 0.95, k)
    return c, r


def _loss(pc, pr, tc, tr, logits=None):
    perm = assign(np.asarray(pc), np.asarray(tc), K)       # host pairing, as in training
    logits = np.zeros(pc.shape[:3]) if logits is None else logits
    return wannier_match_loss(jnp.asarray(pc), jnp.asarray(pr), jnp.asarray(logits),
                              jnp.asarray(tc), jnp.asarray(tr), jnp.asarray(K),
                              jnp.asarray(perm), **KW)


def _shuffled_prediction(tc, tr, seed=1):
    """Prediction = the truth reordered inside the first k candidates, junk beyond k."""
    rng = np.random.default_rng(seed)
    pc, pr = tc.copy(), tr.copy()
    for b in range(B):
        for s in range(S):
            k = K[b, s]
            p = rng.permutation(k)
            pc[b, s, :k], pr[b, s, :k] = tc[b, s, p], tr[b, s, p]
            pc[b, s, k:] = rng.standard_normal((M - k, 3)) * 10 + 30  # unused: far garbage
            pr[b, s, k:] = 5.0
    return pc, pr


def test_reordered_truth_gives_zero():
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    total, parts = _loss(pc, pr, tc, tr)
    assert float(parts["pos"]) < 1e-10 and float(parts["radius"]) < 1e-10
    assert abs(float(parts["guide"])) < 1e-5
    assert float(total) - KW["select_weight"] * float(parts["select"]) < 1e-5   # logits are 0


def test_known_offset():
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    _, parts = _loss(pc + np.array([0.1, 0, 0]), pr, tc, tr)
    assert abs(float(parts["pos"]) - 0.01) < 1e-6                   # (0.1 Å)^2


def test_spins_never_mix():
    tc, tr = _truth()
    swapped_c, swapped_r = tc[:, ::-1].copy(), tr[:, ::-1].copy()   # up <-> down predictions
    _, parts = _loss(swapped_c, swapped_r, tc, tr)
    assert float(parts["pos"]) > 0.1                       # an up center can't fill a down slot


def test_gradient_under_jit():
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    perm = jnp.asarray(assign(pc, tc, K))
    loss = lambda c, r, l: wannier_match_loss(c, r, l, jnp.asarray(tc), jnp.asarray(tr),
                                              jnp.asarray(K), perm, **KW)[0]
    f = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    gc, gr, gl = f(jnp.asarray(pc + 0.05), jnp.asarray(pr + 0.02), jnp.zeros((B, S, M)))
    assert all(np.all(np.isfinite(g)) for g in (gc, gr, gl))
    assert np.abs(np.asarray(gc)[:, :, :3]).sum() > 0        # matched candidates get gradient
    assert np.abs(np.asarray(gc)[1, :, 3:]).sum() == 0       # unmatched get no position pull
    assert np.abs(np.asarray(gl)).min() > 0                  # every logit gets a selection signal


def test_more_candidates_than_centers():
    """C > M: the k true centers pick their k candidates out of many (the head offers ~4/atom)."""
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    extra = np.random.default_rng(3).standard_normal((B, S, 10, 3)) * 5 + 40   # far decoys
    pc2 = np.concatenate([extra, pc], axis=2)                     # true matches now at 10..
    pr2 = np.concatenate([np.full((B, S, 10), 2.0), pr], axis=2)
    total, parts = _loss(pc2, pr2, tc, tr)
    assert float(parts["pos"]) < 1e-10 and float(parts["radius"]) < 1e-10
    perm = assign(pc2, tc, K)
    for b in range(B):
        for s in range(S):
            assert np.all(perm[b, s, :K[b, s]] >= 10)             # decoys never chosen


def test_selection_target_follows_pairing():
    """Logits high on matched candidates and low elsewhere -> small selection loss; reversed ->
    large."""
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    good = np.where(np.arange(M)[None, None, :] < K[:, :, None], 8.0, -8.0)
    _, p_good = _loss(pc, pr, tc, tr, good)
    _, p_bad = _loss(pc, pr, tc, tr, -good)
    assert float(p_good["select"]) < 1e-3 < 1.0 < float(p_bad["select"])


def test_select_centers_orders_by_logit():
    from model.heads import select_centers
    c = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    w = {"centers": c, "radii": c[..., 0], "logits": np.array([[0, 3, 1, 2, -1]] * 2, float)}
    sel = select_centers(w)
    np.testing.assert_array_equal(sel["centers"][0, :3], c[0, [1, 3, 2]])
    np.testing.assert_array_equal(sel["radii"][1], c[1, [1, 3, 2, 0, 4], 0])


def test_assign_with_thread_pool_matches_serial():
    from concurrent.futures import ThreadPoolExecutor
    tc, _ = _truth()
    pc, _ = _shuffled_prediction(tc, _truth()[1])
    with ThreadPoolExecutor(4) as pool:
        assert np.array_equal(assign(pc, tc, K, pool), assign(pc, tc, K))


def test_pairs_used_whole():
    """A ± pair whose two members sit on one spot must not stand in for a single: matching only
    one member would give their shared logit the targets 1 and 0 at once."""
    from losses.wannier_match import _assign_one, pair_siblings
    # one atom, layout (singles=2, pairs=1): candidates [s0, s1, +, -]
    pc = np.array([[0.0, 0, 0.3], [5, 5, 5], [0, 0, 0], [0, 0, 0]])   # pair collapsed on target
    tc = np.array([[0.0, 0, 0]])
    sib = pair_siblings(4, 2, 1)
    assert list(sib) == [0, 1, 3, 2]
    assert _assign_one(pc, tc, 1)[0] in (2, 3)                       # plain Hungarian: half pair
    assert _assign_one(pc, tc, 1, sib)[0] == 0                       # whole-pair rule: the single
    tc2 = np.array([[0.0, 0, 0.4], [0, 0, -0.4]])                    # an off-plane pair of targets
    pc2 = np.array([[3.0, 0, 0], [5, 5, 5], [0, 0, 0.4], [0, 0, -0.4]])
    assert sorted(_assign_one(pc2, tc2, 2, sib)) == [2, 3]           # both members used together


def test_position_loss_l1_known_value():
    tc, tr = _truth()
    pc, pr = _shuffled_prediction(tc, tr)
    perm = assign(pc, tc, K)
    _, parts = wannier_match_loss(jnp.asarray(pc + np.array([0.1, 0, 0])), jnp.asarray(pr),
                                  jnp.zeros((B, S, M)), jnp.asarray(tc), jnp.asarray(tr),
                                  jnp.asarray(K), jnp.asarray(perm), **KW, position_loss="l1")
    assert abs(float(parts["pos"]) - 0.1) < 1e-4                     # mean distance, Å


def test_selection_targets_nms():
    """Twins near a matched candidate are positives too (NMS keeps one at prediction); far
    candidates stay negatives; radius 0 reproduces one-to-one targets; a ± pair is positive only
    if both members are."""
    from losses.wannier_match import selection_targets
    # 1 molecule, 1 spin, 1 true center; candidates (one atom, layout 2 singles + 1 pair):
    # matched single 0, its twin 1 (0.05 Å away), pair 2/3 (+/- 0.6 Å, far)
    pc = jnp.asarray([[[[0.0, 0, 0], [0.05, 0, 0], [0, 0, 0.6], [0, 0, -0.6]]]])
    perm = jnp.zeros((1, 1, 1), jnp.int32)
    mask = jnp.ones((1, 1, 1))
    mc = pc[:, :, :1]
    t1 = selection_targets(pc, mc, mask, perm, 0.3, layout=(2, 1))
    assert np.asarray(t1).tolist() == [[[1, 1, 0, 0]]]
    t0 = selection_targets(pc, mc, mask, perm, 0.0, layout=(2, 1))
    assert np.asarray(t0).tolist() == [[[1, 0, 0, 0]]]
    # a pair with only its "+" member near the matched center is not a positive
    pc2 = pc.at[0, 0, 2].set(jnp.asarray([0.1, 0, 0]))
    t2 = selection_targets(pc2, mc, mask, perm, 0.3, layout=(2, 1))
    assert np.asarray(t2).tolist() == [[[1, 1, 0, 0]]]


def test_select_centers_nms():
    """Coincident twins (equal logits) count once; the next candidate fills the slot."""
    from model.heads import select_centers
    c = np.array([[[0.0, 0, 0], [0.02, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])     # [S=1, C=4, 3]
    w = {"centers": c, "radii": np.ones((1, 4)), "logits": np.array([[3.0, 3.0, 2.0, 1.0]])}
    plain = select_centers(w)["centers"][0, :2]
    assert np.abs(plain[0] - plain[1]).max() < 0.05                  # duplicate kept twice
    nms = select_centers(w, np.array([2]), 0.3)["centers"][0, :2]
    np.testing.assert_allclose(nms, c[0, [0, 2]])                    # one twin + the next one
