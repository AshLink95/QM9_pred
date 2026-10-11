"""The Wannier head can represent centers the old global-slot head could not.

The old head built every slot as centroid + Σ_i A[i,m](x_i − centroid). In methane the four H
are indistinguishable, so every slot sat exactly on the carbon; in a planar molecule every slot
stayed in the plane. Here one molecule is fitted alone (a representability check, not training):
the per-atom candidates must reach the four C–H centers of methane, and the ± pairs must reach
off-plane centers (above/below a C=C bond, an O lone pair) of a planar molecule. Predictions are
scored the way they are used: the k top-scoring candidates (select_centers), Hungarian-matched to
the truth.
"""

import jax

jax.config.update("jax_enable_x64", True)   # same precision whatever test file ran first

import jax.numpy as jnp
import numpy as np
import optax

from losses.wannier_match import assign, wannier_match_loss
from model.heads import select_centers
from model.model import QM9Model
from training.metrics import match_wannier

KW = dict(r_std=0.07, guide_sigma=0.3, radius_weight=1.0, guide_weight=0.1, select_weight=1.0)


def _fit(z, pos, targets, steps=600):
    """Fit the model to one molecule's centers (same set for both spins); return the mean
    distance (Å) of the selected centers to the truth."""
    k = len(targets)
    model = QM9Model(hidden_dim=32, n_layers=2, n_elements=10, n_spins=2, n_basis=8,
                     cutoff=6.0, radius_init=0.8)
    z, pos = jnp.asarray(z), jnp.asarray(pos, jnp.float32)
    params = model.init(jax.random.PRNGKey(0), z, pos)
    tc = np.broadcast_to(np.asarray(targets, np.float32), (1, 2, k, 3)).copy()
    tr = np.full((1, 2, k), 0.8, np.float32)
    kk = np.full((1, 2), k, np.int32)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2))   # as in training
    ost = opt.init(params)

    fwd = jax.jit(lambda p: model.apply(p, z, pos)["wannier"])

    @jax.jit
    def step(p, o, perm):
        def loss(p):
            w = model.apply(p, z, pos)["wannier"]
            return wannier_match_loss(w["centers"][None], w["radii"][None], w["logits"][None],
                                      tc, tr, kk, perm, **KW)[0]
        g = jax.grad(loss)(p)
        u, o = opt.update(g, o, p)
        return optax.apply_updates(p, u), o

    for _ in range(steps):
        perm = assign(np.asarray(fwd(params)["centers"])[None], tc, kk, layout=(2, 1))
        params, ost = step(params, ost, jnp.asarray(perm))
    sel = select_centers(jax.device_get(fwd(params)))
    m = match_wannier(sel["centers"][0], sel["radii"][0], k, tc[0, 0], tr[0, 0])
    return float(np.mean([d for _, _, d, _ in m["pairs"]]))


def test_methane_centers_on_each_bond():
    a = 0.629
    pos = np.array([[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]], float)
    targets = pos[1:] * 0.6                          # one center on each C–H bond
    assert _fit([6, 1, 1, 1, 1], pos, targets) < 0.05


def test_planar_off_plane_pairs():
    # planar vinyl-alcohol-like C2OH4 in the xy plane (only symmetry: the plane). k = 9: five
    # in-plane bond centers, a ± pair 0.4 Å above/below C=C, and an O lone pair ±0.35 Å
    pos = np.array([[0.0, 0, 0], [1.33, 0, 0], [2.05, 1.15, 0], [-0.55, 0.93, 0],
                    [-0.52, -0.95, 0], [1.86, -0.95, 0], [2.98, 1.0, 0]])
    inplane = [0.6 * pos[h] + 0.4 * pos[c] for c, h in ((0, 3), (0, 4), (1, 5), (2, 6))]
    inplane.append(0.5 * (pos[1] + pos[2]))
    mid, lp = 0.5 * (pos[0] + pos[1]), pos[2] + np.array([-0.1, -0.2, 0.0])
    up = np.array([0, 0, 1.0])
    targets = np.array(inplane + [mid + 0.4 * up, mid - 0.4 * up, lp + 0.35 * up, lp - 0.35 * up])
    assert _fit([6, 6, 8, 1, 1, 1, 1], pos, targets, steps=800) < 0.05
