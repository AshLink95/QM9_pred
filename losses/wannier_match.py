"""Wannier loss: Hungarian-matched candidates -> positions + radii + selection, plus a fixed-width
cloud guide (per spin).

Replaces the presence-weighted Gaussian cloud, which let the model blur several centers into a
few wide, faint blobs (test report: count 0.1%, center 0.61 Å, radius 0.22 Å). Here:

  * the COUNT is not predicted: spin s of a molecule has k = Σ valence/2 real centers
    (data/electrons.py);
  * the head proposes C candidates per spin (a few per atom, model/heads.py). The k true centers
    of ONE spin are paired with k of the C candidates by the Hungarian algorithm (min total
    squared distance; rectangular, k <= C), and each pair's squared distance is minimized
    (DETR-style set loss; the pairing itself is not differentiated);
  * RADII: each matched candidate regresses its own true radius (normalized by the dataset std);
  * SELECTION: binary cross-entropy on each candidate's logit, target 1 if it was matched or lies
    within `nms_radius` of a matched one (`selection_targets`), else 0. At prediction time the
    k highest logits, with near-duplicates suppressed, are the centers
    (model/heads.py:select_centers). Pure classification: no amplitude or width enters any
    overlap, so it can't be used to blur;
  * GUIDE: the closed-form cloud overlap (losses/wannier_cloud.py) between the matched
    candidates and the truth, FIXED width, amplitude 1 per center — a smooth, order-free pull
    that steadies early training when pairings still flip.

The pairing is computed on the HOST (`assign`, NumPy + SciPy, multi-threaded) and passed into
the jitted loss as `perm`. Keeping it out of compiled code lets training overlap CPU matching with
GPU work and keeps the compiled programs cacheable (training/train.py).

Up only ever pairs with up and down with down: every term is computed per spin channel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import linear_sum_assignment

from losses.wannier_cloud import cloud_l2


def pair_siblings(n_cand, singles, pairs):
    """sib[c] = the other member of candidate c's ± pair (c itself for singles), for the head's
    layout: per atom `singles` singles, then `pairs` "+" members, then their "-" members."""
    per = singles + 2 * pairs
    q = np.arange(n_cand) % per
    sib = np.arange(n_cand)
    plus, minus = (q >= singles) & (q < singles + pairs), q >= singles + pairs
    sib[plus] += pairs
    sib[minus] -= pairs
    return sib


def _assign_one(pc, tc, k, sib=None):
    """One (molecule, spin): the k true centers paired with k of the C candidates -> perm[M]
    (candidate index per true center; 0 beyond k, masked out by the loss).

    A ± pair is used whole or not at all: its members share one selection logit (a mirror may
    swap them), so a pair with only one member matched would get the targets 1 and 0 at once.
    When that happens the pair is excluded and the assignment re-solved (a few rounds at most;
    if too few candidates would remain, the last solution is kept)."""
    perm = np.zeros(tc.shape[0], dtype=np.int32)
    if not k:
        return perm
    d2 = np.sum((tc[:k, None, :] - pc[None, :, :]) ** 2, axis=-1)         # [k, C]
    allowed = np.ones(pc.shape[0], bool)
    for _ in range(4):
        cand = np.flatnonzero(allowed)
        if len(cand) < k:
            break
        _, c = linear_sum_assignment(d2[:, cand])
        perm[:k] = cand[c]
        if sib is None:
            break
        used = np.zeros(pc.shape[0], bool)
        used[perm[:k]] = True
        half = used & ~used[sib]                                         # pair half-used
        if not half.any():
            break
        allowed[half] = allowed[sib[half]] = False
    return perm


def assign(pred_c, true_c, k, pool: ThreadPoolExecutor | None = None, layout=None):
    """Host-side Hungarian pairing for a batch.

    pred_c[B,S,C,3] candidates, true_c[B,S,M,3] (padded; first k real), k[B,S] -> perm[B,S,M]
    (int32): perm[b,s,i] = candidate paired with true center i (0 for i >= k). `layout` =
    (singles, pairs) per atom enables whole-pair matching (see _assign_one). SciPy releases the
    GIL, so a thread pool solves the B·S small problems on several CPU cores at once.
    """
    pred_c, true_c, k = np.asarray(pred_c), np.asarray(true_c), np.asarray(k)
    B, S, M, _ = true_c.shape
    sib = pair_siblings(pred_c.shape[2], *layout) if layout else None
    jobs = [(pred_c[b, s], true_c[b, s], int(k[b, s]), sib) for b in range(B) for s in range(S)]
    rows = pool.map(lambda a: _assign_one(*a), jobs) if pool else (_assign_one(*a) for a in jobs)
    return np.stack(list(rows)).reshape(B, S, M)


def selection_targets(pred_c, mc, mask, perm, nms_radius, layout=None):
    """BCE target per candidate [B,S,C]: 1 if it is a matched candidate or lies within
    `nms_radius` of one (one-to-many, consistent with NMS at prediction), else 0.

    Why not one-to-one: a center sitting on a symmetry element is approached by identical
    candidates from the atoms the symmetry swaps; one-to-one targets would give these twins 1 and
    0 at once although their features are equal. With the radius they are both positives and
    NMS (model/heads.py:select_centers) keeps one. A ± pair's members share one logit, so the
    pair's target is the AND of its two members (`layout` = (singles, pairs))."""
    C = pred_c.shape[2]
    chosen = jnp.max(jax.nn.one_hot(perm, C, dtype=pred_c.dtype) * mask[..., None], axis=2)
    if nms_radius > 0:
        pc, mcs = jax.lax.stop_gradient(pred_c), jax.lax.stop_gradient(mc)
        d2 = jnp.sum((pc[:, :, :, None, :] - mcs[:, :, None, :, :]) ** 2, axis=-1)   # [B,S,C,M]
        near = jnp.any((d2 < nms_radius ** 2) & (mask[:, :, None, :] > 0), axis=-1)
        chosen = jnp.maximum(chosen, near.astype(chosen.dtype))
        if layout is not None:
            sib = jnp.asarray(pair_siblings(C, *layout))
            chosen = jnp.minimum(chosen, chosen[:, :, sib])
    return chosen


def wannier_match_loss(pred_c, pred_r, pred_l, true_c, true_r, k, perm, *, r_std, guide_sigma,
                       radius_weight, guide_weight, select_weight, position_loss="mse",
                       nms_radius=0.0, layout=None):
    """Batched, per-spin Wannier loss.

    pred_c[B,S,C,3], pred_r[B,S,C], pred_l[B,S,C]: candidates from the head (center, radius,
    selection logit).  true_c[B,S,M,3], true_r[B,S,M]: parsed centers padded to M (first k
    real).  k[B,S]: real centers per spin (= Σ valence/2).  perm[B,S,M]: pairing from `assign`.
    position_loss: "l1" = mean distance (Charbonnier, Å; a few far, misassigned pairs can't
    dominate the gradient) or "mse" = mean squared distance (Å²).
    Returns (total, {"pos", "radius": normalized, "select": BCE, "guide": per-center overlap}).
    """
    M = true_c.shape[2]
    mask = (jnp.arange(M)[None, None, :] < k[:, :, None]).astype(pred_c.dtype)   # [B,S,M]
    n = jnp.maximum(mask.sum(), 1.0)

    perm = jax.lax.stop_gradient(perm)
    mc = jnp.take_along_axis(pred_c, perm[..., None], axis=2)                    # matched xyz
    mr = jnp.take_along_axis(pred_r, perm, axis=2)                               # matched r
    d2 = jnp.sum((true_c - mc) ** 2, axis=-1)
    if position_loss == "l1":
        pos = jnp.sum(mask * jnp.sqrt(d2 + 1e-6)) / n           # Charbonnier, eps = 1e-3 Å
    elif position_loss == "mse":
        pos = jnp.sum(mask * d2) / n
    else:
        raise ValueError(f"wannier.position_loss must be 'l1' or 'mse', got {position_loss!r}")
    rad = jnp.sum(mask * ((true_r - mr) / r_std) ** 2) / n

    # selection: matched candidates (and their near-duplicates) -> 1, all others -> 0
    target = selection_targets(pred_c, mc, mask, perm, nms_radius, layout)
    select = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_l, target))

    # fixed-width cloud guide, one cloud per (molecule, spin), normalized per center
    s2 = jnp.full_like(mask, guide_sigma ** 2)
    flat = lambda a: a.reshape((-1,) + a.shape[2:])
    overlap = jax.vmap(cloud_l2)(flat(mc), flat(mask), flat(s2),
                                 flat(true_c), flat(mask), flat(s2))             # [B*S]
    guide = jnp.mean(overlap / jnp.maximum(flat(mask).sum(-1), 1.0))

    total = pos + radius_weight * rad + select_weight * select + guide_weight * guide
    return total, {"pos": pos, "radius": rad, "select": select, "guide": guide}
