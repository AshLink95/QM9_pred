"""Wannier loss: Hungarian-matched positions + radii, plus a fixed-width cloud guide (per spin).

Replaces the presence-weighted Gaussian cloud, which let the model blur several centers into a
few wide, faint blobs (test report: count 0.1%, center 0.61 Å, radius 0.22 Å). Here:

  * the COUNT is not predicted: spin s of a molecule has k = Σ valence/2 real centers
    (data/electrons.py); the first k predicted slots of that spin are its centers;
  * POSITIONS: the k predicted and k true centers of ONE spin are paired one-to-one by the
    Hungarian algorithm (min total squared distance), then each pair's squared distance is
    minimized (DETR-style set loss; the pairing itself is not differentiated);
  * RADII: each matched slot regresses its own true radius (normalized by the dataset std);
  * GUIDE: the closed-form cloud overlap (losses/wannier_cloud.py) with a FIXED width on both
    sides and amplitude 1 per center — a smooth, order-free pull that steadies early training
    when pairings still flip. Widths here are fixed, so they can't be used to blur.

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
from scipy.optimize import linear_sum_assignment

from losses.wannier_cloud import cloud_l2


def _assign_one(pc, tc, k):
    """One (molecule, spin): perm over M slots; first k true centers paired with first k slots."""
    M = pc.shape[0]
    perm = np.arange(M, dtype=np.int32)
    if k:
        d2 = np.sum((tc[:k, None, :] - pc[None, :k, :]) ** 2, axis=-1)
        _, cols = linear_sum_assignment(d2)
        perm[:k] = cols
    return perm


def assign(pred_c, true_c, k, pool: ThreadPoolExecutor | None = None):
    """Host-side Hungarian pairing for a batch.

    pred_c[B,S,M,3], true_c[B,S,M,3] (padded; first k real), k[B,S] -> perm[B,S,M] (int32):
    perm[b,s,i] = predicted slot paired with true center i (identity for i >= k). SciPy releases
    the GIL, so a thread pool solves the B·S small problems on several CPU cores at once.
    """
    pred_c, true_c, k = np.asarray(pred_c), np.asarray(true_c), np.asarray(k)
    B, S, M, _ = pred_c.shape
    jobs = [(pred_c[b, s], true_c[b, s], int(k[b, s])) for b in range(B) for s in range(S)]
    rows = pool.map(lambda a: _assign_one(*a), jobs) if pool else (_assign_one(*a) for a in jobs)
    return np.stack(list(rows)).reshape(B, S, M)


def wannier_match_loss(pred_c, pred_r, true_c, true_r, k, perm, *, r_std, guide_sigma,
                       radius_weight, guide_weight):
    """Batched, per-spin Wannier loss.

    pred_c[B,S,M,3], pred_r[B,S,M]: all slots from the head.
    true_c[B,S,M,3], true_r[B,S,M]: parsed centers padded to M (first k real).
    k[B,S]: real centers per spin (= Σ valence/2).  perm[B,S,M]: pairing from `assign`.
    Returns (total, {"pos": Å², "radius": normalized, "guide": per-center overlap}).
    """
    M = pred_c.shape[2]
    mask = (jnp.arange(M)[None, None, :] < k[:, :, None]).astype(pred_c.dtype)   # [B,S,M]
    n = jnp.maximum(mask.sum(), 1.0)

    perm = jax.lax.stop_gradient(perm)
    mc = jnp.take_along_axis(pred_c, perm[..., None], axis=2)                    # matched xyz
    mr = jnp.take_along_axis(pred_r, perm, axis=2)                               # matched r
    pos = jnp.sum(mask * jnp.sum((true_c - mc) ** 2, axis=-1)) / n
    rad = jnp.sum(mask * ((true_r - mr) / r_std) ** 2) / n

    # fixed-width cloud guide, one cloud per (molecule, spin), normalized per center
    s2 = jnp.full_like(mask, guide_sigma ** 2)
    flat = lambda a: a.reshape((-1,) + a.shape[2:])
    overlap = jax.vmap(cloud_l2)(flat(pred_c), flat(mask), flat(s2),
                                 flat(true_c), flat(mask), flat(s2))             # [B*S]
    guide = jnp.mean(overlap / jnp.maximum(flat(mask).sum(-1), 1.0))

    total = pos + radius_weight * rad + guide_weight * guide
    return total, {"pos": pos, "radius": rad, "guide": guide}
