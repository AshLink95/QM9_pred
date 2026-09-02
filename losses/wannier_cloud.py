"""Ordering-free Wannier-center loss: closed-form Gaussian-cloud L2 overlap (CLAUDE.md §6).

Each set of centers is a density cloud rho(x) = sum_k w_k G(x; c_k, sigma_k). The loss is

    L = integral (rho_pred - rho_true)^2 dx
      = <p,p> - 2<p,t> + <t,t>

where every term is a closed-form sum of pairwise Gaussian OVERLAPS -- O(n^2), no grid, no
Monte Carlo, gradients everywhere (§6: do NOT grid/MC, do NOT Hungarian/Sinkhorn, do NOT
volumetric IoU). Clouds are built and compared PER SPIN (up<->up, down<->down only).

Overlap of two isotropic 3D Gaussians exp(-||x-a||^2 / (2 s)) (variance s = sigma^2):

    O(a,b,si,sj) = (2 pi)^{3/2} (si sj / (si + sj))^{3/2} exp(-||a-b||^2 / (2 (si+sj)))

Amplitudes w carry presence (pred) / mask (true padding): a slot with w=0 drops out, so
variable center count is handled for free (§6).
"""

from __future__ import annotations

import jax.numpy as jnp

_TWO_PI_32 = (2.0 * jnp.pi) ** 1.5


def _pairwise_overlap_sum(ca, wa, sa, cb, wb, sb):
    """sum_{i,j} wa_i wb_j O(ca_i, cb_j, sa_i, sb_j). Shapes: c[*,3], w[*], s[*]."""
    d2 = jnp.sum((ca[:, None, :] - cb[None, :, :]) ** 2, axis=-1)      # [A,B]
    ssum = sa[:, None] + sb[None, :]                                    # [A,B]
    prefac = _TWO_PI_32 * (sa[:, None] * sb[None, :] / ssum) ** 1.5
    overlap = prefac * jnp.exp(-d2 / (2.0 * ssum))
    return jnp.sum(wa[:, None] * wb[None, :] * overlap)


def cloud_l2(pred_c, pred_w, pred_s, true_c, true_w, true_s):
    """L2 between one spin's predicted and true clouds. Amplitudes w, variances s=sigma^2."""
    pp = _pairwise_overlap_sum(pred_c, pred_w, pred_s, pred_c, pred_w, pred_s)
    tt = _pairwise_overlap_sum(true_c, true_w, true_s, true_c, true_w, true_s)
    pt = _pairwise_overlap_sum(pred_c, pred_w, pred_s, true_c, true_w, true_s)
    return pp - 2.0 * pt + tt


def wannier_loss(pred, true, width_scale: float = 1.0):
    """Sum of per-spin cloud L2 (§6: split by spin).

    pred: {centers[S,M,3], radii[S,M], presence[S,M]}  (from WannierHead)
    true: length-S sequence of {centers[K_s,3], radii[K_s]}  (K_s varies freely per spin;
          no padding needed since this is a per-molecule loss). Pred amplitude = presence,
          true amplitude = 1.
    Widths sigma = width_scale * radii; variance s = sigma^2.
    """
    S = pred["centers"].shape[0]
    total = 0.0
    for s in range(S):
        pc, pr, pw = pred["centers"][s], pred["radii"][s], pred["presence"][s]
        tc, tr = true[s]["centers"], true[s]["radii"]
        tw = jnp.ones(tc.shape[0])
        ps = (width_scale * pr) ** 2 + 1e-6
        ts = (width_scale * tr) ** 2 + 1e-6
        total = total + cloud_l2(pc, pw, ps, tc, tw, ts)
    return total
