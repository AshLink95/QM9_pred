"""Eval metrics (CLAUDE.md §10): energy MAE and an interpretable Wannier-center accuracy.

The Wannier metric Hungarian-matches predicted centers to true ones and reports distances in Å.
Matching is fine HERE (scoring a finished prediction); CLAUDE.md §6 rejects it only as a
training LOSS (needs padding, poor gradients) — the loss stays the Gaussian cloud.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def energy_mae(apply_fn, params, examples) -> float:
    errs = [abs(float(apply_fn(params, ex["z"], ex["pos"])["energy"]) - float(ex["energy"]))
            for ex in examples]
    return float(np.mean(errs)) if errs else float("nan")


def match_wannier(pred_c, pred_r, pred_p, true_c, true_r, thr=0.5):
    """Match one molecule's predicted centers (one spin) to its true centers.

    pred_c[M,3], pred_r[M], pred_p[M]: all slots from the Wannier head. Slots with
    presence > thr count as predicted centers. true_c[K,3], true_r[K]: parsed .wout values.
    Returns {"pairs": [(true_idx, pred_idx, dist_A, radius_err_A)], "kept": pred slot indices,
    "n_true": K, "n_pred": #kept}. Unequal counts: min(K, #kept) pairs; the rest are
    missed (true) or extra (predicted) centers.
    """
    kept = np.flatnonzero(np.asarray(pred_p) > thr)
    true_c, true_r = np.asarray(true_c), np.asarray(true_r)
    pairs = []
    if len(kept) and len(true_c):
        pc = np.asarray(pred_c)[kept]
        d = np.linalg.norm(true_c[:, None, :] - pc[None, :, :], axis=-1)   # [K, kept]
        ti, pi = linear_sum_assignment(d)                                 # min total distance
        pairs = [(int(t), int(kept[p]), float(d[t, p]),
                  float(np.asarray(pred_r)[kept[p]] - true_r[t])) for t, p in zip(ti, pi)]
    return {"pairs": pairs, "kept": kept.tolist(), "n_true": len(true_c), "n_pred": len(kept)}


def wannier_summary(matches):
    """Aggregate match_wannier results (any number of molecule/spin entries)."""
    dist = np.array([p[2] for m in matches for p in m["pairs"]])
    rerr = np.array([p[3] for m in matches for p in m["pairs"]])
    count_err = np.array([m["n_pred"] - m["n_true"] for m in matches])
    vals, cnts = np.unique(count_err, return_counts=True)
    some = len(dist) > 0
    return {
        "n_sets": len(matches),
        "n_matched": int(len(dist)),
        "center_mae": float(dist.mean()) if some else float("nan"),
        "center_rmse": float(np.sqrt((dist ** 2).mean())) if some else float("nan"),
        "center_median": float(np.median(dist)) if some else float("nan"),
        "center_p95": float(np.percentile(dist, 95)) if some else float("nan"),
        "radius_mae": float(np.abs(rerr).mean()) if some else float("nan"),
        "count_accuracy": float((count_err == 0).mean()) if len(count_err) else float("nan"),
        "count_error_breakdown": {int(v): int(c) for v, c in zip(vals, cnts)},
    }
