"""Eval metrics (CLAUDE.md §10): energy MAE, center-set discrepancy."""

from __future__ import annotations

import numpy as np

from losses.wannier_cloud import wannier_loss


def energy_mae(apply_fn, params, examples) -> float:
    errs = [abs(float(apply_fn(params, ex["z"], ex["pos"])["energy"]) - float(ex["energy"]))
            for ex in examples]
    return float(np.mean(errs)) if errs else float("nan")


def wannier_discrepancy(apply_fn, params, examples, width_scale: float = 1.0) -> float:
    """Mean per-molecule Gaussian-cloud L2 between predicted and true center sets."""
    vals = [float(wannier_loss(apply_fn(params, ex["z"], ex["pos"])["wannier"],
                               ex["wannier"], width_scale)) for ex in examples]
    return float(np.mean(vals)) if vals else float("nan")
