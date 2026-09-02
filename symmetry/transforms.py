"""Symmetry transforms for the equivariance suite (CLAUDE.md §8).

Load-bearing: the whole project's correctness claim is the symmetry guarantee, so these are
first-class. Rotations are proper (det +1); mirror is improper (det -1).
"""

from __future__ import annotations

import numpy as np


def translate(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pos + np.asarray(t)


def random_rotation(seed: int = 0) -> np.ndarray:
    """Uniform-ish proper rotation via QR of a random Gaussian matrix (det forced to +1)."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))                 # fix QR sign ambiguity
    if np.linalg.det(q) < 0:                    # ensure proper rotation
        q[:, 0] = -q[:, 0]
    return q


def rotate(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    return pos @ np.asarray(R).T                # row vectors: r' = R r


def mirror(pos: np.ndarray, axis: int = 0) -> np.ndarray:
    """Reflection through the plane normal to `axis` (improper, det -1)."""
    M = np.eye(3)
    M[axis, axis] = -1.0
    return pos @ M.T


def permutation(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)
