"""Fabricated molecules for tests that obey the dataset's physics: closed-shell compositions
(even valence electrons) with exactly Σ valence/2 Wannier centers per spin."""

import numpy as np

from data.electrons import centers_per_spin

WANNIER_CFG = {"max_centers": 10, "spins": ["up", "down"],
               "guide_sigma": 0.3, "radius_weight": 1.0, "guide_weight": 0.1,
               "select_weight": 1.0, "singles_per_atom": 2, "pairs_per_atom": 1}


def molecules(n, seed=0):
    """n molecules C1-2 O0-1 H2/4 (3-7 atoms, ≤ 9 centers per spin)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = np.array([6] * int(rng.integers(1, 3)) + [8] * int(rng.integers(0, 2))
                     + [1] * int(rng.choice([2, 4])), dtype=np.int32)
        k = int(centers_per_spin(z, 2)[0])
        out.append({"z": z, "pos": rng.standard_normal((len(z), 3)) * 1.5,
                    "energy": np.float64(rng.uniform(-500, -100)),
                    "wannier": [{"centers": rng.standard_normal((k, 3)),
                                 "radii": rng.uniform(0.75, 0.95, k)} for _ in range(2)]})
    return out
