"""Wannier center count per spin, computed from composition (deterministic, no learning).

Only valence electrons are simulated (the pseudopotentials replace core electrons). In a
closed-shell molecule half of them are spin-up and half spin-down, and in a spin-polarized
calculation each Wannier function of a spin channel holds exactly one electron of that spin. So
each spin channel has (total valence electrons) / 2 Wannier centers.
Verified on the dataset: matches the parsed .wout count in 100% of 23,588 (molecule, spin) sets.

The valence numbers below must match the SIESTA pseudopotentials the data was generated with.
Open-shell molecules (unequal up/down counts) would need the spin multiplicity as an extra
input — out of scope here (every QM9 molecule in this dataset is closed-shell).
"""

from __future__ import annotations

import numpy as np

VALENCE = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7}      # H, C, N, O, F — pseudopotential valence charges

# Z -> valence lookup table (0 = unknown element)
_TABLE = np.zeros(119, dtype=np.int64)
for _z, _v in VALENCE.items():
    _TABLE[_z] = _v


def centers_per_spin(z, n_spins: int) -> np.ndarray:
    """Atomic numbers z[N] -> number of Wannier centers in each spin channel, int array [S]."""
    z = np.asarray(z)
    unknown = sorted({int(q) for q in z if q >= len(_TABLE) or _TABLE[q] == 0})
    if unknown:
        raise ValueError(f"no valence charge for element(s) Z={unknown}; add them to "
                         "data/electrons.py:VALENCE (must match the SIESTA pseudopotentials)")
    total = int(_TABLE[z].sum())
    if total % 2:
        raise ValueError(f"odd valence electron count ({total}): open-shell molecule, "
                         "not supported (needs spin multiplicity)")
    return np.full(n_spins, total // 2, dtype=np.int32)
