"""Wannier count per spin from composition: Σ valence / 2, with clear errors otherwise."""

import numpy as np
import pytest

from data.electrons import centers_per_spin


@pytest.mark.parametrize("z, expected", [
    ([6, 1, 1, 1, 1], 4),                       # CH4: 4 + 4·1 = 8 electrons -> 4 per spin
    ([8, 1, 1], 4),                             # H2O: 6 + 2 = 8
    ([6] * 6 + [1] * 6 + [8] * 3, 24),          # C6H6O3: 24 + 6 + 18 = 48 (report example)
    ([9, 1], 4),                                # HF: 7 + 1
    ([7, 7], 5),                                # N2: 10
])
def test_count_is_half_the_valence_electrons(z, expected):
    k = centers_per_spin(np.array(z), n_spins=2)
    assert k.tolist() == [expected, expected]   # closed shell: same count in each spin


def test_odd_electron_count_is_rejected():
    with pytest.raises(ValueError, match="open-shell"):
        centers_per_spin([7], 2)                # N atom: 5 electrons


def test_unknown_element_is_rejected():
    with pytest.raises(ValueError, match="Z=\\[16\\]"):
        centers_per_spin([16, 1, 1], 2)         # H2S: sulfur not in the valence table
