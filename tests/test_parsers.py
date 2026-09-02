"""Parser tests on synthetic SIESTA/Wannier90 fixtures (CLAUDE.md §9).

Real-file hand-verification is deferred until sample .fdf/.out/.wout land (see plan);
these fixtures pin the parsing LOGIC now: species->Z map, Bohr->Ang conversion, last-Total
selection, Final-State-only WF extraction, spread->radius.
"""

import numpy as np

from data.parse_siesta import parse_fdf, parse_energy
from data.parse_wout import parse_wout
from data.units import BOHR_TO_ANG

FDF_ANG = """\
# a tiny water-like molecule in Angstrom
NumberOfAtoms 3
%block ChemicalSpeciesLabel
 1  8  O
 2  1  H
%endblock ChemicalSpeciesLabel
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
 0.000  0.000  0.000  1   # O
 0.757  0.586  0.000  2   # H
-0.757  0.586  0.000  2   # H
%endblock AtomicCoordinatesAndAtomicSpecies
"""

FDF_BOHR = FDF_ANG.replace("AtomicCoordinatesFormat Ang",
                           "AtomicCoordinatesFormat Bohr")

OUT = """\
siesta: Total =      -100.5
siesta: Total =      -466.123456
"""

WOUT = """\
 some minimization log
  WF centre and spread    1  (  9.9, 9.9, 9.9 )   99.0   # intermediate, must be ignored
 Final State
  WF centre and spread    1  (  0.000000,  0.000000,  0.100000 )     0.25000000
  WF centre and spread    2  ( -0.500000,  0.300000,  0.000000 )     1.00000000
 Sum of centres and spreads (  ... )   ...
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_parse_fdf_ang(tmp_path):
    z, pos = parse_fdf(_write(tmp_path, "m.fdf", FDF_ANG))
    assert list(z) == [8, 1, 1]
    assert pos.shape == (3, 3)
    np.testing.assert_allclose(pos[1], [0.757, 0.586, 0.0])


def test_parse_fdf_bohr_converts(tmp_path):
    z, pos = parse_fdf(_write(tmp_path, "m.fdf", FDF_BOHR))
    np.testing.assert_allclose(pos[1], np.array([0.757, 0.586, 0.0]) * BOHR_TO_ANG)


def test_parse_energy_takes_last(tmp_path):
    assert parse_energy(_write(tmp_path, "m.out", OUT)) == -466.123456


def test_parse_wout_final_state_only(tmp_path):
    out = parse_wout(_write(tmp_path, "m.up.wout", WOUT), spin="up")
    assert out["centers"].shape == (2, 2 + 1)  # 2 centers, xyz
    np.testing.assert_allclose(out["centers"][0], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(out["radii"], [0.5, 1.0])  # sqrt(0.25), sqrt(1.0)
    assert list(out["spin"]) == ["up", "up"]
