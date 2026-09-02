"""Unit conversions and physical constants — the ONE place they live (CLAUDE.md §9, §11).

Positions are kept in Angstrom, energies in eV throughout the pipeline. SIESTA `.out`
trial-projector centers are in Bohr, but those are never used (§1), so Bohr appears here
only for completeness / the `.wout` spread if ever needed in atomic units.
"""

# CODATA-ish; matches SIESTA's internal constant closely enough for parsing.
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# SIESTA energies are already eV in `siesta: Total`; keep a Ry factor handy just in case.
RY_TO_EV = 13.605693122994
