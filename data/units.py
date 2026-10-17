"""Unit conversions and physical constants — the ONE place they live (CLAUDE.md §9, §11).

Positions are kept in Angstrom throughout the pipeline. Bohr appears in the `.fdf` when
coordinates are given in Bohr (`AtomicCoordinatesFormat Bohr`); SIESTA `.out` trial-projector
centers are also in Bohr, but those are never used (§1).
"""

# CODATA-ish; matches SIESTA's internal constant closely enough for parsing.
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
