"""Parse SIESTA input/output: geometry from `.fdf`, total energy from `.out` (CLAUDE.md §9).

Stage-1 data code (no learning). Returns atomic numbers `Z`, positions in Angstrom, and the
converged total energy in eV. The `.out` trial-projector Wannier centers are NOT parsed here
(§1) — they are never used.

We deliberately prefer the clean `.fdf` for geometry over `.out` coordinate dumps, which can
be duplicated/truncated in the log.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .units import BOHR_TO_ANG

# fdf keys are matched case-insensitively with '.', '-', '_' ignored (SIESTA fdf convention).
def _norm_key(s: str) -> str:
    return re.sub(r"[.\-_]", "", s).lower()


def _strip_comment(line: str) -> str:
    # fdf comment char is '#'. Everything after it (outside no quoting in fdf) is a comment.
    return line.split("#", 1)[0]


def _read_fdf_tokens(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (scalars, blocks). scalars: normkey -> value string (first token after key).
    blocks: normkey -> list of raw content lines (between %block/%endblock)."""
    scalars: dict[str, str] = {}
    blocks: dict[str, list[str]] = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        raw = _strip_comment(lines[i]).strip()
        i += 1
        if not raw:
            continue
        toks = raw.split()
        head = _norm_key(toks[0].lstrip("%"))   # fdf blocks are '%block'/'%endblock'
        if head == "block":
            name = _norm_key(toks[1])
            content: list[str] = []
            while i < len(lines):
                inner = _strip_comment(lines[i]).strip()
                i += 1
                if inner and _norm_key(inner.split()[0].lstrip("%")) == "endblock":
                    break
                if inner:
                    content.append(inner)
            blocks[name] = content
        else:
            # scalar: "Key value [unit]" -> store the value token (units handled by caller)
            scalars[head] = toks[1] if len(toks) > 1 else "true"
    return scalars, blocks


def _species_to_z(blocks: dict[str, list[str]]) -> dict[int, int]:
    """%block ChemicalSpeciesLabel lines: 'index Z label'."""
    key = _norm_key("ChemicalSpeciesLabel")
    if key not in blocks:
        raise ValueError("fdf missing %block ChemicalSpeciesLabel")
    mapping: dict[int, int] = {}
    for line in blocks[key]:
        t = line.split()
        mapping[int(t[0])] = int(t[1])
    return mapping


def _coord_scale(scalars: dict[str, str]) -> float:
    """AtomicCoordinatesFormat -> factor converting stored coords to Angstrom.
    Supports Ang and Bohr forms; fractional/scaled-by-lattice are rejected (QM9 is isolated
    molecules, expected in Ang or Bohr)."""
    fmt = scalars.get(_norm_key("AtomicCoordinatesFormat"), "bohr").lower()
    fmt = re.sub(r"[.\-_]", "", fmt)
    if "ang" in fmt:                       # Ang, NotScaledCartesianAng
        return 1.0
    if "bohr" in fmt:                      # Bohr, NotScaledCartesianBohr
        return BOHR_TO_ANG
    raise ValueError(
        f"Unsupported AtomicCoordinatesFormat '{fmt}'. Only Ang/Bohr cartesian handled "
        "(fractional/ScaledCartesian need lattice vectors; QM9 should not use them)."
    )


def parse_fdf(fdf_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """`.fdf` -> (Z[atoms] int, pos[atoms,3] float, Angstrom)."""
    scalars, blocks = _read_fdf_tokens(Path(fdf_path))
    sp2z = _species_to_z(blocks)
    scale = _coord_scale(scalars)

    key = _norm_key("AtomicCoordinatesAndAtomicSpecies")
    if key not in blocks:
        raise ValueError("fdf missing %block AtomicCoordinatesAndAtomicSpecies")

    zs: list[int] = []
    pos: list[list[float]] = []
    for line in blocks[key]:
        t = line.split()
        x, y, z = float(t[0]), float(t[1]), float(t[2])
        species = int(t[3])
        pos.append([x * scale, y * scale, z * scale])
        zs.append(sp2z[species])
    return np.array(zs, dtype=np.int32), np.array(pos, dtype=np.float64)


# Converged total energy. SIESTA prints many "siesta: Total = ..." lines during SCF; the last
# one is the converged value (§1: targets are the final minimized values).
_TOTAL_RE = re.compile(r"siesta:\s*Total\s*=\s*(-?\d+\.?\d*)")


def parse_energy(out_path: str | Path) -> float:
    """`.out` -> converged total energy (eV). Takes the LAST 'siesta: Total =' match."""
    text = Path(out_path).read_text()
    matches = _TOTAL_RE.findall(text)
    if not matches:
        raise ValueError(f"No 'siesta: Total =' line found in {out_path}")
    return float(matches[-1])
