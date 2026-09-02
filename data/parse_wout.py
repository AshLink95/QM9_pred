"""Parse converged Wannier centers from Wannier90 `.wout` files (CLAUDE.md §9).

Targets are the CONVERGED values in the `Final State` section — NOT the trial-projector
centers in the main SIESTA `.out` (which are never used, §1). Centers are in Angstrom;
each center's radius is derived from its spread. Spin channel = which manifold file
(`*.up.wout` vs `*.down.wout`).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# A WF line inside the Final State block:
#   WF centre and spread    1  ( -0.000005, -0.000234,  1.155586 )     0.71534663
_WF_RE = re.compile(
    r"WF centre and spread\s+\d+\s+"
    r"\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)\s+"
    r"(-?\d+\.\d+)"
)


def _final_state_block(text: str) -> str:
    """Slice from the 'Final State' header to the following 'Sum of centres' line.
    Restricting to this block avoids the per-iteration WF lines printed during minimization.
    """
    start = text.rfind("Final State")
    if start == -1:
        raise ValueError("No 'Final State' section found in .wout")
    tail = text[start:]
    end = tail.find("Sum of centres")
    return tail if end == -1 else tail[:end]


def parse_wout(wout_path: str | Path, spin: str) -> dict[str, np.ndarray]:
    """`.wout` + spin label -> dict with:
        centers[k,3] (Angstrom), radii[k] (Angstrom, = sqrt(spread)), spin[k] (str label).
    """
    text = Path(wout_path).read_text()
    block = _final_state_block(text)
    rows = _WF_RE.findall(block)
    if not rows:
        raise ValueError(f"No WF centre lines in Final State of {wout_path}")

    arr = np.array(rows, dtype=np.float64)      # (k, 4): x, y, z, spread
    centers = arr[:, :3]
    spread = arr[:, 3]
    radii = np.sqrt(spread)                     # spread is <r^2>-<r>^2 (Ang^2); radius ~ sqrt
    return {
        "centers": centers,
        "radii": radii,
        "spin": np.array([spin] * len(centers)),
    }
