"""Assemble parsed molecules into training examples; split + cache (CLAUDE.md §9).

Stage 1-2 glue (no learning). One molecule = one `.fdf` (+ `.out` energy + up/down `.wout`),
matched by shared filename stem. The exact stem->sibling naming is site-specific; the glob
rules below are the one assumption to revisit when real samples land (see plan Open items).

Each example:
    {z[N], pos[N,3], energy, wannier=[{centers[K_s,3], radii[K_s]} for spin in spins]}
`wannier` is ordered to match config `wannier.spins` (e.g. [up, down]).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .graph import build_graph
from .parse_siesta import parse_energy, parse_fdf
from .parse_wout import parse_wout


def _find_one(directory: Path, pattern: str) -> Path:
    hits = sorted(directory.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    return hits[0]


def load_molecule(fdf_path: Path, spins: list[str]) -> dict:
    """Parse one molecule's files (siblings of the .fdf, matched by stem)."""
    fdf_path = Path(fdf_path)
    d, stem = fdf_path.parent, fdf_path.stem
    z, pos = parse_fdf(fdf_path)
    energy = parse_energy(_find_one(d, f"{stem}*.out"))
    wannier = []
    for spin in spins:
        w = parse_wout(_find_one(d, f"{stem}*{spin}*.wout"), spin=spin)
        wannier.append({"centers": w["centers"], "radii": w["radii"]})
    return {"z": z, "pos": pos, "energy": np.float64(energy), "wannier": wannier}


def build_dataset(data_dir: str | Path, spins: list[str]) -> tuple[list[dict], dict]:
    """Parse all molecules under data_dir. Returns (examples, meta).
    meta['max_centers'] sizes the Wannier head (§6, §9)."""
    data_dir = Path(data_dir)
    examples = [load_molecule(f, spins) for f in sorted(data_dir.glob("*.fdf"))]
    if not examples:
        raise FileNotFoundError(f"no .fdf molecules found under {data_dir}")
    max_centers = max((max(len(w["radii"]) for w in ex["wannier"]) for ex in examples),
                      default=0)
    return examples, {"max_centers": int(max_centers), "spins": spins}


def split(examples: list[dict], val_frac: float, test_frac: float, seed: int = 0):
    """Deterministic train/val/test split."""
    idx = np.random.default_rng(seed).permutation(len(examples))
    n_test = int(len(examples) * test_frac)
    n_val = int(len(examples) * val_frac)
    test, val, train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    pick = lambda ids: [examples[i] for i in ids]
    return pick(train), pick(val), pick(test)


def save_cache(path: str | Path, examples: list[dict], meta: dict) -> None:
    Path(path).write_bytes(pickle.dumps({"examples": examples, "meta": meta}))


def load_cache(path: str | Path) -> tuple[list[dict], dict]:
    blob = pickle.loads(Path(path).read_bytes())
    return blob["examples"], blob["meta"]


# `build_graph` re-exported for callers that want precomputed edges (§3); the model recomputes
# distances in-layer for grad flow, so training does not need this.
__all__ = ["load_molecule", "build_dataset", "split", "save_cache", "load_cache",
           "build_graph"]
