"""Assemble parsed molecules into training examples; split + cache (CLAUDE.md §9).

Stage 1-2 glue (no learning). One molecule = one `.fdf` (geometry) + one `.out` (energy) +
one `.wout` per spin (converged Wannier targets). These files may live in SEPARATE roots
(e.g. done/*.fdf, out_files/*.out, wout_files/*.wout), so we match them by MOLECULE ID = the
filename up to the first '.' (e.g. `C2F3N3O_133481.fdf`, `C2F3N3O_133481.out`,
`C2F3N3O_133481.manifold.valence.up.wout` all -> id `C2F3N3O_133481`). Molecules missing an
`.out` or a spin `.wout` are skipped (not fatal), and the skip count is reported.

Each example:
    {z[N], pos[N,3], energy, wannier=[{centers[K_s,3], radii[K_s]} for spin in spins]}
`wannier` is ordered to match config `wannier.spins` (e.g. [up, down]).
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from .graph import build_graph
from .parse_siesta import parse_energy, parse_fdf
from .parse_wout import parse_wout


def _mol_id(path: Path) -> str:
    return path.name.split(".", 1)[0]


def _index(dirs: list, ext: str) -> dict[str, list[Path]]:
    """Map molecule id -> list of matching files (across all roots, recursively)."""
    idx: dict[str, list[Path]] = defaultdict(list)
    for d in dirs:
        for p in Path(d).rglob(f"*.{ext}"):
            idx[_mol_id(p)].append(p)
    return idx


def load_molecule(fdf_path: Path, out_path: Path, wout_by_spin: dict[str, Path],
                  spins: list[str]) -> dict:
    """Parse one molecule from already-resolved file paths."""
    z, pos = parse_fdf(fdf_path)
    energy = parse_energy(out_path)
    wannier = []
    for spin in spins:
        w = parse_wout(wout_by_spin[spin], spin=spin)
        wannier.append({"centers": w["centers"], "radii": w["radii"]})
    return {"z": z, "pos": pos, "energy": np.float64(energy), "wannier": wannier}


def build_dataset(data_dirs: str | Path | list, spins: list[str]) -> tuple[list[dict], dict]:
    """Parse all molecules under one or more roots (recursively). Returns (examples, meta).
    meta['max_centers'] sizes the Wannier head (§6, §9). Files are matched across roots by
    molecule id; incomplete molecules are skipped."""
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    fdfs = sorted(f for d in data_dirs for f in Path(d).rglob("*.fdf"))
    if not fdfs:
        raise FileNotFoundError(f"no .fdf files found under {data_dirs}")
    outs = _index(data_dirs, "out")
    wouts = _index(data_dirs, "wout")

    examples, skipped = [], []
    for f in fdfs:
        mid = _mol_id(f)
        if mid not in outs:
            skipped.append((mid, "no .out")); continue
        wout_by_spin = {}
        for spin in spins:                       # pick the .wout for this molecule + spin
            match = [p for p in wouts.get(mid, []) if spin in p.name]
            if match:
                wout_by_spin[spin] = match[0]
        if len(wout_by_spin) < len(spins):
            skipped.append((mid, "missing spin .wout")); continue
        examples.append(load_molecule(f, outs[mid][0], wout_by_spin, spins))

    if not examples:
        raise FileNotFoundError(
            f"found {len(fdfs)} .fdf but none had a matching .out + {spins} .wout. "
            f"First skips: {skipped[:5]}")
    if skipped:
        print(f"skipped {len(skipped)} incomplete molecules (e.g. {skipped[:3]})")

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
