"""Shared helpers for the demo notebooks ONLY (CLAUDE.md §11: nothing outside notebooks
depends on this). Keeps the 4 notebooks DRY — one demo molecule, one plot function.

Runs with no real SIESTA data: a hardcoded methane geometry + fabricated Wannier targets,
same spirit as tests/test_training_smoke.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np

# notebooks execute with cwd = notebooks/; put the repo root on the path so `model`/`data`/...
# resolve exactly as they do for pytest (pyproject pythonpath=["."]).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.model import model_from_config  # noqa: E402

# Small demo config — enough layers to exercise the equivariance machinery, small enough to
# stay snappy in a notebook.
DEMO_CONFIG = {
    "model": {"hidden_dim": 32, "n_layers": 3, "n_elements": 100,
              "rbf": {"enabled": True, "n_basis": 12, "cutoff": 10.0}},
    "wannier": {"max_centers": 8, "spins": ["up", "down"], "width_scale": 1.0},
    "loss": {"lambda_energy": 1.0, "lambda_wannier": 1.0},
    "train": {"lr": 5e-3, "batch_size": 2, "epochs": 40, "seed": 0,
              "val_frac": 0.25, "test_frac": 0.25},
}

# Z -> (matplotlib color, marker size, label) for readable 3D plots.
_ELEMENT_STYLE = {1: ("lightgray", 120, "H"), 6: ("dimgray", 300, "C"),
                  7: ("royalblue", 300, "N"), 8: ("crimson", 320, "O")}


def demo_molecule():
    """Methane CH4: Z[5], pos[5,3] in Angstrom (C at origin, 4 H tetrahedral)."""
    z = np.array([6, 1, 1, 1, 1], dtype=np.int32)
    bond = 1.09
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pos = np.vstack([np.zeros(3), dirs * bond])
    return z, pos


def fabricate_wannier(pos, spins=("up", "down"), seed=0):
    """Plausible per-spin centers near the C-H bond midpoints (a closed-shell cartoon).
    Returns a length-len(spins) list of {'centers'[K,3], 'radii'[K]} — the shape the loss and
    WannierHead speak."""
    rng = np.random.default_rng(seed)
    midpoints = 0.5 * (pos[0] + pos[1:])          # C-H bond midpoints
    out = []
    for _ in spins:
        jitter = rng.standard_normal(midpoints.shape) * 0.03
        out.append({"centers": midpoints + jitter,
                    "radii": rng.uniform(0.6, 0.8, size=len(midpoints))})
    return out


def init_model(config=DEMO_CONFIG, seed=0):
    """Build the small demo model + random params. Returns (model, params, apply_fn)."""
    z, pos = demo_molecule()
    model = model_from_config(config)
    params = model.init(jax.random.PRNGKey(seed), z, pos)
    return model, params, model.apply


def plot_molecule(ax, z, pos, centers=None, title="", center_color="seagreen"):
    """3D scatter of atoms (colored/sized by element) + optional Wannier centers (x marks)."""
    pos = np.asarray(pos)
    seen = set()
    for zi, p in zip(np.asarray(z), pos):
        color, size, label = _ELEMENT_STYLE.get(int(zi), ("orange", 200, str(int(zi))))
        ax.scatter(*p, c=color, s=size, edgecolors="k", depthshade=True,
                   label=label if label not in seen else None)
        seen.add(label)
    if centers is not None:
        c = np.asarray(centers).reshape(-1, 3)
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=center_color, marker="x", s=80,
                   label="Wannier centers")
    ax.set_title(title)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.legend(loc="upper left", fontsize=8)
