"""End-to-end training smoke test on fabricated molecules (no real data needed).

Verifies the full path -- model init, per-molecule combined loss, optax step, split -- runs
and actually reduces training error. Real-data training waits on samples (see plan).
"""

import jax
import numpy as np

from model.model import model_from_config
from training.loss import example_to_jax
from training.metrics import energy_mae
from training.train import train

CFG = {
    "model": {"hidden_dim": 16, "n_layers": 2, "n_elements": 10,
              "rbf": {"enabled": True, "n_basis": 8, "cutoff": 10.0}},
    "wannier": {"max_centers": 4, "spins": ["up", "down"], "width_scale": 1.0},
    "loss": {"lambda_energy": 1.0, "lambda_wannier": 1.0},
    "train": {"lr": 5e-3, "batch_size": 2, "epochs": 40, "seed": 0,
              "val_frac": 0.25, "test_frac": 0.25},
}


def _fabricate(n_mol=4, seed=0):
    rng = np.random.default_rng(seed)
    ex = []
    for _ in range(n_mol):
        n = int(rng.integers(4, 6))
        k = int(rng.integers(2, 4))
        ex.append({
            "z": rng.integers(1, 9, size=n).astype(np.int32),
            "pos": (rng.standard_normal((n, 3)) * 1.5),
            "energy": np.float64(rng.uniform(-500, -100)),
            "wannier": [{"centers": rng.standard_normal((k, 3)),
                         "radii": rng.uniform(0.3, 0.9, size=k)} for _ in range(2)],
        })
    return ex


def test_training_reduces_error():
    examples = _fabricate()
    model = model_from_config(CFG)
    ex_j = [example_to_jax(e) for e in examples]
    params0 = model.init(jax.random.PRNGKey(0), ex_j[0]["z"], ex_j[0]["pos"])
    before = energy_mae(model.apply, params0, ex_j)

    params = train(CFG, examples)
    after = energy_mae(model.apply, params, ex_j)

    assert np.isfinite(after)
    assert after < before        # the loop actually learns
