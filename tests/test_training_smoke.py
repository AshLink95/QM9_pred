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

from tests._fake import WANNIER_CFG, molecules

CFG = {
    "model": {"hidden_dim": 16, "n_layers": 2, "n_elements": 10,
              "rbf": {"enabled": True, "n_basis": 8, "cutoff": 10.0}},
    "wannier": WANNIER_CFG,
    "loss": {"lambda_energy": 1.0, "lambda_wannier": 1.0},
    "train": {"lr": 5e-3, "batch_size": 2, "epochs": 40, "seed": 0,
              "val_frac": 0.25, "test_frac": 0.25},
}


def _fabricate(n_mol=4, seed=0):
    return molecules(n_mol, seed)


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


def test_fit_atom_ref_recovers_known_references():
    from training.train import _fit_atom_ref
    true = {1: -13.0, 6: -155.0, 8: -430.0}
    rng = np.random.default_rng(1)
    ex = []
    for _ in range(50):
        z = rng.choice([1, 6, 8], size=int(rng.integers(3, 12)))
        ex.append({"z": z, "energy": sum(true[int(q)] for q in z)})
    ref, resid = _fit_atom_ref(ex, n_elements=10)
    for z, r in true.items():
        assert abs(ref[z] - r) < 1e-6
    assert resid < 1e-6 and ref[7] == 0.0          # absent element stays 0
