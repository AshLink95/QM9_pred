"""End-to-end training smoke test on fabricated molecules (no real data needed).

Verifies the full path -- model init, matched Wannier loss, optax step, split -- runs and
actually reduces the Wannier center error on the training molecules.
"""

import jax
import numpy as np

from data.dataset import split
from data.electrons import centers_per_spin
from model.heads import select_centers
from model.model import model_from_config
from training.metrics import match_wannier
from training.train import train

from tests._fake import WANNIER_CFG, molecules

CFG = {
    "model": {"hidden_dim": 16, "n_layers": 2, "n_elements": 10,
              "rbf": {"enabled": True, "n_basis": 8, "cutoff": 10.0}},
    "wannier": WANNIER_CFG,
    "train": {"lr": 5e-3, "batch_size": 2, "epochs": 40, "seed": 0,
              "val_frac": 0.25, "test_frac": 0.25},
}


def _center_mae(model, params, examples):
    """Mean matched distance (Å) of the selected centers, both spins."""
    d = []
    for e in examples:
        k = centers_per_spin(e["z"], 2)
        sel = select_centers(model.apply(params, e["z"], e["pos"])["wannier"], k)
        for s in range(2):
            w = e["wannier"][s]
            m = match_wannier(sel["centers"][s], sel["radii"][s], k[s], w["centers"], w["radii"])
            d += [p[2] for p in m["pairs"]]
    return float(np.mean(d))


def test_training_reduces_error():
    examples = molecules(4)
    train_ex, _, _ = split(examples, CFG["train"]["val_frac"], CFG["train"]["test_frac"], 0)
    model = model_from_config(CFG)
    params0 = model.init(jax.random.PRNGKey(0), train_ex[0]["z"], train_ex[0]["pos"])
    before = _center_mae(model, params0, train_ex)

    params = train(CFG, examples)
    after = _center_mae(model, params, train_ex)

    assert np.isfinite(after)
    assert after < before        # the loop actually learns
