"""Data parallelism: training on 2 (fake CPU) devices gives the same result as on 1."""

import json
import os
import subprocess
import sys

SCRIPT = r"""
import json, jax, numpy as np
from tests._fake import WANNIER_CFG, molecules
from training.train import train_with_status
cfg = {"model": {"hidden_dim": 16, "n_layers": 2, "n_elements": 10,
                 "rbf": {"enabled": True, "n_basis": 8, "cutoff": 10.0}},
       "wannier": WANNIER_CFG, "loss": {"lambda_energy": 1.0, "lambda_wannier": 1.0},
       "train": {"lr": 5e-3, "batch_size": 4, "epochs": 2, "ckpt_every": 1, "seed": 0,
                 "split_seed": 0, "val_frac": 0.25, "test_frac": 0.25}}
params, status = train_with_status(cfg, molecules(12))
leaves = jax.tree_util.tree_leaves(params)
print("RESULT", json.dumps({"devices": jax.device_count(), "status": status,
                            "norm": float(sum(np.abs(np.asarray(x)).sum() for x in leaves))}))
"""


def _run(n_devices):
    env = {**os.environ, "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_devices}",
           "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(__file__)), timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.split("RESULT", 1)[1])


def test_two_devices_match_one():
    one, two = _run(1), _run(2)
    assert one["devices"] == 1 and two["devices"] == 2
    assert abs(one["norm"] - two["norm"]) / one["norm"] < 1e-4
