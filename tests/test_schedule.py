"""Learning-rate schedule, early stopping, best/last files, and exact resume."""

import copy
import json

import jax
import numpy as np

from tests._fake import WANNIER_CFG, molecules
from training.train import make_lr, train_with_status

CFG = {
    "model": {"hidden_dim": 16, "n_layers": 2, "n_elements": 10,
              "rbf": {"enabled": True, "n_basis": 8, "cutoff": 10.0}},
    "wannier": WANNIER_CFG,
    "loss": {"lambda_energy": 1.0, "lambda_wannier": 1.0},
    "train": {"lr": 5e-3, "batch_size": 4, "epochs": 4, "ckpt_every": 1, "seed": 0,
              "split_seed": 0, "val_frac": 0.25, "test_frac": 0.25, "lr_schedule": "constant"},
}


def _cfg(**train):
    c = copy.deepcopy(CFG); c["train"].update(train); return c


def test_cosine_schedule_shape():
    tc = {"lr": 1e-3, "epochs": 10, "lr_schedule": "cosine", "warmup_epochs": 2,
          "lr_end_frac": 0.01}
    _, lr_at = make_lr(tc, spe=5)                    # 50 steps, 10 of warmup
    assert lr_at(0) == 0.0
    assert abs(lr_at(10) - 1e-3) < 1e-9              # peak at the end of the warmup
    assert abs(lr_at(50) - 1e-5) < 1e-9              # 1% of lr at the end
    assert lr_at(20) > lr_at(30) > lr_at(40)         # smooth decay


def test_early_stopping_keeps_best(tmp_path):
    ex = molecules(12)
    _, status = train_with_status(_cfg(lr=0.0, epochs=20, patience=2), ex,
                                  ckpt_path=tmp_path / "p.msgpack")
    assert status == "early_stopped"                 # lr=0 => never improves after epoch 0
    best = json.loads((tmp_path / "p.best.json").read_text())
    assert best["epoch"] == 0
    assert (tmp_path / "p.msgpack").exists() and (tmp_path / "p.last.msgpack").exists()


def test_resume_matches_uninterrupted_run(tmp_path):
    ex = molecules(12)
    (tmp_path / "a").mkdir()
    full, _ = train_with_status(_cfg(epochs=4), ex, ckpt_path=tmp_path / "a" / "p.msgpack")
    (tmp_path / "b").mkdir()
    train_with_status(_cfg(epochs=2), ex, ckpt_path=tmp_path / "b" / "p.msgpack")   # "killed"
    resumed, _ = train_with_status(_cfg(epochs=4), ex, ckpt_path=tmp_path / "b" / "p.msgpack")
    for a, b in zip(jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_stop_request_saves_state(tmp_path):
    _, status = train_with_status(_cfg(), molecules(12), ckpt_path=tmp_path / "p.msgpack",
                                  should_stop=lambda: True)
    assert status == "stopped" and (tmp_path / "p.last.msgpack").exists()
