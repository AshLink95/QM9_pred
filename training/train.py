"""Training loop: optax + per-molecule loss, checkpointing (CLAUDE.md §10).

QM9 molecules have different atom/center counts, so we avoid padded batching: the loss is
per-molecule and gradients are averaged over a minibatch (a plain Python list). jax jits the
single-molecule value_and_grad and caches one compile per distinct shape.

# ponytail: naive per-example grad loop; if throughput matters, switch to padded vmap with
# atom/center masks. QM9 is tiny so this is fine.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import optax
import yaml
from flax import serialization

from data.dataset import split
from model.model import model_from_config
from training.loss import example_to_jax, molecule_loss
from training.metrics import energy_mae


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def train(cfg: dict, examples: list[dict]):
    tcfg = cfg["train"]
    train_ex, val_ex, _ = split(examples, tcfg["val_frac"], tcfg["test_frac"], tcfg["seed"])
    train_j = [example_to_jax(e) for e in train_ex]
    val_j = [example_to_jax(e) for e in val_ex]

    model = model_from_config(cfg)
    params = model.init(jax.random.PRNGKey(tcfg["seed"]),
                        train_j[0]["z"], train_j[0]["pos"])
    opt = optax.adam(tcfg["lr"])
    opt_state = opt.init(params)

    lam_e = cfg["loss"]["lambda_energy"]
    lam_w = cfg["loss"]["lambda_wannier"]
    ws = cfg["wannier"]["width_scale"]

    @jax.jit
    def vgrad(p, ex):
        return jax.value_and_grad(
            lambda pp: molecule_loss(pp, model.apply, ex, lam_e, lam_w, ws)[0])(p)

    rng = np.random.default_rng(tcfg["seed"])
    bs = tcfg["batch_size"]
    for epoch in range(tcfg["epochs"]):
        order = rng.permutation(len(train_j))
        for k in range(0, len(order), bs):
            batch = [train_j[i] for i in order[k:k + bs]]
            grads = None
            for ex in batch:
                _, g = vgrad(params, ex)
                grads = g if grads is None else _tree_add(grads, g)
            grads = jax.tree_util.tree_map(lambda x: x / len(batch), grads)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  val energy MAE = "
                  f"{energy_mae(model.apply, params, val_j):.4f} eV")
    return params


def save_params(path: str | Path, params) -> None:
    Path(path).write_bytes(serialization.to_bytes(params))


def load_params(path: str | Path, template) -> object:
    return serialization.from_bytes(template, Path(path).read_bytes())
