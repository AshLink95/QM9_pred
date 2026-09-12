"""Training loop: padded-batch vmap + optax, with periodic checkpoint/resume (CLAUDE.md §10).

Why batched (not per-molecule): jitting a per-molecule step recompiles for every distinct
shape, and the Wannier loss shape depends on each molecule's true center count K — so per
molecule you get a combinatorial (N, K_up, K_down) shape storm → thousands of XLA/LLVM
compiles → the compiler runs out of memory. Here we instead:

  * bucket molecules by atom count N (same N → no atom padding, clean vmap),
  * pad true Wannier centers to a fixed Kmax with a 0/1 mask (padded centers get amplitude 0,
    so they vanish from the cloud loss — CLAUDE.md §6),

which collapses everything to a handful of fixed shapes: bounded compiles + real batching.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import serialization

from data.dataset import split
from losses.wannier_cloud import cloud_l2
from model.model import model_from_config


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ------------------------------------------------------------------ padding / batching (numpy)
def _pad_wannier(ex, n_spins, kmax):
    """example -> centers[S,Kmax,3], radii[S,Kmax], mask[S,Kmax] (1 real, 0 pad)."""
    c = np.zeros((n_spins, kmax, 3)); r = np.zeros((n_spins, kmax)); m = np.zeros((n_spins, kmax))
    for s in range(n_spins):
        w = ex["wannier"][s]; k = len(w["radii"])
        c[s, :k] = w["centers"]; r[s, :k] = w["radii"]; m[s, :k] = 1.0
    return c, r, m


def _stack(chunk, n_spins, kmax):
    """Stack a list of same-N examples into batched arrays."""
    crm = [_pad_wannier(e, n_spins, kmax) for e in chunk]
    return {
        "z": np.stack([e["z"] for e in chunk]),
        "pos": np.stack([e["pos"] for e in chunk]),
        "energy": np.array([e["energy"] for e in chunk], dtype=np.float64),
        "centers": np.stack([c for c, _, _ in crm]),
        "radii": np.stack([r for _, r, _ in crm]),
        "mask": np.stack([m for _, _, m in crm]),
    }


def _batches(examples, batch_size, n_spins, kmax, rng):
    """Yield stacked batches, bucketed by atom count so each batch has one shape."""
    buckets = defaultdict(list)
    for e in examples:
        buckets[len(e["z"])].append(e)
    order = []
    for n, exs in buckets.items():
        idx = rng.permutation(len(exs))
        for k in range(0, len(idx), batch_size):
            order.append([exs[i] for i in idx[k:k + batch_size]])
    rng.shuffle(order)                       # mix buckets across the epoch
    for chunk in order:
        yield _stack(chunk, n_spins, kmax)


# ------------------------------------------------------------------ batched loss
def _batch_loss(params, model, batch, lam_e, lam_w, ws, n_spins):
    out = jax.vmap(lambda z, p: model.apply(params, z, p))(batch["z"], batch["pos"])
    e = jnp.mean((out["energy"] - batch["energy"]) ** 2)
    w = 0.0
    if lam_w > 0:
        for s in range(n_spins):
            pc, pr, pw = out["wannier"]["centers"][:, s], out["wannier"]["radii"][:, s], \
                out["wannier"]["presence"][:, s]
            tc, tr, tw = batch["centers"][:, s], batch["radii"][:, s], batch["mask"][:, s]
            ps = (ws * pr) ** 2 + 1e-6
            ts = (ws * tr) ** 2 + 1e-6
            w = w + jnp.mean(jax.vmap(cloud_l2)(pc, pw, ps, tc, tw, ts))
    return lam_e * e + lam_w * w


def _energy_mae_batched(model, params, examples, n_spins, kmax, batch_size):
    """Energy MAE over examples, batched (bucketed) so eval is cheap."""
    errs = []
    rng = np.random.default_rng(0)
    for batch in _batches(examples, batch_size, n_spins, kmax, rng):
        pred = jax.vmap(lambda z, p: model.apply(params, z, p)["energy"])(
            jnp.asarray(batch["z"]), jnp.asarray(batch["pos"]))
        errs.append(np.abs(np.asarray(pred) - batch["energy"]))
    return float(np.concatenate(errs).mean()) if errs else float("nan")


# ------------------------------------------------------------------ train
def train(cfg, examples, ckpt_path=None):
    tc = cfg["train"]
    n_spins = len(cfg["wannier"]["spins"])
    kmax = cfg["wannier"]["max_centers"]
    lam_e, lam_w, ws = cfg["loss"]["lambda_energy"], cfg["loss"]["lambda_wannier"], \
        cfg["wannier"]["width_scale"]

    train_ex, val_ex, _ = split(examples, tc["val_frac"], tc["test_frac"], tc["seed"])
    model = model_from_config(cfg)
    params = model.init(jax.random.PRNGKey(tc["seed"]),
                        jnp.asarray(train_ex[0]["z"]), jnp.asarray(train_ex[0]["pos"]))
    if ckpt_path and Path(ckpt_path).exists():       # resume after a wall-kill
        params = serialization.from_bytes(params, Path(ckpt_path).read_bytes())
        print(f"resumed from {ckpt_path}")

    opt = optax.adam(tc["lr"]); opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(_batch_loss)(
            params, model, batch, lam_e, lam_w, ws, n_spins)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(tc["seed"])
    ckpt_every = tc.get("ckpt_every", 10)
    for epoch in range(tc["epochs"]):
        losses = []
        for batch in _batches(train_ex, tc["batch_size"], n_spins, kmax, rng):
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
            params, opt_state, loss = step(params, opt_state, batch)
            losses.append(float(loss))
        if epoch % ckpt_every == 0 or epoch == tc["epochs"] - 1:
            mae = _energy_mae_batched(model, params, val_ex, n_spins, kmax, tc["batch_size"])
            print(f"epoch {epoch:4d}  train loss {np.mean(losses):.4f}  val E-MAE {mae:.4f} eV")
            if ckpt_path:
                save_params(ckpt_path, params)       # periodic: survive a wall-kill
    return params


def save_params(path, params):
    Path(path).write_bytes(serialization.to_bytes(params))


def load_params(path, template):
    return serialization.from_bytes(template, Path(path).read_bytes())
