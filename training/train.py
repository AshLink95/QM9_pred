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

import time
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
    return lam_e * e + lam_w * w, (e, w)     # aux: components, for logging / NaN diagnosis


def _finite(tree):
    """True iff every array leaf is free of NaN/inf."""
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(tree))


def _fit_atom_ref(examples, n_elements):
    """Least-squares per-element reference energies: E_mol ≈ sum_atoms ref[Z].
    Returns (ref tuple of length n_elements, residual std in eV). Deterministic statistic of
    the TRAIN targets — used only to initialize the model's per-element energy offset."""
    counts = np.zeros((len(examples), n_elements))
    for i, e in enumerate(examples):
        np.add.at(counts[i], e["z"], 1)
    energies = np.array([e["energy"] for e in examples])
    present = counts.any(axis=0)
    ref = np.zeros(n_elements)
    ref[present] = np.linalg.lstsq(counts[:, present], energies, rcond=None)[0]
    return tuple(float(r) for r in ref), float(np.std(energies - counts @ ref))


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
    # startup diagnostics (flushed): if these don't appear, the hang is env/load, not training
    print(f"devices: {jax.devices()}", flush=True)
    print(f"train {len(train_ex)}  val {len(val_ex)}  max_centers {kmax}  "
          f"batch_size {tc['batch_size']}", flush=True)

    atom_ref, resid_std = _fit_atom_ref(train_ex, cfg["model"]["n_elements"])
    print("atom_ref (eV): " + "  ".join(f"Z{z}={r:.3f}" for z, r in enumerate(atom_ref) if r),
          flush=True)
    print(f"energy residual std after ref: {resid_std:.4f} eV  (what the network must learn)",
          flush=True)

    model = model_from_config(cfg, atom_ref=atom_ref)
    params = model.init(jax.random.PRNGKey(tc["seed"]),
                        jnp.asarray(train_ex[0]["z"]), jnp.asarray(train_ex[0]["pos"]))
    if ckpt_path and Path(ckpt_path).exists():       # resume after a wall-kill
        try:
            loaded = serialization.from_bytes(params, Path(ckpt_path).read_bytes())
        except Exception as err:                       # checkpoint from an older architecture
            loaded = None
            print(f"WARNING: {ckpt_path} doesn't match this model ({err}) — starting fresh",
                  flush=True)
        if loaded is not None and _finite(loaded):
            params = loaded
            print(f"resumed from {ckpt_path}", flush=True)
        elif loaded is not None:                       # never resume from a diverged run
            print(f"WARNING: {ckpt_path} contains NaN/inf — ignoring it, starting fresh",
                  flush=True)

    opt = optax.chain(optax.clip_by_global_norm(tc.get("grad_clip", 1.0)),  # safety net
                      optax.adam(tc["lr"]))
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        (loss, parts), grads = jax.value_and_grad(_batch_loss, has_aux=True)(
            params, model, batch, lam_e, lam_w, ws, n_spins)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, parts, \
            optax.tree.norm(grads)

    rng = np.random.default_rng(tc["seed"])
    ckpt_every = tc.get("ckpt_every", 10)
    diverged = False
    for epoch in range(tc["epochs"]):
        t0 = time.time()
        losses, e_parts, w_parts = [], [], []
        for i, batch in enumerate(_batches(train_ex, tc["batch_size"], n_spins, kmax, rng)):
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
            prev = params
            params, opt_state, loss, (e, w), gnorm = step(params, opt_state, batch)
            if not np.isfinite(float(loss)):
                # stop at the FIRST bad step: keep the last finite params, report which term broke
                print(f"NON-FINITE loss at epoch {epoch} step {i}: E-loss {float(e):.3e}  "
                      f"W-loss {float(w):.3e}  |grad| {float(gnorm):.3e}  "
                      f"(N atoms {batch['z'].shape[1]}) — stopping, keeping last finite params",
                      flush=True)
                params, diverged = prev, True
                break
            losses.append(float(loss)); e_parts.append(float(e)); w_parts.append(float(w))
        last = diverged or epoch == tc["epochs"] - 1
        if losses and (epoch % ckpt_every == 0 or last):
            mae = _energy_mae_batched(model, params, val_ex, n_spins, kmax, tc["batch_size"])
            print(f"epoch {epoch:4d}  train loss {np.mean(losses):.4f}  "
                  f"(E {np.mean(e_parts):.4f}  W {np.mean(w_parts):.4f})  "
                  f"val E-MAE {mae:.4f} eV  ({time.time() - t0:.1f}s)", flush=True)
        if ckpt_path and (epoch % ckpt_every == 0 or last) and _finite(params):
            save_params(ckpt_path, params)           # periodic: survive a wall-kill
            print(f"checkpoint saved -> {ckpt_path} @ epoch {epoch}", flush=True)
        if diverged:
            break
    return params


def save_params(path, params):
    Path(path).write_bytes(serialization.to_bytes(params))


def load_params(path, template):
    return serialization.from_bytes(template, Path(path).read_bytes())
