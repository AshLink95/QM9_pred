"""Training loop (CLAUDE.md §10): precomputed batches, CPU/GPU pipeline, multi-GPU, validation
with Wannier metrics, best/last checkpoints with exact resume, LR schedule, early stopping.

How one epoch runs (docs/training.md walks through it):

  * Molecules are bucketed by atom count ONCE into stacked arrays (`stack_buckets`). Each epoch
    cuts every bucket into fixed-size batches; a bucket's last batch is topped up with random
    molecules from the same bucket, so every batch has exactly `batch_size` molecules: one
    compiled shape per atom count, and batches always split evenly across GPUs.
  * A producer thread slices the arrays and moves upcoming batches to the devices.
  * Per step: `fwd` (no gradient) predicts Wannier centers for the NEXT batch while `step` trains
    on the current one; meanwhile the CPU pairs predicted and true centers (Hungarian, thread
    pool). The pairing used by a step was thus computed with parameters one update old (a
    "lagged assignment"; pairings rarely change in one step). No host callbacks inside compiled
    code, so compiled programs can be cached on disk and shared between jobs.
  * Multi-GPU: parameters are replicated, batches sharded along the batch axis; XLA inserts the
    gradient all-reduce. Identical code runs on 1 GPU or a CPU.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import serialization
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from data.dataset import split
from data.electrons import centers_per_spin
from losses.wannier_match import assign, wannier_match_loss
from model.heads import select_centers
from model.model import model_from_config
from training.metrics import match_wannier


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def split_seed(tc: dict) -> int:
    """Seed of the train/val/test split. Separate from `seed` (init + batch order), so runs with
    different seeds share one identical test set. Falls back to `seed` for old configs."""
    return tc.get("split_seed", tc["seed"])


# ------------------------------------------------------------------ data: stacked once
def _pad_wannier(ex, n_spins, kmax):
    """example -> centers[S,Kmax,3], radii[S,Kmax] (zero-padded beyond the real centers)."""
    c = np.zeros((n_spins, kmax, 3), np.float32); r = np.zeros((n_spins, kmax), np.float32)
    for s in range(n_spins):
        w = ex["wannier"][s]; k = len(w["radii"])
        c[s, :k] = w["centers"]; r[s, :k] = w["radii"]
    return c, r


def stack_buckets(examples, n_spins, kmax):
    """Group molecules by atom count and stack each group into arrays (done once).
    Returns {n_atoms: {z, pos, centers, radii, k, index}}; `index` = position in
    `examples`."""
    groups = defaultdict(list)
    for i, e in enumerate(examples):
        groups[len(e["z"])].append(i)
    buckets = {}
    for n, idx in sorted(groups.items()):
        exs = [examples[i] for i in idx]
        cr = [_pad_wannier(e, n_spins, kmax) for e in exs]
        buckets[n] = {
            "z": np.stack([e["z"] for e in exs]).astype(np.int32),
            "pos": np.stack([e["pos"] for e in exs]).astype(np.float32),
            "centers": np.stack([c for c, _ in cr]),
            "radii": np.stack([r for _, r in cr]),
            "k": np.stack([centers_per_spin(e["z"], n_spins) for e in exs]),
            "index": np.array(idx),
        }
    return buckets


def epoch_batches(buckets, batch_size, rng):
    """One epoch's batches as (n_atoms, indices[batch_size]). Each bucket is shuffled and cut;
    its last chunk is topped up with random molecules from the same bucket. Order is shuffled
    across buckets. Same rng state => same batches (exact resume)."""
    out = []
    for n in sorted(buckets):
        m = len(buckets[n]["k"])
        order = rng.permutation(m)
        for s in range(0, m, batch_size):
            idx = order[s:s + batch_size]
            if len(idx) < batch_size:
                idx = np.concatenate([idx, rng.choice(m, batch_size - len(idx), replace=True)])
            out.append((n, idx))
    return [out[i] for i in rng.permutation(len(out))]


def eval_batches(buckets, batch_size):
    """Every molecule exactly once, as (n_atoms, indices[batch_size], n_real); the last chunk of
    a bucket is padded with repeats that callers discard."""
    for n in sorted(buckets):
        m = len(buckets[n]["k"])
        for s in range(0, m, batch_size):
            idx = np.arange(s, min(s + batch_size, m))
            real = len(idx)
            yield n, np.concatenate([idx, np.full(batch_size - real, idx[0])]), real


def steps_per_epoch(buckets, batch_size):
    return sum(-(-len(b["k"]) // batch_size) for b in buckets.values())


def _gather(buckets, n, idx):
    return {k: v[idx] for k, v in buckets[n].items() if k != "index"}


def _prefetch(items, make, depth=2):
    """Run `make(item)` in a background thread, `depth` items ahead of the consumer. Closing the
    generator (break / exception) stops the producer, so no thread is left blocked."""
    q, done, stop = queue.Queue(maxsize=depth), object(), threading.Event()

    def produce():
        try:
            for it in items:
                x = make(it)
                while not stop.is_set():
                    try:
                        q.put(x, timeout=0.1); break
                    except queue.Full:
                        pass
                if stop.is_set():
                    return
            q.put(done)
        except BaseException as err:          # surface producer errors in the consumer
            q.put(err)

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            x = q.get()
            if x is done:
                return
            if isinstance(x, BaseException):
                raise x
            yield x
    finally:
        stop.set()


# ------------------------------------------------------------------ loss
def _batch_loss(params, model, batch, perm, wkw):
    """The matched Wannier loss, per spin — the only training objective (CLAUDE.md §1, §6).
    `perm` comes from the host pairing; `wkw` holds the loss settings (python values)."""
    out = jax.vmap(lambda z, p: model.apply(params, z, p))(batch["z"], batch["pos"])
    o = out["wannier"]
    return wannier_match_loss(o["centers"], o["radii"], o["logits"], batch["centers"],
                              batch["radii"], batch["k"], perm, **wkw)


def _finite(tree):
    """True iff every array leaf is free of NaN/inf."""
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jax.tree_util.tree_leaves(tree))


def _fit_radius(examples):
    """Mean and std of all true Wannier radii in the TRAIN split (Å): the mean initializes the
    radius output, the std normalizes the radius loss."""
    r = np.concatenate([w["radii"] for e in examples for w in e["wannier"]])
    return float(r.mean()), float(max(r.std(), 1e-3))


def make_lr(tc, spe):
    """Learning rate: constant, or warmup + cosine decay over the planned epochs. Returns the
    optax learning rate (float or schedule) and a step -> value function for logging."""
    if tc.get("lr_schedule", "constant") != "cosine":
        return tc["lr"], (lambda step: tc["lr"])
    total = max(2, tc["epochs"] * spe)
    warm = int(min(max(1, tc.get("warmup_epochs", 5) * spe), total - 1))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=tc["lr"], warmup_steps=warm, decay_steps=total,
        end_value=tc["lr"] * tc.get("lr_end_frac", 0.01))
    return sched, (lambda step: float(sched(step)))


# ------------------------------------------------------------------ validation
def _validate(fwd, params, buckets, examples, put, spins, wkw, batch_size, pool):
    """Predict every validation molecule once, select k centers per spin exactly as at
    prediction time (top logits + NMS), pair them with the truth on the host, and compute:

        objective = center MAE + radius_weight · radius MAE / r_std

    It doesn't depend on the loss forms (l1/mse), so runs that differ only in those are compared
    on the same scale (r_std comes from the shared train split). Lower is better."""
    S = len(spins)
    n_mol, jobs = 0, []
    for n, idx, real in eval_batches(buckets, batch_size):
        b = _gather(buckets, n, idx)
        out = jax.device_get(fwd(params, put(b["z"]), put(b["pos"])))
        sel = select_centers(out["wannier"], b["k"], wkw["nms_radius"])
        for j in range(real):
            ex = examples[buckets[n]["index"][idx[j]]]
            n_mol += 1
            for s in range(S):
                w = ex["wannier"][s]
                jobs.append((s, sel["centers"][j, s], sel["radii"][j, s],
                             int(b["k"][j, s]), w["centers"], w["radii"]))
    matches = list(pool.map(lambda a: (a[0], match_wannier(*a[1:])), jobs))
    d = {s: np.array([p[2] for ss, m in matches if ss == s for p in m["pairs"]]) for s in range(S)}
    rerr = np.array([p[3] for _, m in matches for p in m["pairs"]])
    alld = np.concatenate([d[s] for s in range(S)])
    c_mae = float(np.mean(alld)) if len(alld) else 0.0
    r_mae = float(np.mean(np.abs(rerr))) if len(rerr) else 0.0
    return {
        "objective": c_mae + wkw["radius_weight"] * r_mae / wkw["r_std"],
        "center_mae": {spins[s]: float(d[s].mean()) if len(d[s]) else float("nan")
                       for s in range(S)},
        "radius_mae": float(np.mean(np.abs(rerr))) if len(rerr) else float("nan"),
        "count_accuracy": float(np.mean([m["n_pred"] == m["n_true"] for _, m in matches])),
        "n_molecules": n_mol,
    }


# ------------------------------------------------------------------ checkpoints
def _paths(ckpt_path):
    out = Path(ckpt_path)
    return out, out.with_name(out.stem + ".last.msgpack"), out.with_name(out.stem + ".best.json")


def save_params(path, params):
    Path(path).write_bytes(serialization.to_bytes(params))


def load_params(path, template):
    """Load parameters into `template`'s structure. Modules the checkpoint has but the model
    doesn't (e.g. `EnergyHead_0` / `atom_ref` in checkpoints from before the energy was
    dropped) are ignored, so older Wannier models still load; missing modules still error."""
    raw = serialization.msgpack_restore(Path(path).read_bytes())
    tp, rp = template.get("params", {}), raw.get("params", {})
    if isinstance(rp, dict) and isinstance(tp, dict):
        raw = {**raw, "params": {k: v for k, v in rp.items() if k in tp}}
    return serialization.from_state_dict(template, raw)


def _atomic_write(path, data: bytes):
    """Write via a temp file + rename, so a crash mid-write never leaves a truncated file."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ------------------------------------------------------------------ train
def train(cfg, examples, ckpt_path=None, should_stop=None):
    """Train and return the BEST parameters (by validation objective). See train_with_status."""
    return train_with_status(cfg, examples, ckpt_path, should_stop)[0]


def train_with_status(cfg, examples, ckpt_path=None, should_stop=None):
    """Returns (best_params, status) with status in {"finished", "early_stopped", "stopped",
    "diverged"}. Files next to `ckpt_path` (e.g. params.msgpack):
      params.msgpack       best parameters (what evaluate/main load)
      params.last.msgpack  full training state for resume (params, optimizer, epoch, best)
      params.best.json     validation metrics of the best epoch
    `should_stop()` is polled between steps; when True the state is saved and status "stopped"
    is returned (used for graceful preemption)."""
    tc, wc = cfg["train"], cfg["wannier"]
    spins = wc["spins"]; n_spins = len(spins); kmax = wc["max_centers"]
    B, ckpt_every, patience = tc["batch_size"], tc.get("ckpt_every", 10), tc.get("patience", 0)
    should_stop = should_stop or (lambda: False)

    train_ex, val_ex, _ = split(examples, tc["val_frac"], tc["test_frac"], split_seed(tc))
    # the loss trusts k = Σ valence/2 per spin; drop (and report) any molecule whose parsed
    # .wout count disagrees, rather than pairing predictions with zero-padding
    def _count_ok(e):
        k = centers_per_spin(e["z"], n_spins)
        return all(len(e["wannier"][s]["radii"]) == k[s] for s in range(n_spins))
    n_before = len(train_ex)
    train_ex = [e for e in train_ex if _count_ok(e)]
    if len(train_ex) < n_before:
        print(f"WARNING: dropped {n_before - len(train_ex)} training molecules whose Wannier "
              f"count != valence/2", flush=True)

    # ---- devices: data parallel over every visible device
    devices = jax.devices()
    if B % len(devices):
        raise ValueError(f"train.batch_size={B} must be divisible by the {len(devices)} visible "
                         "devices (each device gets an equal slice of every batch)")
    # Auto axis type = classic automatic sharding propagation (XLA decides intermediate
    # shardings); this JAX defaults to Explicit, which would require annotating every op.
    mesh = jax.make_mesh((len(devices),), ("data",), axis_types=(AxisType.Auto,))
    data_sh, repl = NamedSharding(mesh, P("data")), NamedSharding(mesh, P())
    put = lambda x: jax.device_put(x, data_sh)
    print(f"devices: {devices}", flush=True)
    print(f"data-parallel over {len(devices)} device(s), {B // len(devices)} molecules per "
          f"device per step", flush=True)

    train_b = stack_buckets(train_ex, n_spins, kmax)
    val_b = stack_buckets(val_ex, n_spins, kmax) if val_ex else {}
    spe = steps_per_epoch(train_b, B)
    per_atom = wc.get("singles_per_atom", 2) + 2 * wc.get("pairs_per_atom", 1)
    for n, b in {**train_b, **val_b}.items():           # the head must offer >= k candidates
        if b["k"].max() > n * per_atom:
            raise ValueError(f"a {n}-atom molecule has {b['k'].max()} centers per spin but the "
                             f"head offers {n * per_atom} candidates; raise "
                             "wannier.singles_per_atom / pairs_per_atom")
    print(f"train {len(train_ex)}  val {len(val_ex)}  max_centers {kmax}  batch_size {B}  "
          f"steps/epoch {spe}", flush=True)

    r_mean, r_std = _fit_radius(train_ex)
    print(f"Wannier radius: mean {r_mean:.4f} A, std {r_std:.4f} A  (radius output starts at "
          f"the mean)", flush=True)
    layout = (wc.get("singles_per_atom", 2), wc.get("pairs_per_atom", 1))
    wkw = {"r_std": r_std, "guide_sigma": wc["guide_sigma"],
           "radius_weight": wc["radius_weight"], "guide_weight": wc["guide_weight"],
           "select_weight": wc.get("select_weight", 1.0),
           "position_loss": wc.get("position_loss", "mse"),
           "nms_radius": wc.get("nms_radius", 0.0), "layout": layout}
    ema_decay = float(tc.get("ema_decay", 0.0))
    print(f"loss: positions {wkw['position_loss']}, NMS radius "
          f"{wkw['nms_radius']} A, EMA decay {ema_decay}", flush=True)

    model = model_from_config(cfg, radius_init=r_mean)
    params = model.init(jax.random.PRNGKey(tc["seed"]),
                        jnp.asarray(train_ex[0]["z"]), jnp.asarray(train_ex[0]["pos"]))
    lr, lr_at = make_lr(tc, spe)
    opt = optax.chain(optax.clip_by_global_norm(tc.get("grad_clip", 1.0)), optax.adam(lr))
    opt_state = opt.init(params)
    print(f"learning rate: {tc.get('lr_schedule', 'constant')} (peak {tc['lr']}, "
          f"{tc['epochs']} epochs)", flush=True)

    # ---- resume from the last full state, if compatible
    state = {"params": params, "ema": params, "opt_state": opt_state, "epoch": np.int64(0),
             "best_obj": np.float64(np.inf), "best_epoch": np.int64(-1),
             "evals_since_best": np.int64(0)}
    best_params = params
    if ckpt_path:
        out_p, last_p, best_p = _paths(ckpt_path)
        if last_p.exists():
            try:
                loaded = serialization.from_bytes(state, last_p.read_bytes())
            except Exception as err:                   # older architecture / other config
                loaded = None
                print(f"WARNING: {last_p} doesn't match this model ({err}) — starting fresh",
                      flush=True)
            if loaded is not None and _finite(loaded["params"]):
                state = loaded
                print(f"resumed from {last_p} at epoch {int(state['epoch'])} (best "
                      f"{float(state['best_obj']):.4f} @ epoch {int(state['best_epoch'])})",
                      flush=True)
                if out_p.exists():
                    best_params = load_params(out_p, params)
            elif loaded is not None:
                print(f"WARNING: {last_p} contains NaN/inf — starting fresh", flush=True)
    params = jax.device_put(state["params"], repl)
    ema = jax.device_put(state["ema"], repl)           # what is validated, saved and returned
    opt_state = jax.device_put(state["opt_state"], repl)
    if int(state["best_epoch"]) < 0:
        best_params = ema

    def save_state(next_epoch):
        if not ckpt_path:
            return
        st = {**state, "params": params, "ema": ema, "opt_state": opt_state,
              "epoch": np.int64(next_epoch)}
        _atomic_write(_paths(ckpt_path)[1], serialization.to_bytes(jax.device_get(st)))

    # ---- compiled functions (no host callbacks inside: cacheable, shardable)
    fwd = jax.jit(lambda prm, z, pos: jax.vmap(lambda zz, pp: model.apply(prm, zz, pp))(z, pos),
                  out_shardings=data_sh)

    def _step(prm, ema_, ost, batch, perm):
        (loss, aux), grads = jax.value_and_grad(_batch_loss, has_aux=True)(
            prm, model, batch, perm, wkw)
        updates, ost = opt.update(grads, ost, prm)
        prm = optax.apply_updates(prm, updates)
        # exponential moving average of the weights (decay 0 -> the weights themselves)
        ema_ = jax.tree_util.tree_map(lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
                                      ema_, prm)
        return prm, ema_, ost, loss, aux
    step = jax.jit(_step, out_shardings=repl)

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    status = "finished"

    def make_batch(item):
        n, idx = item
        host = _gather(train_b, n, idx)
        dev = {k: put(v) for k, v in host.items()}
        return host, dev

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        def pairing(host, out_centers):
            return put(assign(np.asarray(out_centers), host["centers"], host["k"], pool,
                              layout=layout))

        epoch = int(state["epoch"])
        while epoch < tc["epochs"]:
            t0 = time.time()
            start_params, start_ema, start_opt = params, ema, opt_state
            rng = np.random.default_rng([tc["seed"], epoch])
            batches = _prefetch(epoch_batches(train_b, B, rng), make_batch)
            records, stopped = [], False
            cur = next(batches)
            perm = pairing(cur[0], fwd(params, cur[1]["z"], cur[1]["pos"])["wannier"]["centers"])
            for nxt in _chain(batches):
                if nxt is not None:                 # predict NEXT batch with current params...
                    nxt_out = fwd(params, nxt[1]["z"], nxt[1]["pos"])
                params, ema, opt_state, loss, aux = step(params, ema, opt_state, cur[1],
                                                         perm)                       # ...train
                records.append((loss, aux))
                if should_stop():
                    stopped = True
                    break
                if nxt is None:
                    break
                # CPU pairs the next batch while the GPU runs the step above
                perm = pairing(nxt[0], nxt_out["wannier"]["centers"])
                cur = nxt
            batches.close()

            vals = jax.device_get(records)           # one host sync per epoch
            losses = np.array([float(l) for l, _ in vals])
            if not np.all(np.isfinite(losses)):
                bad = int(np.argmax(~np.isfinite(losses)))
                a = vals[bad][1]
                print(f"NON-FINITE loss at epoch {epoch} step {bad}: pos {float(a['pos']):.3e}  "
                      f"radius {float(a['radius']):.3e} — reverting to the epoch-start state and "
                      f"stopping", flush=True)
                params, ema, opt_state = start_params, start_ema, start_opt
                save_state(epoch)
                status = "diverged"
                break
            if stopped:
                save_state(epoch)                    # resume restarts this epoch
                status = "stopped"
                print(f"stop requested — state saved at epoch {epoch}", flush=True)
                break

            last_epoch = epoch == tc["epochs"] - 1
            if epoch % ckpt_every == 0 or last_epoch:
                mean = lambda key: float(np.mean([float(a[key]) for _, a in vals]))
                line = (f"epoch {epoch:4d}  lr {lr_at((epoch + 1) * spe):.2e}  train "
                        f"{losses.mean():.4f} (pos {mean('pos'):.4f}, radius {mean('radius'):.4f}, select "
                        f"{mean('select'):.4f}, guide {mean('guide'):.4f})")
                if val_b:
                    v = _validate(fwd, ema, val_b, val_ex, put, spins, wkw, B, pool)
                    centers = "/".join(f"{x:.4f}" for x in v["center_mae"].values())
                    line += (f" | val obj {v['objective']:.4f}  "
                             f"center {'/'.join(spins)} {centers} A  radius "
                             f"{v['radius_mae']:.4f} A")
                    if v["objective"] < float(state["best_obj"]):
                        state.update(best_obj=np.float64(v["objective"]),
                                     best_epoch=np.int64(epoch), evals_since_best=np.int64(0))
                        best_params = ema
                        line += "  * new best"
                        if ckpt_path:
                            out_p, _, best_p = _paths(ckpt_path)
                            _atomic_write(out_p, serialization.to_bytes(jax.device_get(ema)))
                            best_p.write_text(json.dumps({"epoch": epoch, **v}, indent=2))
                    else:
                        state["evals_since_best"] = np.int64(int(state["evals_since_best"]) + 1)
                print(line + f"  ({time.time() - t0:.1f}s)", flush=True)
                save_state(epoch + 1)
                if ckpt_path:
                    print(f"checkpoint saved -> {_paths(ckpt_path)[1]} @ epoch {epoch}",
                          flush=True)
                    out_p = Path(ckpt_path)          # small progress file, read by sweep status
                    out_p.with_name(out_p.stem + ".progress.json").write_text(json.dumps({
                        "epoch": epoch, "epochs": tc["epochs"],
                        "best_epoch": int(state["best_epoch"]),
                        "best_objective": float(state["best_obj"])}))
                if patience and int(state["evals_since_best"]) >= patience:
                    print(f"early stop: no improvement for {patience} evaluations (best @ epoch "
                          f"{int(state['best_epoch'])})", flush=True)
                    status = "early_stopped"
                    break
            epoch += 1

    if not val_b:                                    # nothing to select on: keep the last
        best_params = ema
        if ckpt_path:
            _atomic_write(_paths(ckpt_path)[0], serialization.to_bytes(jax.device_get(ema)))
    return jax.device_get(best_params), status


def _chain(gen):
    """Yield every item of `gen`, then a final None (marks the last step of the epoch)."""
    yield from gen
    yield None
