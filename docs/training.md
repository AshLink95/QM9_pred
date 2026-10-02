# `training/` — the optimization loop (JAX/optax idioms)

Ties [model](model.md) + [losses](losses.md) together and drives the weights with gradient
descent. This is the doc to read for **how JAX training actually looks**. Syntax:
[jax_flax_primer](jax_flax_primer.md). Rationale: `../CLAUDE.md` §6, §10.

Files: `train.py` (batching, loss, loop, checkpointing), `loss.py` (example → JAX arrays for
eval), `metrics.py` (per-molecule eval metrics).

---

## 1. Why batched, and how molecules become batches

The first version trained **one molecule per step**. On 118k real molecules that failed twice:
it was slow (one Python dispatch per molecule), and it crashed with `LLVM compilation error:
Cannot allocate memory`. The cause was a **shape storm**: `jax.jit` compiles once per distinct
input shape ([primer §6](jax_flax_primer.md)), and the Wannier loss shape depends on each
molecule's true center count `K`, so every `(N, K_up, K_down)` combination compiled separately.
Thousands of compilations exhausted the compiler's memory.

The fix is to make shapes few and fixed:

```python
def _batches(examples, batch_size, n_spins, kmax, rng):
    buckets = defaultdict(list)
    for e in examples:
        buckets[len(e["z"])].append(e)        # group by atom count N
    ...                                        # shuffle, cut each bucket into batch_size chunks
    yield _stack(chunk, n_spins, kmax)
```

- **Bucket by atom count `N`.** Every molecule in a batch has the same `N`, so `z` stacks to
  `[B, N]` and `pos` to `[B, N, 3]` with no atom padding and no masks inside the model.
- **Pad Wannier targets to `Kmax`** (`_pad_wannier`): `centers[S, Kmax, 3]`,
  `radii[S, Kmax]`, and a `mask[S, Kmax]` that is 1 for real centers and 0 for padding. The
  mask becomes the Gaussian amplitude in the cloud loss, so padded centers contribute nothing
  ([losses](losses.md) §5). `Kmax` = `wannier.max_centers` (28 for this dataset).
- Distinct shapes are now one `(B, N)` per atom count, plus one for each bucket's last, smaller
  batch. That's a few dozen compilations in total, all paid in epoch 0 (which is why epoch 0
  takes ~30 min on the full dataset and later epochs ~6 s).

## 2. The batched loss

```python
def _batch_loss(params, model, batch, lam_e, lam_w, ws, n_spins):
    out = jax.vmap(lambda z, p: model.apply(params, z, p))(batch["z"], batch["pos"])
    e = jnp.mean((out["energy"] - batch["energy"]) ** 2)
    w = sum over spins of mean(jax.vmap(cloud_l2)(pred..., true..., mask as amplitude))
    return lam_e * e + lam_w * w, (e, w)
```

- `jax.vmap(...)` turns the single-molecule `model.apply` into a batched one: it maps over the
  leading `B` axis of `z`/`pos` while `params` is shared. Outputs gain a leading `B` axis
  (`energy[B]`, `centers[B, S, M, 3]`, …).
- The Wannier term calls `cloud_l2` from `losses/wannier_cloud.py` under `vmap`, once per spin
  (up compared only with up). Predicted amplitude = presence, true amplitude = the pad mask.
- `total = λ_E·E + λ_W·W` (CLAUDE.md §6). It returns `(total, (e, w))`: the second element is
  **aux** data, so the loop can log the two terms separately
  (`value_and_grad(..., has_aux=True)`, [primer §6](jax_flax_primer.md)).
- `energy_only.yaml` sets `λ_W = 0`, which skips the Wannier computation entirely.

## 3. Energy reference — why the loss was ~1e12

SIESTA total energies are ~−1000 to −3000 eV, and a freshly initialized network outputs ~0. On
the real data the energy MSE started around 1e12 and drowned out the Wannier term (~1e3). The
huge errors drove the weights up until training diverged to NaN at epoch 41.

```python
atom_ref, resid_std = _fit_atom_ref(train_ex, n_elements)   # least squares: E ≈ Σ_atoms ref[Z]
model = model_from_config(cfg, atom_ref=atom_ref)            # seeds the model's per-element offset
```

`_fit_atom_ref` builds a `[molecules, elements]` matrix of atom counts and solves
`counts @ ref ≈ E` with `np.linalg.lstsq`, using the **train** split only. The model adds
`Σ_i ref[Z_i]` to its energy ([model](model.md) §5), so the network only has to learn the
residual. Startup prints the fitted `ref` per element and the residual std. That std is roughly
the MAE of a model that learned nothing beyond composition, so it's the baseline to beat.

## 4. The step and the loop

```python
opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))

@jax.jit
def step(params, opt_state, batch):
    (loss, parts), grads = jax.value_and_grad(_batch_loss, has_aux=True)(params, model, batch, ...)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, parts, optax.tree.norm(grads)
```

- `optax.chain` runs the transforms in order: **clip** the whole gradient pytree to global norm
  ≤ `grad_clip` (config, default 1.0), then Adam. Clipping caps any single bad batch's effect.
- `opt_state` is threaded forward every step. Adam keeps running moments, and dropping the
  state resets them.
- `model` and the λ's are closed over. `jax.jit` sees only arrays as arguments, so it compiles
  once per batch shape (§1).

Each epoch iterates `_batches`, converts each batch with `jnp.asarray`, and calls `step`.

## 5. Checkpointing and the NaN guard

- **Every `ckpt_every` epochs** (config, 5) and at the end, it prints
  `epoch … train loss X (E … W …) val E-MAE … (Ns)` and saves params with
  `flax.serialization.to_bytes` ([primer §8](jax_flax_primer.md)), then prints
  `checkpoint saved -> … @ epoch N`.
- **Resume:** if `--out` already exists, it's loaded as the starting params. It's **ignored**
  (with a WARNING) if it contains NaN/inf or doesn't match the current architecture. An earlier
  run resumed from a NaN checkpoint and trained NaN for 300 epochs; this check prevents that.
- **NaN guard:** after every step it checks the loss. On the first non-finite value it prints
  `NON-FINITE loss at epoch E step S: E-loss … W-loss … |grad| … (N atoms …)`, reverts to the
  last finite params, saves those, and stops. The printed components show which term blew up.
- A checkpoint is only written if every param is finite, so a bad step never overwrites a good
  file.

Loading for inference (`main.py`, `scripts/evaluate.py`) builds a template with `model.init`,
then fills it with `load_params`. The trained `atom_ref` values come from the checkpoint.

## 6. Evaluation: `metrics.py`, `report.py`, `loss.py`

None of these are imported by the training loop.

- **`metrics.py:match_wannier`** scores one molecule and spin. Predicted slots with
  presence > threshold count as centers. They're matched one-to-one to the true centers with
  `scipy.optimize.linear_sum_assignment` (Hungarian: the pairing with the smallest total
  distance). It returns per-pair distance (Å) and radius error, plus both counts. Leftovers on
  either side are missed or extra centers. **`wannier_summary`** aggregates these into center
  MAE / RMSE / median / p95, radius MAE, and count accuracy.
  - Why matching is fine here but not as the loss (CLAUDE.md §6): scoring a finished
    prediction needs no gradient and no padding, so the classic reasons against matching don't
    apply. The training loss stays the Gaussian cloud.
- **`report.py`** drives `scripts/evaluate.py`:
  - `predict_all` is a batched forward pass bucketed by atom count. It returns outputs in input
    order, and padding each chunk to `batch_size` keeps it to one compiled shape per atom count.
  - `build_report` produces the per-molecule rows, per-center rows, and the summary.
    `write_outputs` writes the CSVs, `summary.txt`, and `accuracy.png`.
  - `load_fresh` parses new folders, reusing `data/dataset.py`'s helpers so molecule ids are
    kept.
  - `max_centers_from_checkpoint` reads the slot count from the checkpoint, so the model is
    always rebuilt with the shapes it was trained with.
- `energy_mae` (used by the smoke test) loops over molecules one at a time.
  `loss.py:example_to_jax` converts a parsed NumPy example to JAX arrays for it.
- Validation MAE during training uses the batched `_energy_mae_batched` in `train.py`.

---

## Failure modes & debugging
| Symptom | Cause | Where |
|---|---|---|
| `LLVM compilation error: Cannot allocate memory` | too many distinct shapes being compiled | §1: the bucketing/padding must stay |
| epoch 0 takes ~30 min, later epochs seconds | one-time compile per batch shape | expected; §1 |
| E-loss ~1e12, huge val E-MAE, then NaN | energy reference missing or removed | §3 |
| W-loss explodes within one epoch | coordinate update unbounded | [model](model.md) §2 stabilizer |
| `NON-FINITE loss …` line | divergence; the components tell which term | §5; lower `lr`, check λ balance |
| `WARNING: … doesn't match this model` | checkpoint from an older architecture | expected after model changes; starts fresh |
| only GPU 0 busy with 4 requested | no multi-GPU sharding implemented | request `--gres=gpu:1` |
| job hangs with 0 CPU, empty log | `uv run` inside the job (no internet on compute nodes) | use `scripts/train*.sh` |

## Rebuild-by-hand order
1. `_pad_wannier` + `_stack` + `_batches` (bucket by `N`, pad `K`, mask). Check the batch shapes.
2. `_batch_loss` with `vmap` and aux `(e, w)`.
3. `_fit_atom_ref`, and seed the model's per-element offset with it.
4. Optimizer chain (clip → Adam), jitted `step`.
5. Epoch loop with the NaN guard, periodic finite-only checkpointing, and resume.
6. Train `energy_only.yaml` first. Once val E-MAE drops below the residual std printed at
   startup, move to the joint loss.
