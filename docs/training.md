# `training/` — the optimization loop (JAX/optax idioms)

Ties [model](model.md) + [losses](losses.md) together and drives the weights with gradient
descent. This is the doc to read for **how JAX training actually looks**. Syntax:
[jax_flax_primer](jax_flax_primer.md). Rationale: `../CLAUDE.md` §6, §10.

Files: `loss.py` (combine + convert), `train.py` (the loop + checkpointing), `metrics.py` (eval).

---

## 1. `loss.py` — combine the two objectives

```python
def molecule_loss(params, apply_fn, ex, lambda_e, lambda_w, width_scale):
    out = apply_fn(params, ex["z"], ex["pos"])
    e = energy_loss(out["energy"], ex["energy"])
    w = wannier_loss(out["wannier"], ex["wannier"], width_scale) if lambda_w > 0 else 0.0
    total = lambda_e * e + lambda_w * w
    return total, {"energy": e, "wannier": w}
```

- `total = λ_E · energy + λ_W · wannier` — the weighted sum from CLAUDE.md §6. The weights matter
  because energy (eV²) and the cloud loss live in **different units/scales**; `configs/*.yaml`
  holds them (`energy_only.yaml` sets `λ_W = 0`, which also *skips* the Wannier computation via
  the `if lambda_w > 0` guard).
- Returns `(scalar, aux_dict)` — the `has_aux=True` shape ([primer §6](jax_flax_primer.md)) so you
  can log the two components separately.
- `example_to_jax(ex)` converts a parsed NumPy example into `jnp` arrays at the boundary
  (`np` in [data](data.md) → `jnp` for anything differentiated, [primer §1](jax_flax_primer.md)),
  keeping the per-spin `wannier` list structure intact.

## 2. `train.py` — the loop, decoded

### Setup
```python
train_ex, val_ex, _ = split(examples, val_frac, test_frac, seed)   # from data/dataset.py
model  = model_from_config(cfg)
params = model.init(jax.random.PRNGKey(seed), train_j[0]["z"], train_j[0]["pos"])  # weights
opt = optax.adam(lr); opt_state = opt.init(params)                 # optimizer + its state
```
`init` needs one example just to fix shapes ([primer §5](jax_flax_primer.md)). `opt_state` holds
Adam's running moments as a pytree parallel to `params`.

### The gradient function (a jitted closure)
```python
@jax.jit
def vgrad(p, ex):
    return jax.value_and_grad(
        lambda pp: molecule_loss(pp, model.apply, ex, lam_e, lam_w, ws)[0])(p)
```
- The inner `lambda pp: molecule_loss(...)[0]` exposes a **pure scalar function of the params
  only** (everything else — `model.apply`, `ex`, the λ's — is captured from the closure). That's
  what `value_and_grad` needs. `[0]` drops the aux dict so the differentiated output is scalar.
- `@jax.jit` compiles it. **Recompile-per-shape caveat** ([primer §6](jax_flax_primer.md)): each
  distinct atom count `N` (and center counts) is a new shape → one compile each. QM9 has few
  distinct sizes, so this is cheap; it's the reason we didn't force everything to a padded shape.

### Per-example gradient averaging (no vmap)
```python
for ex in batch:
    _, g = vgrad(params, ex)
    grads = g if grads is None else _tree_add(grads, g)     # tree_map(a+b) over all weights
grads = jax.tree_util.tree_map(lambda x: x / len(batch), grads)   # average
updates, opt_state = opt.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```
- Molecules have different shapes, so we can't stack them into one batched array for `vmap`.
  Instead we compute each molecule's gradient and **average the gradient pytrees leafwise** with
  `tree_map` ([primer §3](jax_flax_primer.md)). Averaging gradients of a mean-loss ≡ the
  minibatch gradient — mathematically the same as batching, just slower.
- This is the deliberate `ponytail:` shortcut noted in `train.py`: fine for tiny QM9; if
  throughput ever matters, pad molecules to a common size and `vmap`/`scan` the batch.
- `opt.update` returns new `updates` **and** new `opt_state` (Adam is stateful — you must thread
  `opt_state` forward each step). `apply_updates` adds `updates` to `params` leafwise
  ([primer §7](jax_flax_primer.md)).

### Checkpointing
```python
save_params: Path.write_bytes(serialization.to_bytes(params))
load_params: serialization.from_bytes(template, blob)   # template from a fresh model.init
```
Loading needs a **template** pytree of the right structure — you get it by running `init` once,
then fill it from disk ([primer §8](jax_flax_primer.md)). `scripts/evaluate.py` and `main.py` do
exactly this: `init` for structure, then `load_params` to overwrite with trained values.

## 3. `metrics.py`
- `energy_mae` — mean absolute error in eV (report-friendly, unlike the squared training loss).
- `wannier_discrepancy` — mean per-molecule `wannier_loss`, i.e. the cloud L2 as an eval number.

Both loop over examples in plain Python (eval, not differentiated, so no need to jit/vmap).

---

## Failure modes & debugging
| Symptom | Cause | Where |
|---|---|---|
| training very slow to start | one XLA compile per distinct molecule size | expected; §2, [primer §6](jax_flax_primer.md) |
| loss dominated by one term | `λ_E`/`λ_W` mismatched to the unit scales | tune in `configs/*.yaml`; §1 |
| `NaN` after a few steps | lr too high, or a head without `softplus`/`sqrt` eps | lower lr; see [model](model.md) failure table |
| checkpoint won't load | `template` structure ≠ saved params (config changed dims) | rebuild template with the *same* config; §2 |
| Wannier loss ignored | using `energy_only.yaml` (`λ_W=0`) | intended ablation; §1 |

## Rebuild-by-hand order
1. `molecule_loss` (combine energy + wannier with λ's) → confirm it returns `(scalar, aux)`.
2. `model.init` + `optax.adam` setup.
3. The jitted `value_and_grad` closure over params.
4. Minibatch loop: per-example grad, `tree_map` average, `opt.update`, `apply_updates`.
5. `save_params`/`load_params`. Then metrics. Train **energy-only first** (§10 build order).
