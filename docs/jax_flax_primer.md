# JAX / Flax primer — the syntax this repo actually uses

Read this once. Every other doc links here instead of re-explaining `jit`, pytrees, or
`init`/`apply`. Scope is deliberately narrow: only the constructs that appear in this codebase,
explained for someone who knows Python and NumPy but is new to JAX.

Related docs: [data](data.md) · [model](model.md) · [losses](losses.md) ·
[training](training.md) · [symmetry](symmetry.md) · [index](README.md).

---

## 1. `jnp` vs `numpy`, and why JAX is "functional"

```python
import jax.numpy as jnp   # the model / loss math
import numpy as np        # the data pipeline (parsing, plotting)
```

`jax.numpy` mirrors the NumPy API (`jnp.sum`, `jnp.exp`, `jnp.einsum`, broadcasting …) but with
three differences that matter here:

1. **Arrays are immutable.** There is no `a[0] = 5`. You build new arrays instead. The
   "in-place" style is `a = a.at[0].set(5)` (used once, in `symmetry/transforms.py` you'll see
   plain NumPy because that's data-side; model code stays pure).
2. **Functions must be pure** to be transformed by `jit`/`grad`/`vmap`: output depends only on
   inputs, no side effects, no hidden global state. That's why parameters are passed *in*
   explicitly (see `apply` below) instead of living inside the object like PyTorch `self.weight`.
3. **Randomness is explicit.** No global seed. You pass a key (next section).

Rule of thumb in this repo: **`np` for parsing/plotting, `jnp` for anything the optimizer will
differentiate.** `jnp.asarray(x)` converts at the boundary (see `training/loss.py:example_to_jax`).

## 2. Explicit randomness — `jax.random.PRNGKey`

```python
key = jax.random.PRNGKey(0)      # a seed, but as an explicit value you thread through
params = model.init(key, ...)    # init consumes the key to sample initial weights
```

A `PRNGKey` is just a small array standing in for "the random state." Because functions are
pure, randomness can't hide in a global — you hand the key to whatever needs it. Same key →
same weights, always (this is why the equivariance tests are reproducible).

## 3. Pytrees and `tree_map` — the shape of "all the parameters"

A **pytree** is any nested structure of dicts/lists/tuples with array leaves. Flax stores every
weight matrix of the model in one big nested dict — that dict *is* a pytree. You rarely index
into it; instead you apply an operation to **every leaf at once**:

```python
jax.tree_util.tree_map(lambda x: x / n, grads)          # divide every gradient array by n
jax.tree_util.tree_map(lambda a, b: a + b, gA, gB)      # add two gradient trees leafwise
```

`training/train.py` uses exactly these two to average per-example gradients. Mental model:
`tree_map` is `map()` over the leaves of a nested dict, preserving the structure.

## 4. Flax modules — `nn.Module`, `@nn.compact`, and the layer primitives

Flax describes a network as a `Module` (a dataclass) whose `__call__` defines the forward pass.

```python
import flax.linen as nn

class EnergyHead(nn.Module):
    hidden_dim: int                     # hyperparameters = dataclass fields

    @nn.compact                         # lets you declare layers inline in __call__
    def __call__(self, h):
        return jnp.sum(nn.Dense(1)(nn.silu(nn.Dense(self.hidden_dim)(h))))
```

- **Fields** (`hidden_dim: int`) are static configuration, set when you construct the module.
- **`@nn.compact`** means "I'll create sub-layers (their weights) *inline* the first time
  `__call__` runs," instead of pre-declaring them in a `setup()` method. Every module in this
  repo uses the compact style. The *first* call (inside `init`) creates the weights; later calls
  (inside `apply`) reuse them.
- Primitives used here:
  - `nn.Dense(features)` — a learned affine map `xW + b`; `features` = output width.
  - `nn.Embed(num, dim)` — a lookup table mapping integer `Z` → a `dim`-vector (see
    [model](model.md), `ElementEmbedding`).
  - `nn.silu` — the SiLU/swish activation `x·sigmoid(x)`; used as a plain function.
  - `nn.softplus`, `nn.sigmoid` — smooth maps to `(0,∞)` and `(0,1)`, used to keep radii
    positive and presence in a probability range.
  - `nn.Sequential([layer, fn, layer, ...])` — chains callables; this repo's `_mlp` helper
    builds these (see [model](model.md)).

### Naming (`name="e0"`)
`nn.Dense(..., name="e0")` fixes the key that layer's weights get in the parameter pytree. It
keeps the tree readable and stable, which matters for checkpoint load/save
([training](training.md)).

## 5. `init` vs `apply` — where the weights live

Because the module holds *no* weights itself, there are two calls:

```python
params = model.init(key, z, pos)      # RUN once: build & return the parameter pytree
out    = model.apply(params, z, pos)  # RUN many times: forward pass with those params
```

- `init(key, *example_inputs)` traces `__call__` once on example inputs, allocates every weight,
  and returns them as a pytree. It needs example inputs only to learn the shapes.
- `apply(params, *inputs)` is the pure forward function: same inputs + same params → same
  output. The optimizer produces new `params`; you keep calling `apply` with the latest.

This split is the whole reason JAX can `grad` the network: `apply` is a plain function of
`(params, inputs)`, so `jax.grad(...)(params)` is well-defined.

## 6. The transforms — `jit`, `grad`, `value_and_grad`, `vmap`

- **`jax.jit(f)`** compiles `f` (via XLA) on first call for a given input *shape/dtype*, then
  reuses the compiled version. Huge speedups. Caveat used in [training](training.md): a **new
  shape triggers a recompile** — QM9 molecules have different atom counts `N`, so you get one
  compile per distinct `N`. Fine for a small dataset.
- **`jax.grad(f)`** returns a function computing `∂f/∂(first arg)`. `f` must return a scalar.
- **`jax.value_and_grad(f)`** returns `(f(x), grad)` in one pass — you usually want the loss
  value *and* its gradient. With `has_aux=True`, `f` returns `(scalar, aux)` and you get
  `((scalar, aux), grad)` (the loss code returns `(total, {"energy":..., "wannier":...})`).
- **`jax.vmap(f)`** auto-vectorizes `f` over a new batch axis. Not used in training here (we loop
  over molecules because their shapes differ) but it's the standard way to batch equal-shaped
  inputs, and it's how you'd batch if you padded molecules to a common size.

## 7. `optax` — the optimizer as three pure functions

Optax optimizers are also stateless objects + explicit state:

```python
opt = optax.adam(lr)
opt_state = opt.init(params)                             # optimizer's own state (moments)
updates, opt_state = opt.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)           # params + updates, leafwise
```

`updates` and `params` are pytrees of the same shape; `apply_updates` is a `tree_map` add under
the hood. See the loop in [training](training.md).

## 8. `flax.serialization` — saving weights

```python
from flax import serialization
serialization.to_bytes(params)                # pytree -> bytes (msgpack)
serialization.from_bytes(template, blob)      # bytes -> pytree, using `template` for structure
```

Loading needs a **template** pytree of the right structure (you get it by running `init` once),
then fills it from the bytes. See `training/train.py:save_params` / `load_params`.

## 9. Broadcasting trick you'll see everywhere: `[:, None, :]`

`None` in an index inserts a length-1 axis (same as `np.newaxis`). It's how we form all-pairs
tensors without loops:

```python
x[:, None, :] - x[None, :, :]        # x is [N,3]  ->  result [N,N,3]
```

Row `i`, column `j` of the result is `x[i] - x[j]`. This one line builds every pairwise
difference at once; the same pattern makes the `[N,N]` distance matrix and the `[A,B]` overlap
matrix in [losses](losses.md). Read `[:, None, :]` as "vary this index down the rows,"
`[None, :, :]` as "vary it across the columns."

---

### Where each idea is used
| Idea | First appears in |
|---|---|
| `nn.Module` / `@nn.compact` / `nn.Embed` | [model](model.md) — `ElementEmbedding` |
| pairwise `[:,None,:]` broadcasting | [model](model.md) — `EGNNLayer`; [losses](losses.md) |
| `einsum` | [model](model.md) — `WannierHead` offsets |
| `jit` recompile-per-shape | [training](training.md) |
| `value_and_grad(has_aux)` | [training](training.md) / [losses](losses.md) |
| `tree_map` gradient averaging | [training](training.md) |
| `PRNGKey` reproducibility | [symmetry](symmetry.md) tests |
