# `model/` — the EGNN and the heads (all the learning)

This is the centerpiece. Everything else exists to feed this directory a graph and to score
what it produces. If you present one thing, present this.

- **Feeds in:** graph tensors from [data](data.md) (`z`, `pos`) — geometry only.
- **Feeds out:** predictions scored by [losses](losses.md), trained by [training](training.md),
  proven correct by [symmetry](symmetry.md).
- **Syntax reference:** [jax_flax_primer](jax_flax_primer.md) for every `nn.*` / `jnp.*` call.
- **Rationale of record:** `../CLAUDE.md` §4 (backbone), §5 (heads). This doc adds the *how*.

Files: `embeddings.py`, `egnn.py`, `heads.py`, `model.py`.

---

## 0. The two tracks (the mental model for the whole file)

Every atom carries **two** representations that travel together through the layers:

| Track | Symbol | Shape | Transforms how under rotation R / translation t | Seeded from |
|---|---|---|---|---|
| invariant scalars | `h` | `[N, H]` | **unchanged** (invariant) | element embedding of `Z` |
| equivariant coords | `x` | `[N, 3]` | `h` stays; `x → R x (+t)` (equivariant) | atom positions |

The single rule that makes the network correct: **`h` may depend on geometry only through
rotation-invariant quantities (distances); `x` may only ever be updated by adding vectors scaled
by invariant weights.** Keep that rule and equivariance is automatic. Break it anywhere and
[symmetry](symmetry.md)'s tests go red.

---

## 1. `embeddings.py` — seeding the invariant track

```python
class ElementEmbedding(nn.Module):
    n_elements: int
    hidden_dim: int
    @nn.compact
    def __call__(self, z):                 # z[N] int -> h[N, hidden_dim]
        return nn.Embed(self.n_elements, self.hidden_dim)(z)
```

- `nn.Embed(n_elements, hidden_dim)` is a learned lookup table with `n_elements` rows of length
  `hidden_dim`. Row `Z` is the feature vector for that element. Carbon and hydrogen thus start
  from different, learnable vectors.
- **Why position is never a node feature (CLAUDE.md §3):** if we concatenated raw `pos` onto
  `h`, the network would see absolute coordinates and could learn frame-dependent shortcuts —
  destroying invariance. Geometry is allowed to enter *only* via edges (distances), which happens
  inside the EGNN layer, never here.
- `n_elements=100` (config) just sizes the table to cover any element; only the rows for elements
  present get trained.

## 2. `egnn.py` — one layer, line by line

`_mlp(dims, name)` builds a small `nn.Sequential` MLP with SiLU between layers (see
[primer §4](jax_flax_primer.md)). `gaussian_rbf` is covered in §3. Here is
`EGNNLayer.__call__(h, x)` with the shape and the reason for every line:

```python
n = h.shape[0]; H = self.hidden_dim

diff  = x[:, None, :] - x[None, :, :]      # [N,N,3]  every r_i - r_j   (EQUIVARIANT)
dist2 = jnp.sum(diff**2, axis=-1)          # [N,N]    squared distances  (INVARIANT)
dist  = jnp.sqrt(dist2 + 1e-12)            # [N,N]    +eps: keep grad finite at d=0
radial = gaussian_rbf(dist, ...)           # [N,N,n_basis]  smooth features of distance
```

`diff` is the all-pairs difference built by the `[:, None, :]` broadcasting trick
([primer §9](jax_flax_primer.md)). It is the only equivariant quantity in the layer. `dist2`/
`dist`/`radial` are functions of it that are **rotation/translation invariant** (a distance
doesn't care about frame) — this is what makes the message legal.

```python
hi = jnp.broadcast_to(h[:, None, :], (n, n, H))    # [N,N,H]  h_i on every edge
hj = jnp.broadcast_to(h[None, :, :], (n, n, H))    # [N,N,H]  h_j on every edge
edge_in = jnp.concatenate([hi, hj, radial], -1)    # [N,N,·]
m_ij    = _mlp([H, H], "phi_e")(edge_in)           # [N,N,H]  the message  (INVARIANT)
```

The message `m_ij = φ_e(h_i, h_j, radial)` depends **only on invariants**. That is the crux: no
raw `diff` goes into `φ_e`. (If you ever passed `diff` here, the message would become
frame-dependent and everything downstream would break.)

```python
off_diag = (1.0 - jnp.eye(n))[..., None]           # [N,N,1] zero on the diagonal
m_ij = m_ij * off_diag                             # drop self-edges i==j
```

We fully connect all pairs but an atom shouldn't message itself; multiplying by `1 - I` zeroes
the diagonal so it contributes to neither sum below.

```python
m_i = jnp.sum(m_ij, axis=1)                         # [N,H] aggregate neighbours (perm-invariant)
h   = h + _mlp([H, H], "phi_h")(jnp.concatenate([h, m_i], -1))   # residual scalar update
```

`sum over j` is **permutation-invariant** (reordering atoms reorders the summands, same total) —
that's where permutation symmetry comes from. `h` stays invariant because everything feeding it
is invariant. The `h +` is a residual connection (helps gradients/stability).

```python
w = _mlp([H, 1], "phi_x")(m_ij)                     # [N,N,1] ONE invariant scalar per pair
x = x + jnp.sum(diff * w, axis=1) / (n - 1 + 1e-9)  # [N,3] equivariant coord update
```

This is the equivariant heart. `φ_x` outputs a **scalar** weight per pair (last dim = 1). The
update is `x_i += Σ_j (x_i − x_j) · w_ij`: equivariant vectors `diff` scaled by invariant scalars
`w`, summed. Dividing by `n-1` averages over neighbours so deep stacks don't blow up.

`EGNNBackbone` just runs `n_layers` of these in sequence, threading `(h, x)`.

## 3. `gaussian_rbf` — why expand the distance

```python
centers = jnp.linspace(0, cutoff, n_basis)
width   = cutoff / n_basis
return jnp.exp(-((dist[..., None] - centers)**2) / (2*width**2))   # [..., n_basis]
```

Instead of feeding the raw scalar distance into `φ_e`, we expand it onto `n_basis` Gaussian
bumps spaced along `[0, cutoff]`. Each output channel responds to a distance band, which gives
the MLP a smoother, higher-resolution handle on geometry (standard trick in message-passing nets;
CLAUDE.md §4 calls it optional — it's on by default via config). `cutoff` here only sets where
the basis functions sit; it is **not** a graph cutoff (the graph is still fully connected).

## 4. The equivariance proof, mapped to the code

Let every atom move `x_i → R x_i + t` for a rotation `R` and shift `t`. Walk the layer:

1. `diff_ij = (R x_i + t) − (R x_j + t) = R(x_i − x_j)` → **`diff` rotates, translation cancels.**
2. `dist2 = ‖R diff‖² = ‖diff‖²` (rotations preserve length) → **`dist`, `radial` unchanged.**
3. `m_ij = φ_e(h_i, h_j, radial)` uses only unchanged inputs → **`m_ij` unchanged.**
4. `h` update uses only `h` and `m_ij` → **`h` stays invariant, all layers deep.**
5. `w = φ_x(m_ij)` unchanged (scalar) → coord update `Σ_j diff_ij · w_ij → Σ_j R diff_ij · w_ij
   = R (Σ_j diff_ij w_ij)` → **the update rotates**, and `x_i` already rotates, so
   `x_i → R x_i + t` is preserved.

Permutation: every cross-atom combination is a **sum over j** (`m_i`, the coord update) or a
symmetric readout — reindexing atoms leaves sums unchanged. Reflection: `R` with `det = −1` works
in exactly the same algebra (nothing above assumed `det = +1`), so mirrored inputs give mirrored
outputs — the architecture has **no chiral feature** (no triple product / signed volume), which
is why the mirror test passes exactly (CLAUDE.md §8).

### The "do NOT alter" list (each line here is load-bearing)
- Don't pass `diff` (or any raw vector/angle) into `φ_e` or `φ_h`. Messages must be invariant.
- Don't let `φ_x` output more than a scalar per pair, and don't multiply `diff` by anything
  frame-dependent. Vectors may only be scaled by invariant scalars.
- Don't add absolute `x` (only differences `x_i − x_j`) into any update — that reintroduces the
  origin and kills translation equivariance.
- Don't concatenate `pos` onto `h` anywhere (see §1).

## 5. `heads.py` — reading out predictions

### EnergyHead (invariant scalar, size-extensive)
```python
per_atom = _mlp(...)(h)     # [N,1]  one number per atom from its invariant features
return jnp.sum(per_atom)    # scalar
```
Invariant because `h` is invariant. **Sum, not mean**, so energy scales with molecule size the
way a total energy physically does (add an atom → add its contribution). CLAUDE.md §5.

### WannierHead (equivariant point set per spin)
```python
centroid = jnp.mean(x, axis=0)                       # [3]  equivariant
A = _mlp(...)(h).reshape(N, S, M)                     # [N,S,M]  INVARIANT per-atom weights
offsets = jnp.einsum("nsm,nd->smd", A, x - centroid)  # [S,M,3]
centers = centroid + offsets                          # [S,M,3]  equivariant
```
- `S` = number of spin channels (2: up/down), `M` = `max_centers` slots per spin.
- A center is the **centroid plus a weighted sum of atom offsets**, where the weights `A` come
  from invariant `h`. Equivariance: `centroid` moves with the molecule, `x − centroid` is
  translation-free and rotates with `R`, weights are invariant ⇒ `centers → R·centers (+t)`.
  This is the §5 requirement "centers must come out of the equivariant track, never a raw MLP."
- `jnp.einsum("nsm,nd->smd", A, V)` means: for each spin `s`, slot `m`, coordinate `d`, sum over
  atoms `n` of `A[n,s,m] · V[n,d]`. It's the batched weighted sum in one call
  ([primer §9](jax_flax_primer.md) for the index intuition).

```python
rp = jnp.sum(_mlp(...)(h), axis=0).reshape(S, M, 2)   # [S,M,2] invariant per-slot readouts
radii    = nn.softplus(rp[..., 0])                    # [S,M] > 0
presence = nn.sigmoid(rp[..., 1])                     # [S,M] in (0,1)
```
- `radii` and `presence` are invariant scalars (from `h`, summed over atoms). `softplus` keeps
  radius positive; `sigmoid` keeps presence a 0–1 "is this slot a real center?" gate.
- **presence** is what lets a *fixed* `M` slots represent a *variable* number of true centers: an
  unused slot learns `presence ≈ 0`, and in [losses](losses.md) it enters as amplitude, so a
  zero-amplitude Gaussian contributes nothing to the cloud (CLAUDE.md §6).

## 6. `model.py` — assembly

`QM9Model.__call__(z, pos)` = embed `z`→`h`, set `x=pos`, run `EGNNBackbone`, then both heads;
returns `{"energy": scalar, "wannier": {centers, radii, presence}}`. `model_from_config(cfg)`
reads `configs/*.yaml` and constructs it — no magic numbers in code (CLAUDE.md §11).

---

## Failure modes & debugging
| Symptom | Likely cause | Where |
|---|---|---|
| rotation test fails on centers | equivariant track polluted — `diff`/`pos` fed to `φ_e`/`h`, or `φ_x` not scalar | §2, §4 |
| translation test fails | absolute `x` used instead of `x_i − x_j`, or head not built off `centroid` | §2, §5 |
| energy changes under permutation | a non-sum aggregation crept in (e.g. indexing atom 0) | §2, §5 |
| mirror test fails | you added a chiral feature (triple product / signed volume) | §4 |
| `NaN` early in training | remove the `+1e-12` in `sqrt`, or radius not passed through `softplus` | §2, §5 |
| centers all collapse to centroid | `A` weights vanish — check `hidden_dim`, learning rate | §5 |

## Rebuild-by-hand order
1. `ElementEmbedding` (§1) → get `h` from `z`.
2. One `EGNNLayer` (§2): pairwise `diff`/`dist`, invariant message, `sum`-aggregate, residual `h`,
   scalar-weighted coord update. **Test equivariance now on random weights** ([symmetry](symmetry.md)).
3. Stack into `EGNNBackbone`.
4. `EnergyHead` (sum of per-atom scalars) → train energy-only first.
5. `WannierHead` (centroid + einsum offsets, softplus radius, sigmoid presence).
6. Wire in `model.py`, drive hyperparameters from config.
