# `losses/` — energy regression + the Gaussian-cloud Wannier loss

Scores what [model](model.md) predicts against the parsed targets from [data](data.md); the
scalar it returns is what [training](training.md) differentiates. Syntax:
[jax_flax_primer](jax_flax_primer.md). Rationale: `../CLAUDE.md` §6 (the load-bearing decision).

Files: `energy.py` (trivial), `wannier_cloud.py` (the interesting one).

---

## 1. `energy.py`

```python
def energy_loss(pred_energy, true_energy):
    return jnp.mean((pred_energy - true_energy) ** 2)
```

Plain mean-squared error on one invariant scalar. Nothing subtle. (`jnp.mean` over a single
molecule is just the square; the `mean` matters only if you ever pass a batch vector.)

## 2. Why the Wannier loss can't be element-wise

Predicted centers come out of `WannierHead` as `M` fixed slots per spin; true centers are an
**unordered set of variable size** `K`. Predicted slot #k has *no* correspondence to true center
#k. A loss like `Σ ‖pred[k] − true[k]‖²` would punish a perfectly correct set emitted in a
different order — contradictory gradients. So we compare the two sets **without ordering** by
turning each into a smooth density and comparing densities.

## 3. The density and the L2 between two densities

Represent a set of centers as a sum of weighted isotropic Gaussians (a "cloud"):

```
ρ(x) = Σ_k  w_k · G(x; c_k, σ_k) ,    G(x; c, σ) = exp( −‖x − c‖² / (2σ²) )
```

`w_k` is an amplitude (predicted **presence** for the model's cloud, `1` for a true center),
`c_k` the center, `σ_k` the width. The loss is the squared L2 distance between the two clouds:

```
L = ∫ (ρ_pred − ρ_true)²  dx
  = ∫ρ_pred²  −  2∫ρ_pred ρ_true  +  ∫ρ_true²
  =  ⟨p,p⟩    −      2⟨p,t⟩       +    ⟨t,t⟩
```

The trick (CLAUDE.md §6: **do not** grid or Monte-Carlo this integral): every one of those three
terms is a sum of **pairwise Gaussian overlaps**, and the overlap of two isotropic Gaussians has
a closed form. With variances `s = σ²`:

```
O(a, b, s_a, s_b) = ∫ G(x;a,σ_a) G(x;b,σ_b) dx
                  = (2π)^{3/2} · ( s_a s_b / (s_a + s_b) )^{3/2} · exp( −‖a−b‖² / (2(s_a+s_b)) )
```

So `⟨p,t⟩ = Σ_{i,j} w^p_i w^t_j O(c^p_i, c^t_j, s^p_i, s^t_j)` — a finite double sum, **O(n²)** in
the number of centers (tens), no spatial grid, and differentiable everywhere (the `exp` never
flattens to exactly zero, so gradients survive even when clouds don't yet overlap).

## 4. `_pairwise_overlap_sum` — the formula, vectorized

```python
def _pairwise_overlap_sum(ca, wa, sa, cb, wb, sb):
    d2     = jnp.sum((ca[:, None, :] - cb[None, :, :])**2, axis=-1)   # [A,B]  ‖a_i − b_j‖²
    ssum   = sa[:, None] + sb[None, :]                                # [A,B]  s_i + s_j
    prefac = _TWO_PI_32 * (sa[:, None] * sb[None, :] / ssum) ** 1.5   # [A,B]  the (…)^{3/2} factor
    overlap = prefac * jnp.exp(-d2 / (2.0 * ssum))                    # [A,B]  O(a_i, b_j)
    return jnp.sum(wa[:, None] * wb[None, :] * overlap)               # scalar Σ w_i w_j O
```

- Same `[:, None, :]` broadcasting as the model ([primer §9](jax_flax_primer.md)): it forms the
  full `[A,B]` matrix of every pair `(i,j)` at once, no loops.
- `_TWO_PI_32 = (2π)^{3/2}` is precomputed at import.
- The final `jnp.sum(w_i w_j O_ij)` collapses the matrix to the scalar term.

`cloud_l2` assembles `⟨p,p⟩ − 2⟨p,t⟩ + ⟨t,t⟩` from three calls to it.

## 5. `wannier_loss` — split by spin, widths from radii

```python
for s in range(S):                                # one cloud L2 per spin channel
    pc, pr, pw = pred["centers"][s], pred["radii"][s], pred["presence"][s]
    tc, tr     = true[s]["centers"], true[s]["radii"]
    tw = jnp.ones(tc.shape[0])                     # true amplitude = 1
    ps = (width_scale * pr) ** 2 + 1e-6            # variance from radius
    ts = (width_scale * tr) ** 2 + 1e-6
    total += cloud_l2(pc, pw, ps, tc, tw, ts)
```

- **Split by spin (CLAUDE.md §6):** up compares only to up, down only to down. An up center can
  never overlap-cancel a down center because they live in separate `cloud_l2` calls. (This is the
  `test_spin_split` property — swapping spins does not give zero.)
- **Amplitudes:** predicted = `presence` (so an unused slot with presence≈0 vanishes from the
  cloud — that's how a fixed `M` slots represent a variable count), true = `1`.
- **Widths from `w_radii`:** `σ = width_scale · radius`, i.e. each Gaussian is as wide as that
  center physically is. This uses the real per-center size and honors "not all Wanniers are the
  same." `+1e-6` keeps variance strictly positive.
- **Shapes:** `pred` is the stacked `[S,M,·]` dict from `WannierHead`; `true` is a length-`S`
  **list** of `{centers[K_s,3], radii[K_s]}` — a list, not an array, because `K_up` and `K_down`
  differ per molecule and per-molecule loss needs no padding.

## 6. Why not the alternatives (so you can defend it)
- **Hungarian / Sinkhorn matching:** treats centers as identical points, needs padding to a
  common count, and still imposes a (soft) correspondence. The cloud handles variable count for
  free and is size-aware.
- **Literal region/IoU overlap (union of balls):** volume has no closed form → forces a 3D grid
  (millions of points/molecule/step) or Monte Carlo, and its gradient is exactly **zero** when
  shapes don't yet touch (nothing tells the optimizer which way to move). The Gaussian form is
  orders of magnitude cheaper and has gradients everywhere — see the "money plot" in
  `notebooks/wannier_loss.ipynb`.

---

## Failure modes & debugging
| Symptom | Cause | Fix |
|---|---|---|
| loss can't separate two nearby centers | widths too large → clouds smear into one blob | lower `width_scale`; keep width tied to physical radius, no big free smoothing (§6) |
| `test_zero_on_identical` fails at ~1e-7 | float32 rounding, not a bug | tolerance is `1e-5` in `tests/test_loss.py` on purpose |
| gradients vanish | you reintroduced a hard cutoff / IoU-style term | keep the analytic Gaussian overlap |
| up matches down | you built one cloud over both spins | keep the per-spin loop in `wannier_loss` |
| `NaN` in loss | a radius reached 0 (no `softplus`) or `ssum=0` | `softplus` in the head (§5 of [model](model.md)), `+1e-6` here |

## Rebuild-by-hand order
1. Write `O(a,b,s_a,s_b)` and unit-test that `∫G_a G_b` matches it (or that identical clouds give
   `⟨p,p⟩−2⟨p,t⟩+⟨t,t⟩ = 0`).
2. `_pairwise_overlap_sum` (the `[A,B]` broadcast), then `cloud_l2` (the three terms).
3. `wannier_loss`: loop spins, widths from radii, presence as amplitude.
4. Sanity properties (mirror in `tests/test_loss.py`): zero on identical, order-invariant,
   spin-split, positive when different.
