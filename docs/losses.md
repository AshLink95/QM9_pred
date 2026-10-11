# `losses/` — energy regression + the matched Wannier loss

Scores what [model](model.md) predicts against the parsed targets from [data](data.md); the
scalar it returns is what [training](training.md) differentiates. Syntax:
[jax_flax_primer](jax_flax_primer.md). Rationale: `../CLAUDE.md` §6 (the load-bearing decision,
including the "Revised loss" and why the first version was replaced).

Files: `energy.py` (trivial), `wannier_match.py` (the Wannier loss), `wannier_cloud.py` (the
closed-form Gaussian overlap, used as the fixed-width guide term).

---

## 1. `energy.py`

```python
def energy_loss(pred_energy, true_energy):
    return jnp.mean((pred_energy - true_energy) ** 2)
```

Plain mean-squared error on one invariant scalar. (Training computes the same thing inline over
a batch in `training/train.py:_batch_loss`.)

## 2. The problem: comparing two unordered sets

Predicted centers come out of `WannierHead` as `C` candidates per spin (a few per atom,
[model](model.md) §5). True centers are an **unordered set**: candidate #k has no natural
correspondence to true center #k. A loss
like `Σ ‖pred[k] − true[k]‖²` would punish a perfectly correct set emitted in a different order,
which gives contradictory gradients. Any Wannier loss has to be order-free.

### What went wrong with the first loss
The first version turned each set into a smooth density of Gaussians and compared densities
(§5 below). The model controlled each predicted Gaussian's **width** (radius) and **strength**
(a presence score). On the test split it learned to cover the true centers with a few wide, faint
blobs: count right 0.1% of the time, centers off by 0.61 Å, radii worse than a constant guess.
With width and strength under its control, blurring scored well.

## 3. `wannier_match.py` — the current loss, per spin

Three facts make a sharper loss possible:
- **The count is known.** Spin `s` of a molecule has `k = Σ valence(Z) / 2` centers
  (`data/electrons.py`; true for 100% of the dataset). The head offers more candidates than that
  (`C ≥ k`); the `k` with the highest selection logits are the prediction.
- With a known count, **one-to-one matching** is clean: each true center gets exactly one
  candidate, and the rest are "not chosen".
- **Spin is separate everywhere:** every term below is computed per spin, so an up center can
  never pair with a down center.

### 3a. The pairing (Hungarian algorithm) — computed on the CPU
```python
perm = assign(pred_centers, true_centers, k, pool)   # NumPy + SciPy, on the host
loss = wannier_match_loss(..., k, perm, ...)          # jitted, on the GPU
```
- For each molecule and spin, `scipy.optimize.linear_sum_assignment` on the `k × C` matrix of
  squared distances (true centers × candidates; rectangular is fine) picks `k` candidates and the
  one-to-one pairing with the smallest total. `assign` does this for a whole batch, spread over a
  **thread pool** (SciPy releases Python's GIL, so the threads truly run in parallel).
- `perm[b, s, i]` = which candidate is paired with true center `i` (entries `i ≥ k` are padding,
  masked out).
- **Pairs are used whole** (`layout=(singles, pairs)`): the two members of a ± pair share one
  selection logit, because a mirror may swap them. If the matching uses only one member — e.g. a
  pair collapsed onto an in-plane spot standing in for a single — that logit would get the targets
  1 and 0 at once and settle at "maybe". So a half-used pair is excluded and the assignment
  re-solved (a few rounds at most). Off-plane centers come in mirror pairs, so the pair that sits
  on them is matched whole.
- The pairing is a discrete choice and **isn't differentiated** (as in DETR): inside the loss it
  passes through `jax.lax.stop_gradient`. Gradients flow through the distances of the chosen pairs.
- **Why on the CPU, outside the compiled step:** training overlaps it with GPU work. The CPU pairs
  the *next* batch while the GPU trains on the current one, using predictions made one update
  earlier (a "lagged assignment"; [training](training.md) §3). Keeping it out of compiled code also
  lets the compiled programs be cached on disk and split across several GPUs. (An earlier version
  called SciPy from inside the compiled step via `jax.pure_callback`; the GPU then sat idle during
  every pairing.)

### 3b. The terms
```python
mc  = take_along_axis(pred_c, perm)                  # predicted center paired with each true one
mr  = take_along_axis(pred_r, perm)
pos = Σ mask·‖true_c − mc‖² / n                      # Å²
rad = Σ mask·((true_r − mr) / r_std)² / n            # dimensionless
select = mean BCE(logit, chosen)                     # chosen = 1 if matched, else 0
guide = mean over (molecule, spin) of cloud_l2(matched, true)/k   # fixed width, unit amplitudes
total = pos + radius_weight·rad + select_weight·select + guide_weight·guide
```
- `mask[b,s,i] = i < k[b,s]`: only real centers count, and `n` is how many there are.
- **Position** is plain squared error on matched pairs. That's sharp: it keeps pushing until each
  center sits on its partner.
- **Radius** is regressed directly per matched pair, divided by the dataset's radius std
  (`r_std`, measured on the train split at startup). The head's radius output starts at the
  dataset mean ([model](model.md) §5), so training starts at "constant guess" quality.
- **Selection** is binary cross-entropy on every candidate's logit: 1 if the matching chose it,
  0 otherwise. At prediction time the `k` highest logits are the centers
  (`model/heads.py:select_centers`). It is classification only — no amplitude or width ever
  enters an overlap — so it can't be used to blur (the first loss's failure).
  **Validation, the report and `main.py` score the selected candidates**, never a match over all
  `C` (that would flatter the numbers: with 4 candidates per atom, something is always close).
- **Guide** is the closed-form cloud overlap (§4–§5) between the matched candidates and the truth,
  with a **fixed** width `guide_sigma` (0.3 Å) and amplitude 1 for every real center. Early in
  training the pairing can flip
  from step to step. The guide doesn't care about pairing and gives a smooth pull toward the
  right region. Because its width is fixed, it can't be gamed by widening blobs.
- Weights come from `configs/*.yaml` (`wannier.radius_weight`, `select_weight`, `guide_weight`); the
  whole Wannier term is scaled by `loss.lambda_wannier` (0 in `energy_only.yaml` skips it
  entirely).

## 4. `wannier_cloud.py` — the closed-form overlap (now the guide)

A set of centers becomes a sum of isotropic Gaussians:
```
ρ(x) = Σ_k  w_k · G(x; c_k, σ) ,    G(x; c, σ) = exp( −‖x − c‖² / (2σ²) )
L    = ∫ (ρ_pred − ρ_true)² dx  =  ⟨p,p⟩ − 2⟨p,t⟩ + ⟨t,t⟩
```
Every term is a sum of **pairwise Gaussian overlaps**, and the overlap of two isotropic
Gaussians has a closed form (variances `s = σ²`):
```
O(a, b, s_a, s_b) = (2π)^{3/2} · ( s_a s_b / (s_a + s_b) )^{3/2} · exp( −‖a−b‖² / (2(s_a+s_b)) )
```
So it's a finite double sum, **O(n²)** in the number of centers, with no spatial grid, and
differentiable everywhere (the `exp` never reaches exactly zero). CLAUDE.md §6: never compute it
on a grid or by Monte Carlo.

## 5. `_pairwise_overlap_sum` — the formula, vectorized
```python
def _pairwise_overlap_sum(ca, wa, sa, cb, wb, sb):
    d2     = jnp.sum((ca[:, None, :] - cb[None, :, :])**2, axis=-1)   # [A,B]  ‖a_i − b_j‖²
    ssum   = sa[:, None] + sb[None, :]                                # [A,B]  s_i + s_j
    prefac = _TWO_PI_32 * (sa[:, None] * sb[None, :] / ssum) ** 1.5
    overlap = prefac * jnp.exp(-d2 / (2.0 * ssum))                    # [A,B]  O(a_i, b_j)
    return jnp.sum(wa[:, None] * wb[None, :] * overlap)               # Σ w_i w_j O
```
The `[:, None, :]` broadcasting ([primer §9](jax_flax_primer.md)) forms every pair at once.
`cloud_l2` assembles `⟨p,p⟩ − 2⟨p,t⟩ + ⟨t,t⟩` from three calls. In the guide, padded entries
have amplitude 0 and drop out.

## 6. Why not the alternatives (so you can defend it)
- **Learned count / presence score:** the count is physics (valence electrons / 2), so learning
  it only adds a way to fail. It's what let the first loss blur. (The selection logit only
  chooses *which* candidates; the count stays computed.)
- **Density loss with predicted widths:** width becomes a blur knob. That's the measured failure.
- **Fixed per-atom assignment (Deep Wannier style):** works in water, where every center clearly
  belongs to one oxygen. In organics a C–C bond center sits exactly halfway between two carbons,
  so "nearest atom" is a coin flip and per-atom counts vary. The head's candidates are per atom
  too, but no center is *assigned* to an atom: the matching lets any atom's candidate take any
  center, and the selection decides how many each atom contributes.
- **Chamfer distance:** no matching needed, but it allows duplicates (two predictions on one true
  center). Matching rules that out.
- **Literal region/IoU overlap:** no closed form (needs a grid) and zero gradient until shapes
  touch.

---

## Failure modes & debugging
| Symptom | Cause | Fix |
|---|---|---|
| centers good but radius drifts | `radius_weight` too low, or `r_std` tiny | check the startup "Wannier radius: mean … std …" line |
| training jumpy early, pairings flip | normal for matching at the start | the guide steadies it; raise `guide_weight` a bit if needed |
| two predicted centers merge onto one true center | guide σ too large | keep `guide_sigma` below the ~0.6–1 Å spacing of neighboring centers |
| `ValueError: no valence charge for element(s)` | a new element appeared | add it to `data/electrons.py:VALENCE` (must match the pseudopotentials) |
| `WARNING: dropped N training molecules whose Wannier count != valence/2` | parsed count disagrees with the rule | inspect those molecules (open-shell? bad `.wout`?) |
| up center paired with down | someone merged spins | every term must loop/index per spin |
| `select` stuck near 0.69 (log 2) | logits never separate | check `select_weight`; check `layout` is passed to `assign` (half-used pairs) |
| val center error far above train `pos` | selection picks wrong candidates | compare with the matched training error; raise `select_weight` |
| `test_zero_on_identical` fails at ~1e-7 | float32 rounding | tolerance is `1e-5` on purpose |

## Rebuild-by-hand order
1. `centers_per_spin` (valence table, closed-shell rule).
2. `cloud_l2` (closed-form overlap; zero on identical sets).
3. Host `assign` (per-(molecule, spin) rectangular Hungarian on a thread pool, pairs used whole)
   returning `perm`.
4. `wannier_match_loss(…, perm)`: mask from `k`, matched position and radius terms
   (`stop_gradient` on `perm`), selection BCE, fixed-width guide.
5. Tests (`tests/test_wannier_match.py`): zero on a reordered truth, a known offset gives a known
   value, spins never mix, more candidates than centers, the selection target follows the
   pairing, pairs used whole, gradients finite under `jit`, unmatched candidates get no position
   pull.
