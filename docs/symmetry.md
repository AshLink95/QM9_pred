# `symmetry/` + `tests/` — the correctness claim

The whole project rests on one promise: **outputs respect the molecule's symmetries by
construction** (CLAUDE.md §1, §8). This directory is where that promise is *tested*, so it's
load-bearing, not an afterthought. The transforms live in `symmetry/transforms.py`; the assertions
live in `tests/test_equivariance.py`. They exercise [model](model.md); syntax in
[jax_flax_primer](jax_flax_primer.md).

---

## 1. `transforms.py` — the group actions

Plain `numpy` (data-side). Each returns transformed positions (or a permutation):

```python
translate(pos, t)  ->  pos + t
rotate(pos, R)     ->  pos @ R.T          # row vectors: r' = R r
mirror(pos, axis)  ->  reflect through the plane normal to `axis` (a diag(±1) matrix)
permutation(n)     ->  a random index permutation
```

### `random_rotation` — why QR, why the det fix
```python
q, r = np.linalg.qr(rng.standard_normal((3, 3)))
q = q * np.sign(np.diag(r))          # remove QR's sign ambiguity → uniform-ish
if np.linalg.det(q) < 0: q[:, 0] = -q[:, 0]   # force det = +1 (proper rotation)
```
QR of a random Gaussian matrix gives a random orthogonal `Q` (`QᵀQ = I`). Orthogonal matrices
split into **proper rotations** (`det = +1`) and **improper** ones (`det = −1`, i.e. rotation +
reflection). For the *rotation* test we want a proper rotation, so we flip a column if the det is
negative. `mirror` is the deliberately **improper** transform (`det = −1`) for the reflection
test.

## 2. `tests/test_equivariance.py` — the four checks

The file first does:
```python
jax.config.update("jax_enable_x64", True)   # float64 → tolerances of 1e-8 are meaningful
```
JAX defaults to float32; enabling x64 lets the tests assert machine-precision agreement instead
of loosening to ~1e-4.

A module fixture builds a small model, `init`s it once (fixed `PRNGKey` →
reproducible, [primer §2](jax_flax_primer.md)), and jits `apply`. Then, comparing to a baseline
prediction:

| Test | Assertion | What it proves |
|---|---|---|
| `test_translation` | energy unchanged; `centers_out == centers + t` | translation equivariance |
| `test_rotation` | energy unchanged; `centers_out == centers @ Rᵀ`; radii unchanged | rotation equivariance |
| `test_permutation` | energy unchanged; center **set** unchanged (compared via `np.sort`) | atom-order invariance |
| `test_mirror` | energy unchanged; `centers_out == centers @ Mᵀ` | reflection equivariance |

All at `atol=1e-8`. The permutation test sorts both center arrays before comparing because the
set is unordered — it checks the *set* is preserved, not slot-by-slot identity (which also holds
here, but sorting is the honest test).

## 3. The point that makes this special: **it runs on the UNTRAINED net**

The fixture never trains. Random weights already satisfy every assertion, because equivariance is
a property of the **architecture**, not something learned (CLAUDE.md §10 step 4). That's the
strongest possible demonstration and the reason these tests are the *gate*: they must pass before
any training. A model that fails them is wrong regardless of its loss — training cannot fix a
structural asymmetry, and no amount of data augmentation is needed to create the symmetry.

Trace any failure straight back to [model](model.md) §4 ("the do NOT alter list"): a broken
translation test ⇒ absolute `x` leaked into an update; broken rotation ⇒ a raw vector entered a
message or `φ_x` returned more than a scalar; broken permutation ⇒ a non-sum aggregation.

## 4. The chirality caveat (CLAUDE.md §8)
Pure distances are reflection-invariant, and this architecture has **no chiral feature**, so the
mirror test passes exactly — mirror in, mirror out. That's correct for energy and centers here.
*If* you ever needed to distinguish enantiomers (a genuinely chiral target), you'd add a signed
feature (a triple product / signed volume of edge vectors) and then the mirror test *should*
legitimately differ on that target. Don't add it preemptively.

---

## Failure modes & debugging
| Symptom | Meaning | Where to look |
|---|---|---|
| translation test red | origin leaked in — absolute `x`, not `x_i − x_j` | [model](model.md) §2/§4, `WannierHead` centroid |
| rotation test red | frame-dependent op in a message, or vector mixing | [model](model.md) §4 do-not-alter list |
| permutation test red | aggregation isn't a symmetric sum | [model](model.md) §2 `sum(axis=1)` |
| mirror test red (unexpected) | a chiral feature crept in | [model](model.md) §4 |
| tests pass at 1e-4 but not 1e-8 | x64 not enabled | the `jax_enable_x64` line, §2 |

## Rebuild-by-hand order
1. `transforms.py`: translate, `rotate` (`@ R.T`), `mirror` (diag ±1), `random_rotation`
   (QR + det fix), `permutation`.
2. Build the untrained model, one fixed-seed `init`.
3. Assert the four properties at `atol=1e-8` with x64 on — **before** writing any training code.
4. Keep them in CI/pre-commit so a regression fails loudly (CLAUDE.md §11).
