# `data/` — parsing SIESTA files and building the graph (no learning)

Stage 1–2 of the pipeline (CLAUDE.md §3, §9): turn files on disk into the tensors
[model](model.md) consumes and the targets [losses](losses.md) scores against. **Deterministic —
nothing here is learned.** This directory is the boundary: it hands the model an already-formed
graph, and the model never reaches back to parse files. Mostly plain `numpy`
([primer §1](jax_flax_primer.md) on why `np` here, `jnp` in the model).

Files: `units.py`, `parse_siesta.py`, `parse_wout.py`, `graph.py`, `dataset.py`,
`electrons.py`.

New to the physics (what a `.fdf` or a Wannier center is)? Read
[physics_primer](physics_primer.md) first.

---

## 1. `units.py`
One place for conversions so they bite you exactly once (CLAUDE.md §9): `BOHR_TO_ANG`,
`ANG_TO_BOHR`. Positions are kept in **Å**. Bohr shows up when an `.fdf` gives coordinates in
Bohr; the `.out` trial-projector centers are in Bohr too, but we never use them (§1).

## 2. `parse_siesta.py` — geometry from `.fdf`

The SIESTA `.out` log is not read at all. It used to supply the total energy, which the model
no longer predicts (CLAUDE.md §1: energy and Wannier traded off when trained in one network).

### The fdf tokenizer
fdf is SIESTA's input format. Two quirks the code handles:
- **Key normalization.** `_norm_key(s)` lowercases and strips `.`, `-`, `_` so
  `AtomicCoordinatesFormat`, `atomic-coordinates-format`, `atomic_coordinates_format` all match.
  Block markers `%block` / `%endblock` are detected after `.lstrip("%")` (the `%` isn't part of
  the key).
- **Blocks vs scalars.** `_read_fdf_tokens` returns `(scalars, blocks)`: a `Key value` line
  becomes `scalars[key] = value`; everything between `%block X … %endblock X` becomes
  `blocks[X] = [content lines]`. Comments (`#`) are stripped first.

### From tokens to tensors
- `_species_to_z(blocks)` reads `%block ChemicalSpeciesLabel` lines `index Z label` → `{species
  index: Z}`.
- `_coord_scale(scalars)` reads `AtomicCoordinatesFormat`: returns `1.0` for Ångström forms,
  `BOHR_TO_ANG` for Bohr forms, and **raises** on fractional/ScaledCartesian (those need lattice
  vectors; QM9 is isolated molecules and shouldn't use them — better a loud error than a silent
  wrong geometry).
- `parse_fdf(path) -> (Z[N] int32, pos[N,3] float64 in Å)`: reads `%block
  AtomicCoordinatesAndAtomicSpecies` (`x y z species`), maps species→Z, scales coords to Å.

## 3. `parse_wout.py` — converged Wannier centers

`.wout` is the Wannier90 output. The centers we want are the **converged** ones in the
`Final State` section — *not* the trial-projector centers in the SIESTA `.out` (§1), and not the
per-iteration lines printed during minimization.

```python
def _final_state_block(text):
    start = text.rfind("Final State")        # last occurrence
    tail  = text[start:]
    end   = tail.find("Sum of centres")      # section ends here
    return tail[:end]
```
Slicing to just this block is what prevents matching the intermediate `WF centre and spread`
lines from earlier iterations.

```python
_WF_RE = re.compile(r"WF centre and spread\s+\d+\s+\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,"
                    r"\s*(-?\d+\.\d+)\s*\)\s+(-?\d+\.\d+)")
```
Each match yields `(x, y, z, spread)`. Then:
- `centers = arr[:, :3]` (Å),
- `radii = sqrt(spread)` — the spread is `⟨r²⟩−⟨r⟩²` in Å², so its square root is a length: the
  per-center "size" the model learns to predict ([losses](losses.md) §3b),
- `spin` = the label you pass in (`"up"`/`"down"`), determined by **which file** (`*.up.wout` vs
  `*.down.wout`), not by anything inside the file.

## 4. `graph.py` — positions → fully-connected graph (CLAUDE.md §3)

```python
def fully_connected_edges(n):
    idx = np.arange(n); send, recv = np.meshgrid(idx, idx, indexing="ij")
    mask = send != recv
    return np.stack([send[mask], recv[mask]], 0)     # edge_index [2, E], E = N(N-1)
```
`meshgrid` builds every `(i,j)` pair; `send != recv` drops self-loops. QM9 is tiny (≤~29 atoms →
≤~400 edges) so all-pairs is trivial and preserves every neighbour — no cutoff, no
sparsification, no "canonical" graph (which would be unstable and lossy, §3).

```python
def build_graph(pos):
    edge_index = fully_connected_edges(len(pos)); i, j = edge_index
    rel_vec   = pos[i] - pos[j]                       # [E,3] equivariant
    distances = np.linalg.norm(rel_vec, axis=-1)      # [E]   invariant
    return {"edge_index": edge_index, "rel_vec": rel_vec, "distances": distances}
```
Shapes: `edge_index [2,E]`, `rel_vec [E,3]`, `distances [E]`. **Note:** the EGNN actually
recomputes distances *in-layer* from `pos` (so `jax.grad` can reach positions, §3) — this
precompute is here for inspection/notebooks, not required by training.

## 5. `dataset.py` — molecules → examples, split, cache

On the cluster the file types live in **separate folders**, e.g. `done/C2F3N3O_133481.fdf`
and `wout_files/C2F3N3O_133481.manifold.valence.up.wout` (and `.down.wout`). Files are matched
by **molecule id**: the filename up to the first `.` (here `C2F3N3O_133481`). The `.out` files
(e.g. in `out_files/`) are not needed.

- `_mol_id(path)`: the id (`path.name.split(".", 1)[0]`).
- `_index(dirs, ext)`: walks every root **recursively** (`rglob`) and maps id → list of files
  with that extension (here `.wout`).
- `load_molecule(fdf_path, wout_by_spin, spins)`: parses one molecule from already-found paths.
  Returns `{z, pos, wannier=[{centers, radii} per spin]}`, with the `wannier` list ordered like
  config `wannier.spins` (`[up, down]`).
- `build_dataset(dirs, spins)`: accepts one or many roots. For every `.fdf` it finds one `.wout`
  per spin (the spin name must appear in the filename). Molecules missing a file,
  or whose `.wout` has no `Final State` (an unconverged Wannier run), are **skipped and counted**
  instead of stopping the build. It also computes `meta["max_centers"]` (the largest per-spin
  count), which sizes the Wannier head.
- `split(examples, val_frac, test_frac, seed)`: a fixed shuffle (seed 0) cut into train / val /
  test, 80 / 10 / 10 by default. Same input list + same seed ⇒ same split, which is how
  `scripts/evaluate.py` finds the exact test molecules again
  ([ml_primer](ml_primer.md) explains why there are three groups).
- `save_cache` / `load_cache`: pickle the parsed examples + meta into `dataset.pkl`, so the slow
  parsing happens once.
- **Older caches:** a `dataset.pkl` built while the energy was still a target has an extra
  `energy` field per molecule and only contains molecules that also had a `.out`. Both are
  harmless: the field is ignored. **Keep using that cache** so the train/val/test split, and so
  the test molecules, stay identical to the earlier runs. Rebuilding now no longer requires a
  `.out`, so molecules that have `.wout` files but no `.out` would join, and the split would
  shift (the split is a shuffle of the list).

## 6. `electrons.py` — how many Wannier centers each spin has

```python
VALENCE = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7}         # H, C, N, O, F
centers_per_spin(z, n_spins)  ->  [Σ VALENCE[z] / 2] * n_spins
```
Each spin channel of a closed-shell molecule has (valence electrons) / 2 Wannier centers
([physics_primer](physics_primer.md) §5). This matched the parsed `.wout` count for 100% of the
dataset, so the count is **computed, never learned**. Training, the report, and `main.py` all use
this function. It raises on an element missing from `VALENCE` (add it; the value must match the
SIESTA pseudopotential) or on an odd electron count (an open-shell molecule, not supported).

---

## Failure modes & debugging
| Symptom | Cause | Where |
|---|---|---|
| `Unsupported AtomicCoordinatesFormat` | fractional/scaled coords need lattice vectors | `_coord_scale`; convert upstream or add lattice support |
| way too many / garbage centers | matched per-iteration WF lines, not Final State | `_final_state_block` slicing; §3 |
| wrong element identities | `ChemicalSpeciesLabel` species index misread | `_species_to_z`; §2 |
| many molecules "skipped … missing spin .wout" | filenames don't share the molecule id, or the spin name isn't in the `.wout` filename | check `_mol_id` against real names; §5 |
| skipped with "No 'Final State' section found in …" | that Wannier90 run didn't converge | expected for a few hundred molecules; they're excluded |
| `ValueError: no valence charge for element(s)` | an element not in `VALENCE` | add it to `electrons.py`; §6 |
| head too small at train | `max_centers` in config < dataset max | `scripts/train.py` raises it to `meta["max_centers"]` |

## Rebuild-by-hand order
1. `units.py` constants.
2. fdf tokenizer (`_norm_key`, `_read_fdf_tokens`) → `_species_to_z`, `_coord_scale` →
   `parse_fdf`. Verify against a real `.fdf` by hand.
3. `parse_wout` (Final-State slice + WF regex + `sqrt(spread)`).
4. `graph.build_graph` (meshgrid edges, rel-vec, distances).
5. `dataset` (id index across roots, skip-and-count, `max_centers`, split, cache).
6. `electrons.centers_per_spin` (valence table). Then feed [model](model.md).
