# `data/` — parsing SIESTA files and building the graph (no learning)

Stage 1–2 of the pipeline (CLAUDE.md §3, §9): turn files on disk into the tensors
[model](model.md) consumes and the targets [losses](losses.md) scores against. **Deterministic —
nothing here is learned.** This directory is the boundary: it hands the model an already-formed
graph, and the model never reaches back to parse files. Mostly plain `numpy`
([primer §1](jax_flax_primer.md) on why `np` here, `jnp` in the model).

Files: `units.py`, `parse_siesta.py`, `parse_wout.py`, `graph.py`, `dataset.py`.

---

## 1. `units.py`
One place for conversions so they bite you exactly once (CLAUDE.md §9): `BOHR_TO_ANG`,
`ANG_TO_BOHR`, `RY_TO_EV`. Positions are kept in **Å**, energies in **eV**. Note the `.out`
trial-projector centers are in Bohr — but we never use them (§1), so Bohr only shows up if you
ever touch coordinates in atomic units.

## 2. `parse_siesta.py` — geometry from `.fdf`, energy from `.out`

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

### Energy
```python
_TOTAL_RE = re.compile(r"siesta:\s*Total\s*=\s*(-?\d+\.?\d*)")
def parse_energy(out): return float(_TOTAL_RE.findall(text)[-1])   # LAST match
```
SIESTA prints `siesta: Total = …` on **every SCF iteration**; the **converged** value is the
last one. `findall(...)[-1]` takes it. Taking the first (or a mid-file) value would give an
unconverged energy — a subtle correctness bug (CLAUDE.md §1: targets are the final values).

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
- `radii = sqrt(spread)` — the spread is `⟨r²⟩−⟨r⟩²` in Å², so its square root is a length; this
  is the per-center "size" that [losses](losses.md) uses as the Gaussian width,
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
recomputes distances *in-layer* from `pos` (so `jax.grad` can reach positions if forces ever
become a target, §3) — this precompute is here for inspection/notebooks, not required by
training.

## 5. `dataset.py` — molecules → examples, split, cache

- `load_molecule(fdf_path, spins)`: parses one molecule; finds its siblings by **filename stem**
  glob (`{stem}*.out`, `{stem}*{spin}*.wout`). Returns
  `{z, pos, energy, wannier=[{centers,radii} per spin]}`, the `wannier` list ordered to match
  `config wannier.spins`.
- `build_dataset(dir, spins)`: parses all `*.fdf`; computes `meta["max_centers"] =` the largest
  per-spin center count in the set — this **sizes the Wannier head** (`M`) so every molecule fits
  (CLAUDE.md §6, §9).
- `split(examples, val_frac, test_frac, seed)`: deterministic permutation → train/val/test.
- `save_cache`/`load_cache`: pickle the parsed examples + meta so you parse once.

---

## Failure modes & debugging
| Symptom | Cause | Where |
|---|---|---|
| `Unsupported AtomicCoordinatesFormat` | fractional/scaled coords need lattice vectors | `_coord_scale`; convert upstream or add lattice support |
| energy looks wrong / not converged | grabbed a non-final `siesta: Total` | must be `findall(...)[-1]`; §2 |
| way too many / garbage centers | matched per-iteration WF lines, not Final State | `_final_state_block` slicing; §3 |
| wrong element identities | `ChemicalSpeciesLabel` species index misread | `_species_to_z`; §2 |
| `FileNotFoundError` on `.wout` | the stem-glob assumption doesn't match real filenames | adjust the globs in `load_molecule`; **this is the one thing to confirm when real data lands** |
| head too small at train | `max_centers` in config < dataset max | `scripts/train.py` raises it to `meta["max_centers"]` |

## Rebuild-by-hand order
1. `units.py` constants.
2. fdf tokenizer (`_norm_key`, `_read_fdf_tokens`) → `_species_to_z`, `_coord_scale` →
   `parse_fdf`; `parse_energy` (last-match regex). Verify against a real `.fdf`/`.out` by hand.
3. `parse_wout` (Final-State slice + WF regex + `sqrt(spread)`).
4. `graph.build_graph` (meshgrid edges, rel-vec, distances).
5. `dataset` (stem matching, `max_centers`, split, cache). Then feed [model](model.md).
