# QM9 → Wannier + Energy Predictor

An **E(3)-equivariant graph neural network** that predicts converged SIESTA electronic-structure
outputs — total energy and Wannier centers (with radius + spin) — from atomic geometry alone
(`Z` + positions). Symmetry (rotation / translation / permutation / mirror) is **guaranteed by
the architecture** (EGNN), not approximated by data augmentation.

The design decisions and their rationale live in [`CLAUDE.md`](CLAUDE.md) — that file is the
source of truth; read it before changing anything load-bearing. For **how it's implemented**
(EGNN line-by-line, the equivariance proof, JAX/Flax syntax, failure modes), see
[`docs/`](docs/README.md).

## Stack

JAX + Flax (model) · Optax (optimization) · closed-form Gaussian-cloud loss for the unordered
Wannier center set. No `e3nn` (L≤1 targets only). Python ≥3.14.

```bash
uv sync                      # install deps
uv run pytest tests/ -q      # parsers, equivariance gate, loss, end-to-end training
```

## Pipeline (mirrors the directory layout)

| Stage | Module | Role |
|---|---|---|
| Parse | `data/parse_siesta.py`, `data/parse_wout.py` | `.fdf`/`.out` → Z, pos, energy; `.wout` Final State → centers, radii, spin |
| Graph | `data/graph.py` | fully-connected edges, distances, relative vectors (deterministic, no learning) |
| Model | `model/egnn.py`, `model/heads.py`, `model/model.py` | EGNN backbone + energy/Wannier heads (all learning) |
| Loss | `losses/energy.py`, `losses/wannier_cloud.py` | squared-error energy + ordering-free Gaussian-cloud Wannier loss (split by spin) |
| Train | `training/` | optax loop, combined weighted loss, metrics |
| Symmetry | `symmetry/transforms.py` + `tests/test_equivariance.py` | the correctness guarantee, tested explicitly |

`data/` is deterministic prep; `model/` is all learning. The graph is the interface — the model
never parses files, the loss never builds graphs.

## Usage

```bash
uv run python -m scripts.build_dataset  root1/ root2/ ...  --out dataset.pkl   # 1+ roots, each recursed
uv run python -m scripts.train          --config configs/default.yaml --dataset dataset.pkl --out params.msgpack
uv run python -m scripts.evaluate       --dataset dataset.pkl --params params.msgpack
uv run python main.py                   path/to/molecule.fdf --params params.msgpack   # one molecule → stdout
uv run python main.py                   path/to/fdf_dir/ --params params.msgpack --out predictions.json  # batch → one .json
```

`build_dataset` walks each root recursively and pools all molecules into one `dataset.pkl` —
pass as many roots as you like. Files are matched by **molecule id** (the filename up to the
first `.`, e.g. `C2F3N3O_133481`), so the `.fdf`, `.out`, and `.wout` for a molecule may live in
separate roots (`done/*.fdf`, `out_files/*.out`, `wout_files/*.wout`). `.wout` files supply the
converged Wannier targets for training; molecules missing an `.out` or a spin `.wout` are
skipped with a count.

`main.py` predicts from `.fdf` geometry alone (no `.wout` needed — those are training labels).
A directory reads only its `.fdf` files and writes one JSON array of
`{input, energy, wannier}` records. Energy-only ablation (build-order step 5):
`--config configs/energy_only.yaml`.

## Demo notebooks (`notebooks/`)

Presentation + onboarding; run on a hardcoded methane molecule, no real data required.
Install the group and open with `uv run --group notebook jupyter lab`.

| Notebook | Shows |
|---|---|
| `equivariance_showcase.ipynb` | the flagship: rotate/translate/mirror/permute → energy fixed, centers transform exactly (on the **untrained** net) |
| `pipeline_walkthrough.ipynb` | `.fdf` → parse → graph → EGNN → prediction, stage by stage (the data/model boundary) |
| `wannier_loss.ipynb` | the Gaussian-cloud loss: zero on identical sets, order-invariant, spin-split, smooth gradient everywhere |
| `inspect_predictions.ipynb` | predicted vs true energy/centers (fabricated data now; auto-loads `params.msgpack`/`dataset.pkl` when present) |

Shared helpers live in `notebooks/demo_utils.py`. Re-run headless with
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.

## Status

Full pipeline is wired and tested end-to-end on fabricated molecules. **Real SIESTA samples are
not yet present** — parsers are written to the format spec (`CLAUDE.md §9`); hand-verification
against real `.fdf`/`.out`/`.wout` and real training are the remaining steps once data lands.
The file-stem matching convention in `data/dataset.py` is the one assumption to confirm against
real filenames.
