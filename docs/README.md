# Implementation notes — how this repo is built

`../CLAUDE.md` says **why** every decision was made (it's the design rationale, and the source of
truth). These docs say **how** it's implemented: the actual functions, the JAX/Flax syntax, the
tensor shapes, the equivariance proof mapped to real lines, and the failure modes. Written for
the engineer who designed it and may one day rebuild it by hand or debug it at 3am.

## Reading order

1. **[jax_flax_primer.md](jax_flax_primer.md)** — the JAX/Flax vocabulary used everywhere
   (`init`/`apply`, `@nn.compact`, `jit`, pytrees, `value_and_grad`, broadcasting). Read first;
   the rest link back to it instead of repeating.
2. **[data.md](data.md)** — parsing SIESTA `.fdf`/`.out`/`.wout` and building the graph.
3. **[model.md](model.md)** — the EGNN layer line-by-line, the equivariance proof, the heads.
   *The centerpiece.*
4. **[losses.md](losses.md)** — energy MSE + the closed-form Gaussian-cloud Wannier loss.
5. **[training.md](training.md)** — the optax loop and JAX training idioms.
6. **[symmetry.md](symmetry.md)** — the transforms and the equivariance tests (the correctness
   claim).

## The pipeline, end to end

```
 files on disk                         data/ (deterministic, no learning)
   .fdf ─ parse_siesta ─┐
   .out ─ parse_siesta ─┼─► {Z, pos, energy, wannier targets} ─► graph (fully connected)
  .wout ─ parse_wout ───┘                    │
                                             ▼                 model/ (all the learning)
                        ElementEmbedding → h │ pos → x
                                  EGNNBackbone (h invariant, x equivariant)
                                             │
                              ┌──────────────┴──────────────┐
                        EnergyHead (Σ scalars)        WannierHead (equivariant point set)
                                             │
                                             ▼                 losses/
                          λ_E·energy_loss + λ_W·wannier_cloud_loss
                                             │
                                             ▼                 training/
                              optax loop → params → checkpoint
                                             │
                                             ▼                 symmetry/ + tests/
                        rotate/translate/permute/mirror must hold (on the untrained net!)
```

Each arrow into `model/` crosses the **pipeline boundary** (CLAUDE.md §3): `data/` builds the
graph by a fixed rule; `model/` turns it into predictions with all the weights. The graph is the
interface — neither side reaches across.

## Where each CLAUDE.md section is implemented

| CLAUDE.md § | Topic | Code | Doc |
|---|---|---|---|
| §3 | graph / pipeline boundary | `data/graph.py`, `data/dataset.py` | [data](data.md) |
| §4 | EGNN backbone | `model/egnn.py` | [model](model.md) |
| §5 | energy + Wannier heads | `model/heads.py` | [model](model.md) |
| §6 | Gaussian-cloud loss | `losses/wannier_cloud.py` | [losses](losses.md) |
| §8 | symmetry verification | `symmetry/transforms.py`, `tests/test_equivariance.py` | [symmetry](symmetry.md) |
| §9 | data parsing | `data/parse_siesta.py`, `data/parse_wout.py`, `data/units.py` | [data](data.md) |
| §10 | build/train order | `training/` | [training](training.md) |

## The thin directories (no dedicated doc)

- **`configs/`** — YAML hyperparameters (`default.yaml`, `energy_only.yaml`). Code reads these;
  no magic numbers in code (CLAUDE.md §11). `default.yaml` sizes the model, loss weights, lr,
  `max_centers`; `energy_only.yaml` sets `λ_W = 0` for the energy-only ablation.
- **`scripts/`** — thin CLI wiring only (`build_dataset`, `train`, `evaluate`); run with
  `python -m scripts.<name>`. Logic lives in the packages, not here.
- **`notebooks/`** — presentation/demos on a hardcoded methane molecule (no data needed): the
  equivariance showcase, pipeline walkthrough, loss visualization, prediction inspection. Shared
  helpers in `notebooks/demo_utils.py`. Nothing importable depends on these (CLAUDE.md §11).
- **`tests/`** — `test_parsers.py`, `test_equivariance.py` (the gate — see
  [symmetry](symmetry.md)), `test_loss.py`, `test_training_smoke.py`. Run `uv run pytest -q`.

## If you rebuild from scratch
Follow the build order in `../CLAUDE.md §10`; each doc ends with a "Rebuild-by-hand" section for
its piece. The one hard rule: **get the equivariance tests passing on the untrained model before
writing any training code** ([symmetry](symmetry.md)).
