# Implementation notes — how this repo is built

`../CLAUDE.md` says **why** every decision was made (it's the design rationale, and the source of
truth). These docs say **how** it's implemented, and start from zero: someone without a physics
or machine-learning background should be able to read them in order and then develop on the
code alone. Commands for running everything are in the top-level [README](../README.md).

## Reading order

**Background (start here if the science or ML is new to you):**
1. **[physics_primer.md](physics_primer.md)** — atoms, valence electrons, DFT/SIESTA, total
   Wannier centers and radii, spin, the files on disk, why symmetry matters.
2. **[ml_primer.md](ml_primer.md)** — loss, gradient descent, batches/epochs, train/val/test,
   graph neural networks, invariance/equivariance, predicting a set (matching), loss balance.
3. **[developer_guide.md](developer_guide.md)** — repo map, setup, the development loop, rules you
   must not break, recipes for common changes, tests, debugging, cluster use.
4. **[glossary.md](glossary.md)** — every term, one line each. Keep it open.

**Implementation, one doc per code directory:**
5. **[jax_flax_primer.md](jax_flax_primer.md)** — the JAX/Flax syntax used everywhere
   (`init`/`apply`, `@nn.compact`, `jit`, `vmap`, `value_and_grad`, sharding, schedules).
6. **[data.md](data.md)** — parsing `.fdf`/`.wout`, matching files by molecule id, the
   center count per spin, the split.
7. **[model.md](model.md)** — the EGNN layer line by line, the equivariance proof, the heads.
   *The centerpiece.*
8. **[losses.md](losses.md)** — the Wannier loss (count from composition,
   Hungarian-matched positions and radii per spin, fixed-width cloud guide), and why the first
   Wannier loss failed.
9. **[training.md](training.md)** — the training loop step by step: precomputed batches, the
   CPU/GPU pipeline, multi-GPU, learning-rate schedule, validation and best epoch, resume, early
   stopping, the evaluation/report code.
10. **[sweeps.md](sweeps.md)** — running many trainings in parallel across cluster jobs: the
    lock-free queue, dead-job recovery, the two-stage plan, reading `status`.
11. **[symmetry.md](symmetry.md)** — the transforms and the equivariance tests (the correctness
    claim).
12. **[refs.md](refs.md)** — every academic source the project uses, and what for.

## The pipeline, end to end

```
 files on disk                         data/ (deterministic, no learning)
   .fdf ─ parse_siesta ─┐
                        ├─► {Z, pos, wannier targets} ─► graph (fully connected)
  .wout ─ parse_wout ───┘                    │
                                             ▼                 model/ (all the learning)
                        ElementEmbedding → h │ pos → x
                                  EGNNBackbone (h invariant, x equivariant)
                                             │
                                             │
                          WannierHead (per-atom candidates → set; NMS picks k)
                                             │
                                             ▼                 losses/
           matched positions + radii + selection + fixed-width guide, per spin (Wannier only)
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
| §5 | Wannier head | `model/heads.py` | [model](model.md) |
| §6 | Wannier count + loss | `data/electrons.py`, `losses/wannier_match.py`, `losses/wannier_cloud.py` | [losses](losses.md) |
| §8 | symmetry verification | `symmetry/transforms.py`, `tests/test_equivariance.py` | [symmetry](symmetry.md) |
| §9 | data parsing | `data/parse_siesta.py`, `data/parse_wout.py`, `data/units.py` | [data](data.md) |
| §10 | build/train order | `training/` | [training](training.md) |

## The thin directories (no dedicated doc)

- **`configs/`** — YAML hyperparameters; code reads these, no magic numbers in code (CLAUDE.md
  §11). `default.yaml` (CPU, batch 32), `gpu.yaml` (same model, batch 256),
  `sweep.yaml` (the planned two-stage sweep). Keys: model size (`model.*`), Wannier settings and
  loss weights (`wannier.*`), training (`train.*`).
- **`scripts/`** — thin CLI wiring only: `build_dataset`, `train`, `evaluate` (accuracy report;
  logic in `training/report.py`), `sweep` (work / status / requeue / promote; logic in
  `training/sweep.py`), run with `python -m scripts.<name>`; plus the launchers to call from your
  own Slurm script: `train.sh` (CPU), `train-gpu.sh` (one GPU run), `train-bunch.sh` (sweep worker).
- **`runs/`** (created by sweeps, not in git) — one folder per sweep: the queue markers and one
  folder per run ([sweeps](sweeps.md) §2). **`.jax_cache/`** — shared compiled programs.
- **`notebooks/`** — presentation/demos on a hardcoded methane molecule (no data needed): the
  equivariance showcase, pipeline walkthrough, loss visualization, prediction inspection. Shared
  helpers in `notebooks/demo_utils.py`. Nothing importable depends on these (CLAUDE.md §11).
- **`tests/`** — `test_parsers.py`, `test_equivariance.py` (the gate — see
  [symmetry](symmetry.md)), `test_loss.py`, `test_training_smoke.py`, `test_metrics.py` +
  `test_report.py` (accuracy report), `test_electrons.py` (center count per spin) +
  `test_wannier_match.py` (the matched Wannier loss) + `test_set_head.py` (what the per-atom
  candidate head can represent). Run `uv run pytest -q`.

## If you rebuild from scratch
Follow the build order in `../CLAUDE.md §10`; each doc ends with a "Rebuild-by-hand" section for
its piece. The one hard rule: **get the equivariance tests passing on the untrained model before
writing any training code** ([symmetry](symmetry.md)).
