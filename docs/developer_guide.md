# Developer guide — working on this repo without help

For someone taking over the code. Read [physics_primer](physics_primer.md) and
[ml_primer](ml_primer.md) first if the science or the machine learning is new to you; this page
is about the *work*: where things live, how to run and change them, what you must not break, and
how to debug. Commands for running everything are in the top-level [README](../README.md).

---

## 1. Repo map

| Path | What it is | Learns? |
|---|---|---|
| `data/parse_siesta.py` | reads `.fdf` (atoms, positions) and `.out` (energy) | no |
| `data/parse_wout.py` | reads `.wout` (Wannier centers, spreads) | no |
| `data/dataset.py` | finds files by molecule id across folders, builds/loads `dataset.pkl`, the train/val/test split | no |
| `data/electrons.py` | valence table; Wannier centers per spin = valence electrons / 2 | no |
| `data/graph.py` | all-pairs graph (used by notebooks; the model recomputes distances itself) | no |
| `data/units.py` | Å / Bohr / eV / Ry conversions | no |
| `model/embeddings.py` | element → feature vector | **yes** |
| `model/egnn.py` | the message-passing layers (the symmetry-preserving core) | **yes** |
| `model/heads.py` | energy readout + Wannier slots (centers, radii) per spin | **yes** |
| `model/model.py` | assembles everything into `QM9Model`; `model_from_config` | **yes** |
| `losses/wannier_match.py` | the Wannier loss: Hungarian-matched positions + radii + fixed-width guide, per spin | — |
| `losses/wannier_cloud.py` | closed-form Gaussian-cloud overlap (the guide term) | — |
| `losses/energy.py` | energy MSE (reference; training computes it inline) | — |
| `training/train.py` | the training loop: precomputed batches, CPU/GPU pipeline, multi-GPU, LR schedule, validation, best/last checkpoints, resume, early stopping | — |
| `training/sweep.py` | the parallel sweep: lock-free run queue, dead-job recovery, two stages | — |
| `training/metrics.py` | evaluation metrics (energy MAE, Wannier matching) | — |
| `training/report.py` | the accuracy report (CSVs, summary, figure) | — |
| `training/loss.py` | small converter used by evaluation helpers | — |
| `symmetry/transforms.py` | rotate / translate / mirror / permute, for the symmetry tests | — |
| `scripts/build_dataset.py` | CLI: parse files → `dataset.pkl` | — |
| `scripts/train.py` | CLI: config + `dataset.pkl` → `params.msgpack` | — |
| `scripts/evaluate.py` | CLI: accuracy report | — |
| `scripts/sweep.py` | CLI: `work` / `status` / `requeue` / `promote` for a sweep | — |
| `scripts/train.sh`, `scripts/train-gpu.sh` | launchers for one run (CPU / GPU), called from your Slurm script | — |
| `scripts/train-bunch.sh` | launcher for a sweep worker; submit it from many jobs | — |
| `main.py` | CLI: predict for one `.fdf` or a folder of them | — |
| `configs/*.yaml` | every tunable number (no magic numbers in code); `sweep.yaml` = the planned sweep | — |
| `runs/` (not in git) | sweep outputs: queue markers + one folder per run | — |
| `tests/` | pytest suite (see §6) | — |
| `notebooks/` | demos/presentation only; nothing imports them | — |
| `CLAUDE.md` | the design decisions and *why* — read before changing anything load-bearing | — |
| `docs/` | how it's implemented (this folder) | — |

The split between `data/` (fixed rules, no learning) and `model/` (all learning) is deliberate:
the model never reads files, and the data code never contains trainable parameters.

## 2. Setup

```bash
# local machine, once
uv sync --group eval --group notebook     # code deps + plotting + notebooks
uv run pytest -q                          # should print "N passed"
```
- Python 3.14, managed by `uv` (`pyproject.toml` + `uv.lock` pin every version).
- **On the cluster:** run `uv sync` on the **login node** (it needs internet). In jobs, call
  `.venv/bin/python` directly, **never `uv run`**: compute nodes have no internet, and `uv` blocks
  forever trying to sync (a 24 h timeout with 0% CPU and an empty log).
- `uv sync --group X` removes packages from groups you don't list. On a GPU setup always run
  `uv sync --group gpu --group eval`, or the CUDA jax gets uninstalled.

## 3. The development loop

1. **Change code** (and its config keys, if it adds a tunable number).
2. **Run the tests locally:** `uv run pytest -q` (≈20 s). Never skip this: the equivariance tests
   catch architecture mistakes before you spend GPU hours.
3. **Train on the cluster**: one run with `scripts/train-gpu.sh` (or `train.sh` for CPU), or a
   whole sweep with `scripts/train-bunch.sh` submitted from several jobs
   ([sweeps](sweeps.md)). Watch the log: per-epoch loss terms, validation line, `* new best`.
4. **Evaluate** with `scripts/evaluate.py` on the test split; read `summary.txt` and
   `accuracy.png`.
5. **Compare** with the previous model's report (keep old `params.msgpack` files under a new
   name, e.g. `params_v1.msgpack`, and old `accuracy_report/` folders).
6. **Record** design changes in `CLAUDE.md` and the matching `docs/` page.

Don't run full training on a laptop or shared login node: it takes ~1 hour on a GPU.

## 4. Rules you must not break

| Rule | Why | Where |
|---|---|---|
| Messages in `EGNNLayer` use only invariants (element features, distances) | otherwise rotation/translation symmetry breaks | [model](model.md) §4 "do NOT alter" |
| Positions are only updated by relative vectors × plain numbers | same | [model](model.md) §2 |
| Wannier centers are built from the equivariant track (`centroid + Σ A·(x − centroid)`) | an ordinary MLP output can't rotate with the molecule | [model](model.md) §5 |
| Energy is a **sum** over atoms | energy must scale with molecule size | [model](model.md) §5 |
| The Wannier count is computed, never predicted; loss widths are fixed, never predicted | a model that controls these "games" the loss (measured) | [losses](losses.md) §2–3 |
| Spins are never mixed in any loss or metric | up and down are separate physical sets | [losses](losses.md) §3 |
| Everything tunable lives in `configs/` | reproducible experiments | `configs/*.yaml` |
| Units: Å and eV inside the pipeline | conversions bite once, in `data/units.py` | [data](data.md) §1 |
| No `uv run` inside cluster jobs | it hangs offline | §2 |
| The equivariance tests must pass on the **untrained** model | proves the symmetry is structural | [symmetry](symmetry.md) |

## 5. Recipes

### 5a. Change a hyperparameter
Edit or copy a file in `configs/`. Model size: `model.hidden_dim`, `model.n_layers`. Training:
`train.lr`, `train.batch_size`, `train.epochs`, `train.grad_clip`. Loss balance:
`loss.lambda_energy`, `loss.lambda_wannier`, `wannier.radius_weight`, `wannier.guide_weight`,
`wannier.guide_sigma`. Point `scripts/train.sh` / `train-gpu.sh` (the `--config` line) at the
file. A model trained with a config must be **evaluated with the same config** (sizes and split
seed come from it).

### 5a′. Learning-rate schedule and epochs
`train.lr_schedule: cosine` (warmup `train.warmup_epochs`, final rate `lr · train.lr_end_frac`) or
`constant`. The schedule spans `train.epochs`, so changing the epoch count changes the curve.
`train.patience` > 0 enables early stopping. How to pick: [ml_primer](ml_primer.md) §14.

### 5b. Energy-only run (sanity check or ablation)
Use `configs/energy_only.yaml` (`lambda_wannier: 0`, which skips the Wannier loss entirely).

### 5c. Add a new element (e.g. sulfur)
1. Add it to `VALENCE` in `data/electrons.py`; the value must equal the valence charge of the
   SIESTA pseudopotential used for the data (S → 6 for a standard pseudopotential; check yours).
2. Make sure `model.n_elements` exceeds its Z (it's 100).
3. Rebuild `dataset.pkl` with molecules containing it, then retrain. The model has never seen
   the element otherwise and its predictions would be meaningless.

### 5d. Add a new predicted quantity (e.g. the dipole moment)
1. **Parse** it into each example dict (`data/parse_*.py`, `data/dataset.py:load_molecule`),
   rebuild `dataset.pkl`.
2. **Batch** it: add a field in `training/train.py:_stack`.
3. **Predict** it: add a head in `model/heads.py` of the right symmetry type (a scalar must come
   from invariant features, a vector from the equivariant track), wire it in `model/model.py`.
4. **Loss:** add a term in `training/train.py:_batch_loss` with a weight in the config, and log
   it in the epoch line.
5. **Report:** add it to `training/report.py`.
6. **Tests:** extend `tests/test_equivariance.py` (it must transform correctly) and add a loss
   test.
7. **Docs:** CLAUDE.md §1/§5 and the matching `docs/` page.

### 5e. Change the Wannier loss
All of it is in `losses/wannier_match.py`; the weights are in `configs/` (`wannier.*`). Keep
`tests/test_wannier_match.py` passing (zero on a reordered truth, spins never mix, gradients
finite) and add a test for any new property. Read [losses](losses.md) §2 first: it explains the
failure the current design avoids.

### 5f. Read a new input file format
Write a parser that returns the same example dict
(`{z, pos, energy, wannier: [{centers, radii} per spin]}`, Å and eV). Nothing downstream changes.
For prediction only, add a branch in `main.py` that turns the new file into `z, pos`.

### 5g. Add or change data
Rebuild with `scripts.build_dataset` (it lists skipped molecules). **Any rebuild changes the
train/val/test split**, so retrain before evaluating; never evaluate an old model on a rebuilt
`dataset.pkl`.

### 5h′. Run a sweep
1. Edit `configs/sweep.yaml` (base config, runs and their overrides, stage-1/stage-2 overrides).
2. On the login node: `uv sync --group gpu --group eval`.
3. Submit an sbatch job array that calls `bash scripts/train-bunch.sh` (template in
   [sweeps](sweeps.md) §8).
4. Watch: `.venv/bin/python -m scripts.sweep status configs/sweep.yaml`.
5. Failed run: read `runs/<sweep>/<run>/error.txt`, fix, then
   `python -m scripts.sweep requeue configs/sweep.yaml <run>`.
6. Add a variant later: add a new entry (new name) to `runs:`; the next worker to start queues it.
7. Results: each run has `report/` (test split) and `params.msgpack` (best). The stage-2 `final`
   run is the one to use.

### 5i. Use several GPUs for one run
Request them (`#SBATCH --gres=gpu:4`) and keep `train.batch_size` divisible by the GPU count
(256 works for 1, 2, 4, 8). Nothing else changes; the log shows
`data-parallel over 4 device(s), 64 molecules per device per step`. More GPUs per run only pays off
if the per-step matching keeps up: give the job a few CPUs per GPU.

### 5h. Compare two models
Keep both checkpoints and run `scripts.evaluate` for each with its own `--out-dir`; compare
`summary.txt` side by side. Same `dataset.pkl` and config split settings ⇒ same test molecules.

## 6. Tests

`uv run pytest -q` runs everything in ~20 s. One file: `uv run pytest tests/test_metrics.py -q`.

| Test file | Guards |
|---|---|
| `test_equivariance.py` | **the symmetry guarantee** (rotate, translate, permute, mirror) on an untrained model |
| `test_parsers.py` | `.fdf` / `.out` / `.wout` parsing on small synthetic files |
| `test_electrons.py` | center count per spin; errors on unknown elements / odd electron counts |
| `test_wannier_match.py` | the Wannier loss: zero on a reordered truth, known offsets, spin separation, gradients |
| `test_loss.py` | the Gaussian-cloud overlap |
| `test_metrics.py` | the evaluation matching metric |
| `test_training_smoke.py` | the training loop runs and reduces error on fake molecules |
| `test_batches.py` | fixed-size batches, one atom count per batch, every molecule seen each epoch, reproducible order |
| `test_schedule.py` | cosine schedule shape, early stopping keeps the best, resume equals an uninterrupted run, stop request saves state |
| `test_multidevice.py` | training on 2 (fake) devices gives the same result as on 1 |
| `test_sweep.py` | the queue: no duplicate runs, exclusive claims under concurrency, dead-owner reclaim, preemption, exactly-once promotion, a tiny sweep end to end |
| `test_report.py` | the accuracy report and `scripts/evaluate.py` end to end |
| `test_main_batch.py` | `main.py` folder mode and its JSON output |

`tests/_fake.py` makes fake molecules that obey the physics (even electron counts, the right
number of centers). Use it for new tests.

## 7. Debugging playbook

| Symptom | Likely cause | Fix |
|---|---|---|
| job runs 24 h, 0% CPU, empty log | `uv run` inside the job | use `scripts/train*.sh` |
| `LLVM compilation error: Cannot allocate memory` | too many distinct array shapes being compiled | keep the bucketing/padding in `training/train.py` ([training](training.md) §1) |
| first epoch very slow, later fast | one-time compilation | expected |
| `train loss nan` / `NON-FINITE loss …` | divergence | the log line shows which term; see [training](training.md) §5 |
| `WARNING: … doesn't match this model` | checkpoint from an older architecture | expected after model changes; it starts fresh |
| `train-gpu.sh`: `no GPU visible to JAX` | CUDA jax not installed, or no GPU on the node | `uv sync --group gpu --group eval` on the login node; check `nvidia-smi` |
| only 1 of several GPUs busy | JAX sees one device | check the `data-parallel over N device(s)` line and `--gres` |
| `batch_size … must be divisible by the N visible devices` | batch doesn't split evenly | use a multiple of the GPU count |
| sweep run stuck in `running` with its job gone | `squeue` unavailable, heartbeat not yet expired | wait `stale_minutes` (sweep file) |
| sweep run in `failed` | exception during training/report | `runs/<sweep>/<run>/error.txt`, then `requeue` |
| `evaluate`: params don't match config | evaluated with a different config than training | pass the training config |
| `ValueError: no valence charge for element(s)` | new element | §5c |
| Wannier error high but energy fine | loss balance, or a Wannier-loss regression | read the per-term epoch log; [losses](losses.md) failure table |

## 8. Cluster cheat sheet

Your own sbatch carries the resources; the repo's shell scripts only run Python:
```bash
#!/bin/bash
#SBATCH --job-name=train_QM9
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
cd /path/to/QM9_pred
bash scripts/train-gpu.sh
```
- One GPU per run is the efficient default; several GPUs per run also work (§5i). Give each job
  a few CPUs (4–8): they prepare batches and run the Hungarian matching every step.
- Add `#SBATCH --signal=B:USR1@300` for a clean save 5 minutes before the time limit.
- Training resumes from `params.last.msgpack` if it exists, so a time-out just means resubmit. To
  start over, move the old files away (`params*.msgpack`, `params.*.json`).
- For many runs, use a job array with `scripts/train-bunch.sh` ([sweeps](sweeps.md)).
- Memory use is ~4–5 GB; 32 GB is plenty.

## 9. Where decisions are recorded
- **`CLAUDE.md`**: what was decided and why, including what was tried and rejected (e.g. the first
  Wannier loss, with its measured failure). Change it when you change a design decision.
- **`docs/`**: how each part is implemented, failure modes, rebuild order.
- **`README.md`**: how to run things.
