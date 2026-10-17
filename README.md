# QM9 → Wannier Center Predictor

A neural network that takes a molecule's **atoms and their positions** and predicts what an
expensive quantum-chemistry calculation (SIESTA + Wannier90) would produce: its **Wannier
centers**, the points where its electrons sit (one per bond / lone pair and spin), each with a
**radius**, separately for **spin up** and **spin down**. The model optimizes these alone; the
total energy was dropped because training both in one network traded one off against the other
(CLAUDE.md §1).

A calculation that takes minutes per molecule becomes a millisecond prediction. The network is an
**E(3)-equivariant graph neural network (EGNN)**: rotating, moving, mirroring, or relabeling a
molecule is *guaranteed* to rotate, move, mirror, or leave unchanged the predictions, by the
architecture itself.

**New to the project?** Start with [`docs/README.md`](docs/README.md). It gives a reading order
that starts from zero physics and zero machine learning (physics primer → ML primer → developer
guide → one doc per code directory) plus a glossary. Design decisions and their reasons are in
[`CLAUDE.md`](CLAUDE.md); academic sources in [`docs/refs.md`](docs/refs.md).

---

## 1. Requirements
- Linux or macOS, Python 3.14 (installed automatically by `uv`).
- [`uv`](https://docs.astral.sh/uv/) for environments: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- For training: an NVIDIA GPU is recommended (one is enough). CPU works but is slower.
- The data: SIESTA/Wannier90 output folders (`.fdf` and one `.wout` per spin, per molecule).

## 2. Install
```bash
git clone <repo> && cd QM9_pred
uv sync --group eval              # code + plotting for the accuracy report
uv sync --group eval --group gpu  # same, plus CUDA jax, on a machine with an NVIDIA GPU
uv run pytest -q                  # all tests should pass (~20 s)
```
`uv sync --group X` installs exactly the groups you list and **removes the others**. On a GPU
machine always include `--group gpu`, or the CUDA version of jax gets uninstalled.

| Group | Adds | Needed for |
|---|---|---|
| *(default)* | jax, flax, optax, scipy, pyyaml, pytest | everything |
| `gpu` | `jax[cuda12]` (bundles its own CUDA; needs only the NVIDIA driver) | GPU training |
| `eval` | matplotlib | `scripts/evaluate.py` (the accuracy report) |
| `notebook` | jupyter kernel, matplotlib, nbconvert | the demo notebooks |

## 3. The workflow, step by step

Locally, prefix commands with `uv run`. **On a cluster compute node use `.venv/bin/python`
instead of `uv run`** (see §4). Every script has `--help`.

### Step 1 — Build the dataset (once)
```bash
uv run python -m scripts.build_dataset /path/to/parent --out dataset.pkl
```
- Give one or more folders. Each is searched **recursively**. The `.fdf` and `.wout` files of a
  molecule may sit in different subfolders (e.g. `done/`, `wout_files/`). SIESTA's `.out` is not
  needed. An existing `dataset.pkl` built with energies still works (the energy is ignored);
  keep using it so the test molecules stay the same.
  They're matched by **molecule id**, the filename up to the first `.` (e.g. `C2F3N3O_133481`).
- Each molecule needs `<id>.fdf`, `<id>…out`, and one `.wout` per spin whose name contains `up`
  or `down`.
- Incomplete or unconverged molecules are **skipped and counted**, not fatal (a few hundred of
  ~118k here).
- Output: `dataset.pkl` (all parsed molecules, ~118k). Rebuilding it changes the train/val/test
  split, so retrain after any rebuild.

### Step 2 — Train (one run, or a whole sweep)
**One run:**
```bash
uv run python -m scripts.train --config configs/gpu.yaml --dataset dataset.pkl --out params.msgpack
```
(On a cluster: `bash scripts/train-gpu.sh` from your sbatch file, §4.)
- Splits the molecules 80 / 10 / 10 into train / validation / test using `train.split_seed`
  (fixed, so every run is tested on the same molecules).
- Uses every GPU the job can see (data parallel); the log says `data-parallel over N device(s)`.
- Learning rate: 5-epoch warmup, then cosine decay to 1% (`train.lr_schedule`).
- Every `ckpt_every` epochs it validates and prints one line: training loss terms, then
  `val obj` (lower = better), per-spin center error and radius error
  (Å), and `* new best` when it improves.
- Files next to `--out`: `params.msgpack` (**best** epoch: what evaluate/predict use),
  `params.last.msgpack` (resumable state), `params.best.json` (best epoch's metrics),
  `params.progress.json`.
- **Resumes automatically** from `params.last.msgpack`, identically to an uninterrupted run, so a
  time-out means resubmit. On Slurm's time-limit signal it saves first. To start fresh, move the
  `params*` files away.
- `train.patience` > 0 stops a run early when validation stops improving.
- The first epoch is slow (~30 min of one-time compilation); the compile cache (`.jax_cache/`)
  makes later jobs skip it.

**A sweep** (planned variants, then a long final run of the winner), spread over many jobs:
```bash
bash scripts/train-bunch.sh                                  # submit this from N jobs (§4)
.venv/bin/python -m scripts.sweep status configs/sweep.yaml  # progress table
```
Each job claims the next unclaimed run, trains it, writes its test report, and takes another. A
run whose job dies resumes in another job. Results land in `runs/<sweep name>/<run>/`. Full
guide: [docs/sweeps.md](docs/sweeps.md).

### Step 3 — Measure accuracy
```bash
uv run python -m scripts.evaluate --config configs/gpu.yaml --params params.msgpack
uv run python -m scripts.evaluate --config configs/gpu.yaml --params params.msgpack \
    --data-dir /path/to/new_molecules      # optional: your own test set, parsed fresh
```
- **Pass the same config the model was trained with**: it fixes the model size and the split.
- Default: scores the **test split**, ~11.8k molecules training never saw. `--subset
  val|train|all` scores other parts; train vs test shows memorization.
- Writes `accuracy_report/` (or `--out-dir`):
  - `summary.txt`: per-spin Wannier center error (MAE, RMSE, median, p95, share within 0.1 Å,
    share over 1 Å), radius error, count accuracy; a rotation symmetry check. Also printed.
  - `molecules.csv`: per molecule, the per-spin center and radius errors.
  - `centers.csv`: per center, SIESTA's position and radius next to the matched prediction.
  - `plots/`: one image per plot, spin up and down pooled — `center_error_hist.png`,
    `radius_parity.png`, `molecule_center_error_hist.png`.
- How to read these numbers (what is good or bad): [docs/ml_primer.md](docs/ml_primer.md) §13.

### Step 4 — Predict new molecules
```bash
uv run python main.py molecule.fdf --config configs/gpu.yaml --params params.msgpack
uv run python main.py folder_of_fdfs/ --config configs/gpu.yaml --params params.msgpack --out predictions.json
```
Only `.fdf` geometry is needed. One file prints a readable block; a folder (only its `.fdf`
files, not recursive) writes one JSON array:
```json
[{"input": "mol1.fdf",
  "wannier": {"up":   [{"position": [0.01, 0.02, 0.70], "radius": 0.85}, ...],
              "down": [{"position": [0.01, 0.02, 0.70], "radius": 0.85}, ...]}}]
```
Each spin lists (valence electrons) / 2 centers. A file that fails to parse becomes
`{"input": …, "error": …}` instead of stopping the run.

## 4. Running on a Slurm cluster
1. On the **login node** (needs internet): `uv sync --group gpu --group eval`.
2. Put resources in **your own** sbatch file and call the repo's launcher:
   ```bash
   #!/bin/bash
   #SBATCH --partition=<gpu partition>
   #SBATCH --gres=gpu:1            # more GPUs also work (batch_size must divide evenly)
   #SBATCH --cpus-per-task=4       # CPUs prepare batches and run the matching each step
   #SBATCH --mem=32G
   #SBATCH --time=12:00:00
   #SBATCH --output=train_%j.out
   #SBATCH --error=train_%j.err
   cd /path/to/QM9_pred
   bash scripts/train-gpu.sh       # or: bash scripts/train.sh  (CPU, configs/default.yaml)
   ```
3. Edit the `--config`, `--dataset`, `--out` lines inside the launcher if needed.

**For a sweep**, the same file with a job array and the sweep launcher:
```bash
#SBATCH --array=1-6               # 6 jobs work through configs/sweep.yaml together
#SBATCH --signal=B:USR1@300       # 5-min warning before the time limit -> clean hand-back
...
bash scripts/train-bunch.sh       # status: .venv/bin/python -m scripts.sweep status configs/sweep.yaml
```
Failed run: read `runs/<sweep>/<run>/error.txt`, then
`python -m scripts.sweep requeue configs/sweep.yaml <run>`. Start stage 2 early:
`python -m scripts.sweep promote configs/sweep.yaml`.

- **Never call `uv run` inside a job.** Compute nodes have no internet; `uv` blocks forever
  (symptom: timeout, 0% CPU, empty log). The launchers call `.venv/bin/python` directly.
- `train-gpu.sh` stops with a clear error if JAX can't see a GPU. Don't `module load cuda`:
  `jax[cuda12]` brings its own CUDA.
- Evaluation and prediction are light; run them as a small job too, with
  `.venv/bin/python -m scripts.evaluate …`.

## 5. Configuration (`configs/*.yaml`)
| Key | Meaning |
|---|---|
| `model.hidden_dim`, `model.n_layers` | network width and depth |
| `model.update_coords`, `model.edge_gate` | EGNN options from its QM9 setup: freeze the coordinate track; soft edge gate (both off by default) |
| `model.rbf.*` | how distances are expanded into features |
| `wannier.max_centers` | padding for the TRUE centers per spin (≥ the largest molecule's count; 28 here) |
| `wannier.singles_per_atom`, `wannier.pairs_per_atom` | candidate centers each atom proposes per spin (default 2 singles + 1 ± pair) |
| `wannier.select_weight` | weight of the candidate-selection term |
| `wannier.position_loss` | `l1` (distance, default) or `mse` (squared distance) for matched centers |
| `wannier.nms_radius` | Å; selected centers closer than this are merged, and candidates this close to a matched one count as positives (0 = off; default 0.3) |
| `wannier.spins` | `[up, down]` |
| `wannier.guide_sigma` | fixed width (Å) of the smooth guide term in the Wannier loss |
| `wannier.radius_weight`, `wannier.guide_weight` | weights of the radius and guide terms |
| `train.lr`, `train.grad_clip` | peak learning rate, gradient-clipping limit |
| `train.ema_decay` | moving average of the weights used for validation and checkpoints (0.99; 0 = off) |
| `train.lr_schedule`, `train.warmup_epochs`, `train.lr_end_frac` | `cosine` (warmup, then decay to `lr·lr_end_frac`) or `constant` |
| `train.patience` | early stopping: validations without improvement before stopping (0 = off) |
| `train.batch_size`, `train.epochs`, `train.ckpt_every` | batch size (32 CPU / 256 GPU), epochs, checkpoint interval |
| `train.seed` | initial weights + batch order |
| `train.split_seed`, `train.val_frac`, `train.test_frac` | the train/val/test split (keep `split_seed` fixed) |

Files: `default.yaml` (CPU), `gpu.yaml` (GPU, larger batches), `sweep.yaml` (the planned sweep: base config, variants, stage
settings; see [docs/sweeps.md](docs/sweeps.md)).

## 6. Tests and notebooks
```bash
uv run pytest -q                               # whole suite
uv run pytest tests/test_equivariance.py -q    # the symmetry guarantee alone
uv sync --group eval --group notebook
uv run --with jupyterlab jupyter lab notebooks/   # or open them in VS Code's notebook editor
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # re-run headless
```
| Notebook | Shows |
|---|---|
| `equivariance_showcase.ipynb` | rotate/move/mirror/relabel a molecule; outputs follow exactly, even untrained |
| `pipeline_walkthrough.ipynb` | `.fdf` → parsing → graph → network → prediction |
| `wannier_loss.ipynb` | how the Wannier loss works: count from composition, matched positions and radii per spin |
| `inspect_predictions.ipynb` | predicted vs true (fake data, or your `params.msgpack` + `dataset.pkl` if present) |

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| cluster job times out with 0% CPU and an empty log | you used `uv run` in the job; use the launchers |
| `no GPU visible to JAX` | `uv sync --group gpu --group eval` on the login node; check `nvidia-smi` |
| `LLVM compilation error: Cannot allocate memory` | don't change the batching in `training/train.py` (see `docs/training.md` §1) |
| `train loss nan` / `NON-FINITE loss …` | training diverged; see `docs/training.md` §5 |
| `params.msgpack doesn't match …` in evaluate | evaluate with the config used for training |
| `no valence charge for element(s)` | a new element; see `docs/developer_guide.md` §5c |
More: [docs/developer_guide.md](docs/developer_guide.md) §7.

## 8. Project layout
```
data/       parsing, file matching, the split, center counts          (no learning)
model/      the EGNN and its Wannier output                            (all the learning)
losses/     the Wannier loss (matched, per spin) and the cloud overlap
training/   training loop, metrics, accuracy report
symmetry/   transforms used by the symmetry tests
scripts/    command-line entry points + Slurm launchers
configs/    every tunable number
tests/      pytest suite
notebooks/  demos only
docs/       documentation, from zero background
```

## 9. Status
- **v1 model** (first full training, Gaussian-cloud Wannier loss): test energy MAE **0.168 eV**
  (composition baseline 0.913 eV). Wannier centers were not usable (center error 0.61 Å, count
  right 0.1%).
- **v2** (sweep `wannier_v2`: center count from composition, Hungarian-matched loss, global
  slot head): test energy MAE **0.058 eV**, radius MAE **0.022 Å**, count 100%, but center error
  stuck at **0.36 Å**. Cause: global slots must sit on a molecule's symmetry axes/planes (all of
  methane's slots on the carbon; nothing out of plane in flat molecules), measured as 19% of
  predicted centers piled up.
- **v3** (sweep `wannier_v3`: per-atom candidate centers + ± pairs, selection logits): test
  energy MAE **0.075 eV**, center MAE **0.127 Å** (median **0.056 Å**), radius MAE **0.011 Å**.
  The center mean is dominated by duplicates: 2.6% of centers are > 1 Å off, mostly next to a
  duplicate selection.
- **Current code** (sweep `wannier_v4`): **Wannier only** — the energy target, head and loss
  are removed (CLAUDE.md §1: energy and Wannier traded off in one network). NMS selection with
  NMS-consistent targets, L1 distance loss, weight EMA; EGNN QM9 options as sweep variants
  (CLAUDE.md §5–§6, [docs/refs.md](docs/refs.md)). v1–v3 numbers above include energy because
  those models predicted it. v3 checkpoints still load (their energy weights are ignored), and
  re-evaluating one applies NMS (`nms_radius` defaults to 0.3 Å for older configs). **Awaiting
  the v4 sweep.**
- Not implemented: molecule ids saved in `dataset.pkl`; multi-node training.
