# Sweeps — running many trainings in parallel

A **sweep** is a planned list of training runs that differ in one setting each (a seed, a loss
weight, a model size). This page explains how the repo runs a sweep across many cluster jobs at
once without the jobs getting in each other's way. Background: [ml_primer](ml_primer.md) §15
(why sweeps, seeds, two stages); the training loop itself: [training](training.md).

Code: `training/sweep.py` (logic), `scripts/sweep.py` (command line), `scripts/train-bunch.sh`
(what each cluster job runs), `configs/sweep.yaml` (the plan).

---

## 1. The problem it solves

Without it, 13 experiments mean 13 config files, 13 hand-submitted jobs, and keeping track of
which ones finished, which died at the time limit, and which need resubmitting. With it:
1. You describe the runs once in `configs/sweep.yaml`.
2. You submit the **same** job as many times as you have GPUs to spare (e.g. a job array of 6).
3. Each job takes the next unclaimed run, trains it, writes its report, and takes another, until
   the list is empty. A run whose job dies is picked up and **resumed** by another job.
4. `python -m scripts.sweep status configs/sweep.yaml` shows everything in one table.

## 2. Folder layout

```
runs/<sweep name>/
  queue/queued/<run>      ← empty "marker" files; the folder a marker sits in IS the run's state
  queue/running/<run>
  queue/finished/<run>
  queue/failed/<run>
  <run>/config.yaml            the run's full config (base + overrides)
        params.msgpack         best parameters        (see training.md §7)
        params.last.msgpack    resumable state
        params.best.json       best epoch + validation metrics
        params.progress.json   current epoch
        train.log              the run's log (also printed to the job's output)
        owner.json             which Slurm job/host/process is running it
        report/                accuracy report on the test split (after training)
        result.json            test-set headline numbers
        error.txt              traceback, if it failed
```

## 3. How jobs coordinate without locks

Many jobs on different machines share only the filesystem. Two operations on a single filesystem
are **atomic**: they happen completely or not at all, and if two processes try them at once,
exactly one succeeds:
- `os.mkdir(path)`: fails if the folder already exists;
- `os.rename(a, b)`: moving a file; if two processes rename the same file, only the first finds it.

The queue uses only these, so no job ever waits on a lock (the "lock-free" property):

| Action | Operation | Why it's safe |
|---|---|---|
| **create a run** | `mkdir runs/<sweep>/<run>`, then write `config.yaml` and the `queued` marker | only the job whose `mkdir` succeeds writes anything, so runs are never duplicated, even when 10 jobs start together |
| **claim a run** | `rename queue/queued/<run> → queue/running/<run>` | exactly one job wins; the others get "file not found" and try the next run |
| **finish / fail** | `rename running → finished` (or `failed`) | |
| **give a run back** | `rename running → queued` | |

A plain counter file ("next run number") would **not** be safe: reading, adding one, and writing
back are three steps, and two jobs can interleave them and take the same number.

## 4. Detecting dead jobs

A running job **heartbeats**: every 60 s it updates the timestamp of its `running/<run>` marker.
Before claiming, every worker checks the running runs:
1. If `owner.json` has a Slurm job id, ask Slurm: `squeue -h -j <id>`. Not listed (or "Invalid
   job id") means the job is gone.
2. If Slurm can't be asked, fall back to the heartbeat: no update for `stale_minutes` (sweep file,
   default 30) means gone.
3. A gone owner's run is moved back to `queued/` with the same atomic rename (only one worker
   wins). The next claimer **resumes** it from `params.last.msgpack`.

A worker never reclaims its own run.

## 5. Time limits and preemption

`train-bunch.sh` starts Python with `exec`, so Slurm's signals reach it directly. On **SIGTERM**
(sent at the time limit) or **SIGUSR1**, the worker finishes the current step, saves the run's
state, moves it back to `queued/`, and exits; another job continues it right away. To get a
warning a few minutes before the limit, add `#SBATCH --signal=B:USR1@300` (5 minutes early).

## 6. The two stages

**Stage 1 (screening):** the runs listed in `configs/sweep.yaml`. `stage1_overrides` applies to
all of them: 300 epochs, early stopping after 6 validations (30 epochs) without improvement. Poor
variants stop early and free their GPU.

**Stage 2 (final run):** when a worker finds nothing to claim and every stage-1 run is finished
or failed, it creates run `final`. That's the best stage-1 run's config (lowest **validation**
objective, never the test score) plus `stage2_overrides`: 1000 epochs, no early stopping, the
full cosine schedule. Creation uses the same atomic `mkdir`, so exactly one worker creates it;
that worker then trains it. `promoted_from.txt` records the source run.
`python -m scripts.sweep promote configs/sweep.yaml` creates it immediately (e.g. if you don't
want to wait for a slow run).

Selecting the winner on validation and reporting on test keeps the test score honest: the test
molecules never influence any choice.

## 7. The planned stage-1 runs (`configs/sweep.yaml`)

All runs share `split_seed: 0`, so all are scored on the **same test molecules**.

**Current plan: `wannier_v3`** (the per-atom candidate head, [model](model.md) §5). The sweep file
is self-contained: its `base:` is the full run config written inline (a copy of
`configs/gpu.yaml`), so nothing outside it is read. (`base:` may also be a path to a config file.)
Every override must name a key that exists in the base; a base written before a key was added
fails at start-up with `no such config key`.

| Runs | What changes | Question it answers |
|---|---|---|
| `base_s0`, `base_s1` | seed 0 / 1 | Run-to-run spread with the new head |
| `select_w0.3`, `select_w3` | `wannier.select_weight` (base 1) | How hard to push candidate selection |
| `singles3` | `wannier.singles_per_atom` 3 (base 2) | Do atoms need more candidates? |
| `pairs2` | `wannier.pairs_per_atom` 2 (base 1) | More ± pairs per atom (several off-plane centers on one atom)? |

**Previous plan: `wannier_v2`** (global-slot head, finished; results in `runs/wannier_v2/`; the
winner's `final` reached test energy 0.058 eV, radius 0.022 Å, centers 0.36 Å):

| Runs | What changes | Question it answers |
|---|---|---|
| `base_s0`, `base_s1`, `base_s2` | seed 0 / 1 / 2 | How much do results vary by chance alone? Differences smaller than this spread aren't real. |
| `guide_w0.03`, `guide_w0.3` | `wannier.guide_weight` (base 0.1) | Does the smooth guide help, or does it blur centers? |
| `guide_s0.2`, `guide_s0.5` | `wannier.guide_sigma` Å (base 0.3) | Sharper vs smoother guide |
| `radius_w0.3`, `radius_w3` | `wannier.radius_weight` (base 1) | Trade-off between radius and position accuracy |
| `wannier_l0.3`, `wannier_l3` | `loss.lambda_wannier` (base 1) | Trade-off between energy and Wannier accuracy |
| `lr_5e-4` | `train.lr` (base 1e-3) | Is the learning rate too aggressive? |
| `big_model` | `model.hidden_dim` 192, `n_layers` 6 | Does a bigger network help? |

To change the plan, edit the file. Adding a run with a new name adds it to the queue the next time
any worker starts; existing runs (by name) are never duplicated or changed. To re-run a variant
with different settings, give it a new name.

## 8. Commands

```bash
bash scripts/train-bunch.sh [configs/sweep.yaml]               # what each job runs (the worker)
.venv/bin/python -m scripts.sweep status  configs/sweep.yaml   # table of all runs
.venv/bin/python -m scripts.sweep requeue configs/sweep.yaml RUN   # retry a failed run
.venv/bin/python -m scripts.sweep promote configs/sweep.yaml   # start stage 2 now
```

An sbatch file for a 6-job array:
```bash
#!/bin/bash
#SBATCH --job-name=qm9_sweep
#SBATCH --array=1-6                 # 6 jobs work through the queue together
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6           # threads for the per-step matching + batch producer
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300         # 5-minute warning before the time limit -> clean hand-back
#SBATCH --output=sweep_%A_%a.out
cd /path/to/QM9_pred
bash scripts/train-bunch.sh
```
Prerequisite on the login node: `uv sync --group gpu --group eval`.

## 9. Reading `status`

```
sweep 'wannier_v2' (runs/wannier_v2)
run             state        epoch   val obj    val E  val ctr  val rad   test E test ctr  job
base_s1 *       finished   185/300    0.3120   0.1410   0.1902   0.0398   0.1432   0.1931  4711
...
final           running    212/1000   0.2801   ...                                         4720
```
- Sorted by **val obj** (lower = better). `*` marks the best finished stage-1 run, the one promoted
  to `final`.
- `epoch` is current / planned; a finished run with fewer epochs than planned stopped early.
- `val E`, `val ctr`, `val rad`: validation energy error (eV), center error (Å, mean of the spins),
  radius error (Å) at the best epoch. `test …`: the same on the test split, from the run's report.
- Compare a variant's `val obj` with the spread of the three seed runs: smaller differences are
  noise.

## 10. Isolation guarantees
- Every run writes only inside its own folder; the only shared writes are marker renames.
- `dataset.pkl` and the configs are only read.
- Each job gets its own GPU(s) from Slurm.
- Compiled programs go to a shared cache (`.jax_cache/`); JAX writes entries safely, and a cache
  miss just means compiling again.

## 11. Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| a run stays `running` but its job is gone | `squeue` unreachable from the nodes and the heartbeat hasn't expired yet | wait `stale_minutes`, or lower it in the sweep file |
| a run is `failed` | exception during training/report | read `runs/<sweep>/<run>/error.txt`, fix, then `requeue` |
| `final` never appears | some stage-1 run is still queued or running | check `status`; or `promote` |
| jobs exit immediately with "nothing left to claim" | everything is finished/running | expected |
| a new run in the file doesn't show up | no worker started since you edited the file | start a worker (it creates missing runs) |
| `no GPU visible to JAX` | CUDA jax missing or a CPU node | `uv sync --group gpu --group eval` on the login node; or `ALLOW_CPU=1` |
| `WARNING: '<run>' was reclaimed by another job while running` | the owner was wrongly judged dead (e.g. `squeue` errors plus an expired heartbeat) | raise `stale_minutes`; the run's files may need a fresh start |
