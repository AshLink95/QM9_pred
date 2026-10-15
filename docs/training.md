# `training/` — the training loop, step by step

How the weights actually get trained. Background first if needed: [ml_primer](ml_primer.md)
(loss, gradient descent, epochs, schedules, data parallelism) and [jax_flax_primer](jax_flax_primer.md)
(`jit`, `vmap`, sharding syntax). Running many trainings in parallel is in [sweeps](sweeps.md).
Rationale of record: `../CLAUDE.md` §6, §10.

Files: `train.py` (the loop), `metrics.py` + `report.py` (evaluation), `sweep.py` (the parallel
queue, see [sweeps](sweeps.md)), `loss.py` (a small converter for evaluation helpers).

---

## 0. The big picture

```
dataset.pkl ──split (split_seed)──► train 80% / val 10% / test 10%
train molecules ──stack_buckets──► arrays grouped by atom count (done once)
every epoch:
   producer thread: pick batches ──► move to GPU(s) ─┐
   main loop:   fwd(next batch) ──► CPU pairs centers (Hungarian, threads)
                step(current batch, pairing) ──► updated weights
every ckpt_every epochs: validate ──► print line ──► save "last" (+ "best" if improved)
```

Entry points: `scripts/train.py` (one run, launched by `scripts/train.sh` / `train-gpu.sh`) and
`training/sweep.py` (many runs, launched by `scripts/train-bunch.sh`). Both call
`train_with_status(cfg, examples, ckpt_path, should_stop)`.

## 1. Data preparation, done once — `stack_buckets`

- The split uses `train.split_seed` (default 0), **not** `train.seed`. `seed` controls weight
  initialization and batch order; keeping the split seed fixed means runs with different seeds
  are scored on the **same test molecules**.
- Training molecules whose parsed Wannier count differs from valence electrons / 2 are dropped
  with a warning (none expected).
- Molecules are grouped by **atom count** and each group is stacked into NumPy arrays
  (`z`, `pos`, `energy`, Wannier `centers`/`radii` padded to `max_centers`, the per-spin count
  `k`). About 120 MB for the full dataset.
- Why group by atom count: a batch is one array, and arrays need equal shapes. Same atom count
  means no padding inside the model.

## 2. Batches, every epoch — `epoch_batches`

- Each group is shuffled and cut into `batch_size` chunks. The last chunk of a group is **topped
  up** with random molecules from the same group, so **every batch has exactly `batch_size`
  molecules**. That gives:
  - one compiled program per atom count (JAX compiles once per shape; [primer §6](jax_flax_primer.md));
  - batches that always split evenly across GPUs (§4).
- Batch order is shuffled across groups. The shuffle uses `default_rng([seed, epoch])`, so a
  resumed run replays exactly the same batches.
- A **producer thread** (`_prefetch`) slices the arrays and copies the next two batches to the
  devices while the main loop is busy, so the GPU never waits for batch preparation.

> History: the first version trained one molecule per step and compiled a new program for every
> combination of atom count and Wannier count, which ran out of compiler memory (`LLVM compilation
> error: Cannot allocate memory`). Fixed shapes solved that.

## 3. One training step, and the CPU/GPU overlap

The Wannier loss needs a **pairing** of predicted to true centers (the Hungarian algorithm,
[losses](losses.md) §3a), which runs on the CPU. To keep the GPU busy the loop is pipelined:

```python
nxt_out = fwd(params, next_batch)                     # GPU: predict the NEXT batch (no gradient)
params, opt_state, loss, aux = step(params, opt_state, cur_batch, perm)   # GPU: train on current
perm = pairing(next_batch, nxt_out["wannier"]["centers"])   # CPU, at the same time as `step`
```

- `fwd` and `step` are `jax.jit`-compiled. JAX dispatches them **asynchronously**: Python
  continues immediately and the GPU executes in order. So while the GPU runs `step`, the CPU
  computes the next pairing on a thread pool sized to the job's CPUs (`SLURM_CPUS_PER_TASK`).
  SciPy releases Python's GIL, so the threads really run in parallel.
- **Lagged assignment:** the pairing for a step is computed from predictions made one update
  earlier. Pairings almost never change in one step, so this costs nothing measurable and buys
  the overlap. Cost: one extra forward pass per step.
- No host callback sits inside compiled code, which keeps both programs cacheable on disk (§8).
- Per-step losses stay on the device and are fetched **once per epoch**. Fetching every step
  would force the CPU to wait for the GPU each time.
- `lambda_wannier: 0` (`energy_only.yaml`) skips `fwd` and the pairing entirely.

`step` computes `λ_E · energy MSE + λ_W · Wannier loss` (`_batch_loss`), its gradient, clips it
(`train.grad_clip`), and applies Adam with the current learning rate.

## 4. Several GPUs — data parallelism

```python
mesh = jax.make_mesh((n_devices,), ("data",), axis_types=(AxisType.Auto,))
data_sh = NamedSharding(mesh, P("data"))   # split along the batch axis
repl    = NamedSharding(mesh, P())         # full copy on every device
```
- Weights and optimizer state are **replicated**. Each batch (and its pairing) is **sharded**:
  with 4 GPUs and `batch_size` 256, each GPU gets 64 molecules.
- `jit` sees the shardings and splits the computation automatically. Each GPU computes gradients
  on its slice, and XLA inserts an **all-reduce** that combines them. Every GPU then applies the
  same update, so the replicas stay identical. The result equals single-GPU training up to float
  rounding (`tests/test_multidevice.py` checks this).
- `AxisType.Auto` asks XLA to work out the intermediate shardings itself; this JAX version's
  default (`Explicit`) would require annotating every operation.
- `batch_size` must be divisible by the number of devices (clear error otherwise). The log prints
  `data-parallel over N device(s), M molecules per device per step`.
- Only the GPUs of one job (one node). For more throughput across nodes, run a sweep
  ([sweeps](sweeps.md)).

## 5. Learning-rate schedule — `make_lr`

`train.lr_schedule: cosine` (the default in the provided configs) builds
`optax.warmup_cosine_decay_schedule`:
- **Warmup:** the rate rises linearly from 0 to `lr` over `warmup_epochs` (5). Adam's internal
  averages are unreliable in the first steps; small steps there avoid a bad start.
- **Cosine decay:** then it falls along half a cosine wave to `lr · lr_end_frac` (1% of `lr`) at
  the last planned epoch. Big steps early make fast progress; small steps late let the weights
  settle instead of bouncing around ([ml_primer](ml_primer.md) §3).
- Steps per epoch are fixed (§2), so the total length is known in advance. The schedule's step
  counter lives inside the optimizer state, so a resumed run continues the curve exactly.
- `lr_schedule: constant` keeps `lr` fixed (the old behavior).

## 6. Validation, best epoch, early stopping

Every `ckpt_every` epochs (and at the last epoch), `_validate` predicts every validation molecule
once, pairs centers on the host, and computes:

| Value | Meaning |
|---|---|
| **val objective** | `λ_E · E_MAE/σ_E + λ_W · (center MAE + radius_weight · radius MAE/r_std)`, on the **selected** centers (top logits with NMS, exactly as at prediction time). It uses mean absolute errors whatever the training loss forms (`l1`/`mse`), so sweep runs that differ in loss form are comparable (σ_E and r_std come from the shared train split). **Lower = better; this picks the best epoch.** |
| `E-MAE` | validation energy error, eV |
| `center up/down` | per-spin matched center error, Å |
| `radius` | matched radius error, Å |

The line looks like:
```
epoch   40  lr 9.71e-04  train 0.4123 (E 0.0210  W 0.3913 = pos 0.3101, radius 0.6012, select 0.0950, guide 0.2210) | val obj 0.3811  E-MAE 0.1512 eV  center up/down 0.2101/0.2098 A  radius 0.0412 A  * new best  (63.2s)
```
- `* new best`: the validation objective improved, so the parameters are saved as the best.
- **Which parameters:** validation, `params.msgpack` and the returned model use an exponential
  moving average (EMA) of the weights, `ema ← d·ema + (1−d)·weights` after every step
  (`train.ema_decay`, default 0.99; 0 = the raw weights). Averaging over the last ~100 steps
  smooths out step-to-step noise; NequIP and MACE validate and ship their EMA weights the same way
  ([refs](refs.md)). Training itself continues from the raw weights; both are saved in
  `params.last.msgpack`, so resume stays exact.
- **Early stopping** (`train.patience`, 0 = off): after `patience` validations in a row without
  a new best, training stops (`early stop: no improvement for N evaluations (best @ epoch E)`).
  The best parameters are already on disk, so nothing is lost.

## 7. Files, resume, stopping

Next to `--out` (default `params.msgpack`):

| File | Contents | Used by |
|---|---|---|
| `params.msgpack` | **best** parameters (lowest val objective) | `evaluate`, `main.py` |
| `params.last.msgpack` | full training state: parameters, optimizer state (incl. schedule step), next epoch, best objective, epochs since best | resume |
| `params.best.json` | the best epoch and its validation metrics | you, `sweep status` |
| `params.progress.json` | current epoch, total, best so far | `sweep status` |

- **Resume:** if `params.last.msgpack` exists, is finite, and matches the model, training
  continues at the saved epoch with the same batches and learning-rate value; it's identical to
  an uninterrupted run (`tests/test_schedule.py`). An incompatible or NaN file is ignored with a
  WARNING. To start over, move these files away.
- Files are written via a temp file + rename, so a kill mid-write never leaves a broken file.
- **NaN guard:** if any step in an epoch produced a non-finite loss, the epoch is undone (weights
  revert to the epoch start), the state is saved, and training stops with
  `NON-FINITE loss at epoch E step S: E-loss … W-loss …`.
- **Graceful stop:** `scripts/train.py` and the sweep worker catch SIGTERM / SIGUSR1 (Slurm sends
  these at the time limit). The loop notices between steps, saves `last`, and returns `stopped`;
  resubmitting resumes.
- `train()` returns the best parameters; `train_with_status()` also returns the status:
  `finished`, `early_stopped`, `stopped`, or `diverged`.

## 8. Shared compile cache

The launchers set `JAX_COMPILATION_CACHE_DIR` (default `.jax_cache/` in the repo). JAX stores
compiled programs there and reuses them in later processes with the same shapes and devices, so
the slow first-epoch compile (~30 min on the full dataset) is paid once instead of in every job
and resubmission. Set `JAX_COMPILATION_CACHE_DIR=` (empty) to disable.

## 9. Evaluation code

- **`metrics.py:match_wannier`** scores one molecule and spin: the first `k` candidates after
  `model.heads.select_centers` (highest logits first) are matched one-to-one to the true centers (`scipy.optimize.linear_sum_assignment`), returning per-pair
  distance (Å), radius error, and both counts. `wannier_summary` aggregates MAE / RMSE /
  median / p95 / radius MAE / count accuracy.
- **`report.py:evaluate_checkpoint`** is the whole accuracy report in one call (used by
  `scripts/evaluate.py` and by the sweep after each run): load the checkpoint, pick the molecules
  (test split by `split_seed`, or fresh folders), predict in batches (`predict_all`), compare
  (`build_report`), write CSVs, `summary.txt`, and one PNG per plot in `plots/` (`PLOTS`; spins
  pooled, no count plot since the count is computed).

---

## Failure modes & debugging
| Symptom | Cause | Where |
|---|---|---|
| `train.batch_size=… must be divisible by the N visible devices` | batch doesn't split evenly | use a multiple of the GPU count (e.g. 256 for 1/2/4/8) |
| first epoch ~30 min, later seconds | one-time compilation | expected; the compile cache (§8) skips it next time |
| `ShardingTypeError … out_sharding` | mesh created in Explicit mode | keep `axis_types=(AxisType.Auto,)` (§4) |
| `LLVM compilation error: Cannot allocate memory` | too many distinct shapes | keep fixed-size batches (§2) |
| E-loss huge, then NaN | energy reference removed | [model](model.md) §5 |
| `NON-FINITE loss …` | divergence; the line shows which term | lower `lr`, check λ balance |
| val objective stops improving early | too high a learning rate, or real plateau | try `lr` 5e-4; cosine schedule; see [ml_primer](ml_primer.md) §12 |
| `WARNING: … doesn't match this model` | `last` file from another architecture/config | expected after model changes; starts fresh |
| only GPU 0 busy | `jax.devices()` sees one GPU | request `--gres=gpu:N`; check the `data-parallel over …` line |
| job hangs with 0 CPU, empty log | `uv run` inside the job | use `scripts/train*.sh` |

## Rebuild-by-hand order
1. `stack_buckets` + `epoch_batches` (fixed-size, top-up, seeded per epoch). Check shapes.
2. `_batch_loss` with a given pairing; host `assign` ([losses](losses.md)).
3. `fwd` / `step` under `jit`; the pipelined loop (§3); per-epoch loss fetch.
4. Mesh + shardings (§4).
5. `make_lr` (§5).
6. `_validate`, best/last files, early stopping, resume, NaN guard, graceful stop (§6–7).
7. Train `energy_only.yaml` first, then the full loss.
