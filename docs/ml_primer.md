# Machine-learning primer — the ideas this project uses, from zero

This explains the machine-learning concepts behind the code, assuming you can program but have
never trained a neural network. It's about *ideas*; the JAX/Flax *syntax* is in
[jax_flax_primer](jax_flax_primer.md). Terms in **bold** are also in the [glossary](glossary.md).

Related: [physics_primer](physics_primer.md) · [developer_guide](developer_guide.md) ·
[index](README.md).

---

## 1. What "learning" means here

We have ~118,000 molecules where we already know the right answers (from SIESTA, see
[physics_primer](physics_primer.md)): an energy, plus the Wannier centers and radii for each
spin. We want a function that, given **only** a new molecule's atoms and positions, produces
those answers. This is **supervised learning**, and since the answers are numbers it's
**regression**.

The function is a **neural network**: a big formula with many adjustable numbers called
**parameters** (or weights). "Training" means adjusting the parameters until the formula's
outputs match the known answers as closely as possible.

## 2. The loss: one number that says "how wrong"

To adjust parameters you need a single number measuring how wrong the network currently is: the
**loss**. Lower is better.

- **MSE (mean squared error):** the average of (prediction − truth)². Squaring makes all errors
  positive and punishes big errors more. Used for training the energy (`E` in the log, in eV²)
  and the Wannier positions (`pos`, in Å²).
- **MAE (mean absolute error):** the average of |prediction − truth|. Same units as the quantity
  itself, so it's easier to read. Used for *reporting* (e.g. "energy MAE 0.168 eV").
- With several targets the total loss is a **weighted sum**, e.g.
  `λ_E · energy_loss + λ_W · wannier_loss`. The weights (λ, "lambda") matter: the terms have
  different units, and if one term is huge it dominates training (§10).

## 3. How parameters get adjusted: gradient descent

- The **gradient** of the loss is, for every parameter, "how much the loss would change if this
  parameter grew a little". JAX computes it automatically (**automatic differentiation**,
  `jax.grad`), no matter how complicated the network is.
- **Gradient descent:** nudge every parameter a small step *against* its gradient, so the loss
  goes down. Repeat thousands of times.
- The step size is the **learning rate** (`train.lr`, 0.001 here). Too big and training becomes
  unstable or diverges to NaN; too small and it crawls. It usually changes during training
  according to a **schedule** (§14).
- **Adam** is the step rule we use: a refinement of gradient descent that adapts the step size
  for each parameter separately. It's the standard default.
- **Gradient clipping** (`train.grad_clip`): if the gradient is unusually large on some step,
  scale it down. That stops one bad batch from wrecking the parameters.

## 4. Batches, steps, epochs

- Computing the loss on all 94,000 training molecules for every step would be slow. Each step
  uses a **batch** (`train.batch_size`, e.g. 256 molecules).
- One **step** = one batch → one parameter update.
- One **epoch** = one pass over every training molecule (about 370 steps at batch size 256).
- Training runs for `train.epochs` epochs (300 here).
- In this code, a batch always contains molecules with the **same number of atoms**, so they can
  be stacked into one array and processed in parallel ([training](training.md) §1).

## 5. Train / validation / test, and over- vs underfitting

The data is split once, with a fixed random seed, into three non-overlapping groups:

| Group | Share | Used for |
|---|---|---|
| **train** | 80% | the only molecules the network learns from |
| **validation** (val) | 10% | checked every few epochs, never learned from; shows progress on unseen data |
| **test** | 10% | never touched during training; the honest final score |

- **Overfitting:** the network memorizes the training molecules and does worse on new ones.
  Signature: training error keeps falling while validation error rises.
- **Underfitting:** the network can't even fit the training data well. Signature: training and
  validation errors are about equal and both too high. (The first energy model here was
  underfitting: about 0.20 eV on train, 0.17 eV on val.)
- Report the **test** number. Scoring on training molecules always looks too good.

## 6. Building blocks of the network

- **Dense layer** (`nn.Dense`): output = input × weight matrix + bias vector. The weights and
  biases are parameters.
- **Activation function** (SiLU here): a simple nonlinear function applied after a dense layer.
  Without it, stacking dense layers would collapse into a single linear map, so nonlinearity is
  what lets a network represent complicated relationships.
- **MLP (multi-layer perceptron):** dense → activation → dense. The repo's small `_mlp` helpers
  ([model](model.md) §2) are exactly this.
- **Embedding** (`nn.Embed`): a lookup table with one learned vector per element. Carbon starts
  as one vector, oxygen as another, and training shapes what they mean.
- **Residual connection:** a layer computes a *change* that's added to its input (`h = h +
  update`), instead of replacing it. This makes deep stacks easier to train.

## 7. Graph neural networks (GNNs)

A molecule is naturally a **graph**: atoms are **nodes**, and here every pair of atoms is
connected by an **edge** (QM9 molecules are small, so all pairs is cheap).

A GNN works by **message passing**, repeated over several **layers**:
1. For every pair of atoms (i, j), compute a **message** from both atoms' current features and
   the distance between them.
2. Each atom **sums** the messages it receives (summing doesn't care about atom order).
3. Each atom **updates** its features from its old features plus that sum.

After a few layers each atom's features encode its whole chemical environment. A **readout**
then turns per-atom features into outputs: the energy is the **sum** of one number per atom.
Summing makes the energy grow with molecule size, as real energies do.

> **In the code:** `model/egnn.py` (message passing), `model/heads.py` (readouts).

## 8. Symmetry: invariance and equivariance

From [physics_primer](physics_primer.md) §8: rotating or moving a molecule must not change the
energy (**invariance**) and must move the Wannier centers the same way (**equivariance**).

Two ways to get that:
- **Data augmentation:** train on many randomly rotated copies and hope the network learns the
  rule. It only ever approximates it.
- **Build it into the architecture:** make the formula unable to break the rule. This project
  does that, using an **EGNN** (E(n)-equivariant graph neural network). Its two tricks:
  - messages only ever see **distances**, which don't change under rotation or translation;
  - positions are only ever updated by adding **relative vectors** (atom i minus atom j)
    multiplied by plain numbers computed from distances, so they rotate along with the molecule.

The consequence is checked by tests on an **untrained** network: a randomly initialized model
already passes the rotation, translation, permutation, and mirror tests
([symmetry](symmetry.md)). A network that fails them is wrong regardless of its accuracy.

## 9. Predicting a *set*: the Wannier centers

The energy is one number. The Wannier centers of a spin are a **set** of points with **no
natural order**. Nothing makes one center "the first". That breaks the obvious loss:

> Pair predicted center #1 with true center #1, #2 with #2, and so on: a correct prediction
> listed in a different order would be scored as badly wrong.

How this project scores a predicted set against the true set, per spin:

1. **The count is not predicted.** Each spin has (valence electrons) / 2 centers, a fact of the
   physics ([physics_primer](physics_primer.md) §5). Every atom proposes a few **candidate**
   centers (4 per spin by default), each with a **selection score**; the k highest-scoring
   candidates are the prediction.
2. **Matching:** pair each true center with one candidate so that the total distance is
   as small as possible. This is the **assignment problem**, solved exactly by the **Hungarian
   algorithm** (`scipy.optimize.linear_sum_assignment`). Example with 3 centers: if prediction A
   is closest to truth #2, B to #3, and C to #1, the matching pairs them that way regardless of
   output order.
3. **Loss on the pairs:** squared distance for each matched pair (positions), and squared
   difference of radii. The scores are trained as a yes/no classification: candidates the matching
   chose → "yes", the rest → "no".
4. **A smooth "guide" term:** each set is also drawn as a blurred cloud of fixed width, and the
   two clouds are compared. That needs no pairing, so it gives steady guidance early in training,
   when the matching still flips around.

**A lesson from this project:** the first version let the network choose each predicted blob's
width and strength. It learned to cover the true centers with a few wide, faint blobs, which
scored well on that loss and was useless in practice (center count right 0.1% of the time).
**Never let the model control the measuring stick.** If a quantity can be computed (the count)
or should be fixed (the guide's width), don't make it a prediction inside the loss.

### Lists vs sets: why candidates belong to atoms
A first version output a fixed **list** of slots, each computed from all atoms by one shared
rule. That rotates correctly, but think of methane: its four hydrogens look identical to the
network, so each slot gets the same contribution from all four, and those cancel — every slot
landed exactly on the carbon, for any weights. The true centers are on the four C–H bonds. The
general rule: **an equivariant function can't make its output less symmetric than its input**
(Xie & Smidt 2024, [refs](refs.md)). Turn methane by 120° and you get the same methane; the
network must then return the same list, and each slot must stay put under the turn — only points
on the rotation axis do. The physics asks for less: the *set* of centers must come back, but
individual centers may swap places. Outputs indexed by **atoms** have exactly that freedom: when
the turn swaps two hydrogens, their candidates swap too. Nothing random is added; the same
molecule always gives the same set. (For centers above and below the plane of a flat molecule,
candidates come in mirror pairs; see [model](model.md) §5.)

## 10. Balancing several targets

The total loss adds terms in different units (eV² for energy, Å² for positions). If one term is
1,000× bigger, nearly all of the gradient serves it and the other target barely trains. That
happened twice here: raw energies once made the energy loss ~10¹² and training diverged; later
the old Wannier term (~44) swamped the energy term (~0.06). Remedies, in the order this code uses
them:
- **Normalize targets** so terms are on similar scales (energy reference, radius divided by its
  spread, §11).
- **Weights** in the config (`loss.lambda_*`, `wannier.*_weight`), checked by reading each term
  in the epoch log line.
- (Not implemented yet) learning the weights automatically, e.g. uncertainty weighting
  (Kendall et al., 2018).

## 11. Baselines and normalization

- A **baseline** is the error of a trivial predictor. Always compare against one:
  - energy: "sum of one number per element" (composition baseline) → ~0.9 eV;
  - radius: "always predict the average radius" → 0.052 Å.
  A model that doesn't beat its baseline has learned nothing useful.
- **Normalization:** shift and scale targets so the network works with numbers of order 1. Here
  the energy is predicted as `composition baseline + learned correction`, and the radius starts
  at the dataset average with errors divided by the dataset spread.

## 12. Practical training issues

| Situation | What it means | Where to look |
|---|---|---|
| loss becomes `nan` | training diverged (steps too large, unnormalized targets) | [training](training.md) §5 NaN guard |
| first epoch takes ~30 min, later ones seconds | **compilation**: JAX turns the code into fast machine code once per input shape | [jax_flax_primer](jax_flax_primer.md) §6 |
| job killed by the time limit | expected on long runs | checkpoints save every few epochs; resubmit to **resume** |
| training error ≈ validation error, both high | underfitting | more epochs, bigger model, learning-rate schedule |
| validation error rising | overfitting | nothing to do by hand: the best epoch is kept automatically (§14), and early stopping can end the run |

Three techniques this project uses because they are standard where accuracy matters
([refs](refs.md)):
- **Averaging the weights (EMA).** Each step moves the weights a little in a noisy direction.
  Keeping a running average, `avg ← 0.99·avg + 0.01·weights`, and evaluating that average gives a
  steadier model than the last step. Interatomic-potential codes (NequIP, MACE) do this.
- **L1 vs squared losses.** A squared error lets a few large mistakes dominate: one 2 Å miss
  counts like 400 misses of 0.1 Å. The absolute (L1) error counts it like 20, so training keeps
  improving the many nearly-right predictions, which lowers the median. L1 is also what the
  reported MAE measures.
- **Non-maximum suppression (NMS).** When several predictions land on the same spot, keep the
  highest-scoring one and drop the others within a small radius. Object detectors use it; here
  it removes duplicate Wannier centers that an exactly symmetric model can't avoid producing
  ([model](model.md) §5).

## 13. Reading the metrics

| Metric | Meaning |
|---|---|
| MAE | typical error, in the quantity's units |
| RMSE | like MAE but weights big errors more |
| median | half the errors are smaller than this |
| p95 | 95% of errors are smaller than this; shows the bad tail |
| count accuracy | share of (molecule, spin) where predicted count = true count (100% by construction now; it checks the valence rule) |

To judge a number: compare it with the baseline, with the physical scale (chemical accuracy
0.043 eV; bonds are 1.1–1.5 Å long), and with the noise floor (Wannier labels: ~0.013 Å).

## 14. Learning-rate schedules, early stopping, and how many epochs

### Why change the learning rate during training
Every batch is a random sample, so each step contains some noise. Early in training the weights
are far from good and big steps make fast progress; the noise doesn't matter. Late in training,
big steps make the weights **bounce around** a good solution instead of settling into it, and the
loss only creeps down. That's what the first run here showed: the loss was still drifting at
epoch 300 with a constant learning rate.

**Cosine decay** (`train.lr_schedule: cosine`) lowers the rate smoothly along half a cosine wave,
from `lr` to 1% of `lr` at the last epoch:

| Point in training | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| learning rate (lr = 1e-3) | 0.001 | 0.00085 | 0.0005 | 0.00015 | 0.00001 |

Think of walking downhill in fog: big strides at first, small careful steps near the bottom.

**Warmup** (`train.warmup_epochs`, 5) comes first: the rate rises from 0 to `lr`. Adam's internal
estimates are unreliable in the very first steps, and full-size steps there can throw training off.

### Keeping the best epoch
The model is checked on the validation molecules every few epochs. Whenever the **validation
objective** improves, those weights are saved as "best"; later, worse epochs can't overwrite them.
The model you evaluate and use is always the best one seen, not simply the last.

### Early stopping
`train.patience` = how many validation checks in a row may pass without a new best before the run
stops. It saves time on runs that have stopped improving (the best weights are already saved).
It's used in sweep screening (§15); for a final long run it's usually off (0).

### So how many epochs?
Read the validation curve (the `val obj` column of the log over time):
- still improving at the end → the run was too short (or the schedule ended too early);
- flat for a long time → more epochs won't help; change something else (learning rate, model size,
  loss weights);
- rising while training loss falls → overfitting; the kept best epoch protects you.

With a cosine schedule, the epoch count also sets how long the slow, careful phase lasts. That's
why the final run in the sweep is long (1000 epochs) while screening runs are shorter (300).

## 15. Sweeps: comparing settings fairly

A **sweep** trains the same model several times with one setting changed per run. Three ideas make
its results trustworthy:
- **Seeds measure noise.** The random seed changes the initial weights and the batch order. Training
  the same config with seeds 0, 1, 2 shows how much results vary **by chance alone**. A different
  setting only "wins" if it beats the base by more than that spread.
- **One factor at a time.** Each variant changes a single setting, so a difference can be
  attributed to it.
- **Same test set for everyone** (`train.split_seed` fixed), and the winner chosen on
  **validation**, never on test. Then the test score of the winner is still an honest estimate.

This repo runs sweeps in **two stages**: short screening runs of every variant (with early
stopping), then one long run of the winner. How the runs are spread over many cluster jobs:
[sweeps](sweeps.md).

## 16. Training on several GPUs

**Data parallelism:** every GPU holds a full copy of the weights. Each batch is split into equal
slices (with 4 GPUs and batch 256, each GPU gets 64 molecules). Every GPU computes gradients on its
slice; the gradients are then **combined (all-reduce)** so that every GPU applies the same update
and the copies stay identical. The result equals training on one GPU with the whole batch, just
faster. JAX does this automatically once the arrays are marked as "split across devices"
([training](training.md) §4).

**Lagged assignment:** the Wannier loss needs a pairing of predicted to true centers, computed on
the CPU. To keep the GPU busy, the CPU pairs the *next* batch while the GPU trains on the current
one, using predictions made one step earlier. A pairing almost never changes in one step, so the
result is practically the same, and CPU and GPU no longer wait for each other.

## 17. Further reading
- Goodfellow, Bengio, Courville, *Deep Learning* (free at deeplearningbook.org), chapters 5–8.
- Sanchez-Lengeling et al., "A Gentle Introduction to Graph Neural Networks", distill.pub (2021).
- Satorras, Hoogeboom, Welling, "E(n) Equivariant Graph Neural Networks", ICML (2021).
- Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV (2020) — the
  matched set loss.
- Xie & Smidt, "Equivariant Symmetry Breaking Sets", TMLR (2024) — why a symmetric model must
  output a set, not a list, for symmetric inputs.
- The JAX documentation, "JAX 101" tutorials (jax.readthedocs.io).
- Full bibliography with what each source is used for: [refs](refs.md).
