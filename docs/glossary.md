# Glossary

Short definitions of the physics, machine-learning, and tooling terms used in this repo. Longer
explanations: [physics_primer](physics_primer.md), [ml_primer](ml_primer.md),
[jax_flax_primer](jax_flax_primer.md).

| Term | Meaning |
|---|---|
| **Adam** | The optimizer (step rule) used for training; adapts the step size per parameter. |
| **All-reduce** | Combining (summing) a value across all GPUs so each ends up with the total; used for gradients in multi-GPU training. |
| **Assignment problem** | Pair the items of two equal-size sets so the total cost is minimal; solved by the Hungarian algorithm. |
| **Asynchronous dispatch** | JAX returns from a jitted call immediately and the GPU works in the background; Python waits only when it reads a value. |
| **Atomic number (Z)** | Which element an atom is: H 1, C 6, N 7, O 8, F 9. The model's input is `z` + positions. |
| **Atomic operation** | A file-system action that completes entirely or not at all, with one winner if several processes race (`os.rename`, `os.mkdir`); the basis of the run queue. |
| **Baseline** | The error of a trivial predictor (e.g. always predicting the mean radius). A model must beat it. |
| **Batch** | The group of molecules used for one training step (e.g. 256). Here a batch always has one atom count. |
| **Best / last checkpoint** | `params.msgpack` = parameters of the best validation epoch (used for evaluation); `params.last.msgpack` = full state for resuming. |
| **Bohr** | Atomic length unit, 0.529 Å. Appears in `.fdf` files that give coordinates in Bohr (converted to Å on parsing). |
| **Bucket** | All training molecules with the same number of atoms; batches are cut from buckets. |
| **Candidate** | One proposed Wannier center: each atom proposes a few per spin (singles and ± pairs); the k with the highest selection logits are the prediction. |
| **Charbonnier loss** | `sqrt(d² + ε²)`: the distance itself, smoothed at zero so it has a gradient everywhere; the default Wannier position loss (`wannier.position_loss: l1`). |
| **Checkpoint** | Saved model parameters (`params.msgpack`); training resumes from it. |
| **Chiral** | A molecule that differs from its mirror image. |
| **Closed-shell** | All electrons paired; equal numbers of spin-up and spin-down. Every molecule here. |
| **Compilation (jit)** | JAX translating Python code into fast machine code, once per input shape; why the first epoch is slow. |
| **Compile cache** | Folder (`.jax_cache/`) where JAX stores compiled programs so later jobs skip the slow first-epoch compilation. |
| **Config** | A YAML file in `configs/` holding every tunable number for a run. |
| **Converged** | A calculation that has stopped changing between iterations; only converged values are targets. |
| **Core electrons** | Inner electrons that don't take part in chemistry; not simulated (replaced by pseudopotentials). |
| **Cosine decay** | Learning-rate schedule that falls smoothly along half a cosine wave from `lr` to a small final value. |
| **Data parallelism** | Splitting each batch across GPUs, each with a full copy of the weights, and combining their gradients. |
| **Dense layer** | `output = input × W + b`; the basic trainable layer. |
| **DFT (density functional theory)** | The approximate quantum-mechanics method SIESTA uses to compute energies and electrons. |
| **Dipole moment** | A vector describing how charge is separated in a molecule; computable from nuclei + Wannier centers. |
| **Early stopping** | Ending a run after `patience` validation checks without improvement; the best weights are already saved. |
| **EGNN** | E(n)-equivariant graph neural network: the architecture, built so rotations/translations are respected exactly. |
| **EMA (exponential moving average of weights)** | A running average of the weights, `avg ← d·avg + (1−d)·weights` after each step (`train.ema_decay`); validation and checkpoints use it. |
| **Embedding** | A learned lookup table: one vector per element. |
| **Epoch** | One pass over all training molecules. |
| **Equivariant** | Transforms *with* the input: rotate the molecule and the Wannier centers rotate the same way. |
| **`.fdf` / `.out` / `.wout`** | SIESTA input (atoms), SIESTA log (not read), Wannier90 log (centers + spreads, one per spin). |
| **Gradient clipping** | Scaling down unusually large gradients to keep training stable. |
| **Gradient** | How much the loss changes per small change of each parameter; computed automatically by JAX. |
| **Graph neural network (GNN)** | A network that works on a graph by passing messages between connected nodes. |
| **Guide term** | The fixed-width Gaussian-cloud part of the Wannier loss; smooth, needs no pairing. |
| **Heartbeat** | A running sweep job touches its run's marker file every 60 s, so others can tell it's alive. |
| **Hungarian algorithm** | Exact solver for the assignment problem (`scipy.optimize.linear_sum_assignment`). Pairs predicted with true Wannier centers. |
| **Invariant** | Unchanged by a transformation: a Wannier radius doesn't change when the molecule is rotated. |
| **Lagged assignment** | Computing the Wannier pairing for a step from predictions one update older, so CPU pairing overlaps GPU training. |
| **Learning rate** | Step size of each parameter update (`train.lr`). |
| **Lease / run queue** | The sweep's coordination: a job claims a run by an atomic rename and keeps it while its heartbeat/Slurm job is alive; dead jobs' runs go back to the queue. |
| **Lone pair** | Two valence electrons on one atom that aren't in a bond (e.g. on oxygen). |
| **Loss** | One number measuring how wrong the predictions are; training minimizes it. |
| **MAE / RMSE** | Mean absolute error / root-mean-square error; RMSE weights large errors more. |
| **Mesh / sharding** | JAX's description of devices (mesh) and how an array is laid out on them: split across devices or replicated. |
| **Message passing** | A GNN step: each atom gathers information from the others and updates its features. |
| **MLP** | Multi-layer perceptron: dense → activation → dense. |
| **MLWF** | Maximally localized Wannier function: orbitals made as compact as possible (Wannier90). |
| **MSE** | Mean squared error; used for the normalized radius term (and, optionally, positions via `wannier.position_loss: mse`). |
| **NaN** | "Not a number": what a diverged training produces. The NaN guard stops and keeps the last good parameters. |
| **Negative transfer** | Two tasks trained in one network doing worse than trained apart; why the energy target was removed. |
| **NMS (non-maximum suppression)** | When choosing centers, skip a candidate that lies within `wannier.nms_radius` of one already chosen; merges duplicate predictions. |
| **Noise floor** | The best error any model could reach given label noise; ~0.013 Å for Wannier centers here. |
| **Objective (validation)** | The number that picks the best epoch: center MAE + radius_weight · radius MAE / r_std on the selected validation centers (lower is better). |
| **One-to-many targets** | Training targets that mark every candidate near a true center as "yes" (not just the single matched one); consistent with NMS at prediction. |
| **Open-shell** | Unpaired electrons (odd electron count); not supported. |
| **Overfitting / underfitting** | Memorizing the training set vs. not fitting it well enough. |
| **± pair** | Two candidates `b ± s·(u × v)` placed symmetrically about an atom's local plane, for centers above/below a flat molecule; always used together. |
| **Parameters (weights)** | The adjustable numbers inside the network. |
| **Patience** | How many validation checks in a row may pass without a new best before early stopping (`train.patience`; 0 = off). |
| **Promotion (stage 2)** | When screening is done, the best stage-1 run is re-run as `final` with a long schedule; created exactly once. |
| **Pseudopotential** | Replaces a nucleus + its core electrons with an effective potential, so only valence electrons are simulated. |
| **Pseudovector** | A vector that flips sign under a reflection, like a cross product `u × v`; used for the ± pairs, whose two members a mirror swaps. |
| **Radius (of a Wannier function)** | √spread, in Å; the size of the orbital. Typical 0.74–0.97 Å. |
| **Readout** | The final step turning per-atom features into outputs (e.g. summing per-atom energies). |
| **Residual connection** | A layer adds a change to its input (`h = h + update`) instead of replacing it. |
| **Schedule (learning rate)** | How the learning rate changes over training (`train.lr_schedule`). |
| **Seed** | Number that fixes the randomness (initial weights, batch order). `train.seed`; different seeds measure run-to-run noise. |
| **Selection logit** | A candidate's score; trained as yes/no (did the matching choose it?). `select_centers` ranks by it. The count k is still computed, not learned. |
| **Self-symmetry** | A rotation or mirror that maps a molecule onto itself (methane under a 120° turn). An equivariant model's output must keep it, which is why outputs are a set of per-atom candidates rather than a list of slots. |
| **SIESTA** | The DFT program whose output Wannier90 localized (`.fdf` input; its `.out` log isn't read). |
| **Slot** | The old Wannier head's fixed outputs per spin (replaced by per-atom candidates: global slots collapse onto a molecule's symmetry axes). |
| **Spin (up/down)** | An electron property with two values; each spin has its own Wannier set and `.wout` file. |
| **Spin-polarized** | A calculation that treats spin-up and spin-down electrons separately. |
| **Split (train/val/test)** | The fixed 80/10/10 division of molecules: learn / monitor / final score. |
| **Split seed** | `train.split_seed`: fixes which molecules are train/val/test; kept constant so all runs share one test set. |
| **Spread** | ⟨r²⟩ − ⟨r⟩² of a Wannier function, in Å²; how extended it is. |
| **Supervised learning / regression** | Learning a mapping from inputs to known numeric answers. |
| **Sweep** | A planned set of training runs differing in one setting each, run in parallel by many jobs (`configs/sweep.yaml`). |
| **Valence electrons** | Outer electrons that form bonds: H 1, C 4, N 5, O 6, F 7. Wannier centers per spin = their total / 2. |
| **Wannier center** | The average position ⟨r⟩ of a Wannier function; usually on a bond or lone pair. |
| **Wannier90** | The program that computes the Wannier functions (`.wout` output, `Final State` section). |
| **Warmup** | The first epochs, during which the learning rate rises from 0 to `lr`. |
| **Ångström (Å)** | 10⁻¹⁰ m. All positions in the pipeline are in Å. A C–H bond is ≈ 1.09 Å. |
