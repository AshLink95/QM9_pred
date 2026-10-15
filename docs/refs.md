# References

Every academic source this project builds on, grouped by topic, each with what it is used for
here. Where a doc explains the idea, it is linked.

## Data: the quantum-chemistry side
| Reference | Used for |
|---|---|
| Soler, Artacho, Gale, García, Junquera, Ordejón, Sánchez-Portal, "The SIESTA method for ab initio order-N materials simulation", *J. Phys.: Condens. Matter* **14**, 2745 (2002). | The DFT code that produced every training target (`.fdf`/`.out`; [physics_primer](physics_primer.md)). |
| Ramakrishnan, Dral, Rupp, von Lilienfeld, "Quantum chemistry structures and properties of 134 kilo molecules", *Scientific Data* **1**, 140022 (2014). | The QM9 molecule set (geometries); its published energies are B3LYP, ours are recomputed with SIESTA. |
| Marzari, Vanderbilt, "Maximally localized generalized Wannier functions for composite energy bands", *Phys. Rev. B* **56**, 12847 (1997). | Definition of the maximally localized Wannier functions whose centers and spreads are targets. |
| Marzari, Mostofi, Yates, Souza, Vanderbilt, "Maximally localized Wannier functions: Theory and applications", *Rev. Mod. Phys.* **84**, 1419 (2012). | Review: what Wannier centers mean (bonds, lone pairs), spreads, dipoles from centers. |
| Pizzi et al., "Wannier90 as a community code: new features and applications", *J. Phys.: Condens. Matter* **32**, 165902 (2020). | The program that wrote the `.wout` files (`Final State` section parsed in `data/parse_wout.py`). |
| Zhang, Chen, Wu, Wang, E, Car, "Deep neural network for the dielectric response of insulators" (Deep Wannier), *Phys. Rev. B* **102**, 041121(R) (2020). | Reference accuracy for ML-predicted Wannier centroids, quoted in the accuracy report. |
| Krishnamoorthy, Nomura, Baradwaj, Shimamura, Rajak, Mishra, Fukushima, Shimojo, Kalia, Nakano, Vashishta, "Dielectric Constant of Liquid Water Determined with Neural Network Quantum Molecular Dynamics", *Phys. Rev. Lett.* **126**, 216403 (2021). | A neural network predicting MLWF centers (water) to get dipoles/polarization — the downstream use our centers aim at. |
| Amano, Yamazaki, Tsuneyuki, "Chemical bond based machine learning model for dipole moment: Application to dielectric properties of liquid methanol and ethanol", *Phys. Rev. B* **110**, 165159 (2024). | Wannier centers assigned to chemical bonds and predicted per bond — a physics-informed alternative to our generic per-atom candidates (not used: it needs bond perception). |

## Model
| Reference | Used for |
|---|---|
| Satorras, Hoogeboom, Welling, "E(n) Equivariant Graph Neural Networks", *ICML* (2021), arXiv:2102.09844. | The backbone: invariant `h` + equivariant `x` tracks, distance-only messages (`model/egnn.py`, [model](model.md)). |
| EGNN reference code, github.com/vgsatorras/egnn (`main_qm9.py`, `qm9/models.py`). | EGNN's QM9 recipe behind its 11 meV: one model per target, **L1 loss** on targets **normalized** by mean/MAD, 7 layers × 128, batch 96, 1000 epochs, cosine schedule, edge attention, and **coordinates not updated**. Source of `loss.energy_loss: l1` and of the v4 sweep variants `layers7`, `batch96`, `coords_fixed`, `edge_gate`. |
| Schütt, Unke, Gastegger, "Equivariant message passing for the prediction of tensorial properties and molecular spectra" (PaiNN), *ICML* (2021), arXiv:2102.03150. | Data-driven reference: U0 MAE 5.85 meV on QM9 (B3LYP labels). |
| Gasteiger, Giri, Margraf, Günnemann, "Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" (DimeNet++), ML4Molecules workshop, NeurIPS (2020), arXiv:2011.14115. | Data-driven reference: U0 MAE 6.32 meV on QM9. |
| Schütt, Sauceda, Kindermans, Tkatchenko, Müller, "SchNet – A deep learning architecture for molecules and materials", *J. Chem. Phys.* **148**, 241722 (2018). | Gaussian radial-basis expansion of distances (`gaussian_rbf`). |
| Xie, Smidt, "Equivariant Symmetry Breaking Sets", *TMLR* (2024), arXiv:2402.02681. | Why an equivariant model must output the whole symmetric *set* when a molecule maps onto itself; motivates per-atom candidates (`model/heads.py`). |
| Smidt, Geiger, Miller, "Finding symmetry breaking order parameters with Euclidean neural networks", *Phys. Rev. Research* **3**, L012002 (2021). | The same constraint from the physics side: equivariant outputs keep the input's symmetry. |
| Lawrence, Portilheiro, Zhang, Kaba, "Improving Equivariant Networks with Probabilistic Symmetry Breaking", *ICLR* (2025), arXiv:2503.21985. | States the limitation (outputs keep the input's self-symmetries); random symmetry breaking (SymPE) was considered and not used, since the outputs must stay deterministic. |
| Kaba, Ravanbakhsh, "Symmetry Breaking and Equivariant Neural Networks", NeurIPS 2023 workshop (NeurReps), arXiv:2312.09016. | Relaxed equivariance; considered, not used. |
| Abboud, Ceylan, Grohe, Lukasiewicz, "The Surprising Power of Graph Neural Networks with Random Node Initialization", *IJCAI* (2021), arXiv:2010.01179. | Random features to break ties; considered, not used (non-deterministic). |
| Locatello et al., "Object-Centric Learning with Slot Attention", *NeurIPS* (2020), arXiv:2006.15055. | Noise-initialized slots for set prediction; considered, not used (non-deterministic). |
| Hoogeboom, Satorras, Vignac, Welling, "Equivariant Diffusion for Molecule Generation in 3D", *ICML* (2022). | Possible future direction: generate the center set by equivariant denoising. |
| Song et al., "Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation", *NeurIPS* (2023). | Same, with flow matching. |

## Predicting a set: matching
| Reference | Used for |
|---|---|
| Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko, "End-to-End Object Detection with Transformers" (DETR), *ECCV* (2020), arXiv:2005.12872. | Matched set loss: pair predictions with targets by the Hungarian algorithm, don't differentiate the pairing ([losses](losses.md)). |
| Kuhn, "The Hungarian method for the assignment problem", *Naval Research Logistics Quarterly* **2**, 83 (1955). | The assignment algorithm. |
| Crouse, "On implementing 2D rectangular assignment algorithms", *IEEE Trans. Aerospace and Electronic Systems* **52**, 1679 (2016). | The algorithm behind `scipy.optimize.linear_sum_assignment` (rectangular: k centers out of C candidates). |
| Ouyang-Zhang, Cho, Zhou, Krähenbühl, "NMS Strikes Back", arXiv:2212.06137 (2022). | One-to-one matching makes the model learn global de-duplication; one-to-many targets plus non-maximum suppression (NMS) do as well or better. Basis of `wannier.nms_radius`: NMS in `select_centers` and NMS-consistent selection targets (`selection_targets`). |
| Li, Zhang, Liu, Guo, Ni, Zhang, "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising", *CVPR* (2022), arXiv:2203.01305. | Bipartite-matching instability early in training, fixed by denoising queries; considered, not used (our pairings settle quickly). |
| Liu et al., "Detection Transformer with Stable Matching" (Stable-DINO), *ICCV* (2023), arXiv:2304.04742. | Position-aware classification targets; considered, not needed once twin candidates get consistent targets. |
| Cai et al., "Align-DETR: Enhancing End-to-end Object Detection with Aligned Loss", *BMVC* (2024), arXiv:2304.07527. | Quality-aware (IoU) BCE targets; considered, not used. |
| Zhu, Su, Lu, Li, Wang, Dai, "Deformable DETR: Deformable Transformers for End-to-End Object Detection", *ICLR* (2021), arXiv:2010.04159. | Iterative refinement of predictions layer by layer; a candidate–candidate refinement layer is a possible next step (deferred: (4N)² edges). |

## Optimization and training
| Reference | Used for |
|---|---|
| Kingma, Ba, "Adam: A Method for Stochastic Optimization", *ICLR* (2015), arXiv:1412.6980. | The optimizer. |
| Loshchilov, Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", *ICLR* (2017), arXiv:1608.03983. | Cosine learning-rate decay ([training](training.md)). |
| Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", arXiv:1706.02677 (2017). | Learning-rate warmup. |
| Pascanu, Mikolov, Bengio, "On the difficulty of training recurrent neural networks", *ICML* (2013). | Gradient clipping by global norm. |
| Prechelt, "Early Stopping — But When?", in *Neural Networks: Tricks of the Trade*, Springer (1998). | Early stopping with patience. |
| Kendall, Gal, Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", *CVPR* (2018). | Background on balancing loss terms in different units ([ml_primer](ml_primer.md)). |
| Polyak, Juditsky, "Acceleration of Stochastic Approximation by Averaging", *SIAM J. Control Optim.* **30**, 838 (1992). | Averaging the iterates of stochastic optimization; the basis of the weight EMA (`train.ema_decay`). |
| Batzner et al., "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials" (NequIP), *Nat. Commun.* **13**, 2453 (2022). | Practice: validation/test and the final model use an EMA of the weights (decay 0.99). |
| Batatia, Kovács, Simm, Ortner, Csányi, "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", *NeurIPS* (2022), arXiv:2206.07697. | Same EMA practice (0.99). |
| Yu, Kumar, Gupta, Levine, Hausman, Finn, "Gradient Surgery for Multi-Task Learning" (PCGrad), *NeurIPS* (2020), arXiv:2001.06782. | Remedy for conflicting gradients between tasks; deferred until the `energy_only` diagnostic shows energy and Wannier conflict. |
| Chen, Badrinarayanan, Lee, Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks", *ICML* (2018). | Automatic loss balancing; deferred for the same reason. |

## Background reading
- Goodfellow, Bengio, Courville, *Deep Learning*, MIT Press (2016), deeplearningbook.org.
- Sanchez-Lengeling, Reif, Pearce, Wiltschko, "A Gentle Introduction to Graph Neural Networks", *Distill* (2021).
- Bradbury et al., JAX: composable transformations of Python+NumPy programs (2018), github.com/jax-ml/jax; Flax and Optax documentation.
