# Physics primer — what this project predicts, for non-physicists

You don't need a chemistry or physics degree to work on this code, but you need the handful of
ideas below. Each section ends with **"In the code"**, which says where the idea shows up.
Terms in **bold** are also in the [glossary](glossary.md).

Related: [ml_primer](ml_primer.md) (the machine-learning side) · [developer_guide](developer_guide.md)
(how to work on the repo) · [index](README.md).

---

## 1. Atoms, electrons, bonds

- A molecule is a set of **atoms**. Each atom is a positively charged nucleus surrounded by
  negatively charged electrons.
- The **atomic number Z** identifies the element: H = 1, C = 6, N = 7, O = 8, F = 9. A neutral
  atom has Z electrons.
- Electrons come in two kinds for our purposes:
  - **Core electrons** sit close to the nucleus and don't take part in chemistry.
  - **Valence electrons** are the outer ones. They form chemical bonds and decide how the
    molecule behaves. H has 1, C 4, N 5, O 6, F 7.
- A **chemical bond** is (roughly) a pair of electrons shared between two atoms. A **lone pair**
  is a pair of valence electrons that belongs to one atom without being shared (oxygen in water
  has two).

> **In the code:** a molecule is just `z` (the atomic numbers) and `pos` (each atom's x, y, z in
> Ångström). That's the model's entire input.

## 2. The dataset: QM9 recomputed with SIESTA

- **QM9** is a standard benchmark of ~134,000 small organic molecules made of H, C, N, O, F,
  with at most 9 non-hydrogen atoms (so at most ~29 atoms in total).
- This project re-ran those molecules through **SIESTA** (a physics simulation program, §3) and
  **Wannier90** (a post-processing program, §5). After dropping failed runs, **117,946**
  molecules remain in `dataset.pkl`.

## 3. How the reference answers were computed (DFT and SIESTA)

- The exact behavior of electrons follows the Schrödinger equation of quantum mechanics, which
  is far too expensive to solve exactly for a molecule.
- **Density functional theory (DFT)** is the standard practical approximation. It computes the
  lowest-energy arrangement of the electrons (the "ground state") for fixed atom positions.
- DFT is solved **iteratively**: guess the electrons, compute the forces they feel, update,
  repeat. Each round is an **SCF step** (self-consistent field) until the answer stops changing,
  i.e. it has **converged**. Only the converged values are meaningful.
- **SIESTA** is the DFT program used here. Two details matter:
  - **Pseudopotentials:** SIESTA doesn't simulate core electrons. Each nucleus plus its core
    electrons is replaced by an effective "pseudo-ion", and only valence electrons are simulated.
    That's why the valence counts in §1 drive everything later.
  - **Spin-polarized** calculation: spin-up and spin-down electrons are treated separately (§6).
- One DFT + Wannier calculation takes minutes per molecule. The trained model gives an answer in
  milliseconds, which is the point of the project.

### Units
| Quantity | Unit used here | Notes |
|---|---|---|
| length | **Å** (ångström, 10⁻¹⁰ m) | a C–H bond is ≈ 1.09 Å |
| energy | **eV** (electron-volt) | |
| length, atomic unit | Bohr = 0.529 Å | appears only in parts of SIESTA's `.out` we don't use |
| energy, atomic unit | Ry (Rydberg) = 13.606 eV | |

> **In the code:** `data/units.py` holds the conversions. Everything inside the pipeline is in Å
> and eV.

## 4. The first target: total energy

- SIESTA reports a **total energy** for each molecule (the line `siesta: Total = …` in the
  `.out` file). It's the energy of the simulated valence electrons and pseudo-ions together.
- The numbers are large and negative (roughly −500 to −3,500 eV here) because they include how
  tightly every valence electron is bound. Only **differences** between energies matter
  physically (reaction energies, which conformation is more stable).
- Most of a molecule's total energy is explained by **which atoms it contains**: roughly, each
  carbon contributes about the same amount wherever it is. A least-squares fit of "energy ≈ sum
  of one number per element" (the **composition baseline**) already gets within ~0.9 eV. The
  network's real job is the remaining part, which depends on geometry and bonding.
- **Chemical accuracy** is 1 kcal/mol ≈ **0.043 eV**: the error level chemists consider good
  enough for predictions. Use it as the yardstick for energy error.

> **In the code:** `data/parse_siesta.py:parse_energy` takes the **last** `siesta: Total` line,
> since earlier lines are unconverged SCF steps. The model's per-element offset (`atom_ref`) is
> initialized from that composition fit (`training/train.py:_fit_atom_ref`).

## 5. The second target: Wannier centers

### Orbitals
Quantum mechanics describes electrons with **orbitals**: functions telling where an electron
is likely to be. The orbitals a DFT program produces (**molecular orbitals**) are usually spread
over the whole molecule, which makes them hard to read chemically.

### Wannier functions
There is freedom in how to write the occupied orbitals: any rotation (a "unitary mix") of them
describes exactly the same electrons. **Maximally localized Wannier functions** use that freedom
to make each orbital as compact as possible. The program **Wannier90** does this. The result
usually matches a chemist's sketch: one compact function per bond and per lone pair.

Each Wannier function is summarized by two numbers we predict:
- its **center** ⟨r⟩: the average position of its electron, a 3D point in Å;
- its **spread** ⟨r²⟩ − ⟨r⟩², in Å². We use **radius = √spread** (Å) as its size. Typical radii
  here are 0.74–0.97 Å.

### How many centers?
In a spin-polarized calculation, each Wannier function of a spin channel holds **one electron of
that spin**. A closed-shell molecule (§6) has half of its valence electrons in each spin, so:

> **centers per spin = (total valence electrons) / 2**

Worked examples:
- **Methane CH₄:** 4 (C) + 4 × 1 (H) = 8 valence electrons → **4 centers per spin**, one on
  each C–H bond.
- **Water H₂O:** 6 (O) + 2 × 1 (H) = 8 → **4 per spin**: two O–H bonds and two lone pairs on the
  oxygen.

This rule matched the parsed files for 100% of the dataset.

### Why they're useful
Wannier centers give a chemically readable picture of where the electrons are. Together with
the nuclei they give the molecule's **dipole moment**, which feeds infrared spectra and
dielectric properties. That's the main use in the literature (e.g. the "Deep Wannier" model for
water).

### One caveat: they're not perfectly unique
Localization is a minimization, and some molecules have several equally good answers (a double
bond can be written as two "banana" bonds or as one σ plus one π bond). Wannier90 may land on
either. In our data, the spin-up and spin-down sets, which should be identical, differ by
**~0.013 Å per center on average** for this reason. No model can beat that noise floor.

> **In the code:** `data/parse_wout.py` reads the **Final State** section of each `.wout`
> (earlier lines are unconverged steps). `data/electrons.py:centers_per_spin` implements the
> counting rule.

## 6. Spin

- Electrons have a property called **spin** with two values, **up** and **down**.
- A **spin-polarized** calculation keeps the two populations separate, so there is one set of
  Wannier functions per spin: two `.wout` files per molecule (`…up.wout`, `…down.wout`).
- A **closed-shell** molecule has every electron paired (equal numbers of up and down). All
  QM9 molecules here are closed-shell, so both spins have the same count, and their centers are
  nearly identical (up to the caveat above).
- An **open-shell** molecule (a radical, with an odd electron count) would have different up
  and down counts. That case isn't supported; the code raises an error.
- Spin is kept separate everywhere in the model and the loss: a spin-up prediction is only ever
  compared with spin-up truth.

## 7. The files on disk

| File | Written by | What we take from it |
|---|---|---|
| `<id>.fdf` | the SIESTA **input** | element list and atom coordinates (the model's input) |
| `<id>.out` | the SIESTA log | the last `siesta: Total = …` (converged energy, eV) |
| `<id>.manifold.valence.up.wout` / `.down.wout` | Wannier90 log, one per spin | the `Final State` block: each center's x, y, z and spread |

The `.out` also lists "trial projector" centers (starting guesses for Wannier90, in Bohr). They
are **never used**: the model must predict the converged centers from geometry alone.

`<id>` is the molecule id, e.g. `C2F3N3O_133481`. The three file types can live in different
folders; they're matched by this id ([data](data.md) §5).

A `.fdf` excerpt:
```
%block ChemicalSpeciesLabel
 1  6  C          # species 1 is carbon (Z = 6)
 2  1  H
%endblock ChemicalSpeciesLabel
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
 0.000  0.000  0.000  1      # x y z species
 0.629  0.629  0.629  2
%endblock AtomicCoordinatesAndAtomicSpecies
```
A `.wout` `Final State` excerpt:
```
 Final State
  WF centre and spread    1  (  0.000005,  0.000234,  1.155586 )     0.71534663
  ...
 Sum of centres and spreads ( ... )
```
The four numbers are the center's x, y, z (Å) and its spread (Å²).

## 8. Symmetry: why the model is built the way it is

Physics doesn't care where you put a molecule or which way it faces:
- **Move or rotate** the molecule → the energy stays the same (it is **invariant**), and the
  Wannier centers move and rotate *with* the molecule (they are **equivariant**).
- **Relabel the atoms** (call atom 3 "atom 1") → nothing physical changes.
- **Mirror** the molecule → the energy is the same and the centers mirror. (A molecule that
  differs from its mirror image is **chiral**. Only properties that tell mirror images apart may
  legitimately differ; none of ours do.)

The model guarantees all of these by construction, not by hoping it learns them. How is in
[ml_primer](ml_primer.md) §8 and [model](model.md) §4.

## 9. Further reading
- *Density Functional Theory: A Practical Introduction*, Sholl & Steckel (Wiley) — gentle DFT.
- Soler et al., "The SIESTA method…", *J. Phys.: Condens. Matter* 14, 2745 (2002).
- Marzari et al., "Maximally localized Wannier functions: Theory and applications",
  *Rev. Mod. Phys.* 84, 1419 (2012).
- Ramakrishnan et al., "Quantum chemistry structures and properties of 134 kilo molecules",
  *Scientific Data* 1, 140022 (2014) — the QM9 dataset.
- Zhang et al., "Deep neural network for the dielectric response of insulators",
  *Phys. Rev. B* 102, 041121 (2020) — ML prediction of Wannier centroids.
