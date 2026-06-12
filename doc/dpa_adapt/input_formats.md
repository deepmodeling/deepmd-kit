# Input Formats

> **Project/package name:** `dpa-adapt`
> **Python import:** `dpa_adapt`
> **Main CLI:** `dpa-adapt`
> **Optional short alias:** `dpaad`
> **Display name:** DPA-ADAPT — Atomistic DPA Adaptation for Property Tasks

`dpa-adapt data convert` and the Python `dpa_adapt.convert()` helper
auto-detect the input type and route it to the correct pipeline:
**SMILES table** → RDKit 3D conformer generation,
**formula table** → random doping from a POSCAR template,
**structure files** → dpdata (auto-detect or explicit `--fmt`).

## 1. SMILES Tables (CSV)

**Trigger:** file extension `.csv` **and** a SMILES column.
By default, the converter reads `SMILES`/`smiles`; use `--smiles-col` for
other column names such as `smi` or `mol`. Or pass `--fmt smiles` explicitly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--smiles-col` | `SMILES` | Column name for SMILES strings |
| `--property-col` | `Property` | Input table column to read target values from; also used as the output label name |
| `--train-ratio` | `0.9` | Fraction of samples used for training set |
| `--mol-dir` | — | Directory of pre-generated `.mol`, `.sdf`, `.xyz`, or `.pdb` structure files (skips RDKit 3D conformer generation) |
| `--mol-template` | `id{row}.mol` | Filename template under `--mol-dir`; use `{row}` for the CSV row index |
| `--split-seed` | `42` | Random seed for train/valid splitting |
| `--conformer-seed` | `42` | Random seed for RDKit 3D conformer generation |

```bash
# Auto-detected via SMILES column
dpa-adapt data convert --input molecules.csv --output ./npy \
    --property-col homo
# Short alias
dpaad data convert --input molecules.csv --output ./npy \
    --property-col homo

# Explicit fmt + custom column names
dpa-adapt data convert --input data.csv --output ./npy --fmt smiles \
    --smiles-col smi --property-col GAP --train-ratio 0.85 \
    --split-seed 42 --conformer-seed 43
# Short alias
dpaad data convert --input data.csv --output ./npy --fmt smiles \
    --smiles-col smi --property-col GAP --train-ratio 0.85 \
    --split-seed 42 --conformer-seed 43
```

## 2. Formula Tables (CSV/TXT + POSCAR Template)

**Trigger:** `--fmt formula`. Reads a table of elemental composition formulas
(e.g. `Ni0.65Gd0.15O2H1`) and a template POSCAR, then generates doped
structures by randomly substituting atoms on the host-element sublattice.

Formula input supports two table styles:

- Headered CSV/TSV: comma- or tab-delimited with named columns, such as
  `formula,Property`.
- Headerless TXT/CSV-style rows: whitespace-delimited with integer column
  indices, such as `Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1    291.9`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--poscar` | *(required)* | Template POSCAR file for the host lattice |
| `--formula-col` | `formula` | Input table column to read composition formulas from; use a column name for headered files or a 0-based index for headerless whitespace files |
| `--base-element` | auto | Host element to substitute. Inferred as the most frequent non-O/H element in the template if omitted. |
| `--sets` | `1` | Number of random structures generated per formula row |
| `--property-col` | `Property` | Input table column to read target values from; use a column name for headered files or a 0-based index for headerless whitespace files |
| `--property-name` | value of `--property-col` | Output label name written as `set.*/{property_name}.npy` |
| `--seed` | `42` | Random seed for selecting substituted host-atom sites |

```bash
dpa-adapt data convert --input compositions.csv --output ./npy --fmt formula \
    --poscar template.POSCAR --sets 3 \
    --formula-col formula --property-col bandgap
# Short alias
dpaad data convert --input compositions.csv --output ./npy --fmt formula \
    --poscar template.POSCAR --sets 3 \
    --formula-col formula --property-col bandgap

# Headerless whitespace-delimited TXT: formula in column 0, target in column 1
dpa-adapt data convert --input 20260514.txt --output ./npy --fmt formula \
    --poscar template.POSCAR --formula-col 0 --property-col 1 \
    --property-name overpotential
```

## 3. Structure Files via dpdata

**Trigger:** inputs not routed to the SMILES or formula pipelines. This means
`--fmt` is neither `smiles` nor `formula`; when `--fmt` is omitted, CSV inputs
are routed here only if they do not contain a recognized SMILES column.
Calls dpdata for format auto-detection or explicit conversion.

### Common Formats

| `--fmt` value | Typical file(s) | Notes |
|---|---|---|
| `extxyz` / `mace/xyz` / `nequip/xyz` / `gpumd/xyz` / `quip/gap/xyz` | `*.xyz` | Extended XYZ variants |
| `xyz` | `*.xyz` | Plain XYZ |
| `vasp/poscar` / `vasp/contcar` | `POSCAR`, `CONTCAR` | VASP input/final structure |
| `vasp/outcar` | `OUTCAR` | VASP output (energies, forces, stress) |
| `vasp/xml` | `vasprun.xml` | VASP XML output |
| `vasp/string` | VASP structure string | VASP structure from a string |
| `abacus/stru` / `stru` | `STRU` | ABACUS input structure |
| `abacus/scf` / `abacus/pw/scf` / `abacus/lcao/scf` | SCF output | ABACUS SCF calculation |
| `abacus/md` / `abacus/pw/md` / `abacus/lcao/md` | MD output | ABACUS molecular dynamics |
| `abacus/relax` / `abacus/pw/relax` / `abacus/lcao/relax` | Relax output | ABACUS relaxation |
| `cp2k/aimd_output` | CP2K MD output | CP2K AIMD output file |
| `cp2k/output` | CP2K SCF output | CP2K single-point output |
| `deepmd/raw` | `set.*/` dirs | DeePMD-kit raw format |
| `deepmd/comp` / `deepmd/npy` | `set.*/` dirs | DeePMD-kit compressed/npy format |
| `deepmd/npy/mixed` | mixed `deepmd/npy` dir | DeePMD-kit mixed npy format |
| `deepmd/hdf5` | `*.hdf5` | DeePMD-kit HDF5 format |
| `lammps/dump` / `dump` | `dump.*` | LAMMPS dump trajectory |
| `lammps/lmp` / `lmp` | `*.lmp` | LAMMPS data file |
| `qe/cp/traj` | CP trajectory | Quantum ESPRESSO Car-Parrinello MD |
| `qe/pw/scf` | PWscf output | Quantum ESPRESSO PWscf |
| `siesta/output` | Siesta output | SIESTA SCF output |
| `siesta/aimd_output` | Siesta MD output | SIESTA AIMD output |
| `gaussian/log` | `*.log` | Gaussian log file |
| `gaussian/fchk` | `*.fchk` | Gaussian formatted checkpoint |
| `gaussian/md` | Gaussian MD output | Gaussian MD trajectory |
| `gaussian/gjf` | `*.gjf` | Gaussian input file |
| `amber/md` | Amber MD output | Amber MD trajectory |
| `gromacs/gro` / `gro` | `*.gro` | GROMACS coordinate file |
| `pwmat/output` / `pwmat/movement` / `pwmat/mlmd` | `REPORT`, `MOVEMENT`, `MLMD` | PWmat output / movement / MLMD |
| `pwmat/final.config` / `pwmat/atom.config` | `final.config`, `atom.config` | PWmat final/input structure |
| `fhi_aims/output` / `fhi_aims/md` | FHI-aims output/MD | FHI-aims calculation or MD trajectory |
| `fhi_aims/scf` | FHI-aims SCF output | FHI-aims SCF |
| `psi4/out` | Psi4 output | Psi4 calculation output |
| `psi4/inp` | Psi4 input | Psi4 input file |
| `orca/spout` | ORCA output | ORCA single-point output |
| `sqm/out` | SQM output | SQM output |
| `sqm/in` | SQM input | SQM input |
| `openmx/md` | OpenMX MD output | OpenMX MD trajectory |
| `n2p2` | n2p2 output | n2p2/NNPack output |
| `dftbplus` | DFTB+ output | DFTB+ detailed.xml |
| `mol` / `mol_file` | `*.mol` | MDL Molfile |
| `sdf` / `sdf_file` | `*.sdf` | MDL SDFile |
| `ase/structure` | Any ASE format | ASE structure (single frame) |
| `ase/traj` | Any ASE trajectory | ASE trajectory (multi-frame) |
| `pymatgen/structure` | pymatgen objects | pymatgen Structure |
| `pymatgen/molecule` | pymatgen objects | pymatgen Molecule |
| `pymatgen/computedstructureentry` | pymatgen objects | pymatgen ComputedStructureEntry |
| `lmdb` | LMDB dir | DeePMD-kit LMDB format |
| `list` | List-format dir | List of system directories |
| `3dmol` | 3Dmol format | 3Dmol.js format |

You can omit `--fmt` and let dpdata infer the input format from the file name
or content. For example, files named `POSCAR`, `OUTCAR`, or `*.xyz` are often
recognized automatically. Use `--fmt` when the file name is ambiguous or
auto-detection fails.

### Single file

```bash
dpa-adapt data convert --input POSCAR --output ./npy
dpaad data convert --input POSCAR --output ./npy

dpa-adapt data convert --input OUTCAR --output ./npy --fmt vasp/outcar
dpaad data convert --input OUTCAR --output ./npy --fmt vasp/outcar

dpa-adapt data convert --input traj.xyz --output ./npy --fmt xyz
dpaad data convert --input traj.xyz --output ./npy --fmt xyz

dpa-adapt data convert --input traj.extxyz --output ./npy --fmt extxyz
dpaad data convert --input traj.extxyz --output ./npy --fmt extxyz
```

### Glob patterns

When `--input` contains wildcards (`*`, `?`, `[`), conversion uses mirrored
batch output:

- **1 or more matches** → each matched file is converted into an output
  directory that mirrors its path relative to the non-wildcard prefix.
- **0 matches** → `FileNotFoundError`.
- A `manifest.json` is written into the output root, recording converted and
  skipped files.

```bash
# Glob output mirrors the input tree under ./npy_root
dpa-adapt data convert --input "calcs/**/OUTCAR" --output ./npy_root --fmt vasp/outcar
dpaad data convert --input "calcs/**/OUTCAR" --output ./npy_root --fmt vasp/outcar
```

For example, `calcs/run1/OUTCAR` is written as `npy_root/run1/OUTCAR/`.
When `--strict` is set, the first conversion error fails immediately. Without
it, errors are skipped and logged in the manifest.
