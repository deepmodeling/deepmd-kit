# Input Formats

> **CLI command:** `dpaad` (PyPI package: `dpa-adapt`).
> `dpaad` is the short alias you type; both names are equivalent.

`dpaad data convert` auto-detects the input type and routes it to the correct pipeline:
**SMILES/CSV** → RDKit conformer generation, **formula CSV** → random doping from
POSCAR template, **everything else** → dpdata (auto-detect or explicit `--fmt`).

## 1. SMILES / Molecular (CSV or Excel)

**Trigger:** file extension `.csv`/`.xlsx`/`.xls` **and** a column named
`smiles`/`smi`/`mol` (case-insensitive).  Or pass `--fmt smiles` explicitly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--smiles-col` | `SMILES` | Column name for SMILES strings |
| `--property-col` | `Property` | Column name for target property |
| `--property-name` | `Property` | Label key written into each system |
| `--train-ratio` | `0.9` | Fraction of rows used for training set |
| `--mol-dir` | — | Directory of pre-generated `.mol` files (skips RDKit conformer generation) |
| `--seed` | `42` | Random seed for conformer generation and train/valid split |

```bash
# Auto-detected via SMILES column
dpaad data convert --input molecules.csv --output ./npy --property-name homo

# Explicit fmt + custom column names
dpaad data convert --input data.xlsx --output ./npy --fmt smiles \
    --smiles-col SMILES --property-col GAP --train-ratio 0.85 --seed 123
```

## 2. Formula Substitution (CSV + template POSCAR)

**Trigger:** `--fmt formula`.  Reads a CSV of elemental composition formulas
(e.g. `Ni0.65Gd0.15O2H1`) and a template POSCAR, then generates doped structures
by randomly substituting atoms on the host-element sublattice.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--poscar` | *(required)* | Template POSCAR file for the host lattice |
| `--formula-col` | `0` | Column index (0-based) or name for the formula string |
| `--base-element` | auto | Host element to substitute. Inferred as the most frequent non-O/H element in the template if omitted. |
| `--sets` | `1` | Number of random structures generated per formula row |
| `--property-col` | `1` | Column index or name for the target property value |
| `--seed` | `42` | Random seed |

```bash
dpaad data convert --input compositions.csv --output ./npy --fmt formula \
    --poscar template.POSCAR --sets 3 --property-col bandgap
```

## 3. Structure Files via dpdata

**Trigger:** all other cases (no SMILES columns, not `--fmt formula`/`smiles`).
Calls dpdata for format auto-detection or explicit conversion.

### Common Formats

| `--fmt` value | Typical file(s) | Notes |
|---|---|---|
| `extxyz` | `*.xyz` | Extended XYZ (includes cell & per-atom properties) |
| `xyz` | `*.xyz` | Plain XYZ |
| `vasp/poscar` | `POSCAR` | VASP input structure |
| `vasp/contcar` | `CONTCAR` | VASP final structure |
| `vasp/outcar` | `OUTCAR` | VASP output (energies, forces, stress) |
| `vasp/xml` | `vasprun.xml` | VASP XML output |
| `abacus/scf` | SCF output | ABACUS SCF calculation |
| `abacus/md` | MD output | ABACUS molecular dynamics |
| `abacus/stru` | `STRU` | ABACUS input structure |
| `abacus/relax` | Relax output | ABACUS relaxation |
| `abacus/pw/scf` | PW SCF output | ABACUS plane-wave SCF |
| `abacus/lcao/scf` | LCAO SCF output | ABACUS LCAO SCF |
| `abacus/pw/md` | PW MD output | ABACUS plane-wave MD |
| `abacus/lcao/md` | LCAO MD output | ABACUS LCAO MD |
| `abacus/pw/relax` | PW relax output | ABACUS plane-wave relaxation |
| `abacus/lcao/relax` | LCAO relax output | ABACUS LCAO relaxation |
| `cp2k/aimd_output` | CP2K MD output | CP2K AIMD output file |
| `cp2k/output` | CP2K SCF output | CP2K single-point output |
| `deepmd/npy` | `set.*/` dirs | DeePMD-kit npy format |
| `deepmd/raw` | `set.*/` dirs | DeePMD-kit raw format |
| `deepmd/comp` | `set.*/` dirs | DeePMD-kit compressed npy |
| `deepmd/hdf5` | `*.hdf5` | DeePMD-kit HDF5 format |
| `lammps/dump` | `dump.*` | LAMMPS dump trajectory |
| `lammps/lmp` | `*.lmp` | LAMMPS data file |
| `qe/cp/traj` | CP trajectory | Quantum ESPRESSO Car-Parrinello MD |
| `qe/pw/scf` | PWscf output | Quantum ESPRESSO PWscf |
| `siesta/output` | Siesta output | SIESTA SCF output |
| `siesta/aimd_output` | Siesta MD output | SIESTA AIMD output |
| `gaussian/log` | `*.log` | Gaussian log file |
| `gaussian/fchk` | `*.fchk` | Gaussian formatted checkpoint |
| `gaussian/md` | Gaussian MD output | Gaussian MD trajectory |
| `gaussian/gjf` | `*.gjf` | Gaussian input file |
| `amber/md` | Amber MD output | Amber MD trajectory |
| `gromacs/gro` | `*.gro` | GROMACS coordinate file |
| `pwmat/output` | `REPORT`/`MOVEMENT` | PWmat output |
| `pwmat/atom.config` | `atom.config` | PWmat input structure |
| `pwmat/movement` | `MOVEMENT` | PWmat MD trajectory |
| `pwmat/mlmd` | `MLMD` | PWmat MLMD output |
| `fhi_aims/output` | FHI-aims output | FHI-aims calculation |
| `fhi_aims/md` | FHI-aims MD output | FHI-aims MD trajectory |
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
| `quip/gap/xyz` | `*.xyz` | QUIP/GAP extended XYZ |
| `mace/xyz` | `*.xyz` | MACE extended XYZ |
| `nequip/xyz` | `*.xyz` | NequIP extended XYZ |
| `gpumd/xyz` | `*.xyz` | GPUMD extended XYZ |
| `lmdb` | LMDB dir | DeePMD-kit LMDB format |
| `list` | List-format dir | List of system directories |
| `3dmol` | 3Dmol format | 3Dmol.js format |

Omit `--fmt` for dpdata auto-detection (works for most common formats like
POSCAR, OUTCAR, extxyz, etc.).  Pass `--fmt` explicitly when the file
extension is ambiguous or auto-detection fails.

### Single file

```bash
dpaad data convert --input POSCAR --output ./npy
dpaad data convert --input OUTCAR --output ./npy --fmt vasp/outcar
dpaad data convert --input traj.xyz --output ./npy --fmt extxyz
```

### Glob patterns

When `--input` contains wildcards (`*`, `?`, `[`):

- **1 match** → treated as a single file (output directly into `--output`).
- **N > 1 matches** → each match is converted into a numbered subdirectory
  `{output}/sys_{i:04d}/` (zero-indexed, sorted).
- **0 matches** → `FileNotFoundError`.

```bash
# Single match (only one OUTCAR found)
dpaad data convert --input "run*/OUTCAR" --output ./npy

# Multi-match: outputs sys_0000/, sys_0001/, …
dpaad data convert --input "calcs/**/OUTCAR" --output ./npy_root --fmt vasp/outcar
```

## 4. Batch Mode

**Trigger:** `--input` with glob wildcards and N > 1 matches.  Uses
`batch_convert()` internally.

Key behaviors:

- Output directory tree mirrors the input tree structure (relative to the
  non-wildcard prefix of the glob pattern).
- A `manifest.json` is written into the output root, recording every
  converted and skipped file.
- When `--strict` is set, the first conversion error fails immediately.
  Without it (default), errors are skipped and logged.

```bash
# Batch convert all OUTCAR files; each lands in a mirrored subdirectory
dpaad data convert --input "scan/**/OUTCAR" --output ./all_npy --fmt vasp/outcar

# Strict mode — abort on first failure
dpaad data convert --input "scan/**/OUTCAR" --output ./all_npy --fmt vasp/outcar --strict

# Check the manifest
cat ./all_npy/manifest.json
```
