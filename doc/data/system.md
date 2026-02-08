# System

DeePMD-kit takes a **system** as the data structure. A snapshot of a system is called a **frame**. A system may contain multiple frames with the same atom types and numbers, i.e. the same formula (like `H2O`). To contains data with different formulas, one usually needs to divide data into multiple systems, which may sometimes result in sparse-frame systems.

A system should contain system properties, input frame properties, and labeled frame properties. The system property contains the following property:

| ID       | Property            | Raw file     | Required/Optional | Shape  | Description                                                                                                                                                                                                                                                                                                                                                                                             |
| -------- | ------------------- | ------------ | ----------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| type     | Atom type indexes   | type.raw     | Required          | Natoms | Integers that start with 0. If both the training parameter {ref}`type_map <model/type_map>` is set and `type_map.raw` is provided, the system atom type should be mapped to `type_map.raw` in `type.raw` and will be mapped to the model atom type when training; otherwise, the system atom type will be always mapped to the model atom type (whether {ref}`type_map <model/type_map>` is set or not) |
| type_map | Atom type names     | type_map.raw | Optional          | Ntypes | Atom names that map to atom type, which is unnecessary to be contained in the periodic table. Only works when the training parameter {ref}`type_map <model/type_map>` is set                                                                                                                                                                                                                            |
| nopbc    | Non-periodic system | nopbc        | Optional          | 1      | If True, this system is non-periodic; otherwise it's periodic                                                                                                                                                                                                                                                                                                                                           |

The input frame properties contain the following property, the first axis of which is the number of frames:

| ID        | Property                                            | Raw file   | Unit | Required/Optional    | Shape                    | Description                               |
| --------- | --------------------------------------------------- | ---------- | ---- | -------------------- | ------------------------ | ----------------------------------------- |
| coord     | Atomic coordinates                                  | coord.raw  | Å    | Required             | Nframes \* Natoms \* 3   |
| box       | Boxes                                               | box.raw    | Å    | Required if periodic | Nframes \* 3 \* 3        | in the order `XX XY XZ YX YY YZ ZX ZY ZZ` |
| fparam    | Extra frame parameters                              | fparam.raw | Any  | Optional             | Nframes \* Any           |
| aparam    | Extra atomic parameters                             | aparam.raw | Any  | Optional             | Nframes \* aparam \* Any |
| numb_copy | Each frame is copied by the `numb_copy` (int) times | prob.raw   | 1    | Optional             | Nframes                  | Integer; Default is 1 for all frames      |

The labeled frame properties are listed as follows, all of which will be used for training if and only if the loss function contains such property:

| ID                    | Property                                                                         | Raw file                  | Unit   | Shape                                 | Description                               |
| --------------------- | -------------------------------------------------------------------------------- | ------------------------- | ------ | ------------------------------------- | ----------------------------------------- |
| energy                | Frame energies                                                                   | energy.raw                | eV     | Nframes                               |
| force                 | Atomic forces                                                                    | force.raw                 | eV/Å   | Nframes \* Natoms \* 3                |
| virial                | Frame virial                                                                     | virial.raw                | eV     | Nframes \* 9                          | in the order `XX XY XZ YX YY YZ ZX ZY ZZ` |
| hessian               | Frame energy Hessian matrices                                                    | hessian.raw               | eV/Å^2 | Nframes \* Natoms \* 3 \* Natoms \* 3 | full Hessian matrices                     |
| atom_ener             | Atomic energies                                                                  | atom_ener.raw             | eV     | Nframes \* Natoms                     |
| atom_pref             | Weights of atomic forces                                                         | atom_pref.raw             | 1      | Nframes \* Natoms                     |
| dipole                | Frame dipole                                                                     | dipole.raw                | Any    | Nframes \* 3                          |
| atomic_dipole         | Atomic dipole                                                                    | atomic_dipole.raw         | Any    | Nframes \* Natoms \* 3                |
| polarizability        | Frame polarizability                                                             | polarizability.raw        | Any    | Nframes \* 9                          | in the order `XX XY XZ YX YY YZ ZX ZY ZZ` |
| atomic_polarizability | Atomic polarizability                                                            | atomic_polarizability.raw | Any    | Nframes \* Natoms \* 9                | in the order `XX XY XZ YX YY YZ ZX ZY ZZ` |
| drdq                  | Partial derivative of atomic coordinates with respect to generalized coordinates | drdq.raw                  | 1      | Nframes \* Natoms \* 3 \* Ngen_coords |

In general, we always use the following convention of units:

| Property | Unit   |
| -------- | ------ |
| Time     | ps     |
| Length   | Å      |
| Energy   | eV     |
| Force    | eV/Å   |
| Virial   | eV     |
| Hessian  | eV/Å^2 |
| Pressure | Bar    |

## Mixed type

:::{note}
Only the [DPA-1](../model/train-se-atten.md) and [DPA-2](../model/dpa2.md) descriptors support this format.
:::

In the standard data format, only those frames with the same fingerprint (i.e. the number of atoms of different elements) can be put together as a unified system.
This may lead to sparse frame numbers in those rare systems.

An ideal way is to put systems with the same total number of atoms together, which is the way we trained DPA-1 on [OC2M](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md).
This system format, which is called `mixed_type`, is proper to put frame-sparse systems together and is slightly different from the standard one.
Take an example, a `mixed_type` may contain the following files:

```
type.raw
type_map.raw
set.*/box.npy
set.*/coord.npy
set.*/energy.npy
set.*/force.npy
set.*/real_atom_types.npy
```

This system contains `Nframes` frames with the same atom number `Natoms`, the total number of element types contained in all frames is `Ntypes`. Most files are the same as those in [standard formats](../data/system.md), here we only list the distinct ones:

| ID       | Property                         | File                | Required/Optional | Shape             | Description                                                                                                              |
| -------- | -------------------------------- | ------------------- | ----------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| /        | Atom type indexes (place holder) | type.raw            | Required          | Natoms            | All zeros to fake the type input                                                                                         |
| type_map | Atom type names                  | type_map.raw        | Required          | Ntypes            | Atom names that map to atom type contained in all the frames, which is unnecessart to be contained in the periodic table |
| type     | Atom type indexes of each frame  | real_atom_types.npy | Required          | Nframes \* Natoms | Integers that describe atom types in each frame, corresponding to indexes in type_map. `-1` means virtual atoms.         |

With these edited files, one can put together frames with the same `Natoms`, instead of the same formula (like `H2O`).

To put frames with different `Natoms` into the same system, one can pad systems by adding virtual atoms whose type is `-1`. Virtual atoms do not contribute to any fitting property, so the atomic property of virtual atoms (e.g. forces) should be given zero.

The API to generate or transfer to `mixed_type` format is available on [dpdata](https://github.com/deepmodeling/dpdata) for a more convenient experience.
