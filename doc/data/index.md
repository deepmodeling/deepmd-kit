# Data

In this section, we will introduce how to convert the DFT labeled data into the data format used by DeePMD-kit.

The DeePMD-kit organize data in `systems`. Each `system` is composed by a number of `frames`. One may roughly view a `frame` as a snap shot on an MD trajectory, but it does not necessary come from an MD simulation. A `frame` records the coordinates and types of atoms, cell vectors if the periodic boundary condition is assumed, energy, atomic forces and virial. It is noted that the `frames` in one `system` share the same number of atoms with the same type.

- [System](system.md)
- [Formats of a system](data-conv.md)
- [Prepare data with dpdata](dpdata.md)
