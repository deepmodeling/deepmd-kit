# Overall

A model has two parts, a descriptor that maps atomic configuration to a set of symmetry invariant features, and a fitting net that takes descriptor as input and predicts the atomic contribution to the target physical property.

DeePMD-kit implements the following descriptors:
1. [`se_e2_a`](train-se-e2-a.md#descriptor): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes the distance between atoms as input.
2. [`se_e2_r`](train-se-e2-r.md): DeepPot-SE constructed from radial information of atomic configurations. The embedding takes the distance between atoms as input.
3. [`se_e3`](train-se-e3.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes angles between two neighboring atoms as input.
4. `loc_frame`: Defines a local frame at each atom, and the compute the descriptor as local coordinates under this frame.
5. [`hybrid`](train-hybrid.md): Concate a list of descriptors to form a new descriptor.

The fitting of the following physical properties are supported
1. [`ener`](train-se-e2-a.md#fitting): Fitting the energy of the system. The force (derivative with atom positions) and the virial (derivative with the box tensor) can also be trained. See [the example](train-se-e2-a.md#loss).
2. `dipole`: The dipole moment.
3. `polar`: The polarizability.