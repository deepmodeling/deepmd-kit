# Overall

A model has two parts, a descriptor that maps atomic configuration to a set of symmetry invariant features, and a fitting net that takes descriptor as input and predicts the atomic contribution to the target physical property. It's defined in the {ref}`model <model>` section of the `input.json`, for example,
```json
    "model": {
        "type_map":	["O", "H"],
        "descriptor" :{
            "...": "..."
        },
        "fitting_net" : {
            "...": "..."
        }
    }
```
The two subsections, {ref}`descriptor <model/descriptor>` and {ref}`fitting_net <model/fitting_net>`, define the descriptor and the fitting net, respectively.

The {ref}`type_map <model/type_map>` is optional, which provides the element names (but not necessarily same as the actual name of the element) of the corresponding atom types. A water model, as in this example, has two kinds of atoms. The atom types are internally recorded as integers, e.g., `0` for oxygen and `1` for hydrogen here. A mapping from the atom type to their names is provided by {ref}`type_map <model/type_map>`.

DeePMD-kit implements the following descriptors:
1. [`se_e2_a`](train-se-e2-a.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes the distance between atoms as input.
2. [`se_e2_r`](train-se-e2-r.md): DeepPot-SE constructed from radial information of atomic configurations. The embedding takes the distance between atoms as input.
3. [`se_e3`](train-se-e3.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes angles between two neighboring atoms as input.
4. [`se_a_mask`](train-se-a-mask.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The input frames in one system can have a varied number of atoms. Input particles are padded with virtual particles of the same length.
5. `loc_frame`: Defines a local frame at each atom and compute the descriptor as local coordinates under this frame.
6. [`hybrid`](train-hybrid.md): Concate a list of descriptors to form a new descriptor.

The fitting of the following physical properties is supported
1. [`ener`](train-energy.md): Fit the energy of the system. The force (derivative with atom positions) and the virial (derivative with the box tensor) can also be trained.
2. [`dipole`](train-fitting-tensor.md): The dipole moment.
3. [`polar`](train-fitting-tensor.md): The polarizability.
