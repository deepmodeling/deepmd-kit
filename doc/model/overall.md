# Overall

A model has two parts, a descriptor that maps atomic configuration to a set of symmetry invariant features, and a fitting net that takes descriptor as input and predicts the atomic contribution to the target physical property. It's defined in the `model` section of the `input.json`, for example
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

Assume that we are looking for a model for water, we will have two types of atoms. The atom types are recorded as integers. In this example, we denote `0` for oxygen and `1` for hydrogen. A mapping from the atom type to their names is provided by `type_map`. 

The model has two subsections `descritpor` and `fitting_net`, which defines the descriptor and the fitting net, respectively. The `type_map` is optional, which provides the element names (but not necessarily to be the element name) of the corresponding atom types.

DeePMD-kit implements the following descriptors:
1. [`se_e2_a`](train-se-e2-a.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes the distance between atoms as input.
2. [`se_e2_r`](train-se-e2-r.md): DeepPot-SE constructed from radial information of atomic configurations. The embedding takes the distance between atoms as input.
3. [`se_e3`](train-se-e3.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes angles between two neighboring atoms as input.
4. `loc_frame`: Defines a local frame at each atom, and the compute the descriptor as local coordinates under this frame.
5. [`hybrid`](train-hybrid.md): Concate a list of descriptors to form a new descriptor.

The fitting of the following physical properties are supported
1. [`ener`](train-energy.md): Fitting the energy of the system. The force (derivative with atom positions) and the virial (derivative with the box tensor) can also be trained. See [the example](train-se-e2-a.md#loss).
2. [`dipole`](train-fitting-tensor.md): The dipole moment.
3. [`polar`](train-fitting-tensor.md): The polarizability.
