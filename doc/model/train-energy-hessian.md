# Fit energy Hessian

To train a model that takes Hessian matrices, i.e., the second order derivatives of energies w.r.t coordinates as input, you only need to prepare full Hessian matrices and modify the `loss` section to define the Hessian-specific settings, keeping other sections the same as the normal energy model's input script.

Note that fitting energy Hessians is only supported in the **PyTorch** backend as of now.


## Energy Hessian Loss

If you want to train with Hessians, you are expected to add the start and limit prefactors of Hessians, i.e.,`start_pref_h` and `limit_pref_h` to the `loss` section in the `input.json`:

```json
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "start_pref_h": 10,
        "limit_pref_h": 1
    },
```

The options `start_pref_e`, `limit_pref_e`, `start_pref_f`, `limit_pref_f`, `start_pref_v`, and `limit_pref_v`,  determine the start and limit prefactors of energy, force, and virial, respectively. The calculation and definition of Hessian loss are the same as for the other terms.

If one does not want to train with virial, they may set the virial prefactors `start_pref_v` and `limit_pref_v` to 0.


## Hessian format in PyTorch

In the PyTorch backend, Hessian matrices are listed in `hessian.npy` files, and the data format may contain the following files:

```
type.raw
set.*/box.npy
set.*/coord.npy
set.*/energy.npy
set.*/force.npy
set.*/hessian.npy
```

This system contains `Nframes` frames with the same atom number `Natoms`, the total number of elements contained in all frames is `Ntypes`. Most files are the same as those in [standard formats](../data/system.md), here we only list the distinct ones:

| ID             | Property         | Raw file      | Unit    | Shape                                   | Description                                                       |
| -------------- | ---------------- | ------------- | ------- | --------------------------------------- | ----------------------------------------------------------------- |
| hessian        | Hessian matrices | hessian.npy   | eV/Å^2  | Nframes \* (Natoms \* 3 \* Natoms \* 3) | Second-order derivatives of energies w.r.t coordinates.            |

Note that the `hessian.npy` should contain the **full** Hessian matrices with shape of `(3Natoms * 3Natoms)` for each frame, rather than the upper or lower triangular matrices with shape of `(3Natoms * (3Natoms + 1) / 2)` for each frame.
