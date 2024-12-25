# Fit energy Hessian {{ pytorch_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

To train a model that takes Hessian matrices, i.e., the second order derivatives of energies w.r.t coordinates as input, you only need to prepare full Hessian matrices and modify the `loss` section to define the Hessian-specific settings, keeping other sections the same as the normal energy model's input script.

## Energy Hessian Loss

If you want to train with Hessians, you are expected to add the start and limit prefactors of Hessians, i.e., {ref}`start_pref_h <loss[ener_hess]/start_pref_h>` and {ref}`limit_pref_h <loss[ener_hess]/limit_pref_h>` to the {ref}`loss <loss>` section in the `input.json`:

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

The options {ref}`start_pref_e <loss[ener_hess]/start_pref_e>`, {ref}`limit_pref_e <loss[ener_hess]/limit_pref_e>`, {ref}`start_pref_f <loss[ener_hess]/start_pref_f>`, {ref}`limit_pref_f <loss[ener_hess]/limit_pref_f>`, {ref}`start_pref_v <loss[ener_hess]/start_pref_v>` and {ref}`limit_pref_v <loss[ener_hess]/limit_pref_v>` determine the start and limit prefactors of energy, force, and virial, respectively. The calculation and definition of Hessian loss are the same as for the other terms.

If one does not want to train with virial, then he/she may set the virial prefactors {ref}`start_pref_v <loss[ener_hess]/start_pref_v>` and {ref}`limit_pref_v <loss[ener_hess]/limit_pref_v>` to 0.

## Hessian Format in PyTorch

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

| ID      | Property         | Raw file    | Unit   | Shape                                   | Description                                             |
| ------- | ---------------- | ----------- | ------ | --------------------------------------- | ------------------------------------------------------- |
| hessian | Hessian matrices | hessian.npy | eV/Å^2 | Nframes \* (Natoms \* 3 \* Natoms \* 3) | Second-order derivatives of energies w.r.t coordinates. |

Note that the `hessian.npy` should contain the **full** Hessian matrices with shape of `(3Natoms * 3Natoms)` for each frame, rather than the upper or lower triangular matrices with shape of `(3Natoms * (3Natoms + 1) / 2)` for each frame.

## Train the Model

There are two approaches to training a Hessian model. The first method involves training the model from scratch using the same command as in the `ener` mode within the PyTorch backend:

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
dp --pt train input.json
```

:::

::::

The second approach is to train a Hessian model from a pretrained energy model, following the same command as the `finetune` strategy within the PyTorch backend:

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
dp --pt train input.json --finetune pretrained_energy.pt
```

:::

::::

The detailed loss can be found in `lcurve.out`:

```
#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn    rmse_h_val  rmse_h_trn         lr
      0      1.05e+02    2.28e+01      2.11e-01    1.59e+00      3.25e+00    3.37e-01      6.00e+00    6.37e+00    1.0e-03
    200      1.86e+01    3.23e+01      9.24e-03    1.54e-01      2.51e-01    4.70e-01      5.31e+00    9.05e+00    1.0e-03
    400      2.69e+01    2.98e+01      1.03e-01    1.07e-01      5.67e-01    4.17e-01      6.35e+00    8.47e+00    1.0e-03
    600      2.00e+01    1.90e+01      7.23e-02    6.90e-03      3.35e-01    2.58e-01      5.37e+00    5.41e+00    1.0e-03
    800      1.68e+01    1.48e+01      4.06e-02    2.27e-01      2.35e-01    1.98e-01      4.76e+00    4.24e+00    1.0e-03
   1000      1.70e+01    1.81e+01      3.90e-01    1.66e-01      2.02e-01    1.99e-01      4.98e+00    5.37e+00    1.0e-03
```

## Test the Model

:::{warning}
A model trained with Hessian cannot be frozen. If freezing is enforced, the model will be treated as a standard energy model, and the frozen one will no longer be able to output Hessian predictions.
:::

If one do freeze and test a Hessian model using the commands:

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash

dp --pt freeze -o frozen_model.pth

dp --pt test -m frozen_model.pth -s test_system -d ${output_prefix} -a -n 1
```

:::

::::

If `dp --pt test -d ${output_prefix} -a` is specified, the output files will be the same as those in the `ener` mode, i.e.,

```
${output_prefix}.e.out   ${output_prefix}.e_peratom.out  ${output_prefix}.f.out
${output_prefix}.v.out   ${output_prefix}.v_peratom.out
```

If one intends to use the trained model for Hessian predictions, then he/she is supposed to test the model directly without performing a freezing operation:

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash

dp --pt test -m model.pt -s test_system -d ${output_prefix} -a -n 1
```

:::

::::

If `dp --pt test -d ${output_prefix} -a` is specified, the predicted Hessian for each frame are output in an additional file in the working directory:

```
${output_prefix}.h.out
```

For `*.h.out.*`, it contains matrix with shape of `(2, n_hess)`:

```
# frame - 0: data_h pred_h (3Na*3Na matrix in row-major order)
5.897392891323943331e+01 2.909700516268236825e+01
-7.682282297964052376e+00 2.535680817045881774e+00
-1.266442953072092514e+01 -2.127310638041492652e+01
5.442541716174009031e-02 7.202825779190234756e-02
5.198263170894957939e-05 -8.110080221576332349e-02
7.443552765043950914e-02 -2.248597801730128215e-02
1.029910175689553675e+00 1.938646932394622047e-03
1.213862217511276764e+00 5.344132558814301825e-02
-1.221943904909605250e+00 1.602557574981743893e-01
```

The full Hessian matrices are stored in a flattened form in the row-major order. Here, `n_hess` is the total number of Hessian matrix elements across all frames, calculated as:

```math
n_\text{hess} = \sum_{i} 3N_{\text{atom}, i}*3N_{\text{atom}, i}
```

where $N_{\text{atom}, i}$ represents the number of atoms in the $i^{\text{th}}$ frame.
