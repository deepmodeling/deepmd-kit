# Input for the DPA4C model

This directory stores a configuration file for training DPA4C, the compact and
compressible degree-wise member of the DPA4 family. It runs on the pt_expt
backend:

```bash
dp --pt-expt train input.json
```

DPA4C is built for extreme-speed molecular dynamics, so its arguments are best
read as a budget split between two quantities: inference throughput and the
largest system that fits in memory.

## Descriptor arguments

`channels` and `lmax` set every derived width. They are the only arguments that
grow the persistent equivariant node state, the per-atom tensor the descriptor
carries from the neighbor reduction into the readout, so raising either one
lowers both throughput and capacity.

`radial_modes` lets each ordered atom-type pair combine several shared radial
shapes instead of rescaling a single one. It leaves the node state untouched
and spends per-edge work instead, which makes it the lever to reach for when
memory rather than throughput is the binding constraint. The compressed CUDA
path is compiled for the values `0`, `2`, `4` and `8`; a model trained with any
other value runs on the portable path but cannot be compressed.

`basis_type` and `n_radial` select the analytic radial basis feeding the radial
network.

`precision` must be `float32` for the compressed CUDA path. `use_amp` is an
execution policy rather than model state: it is read from this file for
training, while evaluation and inference read the `DP_AMP_INFER` environment
variable. The two are independent, so a model trained in full precision can
still be evaluated under mixed precision.

## Choosing the fitting width

The fitting network is sized against the descriptor, because the invariant
output grows with `channels`. The released grades pair `channels` 8, 32, 64 and
128 with hidden widths 96, 192, 256 and 384, at depth three as used here. This
file is the `Neo` grade: `channels: 32` with `radial_modes: 4` and a hidden
width of 192.

Unlike `radial_modes`, fitting width is not a free trade against memory. It
does not enlarge the persistent node state, but it does add per-atom
activations and the derivatives saved for the force backward pass, so widening
or deepening it costs throughput and capacity together.

Widen the fitting only when validation error is limited by fitting capacity
rather than by the descriptor; when either throughput or system size is
binding, spend the budget on the descriptor instead.
