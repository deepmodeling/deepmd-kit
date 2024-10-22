# Floating-point precision of the model

The following options control the precision of the model:

- The environment variable {envvar}`DP_INTERFACE_PREC` controls the interface precision of the model, the descriptor, and the fitting, the precision of the environmental matrix, and the precision of the normalized parameters for the environmental matrix and the fitting output.
- The training parameter {ref}`model[standard]/fitting_net[ener]/precision <precision>` controls the precision of neural networks in the descriptor and the fitting, and the subsequent operations after the output of neural networks.
- The reduced output (e.g. total energy) is always `float64`.

Usually, the following two combinations of options are recommended:

- Setting {envvar}`DP_INTERFACE_PREC` to `high` (default) and all {ref}`model[standard]/fitting_net[ener]/precision <precision>` options to `float64` (default).
- Setting {envvar}`DP_INTERFACE_PREC` to `high` (default) and all {ref}`model[standard]/fitting_net[ener]/precision <precision>` options to `float32`.

The Python and C++ inference interfaces accept both `float64` and `float32` as the input and output arguments, whatever the floating-point precision of the model interface is.
Usually, the MD programs (such as LAMMPS) only use `float64` in their interfaces.
