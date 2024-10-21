# Use deep potential with dpdata

:::{note}
See [Environment variables](../env.md) for the runtime environment variables.
:::

DeePMD-kit provides a driver for [dpdata](https://github.com/deepmodeling/dpdata) >=0.2.7 via the plugin mechanism, making it possible to call the `predict` method for `System` class:

```py
import dpdata

dsys = dpdata.LabeledSystem("OUTCAR")
dp_sys = dsys.predict("frozen_model_compressed.pb", driver="dp")
```

By inferring with the DP model `frozen_model_compressed.pb`, dpdata will generate a new labeled system `dp_sys` with inferred energies, forces, and virials.
