# Determine `sel`

All descriptors require to set `sel`, which means the expected maximum number of type-i neighbors of an atom. DeePMD-kit will allocate memory according to `sel`.

`sel` should not be too large or too small. If `sel` is too large, the computing will become much slower and cost more memory. If `sel` is not enough, the energy will be not conserved, making the accuracy of the model worse.

To determine a proper `sel`, one can calculate the neighbor stat of the training data before training:

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```sh
dp --tf neighbor-stat -s data -r 6.0 -t O H
```

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

```sh
dp --pt neighbor-stat -s data -r 6.0 -t O H
```

:::

:::{tab-item} JAX {{ jax_icon }}

```sh
dp --jax neighbor-stat -s data -r 6.0 -t O H
```

:::

:::{tab-item} Paddle {{ paddle_icon }}

```sh
dp --pd neighbor-stat -s data -r 6.0 -t O H
```

:::

::::

where `data` is the directory of data, `6.0` is the cutoff radius, and `O` and `H` is the type map. The program will give the `max_nbor_size`. For example, `max_nbor_size` of the water example is `[38, 72]`, meaning an atom may have 38 O neighbors and 72 H neighbors in the training data.

The `sel` should be set to a higher value than that of the training data, considering there may be some extreme geometries during MD simulations. As a result, we set `sel` to `[46, 92]` in the water example.
