# Fit charge density on grid points {{ pytorch_icon }}

> [!NOTE]
> **Supported backends**: PyTorch-TorchScript {{ pytorch_icon }}

Here we present an API to the grid density model, which predicts the charge density on a set of grid points for a given atomic configuration.

A complete example can be found in

```bash
$deepmd_source_dir/examples/density/
```

with DPA-2 and DPA-3 configurations, a small QM9 dataset, and a README describing the data format in detail. Note that **the data provided along with the examples are of limited amount, and should not be used to train a production model.**

## Data format

The training/validation data follows the standard `deepmd/npy` format, with two additional files in each `set.000/` directory:

- `grid.npy` of shape `[nframes, ngrid, 3]`: the coordinates of the grid points;
- `density.npy` of shape `[nframes, ngrid, 1]`: the charge density labels on the grid points.

The number of grid points `ngrid` may differ from the number of atoms. The last entry of `type_map` is reserved as a virtual "grid point type": internally, grid points are assigned this type when building the grid-to-atom neighbor list, so `type_map` must contain one more entry than the real element types. For descriptors with a per-type `sel` list (e.g. `se_e2_a`), set its last entry to 0, since grid points are only centers, never neighbors.

It is recommended to set a positive `env_protection` in the descriptor configuration (e.g. `"env_protection": 0.1`). Grid points may legitimately coincide with atoms, and the default `env_protection = 0.0` would let the `1/r` terms in the environment matrix produce `NaN` densities. If `env_protection` is not set for a density model, it is automatically set to `1e-6` with a warning.

## The fitting network

The {ref}`fitting_net <model[standard]/fitting_net>` section tells DP which fitting net to use.

The JSON of `density` type should be provided like

```json
	"fitting_net" : {
		"type": "density",
		"neuron": [120,120,120],
		"resnet_dt": true,
		"seed": 1
	},
```

- `type` specifies which type of fitting net should be used. It should be `density`.
- The rest arguments have the same meaning as they do in `ener` mode. `numb_aparam` is not supported, because the fitting net consumes the descriptor rows of the grid points, which have no per-atom parameters.

## Loss

The grid density model is trained with the `grid_density` loss, which minimises the squared error between the predicted and the reference densities on all grid points, consistent with the other losses in the package:

```json
	"loss" : {
		"type": "grid_density",
		"start_pref_d": 1.0,
		"limit_pref_d": 1.0
	},
```

- {ref}`type <loss/type>` should be written as `grid_density`.
- `start_pref_d` and `limit_pref_d` specify the weight of the density loss at the start and at the end of the training. If both are set to 0, the density label is not required and the term is skipped. Systems without a `density.npy` file are also skipped (with `find_density = 0`) when the label is optional.

## Evaluation

`dp test` evaluates the model on a system containing `grid.npy` and `density.npy` and reports the MAE and RMSE of the density. For programmatic access, `DeepEval` dispatches density models to `DeepDensity`, whose `eval` takes the grid coordinates and returns the density on the grid points:

```python
from deepmd.infer import DeepEval

dp = DeepEval("model.pth")
density = dp.eval(coord, box, atype, grid=grid)  # (nframes, ngrid)
```
