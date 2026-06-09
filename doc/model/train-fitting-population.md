# Fit atomic charge population {{ pytorch_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

Here we present an API to DeepPopulation model, which can be used to fit the atomic charge population.

See the [preprint on arXiv](https://arxiv.org/abs/2606.01763) for details and an example of how the DeepPopulation model was used to study small polaron transport.

In this example, we will show you how to train a model to fit the atomic charge population for a titanium dioxide system. A complete training input script of the examples can be found in

```bash
$deepmd_source_dir/examples/population/train/input_torch.json
```

Similar to the `input.json` used in `ener` mode, training JSON is also divided into {ref}`model <model>`, {ref}`learning_rate <learning_rate>`, {ref}`loss <loss>` and {ref}`training <training>`. Most keywords remain the same as `ener` mode, and their meaning can be found in the [DPA-3 training guide](dpa3.md). To fit the `population`, one needs to modify {ref}`model[standard]/fitting_net <model[standard]/fitting_net>` and {ref}`loss <loss>`.

## The fitting Network

The {ref}`fitting_net <model[standard]/fitting_net>` section tells DP which fitting net to use.

The JSON of `population` type should be provided like

```json
	"fitting_net" : {
		"type": "population",
	},
```

- `type` specifies which type of fitting net should be used. It should be `population`.
- The rest arguments have the same meaning as they do in `ener` mode.

## Loss

The loss function for DeepPopulation has the form:

$$
\begin{equation}
\begin{aligned}
\mathcal{L} =
& \lambda_{N_{\alpha}} \ell\!\left( N^{ML}_{\alpha},\, N^{DFT}_{\alpha} \right)  +
\\ &  \lambda_{N_{\beta}} \ell\!\left( N^{ML}_{\beta},\, N^{DFT}_{\beta} \right) +
\\ & \lambda_S \ell\!\left( S^{ML},\, S^{DFT} \right) +
\\ & \frac{\lambda_s}{N} \sum_{i=1}^{N} \ell\!\left( s_{i}^{ML},\, s_{i}^{DFT} \right) +
\\ & \frac{\lambda_\sigma}{N} \sum_{i=1}^{N} \ell\!\left( \sigma_{i}^{ML},\, \sigma_{i}^{DFT} \right)
\end{aligned}
\end{equation}
$$

where $\ell(\hat{y}, y)$ is the chosen element-wise loss function (`mae`, `smooth_mae`, or `rmse`).

This includes contributions from:

- The total number of ( \\alpha ) electrons ( N\_{\\alpha} ),
- The total number of ( \\beta ) electrons ( N\_{\\beta} ),
- The total spin ( S ),
- The atomic spin moments ( s\_\{i} ),
- The atomic populations ( \\sigma ).

The loss section should be provided like

```json
	"loss": {
		"type": "population",
		"start_pref_spin": 1,
		"limit_pref_spin": 1,
		"start_pref_spin_total": 1,
		"limit_pref_spin_total": 1,
		"start_pref_pop": 1000,
		"limit_pref_pop": 1,
		"start_pref_pop_alpha_total": 1,
		"limit_pref_pop_alpha_total": 1,
		"start_pref_pop_beta_total": 1,
		"limit_pref_pop_beta_total": 1,
		"loss_func": "mae",
		"metric": [
			"mae"
		],
	},
```

## Training Data Preparation

The atomic electronic population file should be named `atomic_population.npy`, with shape [num_frames, num_atoms, 2] where [num_frames, num_atoms, 0] represents the alpha spin channel and [num_frames, num_atoms, 1] represents the beta spin channel.

## Train the Model

The training command is the same as `ener` mode, i.e.

::::{tab-set}

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
dp --pt train input_torch.json
```
:::

::::

The detailed loss can be found in `lcurve.out`:

```text
# step    pop_alpha_total_loss_trn   pop_beta_total_loss_trn   pop_loss_trn   spin_loss_trn   spin_total_trn   spin_total_loss_trn   lr
      1      5.74e+01      2.15e+02      3.95e+02      2.73e+02      2.73e+02      2.72e+02    1.0e-03
   1000      9.26e-02      3.10e-01      6.74e+00      9.13e-01      1.33e+00      2.17e-01    3.1e-05

```
