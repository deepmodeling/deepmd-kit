# DPA4C-LR: DPA4C with a non-periodic long-range term {{ pytorch_icon }}

> [!NOTE]
> **Supported backends**: PyTorch Exportable {{ pytorch_icon }} (`dp --pt-expt`)

DPA4C-LR augments the compact [DPA4C](./dpa4c.md) energy model with a
long-range correction evaluated from per-atom latent charges predicted by a
dedicated fitting branch. The total energy is

```math
E = E_{\text{SR}} + E_{\text{LR}}(\{q_i\}, \{\boldsymbol r_i\}),
```

where $E_{\text{SR}}$ is the standard short-range DPA4C energy and
$E_{\text{LR}}$ is a non-periodic charge-charge interaction. Two kernels are
available for the long-range term, selected by
{ref}`lr_kernel <model[dpa4c_lr]/fitting_net[dpa4c_lr]/lr_kernel>`:

- `les` (default): the LES kernel $\mathrm{erf}(\alpha r)/r$, which behaves as
  $1/r$ at long range. The width $\alpha$ is trainable
  ({ref}`les_alpha <model[dpa4c_lr]/fitting_net[dpa4c_lr]/les_alpha>`) and a
  single latent charge per atom is used.
- `sog`: a trainable sum-of-Gaussians kernel over distinct atom pairs
  ($i \neq j$), initialized from the hyper-parameters
  {ref}`b <model[dpa4c_lr]/fitting_net[dpa4c_lr]/b>`,
  {ref}`sigma <model[dpa4c_lr]/fitting_net[dpa4c_lr]/sigma>`, and
  {ref}`M <model[dpa4c_lr]/fitting_net[dpa4c_lr]/M>`. Multiple latent-charge
  channels are supported via
  {ref}`dim_out_lr <model[dpa4c_lr]/fitting_net[dpa4c_lr]/dim_out_lr>`.

Forces and virials are obtained by differentiating the total energy, including
the response of the latent charges to the atomic positions, so the model
remains conservative.

:::{warning}
Only **non-periodic systems** (no simulation box) are supported. Periodic
long-range summation (Ewald/SPME-style) is not implemented. Freezing to
`.pt2`/`.pte` is not yet implemented for this model type.
:::

## Training input

DPA4C-LR is selected as a model scaffold, `model.type: "dpa4c_lr"`, with the
matching fitting type `dpa4c_lr`:

```json
"model": {
    "type": "dpa4c_lr",
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 6.0,
        "channels": 64,
        "lmax": 2,
        "n_radial": 8
    },
    "fitting_net": {
        "type": "dpa4c_lr",
        "neuron": [128, 128],
        "dim_out_lr": 1,
        "neuron_lr": [64, 64],
        "lr_kernel": "les",
        "les_alpha": 1.0,
        "use_charge_constraint": false
    }
}
```

For the SOG kernel, set `lr_kernel` to `"sog"`; the kernel amplitudes and
bandwidths can be initialized explicitly with
{ref}`amp <model[dpa4c_lr]/fitting_net[dpa4c_lr]/amp>` and
{ref}`bandwidth <model[dpa4c_lr]/fitting_net[dpa4c_lr]/bandwidth>`, and
multiple latent-charge channels can be used by setting `dim_out_lr` larger
than 1.

The latent-charge branch follows the standard energy fitting options (e.g.
`neuron`, `precision`, `seed`); see
{ref}`the fitting_net section <model[dpa4c_lr]/fitting_net[dpa4c_lr]>` for the
full argument list. See [training energy models](train-energy.md) for the
general workflow shared by all energy models.
