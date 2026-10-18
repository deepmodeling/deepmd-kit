# Long-range latent charges {{ pytorch_icon }}

> [!NOTE]
> **Supported backends**: PyTorch Exportable {{ pytorch_icon }} (`dp --pt-expt`)

A short-range model sees each atom only through the neighbors within its
cutoff. Interactions that decay slowly with distance, such as those between
charged groups of a molecule, therefore escape it. The long-range module adds
such an interaction to the energy of a standard model. A second network reads
the same descriptor as the energy fitting and predicts per-atom _latent
charges_, and every pair of atoms of a system interacts through those charges:

```math
E = \sum_i E_i^{\text{SR}} + k_e \sum_{i<j} \sum_{c=1}^{C} q_i^c q_j^c\, K(r_{ij}),
```

where $E_i^{\text{SR}}$ is the atomic energy of the fitting, $q_i^c$ is the
charge of atom $i$ in channel $c$ (in units of the elementary charge),
$K(r)$ is the interaction kernel in Å⁻¹, and $k_e = 14.3996$ eV Å is the
Coulomb constant. Every channel interacts through the same kernel. The sum
runs over the distinct pairs of each frame only: it has no self term, and
no periodic image.

The charges depend on the atomic environments, so forces and virials, which
are obtained by differentiating the energy, include the response of the
charges to the positions as well as the direct dependence of the kernel on
the distance. Half of each pair energy is attributed to each of its atoms,
so the atomic energies still sum to the energy of the frame. Every quantity
derived from the model sees the complete energy: training, `dp test`, the
output-bias statistics and change-bias, the Hessian, and a linear
composition of models.

> [!WARNING]
> The pair sum covers complete non-periodic systems: frames without a cell,
> such as isolated molecules and clusters. A periodic cell is rejected. The
> cost of the sum grows with the square of the number of atoms of a frame.

## Configuration

The module is a `long_range` section of a standard energy model, beside the
descriptor and the fitting. It is currently available with the
[DPA4C](./dpa4c.md) descriptor:

```json
"model": {
    "type_map": ["C", "H", "O"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 6.0
    },
    "fitting_net": {
        "neuron": [128, 128, 128]
    },
    "long_range": {
        "type": "latent_charge",
        "channels": 1,
        "kernel": {
            "type": "les",
            "alpha": 1.0
        },
        "charge_constraint": {
            "source": "neutral"
        }
    }
}
```

The charge network,
{ref}`charge_net <model[standard]/long_range[latent_charge]/charge_net>`,
takes the frame and atomic parameters of the energy fitting and follows its
activation function, skip connection and precision unless they are set; its
seed defaults to one derived from the seed of the fitting.
Its width is set by
{ref}`neuron <model[standard]/long_range[latent_charge]/charge_net/neuron>`.

### Kernels

{ref}`kernel <model[standard]/long_range[latent_charge]/kernel>` selects the
interaction of the charges. Both kernels are trained with the model unless
their `trainable` option is false, and their widths stay positive throughout
training.

- `les`: $K(r) = \mathrm{erf}(\alpha r)/r$, the interaction of two Gaussian
  charge clouds. It approaches $1/r$ beyond a few $1/\alpha$ and stays finite,
  $2\alpha/\sqrt{\pi}$, at contact. The inverse width $\alpha$ is set by
  {ref}`alpha <model[standard]/long_range[latent_charge]/kernel[les]/alpha>`.
- `sog`: $K(r) = \sum_k a_k \exp(-r^2/2\sigma_k^2)$, a sum of Gaussians whose
  amplitudes $a_k$ may take either sign. The widths are given by
  {ref}`width <model[standard]/long_range[latent_charge]/kernel[sog]/width>`
  or default to a geometric sequence, and the amplitudes by
  {ref}`amplitude <model[standard]/long_range[latent_charge]/kernel[sog]/amplitude>`,
  which explicit widths require.

Several charge channels, set by
{ref}`channels <model[standard]/long_range[latent_charge]/channels>`, let
the model represent interactions that one charge per atom cannot.

### Total charge

{ref}`charge_constraint <model[standard]/long_range[latent_charge]/charge_constraint>`
shifts one channel uniformly over the atoms of every frame so that it sums to
the total charge of the frame. The total charge is zero (`neutral`), the
charge of the frame charge/spin condition of the descriptor (`charge_spin`),
or a column of the frame parameters (`fparam`); a frame without the input
takes its default. The constrained channel then carries the total charge,
while the other channels remain free. Without the section, no channel is
constrained.

### Exclusions

Pairs of the types in `pair_exclude_types` see each other neither in the
descriptor nor in the long-range interaction, while the `exclude_types` of
the descriptor act on the descriptor alone. `atom_exclude_types` is not
supported with a long-range module.

## Training and inference

Training uses the usual energy loss; see
[training energy models](train-energy.md). With `enable_compile`, the
long-range interaction is part of the compiled model: the pairs of each frame
are listed outside the compiled region and enter it as inputs of dynamic
length.

`dp --pt-expt freeze` exports the model, long-range module included, to
`.pt2` or `.pte`. A trained checkpoint (`.pt`) and the frozen model are
evaluated by `dp test` and the Python inference interface, which list the
pairs of each frame before calling the model. The C++ interface, and with it
LAMMPS, does not evaluate a model with a long-range module.

The following are not supported for a model with a long-range module:
compression, multi-rank (domain-decomposed) inference, spin models, the
tabulated ZBL potential, and short-range bridging.
