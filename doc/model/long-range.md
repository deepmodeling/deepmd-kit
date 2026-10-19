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
  which explicit widths require. The default amplitudes,
  $a_k = \sqrt{2/\pi}\,\ln b/\sigma_k$ for the ratio $b$ of consecutive
  widths, discretize $1/r$: with the default widths the kernel stays finite,
  $0.46$ Å⁻¹, at contact, and its force stays within 0.4% of that of $1/r$
  from 6.5 Å to 150 Å. It differs from $1/r$ there by a constant, which
  exerts no force.

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
`.pt2` or `.pte`, and `dp --pt-expt compress` compresses its descriptor into a
`.pt2` archive while the networks of the long-range module stay as they are.
A trained checkpoint (`.pt`) and the frozen model are evaluated by `dp test`
and the Python inference interface; the C++ interface and LAMMPS, with
`pair_style deepmd` and `pair_style deepmd/kk`, read the `.pt2` archive.
Since the pair sum covers the complete system, LAMMPS runs such a model with
non-periodic boundaries (`f`, `s` or `m`) in every dimension on a single MPI
rank, and rejects any other setting.

Compressing a descriptor conditioned on `charge_spin` specializes the model
to one charge state at a time, and a constraint that reads its total charge
from `charge_spin` follows the same state. The compressed archive serves any
charge state: the Python interface applies the state of each call, and LAMMPS
the one given by the `charge_spin` keyword of `pair_style deepmd`.

A compressed DPA4C model frozen for a CUDA device evaluates the pair sum with
a dedicated operator, which reads the positions of the atoms and enumerates
their pairs itself: its memory grows linearly with the number of atoms, and
its cost with their square. The operator serves the `les` kernel and the
`sog` kernel of up to 32 Gaussians, with up to 8 charge channels, a charge
network without frame or atomic parameters, and no `pair_exclude_types`. Any
other model, like an uncompressed one, lists the pairs of each frame before
calling the model, so its memory grows with the square of the number of
atoms.

The following are not supported for a model with a long-range module:
multi-rank (domain-decomposed) inference, spin models, the tabulated ZBL
potential, and short-range bridging.
