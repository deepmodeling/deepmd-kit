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
Coulomb constant. Every channel interacts through the same kernel.

For a non-periodic system, the sum runs over the distinct pairs of each frame:
it has no self term, and no periodic image. For a periodic system (with
`periodic: true`, see below) the kernel is summed over the images of the cell
as well, in reciprocal space.

The charges depend on the atomic environments, so forces and virials, which
are obtained by differentiating the energy, include the response of the
charges to the positions as well as the direct dependence of the kernel on
the distance. Every atom carries half of the interaction of its charges with
all others, so the atomic energies still sum to the energy of the frame. Every quantity
derived from the model sees the complete energy: training, `dp test`, the
output-bias statistics and change-bias, the Hessian, and a linear
composition of models.

> [!WARNING]
> The sum covers complete systems. Without `periodic: true`, a periodic cell
> is rejected. The cost of the non-periodic sum grows with the square of the
> number of atoms of a frame.

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
interaction of the charges, `les` by default. Both kernels are trained with
the model unless their `trainable` option is false, and their widths stay
positive throughout training.

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
  $0.46$ Å⁻¹, at contact, and its radial force between fixed charges stays
  within 0.4% of that of $1/r$ from 6.5 Å to 150 Å. It differs from $1/r$
  there by a nearly constant offset $C \approx 1.3 \times 10^{-3}$ Å⁻¹, which
  exerts no force between fixed charges. The charges respond to the positions,
  however, and through them the offset adds
  $-k_e C \sum_c (Q_c^2 - \sum_i (q_i^c)^2)/2$ to the energy of a frame, with
  $Q_c$ the total charge of channel $c$, which the response of the charges
  turns into a force. Below about 5 Å the kernel falls short of $1/r$ by more
  than $C$, by 0.12 Å⁻¹ at 2 Å.

Several charge channels, set by
{ref}`channels <model[standard]/long_range[latent_charge]/channels>`, let
the model represent interactions that one charge per atom cannot.

### Periodic systems

With
{ref}`periodic <model[standard]/long_range[latent_charge]/periodic>` enabled,
a frame given a cell is evaluated over the images of the cell. The sum
converges in reciprocal space, where the Fourier transform of the kernel,

- `les`: $\hat K(\mathbf{k}) = (4\pi/k^2)\, e^{-k^2/4\alpha^2}$, or
- `sog`: $\hat K(\mathbf{k}) = \sum_k a_k (2\pi)^{3/2} \sigma_k^3\, e^{-\sigma_k^2 k^2/2}$,
  which with the default amplitudes approaches the Coulomb transform
  $4\pi/k^2$,

decays like a Gaussian. Over the wave numbers $\mathbf{k} = 2\pi B^{-1}
\mathbf{n}$, $\mathbf{n} \in \mathbb{Z}^3 \setminus \{\mathbf{0}\}$, of a cell
$B$ whose rows are the lattice vectors,

```math
E = k_e \left[\frac{1}{2V}\sum_{\mathbf{k}\neq\mathbf{0}} \sum_{c=1}^{C}
\hat K(\mathbf{k})\, |\hat\rho_c(\mathbf{k})|^2
- \frac{1}{2} K(0) \sum_{i,c} (q_i^c)^2 \right],
\qquad \hat\rho_c(\mathbf{k}) = \sum_i q_i^c e^{i\mathbf{k}\cdot\mathbf{r}_i},
```

with the cell volume $V$ and the kernel value at contact $K(0)$, whose term
removes the interaction of every charge with itself. The
$\mathbf{k}=\mathbf{0}$ mode is dropped, the tin-foil convention, which places
a uniform neutralizing background under a channel with a net charge. The
wave-number cutoff $k_{max}$ is where the weight of the slowest-decaying mode
falls below
{ref}`kspace_tolerance <model[standard]/long_range[latent_charge]/kspace_tolerance>`,
so the truncation converges exponentially; it follows the kernel as it trains,
unless
{ref}`decouple_k_grid <model[standard]/long_range[latent_charge]/decouple_k_grid>`
keeps the cutoff of the initial kernel.

{ref}`k_grid_mode <model[standard]/long_range[latent_charge]/k_grid_mode>`
selects how training evaluates the sum:

- `pme`, the default: by the smooth particle-mesh Ewald method. The charges
  are spread with B-splines of order
  {ref}`pme_order <model[standard]/long_range[latent_charge]/pme_order>` to a
  mesh that resolves every cell of the batch at
  {ref}`pme_spacing <model[standard]/long_range[latent_charge]/pme_spacing>`,
  whose transform gives the sum; the cost grows nearly linearly with the
  number of atoms. With the default order 8 and spacing $0.75\pi/k_{max}$, the
  energy and the forces stay within a few parts in a million of the sum over
  the wave numbers.
- `ragged`: over the wave numbers of every frame within $k_{max}$, listed for
  every batch; their number grows with the volume of the cell.
- `fixed`: over one integer grid shared by the frames of a batch, sized by the
  largest cell of the data statistics or by
  {ref}`k_grid_bounds <model[standard]/long_range[latent_charge]/k_grid_bounds>`;
  a frame whose cell needs a wave number beyond the grid is truncated to it,
  with a warning.

A frozen model evaluates every periodic frame on the mesh of its cell,
whatever its training mode, so a model trained in another mode can differ
from its frozen form by the accuracy of the mesh. Forces and virials hold the
complete periodic term, and a model with `periodic: true` still evaluates
non-periodic frames by the direct sum. In a `linear_ener` composition, a
sub-model whose module serves periodic systems takes no other sub-model with
a long-range module beside it.

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
the descriptor act on the descriptor alone. In a `linear_ener` composition,
`pair_exclude_types` belongs to the composition, whose sub-models share its
neighbor graph and pair topology. The reciprocal-space sum of a periodic
system is global and excludes no pair, so `pair_exclude_types` is not
supported with a module that serves periodic systems, nor
`atom_exclude_types` with any long-range module.

## Training and inference

Training uses the usual energy loss; see
[training energy models](train-energy.md). With `enable_compile`, the
long-range interaction is part of the compiled model: the pairs, the wave
numbers or the mesh of each batch are built outside the compiled region and
enter it as inputs of dynamic size.

`dp --pt-expt freeze` exports the model, long-range module included, to
`.pt2` or `.pte`, and `dp --pt-expt compress` compresses its descriptor into a
`.pt2` archive while the networks of the long-range module stay as they are.
A trained checkpoint (`.pt`) and the frozen model are evaluated by `dp test`
and the Python inference interface; the C++ interface and LAMMPS, with
`pair_style deepmd` and `pair_style deepmd/kk`, read the `.pt2` archive.
Since the sum covers the complete system, LAMMPS runs a model whose module
serves no periodic system with non-periodic boundaries (`f`, `s` or `m`) in
every dimension, and a model with `periodic: true` with boundaries periodic in
every dimension or in none. Several models given to `pair_style deepmd` for
the model deviation evaluate a non-periodic system together only when every
one of them carries a long-range module. A model that the dedicated operator described
below evaluates, a compressed DPA4C model frozen for a CUDA device, also runs
on several MPI ranks: every rank gathers the positions and charges of the
atoms of all ranks over MPI and evaluates its own atoms against all of them,
so that the ranks share the cost of the direct sum, while each evaluates the
mesh of a periodic system whole. This requires DeePMD-kit built with MPI. Any
other model, like an uncompressed one, runs on a single rank;
`pair_style deepmd/kk` evaluates an uncompressed model on non-periodic systems
only.

Compressing a descriptor conditioned on `charge_spin` specializes the model
to one charge state at a time, and a constraint that reads its total charge
from `charge_spin` follows the same state. The compressed archive serves any
charge state: the Python interface applies the state of each call, and LAMMPS
the one given by the `charge_spin` keyword of `pair_style deepmd`.

A compressed DPA4C model frozen for a CUDA device evaluates the long-range
term with a dedicated operator, which reads the positions of the atoms and
the cell. It enumerates the pairs of a non-periodic system itself, so that its
memory grows linearly with the number of atoms and its cost with their
square, and evaluates a periodic system on the mesh of its cell. The operator
serves the `les` kernel and the `sog` kernel of up to 32 Gaussians, with up to
8 charge channels, B-splines of order up to 12, a charge network without
frame or atomic parameters, and no `pair_exclude_types`. Any other model,
like an uncompressed one, receives the pairs or the mesh of each frame from
the interface, so the memory of a non-periodic evaluation grows with the
square of the number of atoms.

The following are not supported for a model with a long-range module:
multi-rank (domain-decomposed) inference of a model that the dedicated
operator does not evaluate, such as an uncompressed one, spin models, the
tabulated ZBL potential, and short-range bridging.
