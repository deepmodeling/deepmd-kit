# Descriptor `"nep"` {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

The `nep` descriptor implements the neuroevolution potential (NEP) representation
of the local atomic environment introduced by Fan et al. and developed in
[GPUMD](https://github.com/brucefan1983/GPUMD). It is numerically equivalent to
the descriptor used by GPUMD's NEP4 models and, combined with the per-element
energy fitting network, reproduces the standard NEP architecture inside the
DeePMD-kit infrastructure (training, inference, and the LAMMPS/i-PI interfaces).

## Theory

The descriptor of atom $i$ is the concatenation of a radial part and an angular
part. Both parts expand the interatomic distances on the Chebyshev radial basis

```math
    g^{ij}_n = \sum_{k=0}^{N_\text{bas}} c^{t_i t_j}_{nk}\, f_k(r_{ij}),
    \qquad
    f_k(r_{ij}) = \tfrac{1}{2}\big[T_k(x) + 1\big]\, f_c(r_{ij}),
    \quad x = 2\left(\frac{r_{ij}}{r_c}-1\right)^2 - 1,
```

where $T_k$ is the Chebyshev polynomial of the first kind, the expansion
coefficients $c^{t_i t_j}_{nk}$ depend on the ordered pair of element types, and
the cutoff function is $f_c(r) = \tfrac{1}{2}\cos(\pi r / r_c) + \tfrac{1}{2}$ for
$r < r_c$ and $0$ otherwise.

The radial descriptor sums the radial embedding over the neighbors within
$r_c^R$,

```math
    q^i_n = \sum_{j \neq i} g^{ij}_n .
```

The angular descriptor contracts the (real) solid harmonics
$Y_{Lm}(\hat{\boldsymbol r}_{ij})$ of the neighbor directions within $r_c^A$ into
rotationally invariant combinations,

```math
    s^i_{nLm} = \sum_{j \neq i} g^{ij}_n\, Y_{Lm}(\hat{\boldsymbol r}_{ij}),
    \qquad
    q^i_{nL} = \sum_{m=-L}^{L} C_{Lm}\, (s^i_{nLm})^2 ,
```

for $L = 1, \dots, L_\text{max}$. Optional four-body ($L_\text{max}^{(4)}=2$) and
five-body ($L_\text{max}^{(5)}=1$) invariants formed from the $L=2$ and $L=1$
components are appended, exactly as in GPUMD. The full descriptor

```math
    \mathcal{D}^i = \big[\, q^i_n,\ q^i_{nL} \,\big]
```

is normalized element-wise by a fixed scaler $1/(q_{\max}-q_{\min})$ estimated
from the training data, which corresponds to the descriptor mean and standard
deviation in DeePMD-kit (zero mean, $q_{\max}-q_{\min}$ standard deviation).

## Instructions

A complete training input script can be found in the directory

```bash
$deepmd_source_dir/examples/water/nep/input.json
```

The {ref}`descriptor <model[standard]/descriptor>` section reads

```json
"descriptor": {
    "type": "nep",
    "sel": "auto",
    "rcut_radial": 8.0,
    "rcut_angular": 4.0,
    "n_max_radial": 6,
    "n_max_angular": 6,
    "basis_size_radial": 6,
    "basis_size_angular": 6,
    "l_max": 4,
    "l_max_4body": 2,
    "l_max_5body": 0,
    "seed": 1
}
```

The descriptor hyper-parameter defaults follow GPUMD's NEP4 (`rcut_radial = 8`,
`rcut_angular = 4`, `n_max = 6`, `basis_size = 6`, `l_max = 4, 2, 0`).

The two cutoffs {ref}`rcut_radial <model[standard]/descriptor[nep]/rcut_radial>`
and {ref}`rcut_angular <model[standard]/descriptor[nep]/rcut_angular>` control
the radial and angular ranges respectively, with
$r_c^A \leq r_c^R$. The descriptor dimension is
`(n_max_radial + 1) + (n_max_angular + 1) * num_L`, where `num_L` is
`l_max + (l_max_4body == 2) + (l_max_5body == 1)`. The radial and angular
Chebyshev basis sizes are
{ref}`basis_size_radial <model[standard]/descriptor[nep]/basis_size_radial>` `+ 1`
and
{ref}`basis_size_angular <model[standard]/descriptor[nep]/basis_size_angular>` `+ 1`.

The descriptor is always paired with a per-element energy fitting network (it
reports `mixed_types() == False`), matching GPUMD, which assigns a separate
network to each element. This keeps the trained model exportable to a
GPUMD-compatible `nep.txt`. For the same reason the descriptor does not support
type exclusion, per-type cutoffs, or a shared fitting network.

{ref}`precision <model[standard]/descriptor[nep]/precision>` defaults to
`float32` to match GPUMD. `float64` may be used for higher-precision training,
but GPUMD inference of an exported `nep.txt` is always single precision.

## Export to GPUMD `nep.txt`

A trained NEP energy model can be exported to a GPUMD-compatible `nep.txt` for
inference and molecular dynamics in GPUMD. The exporter reads a training
checkpoint directly, from the command line or as a Python function:

```bash
# PyTorch checkpoint (dp --pt-expt train) or JAX checkpoint (dp --jax train)
python -m deepmd.tools.nep_txt -i model.ckpt.pt -o nep.txt
python -m deepmd.tools.nep_txt -i model.ckpt.jax -o nep.txt
```

```python
from deepmd.tools.nep_txt import convert_to_nep_txt

convert_to_nep_txt("model.ckpt.pt", "nep.txt")
```

Exported PyTorch (`.pte`/`.pt2`) and JAX (`.savedmodel`) models are also
accepted. The exporter writes the NEP5 format. NEP5 (rather than NEP4) is required because
DeePMD-kit holds a per-element energy baseline (the fitting output-layer bias,
`bias_atom_e`, and the atomic-model `out_bias`), which NEP5 represents exactly
through its per-element bias; the descriptor is identical to NEP4. Export
requires a single-hidden-layer `tanh` fitting network. The conversion is
lossless, so the exported potential reproduces the trained model under GPUMD up
to GPUMD's single-precision arithmetic.

GPUMD recognizes only the first 94 elements (H–Pu). A model whose `type_map`
spans the full periodic table (for example to match an LMDB dataset) is
automatically narrowed to that set on export; pass `--elements` (CLI) or
`elements=` (Python) to select a specific subset and `nep.txt` column order.

## Difference among different backends

The descriptor produces identical values across the DP, PyTorch, and JAX
backends. The expansion coefficients are stored as a single dense tensor of
shape `(ntypes, ntypes, n_desc, k_max)`, so both the memory footprint and the
forward cost follow the neighbor count rather than `ntypes ** 2`.
