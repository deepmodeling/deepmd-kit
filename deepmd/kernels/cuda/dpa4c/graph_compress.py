# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Bindings and immutable artifacts for compressed degree-wise DPA4C.

The CUDA operator evaluates the complete DPA4C graph descriptor: the tabulated
radial branch and its shared mode profiles, the ordered PairFiLM amplitude, one
destination reduction that produces both envelope masses and every degree-wise
moment, the invariant readout, and the fixed output calibration. Its analytical
backward runs one node-readout VJP and one edge recomputation scan.

The radial table stores the learned scalar-distance maps

.. math::

   r\mapsto g_c(r)=\operatorname{RadialMLP}(\operatorname{RBF}(r))_c,\qquad
   r\mapsto q_\rho(r)=\operatorname{ModeHead}(\operatorname{RadialMLP}
   _{\rm hidden}(\operatorname{RBF}(r)))_\rho,

so its width is ``channels + radial_modes``. The fixed C³ envelope remains
analytical and gates the complete FiLM amplitude exactly once; the non-scalar
moments carry a second explicit envelope factor. Ordered scale, shift, and
mode-mixing caches, the readout projections, and the sparse angular coupling
tables are materialized once when compression is enabled.

Degrees one and two dominate the readout and are contracted in closed form.
Degrees three and four carry a single channel each, so their couplings are
driven by a compact sparse Cartesian Gaunt table rather than by specialized
code.
"""

from __future__ import (
    annotations,
)

import copy
import math
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa4c_nn import (
    build_angular_basis,
    build_bispectrum_layout,
    derive_bispectrum_ranks,
    derive_degree_channels,
    packed_l2_to_stf,
)

__all__ = [
    "build_compression_artifacts",
    "build_radial_table",
    "coupling_records",
    "descriptor_profile",
    "dpa4c_graph_compress",
    "dpa4c_graph_compress_energy_force",
    "ef_op_available",
    "ensure_registered",
    "fitting_energy_and_gradient",
    "mega_eligible",
    "op_available",
]

SUPPORTED_CHANNELS = (8, 16, 32, 64, 128)
SUPPORTED_LMAX = (2, 3, 4)
SUPPORTED_RADIAL_MODES = (0, 2, 4, 8)

# Degrees one and two carry the wide channel blocks and are contracted by
# specialized closed forms in every backend; the remaining triples run through
# the shared sparse coupling path.
_CLOSED_FORM_TRIPLES = ((1, 1, 2), (2, 2, 2))

# Lebedev quadrature leaves rounding noise around eighteen orders of magnitude
# below the retained coupling entries, far below the fp32 working precision.
_COUPLING_TOLERANCE = 1.0e-9

# Unit-Frobenius Cartesian normalizations of the two closed-form couplings.
_INV_SQRT_FIVE = 1.0 / math.sqrt(5.0)
_BIS222_SCALE = math.sqrt(12.0 / 35.0)


def op_available() -> bool:
    """Return whether the compiled DPA4C compressed operator is loaded."""
    op = getattr(torch.ops.deepmd, "dpa4c_graph_compress", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def ef_op_available() -> bool:
    """Return whether the descriptor, fitting, and force operators are loaded."""
    return (
        op_available()
        and isinstance(
            getattr(torch.ops.deepmd, "graph_fitting", None),
            torch._ops.OpOverloadPacket,
        )
        and isinstance(
            getattr(torch.ops.deepmd, "edge_force_virial", None),
            torch._ops.OpOverloadPacket,
        )
    )


def mega_eligible(descriptor: Any) -> bool:
    """Return whether the descriptor has a compiled fp32 specialization.

    Parameters
    ----------
    descriptor
        Descriptor instance, which need not be a DPA4C.

    Returns
    -------
    bool
        Whether the compiled operator covers this configuration.
    """
    return (
        int(getattr(descriptor, "channels", -1)) in SUPPORTED_CHANNELS
        and int(getattr(descriptor, "lmax", -1)) in SUPPORTED_LMAX
        and int(getattr(descriptor, "radial_modes", -1)) in SUPPORTED_RADIAL_MODES
        and str(getattr(descriptor, "precision", "")).lower() in ("float32", "single")
    )


# === Compiled profile ===


@dataclass(frozen=True)
class DescriptorProfile:
    r"""Describe every width the compiled operator derives from its inputs.

    Attributes
    ----------
    channels
        Scalar degree-zero width :math:`C_0`.
    lmax
        Maximum angular degree.
    degree_channels
        Channel width of each degree from zero through ``lmax``.
    ranks
        Bispectrum probe rank of each degree from one through ``lmax``.
    moment_width
        Flat moment width :math:`S=\\sum_\\ell(2\\ell+1)C_\\ell`.
    output_width
        Invariant descriptor width consumed by the fitting network.
    gram_base
        Descriptor coordinate of the first degree-one Gram entry.
    bispectrum_base
        Descriptor coordinate of the first bispectrum entry.
    """

    channels: int
    lmax: int
    degree_channels: tuple[int, ...]
    ranks: tuple[int, ...]
    moment_width: int
    output_width: int
    gram_base: int
    bispectrum_base: int

    @property
    def state_width(self) -> int:
        """Return the saved-state width: the moments plus both normalizers."""
        return self.moment_width + 2


def descriptor_profile(channels: int, lmax: int) -> DescriptorProfile:
    """Derive every compiled width from the two structural parameters.

    Parameters
    ----------
    channels
        Scalar degree-zero width.
    lmax
        Maximum angular degree.

    Returns
    -------
    DescriptorProfile
        Widths and descriptor-block offsets shared by all backends.
    """
    degree_channels = tuple(derive_degree_channels(int(channels), int(lmax)))
    ranks = tuple(derive_bispectrum_ranks(degree_channels))
    moment_width = sum(
        (2 * degree + 1) * width for degree, width in enumerate(degree_channels)
    )
    gram_total = sum(width * (width + 1) // 2 for width in degree_channels[1:])
    layout = build_bispectrum_layout(int(lmax), ranks)
    bispectrum_dim = int(layout.probe_index.shape[0])
    gram_base = degree_channels[0]
    bispectrum_base = gram_base + gram_total
    # Geometric block, the two moment divisors, then the center type tail.
    output_width = (
        bispectrum_base + bispectrum_dim + ranks[0] * ranks[1] + 2 + degree_channels[0]
    )
    return DescriptorProfile(
        channels=int(channels),
        lmax=int(lmax),
        degree_channels=degree_channels,
        ranks=ranks,
        moment_width=moment_width,
        output_width=output_width,
        gram_base=gram_base,
        bispectrum_base=bispectrum_base,
    )


@dataclass(frozen=True)
class CouplingRecord:
    r"""Describe one sparse angular coupling consumed by the operator.

    Attributes
    ----------
    degrees
        Degree triple :math:`(\\ell_1,\\ell_2,\\ell_3)`.
    nonzero_begin
        First entry index of the coupling nonzeros.
    nonzero_count
        Number of coupling nonzeros.
    probe_begin
        First entry index of the packed probe coordinates.
    probe_count
        Number of emitted probe contractions.
    coordinate
        Descriptor coordinate of the first emitted contraction.
    """

    degrees: tuple[int, int, int]
    nonzero_begin: int
    nonzero_count: int
    probe_begin: int
    probe_count: int
    coordinate: int


def coupling_records(
    channels: int,
    lmax: int,
) -> tuple[list[CouplingRecord], np.ndarray, np.ndarray]:
    """Build the sparse coupling tables for degrees beyond the closed forms.

    Each record addresses one contiguous run of coupling nonzeros followed by
    one contiguous run of probe coordinates inside the two shared flat arrays.
    A coupling nonzero packs its harmonic components as
    ``m1 | m2 << 8 | m3 << 16`` and carries the Gaunt value; a probe coordinate
    packs its channel indices the same way and carries the isometric scale.

    Parameters
    ----------
    channels
        Scalar degree-zero width.
    lmax
        Maximum angular degree.

    Returns
    -------
    records
        One record per non-closed-form degree triple.
    entry
        Packed component and channel coordinates with shape ``(M,)``, int32.
    value
        Coupling values and probe scales with shape ``(M,)``, float32.
    """
    profile = descriptor_profile(channels, lmax)
    layout = build_bispectrum_layout(profile.lmax, profile.ranks)
    records: list[CouplingRecord] = []
    entries: list[int] = []
    values: list[float] = []
    for index, degrees in enumerate(layout.degree_triples):
        if degrees in _CLOSED_FORM_TRIPLES:
            continue
        degree_1, degree_2, degree_3 = degrees
        start, end = layout.coupling_offsets[index : index + 2]
        coupling = layout.coupling[start:end].reshape(
            2 * degree_1 + 1,
            2 * degree_2 + 1,
            2 * degree_3 + 1,
        )
        nonzero_begin = len(entries)
        for component in np.argwhere(np.abs(coupling) > _COUPLING_TOLERANCE):
            first, second, third = (int(value) for value in component)
            entries.append(first | (second << 8) | (third << 16))
            values.append(float(coupling[first, second, third]))
        nonzero_count = len(entries) - nonzero_begin

        rank_2 = profile.ranks[degree_2 - 1]
        rank_3 = profile.ranks[degree_3 - 1]
        start, end = layout.probe_offsets[index : index + 2]
        probe_begin = len(entries)
        for probe, scale in zip(
            layout.probe_index[start:end],
            layout.probe_scale[start:end],
            strict=True,
        ):
            third = int(probe) % rank_3
            second = (int(probe) // rank_3) % rank_2
            first = int(probe) // (rank_3 * rank_2)
            entries.append(first | (second << 8) | (third << 16))
            values.append(float(scale))
        records.append(
            CouplingRecord(
                degrees=degrees,
                nonzero_begin=nonzero_begin,
                nonzero_count=nonzero_count,
                probe_begin=probe_begin,
                probe_count=end - start,
                coordinate=profile.bispectrum_base + int(start),
            )
        )
    return (
        records,
        np.asarray(entries, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def _coupling_meta(records: list[CouplingRecord]) -> np.ndarray:
    """Flatten coupling records into the int32 metadata table."""
    return np.asarray(
        [
            [
                record.degrees[0],
                record.degrees[1],
                record.degrees[2],
                record.nonzero_begin,
                record.nonzero_count,
                record.probe_begin,
                record.probe_count,
                record.coordinate,
            ]
            for record in records
        ],
        dtype=np.int32,
    ).reshape(len(records), 8)


# === Immutable artifacts ===


def _quintic_coefficients(
    values: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    stride: float,
) -> torch.Tensor:
    """Build C²-matching quintic Hermite coefficients.

    Parameters
    ----------
    values
        Function values with shape ``(S + 1, W)``.
    first
        First derivatives with shape ``(S + 1, W)``.
    second
        Second derivatives with shape ``(S + 1, W)``.
    stride
        Uniform interval width in Å.
    The interval row is split into a leading quartet block holding
    ``[c0, c1, c2, c3]`` of every channel and a trailing pair block holding
    ``[c4, c5]``. Both blocks are naturally aligned for one 128-bit and one
    64-bit load, which evaluates the spline in two memory instructions instead
    of the three a plain channel-major layout would need, at identical traffic.

    Returns
    -------
    torch.Tensor
        Coefficients with shape ``(S, 6 * W)``: ``4 * W`` quartet entries
        followed by ``2 * W`` pair entries.
    """
    left_value, right_value = values[:-1], values[1:]
    left_first, right_first = first[:-1], first[1:]
    left_second, right_second = second[:-1], second[1:]
    delta = right_value - left_value
    h = float(stride)
    c0 = left_value
    c1 = left_first
    c2 = 0.5 * left_second
    c3 = (
        20.0 * delta
        - (8.0 * right_first + 12.0 * left_first) * h
        - (3.0 * left_second - right_second) * h * h
    ) / (2.0 * h**3)
    c4 = (
        -30.0 * delta
        + (14.0 * right_first + 16.0 * left_first) * h
        + (3.0 * left_second - 2.0 * right_second) * h * h
    ) / (2.0 * h**4)
    c5 = (
        12.0 * delta
        - 6.0 * (right_first + left_first) * h
        + (right_second - left_second) * h * h
    ) / (2.0 * h**5)
    intervals = values.shape[0] - 1
    return torch.cat(
        [
            torch.stack([c0, c1, c2, c3], dim=-1).reshape(intervals, -1),
            torch.stack([c4, c5], dim=-1).reshape(intervals, -1),
        ],
        dim=-1,
    )


def build_radial_table(
    descriptor: Any,
    stride: float = 0.002,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tabulate the composed DPA4C radial branch and its mode profiles.

    The table width is ``channels + radial_modes``: the leading block holds the
    shared radial map read by every degree, and the trailing block holds the
    shared mode profiles that each ordered type pair mixes.

    Parameters
    ----------
    descriptor
        pt_expt DPA4C descriptor whose DPA4 radial modules define the table.
    stride
        Uniform distance spacing in Å.

    Returns
    -------
    table
        Quintic coefficients with shape ``(S, 6 * (channels + radial_modes))``,
        fp32.
    info
        CPU metadata ``[stride, table_max, rcut, eps, degree_floor]``, fp64.

    Raises
    ------
    ValueError
        If ``stride`` is not positive.
    """
    if stride <= 0.0:
        raise ValueError(f"`stride` must be positive, got {stride}")
    sample_parameter = next(descriptor.radial_embedding.parameters())
    device = sample_parameter.device
    radial_basis = copy.deepcopy(descriptor.radial_basis).to(
        device=device,
        dtype=torch.float64,
    )
    radial_embedding = copy.deepcopy(descriptor.radial_embedding).to(
        device=device,
        dtype=torch.float64,
    )
    radial_mode_head = (
        None
        if descriptor.radial_mode_head is None
        else copy.deepcopy(descriptor.radial_mode_head).to(
            device=device,
            dtype=torch.float64,
        )
    )
    interval_count = math.ceil(float(descriptor.rcut) / float(stride))
    table_max = interval_count * float(stride)
    distance = torch.arange(
        interval_count + 1,
        dtype=torch.float64,
        device=device,
    )
    distance = (distance * float(stride)).requires_grad_(True)

    # === Step 1. Evaluate every tabulated distance map at the table knots ===
    hidden = radial_embedding.call_hidden(radial_basis(distance[:, None]))
    values = radial_embedding.call_output(hidden)
    if radial_mode_head is not None:
        values = torch.cat([values, radial_mode_head.call(hidden)], dim=-1)

    # === Step 2. Differentiate each output channel independently ===
    first_columns = []
    second_columns = []
    for channel in range(int(values.shape[1])):
        (first_channel,) = torch.autograd.grad(
            values[:, channel].sum(),
            distance,
            create_graph=True,
            retain_graph=True,
        )
        (second_channel,) = torch.autograd.grad(
            first_channel.sum(),
            distance,
            retain_graph=True,
        )
        first_columns.append(first_channel)
        second_columns.append(second_channel)
    first = torch.stack(first_columns, dim=-1)
    second = torch.stack(second_columns, dim=-1)

    # === Step 3. Convert knot data to the runtime spline layout ===
    table = _quintic_coefficients(values, first, second, float(stride))
    table = table.detach().to(torch.float32).contiguous()
    info = torch.tensor(
        [
            float(stride),
            table_max,
            float(descriptor.rcut),
            float(descriptor._EPS),
            float(descriptor._DEGREE_NORM_FLOOR),
        ],
        dtype=torch.float64,
        device="cpu",
    )
    return table, info


def build_compression_artifacts(
    descriptor: Any,
    stride: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Build all immutable tensors consumed by compressed inference.

    Parameters
    ----------
    descriptor
        Evaluated pt_expt DPA4C descriptor.
    stride
        Uniform radial spline spacing in Å.

    Returns
    -------
    dict[str, torch.Tensor]
        Radial spline coefficients, metadata, ordered FiLM, mixing and type
        caches, padded readout projections, sparse angular couplings, and the
        output calibration.

    Raises
    ------
    ValueError
        If the descriptor is not an fp32 model with a compiled specialization.
    """
    sample_parameter = next(descriptor.parameters())
    if sample_parameter.dtype != torch.float32:
        raise ValueError(
            "DPA4C compressed CUDA requires descriptor precision `float32`, "
            f"got {sample_parameter.dtype}"
        )
    if not mega_eligible(descriptor):
        raise ValueError(
            "DPA4C compressed CUDA supports channels "
            f"{SUPPORTED_CHANNELS} with lmax {SUPPORTED_LMAX} and "
            f"radial_modes {SUPPORTED_RADIAL_MODES}, got "
            f"channels={descriptor.channels}, lmax={descriptor.lmax}, "
            f"radial_modes={descriptor.radial_modes}"
        )
    device = sample_parameter.device
    profile = descriptor_profile(descriptor.channels, descriptor.lmax)
    table, info = build_radial_table(descriptor, stride)

    with torch.no_grad():
        type_embedding = descriptor.type_embedding.call().to(
            device=device,
            dtype=torch.float32,
        )
        pair_scale, pair_shift, pair_mixing = descriptor.pair_film.call(type_embedding)
        pair_film = torch.stack((pair_scale, pair_shift), dim=-1)
        # The mode axis is innermost so that the coefficients a lane needs for
        # one channel arrive in one or two vector loads.
        mixing = (
            torch.zeros(0, dtype=torch.float32, device=device)
            if pair_mixing is None
            else pair_mixing.to(torch.float32)
        )
        readout_matrices = _build_readout_matrices(descriptor, profile, device)
        output_mean = descriptor.mean.to(device=device, dtype=torch.float32)
        output_inv_std = torch.reciprocal(
            descriptor.stddev.to(device=device, dtype=torch.float32)
        )

    records, coupling_entry, coupling_value = coupling_records(
        profile.channels,
        profile.lmax,
    )
    to_device = {"device": device}
    return {
        "data": table,
        "info": info,
        "pair_film": pair_film.detach().contiguous(),
        "pair_mixing": mixing.detach().contiguous(),
        "type_embedding": type_embedding.detach().contiguous(),
        "readout_matrices": readout_matrices,
        "coupling_meta": torch.as_tensor(
            _coupling_meta(records),
            dtype=torch.int32,
            **to_device,
        ),
        "coupling_entry": torch.as_tensor(
            coupling_entry,
            dtype=torch.int32,
            **to_device,
        ),
        "coupling_value": torch.as_tensor(
            coupling_value,
            dtype=torch.float32,
            **to_device,
        ),
        "output_mean": output_mean.detach().contiguous(),
        "output_inv_std": output_inv_std.detach().contiguous(),
    }


def _build_readout_matrices(
    descriptor: Any,
    profile: DescriptorProfile,
    device: torch.device,
) -> torch.Tensor:
    """Pack the degree-one and degree-two readout projections.

    The kernel reads the residual channel alignment and the probe projection of
    each wide degree together with its transpose, so all eight matrices are
    stored in one square block padded to the degree-one width. Degrees three
    and above carry a single channel, for which both maps are the identity.

    Parameters
    ----------
    descriptor
        Evaluated pt_expt DPA4C descriptor.
    profile
        Compiled profile of this descriptor.
    device
        Device that receives the packed table.

    Returns
    -------
    torch.Tensor
        Packed projections with shape ``(8, C_1, C_1)``, fp32.
    """
    width = profile.degree_channels[1]
    packed = torch.zeros(8, width, width, dtype=torch.float32, device=device)

    def residual(layer: Any, size: int) -> torch.Tensor:
        return layer.w.to(torch.float32) + torch.eye(
            size,
            dtype=torch.float32,
            device=device,
        )

    def probe(layer: Any, size: int) -> torch.Tensor:
        if layer is None:
            return torch.eye(size, dtype=torch.float32, device=device)
        return layer.w.to(torch.float32)

    alignment_one = residual(
        descriptor.readout.channel_alignment[0],
        profile.degree_channels[1],
    )
    alignment_two = residual(
        descriptor.readout.channel_alignment[1],
        profile.degree_channels[2],
    )
    probe_one = probe(
        descriptor.readout.probe_projections[0],
        profile.degree_channels[1],
    )
    probe_two = probe(
        descriptor.readout.probe_projections[1],
        profile.degree_channels[2],
    )
    for index, matrix in enumerate(
        (
            alignment_one,
            alignment_one.T,
            alignment_two,
            alignment_two.T,
            probe_one,
            probe_one.T,
            probe_two,
            probe_two.T,
        )
    ):
        packed[index, : matrix.shape[0], : matrix.shape[1]] = matrix
    return packed.detach().contiguous()


# === Reference implementation ===


def _table_lookup(
    table: torch.Tensor,
    radius: torch.Tensor,
    stride: float,
    table_max: float,
    width: int,
) -> torch.Tensor:
    """Evaluate a uniform quintic table with a clamped high-distance tail."""
    coordinate = radius.clamp(min=0.0, max=table_max)
    index = torch.floor(coordinate / stride).to(torch.int64)
    index = index.clamp(max=table.shape[0] - 1)
    dx = (coordinate - index.to(coordinate.dtype) * stride)[:, None]
    row = table[index]
    quartet = row[:, : 4 * width].reshape(-1, width, 4)
    pair = row[:, 4 * width :].reshape(-1, width, 2)
    return (
        quartet[..., 0]
        + (
            quartet[..., 1]
            + (
                quartet[..., 2]
                + (quartet[..., 3] + (pair[..., 0] + pair[..., 1] * dx) * dx) * dx
            )
            * dx
        )
        * dx
    )


def _c3_envelope(radius: torch.Tensor, rcut: float) -> torch.Tensor:
    """Evaluate the fixed exponent-five DPA4 C³ envelope."""
    u = ((float(rcut) - radius) / float(rcut)).clamp(0.0, 1.0)
    x = 1.0 - u
    series = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + 35.0 * x)))
    return u**4 * series


def _half_gram(value: torch.Tensor) -> torch.Tensor:
    """Return the Frobenius-isometric upper-triangular channel Gram."""
    width = value.shape[-1]
    row, column = torch.triu_indices(width, width, device=value.device)
    scale = torch.where(
        row == column,
        torch.ones((), dtype=value.dtype, device=value.device),
        torch.full((), math.sqrt(2.0), dtype=value.dtype, device=value.device),
    )
    return (value.transpose(1, 2) @ value)[:, row, column] * scale


def _contract_coupling(
    coupling: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    third: torch.Tensor,
) -> torch.Tensor:
    """Contract one angular coupling into ordered probe combinations.

    Parameters
    ----------
    coupling
        Gaunt tensor with shape ``(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)``.
    first
        Degree block with shape ``(N, 2 * l1 + 1, K_1)``.
    second
        Degree block with shape ``(N, 2 * l2 + 1, K_2)``.
    third
        Degree block with shape ``(N, 2 * l3 + 1, K_3)``.

    Returns
    -------
    torch.Tensor
        Ordered contractions with shape ``(N, K_1 * K_2 * K_3)``.
    """
    nodes = first.shape[0]
    rank_1, rank_2, rank_3 = first.shape[-1], second.shape[-1], third.shape[-1]
    dim_1, dim_2, dim_3 = coupling.shape
    value = first.transpose(1, 2) @ coupling.reshape(dim_1, dim_2 * dim_3)
    value = value.reshape(nodes, rank_1, dim_2, dim_3).transpose(2, 3)
    value = (value.reshape(nodes, rank_1 * dim_3, dim_2) @ second).reshape(
        nodes,
        rank_1,
        dim_3,
        rank_2,
    )
    value = value.transpose(2, 3).reshape(nodes, rank_1 * rank_2, dim_3)
    return (value @ third).reshape(nodes, rank_1 * rank_2 * rank_3)


def _cpu_descriptor(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    table: torch.Tensor,
    pair_film: torch.Tensor,
    pair_mixing: torch.Tensor,
    type_embedding: torch.Tensor,
    readout_matrices: torch.Tensor,
    coupling_meta: torch.Tensor,
    coupling_entry: torch.Tensor,
    coupling_value: torch.Tensor,
    output_mean: torch.Tensor,
    output_inv_std: torch.Tensor,
    canonical: bool,
    lmax: int,
    table_stride: float,
    table_max: float,
    rcut: float,
    eps: float,
    degree_floor: float,
) -> torch.Tensor:
    """Reference implementation of the compressed DPA4C descriptor."""
    del destination_order, destination_row_ptr, canonical, coupling_meta
    compute = edge_vec.to(torch.float32)
    source, destination = edge_index[0].to(torch.long), edge_index[1].to(torch.long)
    node_count = atype.shape[0]
    channels = type_embedding.shape[1]
    type_count = type_embedding.shape[0]
    profile = descriptor_profile(int(channels), int(lmax))
    radial_modes = 0 if pair_mixing.numel() == 0 else int(pair_mixing.shape[2])

    # === Step 1. Build the masked edge geometry ===
    radius = torch.sqrt((compute * compute).sum(dim=-1) + float(eps) ** 2)
    direction = compute / radius[:, None]
    center_type = atype[destination]
    neighbor_type = atype[source]
    mask = edge_mask & (neighbor_type < type_count - 1) & (center_type < type_count - 1)
    maskf = mask.to(compute.dtype)
    envelope = _c3_envelope(radius, float(rcut)) * maskf

    # === Step 2. Evaluate the ordered FiLM amplitude ===
    tabulated = _table_lookup(
        table.to(compute.device),
        radius,
        float(table_stride),
        float(table_max),
        channels + radial_modes,
    )
    pair_index = center_type * type_count + neighbor_type
    film = pair_film[pair_index]
    amplitude = tabulated[:, :channels] * film[..., 0] + film[..., 1]
    if radial_modes > 0:
        amplitude = amplitude + torch.einsum(
            "ecr,er->ec",
            pair_mixing[pair_index],
            tabulated[:, channels:],
        )
    amplitude = amplitude * envelope[:, None]
    basis = build_angular_basis(direction, profile.lmax) * maskf[:, None]

    # === Step 3. Reduce both envelope masses and every degree-wise moment ===
    envelope_squared = envelope * envelope
    scalar_mass = torch.zeros(
        node_count,
        dtype=compute.dtype,
        device=compute.device,
    ).index_add_(0, destination, envelope_squared)
    angular_mass = torch.zeros(
        node_count,
        dtype=compute.dtype,
        device=compute.device,
    ).index_add_(0, destination, envelope_squared * envelope_squared)
    scalar_divisor = torch.sqrt(scalar_mass + float(degree_floor))[:, None]
    angular_divisor = torch.sqrt(angular_mass + float(degree_floor))[:, None]
    scalar_norm = torch.reciprocal(scalar_divisor)
    angular_norm = torch.reciprocal(angular_divisor)[:, :, None]

    blocks = [
        torch.zeros(
            node_count,
            1,
            channels,
            dtype=compute.dtype,
            device=compute.device,
        ).index_add_(0, destination, amplitude[:, None, :])
        * scalar_norm[:, None]
    ]
    for degree, width in enumerate(profile.degree_channels[1:], start=1):
        payload = (
            basis[:, degree**2 : (degree + 1) ** 2, None]
            * amplitude[:, None, :width]
            * envelope[:, None, None]
        )
        blocks.append(
            torch.zeros(
                node_count,
                2 * degree + 1,
                width,
                dtype=compute.dtype,
                device=compute.device,
            ).index_add_(0, destination, payload)
            * angular_norm
        )

    # === Step 4. Align, project, and contract the invariant readout ===
    width_one, width_two = profile.degree_channels[1], profile.degree_channels[2]
    rank_one, rank_two = profile.ranks[0], profile.ranks[1]
    aligned = list(blocks)
    aligned[1] = blocks[1] @ readout_matrices[0, :width_one, :width_one]
    aligned[2] = blocks[2] @ readout_matrices[2, :width_two, :width_two]
    probes = list(aligned[1:])
    probes[0] = aligned[1] @ readout_matrices[4, :width_one, :rank_one]
    probes[1] = aligned[2] @ readout_matrices[6, :width_two, :rank_two]

    vectors = probes[0].transpose(1, 2)
    tensors = packed_l2_to_stf(probes[1].transpose(1, 2))
    tensor_vector = (tensors[:, :, None] @ vectors[:, None, :, :, None]).squeeze(-1)
    parts: dict[int, torch.Tensor] = {}
    for record in coupling_records(int(channels), profile.lmax)[0]:
        degree_1, degree_2, degree_3 = record.degrees
        start = record.nonzero_begin
        components = coupling_entry[start : start + record.nonzero_count]
        coupling = torch.zeros(
            2 * degree_1 + 1,
            2 * degree_2 + 1,
            2 * degree_3 + 1,
            dtype=compute.dtype,
            device=compute.device,
        )
        coupling[
            components & 0xFF,
            (components >> 8) & 0xFF,
            (components >> 16) & 0xFF,
        ] = coupling_value[start : start + record.nonzero_count].to(compute.dtype)
        full = _contract_coupling(
            coupling,
            probes[degree_1 - 1],
            probes[degree_2 - 1],
            probes[degree_3 - 1],
        )
        start = record.probe_begin
        selection = coupling_entry[start : start + record.probe_count]
        rank_2 = profile.ranks[degree_2 - 1]
        rank_3 = profile.ranks[degree_3 - 1]
        flat = (
            ((selection & 0xFF) * rank_2 + ((selection >> 8) & 0xFF)) * rank_3
            + ((selection >> 16) & 0xFF)
        ).to(torch.long)
        parts[record.coordinate] = full[:, flat] * coupling_value[
            start : start + record.probe_count
        ].to(compute.dtype)

    parts[profile.bispectrum_base] = _closed_form_112(
        vectors,
        tensor_vector,
        rank_one,
        rank_two,
    )
    parts[_closed_form_222_coordinate(profile)] = _closed_form_222(tensors)

    # === Step 5. Assemble and calibrate the invariant output ===
    quartic = (tensor_vector * tensor_vector).sum(dim=-1).flatten(start_dim=1)
    descriptor = torch.cat(
        [
            blocks[0][:, 0, :],
            *[_half_gram(block) for block in aligned[1:]],
            *[parts[key] for key in sorted(parts)],
            quartic,
            scalar_divisor,
            angular_divisor,
            type_embedding[atype],
        ],
        dim=-1,
    )
    return (descriptor - output_mean[None, :]) * output_inv_std[None, :]


def _closed_form_112(
    vectors: torch.Tensor,
    tensor_vector: torch.Tensor,
    rank_one: int,
    rank_two: int,
) -> torch.Tensor:
    """Contract the ``112`` triple from the shared ``Q_b v_a`` intermediate."""
    values = []
    for first in range(rank_one):
        for second in range(first, rank_one):
            scale = 1.0 if first == second else math.sqrt(2.0)
            for index in range(rank_two):
                values.append(
                    -scale
                    * _INV_SQRT_FIVE
                    * (vectors[:, first] * tensor_vector[:, index, second]).sum(dim=-1)
                )
    return torch.stack(values, dim=-1)


def _closed_form_222(tensors: torch.Tensor) -> torch.Tensor:
    """Contract the symmetric ``222`` triple over the two degree-two probes.

    One operator ordering suffices because the factors are symmetric, so
    ``tr(ABC) = tr((ABC)^T) = tr(CBA) = tr(ACB)``.
    """
    values = []
    for first, second, third, scale in (
        (0, 0, 0, 1.0),
        (0, 0, 1, math.sqrt(3.0)),
        (0, 1, 1, math.sqrt(3.0)),
        (1, 1, 1, 1.0),
    ):
        product = tensors[:, first] @ tensors[:, second] @ tensors[:, third]
        values.append(
            -_BIS222_SCALE
            * scale
            * torch.linalg.diagonal(product, dim1=-2, dim2=-1).sum(-1)
        )
    return torch.stack(values, dim=-1)


def _closed_form_222_coordinate(profile: DescriptorProfile) -> int:
    """Return the descriptor coordinate of the ``222`` bispectrum block."""
    layout = build_bispectrum_layout(profile.lmax, profile.ranks)
    index = layout.degree_triples.index((2, 2, 2))
    return profile.bispectrum_base + int(layout.probe_offsets[index])


# === Custom-operator registration ===


def _cpu_forward(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU custom-op implementation returning descriptor and opaque state."""
    descriptor = _cpu_descriptor(*args)
    profile = descriptor_profile(int(args[9].shape[1]), int(args[17]))
    state = torch.zeros(
        descriptor.shape[0],
        profile.state_width,
        dtype=descriptor.dtype,
        device=descriptor.device,
    )
    return descriptor, state


def _forward_fake(
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    table: torch.Tensor,
    pair_film: torch.Tensor,
    pair_mixing: torch.Tensor,
    type_embedding: torch.Tensor,
    readout_matrices: torch.Tensor,
    coupling_meta: torch.Tensor,
    coupling_entry: torch.Tensor,
    coupling_value: torch.Tensor,
    output_mean: torch.Tensor,
    output_inv_std: torch.Tensor,
    canonical: bool,
    lmax: int,
    table_stride: float,
    table_max: float,
    rcut: float,
    eps: float,
    degree_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        table,
        pair_film,
        pair_mixing,
        readout_matrices,
        coupling_meta,
        coupling_entry,
        coupling_value,
        output_mean,
        output_inv_std,
        canonical,
        table_stride,
        table_max,
        rcut,
        eps,
        degree_floor,
    )
    profile = descriptor_profile(int(type_embedding.shape[1]), int(lmax))
    descriptor = torch.empty(
        atype.shape[0],
        profile.output_width,
        dtype=torch.float32,
        device=edge_vec.device,
    )
    state = torch.empty(
        atype.shape[0],
        profile.state_width,
        dtype=torch.float32,
        device=edge_vec.device,
    )
    return descriptor, state


def _backward_fake(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    del descriptor_gradient, state, args
    return torch.empty_like(edge_vec)


def _cpu_backward(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> torch.Tensor:
    del state
    if edge_vec.shape[0] == 0:
        return torch.zeros_like(edge_vec)
    value = edge_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        descriptor = _cpu_descriptor(value, *args)
        (gradient,) = torch.autograd.grad(
            (descriptor * descriptor_gradient.to(descriptor.dtype)).sum(),
            value,
        )
    return gradient.to(edge_vec.dtype)


def _setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    ctx.save_for_backward(output[1], *inputs[:16])
    ctx.scalars = inputs[16:]
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)


def _backward(
    ctx: Any,
    descriptor_gradient: torch.Tensor,
    state_gradient: torch.Tensor | None,
) -> tuple:
    del state_gradient
    tensors = ctx.saved_tensors
    edge_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        descriptor_gradient,
        tensors[0],
        *tensors[1:],
        *ctx.scalars,
    )
    return (edge_gradient,) + (None,) * 22


_cpu_library: torch.library.Library | None = None


def ensure_registered() -> None:
    """Register fake, CPU, and autograd implementations once."""
    global _cpu_library
    if _cpu_library is not None or not op_available():
        return
    torch.library.register_fake("deepmd::dpa4c_graph_compress")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4c_graph_compress_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4c_graph_compress",
        _backward,
        setup_context=_setup_context,
    )
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("dpa4c_graph_compress", _cpu_forward, "CPU")
    _cpu_library.impl(
        "dpa4c_graph_compress_backward",
        _cpu_backward,
        "CPU",
    )


def compressed_operator_arguments(descriptor: Any) -> tuple:
    """Return the immutable operator arguments of a compressed descriptor.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.

    Returns
    -------
    tuple
        Radial table, ordered caches, readout projections, coupling tables,
        output calibration, and the trailing scalar configuration.
    """
    return (
        descriptor.compress_data,
        descriptor.compress_pair_film,
        descriptor.compress_pair_mixing,
        descriptor.compress_type_embedding,
        descriptor.compress_readout_matrices,
        descriptor.compress_coupling_meta,
        descriptor.compress_coupling_entry,
        descriptor.compress_coupling_value,
        descriptor.compress_output_mean,
        descriptor.compress_output_inv_std,
    )


def dpa4c_graph_compress(
    descriptor: Any,
    graph: Any,
    atype: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the compressed DPA4C graph descriptor.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.
    graph
        NeighborGraph with destination CSR topology.
    atype
        Flat node types with shape ``(N,)``.

    Returns
    -------
    torch.Tensor
        Degree-wise invariant descriptor with shape
        ``(N, descriptor.get_dim_out())``, fp32.

    Raises
    ------
    ValueError
        If the graph lacks destination CSR topology.
    """
    ensure_registered()
    if graph.destination_order is None or graph.destination_row_ptr is None:
        raise ValueError("DPA4C compressed CUDA requires destination CSR topology")
    descriptor_output, _state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec.contiguous(),
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        *compressed_operator_arguments(descriptor),
        bool(graph.destination_sorted),
        int(descriptor.lmax),
        *descriptor._compression_scalars,
    )
    return descriptor_output.to(graph.edge_vec.dtype)


def dpa4c_graph_compress_energy_force(
    descriptor: Any,
    fitting: Any,
    graph: Any,
    atype: torch.Tensor,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
    node_capacity: int,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate compressed DPA4C energy, force, and virial without a tape.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.
    fitting
        Eligible pt_expt energy fitting network.
    graph
        NeighborGraph with destination and source CSR topology.
    atype
        Flat node atom types with shape ``(N,)``.
    ownership
        Boolean mask selecting energy-contributing nodes with shape ``(N,)``.
    atom_bias
        Combined atomic energy bias with shape ``(ntypes,)``.
    node_capacity
        Force-scatter node capacity.
    do_atomic_virial
        Whether to return per-node virials.

    Returns
    -------
    energy
        Per-frame energy with shape ``(F, 1)``, fp64.
    atom_energy
        Per-node energy with shape ``(N, 1)``, fp64.
    force
        Per-node force with shape ``(N, 3)``, fp32.
    virial
        Per-frame virial with shape ``(F, 3, 3)``, fp32.
    atom_virial
        Per-node virial with shape ``(N, 3, 3)`` or an empty tensor.

    Raises
    ------
    ValueError
        If the graph lacks destination or source CSR topology.
    """
    from deepmd.kernels.cuda.edge_force_virial import (
        edge_force_virial,
    )
    from deepmd.kernels.cuda.edge_force_virial import (
        ensure_registered as ensure_force_registered,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        ensure_registered as ensure_fitting_registered,
    )

    ensure_registered()
    ensure_fitting_registered()
    ensure_force_registered()
    if (
        graph.destination_order is None
        or graph.destination_row_ptr is None
        or graph.source_order is None
        or graph.source_row_ptr is None
    ):
        raise ValueError(
            "DPA4C compressed energy-force inference requires destination "
            "and source CSR topology"
        )

    operator_args = (
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        *compressed_operator_arguments(descriptor),
        bool(graph.destination_sorted),
        int(descriptor.lmax),
        *descriptor._compression_scalars,
    )
    edge_vec = graph.edge_vec.to(torch.float32).contiguous()
    node_descriptor, state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *operator_args,
    )

    energy, atom_energy, descriptor_gradient = fitting_energy_and_gradient(
        fitting,
        node_descriptor,
        atype,
        ownership,
        atom_bias,
        graph.n_node,
    )
    del node_descriptor
    edge_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        descriptor_gradient,
        state,
        edge_vec,
        *operator_args,
    )
    force, atom_virial, virial = edge_force_virial(
        edge_gradient,
        edge_vec,
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
        graph.n_node,
        node_capacity,
        do_atomic_virial,
    )
    return energy, atom_energy, force, virial, atom_virial


def fitting_energy_and_gradient(
    fitting: Any,
    node_descriptor: torch.Tensor,
    atype: torch.Tensor,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
    n_node_per_frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the fused fitting network and its descriptor cotangent.

    Parameters
    ----------
    fitting
        Eligible pt_expt energy fitting network.
    node_descriptor
        Invariant descriptor with shape ``(N, D)``, fp32.
    atype
        Flat node atom types with shape ``(N,)``.
    ownership
        Boolean mask selecting energy-contributing nodes with shape ``(N,)``.
    atom_bias
        Combined atomic energy bias with shape ``(ntypes,)``.
    n_node_per_frame
        Node count of each frame with shape ``(F,)``.

    Returns
    -------
    energy
        Per-frame energy with shape ``(F, 1)``, fp64.
    atom_energy
        Per-node energy with shape ``(N, 1)``, fp64.
    descriptor_gradient
        Cotangent of the invariant descriptor with shape ``(N, D)``.
    """
    from deepmd.kernels.cuda.edge_force_virial import (
        frame_scalar_sum,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        energy_and_input_gradient,
    )

    atom_energy_raw, descriptor_gradient = energy_and_input_gradient(
        fitting,
        node_descriptor,
        atype,
        ownership,
        atom_bias,
    )
    atom_energy = atom_energy_raw * ownership[:, None].to(atom_energy_raw.dtype)
    energy = frame_scalar_sum(atom_energy, n_node_per_frame)
    return energy, atom_energy, descriptor_gradient
