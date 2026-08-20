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
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa4c_nn import (
    build_angular_basis,
    build_bispectrum_layout,
    derive_bispectrum_ranks,
    derive_degree_channels,
    derive_spin_channels,
    packed_l2_to_stf,
)
from deepmd.pt_expt.kernels.utils import (
    backend_device_type,
    operator_available,
)
from deepmd.utils.charge_state import (
    validate_charge_state,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

__all__ = [
    "CHARGE_STATE_ARTIFACTS",
    "ChargeStateFold",
    "build_charge_state_artifacts",
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
    "reduce_edge_spin_gradient",
]

SUPPORTED_CHANNELS = (8, 16, 32, 64, 128)
SUPPORTED_LMAX = (2, 3, 4)
SUPPORTED_RADIAL_MODES = (0, 2, 4, 8)

#: Compression artifacts that carry the frame charge state. They are the
#: complete image of a charge state in a compressed snapshot: overwriting
#: exactly these four re-specializes the snapshot to a different state, and
#: every other artifact depends on the trained weights alone.
CHARGE_STATE_ARTIFACTS = (
    "pair_film",
    "pair_mixing",
    "spin_pair",
    "type_embedding",
)

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


def op_available(spin: bool = False) -> bool:
    """Return whether the backend device carries the DPA4C compressed operator.

    Native spin is a compiled variant of the same operator rather than a
    separate one, so it needs no separate availability probe -- except on the
    CPU, whose kernels carry no magnetic branch: those families need a
    source-major counterpart of the destination scan, and no CPU deployment
    asks for them yet. A spin-conditioned descriptor therefore keeps the
    reference path there.

    Parameters
    ----------
    spin
        Whether the caller needs the native-spin variant.

    Returns
    -------
    bool
        Whether the operator is registered for the backend device.
    """
    if spin and backend_device_type() != "cuda":
        return False
    return operator_available("dpa4c_graph_compress") and operator_available(
        "dpa4c_graph_compress_backward"
    )


def ef_op_available(spin: bool = False) -> bool:
    """Return whether the descriptor, fitting, and force operators are present.

    Parameters
    ----------
    spin
        Whether the caller needs the native-spin variant.

    Returns
    -------
    bool
        Whether every operator of the fused energy-force route is registered.
    """
    return (
        op_available(spin)
        and operator_available("graph_fitting")
        and operator_available("edge_force_virial")
    )


def mega_eligible(descriptor: Any) -> bool:
    """Return whether the descriptor has a compiled fp32 specialization.

    Native spin is a compiled variant of the same kernels rather than a
    separate operator, and every spin width follows from the channel count, so
    a spin-conditioned descriptor is eligible on exactly the conditions a
    spin-free one is. Each condition below is a width the compiled operator
    specializes on.

    Eligibility is a property of the descriptor alone: it decides whether the
    compression artifacts are worth building, which a snapshot does once for
    every device that may later consume it. Whether a device has the kernel to
    consume them is ``op_available``.

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
    spin_width
        Width of the trailing native-spin invariant block.
    """

    channels: int
    lmax: int
    degree_channels: tuple[int, ...]
    ranks: tuple[int, ...]
    moment_width: int
    output_width: int
    gram_base: int
    bispectrum_base: int
    spin_channels: int
    spin_width: int

    @property
    def state_width(self) -> int:
        """Return the saved-state width: the moments plus both normalizers."""
        return self.moment_width + 2

    @property
    def has_spin(self) -> bool:
        """Return whether the compiled profile carries the spin families."""
        return self.spin_channels > 0

    @property
    def spin_slice(self) -> slice:
        """Return the descriptor columns holding the spin invariants.

        The layout closes with the spin invariants, the two moment divisors
        and the center type tail, so the block is addressed from the end and
        is empty for a spin-free profile.
        """
        stop = self.output_width - 2 - self.degree_channels[0]
        return slice(stop - self.spin_width, stop)


def descriptor_profile(
    channels: int,
    lmax: int,
    has_spin: bool = False,
) -> DescriptorProfile:
    """Derive every compiled width from the structural parameters.

    The widths mirror ``Profile<Channels, Lmax, HasSpin>`` on the device side,
    so a mismatch is caught by the operator's own shape validation rather than
    producing a silently misread buffer.

    Parameters
    ----------
    channels
        Scalar degree-zero width.
    lmax
        Maximum angular degree.
    has_spin
        Whether the native spin families are present. Their width is derived
        from the degree profile, so presence is the whole choice.

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
    spin_channels = derive_spin_channels(degree_channels) if has_spin else 0
    if has_spin:
        # Reduced families, then the node-local on-site vector and quadrupole.
        moment_width += 8 * spin_channels + 5 + 8
        # The joint degree-one spin block holds the on-site moment beside the
        # isotropic and the bond-projected neighbor channels, so its Gram is
        # the upper triangle of a ``1 + 2 C_s`` block. The quadrupole Gram
        # drops only its on-site self-term.
        vector_width = 1 + 2 * spin_channels
        spin_dim = (
            vector_width * (vector_width + 1) // 2
            + 2
            + 2 * degree_channels[2]
            + 2 * spin_channels
        )
    else:
        spin_dim = 0
    # Geometric block, the spin invariants, the two moment divisors, then the
    # center type tail.
    output_width = (
        bispectrum_base
        + bispectrum_dim
        + ranks[0] * ranks[1]
        + spin_dim
        + 2
        + degree_channels[0]
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
        spin_channels=spin_channels,
        spin_width=spin_dim,
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
        If the descriptor excludes type pairs, or is not an fp32 model with a
        compiled specialization.
    """
    if getattr(descriptor, "exclude_types", None):
        raise ValueError(
            "DPA4C compressed CUDA has no type-exclusion branch, so a "
            "descriptor with a non-empty `exclude_types` cannot reach the "
            "fused path. Drop the exclusions, or deploy the uncompressed "
            "graph archive."
        )
    sample_parameter = next(descriptor.parameters())
    if sample_parameter.dtype != torch.float32:
        raise ValueError(
            "DPA4C compressed CUDA requires descriptor precision `float32`, "
            f"got {sample_parameter.dtype}"
        )
    if not mega_eligible(descriptor):
        raise ValueError(
            "DPA4C compressed CUDA supports "
            f"channels {SUPPORTED_CHANNELS}, lmax {SUPPORTED_LMAX} and "
            f"radial_modes {SUPPORTED_RADIAL_MODES}, got "
            f"channels={descriptor.channels}, lmax={descriptor.lmax}, "
            f"radial_modes={descriptor.radial_modes}"
        )
    device = sample_parameter.device
    profile = descriptor_profile(
        descriptor.channels,
        descriptor.lmax,
        descriptor.spin is not None,
    )
    table, info = build_radial_table(descriptor, stride)

    with torch.no_grad():
        readout_matrices = _build_readout_matrices(descriptor, profile, device)
        output_mean = descriptor.mean.to(device=device, dtype=torch.float32)
        output_inv_std = torch.reciprocal(
            descriptor.stddev.to(device=device, dtype=torch.float32)
        )
        if profile.has_spin:
            # The operator assembles the spin invariants without reading the
            # branch gate, which the portable path applies to the calibrated
            # block. Scaling the inverse deviation of those columns carries
            # the gate exactly, in the forward and in the backward alike:
            # the kernel pulls every output cotangent back through that same
            # array. A closed gate is a zero slope, so the gate never
            # restricts what may be compressed.
            output_inv_std[profile.spin_slice] *= float(
                torch.as_tensor(descriptor.spin.spin_gate).reshape(())
            )

    records, coupling_entry, coupling_value = coupling_records(
        profile.channels,
        profile.lmax,
    )
    to_device = {"device": device}
    return {
        "data": table,
        "info": info,
        "spin_type": _build_spin_type_cache(descriptor, device),
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
        **build_charge_state_artifacts(descriptor, descriptor.default_chg_spin),
    }


def charge_state_artifacts(
    descriptor: Any,
    charge_spin: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Build the compression artifacts that carry one frame charge state.

    The condition reaches the compiled kernel only through the ordered pair
    encoder and the centre type table, neither of which depends on distance,
    so these four artifacts are the complete image of a charge state in the
    snapshot. Rebuilding them from a different state re-specializes the
    snapshot without touching the radial table, the readout projections, the
    angular couplings, the per-type spin scalars or the output calibration.

    This is the traceable form: it takes the condition as a tensor and reads
    no host value, so the same code both builds the snapshot and, once
    exported, rebuilds it on the deployment device.
    :func:`build_charge_state_artifacts` is the host-side entry that validates
    the condition first.

    Parameters
    ----------
    descriptor
        Evaluated pt_expt DPA4C descriptor.
    charge_spin
        Frame condition with shape ``(2,)``. Read only when the descriptor is
        charge conditioned, and mandatory in that case.

    Returns
    -------
    dict[str, torch.Tensor]
        The artifacts named by :data:`CHARGE_STATE_ARTIFACTS`.

    Raises
    ------
    ValueError
        If a charge-conditioned descriptor is given no condition.
    """
    device = next(descriptor.parameters()).device
    type_embedding = descriptor.type_embedding.call().to(
        device=device,
        dtype=torch.float32,
    )
    pair_hidden_bias, type_shift = None, None
    if descriptor.charge_spin_embedding is not None:
        if charge_spin is None:
            raise ValueError(
                "A charge-conditioned DPA4C snapshot must be built against a "
                "frame condition. Set `default_chg_spin` to supply the state "
                "the snapshot starts from."
            )
        type_shift, pair_hidden_bias = descriptor.charge_spin_embedding.call(
            charge_spin.to(dtype=torch.float32).reshape(1, 2)
        )

    # The ordered pair encoder reads the unconditioned type table and receives
    # the condition as a pre-activation bias, while the centre type tail
    # carries it as an additive shift on its real rows. Both mirror the
    # portable path exactly.
    (
        pair_scale,
        pair_shift,
        pair_mixing,
        spin_scale,
        spin_shift,
    ) = descriptor.pair_film.call(
        type_embedding,
        hidden_bias=None if pair_hidden_bias is None else pair_hidden_bias[0],
    )
    if type_shift is not None:
        # The padding row keeps its zero embedding: the portable path masks
        # the shift by atom type, so shifting it here would break the parity
        # between the two routes on any system carrying padding or ghost
        # nodes.
        rows = torch.arange(type_embedding.shape[0], dtype=torch.int64, device=device)
        real = rows < descriptor.ntypes
        type_embedding = type_embedding + type_shift[0] * real[:, None]

    empty = torch.zeros(0, dtype=torch.float32, device=device)
    return {
        "pair_film": torch.stack((pair_scale, pair_shift), dim=-1).contiguous(),
        # The mode axis is innermost so that the coefficients a lane needs for
        # one channel arrive in one or two vector loads.
        "pair_mixing": (
            empty if pair_mixing is None else pair_mixing.to(torch.float32).contiguous()
        ),
        "spin_pair": (
            empty
            if descriptor.spin is None
            else torch.stack((spin_scale, spin_shift), dim=-1)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        ),
        "type_embedding": type_embedding.contiguous(),
    }


def build_charge_state_artifacts(
    descriptor: Any,
    charge_spin: Sequence[float] | None,
) -> dict[str, torch.Tensor]:
    """Validate a frame charge state and build its compression artifacts.

    Parameters
    ----------
    descriptor
        Evaluated pt_expt DPA4C descriptor.
    charge_spin
        Frame condition ``[charge, multiplicity]``. Read only when the
        descriptor is charge conditioned, and mandatory in that case.

    Returns
    -------
    dict[str, torch.Tensor]
        The artifacts named by :data:`CHARGE_STATE_ARTIFACTS`, detached.

    Raises
    ------
    ValueError
        If a charge-conditioned descriptor is given no condition, or one that
        does not address a row of both embedding tables.
    """
    state = None
    if descriptor.charge_spin_embedding is not None and charge_spin is not None:
        state = torch.tensor(
            validate_charge_state(charge_spin),
            dtype=torch.float32,
            device=next(descriptor.parameters()).device,
        )
    with torch.no_grad():
        return {
            name: value.detach().contiguous()
            for name, value in charge_state_artifacts(descriptor, state).items()
        }


class ChargeStateFold(torch.nn.Module):
    """Rebuild the charge-state artifacts of a compressed snapshot.

    Exporting this module beside the inference lower is what lets one
    deployed artifact serve any charge state: the deployment layer runs it
    once when the state becomes known and writes the four tensors over the
    corresponding constants of the inference lower. The alternative, folding
    inside the inference graph, would repeat an evaluation over
    ``(T + 1) ** 2`` ordered pairs on every step for a value that is constant
    over a molecular-dynamics run, and that evaluation does not shrink with
    the system size.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor carrying charge conditioning.
    """

    def __init__(self, descriptor: Any) -> None:
        super().__init__()
        self.descriptor = descriptor

    def forward(self, charge_spin: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Build the artifacts of one charge state.

        Parameters
        ----------
        charge_spin
            Frame condition with shape ``(1, 2)``, matching the tensor the
            inference lower would receive.

        Returns
        -------
        tuple[torch.Tensor, ...]
            The artifacts named by :data:`CHARGE_STATE_ARTIFACTS`, in order.
        """
        artifacts = charge_state_artifacts(self.descriptor, charge_spin.reshape(2))
        return tuple(artifacts[name] for name in CHARGE_STATE_ARTIFACTS)


def _build_spin_type_cache(descriptor: Any, device: torch.device) -> torch.Tensor:
    """Freeze the per-type table of the native spin branch.

    The table packs the four per-type scalars a node needs into one 128-bit
    row: the gate divided by the reference magnitude, which conditions the
    moment; the bare gate, which the magnetic-coordination family reads
    because it counts neighbours that carry a moment rather than the moments
    themselves; and the two on-site weights. None of them passes through the
    ordered pair encoder, so unlike ``spin_pair`` this table is independent of
    the frame charge state.

    Parameters
    ----------
    descriptor
        Evaluated pt_expt DPA4C descriptor.
    device
        Device that receives the packed table.

    Returns
    -------
    torch.Tensor
        Per-type table with shape ``(T + 1, 4)``, or an empty tensor for a
        spin-free descriptor.
    """
    if descriptor.spin is None:
        return torch.zeros(0, dtype=torch.float32, device=device)
    spin = descriptor.spin
    gate = spin.spin_mask.to(device=device, dtype=torch.float32)
    reference = spin.spin_reference.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        return (
            torch.stack(
                (
                    gate / reference,
                    gate,
                    spin.adam_spin_vector_weight.to(device=device, dtype=torch.float32),
                    spin.adam_spin_quadrupole_weight.to(
                        device=device, dtype=torch.float32
                    ),
                ),
                dim=-1,
            )
            .detach()
            .contiguous()
        )


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


def _reference_descriptor(
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
    *,
    spin: torch.Tensor | None = None,
    spin_pair: torch.Tensor | None = None,
    spin_type: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation of the compressed DPA4C descriptor.

    The native spin block is optional. It comprises the raw per-node moment
    ``spin`` with shape ``(N, 3)``, the ordered scale and shift cache
    ``spin_pair`` with shape ``((T + 1) ** 2, C_s, 2)``, and the per-type
    scalars ``spin_type`` with shape ``(T + 1, 4)``. A spin-free descriptor
    omits it and reproduces the geometric descriptor exactly.
    """
    del destination_order, destination_row_ptr, canonical, coupling_meta
    compute = edge_vec.to(torch.float32)
    source, destination = edge_index[0].to(torch.long), edge_index[1].to(torch.long)
    node_count = atype.shape[0]
    channels = type_embedding.shape[1]
    type_count = type_embedding.shape[0]
    has_spin = spin is not None and spin.ndim == 2
    profile = descriptor_profile(int(channels), int(lmax), has_spin)
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

    # === Step 5. Reduce and contract the native spin families ===
    spin_blocks: list[torch.Tensor] = []
    if has_spin:
        magnitude, coordination, spin_vector, spin_tensor = _reference_spin_moments(
            spin,
            spin_pair,
            spin_type,
            atype,
            source,
            destination,
            pair_index,
            direction,
            tabulated,
            envelope,
            angular_norm,
            node_count,
        )
        spin_blocks = [
            _half_gram(spin_vector),
            # The quadrupole Gram omits its leading diagonal entry, the
            # on-site self-term: the harmonic block is homogeneous, so
            # |B_2(s)|^2 = |s|^4 makes that entry a per-type constant times
            # the square of the vector block's own on-site self-term.
            _half_gram(spin_tensor)[:, 1:],
            # Cross Gram against the unaligned geometric degree-two moments.
            # Both factors carry even spin order, so the product is
            # admissible; with a unit direction it evaluates to the
            # single-ion anisotropy sum over neighbors.
            (spin_tensor.transpose(1, 2) @ blocks[2]).flatten(start_dim=1),
            magnitude,
            coordination,
        ]

    # === Step 6. Assemble and calibrate the invariant output ===
    quartic = (tensor_vector * tensor_vector).sum(dim=-1).flatten(start_dim=1)
    descriptor = torch.cat(
        [
            blocks[0][:, 0, :],
            *[_half_gram(block) for block in aligned[1:]],
            *[parts[key] for key in sorted(parts)],
            quartic,
            *spin_blocks,
            scalar_divisor,
            angular_divisor,
            type_embedding[atype],
        ],
        dim=-1,
    )
    return (descriptor - output_mean[None, :]) * output_inv_std[None, :]


def _reference_spin_moments(
    spin: torch.Tensor,
    spin_pair: torch.Tensor,
    spin_type: torch.Tensor,
    atype: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    pair_index: torch.Tensor,
    direction: torch.Tensor,
    radial: torch.Tensor,
    envelope: torch.Tensor,
    angular_norm: torch.Tensor,
    node_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Reduce the native spin families into their four moment blocks.

    The moment enters through the conditioned node quantity
    :math:`\hat{\mathbf s}_i=w_{a_i}\mathbf s_i`, where :math:`w` is the
    per-type gate divided by the per-type reference magnitude. Every
    neighbor family then shares one edge weight, the spin counterpart of the
    geometric amplitude,

    .. math::

       \phi^{s}_{ij,c}=\chi_{ij}^2\bigl(
         \gamma^{s}_{ab,c}g_c(\rho_{ij})+\beta^{s}_{ab,c}\bigr),

    whose squared envelope matches the weight of every non-scalar geometric
    moment. The reduced families therefore share the angular normalizer and
    need no neighborhood mass of their own. The bond-projected family is the
    one that reads the edge direction, and through it the spin branch
    contributes to the coordinate gradient angularly as well as radially.

    The on-site channel leads each non-scalar block and is node local: it is
    written outside the division, so an invariant pairing it with a neighbor
    channel carries exactly one neighborhood normalizer.

    Parameters
    ----------
    spin
        Raw per-node magnetic moments with shape ``(N, 3)``.
    spin_pair
        Ordered spin scale and shift with shape ``((T + 1) ** 2, C_s, 2)``.
    spin_type
        Per-type gate over reference, bare gate, on-site vector weight and
        on-site quadrupole weight with shape ``(T + 1, 4)``.
    atype
        Flat node types with shape ``(N,)``.
    source
        Source node index of each edge with shape ``(E,)``.
    destination
        Destination node index of each edge with shape ``(E,)``.
    pair_index
        Ordered type-pair index of each edge with shape ``(E,)``.
    direction
        Regularized unit edge directions with shape ``(E, 3)``.
    radial
        Tabulated distance maps with shape ``(E, channels + radial_modes)``;
        the leading ``C_s`` columns are read.
    envelope
        Masked C³ envelope with shape ``(E,)``.
    angular_norm
        Reciprocal angular divisor with shape ``(N, 1, 1)``.
    node_count
        Number of nodes ``N``.

    Returns
    -------
    magnitude
        Neighbor moment magnitudes with shape ``(N, C_s)``.
    coordination
        Magnetic effective coordination with shape ``(N, C_s)``.
    vector
        Joint degree-one spin block with shape ``(N, 3, 1 + 2 C_s)``, holding
        the on-site moment, the isotropic neighbor channels and the
        bond-projected neighbor channels in that order.
    tensor
        Degree-two spin block with shape ``(N, 5, 2)``.
    """
    spin_channels = int(spin_pair.shape[1])
    weights = spin_type[atype]
    conditioned = spin.to(envelope.dtype) * weights[:, 0:1]
    neighbor = conditioned[source]
    film = spin_pair[pair_index]
    weight = (radial[:, :spin_channels] * film[..., 0] + film[..., 1]) * (
        envelope * envelope
    )[:, None]
    # Component of the neighbor moment along the bond, carried back as a
    # vector so that the block Gram turns it into the bond-resolved
    # invariants. The masked envelope already zeroes excluded edges, so the
    # unmasked direction cannot leak through the amplitude.
    bond = direction * (neighbor * direction).sum(dim=-1, keepdim=True)

    # Payload layout: [M0, Mw, V_neighbor, P_neighbor, Q_neighbor] with the
    # harmonic component as the outer axis of each non-scalar family, matching
    # the geometric moment convention. The magnetic coordination reads the bare
    # neighbor gate because it counts neighbors that carry a moment rather
    # than the moments themselves, and is therefore nonzero at vanishing spin.
    payload = torch.cat(
        [
            weight * (neighbor * neighbor).sum(dim=-1, keepdim=True),
            weight * weights[source, 1:2],
            (neighbor[:, :, None] * weight[:, None, :]).flatten(start_dim=1),
            (bond[:, :, None] * weight[:, None, :]).flatten(start_dim=1),
            build_angular_basis(neighbor, 2)[:, 4:9] * weight[:, :1],
        ],
        dim=-1,
    )
    reduced = (
        torch.zeros(
            node_count,
            payload.shape[1],
            dtype=payload.dtype,
            device=payload.device,
        ).index_add_(0, destination, payload)
        * angular_norm[:, :, 0]
    )

    onsite_vector = conditioned * weights[:, 2:3]
    onsite_tensor = build_angular_basis(conditioned, 2)[:, 4:9] * weights[:, 3:4]
    return (
        reduced[:, :spin_channels],
        reduced[:, spin_channels : 2 * spin_channels],
        torch.cat(
            [
                onsite_vector[:, :, None],
                reduced[:, 2 * spin_channels : 5 * spin_channels].reshape(
                    -1,
                    3,
                    spin_channels,
                ),
                reduced[:, 5 * spin_channels : 8 * spin_channels].reshape(
                    -1,
                    3,
                    spin_channels,
                ),
            ],
            dim=-1,
        ),
        torch.cat(
            [
                onsite_tensor[:, :, None],
                reduced[:, 8 * spin_channels :, None],
            ],
            dim=-1,
        ),
    )


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


def _reference_forward(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU custom-op implementation returning descriptor and opaque state."""
    descriptor = _reference_descriptor(
        *args[:16],
        *args[19:],
        spin=args[16],
        spin_pair=args[17],
        spin_type=args[18],
    )
    profile = descriptor_profile(
        int(args[9].shape[1]),
        int(args[20]),
        args[16].ndim == 2,
    )
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
    spin: torch.Tensor,
    spin_pair: torch.Tensor,
    spin_type: torch.Tensor,
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
        spin_pair,
        spin_type,
        canonical,
        table_stride,
        table_max,
        rcut,
        eps,
        degree_floor,
    )
    profile = descriptor_profile(
        int(type_embedding.shape[1]), int(lmax), spin.ndim == 2
    )
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del descriptor_gradient, state
    spin = args[15]
    has_spin = spin.ndim == 2
    # Each absent output is allocated separately: the schema declares three
    # unannotated results, so two of them may not share storage.
    return (
        torch.empty_like(edge_vec),
        spin.new_empty((spin.shape[0], 3))
        if has_spin
        else edge_vec.new_empty((0,), dtype=torch.float32),
        torch.empty_like(edge_vec)
        if has_spin
        else edge_vec.new_empty((0,), dtype=torch.float32),
    )


def _reference_backward(
    descriptor_gradient: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU custom-op backward returning the coordinate and magnetic cotangents.

    The device operator splits the magnetic cotangent because a
    destination-major scan does not own the source of an edge, and the caller
    closes it by reducing the per-edge part onto source nodes. The reference
    differentiates the whole node axis at once, so it returns the complete
    cotangent on the node axis and leaves the per-edge part at zero, which the
    same reduction carries through unchanged.
    """
    del state
    spin = args[15]
    has_spin = spin.ndim == 2
    value = edge_vec.detach().clone().requires_grad_(True)
    moment = spin.detach().clone().requires_grad_(has_spin)
    with torch.enable_grad():
        descriptor = _reference_descriptor(
            value,
            *args[:15],
            *args[18:],
            spin=moment,
            spin_pair=args[16],
            spin_type=args[17],
        )
        gradients = torch.autograd.grad(
            (descriptor * descriptor_gradient.to(descriptor.dtype)).sum(),
            (value, moment) if has_spin else (value,),
        )
    return (
        gradients[0].to(edge_vec.dtype),
        gradients[1] if has_spin else edge_vec.new_empty((0,), dtype=torch.float32),
        torch.zeros_like(edge_vec)
        if has_spin
        else edge_vec.new_empty((0,), dtype=torch.float32),
    )


#: Position of the per-node magnetic moment among the operator inputs, and its
#: position among the tensors :func:`_setup_context` saves, which lead with the
#: opaque state.
_SPIN_INPUT_SLOT = 16
_SPIN_SAVED_SLOT = 1 + _SPIN_INPUT_SLOT


def _setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    ctx.save_for_backward(output[1], *inputs[:19])
    ctx.scalars = inputs[19:]
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)


def _backward(
    ctx: Any,
    descriptor_gradient: torch.Tensor,
    state_gradient: torch.Tensor | None,
) -> tuple:
    """Return the coordinate cotangent of one compressed descriptor call.

    Raises
    ------
    RuntimeError
        If the magnetic moment requires a gradient. The operator emits that
        cotangent in two pieces and the per-edge piece is reduced onto source
        nodes through the source CSR, which the operator schema does not carry;
        this registration therefore cannot close the magnetic force and
        refuses rather than reporting a vanishing one.
    """
    del state_gradient
    tensors = ctx.saved_tensors
    if tensors[_SPIN_SAVED_SLOT].requires_grad:
        raise RuntimeError(
            "deepmd::dpa4c_graph_compress cannot differentiate its magnetic "
            "moment through the registered autograd: closing the magnetic "
            "force needs the source CSR, which the operator schema does not "
            "carry. Call "
            "`deepmd.pt_expt.kernels.dpa4c.graph_compress.dpa4c_graph_compress`, "
            "which supplies it."
        )
    edge_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        descriptor_gradient,
        tensors[0],
        *tensors[1:],
        *ctx.scalars,
    )[0]
    return (edge_gradient,) + (None,) * 25


_registered = False


def ensure_registered() -> None:
    """Register the fake and autograd implementations once.

    Both devices implement the operator in C++, so the only Python-side
    registrations are the meta shapes ``torch.export`` needs and the autograd
    rule that connects the analytical backward.
    """
    global _registered
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::dpa4c_graph_compress")(_forward_fake)
    torch.library.register_fake("deepmd::dpa4c_graph_compress_backward")(_backward_fake)
    torch.library.register_autograd(
        "deepmd::dpa4c_graph_compress",
        _backward,
        setup_context=_setup_context,
    )
    _registered = True


def compressed_operator_arguments(
    descriptor: Any,
    spin: torch.Tensor | None = None,
) -> tuple:
    """Return the immutable operator arguments of a compressed descriptor.

    Parameters
    ----------
    descriptor
        Compressed pt_expt DPA4C descriptor.
    spin
        Per-node magnetic moments with shape ``(N_all, 3)`` for a
        spin-conditioned descriptor. The moment is a runtime input rather than
        an artifact; the two tables that condition it are frozen.

    Returns
    -------
    tuple
        Radial table, ordered caches, readout projections, coupling tables,
        output calibration, and the native spin block.
    """
    empty = descriptor.compress_spin_type.new_empty(0)
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
        empty if spin is None else spin.to(torch.float32).contiguous(),
        descriptor.compress_spin_pair,
        descriptor.compress_spin_type,
    )


def reduce_edge_spin_gradient(
    edge_spin_gradient: torch.Tensor,
    source_order: torch.Tensor,
    source_row_ptr: torch.Tensor,
) -> torch.Tensor:
    """Reduce a per-edge magnetic cotangent onto its source nodes.

    The source CSR groups the outgoing edges of a node contiguously, so the
    reduction is a segment sum over a gathered edge axis. That fixes the
    summation order from the topology rather than from arrival order, which an
    atomic scatter would not.

    The permutation is gathered in full rather than sliced to the physical
    edge count. The segments consume only the leading rows, which the source
    grouping already reserves for the physical edges, and reading the count
    off the row pointers would turn a device value into a Python integer that
    symbolic tracing cannot resolve.

    Parameters
    ----------
    edge_spin_gradient
        Per-edge magnetic cotangent with shape ``(E, 3)``.
    source_order
        Source-grouped edge permutation with shape ``(E,)``.
    source_row_ptr
        Source CSR offsets with shape ``(N + 1,)``.

    Returns
    -------
    torch.Tensor
        Per-node magnetic cotangent with shape ``(N, 3)``.
    """
    ordered = torch.index_select(
        edge_spin_gradient,
        0,
        source_order.to(torch.int64),
    )
    return torch.segment_reduce(
        ordered,
        "sum",
        lengths=(source_row_ptr[1:] - source_row_ptr[:-1]).to(torch.int64),
        axis=0,
        unsafe=True,
    )


class _CompressedDescriptor(torch.autograd.Function):
    """Autograd wrapper that closes the magnetic force of the level-one path.

    The operator emits the on-site magnetic gradient per node and the
    neighbour part per edge, because a destination-major scan does not own the
    source of an edge. Reducing the second onto source nodes needs the source
    CSR, which the operator schema does not carry, so the reduction lives here
    where the graph is in scope.
    """

    @staticmethod
    def forward(
        ctx: Any,
        edge_vec: torch.Tensor,
        spin: torch.Tensor,
        source_order: torch.Tensor,
        source_row_ptr: torch.Tensor,
        operator_args: tuple,
    ) -> torch.Tensor:
        descriptor, state = torch.ops.deepmd.dpa4c_graph_compress(
            edge_vec, *operator_args
        )
        ctx.save_for_backward(state, edge_vec, source_order, source_row_ptr)
        ctx.operator_args = operator_args
        return descriptor

    @staticmethod
    def backward(ctx: Any, descriptor_gradient: torch.Tensor) -> tuple:
        state, edge_vec, source_order, source_row_ptr = ctx.saved_tensors
        (
            edge_gradient,
            spin_gradient,
            edge_spin_gradient,
        ) = torch.ops.deepmd.dpa4c_graph_compress_backward(
            descriptor_gradient.contiguous(),
            state,
            edge_vec,
            *ctx.operator_args,
        )
        spin_gradient = spin_gradient + reduce_edge_spin_gradient(
            edge_spin_gradient,
            source_order,
            source_row_ptr,
        )
        return edge_gradient, spin_gradient, None, None, None


def dpa4c_graph_compress(
    descriptor: Any,
    graph: Any,
    atype: torch.Tensor,
    spin: torch.Tensor | None = None,
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
    spin
        Per-node magnetic moments with shape ``(N, 3)``, or ``None``.

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
    operator_args = (
        graph.edge_index.contiguous(),
        graph.edge_mask.contiguous(),
        graph.destination_order.contiguous(),
        graph.destination_row_ptr.contiguous(),
        atype.contiguous(),
        *compressed_operator_arguments(descriptor, spin),
        bool(graph.destination_sorted),
        int(descriptor.lmax),
        *descriptor._compression_scalars,
    )
    if spin is not None and spin.requires_grad:
        if graph.source_order is None or graph.source_row_ptr is None:
            raise ValueError(
                "DPA4C compressed CUDA requires source CSR topology to close "
                "the magnetic force"
            )
        descriptor_output = _CompressedDescriptor.apply(
            graph.edge_vec.contiguous(),
            spin,
            graph.source_order.contiguous(),
            graph.source_row_ptr.contiguous(),
            operator_args,
        )
    else:
        descriptor_output, _state = torch.ops.deepmd.dpa4c_graph_compress(
            graph.edge_vec.contiguous(),
            *operator_args,
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
    spin: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate compressed DPA4C energy, force, virial and magnetic force.

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
    spin
        Per-node magnetic moments with shape ``(N, 3)``, or ``None``.

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
    force_mag
        Per-node magnetic force with shape ``(N, 3)``, or an empty tensor.

    Raises
    ------
    ValueError
        If the graph lacks destination or source CSR topology.
    """
    from deepmd.pt_expt.kernels.edge_force_virial import (
        edge_force_virial,
    )
    from deepmd.pt_expt.kernels.edge_force_virial import (
        ensure_registered as ensure_force_registered,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
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
        *compressed_operator_arguments(descriptor, spin),
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
    (
        edge_gradient,
        spin_gradient,
        edge_spin_gradient,
    ) = torch.ops.deepmd.dpa4c_graph_compress_backward(
        descriptor_gradient,
        state,
        edge_vec,
        *operator_args,
    )
    # The on-site magnetic gradient closes in the node kernel; the neighbour
    # part belongs to source nodes, and the force assembly already walks that
    # grouping, so it is reduced there rather than in a pass of its own.
    force, atom_virial, virial, source_spin = edge_force_virial(
        edge_gradient,
        edge_vec,
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
        graph.n_node,
        edge_spin_gradient,
        node_capacity,
        do_atomic_virial,
    )
    force_mag = spin_gradient if spin is None else -(spin_gradient + source_spin)
    return energy, atom_energy, force, virial, atom_virial, force_mag


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
    from deepmd.pt_expt.kernels.edge_force_virial import (
        frame_scalar_sum,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
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
