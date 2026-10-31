# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Compressed artifacts and portable equations of the further DPA4C blocks.

A block after the first (:class:`~deepmd.dpmodel.descriptor.dpa4c_nn.BlockAttention`)
reads features of the edge sources, so none of its maps is a function of the
pair distance alone that a radial table could hold. Compression therefore
packs the weights of every further block into one fp32 record in the layout
the compiled operator reads, and the operator evaluates each block exactly.

Block :math:`k` corrects the moments of the first block over the edges inside
its own cutoff. With the attention factor :math:`m=1+t`,
:math:`t_{ijc}=\chi_k(r_{ij})\tanh a_{ijc}`, which vanishes with the envelope
of the block,

.. math::

   X^{(0),k}_i=X^{(0),1}_i+\frac{1}{d^{(0)}_i}\sum_j\phi_{ij}t_{ij},\qquad
   X^{(\ell),k}_i=X^{(\ell),1}_i+\frac{1}{d^{(+)}_i}
   \sum_j\chi_{ij}\phi_{ij}t_{ij}B^{(\ell)}_{ij}+\Delta X^{(\ell),k}_i,

where :math:`d^{(\bullet)}` are the divisors of the first block and
:math:`\Delta X` the transported keys. This is the reduction of
:meth:`~deepmd.dpmodel.descriptor.dpa4c.DescrptDPA4C.reduce_moments` over the
rescaled amplitudes :math:`\phi m`, written so that the correction reads
:math:`t` directly rather than the difference :math:`m-1`. The native spin
moments are those of the first block.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from .graph_compress import (
        DescriptorProfile,
    )

__all__ = [
    "HEADS",
    "STATE_WIDTH",
    "BlockLayout",
    "block_layout",
    "build_block_artifacts",
    "further_block_moments",
]

#: Heads of the attention, a width the compiled operator specializes on.
HEADS = 2
#: Hidden width of the state encoder, a width the compiled operator
#: specializes on.
STATE_WIDTH = 16
#: Floats of the record header: the cutoff of the block, the knees of the
#: three saturations, the logit cap and the null floor, and two padding words.
HEADER_WIDTH = 8


def _aligned(size: int) -> int:
    """Round a section width up to a whole number of 16-byte vectors."""
    return (size + 3) // 4 * 4


@dataclass(frozen=True)
class BlockLayout:
    r"""Float offsets of the sections of the record of one further block.

    Every section starts on a 16-byte boundary. Matrices are stored input-major,
    as ``NativeLayer.w``: entry ``(k, c)`` of a map with ``n_out`` outputs is
    at ``offset + k * n_out + c``.

    The two square maps of the edge network each follow the term of the fields
    that enters the same pre-activation, so that the operator contracts one
    stacked operand per map: the hidden layer reads ``[F, sat(phi)]`` through
    ``[W_F; W_phi]``, and the logit reads ``[F, silu(hid)]`` through
    ``[G; W_O]``, where ``G`` scatters the field gain of head ``h`` onto the
    channels of that head. Both square maps are also stored transposed, for
    the backward.

    Attributes
    ----------
    channels
        Scalar width :math:`C_0`.
    degree_channels
        Channel width of every degree from zero through ``lmax``.
    descriptor_width
        Width of the calibrated invariants that the node maps read.
    keys
        Offset of the key projection of every degree from one, ``(C_l, H)``.
    node_weight
        Offset of the input layer of the node maps, ``(descriptor_width,
        node_width)``: the gate and the value of the state encoder, then the
        hidden-layer term, then the null-mass logits, then padding.
    node_width
        Padded column count of the node input layer.
    node_bias
        Offset of the bias of the node input layer, ``(node_width,)``; only
        the null-mass columns are nonzero.
    state_output
        Offset of the output layer of the state encoder, ``(STATE_WIDTH, C_0)``.
    edge_field
        Offset of the field term of the edge hidden layer, ``(L H, C_0)``,
        immediately followed by its amplitude term ``(C_0, C_0)``.
    edge_amplitude
        Offset of the amplitude term of the edge hidden layer.
    edge_gain
        Offset of the field gain scattered onto the channels of every head,
        ``(L H, C_0)``, immediately followed by the edge output layer.
    edge_output
        Offset of the edge output layer, ``(C_0, C_0)``.
    edge_amplitude_t, edge_output_t
        Offsets of the transposes of the two square maps.
    edge_transport
        Offset of the radial weight of the transport, ``(C_0, H)``.
    outputs
        Offset of the transport output of every degree from one, ``(H, C_l)``.
    width
        Floats of the whole record.
    """

    channels: int
    degree_channels: tuple[int, ...]
    descriptor_width: int
    keys: tuple[int, ...]
    node_weight: int
    node_width: int
    node_bias: int
    state_output: int
    edge_field: int
    edge_amplitude: int
    edge_gain: int
    edge_output: int
    edge_amplitude_t: int
    edge_output_t: int
    edge_transport: int
    outputs: tuple[int, ...]
    width: int

    @property
    def lmax(self) -> int:
        """Return the maximum angular degree."""
        return len(self.degree_channels) - 1

    @property
    def harmonic_width(self) -> int:
        """Return the number of non-scalar harmonic components."""
        return (self.lmax + 1) ** 2 - 1

    @property
    def field_width(self) -> int:
        """Return the number of fields of an edge, one per degree and head."""
        return self.lmax * HEADS

    @property
    def source_width(self) -> int:
        """Return the width of the source features of a node.

        The keys, harmonic-major with the heads innermost, then the encoded
        state and the neighbor term of the edge hidden layer.
        """
        return self.harmonic_width * HEADS + 2 * self.channels


def block_layout(profile: DescriptorProfile) -> BlockLayout:
    """Derive the record layout of a further block from the compiled profile.

    The layout mirrors ``BlockLayout`` of the operator, which validates the
    record width against it.

    Parameters
    ----------
    profile
        Compiled profile of the descriptor.

    Returns
    -------
    BlockLayout
        Offsets of every section of the record.
    """
    channels = profile.channels
    lmax = profile.lmax
    fields = lmax * HEADS
    offset = HEADER_WIDTH
    keys = []
    for width in profile.degree_channels[1:]:
        keys.append(offset)
        offset += _aligned(width * HEADS)
    node_width = _aligned(2 * STATE_WIDTH + channels + HEADS)
    node_weight = offset
    offset += profile.output_width * node_width
    node_bias = offset
    offset += node_width
    state_output = offset
    offset += STATE_WIDTH * channels
    edge_field = offset
    edge_amplitude = edge_field + fields * channels
    offset = edge_amplitude + channels * channels
    edge_gain = offset
    edge_output = edge_gain + fields * channels
    offset = edge_output + channels * channels
    edge_amplitude_t = offset
    offset += channels * channels
    edge_output_t = offset
    offset += channels * channels
    edge_transport = offset
    offset += channels * HEADS
    outputs = []
    for width in profile.degree_channels[1:]:
        outputs.append(offset)
        offset += _aligned(HEADS * width)
    return BlockLayout(
        channels=channels,
        degree_channels=tuple(profile.degree_channels),
        descriptor_width=profile.output_width,
        keys=tuple(keys),
        node_weight=node_weight,
        node_width=node_width,
        node_bias=node_bias,
        state_output=state_output,
        edge_field=edge_field,
        edge_amplitude=edge_amplitude,
        edge_gain=edge_gain,
        edge_output=edge_output,
        edge_amplitude_t=edge_amplitude_t,
        edge_output_t=edge_output_t,
        edge_transport=edge_transport,
        outputs=tuple(outputs),
        width=offset,
    )


def _check_attention(attention: Any, layout: BlockLayout) -> None:
    """Reject a block whose widths or constants the operator does not compile."""
    if attention.HEADS != HEADS or attention.STATE_WIDTH != STATE_WIDTH:
        raise ValueError(
            "The compressed DPA4C operator is compiled for "
            f"{HEADS} attention heads and a state width of {STATE_WIDTH}, got "
            f"{attention.HEADS} and {attention.STATE_WIDTH}."
        )
    if tuple(attention.degree_channels) != layout.degree_channels:
        raise ValueError(
            "A further DPA4C block reads the degree channels "
            f"{tuple(attention.degree_channels)}, but the descriptor profile "
            f"has {layout.degree_channels}."
        )
    if int(attention.descriptor_dim) != layout.descriptor_width:
        raise ValueError(
            "A further DPA4C block reads invariants of width "
            f"{attention.descriptor_dim}, but the descriptor emits "
            f"{layout.descriptor_width}."
        )


def _pack_block(
    attention: Any,
    cutoff: float,
    layout: BlockLayout,
    device: torch.device,
) -> torch.Tensor:
    """Pack the weights of one further block into its record."""
    record = torch.zeros(layout.width, dtype=torch.float32, device=device)

    def put(offset: int, value: Any) -> None:
        flat = torch.as_tensor(value).to(device=device, dtype=torch.float32)
        flat = flat.reshape(-1)
        record[offset : offset + flat.numel()] = flat

    header = (
        cutoff,
        attention.STATE_KNEE,
        attention.KEY_KNEE,
        attention.AMPLITUDE_KNEE,
        attention.LOGIT_CAP,
        attention.NULL_FLOOR,
    )
    put(0, torch.tensor(header, dtype=torch.float64, device=device))
    for offset, projection in zip(layout.keys, attention.key_projections, strict=True):
        put(offset, projection.w)

    # The three maps of the calibrated invariants share their input, so they
    # form one layer: gate and value of the state encoder, the hidden-layer
    # term of the neighbor, then the null-mass logits of the center.
    channels = layout.channels
    node = torch.zeros(
        layout.descriptor_width,
        layout.node_width,
        dtype=torch.float32,
        device=device,
    )
    node[:, : 2 * STATE_WIDTH] = torch.as_tensor(attention.state_encoder.layers[0].w)
    node[:, 2 * STATE_WIDTH : 2 * STATE_WIDTH + channels] = torch.as_tensor(
        attention.hidden_state.w
    )
    null_begin = 2 * STATE_WIDTH + channels
    node[:, null_begin : null_begin + HEADS] = torch.as_tensor(attention.null_mass.w)
    put(layout.node_weight, node)
    bias = torch.zeros(layout.node_width, dtype=torch.float32, device=device)
    bias[null_begin : null_begin + HEADS] = torch.as_tensor(attention.null_mass.b)
    put(layout.node_bias, bias)
    put(layout.state_output, attention.state_encoder.layers[-1].w)

    put(layout.edge_field, attention.hidden_field.w)
    amplitude = torch.as_tensor(attention.hidden_amplitude.w)
    output = torch.as_tensor(attention.hidden_output.w)
    put(layout.edge_amplitude, amplitude)
    put(layout.edge_amplitude_t, amplitude.T)
    # Amplitude channel c follows head c mod H, so the gain of field (l, h)
    # reaches exactly the channels of head h.
    gain = torch.as_tensor(attention.adam_field_gain).to(
        device=device, dtype=torch.float32
    )
    head = torch.arange(channels, dtype=torch.int64, device=device) % HEADS
    field_head = (
        torch.arange(layout.field_width, dtype=torch.int64, device=device) % HEADS
    )
    scatter = gain.reshape(layout.field_width)[:, None] * (
        head[None, :] == field_head[:, None]
    ).to(torch.float32)
    put(layout.edge_gain, scatter)
    put(layout.edge_output, output)
    put(layout.edge_output_t, output.T)
    put(layout.edge_transport, attention.transport_radial.w)
    for offset, projection in zip(
        layout.outputs, attention.transport_outputs, strict=True
    ):
        put(offset, projection.w)
    return record


def build_block_artifacts(
    descriptor: Any,
    profile: DescriptorProfile,
) -> torch.Tensor:
    """Pack every further block of a descriptor into the operator records.

    Parameters
    ----------
    descriptor
        Evaluated pt_expt DPA4C descriptor.
    profile
        Compiled profile of the descriptor.

    Returns
    -------
    torch.Tensor
        Records with shape ``(n_blocks - 1, layout.width)``, fp32, on the
        device of the descriptor; shape ``(0, 0)`` for a single block.

    Raises
    ------
    ValueError
        If a block carries widths or constants the operator does not compile.
    """
    device = descriptor.stddev.device
    if not descriptor.attentions:
        return torch.zeros(0, 0, dtype=torch.float32, device=device)
    layout = block_layout(profile)
    records = []
    with torch.no_grad():
        for attention, cutoff in zip(
            descriptor.attentions, descriptor.block_rcut[1:], strict=True
        ):
            _check_attention(attention, layout)
            records.append(_pack_block(attention, float(cutoff), layout, device))
    return torch.stack(records).contiguous()


# === Portable equations ===


def _soft_saturate(value: torch.Tensor, knee: float) -> torch.Tensor:
    ratio = value / knee
    return value / torch.sqrt(1.0 + ratio * ratio)


def _c3_envelope(radius: torch.Tensor, rcut: float) -> torch.Tensor:
    u = torch.clamp((rcut - radius) / rcut, min=0.0, max=1.0)
    x = 1.0 - u
    series = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + 35.0 * x)))
    return u**4 * series


def _section(
    record: torch.Tensor, offset: int, rows: int, columns: int
) -> torch.Tensor:
    return record[offset : offset + rows * columns].reshape(rows, columns)


def further_block_moments(
    record: torch.Tensor,
    layout: BlockLayout,
    first_moments: list[torch.Tensor],
    previous_moments: list[torch.Tensor],
    invariants: torch.Tensor,
    amplitude: torch.Tensor,
    basis: torch.Tensor,
    envelope: torch.Tensor,
    radius: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    scalar_divisor: torch.Tensor,
    angular_divisor: torch.Tensor,
    exchange: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Evaluate one further block on the compressed record.

    The equations are those of
    :class:`~deepmd.dpmodel.descriptor.dpa4c_nn.BlockAttention` followed by the
    reduction of the block, with the correction of the moments formed as in
    the module docstring.

    Parameters
    ----------
    record
        Packed record of the block with shape ``(layout.width,)``.
    layout
        Layout of the record.
    first_moments
        Normalized geometric moments of the first block, one tensor per
        degree with shape ``(N, 2 l + 1, C_l)``, which the block corrects.
    previous_moments
        Geometric moments of the previous block, shaped as ``first_moments``,
        from which the keys are formed.
    invariants
        Calibrated invariants of the previous block with shape ``(N, D)``.
    amplitude
        Enveloped first-block amplitudes with shape ``(E, C_0)``.
    basis
        Masked Cartesian harmonics with shape ``(E, (L + 1) ** 2)``.
    envelope
        Masked first-block envelope with shape ``(E,)``.
    radius
        Regularized edge lengths with shape ``(E,)`` in Å.
    source, destination
        Source and destination node of every edge with shape ``(E,)``.
    scalar_divisor, angular_divisor
        Divisors of the first block with shape ``(N, 1)``.
    exchange
        Refresh of the halo rows of a node feature from their owners, or
        ``None`` for a graph without halo nodes.

    Returns
    -------
    list[torch.Tensor]
        Moments of the block, one tensor per degree, shaped as
        ``first_moments``.
    """
    cutoff, state_knee, key_knee, amplitude_knee, logit_cap, null_floor = (
        float(value) for value in record[:6].tolist()
    )
    channels = layout.channels
    lmax = layout.lmax
    node_count = invariants.shape[0]
    edge_count = amplitude.shape[0]
    dtype = amplitude.dtype

    # === Step 1. Node maps: saturated keys, state, hidden term, null mass ===
    keys = []
    for degree, width in enumerate(layout.degree_channels[1:], start=1):
        projection = _section(record, layout.keys[degree - 1], width, HEADS)
        block = previous_moments[degree] @ projection.to(dtype)
        squared = (block * block).sum(dim=1, keepdim=True)
        keys.append(block / torch.sqrt(1.0 + squared / (key_knee * key_knee)))
    keys = torch.cat(keys, dim=1)
    saturated_invariants = _soft_saturate(invariants, state_knee)
    pre = saturated_invariants @ _section(
        record, layout.node_weight, layout.descriptor_width, layout.node_width
    ).to(dtype) + record[layout.node_bias : layout.node_bias + layout.node_width].to(
        dtype
    )
    gate = pre[:, :STATE_WIDTH]
    value = pre[:, STATE_WIDTH : 2 * STATE_WIDTH]
    state = (torch.nn.functional.silu(gate) * value) @ _section(
        record, layout.state_output, STATE_WIDTH, channels
    ).to(dtype)
    hidden_node = pre[:, 2 * STATE_WIDTH : 2 * STATE_WIDTH + channels]
    null_logit = pre[:, 2 * STATE_WIDTH + channels : 2 * STATE_WIDTH + channels + HEADS]
    sources = torch.cat(
        [keys.reshape(node_count, -1), state, hidden_node],
        dim=1,
    )
    if exchange is not None:
        sources = exchange(sources)

    # === Step 2. Fields of the neighbor along the reversed bond ===
    key_end = layout.harmonic_width * HEADS
    source_keys = sources[source, :key_end].reshape(
        edge_count, layout.harmonic_width, HEADS
    )
    source_state = sources[source, key_end : key_end + channels]
    source_hidden = sources[source, key_end + channels :]
    sign = torch.cat(
        [
            torch.full(
                (2 * degree + 1,), (-1.0) ** degree, dtype=dtype, device=basis.device
            )
            for degree in range(1, lmax + 1)
        ]
    )
    signed = (basis[:, 1:] * sign)[:, :, None] * source_keys
    fields = torch.stack(
        [
            signed[:, degree * degree - 1 : (degree + 1) * (degree + 1) - 1, :].sum(
                dim=1
            )
            for degree in range(1, lmax + 1)
        ],
        dim=1,
    )

    # === Step 3. Logit of every amplitude channel and its correction ===
    saturated = _soft_saturate(amplitude, amplitude_knee)
    hidden = (
        source_hidden
        + fields.reshape(edge_count, layout.field_width)
        @ _section(record, layout.edge_field, layout.field_width, channels).to(dtype)
        + saturated
        @ _section(record, layout.edge_amplitude, channels, channels).to(dtype)
    )
    flat_fields = fields.reshape(edge_count, layout.field_width)
    logit = (
        source_state
        + flat_fields
        @ _section(record, layout.edge_gain, layout.field_width, channels).to(dtype)
        + torch.nn.functional.silu(hidden)
        @ _section(record, layout.edge_output, channels, channels).to(dtype)
    )
    block_envelope = _c3_envelope(radius, cutoff)
    delta = amplitude * (block_envelope[:, None] * torch.tanh(logit))

    # === Step 4. Softmax transport of the neighbor keys ===
    weight = envelope * block_envelope
    capped = logit_cap * torch.tanh(logit[:, :HEADS] / logit_cap)
    mass = (weight * weight)[:, None] * torch.exp(capped)
    denominator = (
        torch.zeros(node_count, HEADS, dtype=dtype, device=basis.device).index_add_(
            0, destination, mass
        )
        + torch.nn.functional.softplus(null_logit)
        + null_floor
    )
    radial_weight = saturated @ _section(
        record, layout.edge_transport, channels, HEADS
    ).to(dtype)
    transported = torch.zeros(
        node_count, layout.harmonic_width, HEADS, dtype=dtype, device=basis.device
    ).index_add_(0, destination, (mass * radial_weight)[:, None, :] * source_keys)
    transported = transported / denominator[:, None, :]

    # === Step 5. Correct the moments of the first block ===
    corrected = [
        first_moments[0]
        + torch.zeros(
            node_count, 1, channels, dtype=dtype, device=basis.device
        ).index_add_(0, destination, delta[:, None, :])
        / scalar_divisor[:, :, None]
    ]
    for degree, width in enumerate(layout.degree_channels[1:], start=1):
        harmonics = basis[:, degree * degree : (degree + 1) * (degree + 1)]
        payload = harmonics[:, :, None] * (
            delta[:, None, :width] * envelope[:, None, None]
        )
        reduced = torch.zeros(
            node_count, 2 * degree + 1, width, dtype=dtype, device=basis.device
        ).index_add_(0, destination, payload)
        output = _section(record, layout.outputs[degree - 1], HEADS, width).to(dtype)
        start = degree * degree - 1
        corrected.append(
            first_moments[degree]
            + reduced / angular_divisor[:, :, None]
            + transported[:, start : start + 2 * degree + 1, :] @ output
        )
    return corrected
