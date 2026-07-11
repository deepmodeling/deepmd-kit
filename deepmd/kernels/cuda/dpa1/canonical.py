# SPDX-License-Identifier: LGPL-3.0-or-later
"""CUDA bindings for compact canonical compressed DPA1 deployment."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

if TYPE_CHECKING:
    from deepmd.pt_expt.utils.canonical_graph import (
        DPA1CanonicalGraph,
    )

__all__ = [
    "canonical_model_eligible",
    "dpa1_canonical_compress_energy_force",
    "ensure_registered",
    "op_available",
]

_registered = False
_cpu_library: torch.library.Library | None = None


def canonical_model_eligible(model: Any) -> bool:
    """Return whether a model can use the no-mask compact deployment ABI."""
    atomic_model = getattr(model, "atomic_model", None)
    descriptor = getattr(atomic_model, "descriptor", None)
    fitting = getattr(atomic_model, "fitting_net", None)
    if descriptor is None or fitting is None:
        return False
    if not bool(getattr(descriptor, "geo_compress", False)):
        return False
    eligible = getattr(descriptor, "_fused_eligible", None)
    if not callable(eligible) or not bool(eligible("cuda")):
        return False
    if getattr(atomic_model, "pair_excl", None) is not None:
        return False
    if getattr(atomic_model, "atom_excl", None) is not None:
        return False
    from deepmd.kernels.cuda.dpa1.graph_compress import (
        mega_eligible,
    )
    from deepmd.kernels.cuda.graph_fitting import (
        fitting_eligible,
    )

    return mega_eligible(descriptor) and fitting_eligible(fitting)


def op_available() -> bool:
    """Return whether both compact canonical descriptor operators are loaded."""
    forward = getattr(torch.ops.deepmd, "dpa1_canonical_compress", None)
    backward = getattr(torch.ops.deepmd, "dpa1_canonical_compress_backward", None)
    return isinstance(forward, torch._ops.OpOverloadPacket) and isinstance(
        backward, torch._ops.OpOverloadPacket
    )


def _forward_fake(
    edge_vec: torch.Tensor,
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    average: torch.Tensor,
    inverse_stddev: torch.Tensor,
    table: torch.Tensor,
    gate_table: torch.Tensor,
    type_one_side: int,
    concatenate_type_embedding: int,
    write_rotation: int,
    smooth: int,
    axis: int,
    lower: float,
    upper: float,
    table_max: float,
    stride0: float,
    stride1: float,
    rcut: float,
    rcut_smooth: float,
    protection: float,
    neighbors: float,
) -> tuple[torch.Tensor, ...]:
    del (
        source,
        destination_row_ptr,
        average,
        inverse_stddev,
        gate_table,
        type_one_side,
        smooth,
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        rcut,
        rcut_smooth,
        protection,
        neighbors,
    )
    node_count = atype.shape[0]
    width = table.shape[1] // 6
    output_width = width * axis + (
        type_embedding.shape[1] if concatenate_type_embedding else 0
    )
    return (
        edge_vec.new_empty(node_count, output_width),
        edge_vec.new_empty(
            node_count if write_rotation else 0,
            width,
            3,
        ),
        edge_vec.new_empty(node_count, 4, width),
    )


def _backward_fake(
    descriptor_gradient: torch.Tensor,
    rotation_gradient: torch.Tensor | None,
    moment: torch.Tensor,
    edge_vec: torch.Tensor,
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    atype: torch.Tensor,
    average: torch.Tensor,
    inverse_stddev: torch.Tensor,
    table: torch.Tensor,
    gate_table: torch.Tensor,
    type_one_side: int,
    smooth: int,
    axis: int,
    lower: float,
    upper: float,
    table_max: float,
    stride0: float,
    stride1: float,
    rcut: float,
    rcut_smooth: float,
    protection: float,
    neighbors: float,
) -> torch.Tensor:
    del (
        descriptor_gradient,
        rotation_gradient,
        moment,
        source,
        destination_row_ptr,
        atype,
        average,
        inverse_stddev,
        table,
        gate_table,
        type_one_side,
        smooth,
        axis,
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        rcut,
        rcut_smooth,
        protection,
        neighbors,
    )
    return torch.empty_like(edge_vec)


def _generic_topology(
    source: torch.Tensor,
    destination_row_ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    physical_edge_count = int(destination_row_ptr[-1].item())
    node_count = destination_row_ptr.shape[0] - 1
    destination = torch.repeat_interleave(
        torch.arange(node_count, dtype=torch.int64, device=source.device),
        destination_row_ptr[1:] - destination_row_ptr[:-1],
        output_size=physical_edge_count,
    )
    source_i64 = source.to(torch.int64)
    destination_storage = torch.zeros_like(source_i64)
    destination_storage[:physical_edge_count] = destination
    edge_index = torch.stack((source_i64, destination_storage))
    edge_mask = (
        torch.arange(
            source.shape[0],
            dtype=torch.int64,
            device=source.device,
        )
        < physical_edge_count
    )
    destination_order = torch.arange(
        source.shape[0],
        dtype=source.dtype,
        device=source.device,
    )
    return edge_index, edge_mask, destination_order


def _cpu_forward(*args: Any) -> tuple[torch.Tensor, ...]:
    from deepmd.kernels.cuda.dpa1.graph_compress import _cpu_forward as generic_forward

    edge_vec, source, destination_row_ptr, *tail = args
    edge_index, edge_mask, destination_order = _generic_topology(
        source,
        destination_row_ptr,
    )
    return generic_forward(
        edge_vec,
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        *tail[:11],
        True,
        *tail[11:],
    )


def _cpu_backward(*args: Any) -> torch.Tensor:
    from deepmd.kernels.cuda.dpa1.graph_compress import (
        _cpu_backward as generic_backward,
    )

    descriptor_gradient, rotation_gradient, moment, edge_vec = args[:4]
    source, destination_row_ptr = args[4:6]
    tail = args[6:]
    edge_index, edge_mask, destination_order = _generic_topology(
        source,
        destination_row_ptr,
    )
    return generic_backward(
        descriptor_gradient,
        rotation_gradient,
        moment,
        edge_vec,
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        *tail[:8],
        True,
        *tail[8:],
    )


def ensure_registered() -> None:
    """Register fake and CPU implementations for compact canonical operators."""
    global _registered, _cpu_library
    if _registered or not op_available():
        return
    torch.library.register_fake("deepmd::dpa1_canonical_compress")(_forward_fake)
    torch.library.register_fake("deepmd::dpa1_canonical_compress_backward")(
        _backward_fake
    )
    _cpu_library = torch.library.Library("deepmd", "IMPL")
    _cpu_library.impl("dpa1_canonical_compress", _cpu_forward, "CPU")
    _cpu_library.impl(
        "dpa1_canonical_compress_backward",
        _cpu_backward,
        "CPU",
    )
    _registered = True


def dpa1_canonical_compress_energy_force(
    desc: Any,
    fit: Any,
    graph: DPA1CanonicalGraph,
    atype: torch.Tensor,
    type_embedding: torch.Tensor,
    ownership: torch.Tensor,
    atom_bias: torch.Tensor,
    do_atomic_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate compressed DPA1 from a compact canonical edge stream.

    Parameters
    ----------
    desc
        Geometrically compressed DPA1 descriptor.
    fit
        Compatible mixed-type energy fitting network.
    graph
        Compact destination-major graph.
    atype
        Flat node atom types with shape ``(N,)``.
    type_embedding
        Type embedding table.
    ownership
        Energy-contributing node mask with shape ``(N,)``.
    atom_bias
        Combined fitting and atomic-model energy bias.
    do_atomic_virial
        Whether to materialize per-node virial.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Frame energy, atom energy, force, frame virial, and atom virial.
    """
    from deepmd.kernels.cuda.dpa1.graph_compress import (
        mega_eligible,
    )
    from deepmd.kernels.cuda.edge_force_virial import (
        canonical_edge_force_virial,
        canonical_op_available,
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
    if not mega_eligible(desc) or not canonical_op_available():
        raise ValueError("model is not eligible for compact canonical DPA1 inference")

    se = desc.se_atten
    compress_data = desc.compress_data[0].contiguous()
    gate_table = desc.type_embd_data.contiguous()
    inverse_stddev = torch.reciprocal(se.stddev[:, 0, :]).contiguous()
    from torch.fx.experimental.proxy_tensor import (
        disable_proxy_modes_tracing,
    )

    with disable_proxy_modes_tracing():
        lower, upper, table_max, stride0, stride1 = (
            float(value) for value in desc.compress_info[0].tolist()[:5]
        )

    descriptor, _rotation, moment = torch.ops.deepmd.dpa1_canonical_compress(
        graph.edge_vec,
        graph.source,
        graph.destination_row_ptr,
        atype,
        type_embedding,
        se.mean[:, 0, :].contiguous(),
        inverse_stddev,
        compress_data,
        gate_table,
        int(se.type_one_side),
        int(desc.concat_output_tebd),
        0,
        int(se.smooth),
        int(se.axis_neuron),
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
    )

    *hidden, head = fit.nets[0].layers
    empty = hidden[0].w.new_empty(0)
    weights = [layer.w.contiguous() for layer in hidden]
    residuals = [1 if layer.resnet else 0 for layer in hidden]
    from deepmd.kernels.triton.dpa1.activation import (
        ACT_CODES,
    )

    atom_energy_raw, fitting_saved = torch.ops.deepmd.graph_fitting(
        descriptor,
        atype,
        weights,
        [layer.b.contiguous() if layer.b is not None else empty for layer in hidden],
        [
            layer.idt.contiguous() if layer.idt is not None else empty
            for layer in hidden
        ],
        residuals,
        head.w.reshape(-1).contiguous(),
        (
            head.b.reshape(-1).to(torch.float32).contiguous()
            if head.b is not None
            else empty
        ),
        atom_bias.to(torch.float64).contiguous(),
        ACT_CODES[str(hidden[0].activation_function).lower()],
    )
    energy_seed = ownership[:, None].to(atom_energy_raw.dtype)
    atom_energy = atom_energy_raw * energy_seed
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
    )

    frame_index = frame_id_from_n_node(
        graph.n_node,
        n_total=atom_energy.shape[0],
    )
    energy = torch.zeros(
        graph.n_node.shape[0],
        1,
        dtype=atom_energy.dtype,
        device=atom_energy.device,
    ).index_add_(0, frame_index, atom_energy)
    del descriptor
    descriptor_gradient = torch.ops.deepmd.graph_fitting_backward(
        energy_seed,
        fitting_saved,
        weights,
        residuals,
        head.w.reshape(-1).contiguous(),
    )
    del fitting_saved
    edge_gradient = torch.ops.deepmd.dpa1_canonical_compress_backward(
        descriptor_gradient,
        None,
        moment,
        graph.edge_vec,
        graph.source,
        graph.destination_row_ptr,
        atype,
        se.mean[:, 0, :].contiguous(),
        inverse_stddev,
        compress_data,
        gate_table,
        int(se.type_one_side),
        int(se.smooth),
        int(se.axis_neuron),
        lower,
        upper,
        table_max,
        stride0,
        stride1,
        float(se.rcut),
        float(se.rcut_smth),
        float(se.env_protection),
        float(se.nnei),
    )
    force, atom_virial, virial = canonical_edge_force_virial(
        edge_gradient,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
        graph.n_node,
        atype.shape[0],
        do_atomic_virial,
    )
    return energy, atom_energy, force, virial, atom_virial
