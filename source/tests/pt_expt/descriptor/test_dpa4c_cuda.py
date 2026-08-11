# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numerical contract of the compressed DPA4C CUDA mega kernel."""

import dataclasses
from collections.abc import (
    Sequence,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    attach_edge_csr,
    graph_from_dense_quartet,
)
from deepmd.pt_expt.kernels.cuda.dpa4c.graph_compress import (
    _cpu_descriptor,
    _cpu_forward,
    _table_lookup,
    build_compression_artifacts,
    build_radial_table,
    descriptor_profile,
    dpa4c_graph_compress_energy_force,
    ensure_registered,
    mega_eligible,
    op_available,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.descriptor.dpa4c import (
    DescrptDPA4C,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="CUDA and the compiled DPA4C operator are required",
)


def _build_descriptor(
    channels: int,
    lmax: int = 2,
    radial_modes: int = 0,
) -> DescrptDPA4C:
    return (
        DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=channels,
            lmax=lmax,
            n_radial=8,
            radial_modes=radial_modes,
            precision="float32",
            seed=17,
        )
        .cuda()
        .eval()
    )


def _build_graph(
    descriptor: DescrptDPA4C,
    canonical: bool,
    node_count: int = 24,
):
    generator = torch.Generator(device="cuda").manual_seed(23)
    coordinate = torch.rand(
        1,
        node_count,
        3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    coordinate = coordinate * 5.0
    atype = torch.arange(node_count, device="cuda").reshape(1, -1) % 2
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coordinate,
        atype,
        descriptor.rcut,
        [48],
        mixed_types=True,
        box=None,
    )
    graph, flat_type = graph_from_dense_quartet(
        coord_ext,
        atype_ext,
        nlist,
        mapping,
    )
    graph = attach_edge_csr(
        graph,
        flat_type.shape[0],
        canonicalize=canonical,
    )
    return graph, flat_type


def _arguments(
    descriptor: DescrptDPA4C,
    graph,
    atype: torch.Tensor,
):
    ensure_registered()
    artifacts = build_compression_artifacts(descriptor)
    return (
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        atype,
        artifacts["data"],
        artifacts["pair_film"],
        artifacts["pair_mixing"],
        artifacts["type_embedding"],
        artifacts["readout_matrices"],
        artifacts["coupling_meta"],
        artifacts["coupling_entry"],
        artifacts["coupling_value"],
        artifacts["output_mean"],
        artifacts["output_inv_std"],
        artifacts["spin_type"][:0],
        artifacts["spin_pair"],
        artifacts["spin_type"],
        bool(graph.destination_sorted),
        int(descriptor.lmax),
        *(float(value) for value in artifacts["info"]),
    )


def _spin_free(arguments: tuple) -> tuple:
    """Drop the native spin block for the spin-free CPU reference."""
    return (*arguments[:15], *arguments[18:])


def _with_spin(arguments: tuple, spin: torch.Tensor) -> tuple:
    """Place per-node magnetic moments in the operator's spin slot."""
    return (*arguments[:15], spin, *arguments[16:])


def _assert_dispatched(actual: torch.Tensor, portable: torch.Tensor) -> None:
    """Assert that a descriptor came from the compiled operator.

    The compressed path evaluates the radial embedding from a table, which
    never reproduces the portable evaluation bit for bit. A run that fell back
    to the portable code -- because the compression gate closed, or because
    ``DP_CUDA_INFER`` left the operator disabled -- reproduces it exactly, and
    would otherwise be compared with itself and pass every tolerance.
    """
    assert not torch.equal(actual, portable), (
        "the compressed descriptor is bitwise identical to the portable one, "
        "so the compiled operator did not run"
    )


@_GPU
@pytest.mark.parametrize("channels", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("canonical", [False, True])
def test_forward_backward_parity(channels: int, canonical: bool) -> None:
    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical)
    arguments = _arguments(descriptor, graph, atype)
    ensure_registered()

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    output, state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *arguments,
    )
    assert state.shape == (
        atype.shape[0],
        descriptor_profile(channels, descriptor.lmax).state_width,
    )
    cotangent = torch.linspace(
        -0.7,
        1.3,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    (gradient,) = torch.autograd.grad((output * cotangent).sum(), edge_vec)

    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference = _cpu_descriptor(reference_edge, *_spin_free(arguments))
    (reference_gradient,) = torch.autograd.grad(
        (reference * cotangent).sum(),
        reference_edge,
    )
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        gradient,
        reference_gradient,
        atol=8e-6 if channels >= 64 else 3e-6,
        rtol=1e-4 if channels >= 64 else 3e-5,
    )


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_backward_tail_node_groups(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=True, node_count=23)
    arguments = _arguments(descriptor, graph, atype)
    # The cotangent fixes which element sits closest to the tolerance, so
    # drawing it from the unseeded global stream would make the outcome vary
    # between processes.
    cotangent = torch.randn(
        atype.shape[0],
        descriptor.get_dim_out(),
        dtype=torch.float32,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(37),
    )

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    output, _state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *arguments,
    )
    (gradient,) = torch.autograd.grad((output * cotangent).sum(), edge_vec)

    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference = _cpu_descriptor(reference_edge, *_spin_free(arguments))
    (reference_gradient,) = torch.autograd.grad(
        (reference * cotangent).sum(),
        reference_edge,
    )
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        gradient,
        reference_gradient,
        atol=8e-6 if channels >= 64 else 3e-6,
        rtol=1e-4 if channels >= 64 else 3e-5,
    )


@_GPU
@pytest.mark.parametrize("channels", [64, 128])
def test_wide_backward_is_deterministic(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=True)
    arguments = _arguments(descriptor, graph, atype)
    output, state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *arguments,
    )
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        first = torch.ops.deepmd.dpa4c_graph_compress_backward(
            torch.ones_like(output),
            state,
            graph.edge_vec,
            *arguments,
        )
        second = torch.ops.deepmd.dpa4c_graph_compress_backward(
            torch.ones_like(output),
            state,
            graph.edge_vec,
            *arguments,
        )
    finally:
        torch.use_deterministic_algorithms(previous)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@_GPU
@pytest.mark.parametrize("channels", [8, 16, 32, 64, 128])
def test_compressed_matches_uncompressed_descriptor(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = _arguments(descriptor, graph, atype)
    compressed, _state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *arguments,
    )
    reference, _ = descriptor.call_graph(
        graph,
        atype,
        type_embedding=descriptor.type_embedding.call(),
    )
    torch.testing.assert_close(compressed, reference, atol=3e-5, rtol=3e-5)

    cotangent = torch.linspace(
        -0.7,
        1.3,
        compressed.numel(),
        dtype=compressed.dtype,
        device=compressed.device,
    ).reshape_as(compressed)
    compressed_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    compressed_value, _state = torch.ops.deepmd.dpa4c_graph_compress(
        compressed_edge,
        *arguments,
    )
    (compressed_gradient,) = torch.autograd.grad(
        (compressed_value * cotangent).sum(),
        compressed_edge,
    )
    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference_graph = dataclasses.replace(graph, edge_vec=reference_edge)
    reference_value, _ = descriptor.call_graph(
        reference_graph,
        atype,
        type_embedding=descriptor.type_embedding.call(),
    )
    (reference_gradient,) = torch.autograd.grad(
        (reference_value * cotangent).sum(),
        reference_edge,
    )
    torch.testing.assert_close(
        compressed_gradient,
        reference_gradient,
        atol=1e-4,
        rtol=5e-4,
    )


@_GPU
def test_output_calibration_matches_uncompressed_descriptor() -> None:
    descriptor = _build_descriptor(8)
    output_width = descriptor.get_dim_out()
    mean = torch.linspace(
        -0.4,
        0.6,
        output_width,
        dtype=torch.float32,
        device="cuda",
    )
    stddev = torch.linspace(
        0.7,
        1.9,
        output_width,
        dtype=torch.float32,
        device="cuda",
    )
    descriptor.set_stat_mean_and_stddev(mean, stddev)
    graph, atype = _build_graph(descriptor, canonical=False)
    reference, _ = descriptor.call_graph(graph, atype)
    actual, _state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *_arguments(descriptor, graph, atype),
    )
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_descriptor_compression_routing_and_serialization(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=False)
    reference, _ = descriptor.call_graph(graph, atype)
    descriptor.enable_compression(min_nbor_dist=0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    actual, _ = descriptor.call_graph(graph, atype)
    restored = DescrptDPA4C.deserialize(descriptor.serialize()).cuda().eval()
    restored_output, _ = restored.call_graph(graph, atype)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(restored_output, actual)


@_GPU
def test_compression_uses_immutable_type_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    descriptor.enable_compression(min_nbor_dist=0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    reference, _ = descriptor.call_graph(graph, atype)
    alternative = descriptor.type_embedding.call().detach().clone() + 0.25
    actual, _ = descriptor.call_graph(
        graph,
        atype,
        type_embedding=alternative,
    )
    torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)

    monkeypatch.setenv("DP_CUDA_INFER", "0")
    portable, _ = descriptor.call_graph(
        graph,
        atype,
        type_embedding=alternative,
    )
    assert not torch.allclose(portable, reference)


@_GPU
def test_post_compression_statistics_update_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    descriptor.enable_compression(min_nbor_dist=0.5)
    output_width = descriptor.get_dim_out()
    mean = torch.linspace(-0.2, 0.3, output_width, device="cuda")
    stddev = torch.linspace(0.8, 1.6, output_width, device="cuda")
    descriptor.set_stat_mean_and_stddev(mean, stddev)

    monkeypatch.setenv("DP_CUDA_INFER", "0")
    reference, _ = descriptor.call_graph(graph, atype)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    actual, _ = descriptor.call_graph(graph, atype)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)


def _build_charge_descriptor(
    channels: int = 8,
    radial_modes: int = 0,
    default_chg_spin: Sequence[float] | None = (2.0, 3.0),
) -> DescrptDPA4C:
    """Return a charge-conditioned descriptor with an active condition head.

    The condition output projection is zero initialized, so it is replaced by
    deterministic weights: a parity test against an inert condition would
    pass for the wrong reason.
    """
    descriptor = (
        DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=channels,
            lmax=2,
            n_radial=8,
            radial_modes=radial_modes,
            precision="float32",
            seed=17,
            add_chg_spin_ebd=True,
            default_chg_spin=(
                None if default_chg_spin is None else list(default_chg_spin)
            ),
        )
        .cuda()
        .eval()
    )
    head = descriptor.charge_spin_embedding.network.layers[-1]
    generator = torch.Generator(device="cuda").manual_seed(11)
    with torch.no_grad():
        head.w.copy_(
            torch.randn(
                head.w.shape,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            * 0.5
        )
    return descriptor


@_GPU
@pytest.mark.parametrize("channels", [8, 32, 128])
@pytest.mark.parametrize("radial_modes", [0, 4])
def test_charge_condition_folds_into_the_frozen_tables(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    radial_modes: int,
) -> None:
    """Compression must reproduce the conditioned portable descriptor.

    The frame condition reaches only the finite type and ordered pair tables,
    so folding it there leaves every artifact shape and the compiled kernel
    untouched. Parity against the portable path is what establishes that the
    fold is the same function the edge axis evaluates.
    """
    descriptor = _build_charge_descriptor(channels, radial_modes)
    graph, atype = _build_graph(descriptor, canonical=False)
    reference, _ = descriptor.call_graph(
        graph,
        atype,
        charge_spin=torch.tensor([[2.0, 3.0]], device="cuda"),
    )
    descriptor.enable_compression(min_nbor_dist=0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    actual, _ = descriptor.call_graph(graph, atype)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)


@_GPU
def test_a_baked_charge_state_differs_from_a_neutral_one() -> None:
    """The baked state must actually reach the frozen tables.

    Two snapshots of the same weights that differ only in the charge state
    they were compressed against have to disagree; otherwise the fold is
    writing an unconditioned table and the parity test above would hold
    vacuously.
    """
    graph, atype = _build_graph(_build_charge_descriptor(), canonical=False)
    outputs = []
    for state in ([0.0, 1.0], [2.0, 3.0]):
        descriptor = _build_charge_descriptor(default_chg_spin=state)
        descriptor.enable_compression(min_nbor_dist=0.5)
        outputs.append(
            torch.ops.deepmd.dpa4c_graph_compress(
                graph.edge_vec,
                *_arguments(descriptor, graph, atype),
            )[0]
        )
    assert not torch.allclose(outputs[0], outputs[1])


@_GPU
def test_a_compressed_snapshot_declares_no_runtime_condition() -> None:
    # The compact canonical lower carries no conditioning slot, so a baked
    # snapshot has to report that it consumes none.
    descriptor = _build_charge_descriptor()
    assert descriptor.get_dim_chg_spin() == 2
    descriptor.enable_compression(min_nbor_dist=0.5)
    assert descriptor.get_dim_chg_spin() == 0
    assert descriptor.supports_charge_spin()


@_GPU
def test_compression_requires_a_baked_charge_state() -> None:
    # Without a default there is no state to fold, and a snapshot that
    # silently evaluated the unconditioned tables would be a different model.
    descriptor = _build_charge_descriptor(default_chg_spin=None)
    with pytest.raises(ValueError, match="`default_chg_spin`"):
        descriptor.enable_compression(min_nbor_dist=0.5)


@_GPU
def test_compressed_descriptor_cannot_reenter_training() -> None:
    descriptor = _build_descriptor(8)
    descriptor.enable_compression(min_nbor_dist=0.5)
    with pytest.raises(RuntimeError, match="immutable inference snapshot"):
        descriptor.train()


@_GPU
def test_compression_rejects_float64_descriptor() -> None:
    descriptor = (
        DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=8,
            lmax=2,
            n_radial=8,
            precision="float64",
            seed=17,
        )
        .cuda()
        .eval()
    )
    with pytest.raises(ValueError, match="requires descriptor precision `float32`"):
        descriptor.enable_compression(min_nbor_dist=0.5)


@_GPU
@pytest.mark.parametrize("channels", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("lmax", [2, 3, 4])
@pytest.mark.parametrize("radial_modes", [0, 2, 4, 8])
@pytest.mark.parametrize("canonical", [False, True])
def test_supported_surface_parity(
    channels: int,
    lmax: int,
    radial_modes: int,
    canonical: bool,
) -> None:
    """Cover the complete compiled surface against the portable equations.

    Each scalar width owns a distinct lane mapping and each angular degree a
    distinct instantiation, so the cross product is the contract the operator
    advertises rather than a sample of it.
    """
    descriptor = _build_descriptor(channels, lmax, radial_modes)
    graph, atype = _build_graph(descriptor, canonical)
    arguments = _arguments(descriptor, graph, atype)
    ensure_registered()

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    output, _state = torch.ops.deepmd.dpa4c_graph_compress(edge_vec, *arguments)
    reference, _ = descriptor.call_graph(graph, atype)
    torch.testing.assert_close(output, reference, atol=3e-5, rtol=3e-5)

    cotangent = torch.linspace(
        -0.7,
        1.3,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    (gradient,) = torch.autograd.grad((output * cotangent).sum(), edge_vec)
    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference_value = _cpu_descriptor(reference_edge, *_spin_free(arguments))
    (reference_gradient,) = torch.autograd.grad(
        (reference_value * cotangent).sum(),
        reference_edge,
    )
    torch.testing.assert_close(gradient, reference_gradient, atol=8e-6, rtol=1e-4)


@_GPU
@pytest.mark.parametrize(
    ("index", "shape", "dtype", "message"),
    [
        (6, (9, 4, 2), torch.float32, "PairFiLM"),
        (9, (8, 5, 5), torch.float32, "invalid readout matrix"),
        (10, (1, 8), torch.int32, "degree triples"),
    ],
)
def test_operator_rejects_inconsistent_artifacts(
    index: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    message: str,
) -> None:
    """Device-side shape assumptions are enforced at the operator boundary."""
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = list(_arguments(descriptor, graph, atype))
    arguments[index] = torch.zeros(shape, dtype=dtype, device="cuda")
    with pytest.raises(RuntimeError, match=message):
        torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)


@_GPU
def test_operator_rejects_unsupported_mode_rank() -> None:
    """An unsupported mode rank would overflow the shared mode cache."""
    descriptor = _build_descriptor(8, radial_modes=2)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = list(_arguments(descriptor, graph, atype))
    table = arguments[5]
    channels = descriptor.channels
    # Three modes keep the table and cache shapes mutually consistent while
    # leaving the rank outside the compiled set.
    arguments[5] = torch.zeros(
        table.shape[0],
        6 * (channels + 3),
        dtype=torch.float32,
        device="cuda",
    )
    arguments[7] = torch.zeros(
        arguments[7].shape[0],
        channels,
        3,
        dtype=torch.float32,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="radial_modes must be"):
        torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)


@_GPU
def test_generic_compression_restores_input_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    graph64 = dataclasses.replace(graph, edge_vec=graph.edge_vec.to(torch.float64))
    descriptor.enable_compression(min_nbor_dist=0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "0")
    reference, _ = descriptor.call_graph(graph64, atype)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    actual, _ = descriptor.call_graph(graph64, atype)
    assert actual.dtype == graph64.edge_vec.dtype
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)


@_GPU
def test_radial_table_accuracy() -> None:
    descriptor = _build_descriptor(8)
    table, info = build_radial_table(descriptor)
    radius = torch.linspace(0.0, descriptor.rcut, 2001, device="cuda")
    reference = descriptor.radial_embedding(descriptor.radial_basis(radius[:, None]))
    actual = _table_lookup(
        table,
        radius,
        float(info[0]),
        float(info[1]),
        descriptor.channels,
    )
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)


@_GPU
def test_radial_table_is_c2_at_internal_knots() -> None:
    descriptor = _build_descriptor(8)
    table, info = build_radial_table(descriptor)
    stride = float(info[0])
    knot = 617 * stride
    cotangent = torch.linspace(
        -0.7,
        1.3,
        descriptor.channels,
        device="cuda",
    )

    def derivatives(radius_value: float) -> list[torch.Tensor]:
        radius = torch.tensor(
            [radius_value],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        value = (
            _table_lookup(
                table,
                radius,
                stride,
                float(info[1]),
                descriptor.channels,
            )[0]
            * cotangent
        ).sum()
        first = torch.autograd.grad(value, radius, create_graph=True)[0]
        second = torch.autograd.grad(first, radius, create_graph=True)[0]
        return [value, first[0], second[0]]

    left = derivatives(knot - 1e-6)
    right = derivatives(knot + 1e-6)
    tolerances = (3e-6, 1e-4, 2e-3)
    for lhs, rhs, atol in zip(left, right, tolerances, strict=True):
        torch.testing.assert_close(lhs, rhs, atol=atol, rtol=0.0)


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_compressed_cutoff_matches_removed_topology(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    edge_index = torch.tensor(
        [[1, 0], [0, 1]],
        dtype=torch.long,
        device="cuda",
    )
    radius = torch.tensor(descriptor.rcut, device="cuda")
    zero = torch.zeros_like(radius)
    edge_vec = torch.stack(
        [
            torch.stack([radius, zero, zero]),
            torch.stack([-radius, zero, zero]),
        ]
    ).requires_grad_(True)
    graph = NeighborGraph(
        n_node=torch.tensor([2], dtype=torch.long, device="cuda"),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    graph = attach_edge_csr(graph, 2, canonicalize=False)
    atype = torch.zeros(2, dtype=torch.long, device="cuda")
    arguments = _arguments(descriptor, graph, atype)
    retained, _state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *arguments,
    )
    (gradient,) = torch.autograd.grad(retained.sum(), edge_vec)

    removed_graph = dataclasses.replace(
        graph,
        edge_mask=torch.zeros_like(graph.edge_mask),
    )
    removed_edge = edge_vec.detach().clone().requires_grad_(True)
    removed, _removed_state = torch.ops.deepmd.dpa4c_graph_compress(
        removed_edge,
        *_arguments(descriptor, removed_graph, atype),
    )
    (removed_gradient,) = torch.autograd.grad(removed.sum(), removed_edge)
    torch.testing.assert_close(retained, removed, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        gradient,
        torch.zeros_like(gradient),
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        removed_gradient,
        torch.zeros_like(removed_gradient),
        atol=0.0,
        rtol=0.0,
    )


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_in_row_mask_matches_removed_edge(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    edge_index = torch.tensor(
        [[1, 0], [0, 1]],
        dtype=torch.long,
        device="cuda",
    )
    edge_vec = torch.tensor(
        [[1.0, 0.2, -0.1], [-0.7, 0.3, 0.4]],
        dtype=torch.float32,
        device="cuda",
    )
    graph = NeighborGraph(
        n_node=torch.tensor([2], dtype=torch.long, device="cuda"),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=torch.ones(2, dtype=torch.bool, device="cuda"),
    )
    graph = attach_edge_csr(graph, 2, canonicalize=False)
    atype = torch.zeros(2, dtype=torch.long, device="cuda")

    full, _full_state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *_arguments(descriptor, graph, atype),
    )
    masked_graph = dataclasses.replace(
        graph,
        edge_mask=torch.tensor([True, False], dtype=torch.bool, device="cuda"),
    )
    masked_edge = edge_vec.detach().clone().requires_grad_(True)
    masked, _masked_state = torch.ops.deepmd.dpa4c_graph_compress(
        masked_edge,
        *_arguments(descriptor, masked_graph, atype),
    )
    cotangent = torch.linspace(
        -0.7,
        1.3,
        masked.numel(),
        dtype=masked.dtype,
        device=masked.device,
    ).reshape_as(masked)
    (masked_gradient,) = torch.autograd.grad(
        (masked * cotangent).sum(),
        masked_edge,
    )

    removed_edge = edge_vec[:1].detach().clone().requires_grad_(True)
    removed_graph = NeighborGraph(
        n_node=graph.n_node,
        edge_index=edge_index[:, :1].contiguous(),
        edge_vec=removed_edge,
        edge_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
    )
    removed_graph = attach_edge_csr(removed_graph, 2, canonicalize=False)
    removed, _removed_state = torch.ops.deepmd.dpa4c_graph_compress(
        removed_edge,
        *_arguments(descriptor, removed_graph, atype),
    )
    (removed_gradient,) = torch.autograd.grad(
        (removed * cotangent).sum(),
        removed_edge,
    )

    assert not torch.allclose(full, masked)
    torch.testing.assert_close(masked, removed, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        masked_gradient[:1],
        removed_gradient,
        atol=3e-6,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        masked_gradient[1],
        torch.zeros_like(masked_gradient[1]),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(masked_gradient[0]).item() > 0


@_GPU
@pytest.mark.parametrize("channels", [8, 32])
def test_padding_type_edge_has_zero_gradient(channels: int) -> None:
    descriptor = _build_descriptor(channels)
    edge_vec = torch.tensor(
        [[1.0, 0.2, -0.1]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    graph = NeighborGraph(
        n_node=torch.tensor([2], dtype=torch.long, device="cuda"),
        edge_index=torch.tensor([[1], [0]], dtype=torch.long, device="cuda"),
        edge_vec=edge_vec,
        edge_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
    )
    graph = attach_edge_csr(graph, 2, canonicalize=False)
    atype = torch.tensor([0, descriptor.ntypes], dtype=torch.long, device="cuda")

    output, _state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *_arguments(descriptor, graph, atype),
    )
    (gradient,) = torch.autograd.grad(output.sum(), edge_vec)
    torch.testing.assert_close(
        gradient,
        torch.zeros_like(gradient),
        atol=0.0,
        rtol=0.0,
    )


@_GPU
def test_int32_edge_indices() -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    graph32 = dataclasses.replace(
        graph,
        edge_index=graph.edge_index.to(torch.int32),
        destination_order=graph.destination_order.to(torch.int32),
    )
    output64, _state64 = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *_arguments(descriptor, graph, atype),
    )
    output32, _state32 = torch.ops.deepmd.dpa4c_graph_compress(
        graph32.edge_vec,
        *_arguments(descriptor, graph32, atype),
    )
    torch.testing.assert_close(output32, output64)


@_GPU
@pytest.mark.parametrize("channels", [8, 64, 128])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.uint32])
def test_compact_canonical_parity(
    channels: int,
    index_dtype: torch.dtype,
) -> None:
    from deepmd.pt_expt.kernels.cuda.dpa4c.canonical import (
        ensure_registered as ensure_canonical_registered,
    )

    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=True)
    arguments = _arguments(descriptor, graph, atype)
    ensure_canonical_registered()
    generic_output, generic_state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *arguments,
    )
    canonical_arguments = (
        graph.edge_index[0].to(index_dtype),
        graph.destination_row_ptr,
        atype,
        *arguments[5:18],
        *arguments[19:],
    )
    compact_output, compact_state = torch.ops.deepmd.dpa4c_canonical_compress(
        graph.edge_vec,
        *canonical_arguments,
    )
    torch.testing.assert_close(compact_output, generic_output)
    torch.testing.assert_close(compact_state, generic_state)

    cotangent = torch.randn_like(generic_output)
    generic_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        cotangent,
        generic_state,
        graph.edge_vec,
        *arguments,
    )
    compact_gradient = torch.ops.deepmd.dpa4c_canonical_compress_backward(
        cotangent,
        compact_state,
        graph.edge_vec,
        *canonical_arguments,
    )
    torch.testing.assert_close(
        compact_gradient,
        generic_gradient,
        atol=2e-6,
        rtol=2e-6,
    )


@_GPU
@pytest.mark.parametrize("channels", [8, 128])
def test_compact_inplace_backward_reuses_state(channels: int) -> None:
    from deepmd.pt_expt.kernels.cuda.dpa4c.canonical import (
        ensure_registered as ensure_canonical_registered,
    )

    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=True, node_count=23)
    arguments = _arguments(descriptor, graph, atype)
    canonical_arguments = (
        graph.edge_index[0].to(torch.uint32),
        graph.destination_row_ptr,
        atype,
        *arguments[5:18],
        *arguments[19:],
    )
    ensure_canonical_registered()
    output, state = torch.ops.deepmd.dpa4c_canonical_compress(
        graph.edge_vec,
        *canonical_arguments,
    )
    cotangent = torch.randn_like(output)
    reference = torch.ops.deepmd.dpa4c_canonical_compress_backward(
        cotangent,
        state,
        graph.edge_vec,
        *canonical_arguments,
    )
    inplace_state = state.clone()
    actual = torch.ops.deepmd.dpa4c_canonical_compress_backward_inplace(
        cotangent,
        inplace_state,
        graph.edge_vec,
        *canonical_arguments,
    )
    torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
    assert not torch.equal(inplace_state, state)


@_GPU
@pytest.mark.parametrize("channels", [8, 128])
@pytest.mark.parametrize("fitting_width", [32, 64, 128, 192, 256])
@pytest.mark.parametrize("fitting_depth", [1, 2, 3])
def test_fused_energy_force_parity(
    channels: int,
    fitting_width: int,
    fitting_depth: int,
) -> None:
    from deepmd.pt_expt.kernels.cuda.edge_force_virial import (
        edge_force_virial,
    )
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )

    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=False)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    fitting = (
        EnergyFittingNet(
            ntypes=2,
            dim_descrpt=descriptor.get_dim_out(),
            neuron=[fitting_width] * fitting_depth,
            resnet_dt=False,
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=29,
        )
        .cuda()
        .eval()
    )
    fitting.bias_atom_e = torch.tensor(
        [[0.3], [-0.2]],
        dtype=torch.float64,
        device="cuda",
    )
    ownership = torch.ones(atype.shape[0], dtype=torch.bool, device="cuda")
    fused = dpa4c_graph_compress_energy_force(
        descriptor,
        fitting,
        graph,
        atype,
        ownership,
        fitting.bias_atom_e[:, 0],
        atype.shape[0],
        True,
    )

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    arguments = _arguments(descriptor, graph, atype)
    node_descriptor, _state = torch.ops.deepmd.dpa4c_graph_compress(
        edge_vec,
        *arguments,
    )
    atom_energy = fitting.call_graph(node_descriptor, atype)[fitting.var_name]
    (edge_gradient,) = torch.autograd.grad(atom_energy.sum(), edge_vec)
    force, atom_virial, virial, _ = edge_force_virial(
        edge_gradient,
        edge_vec.detach(),
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
        graph.n_node,
        edge_vec.new_empty(0),
        atype.shape[0],
        True,
    )
    torch.testing.assert_close(
        fused[1],
        atom_energy.to(torch.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(fused[2], force, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused[3], virial, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused[4], atom_virial, atol=1e-6, rtol=1e-5)


class _ExportModule(torch.nn.Module):
    def forward(self, edge_vec: torch.Tensor, *arguments):
        return torch.ops.deepmd.dpa4c_graph_compress(edge_vec, *arguments)


@_GPU
def test_torch_export() -> None:
    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = _arguments(descriptor, graph, atype)
    module = _ExportModule().cuda().eval()
    exported = torch.export.export(
        module,
        (graph.edge_vec, *arguments),
        strict=False,
    )
    actual = exported.module()(graph.edge_vec, *arguments)
    reference = module(graph.edge_vec, *arguments)
    torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_energy_force_refuses_ineligible_fitting() -> None:
    """The fused path must refuse a network the operator cannot represent.

    A layer timestep has no representation in the fused fitting operator,
    which would otherwise evaluate the network without it and return an
    energy that no reference path produces.
    """
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )

    descriptor = _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=False)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    fitting = (
        EnergyFittingNet(
            ntypes=2,
            dim_descrpt=descriptor.get_dim_out(),
            neuron=[64, 64],
            resnet_dt=True,
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=29,
        )
        .cuda()
        .eval()
    )
    ownership = torch.ones(atype.shape[0], dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="cannot reproduce this network"):
        dpa4c_graph_compress_energy_force(
            descriptor,
            fitting,
            graph,
            atype,
            ownership,
            fitting.bias_atom_e[:, 0],
            atype.shape[0],
            True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels", [8, 64])
def test_compact_canonical_tiling_is_equivalent(
    channels: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node tiling must not change energy, force or virial.

    Every node tile owns a contiguous span of the destination-sorted edge
    axis, so the runs partition the work rather than splitting any reduction.
    """
    from deepmd.pt_expt.kernels.cuda.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    descriptor = _build_descriptor(channels)
    neighbor_graph, atype = _build_graph(descriptor, canonical=True, node_count=97)
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    descriptor._set_compression(build_compression_artifacts(descriptor))
    fitting = (
        EnergyFittingNet(
            ntypes=2,
            dim_descrpt=descriptor.get_dim_out(),
            neuron=[64, 64],
            resnet_dt=False,
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=29,
        )
        .cuda()
        .eval()
    )
    ownership = torch.ones(atype.shape[0], dtype=torch.bool, device="cuda")

    def run() -> tuple[torch.Tensor, ...]:
        return dpa4c_canonical_compress_energy_force(
            descriptor,
            fitting,
            graph,
            atype,
            ownership,
            fitting.bias_atom_e[:, 0],
            True,
        )

    monkeypatch.setenv("DP_NODE_TILE", "0")
    reference = run()
    for tile in ("7", "32", "96"):
        monkeypatch.setenv("DP_NODE_TILE", tile)
        for actual, expected in zip(run(), reference, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def _build_spin_descriptor(
    channels: int,
    lmax: int = 2,
    radial_modes: int = 0,
    device: str = "cuda",
) -> DescrptDPA4C:
    """Return a spin-conditioned descriptor with a non-unit reference moment.

    A reference magnitude other than one makes the conditioning factor visible
    in the magnetic gradient, so a missing chain factor shows up as a constant
    ratio rather than cancelling. The branch gate is opened, since a fresh
    descriptor starts spin-free by design and these tests are about the branch
    behind the gate; the gate itself is covered in ``test_dpa4c.py``.
    """
    descriptor = (
        DescrptDPA4C(
            rcut=3.0,
            ntypes=2,
            channels=channels,
            lmax=lmax,
            n_radial=8,
            radial_modes=radial_modes,
            precision="float32",
            seed=17,
            use_spin=[True, False],
        )
        .to(device)
        .eval()
    )
    descriptor.spin.set_spin_reference(np.array([1.7, 1.0, 1.0]))
    with torch.no_grad():
        descriptor.spin.spin_gate.fill_(1.0)
    return descriptor


def _empty_graph(device: str) -> tuple[NeighborGraph, torch.Tensor]:
    """Return a native graph whose node and edge axes are both empty."""
    n_node = torch.zeros(1, dtype=torch.int64, device=device)
    edge_index = torch.empty(2, 0, dtype=torch.int64, device=device)
    edge_vec = torch.empty(0, 3, dtype=torch.float32, device=device)
    edge_order = torch.empty(0, dtype=torch.int64, device=device)
    row_pointer = torch.zeros(1, dtype=torch.int64, device=device)
    return (
        NeighborGraph(
            n_node=n_node,
            n_local=n_node,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=torch.empty(0, dtype=torch.bool, device=device),
            destination_order=edge_order,
            destination_row_ptr=row_pointer,
            source_order=edge_order,
            source_row_ptr=row_pointer,
        ),
        torch.empty(0, dtype=torch.int64, device=device),
    )


def test_empty_native_spin_cpu_profile_preserves_spin_contract() -> None:
    """The CPU reference keeps the native-spin state width at zero nodes."""
    descriptor = _build_spin_descriptor(8, device="cpu")
    graph, atype = _empty_graph("cpu")
    arguments = _with_spin(
        _arguments(descriptor, graph, atype),
        graph.edge_vec,
    )

    output, state = _cpu_forward(graph.edge_vec, *arguments)

    profile = descriptor_profile(8, 2, True)
    assert output.shape == (0, profile.output_width)
    assert state.shape == (0, profile.state_width)


@_GPU
def test_empty_native_spin_cuda_backward_preserves_spin_contract() -> None:
    """The CUDA backward distinguishes present spin from its empty axis."""
    descriptor = _build_spin_descriptor(8)
    graph, atype = _empty_graph("cuda")
    arguments = _with_spin(
        _arguments(descriptor, graph, atype),
        graph.edge_vec,
    )
    output, state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *arguments,
    )

    edge_gradient, spin_gradient, edge_spin_gradient = (
        torch.ops.deepmd.dpa4c_graph_compress_backward(
            torch.empty_like(output),
            state,
            graph.edge_vec,
            *arguments,
        )
    )

    assert edge_gradient.shape == (0, 3)
    assert spin_gradient.shape == (0, 3)
    assert edge_spin_gradient.shape == (0, 3)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes"),
    [(8, 2, 0), (16, 2, 0), (32, 2, 4), (64, 3, 0), (128, 3, 8), (32, 4, 0)],
)
def test_spin_compressed_matches_portable(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    lmax: int,
    radial_modes: int,
) -> None:
    """Descriptor, coordinate gradient and magnetic force all match.

    The portable path is the oracle for the spin families. Probing every
    output column at once keeps the geometric and the spin blocks in the same
    comparison, because the two are coupled through the shared normalizer and
    through the cross Gram. The cotangent stays bounded: weighting columns by
    their index inflates the gradient magnitude at the widest profile until
    fp32 tabulation noise alone exceeds the tolerance.
    """
    descriptor = _build_spin_descriptor(channels, lmax, radial_modes)
    assert mega_eligible(descriptor)
    graph, atype = _build_graph(descriptor, canonical=True)
    generator = torch.Generator(device="cuda").manual_seed(11)
    spin = torch.randn(
        atype.shape[0],
        3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    def run() -> tuple[torch.Tensor, ...]:
        moment = spin.detach().clone().requires_grad_(True)
        edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
        output, _ = descriptor.call_graph(
            dataclasses.replace(graph, edge_vec=edge_vec),
            atype,
            spin=moment,
        )
        cotangent = torch.linspace(
            -0.7,
            1.3,
            output.numel(),
            dtype=output.dtype,
            device=output.device,
        ).reshape_as(output)
        gradients = torch.autograd.grad(
            (output * cotangent).sum(),
            [edge_vec, moment],
        )
        return (output, *gradients)

    reference = run()
    descriptor.enable_compression(0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    assert descriptor.compress
    actual = run()
    _assert_dispatched(actual[0], reference[0])
    # Tabulation error reaches the gradients through the table derivative, so
    # they carry the wider tolerance the geometric backward already uses.
    tolerances = ((3e-5, 3e-5), (8e-6, 1e-4), (8e-6, 1e-4))
    for (atol, rtol), value, expected in zip(
        tolerances, actual, reference, strict=True
    ):
        torch.testing.assert_close(value, expected, atol=atol, rtol=rtol)


@_GPU
@pytest.mark.parametrize("family", ["neighbour", "onsite"])
def test_spin_magnetic_force_splits_into_onsite_and_neighbour(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    """Each half of the magnetic force is correct on its own.

    The on-site half closes inside the node kernel while the neighbour half is
    emitted per edge and reduced onto source nodes, so they fail
    independently. Silencing one at a time keeps a fault in either from being
    masked by the other's magnitude.
    """
    descriptor = _build_spin_descriptor(16)
    with torch.no_grad():
        if family == "neighbour":
            descriptor.spin.adam_spin_vector_weight.zero_()
            descriptor.spin.adam_spin_quadrupole_weight.zero_()
        else:
            geometric = descriptor.channels * (2 + descriptor.radial_modes)
            for parameter in descriptor.pair_film.parameters():
                if parameter.dim() == 2 and parameter.shape[1] > geometric:
                    parameter[:, geometric:] = 0.0
                elif parameter.dim() == 1 and parameter.shape[0] > geometric:
                    parameter[geometric:] = 0.0
    graph, atype = _build_graph(descriptor, canonical=True)
    generator = torch.Generator(device="cuda").manual_seed(11)
    spin = torch.randn(
        atype.shape[0],
        3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        moment = spin.detach().clone().requires_grad_(True)
        output, _ = descriptor.call_graph(graph, atype, spin=moment)
        cotangent = torch.linspace(
            -0.7,
            1.3,
            output.numel(),
            dtype=output.dtype,
            device=output.device,
        ).reshape_as(output)
        gradient = torch.autograd.grad((output * cotangent).sum(), moment)[0]
        return output, gradient

    reference_output, reference_gradient = run()
    descriptor.enable_compression(0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    output, gradient = run()
    _assert_dispatched(output, reference_output)
    assert torch.count_nonzero(gradient).item() > 0
    torch.testing.assert_close(gradient, reference_gradient, atol=8e-6, rtol=1e-4)


@_GPU
def test_spin_bond_family_couples_the_kernel_to_the_edge_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiled spin path reads the edge direction, and only through it.

    Rotating the neighbourhood while holding the moments fixed leaves every
    spin family except the bond-projected one invariant, so a kernel that
    omitted that family would return an unchanged readout. Rotating the
    moments alongside the geometry restores the invariance, which is what
    separates a genuine bond coupling from a defect in the geometric block.
    """
    descriptor = _build_spin_descriptor(16)
    graph, atype = _build_graph(descriptor, canonical=True)
    generator = torch.Generator(device="cuda").manual_seed(11)
    spin = torch.randn(
        atype.shape[0],
        3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    angle = 0.4
    rotation = torch.tensor(
        [
            [float(np.cos(angle)), -float(np.sin(angle)), 0.0],
            [float(np.sin(angle)), float(np.cos(angle)), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    def run(edge_vec: torch.Tensor, moment: torch.Tensor) -> torch.Tensor:
        output, _ = descriptor.call_graph(
            dataclasses.replace(graph, edge_vec=edge_vec),
            atype,
            spin=moment,
        )
        return output

    portable = run(graph.edge_vec, spin)
    descriptor.enable_compression(0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    upright = run(graph.edge_vec, spin)
    _assert_dispatched(upright, portable)
    rotated = run(graph.edge_vec @ rotation.T, spin)
    assert not torch.allclose(rotated, upright, atol=1e-4)
    covariant = run(graph.edge_vec @ rotation.T, spin @ rotation.T)
    torch.testing.assert_close(covariant, upright, atol=3e-5, rtol=3e-5)


@_GPU
@pytest.mark.parametrize("spin_conditioned", [False, True])
def test_backward_operator_satisfies_its_schema(spin_conditioned: bool) -> None:
    """The backward operator declares three independent results.

    All three are unannotated, so any two of them sharing storage would be an
    alias the schema does not describe, which is undefined under
    functionalization. ``opcheck`` decides that mechanically, including on the
    spin-free path where two of the three are absent and an allocation shared
    between them would otherwise go unnoticed.
    """
    descriptor = _build_spin_descriptor(8) if spin_conditioned else _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=True)
    arguments = _arguments(descriptor, graph, atype)
    if spin_conditioned:
        arguments = _with_spin(
            arguments,
            torch.randn(atype.shape[0], 3, dtype=torch.float32, device="cuda"),
        )
    output, state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *arguments,
    )
    torch.library.opcheck(
        torch.ops.deepmd.dpa4c_graph_compress_backward.default,
        (torch.ones_like(output), state, graph.edge_vec, *arguments),
    )


@_GPU
def test_registered_autograd_refuses_the_magnetic_moment() -> None:
    """A direct operator call cannot silently lose the magnetic force.

    The operator emits that cotangent in two pieces and the per-edge piece is
    reduced onto source nodes through the source CSR, which the schema does
    not carry. The registration therefore cannot close the magnetic force, and
    refusing is the only alternative to reporting a vanishing one.
    """
    descriptor = _build_spin_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=True)
    spin = torch.randn(
        atype.shape[0],
        3,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    output, _state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec,
        *_with_spin(_arguments(descriptor, graph, atype), spin),
    )
    with pytest.raises(RuntimeError, match="cannot differentiate its magnetic"):
        output.sum().backward()
