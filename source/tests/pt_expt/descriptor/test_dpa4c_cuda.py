# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numerical contract of the compressed DPA4C CUDA mega kernel."""

import dataclasses
import functools
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import types
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    Any,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    attach_edge_csr,
    graph_from_dense_quartet,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt_expt.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
    _portable_inputs,
    _reference_descriptor,
    _reference_forward,
    _table_lookup,
    build_compression_artifacts,
    build_radial_table,
    contact_radius_input,
    descriptor_profile,
    dpa4c_graph_compress_energy_force,
    ensure_registered,
    mega_eligible,
    op_available,
)

from .test_dpa4c_cpu import (
    _restrict_scalar_rank,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not op_available(),
    reason="CUDA and the compiled DPA4C operator are required",
)


#: Element symbols of the two test types. They fix the covalent radii, whose
#: pairwise sums give the three distinct length scales the window is measured
#: against, so a kernel that gathered the wrong radius would misplace it.
_TYPE_MAP = ["O", "H"]

#: Element symbols of the systems of up to nine types, whose first two are
#: those of `_TYPE_MAP`. The bridging window `_WINDOW` of every pair of the
#: first eight stays inside the 3 Å cutoff of the descriptors; that of argon,
#: the ninth, does not.
_WIDE_TYPE_MAP = ["O", "H", "C", "N", "B", "F", "Ne", "He", "Ar"]

#: A bridging window wide enough that the random test cluster populates its
#: frozen zone, its transition zone and the plain zone beyond it. The bounds
#: are fractions of each pair's own length scale, not distances in Å.
_WINDOW = {"inner_clamp_f_inner": 0.9, "inner_clamp_f_outer": 1.6}

#: Factor on the mode-mixing columns of the output layer of the pair encoder of
#: every test descriptor (see `_strengthen_mode_mixing`).
_MIXING_GAIN = 2000.0


def _strengthen_mode_mixing(descriptor: DescrptDPA4C) -> None:
    """Raise the mode mixing of a fresh descriptor to the scale of its FiLM terms.

    The output layer of a fresh pair encoder gives mixing coefficients of
    order 1e-4, against FiLM scales of order one, so the mixing of a wrong
    slot, mode or source type stays below the tolerance of several
    comparisons, those inside a bridging window among them. Scaling the mixing
    columns ``[2C, 2C + C R)`` of that layer by ``_MIXING_GAIN`` gives the
    coefficients of the profiles of 32 channels and more a mean magnitude of
    0.06 to 0.31 and none above 0.89; those of the profiles of 8 and 16
    channels reach a mean of 0.47 to 0.61, and in part the saturation of their
    ``tanh`` bound, which keeps them below one.
    """
    film = descriptor.pair_film
    mixing = slice(2 * film.channels, (2 + film.radial_modes) * film.channels)
    with torch.no_grad():
        film.network.layers[-1].w[:, mixing] *= _MIXING_GAIN


def _build_descriptor(
    channels: int,
    lmax: int = 2,
    radial_modes: int = 0,
    rcut: float = 3.0,
    type_map: Sequence[str] = tuple(_TYPE_MAP),
    **window: float,
) -> DescrptDPA4C:
    """Return an fp32 descriptor on the GPU whose mode mixing is strengthened."""
    descriptor = (
        DescrptDPA4C(
            rcut=rcut,
            ntypes=len(type_map),
            channels=channels,
            lmax=lmax,
            n_radial=8,
            radial_modes=radial_modes,
            precision="float32",
            seed=17,
            type_map=list(type_map),
            **window,
        )
        .cuda()
        .eval()
    )
    _strengthen_mode_mixing(descriptor)
    return descriptor


def _reduced_distance(
    descriptor: DescrptDPA4C,
    graph: Any,
    atype: torch.Tensor,
) -> torch.Tensor:
    """Return the valid edge distances in units of each pair's length scale."""
    source, destination = graph.edge_index[0], graph.edge_index[1]
    distance = graph.edge_vec.norm(dim=-1, keepdim=True)
    contact = descriptor.pair_contact(atype[destination], atype[source], distance.dtype)
    return (distance / contact)[graph.edge_mask, 0]


def _build_graph(
    descriptor: DescrptDPA4C,
    canonical: bool,
    node_count: int = 24,
    single_type: bool = False,
    side: float = 5.0,
    frames: int = 1,
    types: Sequence[int] = (0, 1),
):
    """Return a graph of ``frames`` random clusters of ``node_count`` atoms.

    The atoms of each frame are uniform in a cube of ``side`` Å, and atom ``i``
    carries type ``types[i % len(types)]``.
    """
    generator = torch.Generator(device="cuda").manual_seed(23)
    coordinate = torch.rand(
        frames,
        node_count,
        3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    coordinate = coordinate * side
    pattern = torch.as_tensor(list(types), device="cuda")
    atype = pattern[torch.arange(node_count, device="cuda") % len(types)].repeat(
        frames, 1
    )
    if single_type:
        atype = torch.ones_like(atype)
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
        artifacts["basis_table"],
        artifacts["basis_matrix"],
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
        contact_radius_input(descriptor),
        bool(graph.destination_sorted),
        int(descriptor.lmax),
        *(float(value) for value in artifacts["info"]),
        descriptor.bridging_f_inner or 0.0,
        descriptor.bridging_f_outer or 0.0,
    )


def _descriptor_backward(
    cotangent: torch.Tensor,
    state: torch.Tensor,
    edge_vec: torch.Tensor,
    *arguments: Any,
) -> tuple[torch.Tensor, ...]:
    """Differentiate the descriptor alone, without an analytical pair potential."""
    no_pair = edge_vec.new_empty(0)
    return torch.ops.deepmd.dpa4c_graph_compress_backward(
        cotangent, state, edge_vec, *arguments, no_pair, no_pair
    )[:3]


def _spin_free(arguments: tuple) -> tuple:
    """Return the positional arguments of the reference of a spin-free descriptor.

    The reference (`_reference_descriptor`) reads the radial table itself, so
    the low-rank table and its expansion are dropped, and takes the native
    spin block by keyword, so the empty spin block is dropped as well.
    """
    portable = _portable_inputs(arguments)
    return (*portable[:15], *portable[18:])


def _with_spin(arguments: tuple, spin: torch.Tensor) -> tuple:
    """Place per-node magnetic moments in the operator's spin slot."""
    return (*arguments[:17], spin, *arguments[18:])


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


def _assert_close_at_scale(
    actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
) -> None:
    """Assert that ``actual`` matches ``expected`` at tolerances stated for unit scale.

    The rounding of an fp32 sum follows the magnitude of its terms, not that of
    its result, and the terms of a component that cancels to a small value can
    be as large as the largest components of the output. The absolute
    tolerance ``atol`` is therefore stated for an output whose largest
    magnitude is one, and scales with the largest magnitude of ``expected``
    beyond it.
    """
    scale = max(expected.abs().max().item(), 1.0)
    torch.testing.assert_close(actual, expected, atol=atol * scale, rtol=rtol)


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
    reference = _reference_descriptor(reference_edge, *_spin_free(arguments))
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
    reference = _reference_descriptor(reference_edge, *_spin_free(arguments))
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
        first = _descriptor_backward(
            torch.ones_like(output),
            state,
            graph.edge_vec,
            *arguments,
        )
        second = _descriptor_backward(
            torch.ones_like(output),
            state,
            graph.edge_vec,
            *arguments,
        )
    finally:
        torch.use_deterministic_algorithms(previous)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@_GPU
@pytest.mark.parametrize("window", [{}, _WINDOW], ids=["plain", "bridged"])
@pytest.mark.parametrize("channels", [8, 16, 32, 64, 128])
def test_compressed_matches_uncompressed_descriptor(
    channels: int,
    window: dict[str, float],
) -> None:
    descriptor = _build_descriptor(channels, **window)
    graph, atype = _build_graph(descriptor, canonical=False)
    if window:
        reduced = _reduced_distance(descriptor, graph, atype)
        inner, outer = window.values()
        assert (reduced < inner).any() and (reduced > outer).any()
        assert ((reduced > inner) & (reduced < outer)).sum() > 8
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
def test_a_snapshot_without_the_low_rank_table_reads_the_direct_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compressed snapshot without the low-rank artifacts reads the radial table.

    A payload that omits ``basis_table`` and ``basis_matrix`` restores both
    as empty on both backends, which selects the direct table: on a
    single-type canonical graph, where the scans would read a low-rank table,
    the restored descriptor evaluates bit for bit as the original with its
    low-rank artifacts cleared, and differs from the original that reads
    them. It writes the empty artifacts back.
    """
    import copy

    from deepmd.dpmodel.descriptor.dpa4c import DescrptDPA4C as DPDescrptDPA4C

    descriptor = _build_descriptor(64)
    _restrict_scalar_rank(descriptor, rank=8)
    descriptor.enable_compression(min_nbor_dist=0.5)
    data = descriptor.serialize()
    variables = data["compress"]["@variables"]
    assert variables["basis_table"].size > 0
    del variables["basis_table"], variables["basis_matrix"]
    restored_dpmodel = DPDescrptDPA4C.deserialize(copy.deepcopy(data))
    assert restored_dpmodel.compress_basis_table.size == 0
    assert restored_dpmodel.compress_basis_matrix.size == 0
    restored = DescrptDPA4C.deserialize(copy.deepcopy(data)).cuda().eval()

    low_rank = _check_canonical_against_reference(descriptor, False, monkeypatch)
    empty = descriptor.compress_basis_table.new_empty(0)
    descriptor._store_compression_buffer("basis_table", empty)
    descriptor._store_compression_buffer("basis_matrix", empty.clone())
    direct = _check_canonical_against_reference(descriptor, False, monkeypatch)
    actual = _check_canonical_against_reference(restored, False, monkeypatch)
    assert torch.equal(actual[2], direct[2])
    assert not torch.equal(low_rank[2], direct[2])
    written = restored.serialize()["compress"]["@variables"]
    assert written["basis_table"].size == 0
    assert written["basis_matrix"].size == 0


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
    pass for the wrong reason. The mode mixing is strengthened as in
    `_build_descriptor`.
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
            type_map=_TYPE_MAP,
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
    _strengthen_mode_mixing(descriptor)
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
            type_map=_TYPE_MAP,
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
@pytest.mark.parametrize("windowed", [False, True])
def test_supported_surface_parity(
    channels: int,
    lmax: int,
    radial_modes: int,
    canonical: bool,
    windowed: bool,
) -> None:
    """Cover the complete compiled surface against the portable equations.

    Each scalar width owns a distinct lane mapping and each angular degree a
    distinct instantiation, so the cross product is the contract the operator
    advertises rather than a sample of it. A bridging window selects the
    bridged specialization of every profile. The tolerances follow the
    magnitude of each output (`_assert_close_at_scale`).
    """
    descriptor = _build_descriptor(
        channels, lmax, radial_modes, **(_WINDOW if windowed else {})
    )
    graph, atype = _build_graph(descriptor, canonical)
    arguments = _arguments(descriptor, graph, atype)
    ensure_registered()

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    output, _state = torch.ops.deepmd.dpa4c_graph_compress(edge_vec, *arguments)
    reference, _ = descriptor.call_graph(graph, atype)
    _assert_close_at_scale(output, reference, atol=3e-5, rtol=3e-5)

    cotangent = torch.linspace(
        -0.7,
        1.3,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    (gradient,) = torch.autograd.grad((output * cotangent).sum(), edge_vec)
    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference_value = _reference_descriptor(reference_edge, *_spin_free(arguments))
    (reference_gradient,) = torch.autograd.grad(
        (reference_value * cotangent).sum(),
        reference_edge,
    )
    # Inside the window the switch is steep, and the float32 reference then
    # carries a rounding error of its own that exceeds the plain tolerance.
    atol, rtol = (1e-4, 5e-4) if windowed else (8e-6, 1e-4)
    _assert_close_at_scale(gradient, reference_gradient, atol=atol, rtol=rtol)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "type_count", "windowed", "low_rank"),
    [
        *(
            (channels, lmax, radial_modes, 2, False, False)
            for channels in (32, 64, 128)
            for lmax in (2, 3)
            for radial_modes in (2, 4)
        ),
        *(
            (channels, 3, 4, type_count, windowed, False)
            for channels in (64, 128)
            for type_count in (3, 4, 5, 8, 9)
            for windowed in (False, True)
            if type_count < 9 or not windowed
        ),
        (64, 3, 4, 2, True, False),
        (128, 3, 4, 2, True, False),
        (64, 4, 4, 3, False, False),
        (64, 4, 4, 6, True, False),
        *(
            (channels, lmax, radial_modes, type_count, windowed, True)
            for channels, lmax, radial_modes in (
                (64, 2, 0),
                (64, 3, 0),
                (64, 3, 4),
                (128, 3, 0),
                (128, 3, 4),
            )
            for type_count in (2, 4, 5, 8, 9)
            for windowed in (False, True)
            if type_count < 9 or not windowed
        ),
    ],
)
@pytest.mark.parametrize("system", ["unsplit", "coscheduled"])
def test_slotted_scans_match_the_portable_equations(
    channels: int,
    lmax: int,
    radial_modes: int,
    type_count: int,
    windowed: bool,
    low_rank: bool,
    system: str,
) -> None:
    """The scans of a system of a few types follow the portable equations.

    A canonical system of two to eight node types slots the mode profiles and
    the basis functions of a low-rank table of its edges by the types of their
    sources, in four slots up to four types and in eight from five, in the edge
    scans of both directions, except the backward of eight slots of a
    descriptor without radial modes, which scans pairwise; a system of nine
    types scans every edge pairwise. Scalar-only channels confined to the span
    of 32 functions make the compression adopt the low-rank table, every
    function of which then carries a share of the channels, and which the
    portable equations do not read. A bridging window selects the bridged
    specialization. The graph carries 64-bit indices, whose rows always take a
    warp each; a co-scheduled system holds four waves of single 24-row blocks
    per multiprocessor, twice as many as the profiles that co-schedule need.
    The atoms of the random cluster keep the density of the 24-atom one, so its
    rows hold the short bonds of a random cluster as well. The tolerances
    follow the magnitude of each output (`_assert_close_at_scale`), which the
    strengthened mode mixing raises well above one.
    """
    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count = {
        "unsplit": 2 * multiprocessors + 17,
        "coscheduled": 4 * 24 * multiprocessors + 17,
    }[system]
    descriptor = _build_descriptor(
        channels,
        lmax,
        radial_modes,
        type_map=_WIDE_TYPE_MAP[:type_count],
        **(_WINDOW if windowed else {}),
    )
    if low_rank:
        _restrict_scalar_rank(descriptor, rank=32)
    graph, atype = _build_graph(
        descriptor,
        canonical=True,
        node_count=node_count,
        side=5.0 * (node_count / 24) ** (1.0 / 3.0),
        types=range(type_count),
    )
    arguments = _arguments(descriptor, graph, atype)
    assert (arguments[6].numel() > 0) is low_rank

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    output, _state = torch.ops.deepmd.dpa4c_graph_compress(edge_vec, *arguments)
    reference, _ = descriptor.call_graph(graph, atype)
    _assert_close_at_scale(output, reference, atol=3e-5, rtol=3e-5)

    cotangent = torch.linspace(
        -0.7,
        1.3,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    (gradient,) = torch.autograd.grad((output * cotangent).sum(), edge_vec)
    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference_value = _reference_descriptor(reference_edge, *_spin_free(arguments))
    (reference_gradient,) = torch.autograd.grad(
        (reference_value * cotangent).sum(),
        reference_edge,
    )
    # Inside the window the switch is steep, and the float32 reference then
    # carries a rounding error of its own that exceeds the plain tolerance.
    atol, rtol = (1e-4, 5e-4) if windowed else (8e-6, 1e-4)
    _assert_close_at_scale(gradient, reference_gradient, atol=atol, rtol=rtol)


@_GPU
@pytest.mark.parametrize("type_count", [3, 6])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("perturbation", ["source", "mode"])
def test_wrong_mode_mixing_fails_the_comparisons(
    type_count: int,
    windowed: bool,
    perturbation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan that applied the wrong mode mixing fails every comparison.

    A slotted scan that misplaced a slot or a mode would mix the mode profiles
    of an edge with the coefficients of another source type or of another mode.
    The operator given the mixing table with the source types or with the modes
    of every pair rotated stands for such a scan, on a system of three types
    that the scans of four slots serve and on one of six that those of eight
    slots serve. Its descriptor and its coordinate gradient fail the
    comparisons of `test_slotted_scans_match_the_portable_equations`, at the
    wider tolerances of a bridging window, and its forces that of
    `_check_canonical_against_reference` without a pair potential, which that
    check also makes for a bridged descriptor: with the strengthened mixing of
    the test descriptors (`_strengthen_mode_mixing`), such an error exceeds the
    tolerance of every comparison.
    """
    descriptor = _build_descriptor(
        64,
        3,
        4,
        type_map=_WIDE_TYPE_MAP[:type_count],
        **(_WINDOW if windowed else {}),
    )
    graph, atype = _build_graph(descriptor, canonical=True, types=range(type_count))
    arguments = _arguments(descriptor, graph, atype)
    mixing = arguments[9]
    pairs = mixing.reshape(type_count + 1, type_count + 1, *mixing.shape[1:])
    wrong = pairs.clone()
    if perturbation == "source":
        wrong[:, :type_count] = pairs[:, :type_count].roll(1, dims=1)
    else:
        wrong = wrong.roll(1, dims=-1)
    wrong = wrong.reshape(mixing.shape).contiguous()
    perturbed = (*arguments[:9], wrong, *arguments[10:])

    edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
    output, _ = torch.ops.deepmd.dpa4c_graph_compress(edge_vec, *perturbed)
    reference, _ = descriptor.call_graph(graph, atype)
    with pytest.raises(AssertionError):
        _assert_close_at_scale(output, reference, atol=3e-5, rtol=3e-5)
    cotangent = torch.linspace(
        -0.7,
        1.3,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    (gradient,) = torch.autograd.grad((output * cotangent).sum(), edge_vec)
    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    reference_value = _reference_descriptor(reference_edge, *_spin_free(arguments))
    (reference_gradient,) = torch.autograd.grad(
        (reference_value * cotangent).sum(),
        reference_edge,
    )
    with pytest.raises(AssertionError):
        _assert_close_at_scale(gradient, reference_gradient, atol=1e-4, rtol=5e-4)

    artifacts = build_compression_artifacts(descriptor)
    check = functools.partial(
        _check_canonical_against_reference,
        descriptor,
        False,
        monkeypatch,
        single_type=False,
        types=range(type_count),
    )
    descriptor._set_compression(artifacts)
    forces = check()[2]
    descriptor._set_compression({**artifacts, "pair_mixing": wrong})
    wrong_forces = check()[2]
    scale = max(forces.abs().max().item(), 1.0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong_forces, forces, atol=2e-5 * scale, rtol=2e-5)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "low_rank"),
    [
        (64, 2, 0, True),
        (64, 3, 4, False),
        (64, 3, 4, True),
        (128, 3, 4, False),
        (128, 3, 4, True),
    ],
)
@pytest.mark.parametrize("system", ["split", "unsplit", "coscheduled"])
@pytest.mark.parametrize(
    ("types", "bridged"),
    [((0, 1), False), ((0, 2, 3, 5, 6, 7), False), ((0, 2, 3, 5, 6, 7), True)],
)
def test_fused_slotted_scans_match_the_reference(
    channels: int,
    lmax: int,
    radial_modes: int,
    low_rank: bool,
    system: str,
    types: tuple[int, ...],
    bridged: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused operator scans systems of a few types slotted, as LAMMPS does.

    The compact canonical graph carries 32-bit indices, whose rows a small
    system shares among several warps: the split system of 40 atoms shares
    them, and the larger sizes follow the multiprocessor count so that the rows
    take a warp each: an unsplit system holds more rows than half a wave of the
    warps of independent rows, of which a multiprocessor keeps at most 32, and
    a co-scheduled system holds four waves of single 24-row blocks per
    multiprocessor. The descriptor has eight types: a system of two takes the
    scans of four slots, and one of six, which are not consecutive, those of
    eight, bridged as well. The CPU reference reads the radial table, so the
    low-rank table of a 32-function basis, which the slotted scans read as the
    single-type scans do, sums other terms: its forces differ from those of the
    same descriptor without the basis in the last bits, which shows that the
    slotted low-rank scans ran.
    """
    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count = {
        "split": 40,
        "unsplit": 16 * multiprocessors + 17,
        "coscheduled": 4 * 24 * multiprocessors + 17,
    }[system]
    descriptor = _build_descriptor(
        channels,
        lmax,
        radial_modes,
        type_map=_WIDE_TYPE_MAP[:8],
        **(_WINDOW if bridged else {}),
    )
    if low_rank:
        _restrict_scalar_rank(descriptor, rank=32)
    artifacts = build_compression_artifacts(descriptor)
    assert (artifacts["basis_table"].numel() > 0) is low_rank
    descriptor._set_compression(artifacts)
    check = functools.partial(
        _check_canonical_against_reference,
        descriptor,
        bridged,
        monkeypatch,
        node_count=node_count,
        side=5.0 * (node_count / 40) ** (1.0 / 3.0),
        single_type=False,
        types=types,
    )
    slotted = check()
    if low_rank:
        empty = artifacts["basis_table"].new_empty(0)
        descriptor._set_compression(
            {**artifacts, "basis_table": empty, "basis_matrix": empty}
        )
        direct = check()
        assert not torch.equal(slotted[2], direct[2])


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "low_rank"),
    [
        (64, 2, 0, False),
        (64, 2, 0, True),
        (64, 3, 4, False),
        (64, 3, 4, True),
        (128, 3, 4, True),
    ],
)
@pytest.mark.parametrize("system", ["split", "unsplit", "coscheduled"])
@pytest.mark.parametrize("slot_types", [2, 6])
def test_systems_of_a_few_types_take_the_slotted_scans(
    channels: int,
    lmax: int,
    radial_modes: int,
    low_rank: bool,
    system: str,
    slot_types: int,
) -> None:
    """A system of two to eight node types takes the slotted scans.

    Both systems hold the same atoms of two types and seven isolated atoms,
    which interact with none: in one the isolated atoms carry the two types of
    the others, or four other types besides them, a system of two or six types,
    and in the other the seven remaining types of the descriptor, which make a
    system of nine types that the slotted scans do not serve. The node counts
    agree, so every launch choice that follows from them does as well: the rows
    of the split system of 47 atoms are shared among the warps of a block,
    those of the larger systems take a warp each, and those of the largest the
    co-scheduled blocks of the profiles that co-schedule. The slotted scans
    apply the pair coefficients of each slot to reduced sums, so their forces
    differ from the pairwise ones in the last bits. A descriptor without radial
    modes and without a low-rank table has nothing to slot and scans pairwise
    either way: its forces agree bit for bit.
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count = {
        "split": 40,
        "unsplit": 16 * multiprocessors + 17,
        "coscheduled": 4 * 24 * multiprocessors + 17,
    }[system]
    descriptor = _build_descriptor(
        channels, lmax, radial_modes, type_map=_WIDE_TYPE_MAP
    )
    if low_rank:
        _restrict_scalar_rank(descriptor, rank=32)
    artifacts = build_compression_artifacts(descriptor)
    assert (artifacts["basis_table"].numel() > 0) is low_rank
    descriptor._set_compression(artifacts)
    neighbor_graph, atype = _build_graph(
        descriptor,
        canonical=True,
        node_count=node_count,
        side=5.0 * (node_count / 40) ** (1.0 / 3.0),
    )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    # The isolated atoms close both row-pointer arrays with empty rows.
    isolated = 7
    extended = dataclasses.replace(
        graph,
        n_node=graph.n_node + isolated,
        n_local=graph.n_local + isolated,
        destination_row_ptr=torch.cat(
            [graph.destination_row_ptr] + [graph.destination_row_ptr[-1:]] * isolated
        ),
        source_row_ptr=torch.cat(
            [graph.source_row_ptr] + [graph.source_row_ptr[-1:]] * isolated
        ),
    )
    isolated_types = {2: [0, 1, 0, 1, 0, 1, 0], 6: [2, 3, 4, 5, 0, 1, 0]}[slot_types]
    few_types = torch.cat([atype, atype.new_tensor(isolated_types)])
    all_types = torch.cat([atype, atype.new_tensor([2, 3, 4, 5, 6, 7, 8])])
    fitting = _energy_fitting(descriptor)
    nodes = atype.shape[0]
    slotted = dpa4c_canonical_compress_energy_force(
        descriptor, fitting, extended, few_types, fitting.bias_atom_e[:, 0]
    )
    pairwise = dpa4c_canonical_compress_energy_force(
        descriptor, fitting, extended, all_types, fitting.bias_atom_e[:, 0]
    )
    scale = max(pairwise[2].abs().max().item(), 1.0)
    torch.testing.assert_close(
        slotted[2][:nodes], pairwise[2][:nodes], atol=2e-5 * scale, rtol=2e-5
    )
    assert not slotted[2][nodes:].any()
    assert not pairwise[2][nodes:].any()
    slots = radial_modes > 0 or low_rank
    assert torch.equal(slotted[2][:nodes], pairwise[2][:nodes]) is not slots


#: An edge scan of 32-bit indices in a kernel trace: its direction, its rows
#: per block, its warps per row and its scan follow the index type among its
#: template arguments, the scan as its value in `deepmd_dpa4c::Scan`.
_EDGE_SCAN = re.compile(
    r"(forward|edge_backward)_kernel<[^>]*unsigned int, (\d+), (\d+), "
    r"\(deepmd_dpa4c::Scan\)(\d)>"
)

#: Values of `deepmd_dpa4c::Scan` of the single-type scans, which read the
#: radial or the low-rank table, and of the slotted scans of four and of eight
#: slots.
_SINGLE_TYPE_SCANS = {1, 2}
_SLOTTED_SCANS = {3, 4, 5, 6}


def _edge_scans(trace: Any) -> list[tuple[str, int, int, int, float]]:
    """Return the edge scans of 32-bit indices of a CUDA kernel trace.

    Every entry holds the direction, the rows per block, the warps per row and
    the scan of one kernel (see `_EDGE_SCAN`), and its duration in µs.
    """
    scans = []
    for event in trace.events():
        match = _EDGE_SCAN.search(event.name)
        if match is not None:
            direction, rows, split, scan = match.groups()
            duration = event.time_range.elapsed_us()
            scans.append((direction, int(rows), int(split), int(scan), duration))
    return scans


def _run_in_process(call: str) -> Any:
    """Return the result of ``call`` in a fresh Python process.

    A check that reads from a CUDA kernel trace which kernels ran evaluates in
    a process of its own: late in a long process, CUPTI can stop delivering the
    kernel records, and the trace then shows the launch calls without the
    kernels. The process imports this module and evaluates ``call``, an
    expression of its names whose result passes through JSON.
    """
    # The module imports from its package, so the process finds it by its
    # name from the directory above its top package.
    root = pathlib.Path(__file__).parents[__name__.count(".")]
    code = (
        "import importlib, json, sys\n"
        f"sys.path[:0] = {[str(root), *sys.path]!r}\n"
        f"module = importlib.import_module({__name__!r})\n"
        f"print('result:', json.dumps(module.{call}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    (report,) = [
        line for line in result.stdout.splitlines() if line.startswith("result:")
    ]
    return json.loads(report.removeprefix("result:"))


#: Atom counts of the systems of `_row_split_sweep_in_process`: doubling from
#: one small enough for every edge scan to share its rows among eight warps on
#: any device.
_SWEEP_NODE_COUNTS = tuple(8 << k for k in range(13))


def _row_split_sweep_in_process(
    channels: int,
    lmax: int,
    radial_modes: int,
    rank: int,
    spin: bool,
    types: tuple[int, ...],
) -> list[list]:
    """Compare the fused operator with its CPU reference on systems of doubling size.

    The systems of `_SWEEP_NODE_COUNTS`, at the density of the 40-atom system
    and with atoms that cycle through ``types`` of a descriptor of nine types,
    run until every edge scan takes one warp per row. A positive ``rank``
    confines the scalar-only channels to the span of that many functions, so
    that the compression adopts the low-rank table. Returns, per system, its
    atom count and the sorted direction, scan and warps per row of every edge
    scan that ran (see `_edge_scans`).
    """
    from torch.profiler import (
        ProfilerActivity,
        profile,
    )

    build = _build_spin_descriptor if spin else _build_descriptor
    descriptor = build(channels, lmax, radial_modes, type_map=_WIDE_TYPE_MAP)
    if rank:
        _restrict_scalar_rank(descriptor, rank=rank)
    artifacts = build_compression_artifacts(descriptor)
    assert (artifacts["basis_table"].numel() > 0) is bool(rank)
    descriptor._set_compression(artifacts)
    sweep = []
    for node_count in _SWEEP_NODE_COUNTS:
        with (
            pytest.MonkeyPatch.context() as patch,
            profile(activities=[ProfilerActivity.CUDA]) as trace,
        ):
            _check_canonical_against_reference(
                descriptor,
                False,
                patch,
                node_count=node_count,
                side=5.0 * (node_count / 40) ** (1.0 / 3.0),
                single_type=False,
                types=types,
            )
        scans = sorted(
            {
                (direction, scan, split)
                for direction, _, split, scan, _ in _edge_scans(trace)
            }
        )
        assert scans, f"the trace of {node_count} atoms records no edge scan"
        sweep.append([node_count, scans])
        if all(split == 1 for _, _, split in scans):
            return sweep
    raise AssertionError(
        f"an edge scan still shares its rows at {_SWEEP_NODE_COUNTS[-1]} atoms"
    )


_FEW = (0, 1)
_SIX = (0, 2, 3, 5, 6, 8)
_NINE = tuple(range(9))


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "rank", "spin", "types"),
    [
        (64, 2, 0, 32, False, _FEW),
        (64, 3, 4, 32, False, _FEW),
        (128, 3, 4, 32, False, _FEW),
        (64, 3, 4, 0, False, _FEW),
        (64, 3, 4, 0, True, _FEW),
        (128, 3, 4, 0, True, _FEW),
        (64, 2, 0, 32, False, _SIX),
        (64, 3, 4, 32, False, _SIX),
        (128, 3, 4, 32, False, _SIX),
        (64, 3, 4, 0, False, _SIX),
        (64, 3, 4, 0, True, _SIX),
        (64, 3, 4, 32, False, _NINE),
        (64, 3, 4, 0, True, _NINE),
    ],
)
def test_rows_shared_among_warps_match_the_reference(
    channels: int,
    lmax: int,
    radial_modes: int,
    rank: int,
    spin: bool,
    types: tuple[int, ...],
) -> None:
    """Rows shared among two, four or eight warps follow the reference.

    An edge scan of a canonical system shares each row among the most of 1, 2,
    4 and 8 warps whose warps fit in one wave of its resident warps, so once a
    system is too large for eight warps per row, every doubling of its atoms
    halves the warps per row. The systems of doubling size of
    `_row_split_sweep_in_process`, in a process of their own
    (`_run_in_process`), whose kernel traces show the warps per row of every
    edge scan, therefore take, in every scan of both directions, each split
    from that of the smallest system down to one warp per row, and every system
    follows the reference. The shared rows merge the slot sums of the warps of
    a row in the slotted scans of four slots of a system of two types and in
    those of eight slots of a system of six, except for the edge backward of a
    descriptor without radial modes, which scans six types pairwise, as every
    profile scans the systems of nine types from sixteen atoms on; the
    smallest system of that sweep holds eight of the nine types.
    """
    sweep = _run_in_process(
        f"_row_split_sweep_in_process({channels}, {lmax}, {radial_modes}, "
        f"{rank}, {spin}, {types!r})"
    )
    splits: dict[tuple[str, int], list[int]] = {}
    for _, scans in sweep:
        for direction, scan, split in scans:
            splits.setdefault((direction, scan), []).append(split)
    for kernel, taken in splits.items():
        assert len(taken) == len(sweep), kernel
        assert taken[0] > 1, kernel
        assert taken == sorted(taken, reverse=True), kernel
        assert set(taken) == {taken[0] >> k for k in range(taken[0].bit_length())}, (
            kernel
        )


def _padded_storage(graph: Any, slots: int) -> Any:
    """Return ``graph`` with ``slots`` guard slots past its edge storage.

    The guard slots hold zero bond vectors and the identity entries of the
    source order, and belong to no row of either CSR, as the guard slots of a
    storage that a candidate count sizes do.
    """
    storage = graph.source.shape[0]
    order = torch.arange(
        storage, storage + slots, dtype=torch.int64, device=graph.source.device
    )
    return dataclasses.replace(
        graph,
        source=torch.cat([graph.source, graph.source.new_zeros(slots)]),
        edge_vec=torch.cat([graph.edge_vec, graph.edge_vec.new_zeros(slots, 3)]),
        source_order=torch.cat(
            [graph.source_order, order.to(graph.source_order.dtype)]
        ),
    )


@_GPU
@pytest.mark.parametrize("spin", [False, True])
def test_bond_prefetch_leaves_every_output_unchanged(spin: bool) -> None:
    """The bond prefetch leaves every output of the fused operator unchanged.

    The edge scans of a canonical system request the bond vectors of every row,
    and the slotted scans its source indices, into L2 when the bond vectors of
    the edge storage exceed the L2 cache. A descriptor of 128 channels, degree
    three, four radial modes and nine types, with the low-rank table when it is
    spin-free, evaluates systems of the six types of `_SIX`: one of 40 atoms,
    whose rows the warps of a block share, one of more rows than half a wave of
    the warps of independent rows, which take a warp each, and one of four
    waves of single 24-row blocks per multiprocessor, which takes the
    co-scheduled blocks. Each system evaluates on its own edge storage, whose
    bond vectors fit in L2, and on the same storage with guard slots that carry
    it past the L2 size (`_padded_storage`), and every output of the two agrees
    bit for bit.
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    build = _build_spin_descriptor if spin else _build_descriptor
    descriptor = build(128, 3, 4, type_map=_WIDE_TYPE_MAP)
    if not spin:
        _restrict_scalar_rank(descriptor, rank=32)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    fitting = _energy_fitting(descriptor)
    properties = torch.cuda.get_device_properties("cuda")
    multiprocessors = properties.multi_processor_count
    # Bytes of the fp32 bond vector of one storage slot.
    slot_bytes = 3 * 4
    prefetch_slots = properties.L2_cache_size // slot_bytes + 1
    for node_count in (40, 16 * multiprocessors + 17, 4 * 24 * multiprocessors + 17):
        neighbor_graph, atype = _build_graph(
            descriptor,
            canonical=True,
            node_count=node_count,
            side=5.0 * (node_count / 40) ** (1.0 / 3.0),
            types=_SIX,
        )
        graph = canonical_graph_from_neighbor_graph(
            dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
        )
        storage = graph.source.shape[0]
        assert storage < prefetch_slots, node_count
        moments = None
        if spin:
            generator = torch.Generator(device="cuda").manual_seed(13)
            moments = torch.randn(
                atype.shape[0],
                3,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
        outputs = [
            dpa4c_canonical_compress_energy_force(
                descriptor, fitting, edges, atype, fitting.bias_atom_e[:, 0], moments
            )
            for edges in (graph, _padded_storage(graph, prefetch_slots - storage))
        ]
        for index, (compact, padded) in enumerate(zip(*outputs, strict=True)):
            assert torch.equal(compact, padded), (node_count, index)


@_GPU
@pytest.mark.parametrize(
    ("index", "shape", "dtype", "message"),
    [
        (8, (9, 4, 2), torch.float32, "PairFiLM"),
        (11, (8, 5, 5), torch.float32, "invalid readout matrix"),
        (12, (1, 8), torch.int32, "degree triples"),
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


def _misaligned(tensor: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``tensor`` that starts four bytes past its alignment."""
    storage = torch.zeros(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    copy = storage[1:].view(tensor.shape)
    copy.copy_(tensor)
    return copy


@_GPU
@pytest.mark.parametrize(
    ("channels", "case", "message"),
    [
        (8, "atype", "atype must be one-dimensional"),
        (8, "stride", "table_stride must be positive and finite"),
        (8, "reach", "table_max must cover rcut within"),
        (8, "basis", "8 channels admit no low-rank radial table"),
        (8, "film", "8-byte aligned"),
        (64, "basis_width", "invalid low-rank radial table shape"),
        (64, "expansion_shape", "the low-rank expansion must have shape"),
        (64, "expansion_alone", "a low-rank expansion requires its table"),
        (64, "expansion_alignment", "16-byte aligned"),
    ],
)
def test_operator_rejects_invalid_launch_inputs(
    channels: int, case: str, message: str
) -> None:
    """The operator refuses launch inputs that the scans cannot read.

    A two-dimensional atom-type tensor, a stride that is not finite, a table
    reach beyond the knots of the radial table, a low-rank table for a profile
    without the expansion or of the wrong width, an expansion of the wrong
    shape or without its table, and tables off the alignment of the vector
    loads of the scans each fail with their message. The descriptor of 64
    channels carries the low-rank table of `_restrict_scalar_rank`.
    """
    descriptor = _build_descriptor(channels)
    if channels == 64:
        _restrict_scalar_rank(descriptor, rank=32)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = list(_arguments(descriptor, graph, atype))
    table, basis_table, basis_matrix = arguments[5:8]
    stride = arguments[23]
    if case == "atype":
        arguments[4] = atype[:, None]
    elif case == "stride":
        arguments[23] = math.nan
    elif case == "reach":
        arguments[24] = 1.01 * table.shape[0] * stride
    elif case == "basis":
        arguments[6] = torch.zeros_like(table)
    elif case == "film":
        arguments[8] = _misaligned(arguments[8])
    elif case == "basis_width":
        arguments[6] = basis_table[:, 6:].contiguous()
    elif case == "expansion_shape":
        arguments[7] = basis_matrix[1:].contiguous()
    elif case == "expansion_alone":
        arguments[6] = basis_table[:0]
    else:
        arguments[7] = _misaligned(basis_matrix)
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
    arguments[9] = torch.zeros(
        arguments[9].shape[0],
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
@pytest.mark.parametrize("frozen_by", ["mask", "window"])
@pytest.mark.parametrize("channels", [8, 64, 128])
def test_in_row_mask_matches_removed_edge(channels: int, frozen_by: str) -> None:
    """An edge switched off in its row equals an edge absent from the graph.

    The mask and the bridging window are the two ways a row may hold an edge
    that contributes nothing; the second edge lies inside the frozen zone of
    the window and is switched off by either.
    """
    descriptor = _build_descriptor(
        channels, **(_WINDOW if frozen_by == "window" else {})
    )
    edge_index = torch.tensor(
        [[1, 0], [0, 1]],
        dtype=torch.long,
        device="cuda",
    )
    edge_vec = torch.tensor(
        [[1.0, 1.4, -0.1], [-0.7, 0.3, 0.4]],
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
        edge_mask=torch.tensor(
            [True, frozen_by == "window"], dtype=torch.bool, device="cuda"
        ),
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

    if frozen_by == "mask":
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
@pytest.mark.parametrize("mode", ["zbl", "nlh"])
@pytest.mark.parametrize("windowed", [True, False])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("channels", [8, 32, 128])
@pytest.mark.parametrize(("lmax", "radial_modes"), [(2, 0), (3, 4), (4, 2)])
def test_pair_potential_matches_the_portable_term(
    channels: int,
    canonical: bool,
    windowed: bool,
    lmax: int,
    radial_modes: int,
    mode: str,
) -> None:
    """The backward scan evaluates the analytical pair potential of a bridged model.

    A vanishing descriptor cotangent isolates the term: the edge gradient is
    then the pair slope alone, weighted by the energy cotangent of the
    destination, and the scan returns the pair energy of every node. The three
    widths cover the two-, four- and eight-lane splits of the four series
    terms. Without a window the pair table alone selects the bridged backward,
    behind a plain forward. Both supported modes ride the kernel, which reads
    the coefficients of whichever potential filled the table; their decay rates
    differ by three orders of magnitude, so the single-precision table has to
    carry either.
    """
    from deepmd.dpmodel.atomic_model.inner_potential import (
        InnerPotential,
    )

    descriptor = _build_descriptor(
        channels, lmax, radial_modes, **(_WINDOW if windowed else {})
    )
    graph, atype = _build_graph(descriptor, canonical=canonical)
    arguments = _arguments(descriptor, graph, atype)
    output, state = torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)
    potential = InnerPotential(type_map=["O", "H"], mode=mode)
    pair_table = torch.as_tensor(potential.pair_table, device="cuda")
    # Ghost nodes carry no energy, so their cotangent vanishes.
    seed = (torch.arange(atype.shape[0], device="cuda") % 3 != 0).to(torch.float64)

    edge_gradient, _, _, pair_energy = torch.ops.deepmd.dpa4c_graph_compress_backward(
        torch.zeros_like(output),
        state,
        graph.edge_vec,
        *arguments,
        pair_table,
        seed,
    )
    edge = graph.edge_vec.double().requires_grad_(True)
    reference = potential.call(
        edge, graph.edge_index, atype, graph.edge_mask, atype.shape[0]
    )[0]
    (reference_gradient,) = torch.autograd.grad((reference[:, 0] * seed).sum(), edge)
    assert reference.max() > 10.0
    torch.testing.assert_close(pair_energy, reference, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        edge_gradient.double(), reference_gradient, atol=3e-5, rtol=3e-5
    )

    no_pair = graph.edge_vec.new_empty(0)
    plain_gradient, _, _, absent = torch.ops.deepmd.dpa4c_graph_compress_backward(
        torch.zeros_like(output), state, graph.edge_vec, *arguments, no_pair, no_pair
    )
    assert absent.numel() == 0
    assert torch.count_nonzero(plain_gradient).item() == 0

    # A pair term without energy cotangent leaves the descriptor gradient of
    # the plain backward. Without a window that gradient comes from the plain
    # specialization, which the compiler schedules and contracts independently
    # of the bridged one, so the two agree to single-precision rounding, far
    # below any slope of the pair term.
    cotangent = torch.randn_like(output)
    descriptor_gradient = _descriptor_backward(
        cotangent, state, graph.edge_vec, *arguments
    )[0]
    unseeded_gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        cotangent,
        state,
        graph.edge_vec,
        *arguments,
        pair_table,
        torch.zeros_like(seed),
    )[0]
    scale = descriptor_gradient.abs().max().item()
    torch.testing.assert_close(
        unseeded_gradient, descriptor_gradient, atol=1e-6 * scale, rtol=1e-6
    )


@_GPU
@pytest.mark.parametrize("channels", [8, 32])
@pytest.mark.parametrize("bridged", [False, True])
def test_padding_type_edge_has_zero_gradient(channels: int, bridged: bool) -> None:
    descriptor = _build_descriptor(channels, **(_WINDOW if bridged else {}))
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
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("channels", [8, 64, 128])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.uint32])
def test_compact_canonical_parity(
    channels: int,
    index_dtype: torch.dtype,
    windowed: bool,
) -> None:
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        ensure_registered as ensure_canonical_registered,
    )

    descriptor = _build_descriptor(channels, **(_WINDOW if windowed else {}))
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
        *arguments[5:21],
        *arguments[22:],
    )
    compact_output, compact_state = torch.ops.deepmd.dpa4c_canonical_compress(
        graph.edge_vec,
        *canonical_arguments,
    )
    torch.testing.assert_close(compact_output, generic_output)
    torch.testing.assert_close(compact_state, generic_state)

    cotangent = torch.randn_like(generic_output)
    generic_gradient = _descriptor_backward(
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
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        ensure_registered as ensure_canonical_registered,
    )

    descriptor = _build_descriptor(channels)
    graph, atype = _build_graph(descriptor, canonical=True, node_count=23)
    arguments = _arguments(descriptor, graph, atype)
    canonical_arguments = (
        graph.edge_index[0].to(torch.uint32),
        graph.destination_row_ptr,
        atype,
        *arguments[5:21],
        *arguments[22:],
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
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )
    from deepmd.pt_expt.kernels.edge_force_virial import (
        edge_force_virial,
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


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "low_rank", "system"),
    [
        (8, 2, 0, False, "split"),
        (64, 2, 0, False, "split"),
        (64, 2, 0, False, "unsplit"),
        (128, 2, 0, False, "coscheduled"),
        (64, 3, 4, False, "split"),
        (64, 3, 4, False, "unsplit"),
        (64, 2, 0, True, "split"),
        (64, 2, 0, True, "unsplit"),
        (128, 3, 4, True, "coscheduled"),
    ],
)
@pytest.mark.parametrize("type_count", [2, 6])
def test_compact_canonical_tiling_is_equivalent(
    channels: int,
    lmax: int,
    radial_modes: int,
    low_rank: bool,
    system: str,
    type_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node tiling changes no output in any bit.

    Every node tile owns a contiguous span of the destination-sorted edge axis,
    so the runs partition the work rather than splitting any reduction, and
    every run scans its rows with the schedule of the whole system, whose row
    split, co-scheduled blocks and slotted scans a run of other length or of
    fewer node types would not take. The rows of the graph are sorted by bond
    length, so the lockstep windows of a co-scheduled row follow from the row
    alone, whichever rows share its block. The sizes follow the multiprocessor
    count: an unsplit system holds more rows than half a wave of the warps of
    independent rows, of which a multiprocessor keeps at most 32, so it scans
    one warp per row while its shorter runs would share each row; a
    co-scheduled system holds four waves of single 24-row blocks per
    multiprocessor, which its runs of less than one wave would not take. The
    system holds two or six types, so the profiles with radial modes or a
    low-rank table scan it slotted, in four or eight slots, but for the edge
    backward of six types of a profile without radial modes, which scans
    pairwise. The atoms keep the density of the smallest system.
    """
    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count, tiles = {
        "split": (97, (7, 32, 96)),
        "unsplit": (16 * multiprocessors + 17, (100, 4 * multiprocessors)),
        "coscheduled": (
            4 * 24 * multiprocessors + 17,
            (12 * multiprocessors, 24 * multiprocessors),
        ),
    }[system]
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    descriptor = _build_descriptor(
        channels, lmax, radial_modes, type_map=_WIDE_TYPE_MAP[:type_count]
    )
    if low_rank:
        _restrict_scalar_rank(descriptor, rank=8)
    neighbor_graph, atype = _build_graph(
        descriptor,
        canonical=True,
        node_count=node_count,
        side=5.0 * (node_count / 97) ** (1.0 / 3.0),
        types=range(type_count),
    )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    artifacts = build_compression_artifacts(descriptor)
    assert (artifacts["basis_table"].numel() > 0) is low_rank
    descriptor._set_compression(artifacts)
    fitting = (
        EnergyFittingNet(
            ntypes=type_count,
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

    def run() -> tuple[torch.Tensor, ...]:
        return dpa4c_canonical_compress_energy_force(
            descriptor,
            fitting,
            graph,
            atype,
            fitting.bias_atom_e[:, 0],
        )

    monkeypatch.setenv("DP_NODE_TILE", "0")
    reference = run()
    for tile in tiles:
        monkeypatch.setenv("DP_NODE_TILE", str(tile))
        for actual, expected in zip(run(), reference, strict=True):
            assert torch.equal(actual, expected), f"tile {tile}"


@_GPU
@pytest.mark.parametrize("bridged", [False, True])
@pytest.mark.parametrize(("frames", "halo"), [(1, 0), (2, 0), (2, 3)])
def test_fused_canonical_cpu_reference_matches_the_kernel(
    bridged: bool,
    frames: int,
    halo: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU implementation of the fused canonical operator reproduces it.

    The operator runs once on CUDA through the energy-force helper and once on
    CPU copies of the same arguments, where it dispatches to the reference
    sequence of descriptor, fitting, fitting backward and descriptor backward,
    with the pair potential when the model is bridged. A batch of two frames
    takes the multi-frame path of the operator, its frame lookup and its
    segment sums, with the last ``halo`` nodes of each frame unowned.
    """
    window = {"inner_clamp_f_inner": 0.9, "inner_clamp_f_outer": 1.3} if bridged else {}
    descriptor = _build_descriptor(8, **window)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    _check_canonical_against_reference(
        descriptor,
        bridged,
        monkeypatch,
        node_count=24,
        single_type=False,
        halo=halo,
        frames=frames,
    )


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "bridged"),
    [
        (32, 2, 0, False),
        (64, 2, 0, True),
        (64, 3, 4, False),
        (128, 3, 4, True),
        (64, 3, 2, False),
        (64, 2, 8, False),
    ],
)
def test_single_type_canonical_matches_the_reference(
    channels: int,
    lmax: int,
    radial_modes: int,
    bridged: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-type system matches the CPU reference of the fused operator.

    Every edge of such a system carries one ordered pair, which the CUDA scans
    exploit by applying the PairFiLM and mode coefficients to reduced sums and
    folded cotangents instead of to every edge. Eight modes exceed what the
    folded scans carry and pin the unfolded path of the same system.
    """
    window = {"inner_clamp_f_inner": 0.9, "inner_clamp_f_outer": 1.3} if bridged else {}
    descriptor = _build_descriptor(channels, lmax, radial_modes, **window)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    _check_canonical_against_reference(descriptor, bridged, monkeypatch)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "folds"),
    [(64, 2, 0, True), (128, 3, 4, True), (64, 2, 8, False)],
)
@pytest.mark.parametrize("spin", [False, True])
def test_single_type_systems_take_the_folded_scans(
    channels: int,
    lmax: int,
    radial_modes: int,
    folds: bool,
    spin: bool,
) -> None:
    """A single-type system takes the folded scans, which sum other terms.

    Both systems hold the same atoms and one isolated atom, which interacts
    with none; in one of them it carries the type of the others and in the
    other the second type, which ends the single-type system, so the scans
    read the ordered pair of every edge again, or slot the mode profiles of
    the two types for a profile that carries modes. The node counts agree, so
    every launch choice that follows from them does as well. The folded scans
    apply the pair coefficients to reduced sums once per node, so their forces
    differ from the pairwise and the slotted ones in the last bits. Eight
    modes exceed what the folded and the slotted scans carry: that profile
    scans pairwise either way, and its forces agree bit for bit. A
    spin-conditioned profile has no single-type scan and scans both systems
    alike, pairwise or slotted, and its forces and magnetic forces agree bit
    for bit; its atoms are of type 0, which carries the spin branch.
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    build = _build_spin_descriptor if spin else _build_descriptor
    descriptor = build(channels, lmax, radial_modes)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    neighbor_graph, atype = _build_graph(
        descriptor, canonical=True, node_count=40, single_type=True
    )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    system_type, isolated_type = (0, 1) if spin else (1, 0)
    atype = torch.full_like(atype, system_type)
    moments = None
    if spin:
        generator = torch.Generator(device="cuda").manual_seed(13)
        moments = torch.randn(
            atype.shape[0] + 1,
            3,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    # The isolated atom closes both row-pointer arrays with an empty row.
    extended = dataclasses.replace(
        graph,
        n_node=graph.n_node + 1,
        n_local=graph.n_local + 1,
        destination_row_ptr=torch.cat(
            [graph.destination_row_ptr, graph.destination_row_ptr[-1:]]
        ),
        source_row_ptr=torch.cat([graph.source_row_ptr, graph.source_row_ptr[-1:]]),
    )
    uniform_type = torch.cat([atype, torch.full_like(atype[:1], system_type)])
    mixed_type = torch.cat([atype, torch.full_like(atype[:1], isolated_type)])
    fitting = _energy_fitting(descriptor)
    nodes = atype.shape[0]
    single = dpa4c_canonical_compress_energy_force(
        descriptor, fitting, extended, uniform_type, fitting.bias_atom_e[:, 0], moments
    )
    paired = dpa4c_canonical_compress_energy_force(
        descriptor, fitting, extended, mixed_type, fitting.bias_atom_e[:, 0], moments
    )
    scale = max(single[2].abs().max().item(), 1.0)
    torch.testing.assert_close(
        paired[2][:nodes], single[2][:nodes], atol=2e-5 * scale, rtol=2e-5
    )
    torch.testing.assert_close(
        paired[1][:nodes], single[1][:nodes], atol=2e-5, rtol=2e-5
    )
    assert not paired[2][nodes:].any()
    assert not single[2][nodes:].any()
    folded = folds and not spin
    assert torch.equal(paired[2][:nodes], single[2][:nodes]) is not folded
    if spin:
        assert torch.equal(paired[5][:nodes], single[5][:nodes])


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "bridged"),
    [(32, 2, 0, False), (64, 2, 0, True), (64, 3, 4, False), (128, 3, 4, True)],
)
def test_single_type_spin_canonical_matches_the_reference(
    channels: int,
    lmax: int,
    radial_modes: int,
    bridged: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-type spin system matches the CPU reference of the fused operator.

    A spin-conditioned descriptor has no single-type scan: its single-type
    system takes the slotted scans of a profile with radial modes, and the
    pairwise scans of any other; energy, force, virial and magnetic force
    follow the reference.
    """
    window = {"inner_clamp_f_inner": 0.9, "inner_clamp_f_outer": 1.3} if bridged else {}
    descriptor = _build_spin_descriptor(channels, lmax, radial_modes, **window)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    _check_canonical_against_reference(descriptor, bridged, monkeypatch)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "bridged"),
    [(64, 2, 0, False), (64, 3, 4, True), (128, 3, 4, False)],
)
def test_low_rank_table_matches_the_reference(
    channels: int,
    lmax: int,
    radial_modes: int,
    bridged: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The low-rank single-type scans match the CPU reference.

    Scalar-only channels confined to few functions are reproduced by the
    basis, so the compression adopts the low-rank table, which the single-type
    scans read in place of the radial table, while the reference evaluates the
    radial table itself. The low-rank scans sum other terms than the direct
    ones, so their forces differ from those of the same descriptor without the
    basis in the last bits, which shows that they ran.
    """
    window = {"inner_clamp_f_inner": 0.9, "inner_clamp_f_outer": 1.3} if bridged else {}
    descriptor = _build_descriptor(channels, lmax, radial_modes, **window)
    _restrict_scalar_rank(descriptor, rank=8)
    artifacts = build_compression_artifacts(descriptor)
    first = descriptor_profile(channels, lmax).degree_channels[1]
    assert artifacts["basis_matrix"].shape == (32, channels - first)
    assert artifacts["basis_table"].shape == (
        artifacts["data"].shape[0],
        6 * (first + 32 + radial_modes),
    )
    descriptor._set_compression(artifacts)
    low_rank = _check_canonical_against_reference(descriptor, bridged, monkeypatch)
    empty = artifacts["basis_table"].new_empty(0)
    descriptor._set_compression(
        {**artifacts, "basis_table": empty, "basis_matrix": empty}
    )
    direct = _check_canonical_against_reference(descriptor, bridged, monkeypatch)
    assert not torch.equal(low_rank[2], direct[2])


@_GPU
@pytest.mark.parametrize("single_type", [True, False])
@pytest.mark.parametrize("placeholders", [False, True])
def test_canonical_halo_nodes_match_the_reference(
    single_type: bool,
    placeholders: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unowned trailing nodes and source-CSR placeholders match the reference.

    The nodes past ``n_local`` lend their bonds to the owned nodes but carry
    no energy of their own, and the two-entry placeholders that stand in for
    the source CSR of a caller that declines the per-atom virial leave that
    output zero.
    """
    descriptor = _build_descriptor(
        64, 3, 4, inner_clamp_f_inner=0.9, inner_clamp_f_outer=1.3
    )
    descriptor._set_compression(build_compression_artifacts(descriptor))
    kernel = _check_canonical_against_reference(
        descriptor,
        True,
        monkeypatch,
        single_type=single_type,
        halo=9,
        placeholders=placeholders,
    )
    assert not kernel[6][-9:].any()
    if placeholders:
        assert not kernel[4].any()


def _device_filling_scans(
    single_type: bool, spin: bool
) -> list[tuple[str, int, int, int, float]]:
    """Compare the fused operator with its CPU reference on a device-filling system.

    The system holds two blocks of 24 rows per multiprocessor and 17 atoms
    more, of one type or of two, at a density of 0.08 atoms per cubic Å, which
    puts about 40 atoms within the 5 Å cutoff of an atom inside the cube. Its
    descriptor has 128 channels and degree two, and four radial modes when it
    is spin-free and none when it is spin-conditioned. Returns the edge scans
    that the comparison launched (see `_edge_scans`).
    """
    from torch.profiler import (
        ProfilerActivity,
        profile,
    )

    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count = 2 * 24 * multiprocessors + 17
    if spin:
        descriptor = _build_spin_descriptor(128, 2, 0, rcut=5.0)
    else:
        descriptor = _build_descriptor(128, 2, 4, rcut=5.0)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    with (
        pytest.MonkeyPatch.context() as patch,
        profile(activities=[ProfilerActivity.CUDA]) as trace,
    ):
        _check_canonical_against_reference(
            descriptor,
            False,
            patch,
            node_count=node_count,
            side=(node_count / 0.08) ** (1.0 / 3.0),
            single_type=single_type,
        )
    return _edge_scans(trace)


@_GPU
@pytest.mark.parametrize(
    ("single_type", "spin"), [(False, False), (True, False), (False, True)]
)
def test_coscheduled_scans_match_the_reference(
    single_type: bool,
    spin: bool,
) -> None:
    """Co-scheduled edge scans of a device-filling graph match the reference.

    A system of at least two blocks of 24 co-scheduled rows per multiprocessor
    (`_device_filling_scans`) walks the rows of a wide profile in lockstep
    windows of 4 Å of bond length, and a cutoff beyond one window makes every
    block realign at a barrier inside its rows. The small systems of the other
    tests leave every row to warps of its own instead. The kernel trace shows
    that both scans ran with co-scheduled blocks in the kernels that serve the
    system: the longest edge scan of each direction is co-scheduled, and is a
    slotted scan for a system of two types and a single-type scan for one of
    one type, whose launches also start the kernels of the other, whose blocks
    retire at once. A spin profile without radial modes has nothing to slot and
    scans its system of two types pairwise, in the kernels of the single-type
    scans. The node count gives every multiprocessor two blocks of the widest
    co-scheduled shape on any device. The comparison and its kernel trace run
    in a process of their own (`_run_in_process`).
    """
    scans = _run_in_process(f"_device_filling_scans({single_type}, {spin})")
    assert scans
    # Blocks of more than the four independent rows of a slotted or a spin
    # scan are co-scheduled ones.
    slotted = not (single_type or spin)
    serving = _SLOTTED_SCANS if slotted else _SINGLE_TYPE_SCANS
    for direction in ("forward", "edge_backward"):
        _, rows, _, scan, _ = max(
            (entry for entry in scans if entry[0] == direction),
            key=lambda entry: entry[4],
        )
        assert rows > 4, direction
        assert scan in serving, direction


@_GPU
@pytest.mark.parametrize("frames", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("spin", [False, True])
def test_non_finite_energy_marks_the_forces_of_its_frame(
    frames: int, device: str, spin: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame whose energy is not finite reports its forces as not a number.

    A fixed-point force word cannot carry a non-finite value, so the operator
    marks the forces of every frame whose totals are not finite, and only
    those, together with their magnetic forces in a spin-conditioned model.
    The CPU reference of the operator marks the same forces.
    """
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    build = _build_spin_descriptor if spin else _build_descriptor
    descriptor = build(64)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    neighbor_graph, atype = _build_graph(
        descriptor, canonical=True, node_count=40, frames=frames
    )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    fitting = _energy_fitting(descriptor)
    # The first frame consists of the type whose energy bias is not a number,
    # which is the type that carries the spin branch.
    frame = torch.repeat_interleave(
        torch.arange(frames, device="cuda"), neighbor_graph.n_node
    )
    atype = (frame != 0).to(atype.dtype)
    atom_bias = torch.tensor([math.nan, 0.0], dtype=torch.float64, device="cuda")
    moments = None
    if spin:
        generator = torch.Generator(device="cuda").manual_seed(13)
        moments = torch.randn(
            atype.shape[0], 3, dtype=torch.float32, device="cuda", generator=generator
        )
    operator = torch.ops.deepmd.dpa4c_canonical_compress_energy_force
    arguments = _fused_arguments(
        monkeypatch, descriptor, fitting, graph, atype, atom_bias, moments
    )
    if device == "cpu":
        arguments = [_to_cpu(argument) for argument in arguments]
    energy, _, force, _, _, force_mag, _ = operator(*arguments)
    broken = (frame == 0).to(force.device)
    assert torch.isnan(energy[0]).all()
    assert torch.isfinite(energy[1:]).all()
    assert torch.isnan(force[broken]).all()
    assert torch.isfinite(force[~broken]).all()
    if spin:
        assert torch.isnan(force_mag[broken]).all()
        assert torch.isfinite(force_mag[~broken]).all()


@_GPU
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_an_incomplete_source_csr_is_refused(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source CSR is complete or two-entry placeholders, and nothing else.

    The operator and its CPU reference refuse a source CSR of any other
    extent rather than reading a per-atom virial from it.
    """
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    descriptor = _build_descriptor(8)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    neighbor_graph, atype = _build_graph(descriptor, canonical=True)
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    fitting = _energy_fitting(descriptor)
    operator = torch.ops.deepmd.dpa4c_canonical_compress_energy_force
    truncated = dataclasses.replace(graph, source_row_ptr=graph.source_row_ptr[:-1])
    arguments = _fused_arguments(
        monkeypatch, descriptor, fitting, truncated, atype, fitting.bias_atom_e[:, 0]
    )
    if device == "cpu":
        arguments = [_to_cpu(argument) for argument in arguments]
    with pytest.raises(RuntimeError, match="source CSR"):
        operator(*arguments)


@_GPU
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("spin", [False, True])
def test_an_empty_graph_has_no_energy(
    device: str, spin: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph without nodes has zero frame totals and empty node outputs.

    The canonical graph of an empty system keeps its two storage slots, which
    name a node that does not exist. The operator and its CPU reference
    return a zero energy and virial for every frame and node outputs of the
    declared layout without reading those slots.
    """
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_pairs,
    )

    build = _build_spin_descriptor if spin else _build_descriptor
    descriptor = build(8)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    graph = canonical_graph_from_pairs(
        empty,
        empty,
        torch.empty(0, 3, dtype=torch.float32, device="cuda"),
        torch.zeros(2, dtype=torch.int64, device="cuda"),
        0,
    )
    moments = torch.empty(0, 3, dtype=torch.float32, device="cuda") if spin else None
    fitting = _energy_fitting(descriptor)
    arguments = _fused_arguments(
        monkeypatch,
        descriptor,
        fitting,
        graph,
        empty,
        fitting.bias_atom_e[:, 0],
        moments,
    )
    if device == "cpu":
        arguments = [_to_cpu(argument) for argument in arguments]
    energy, atom_energy, force, virial, atom_virial, force_mag, mask = (
        torch.ops.deepmd.dpa4c_canonical_compress_energy_force(*arguments)
    )
    assert torch.equal(energy, energy.new_zeros(2, 1))
    assert torch.equal(virial, virial.new_zeros(2, 3, 3))
    assert energy.dtype == virial.dtype == torch.float64
    assert atom_energy.shape == (0, 1)
    assert force.shape == (0, 3)
    assert atom_virial.shape == (0, 3, 3)
    assert force_mag.shape == ((0, 3) if spin else (0,))
    assert mask.shape == (0,)


@_GPU
@pytest.mark.parametrize("frames", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("spin", [False, True])
def test_out_of_range_forces_mark_their_frame(
    frames: int, device: str, spin: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A force contribution of 2^30 eV/Å or more marks its frame.

    The fixed-point force totals of a node hold magnitudes below 2^31 eV/Å. A
    contribution of 2^30 eV/Å or more, the row total of a destination or an
    edge onto its source, makes its node report its energy as not a number,
    which marks the forces of its frame like a non-finite total; the CPU
    reference of the operator marks the same frames. The energy is affine in
    the head weight and the forces are linear in it, so scaling the weight
    scales every contribution. The largest contribution of the unscaled weight
    is measured on the CPU reference, by lowering its limit until a frame is
    marked, which gives the scale that brings that contribution to 2^30 eV/Å: a
    thousandth below it no frame is marked, and a thousandth above it the
    frames that the CPU reference marks are marked, with every force of theirs
    and none of the others. The forces scale with the weight up to the largest
    scale that keeps every total within the range of the totals as well; the
    check covers each contribution, and three or more contributions below the
    limit, such as the uniform scaling makes, can carry a total out of the
    range undetected.

    A spin-conditioned model checks the magnetic contribution of every edge
    onto its source against the same limit, in the units of the magnetic
    force. Its moments are divided by 2^10 and its conditioning factor, the
    gate over the reference magnitude, multiplied by 2^10: the conditioned
    moments, and with them the energy and every coordinate contribution,
    are bitwise those of the unscaled moments, while every magnetic
    contribution, a cotangent of the raw moment, grows by 2^10 and reaches
    the limit first. The frames are then marked at a far smaller weight
    than with the unscaled moments, which the coordinate contributions
    alone cannot cause, and the marked frames report their magnetic forces
    as not a number as well.
    """
    from deepmd.pt_expt.kernels.dpa4c import (
        canonical,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    build = _build_spin_descriptor if spin else _build_descriptor
    descriptor = build(64)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    neighbor_graph, atype = _build_graph(
        descriptor, canonical=True, node_count=40, frames=frames
    )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    fitting = _energy_fitting(descriptor)
    operator = torch.ops.deepmd.dpa4c_canonical_compress_energy_force
    moments = None
    if spin:
        generator = torch.Generator(device="cuda").manual_seed(13)
        moments = torch.randn(
            atype.shape[0], 3, dtype=torch.float32, device="cuda", generator=generator
        )
    arguments = _fused_arguments(
        monkeypatch,
        descriptor,
        fitting,
        graph,
        atype,
        fitting.bias_atom_e[:, 0],
        moments,
    )
    reference = [_to_cpu(argument) for argument in arguments]

    head_slot = [
        argument.name for argument in operator.default._schema.arguments
    ].index("w_head")

    def evaluate(values: list[Any], scale: float) -> tuple[torch.Tensor, ...]:
        scaled = list(values)
        scaled[head_slot] = values[head_slot] * scale
        return operator(*scaled)

    def threshold_of(values: list[Any]) -> float:
        """Return the weight scale that brings the largest contribution to 2^30."""

        def marked_at(limit: float) -> bool:
            with monkeypatch.context() as patch:
                patch.setattr(canonical, "_FORCE_FIXED_LIMIT", limit)
                energy, *_ = evaluate(values, 1.0)
            return bool(torch.isnan(energy).any())

        # The largest contribution of the unscaled weight lies between a limit
        # that marks a frame and one that marks none; the bisection of their
        # exponents narrows the bracket below a relative 1e-7.
        low, high = -40.0, 30.0
        assert marked_at(2.0**low)
        assert not marked_at(2.0**high)
        for _ in range(30):
            middle = 0.5 * (low + high)
            if marked_at(2.0**middle):
                low = middle
            else:
                high = middle
        return 2.0 ** (30.0 - high)

    # === The largest contribution of the unscaled weight ===
    threshold = threshold_of(reference)
    if spin:
        # === The magnetic contributions reach the limit first ===
        names = [argument.name for argument in operator.default._schema.arguments]
        spin_slot, type_slot = names.index("spin"), names.index("spin_type")

        def rescaled(values: list[Any]) -> list[Any]:
            values = list(values)
            values[spin_slot] = values[spin_slot] * 2.0**-10
            values[type_slot] = values[type_slot].clone()
            values[type_slot][:, 0] *= 2.0**10
            return values

        arguments, reference = rescaled(arguments), rescaled(reference)
        unscaled_threshold, threshold = threshold, threshold_of(reference)
        assert threshold < unscaled_threshold / 16
    tested = reference if device == "cpu" else arguments

    # === Within the range, the forces scale with the weight ===
    _, _, force, *_ = evaluate(tested, 1.0)
    largest_total = force.abs().max().item()
    scale = min(threshold, 2.0**31 / largest_total) * (1.0 - 1e-3)
    energy, _, scaled, *_ = evaluate(tested, scale)
    assert torch.isfinite(energy).all()
    torch.testing.assert_close(
        scaled, force * scale, rtol=1e-5, atol=1e-5 * scale * largest_total
    )

    # === Below the limit, no frame is marked ===
    energy, *_ = evaluate(tested, threshold * (1.0 - 1e-3))
    assert torch.isfinite(energy).all()

    # === Above it, the frames of the contributions beyond it are marked ===
    scale = threshold * (1.0 + 1e-3)
    expected, *_ = evaluate(reference, scale)
    energy, _, marked, _, _, marked_magnetic, _ = evaluate(tested, scale)
    frame_marked = torch.isnan(energy).flatten().cpu()
    assert frame_marked.any()
    assert torch.equal(frame_marked, torch.isnan(expected).flatten())
    node_marked = torch.repeat_interleave(frame_marked, reference[5])
    assert node_marked.numel() == marked.shape[0]
    marked = marked.cpu()
    assert torch.isnan(marked[node_marked]).all()
    assert torch.isfinite(marked[~node_marked]).all()
    if spin:
        marked_magnetic = marked_magnetic.cpu()
        assert torch.isnan(marked_magnetic[node_marked]).all()
        assert torch.isfinite(marked_magnetic[~node_marked]).all()


def _to_cpu(value: Any) -> Any:
    """Return a tensor, or a list of tensors, on the CPU, and anything else unchanged."""
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _energy_fitting(descriptor: DescrptDPA4C) -> Any:
    """Return the two-layer fp32 SiLU energy fitting of the fused-operator cases."""
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )

    return (
        EnergyFittingNet(
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            neuron=[32, 32],
            resnet_dt=False,
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=29,
        )
        .cuda()
        .eval()
    )


def _fused_arguments(
    monkeypatch: pytest.MonkeyPatch,
    descriptor: DescrptDPA4C,
    fitting: Any,
    graph: Any,
    atype: torch.Tensor,
    atom_bias: torch.Tensor,
    spin: torch.Tensor | None = None,
    pair_table: torch.Tensor | None = None,
) -> list[Any]:
    """Return the arguments with which the fused wrapper calls the operator.

    The call is recorded rather than evaluated. The CPU reference of the
    operator is registered first, so that the arguments can be evaluated on
    either device.
    """
    from deepmd.pt_expt.kernels.dpa4c import (
        canonical,
    )

    canonical.ensure_registered()
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.ops.deepmd,
            "dpa4c_canonical_compress_energy_force",
            lambda *arguments: calls.append(list(arguments)),
        )
        canonical.dpa4c_canonical_compress_energy_force(
            descriptor, fitting, graph, atype, atom_bias, spin, pair_table
        )
    (arguments,) = calls
    return arguments


def _check_canonical_against_reference(
    descriptor: DescrptDPA4C,
    bridged: bool,
    monkeypatch: pytest.MonkeyPatch,
    *,
    node_count: int = 40,
    side: float = 5.0,
    single_type: bool = True,
    halo: int = 0,
    placeholders: bool = False,
    frames: int = 1,
    types: Sequence[int] = (0, 1),
) -> tuple[torch.Tensor, ...]:
    """Compare the fused CUDA operator with its CPU reference.

    The graph holds ``frames`` frames, the last ``halo`` nodes of each left
    unowned, and ``placeholders`` replaces the source CSR by the two-entry
    placeholders of a caller that declines the per-atom virial. The atoms of a
    system of several types cycle through ``types``. A bridged descriptor takes
    the pair potential of its own type map, and is compared without it as
    well: the tolerance follows the largest magnitude of every output, and the
    pair potential of the short bonds of the random cluster dominates the
    forces. A spin-conditioned descriptor receives random moments, and its
    single-type system consists of type 0, which carries the spin branch.
    Returns the outputs of the CUDA operator, with the pair potential of a
    bridged descriptor.
    """
    from deepmd.dpmodel.atomic_model.inner_potential import (
        InnerPotential,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    neighbor_graph, atype = _build_graph(
        descriptor,
        canonical=True,
        node_count=node_count,
        single_type=single_type,
        side=side,
        frames=frames,
        types=types,
    )
    spin = None
    if descriptor.spin is not None:
        if single_type:
            atype = torch.zeros_like(atype)
        generator = torch.Generator(device="cuda").manual_seed(13)
        spin = torch.randn(
            atype.shape[0], 3, dtype=torch.float32, device="cuda", generator=generator
        )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node - halo)
    )
    if placeholders:
        graph = dataclasses.replace(
            graph,
            source_row_ptr=graph.source_row_ptr.new_zeros(2),
            source_order=graph.source_order.new_zeros(2),
        )
    fitting = _energy_fitting(descriptor)
    pair_table = (
        torch.as_tensor(
            InnerPotential(descriptor.get_type_map()).pair_table, device="cuda"
        )
        if bridged
        else None
    )
    operator = torch.ops.deepmd.dpa4c_canonical_compress_energy_force
    outputs = []
    for table in (pair_table, None) if bridged else (None,):
        arguments = _fused_arguments(
            monkeypatch,
            descriptor,
            fitting,
            graph,
            atype,
            fitting.bias_atom_e[:, 0],
            spin,
            table,
        )
        kernel = operator(*arguments)
        reference = operator(*[_to_cpu(argument) for argument in arguments])
        for actual, expected in zip(kernel, reference, strict=True):
            scale = max(expected.abs().max().item(), 1.0) if expected.numel() else 1.0
            torch.testing.assert_close(
                actual.cpu(), expected, atol=2e-5 * scale, rtol=2e-5
            )
        outputs.append(kernel)
    return outputs[0]


def _build_spin_descriptor(
    channels: int,
    lmax: int = 2,
    radial_modes: int = 0,
    device: str = "cuda",
    type_map: Sequence[str] = tuple(_TYPE_MAP),
    rcut: float = 3.0,
    **window: float,
) -> DescrptDPA4C:
    """Return a spin-conditioned descriptor with a non-unit reference moment.

    A reference magnitude other than one makes the conditioning factor visible
    in the magnetic gradient, so a missing chain factor shows up as a constant
    ratio rather than cancelling. The branch gate is opened, since a fresh
    descriptor starts spin-free by design and these tests are about the branch
    behind the gate; the gate itself is covered in ``test_dpa4c.py``. The
    first type carries the spin branch, and the others none. The mode mixing
    is strengthened as in `_build_descriptor`.
    """
    descriptor = (
        DescrptDPA4C(
            rcut=rcut,
            ntypes=len(type_map),
            channels=channels,
            lmax=lmax,
            n_radial=8,
            radial_modes=radial_modes,
            precision="float32",
            seed=17,
            type_map=list(type_map),
            use_spin=[True] + [False] * (len(type_map) - 1),
            **window,
        )
        .to(device)
        .eval()
    )
    descriptor.spin.set_spin_reference(np.array([1.7] + [1.0] * len(type_map)))
    with torch.no_grad():
        descriptor.spin.spin_gate.fill_(1.0)
    _strengthen_mode_mixing(descriptor)
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

    output, state = _reference_forward(graph.edge_vec, *arguments)

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

    edge_gradient, spin_gradient, edge_spin_gradient = _descriptor_backward(
        torch.empty_like(output),
        state,
        graph.edge_vec,
        *arguments,
    )

    assert edge_gradient.shape == (0, 3)
    assert spin_gradient.shape == (0, 3)
    assert edge_spin_gradient.shape == (0, 3)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes"),
    [(8, 2, 0), (16, 2, 0), (32, 2, 4), (64, 3, 0), (128, 3, 8), (32, 4, 0)],
)
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("single_type", [False, True])
def test_spin_compressed_matches_portable(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    lmax: int,
    radial_modes: int,
    windowed: bool,
    single_type: bool,
) -> None:
    """Descriptor, coordinate gradient and magnetic force all match.

    The portable path is the oracle for the spin families. Probing every
    output column at once keeps the geometric and the spin blocks in the same
    comparison, because the two are coupled through the shared normalizer and
    through the cross Gram. The cotangent stays bounded: weighting columns by
    their index inflates the gradient magnitude at the widest profile until
    fp32 tabulation noise alone exceeds the tolerance, and the tolerances
    follow the magnitude of each output (`_assert_close_at_scale`). A system
    of the spin-carrying type alone is covered as well as a system of two
    types.
    """
    descriptor = _build_spin_descriptor(
        channels, lmax, radial_modes, **(_WINDOW if windowed else {})
    )
    assert mega_eligible(descriptor)
    graph, atype = _build_graph(descriptor, canonical=True)
    if single_type:
        # Type 0 carries the spin branch; the graph does not depend on types.
        atype = torch.zeros_like(atype)
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
    # they carry the wider tolerance the geometric backward already uses, and
    # the steep switch inside a bridging window widens it as in the surface
    # test.
    gradient = (1e-4, 5e-4) if windowed else (8e-6, 1e-4)
    tolerances = ((3e-5, 3e-5), gradient, gradient)
    for (atol, rtol), value, expected in zip(
        tolerances, actual, reference, strict=True
    ):
        _assert_close_at_scale(value, expected, atol=atol, rtol=rtol)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "type_count", "windowed"),
    [
        (64, 3, 4, 1, False),
        (64, 3, 4, 3, False),
        (64, 3, 4, 3, True),
        (128, 3, 4, 1, False),
        (128, 3, 4, 4, False),
        (128, 3, 4, 4, True),
        (32, 2, 4, 2, False),
        (64, 3, 2, 3, False),
        (32, 2, 2, 2, False),
        (128, 3, 2, 1, False),
        (64, 3, 4, 6, False),
        (64, 3, 4, 8, True),
        (128, 3, 4, 6, False),
        (128, 3, 4, 8, True),
        (64, 3, 2, 6, False),
        (128, 3, 8, 2, False),
        (128, 3, 8, 2, True),
    ],
)
@pytest.mark.parametrize("system", ["unsplit", "coscheduled"])
def test_spin_slotted_scans_match_the_portable_equations(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    lmax: int,
    radial_modes: int,
    type_count: int,
    windowed: bool,
    system: str,
) -> None:
    """The slotted spin scans follow the portable equations.

    A canonical spin system of one to eight node types slots the mode profiles
    of its edges by the types of their sources in the edge scans of both
    directions, in four slots up to four types and in eight from five, for two
    modes as well as for four. The graph carries 64-bit indices, whose rows
    always take a warp each; a co-scheduled system holds four waves of single
    24-row blocks per multiprocessor, more than the co-scheduled blocks of
    every slotted spin scan need. A descriptor of eight radial modes, more than
    the slots carry, scans pairwise, and the co-scheduled system then in the
    co-scheduled blocks of the pairwise spin scans. The atoms of the random
    cluster keep the density of the 24-atom one. Descriptor, coordinate
    gradient and magnetic force follow the portable path, at tolerances that
    follow the magnitude of each output (`_assert_close_at_scale`).
    """
    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count = {
        "unsplit": 2 * multiprocessors + 17,
        "coscheduled": 4 * 24 * multiprocessors + 17,
    }[system]
    descriptor = _build_spin_descriptor(
        channels,
        lmax,
        radial_modes,
        type_map=_WIDE_TYPE_MAP[: max(type_count, 4)],
        **(_WINDOW if windowed else {}),
    )
    graph, atype = _build_graph(
        descriptor,
        canonical=True,
        node_count=node_count,
        side=5.0 * (node_count / 24) ** (1.0 / 3.0),
        types=range(type_count),
    )
    generator = torch.Generator(device="cuda").manual_seed(11)
    spin = torch.randn(
        atype.shape[0], 3, dtype=torch.float32, device="cuda", generator=generator
    )

    def run() -> tuple[torch.Tensor, ...]:
        moment = spin.detach().clone().requires_grad_(True)
        edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
        output, _ = descriptor.call_graph(
            dataclasses.replace(graph, edge_vec=edge_vec), atype, spin=moment
        )
        cotangent = torch.linspace(
            -0.7, 1.3, output.numel(), dtype=output.dtype, device=output.device
        ).reshape_as(output)
        gradients = torch.autograd.grad((output * cotangent).sum(), [edge_vec, moment])
        return (output, *gradients)

    reference = run()
    descriptor.enable_compression(0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    actual = run()
    _assert_dispatched(actual[0], reference[0])
    gradient = (1e-4, 5e-4) if windowed else (8e-6, 1e-4)
    tolerances = ((3e-5, 3e-5), gradient, gradient)
    for (atol, rtol), value, expected in zip(
        tolerances, actual, reference, strict=True
    ):
        _assert_close_at_scale(value, expected, atol=atol, rtol=rtol)


@_GPU
@pytest.mark.parametrize(
    ("channels", "lmax", "radial_modes", "slots"),
    [(64, 3, 4, True), (128, 3, 4, True), (64, 2, 0, False)],
)
@pytest.mark.parametrize("system", ["unsplit", "coscheduled"])
@pytest.mark.parametrize("slot_types", [1, 3, 6])
def test_spin_systems_of_a_few_types_take_the_slotted_scans(
    channels: int,
    lmax: int,
    radial_modes: int,
    slots: bool,
    system: str,
    slot_types: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spin system of one to eight node types takes the slotted scans.

    Both systems hold the same atoms of the spin-carrying type, and eight
    isolated atoms, which interact with none: in one the isolated atoms carry
    that type, or two or five other types besides it, a system of one, three or
    six types, and in the other the eight other types of the descriptor, a
    system of nine types that the slotted scans do not serve. The node counts
    agree, so every launch choice that follows from them does as well, and the
    rows of both take a warp each. The slotted scans apply the mode mixing of
    each slot to reduced sums, so their forces and magnetic forces differ from
    the pairwise ones in the last bits. A descriptor without radial modes has
    nothing to slot and scans pairwise either way: its outputs agree bit for
    bit. The forces of the system of one type follow the CPU reference of the
    fused operator.
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count = {
        "unsplit": 16 * multiprocessors + 17,
        "coscheduled": 4 * 24 * multiprocessors + 17,
    }[system]
    descriptor = _build_spin_descriptor(
        channels, lmax, radial_modes, type_map=_WIDE_TYPE_MAP
    )
    descriptor._set_compression(build_compression_artifacts(descriptor))
    side = 5.0 * (node_count / 40) ** (1.0 / 3.0)
    if slot_types == 1:
        _check_canonical_against_reference(
            descriptor, False, monkeypatch, node_count=node_count, side=side
        )
    neighbor_graph, atype = _build_graph(
        descriptor, canonical=True, node_count=node_count, side=side
    )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    atype = torch.zeros_like(atype)
    # The isolated atoms close both row-pointer arrays with empty rows.
    isolated = 8
    extended = dataclasses.replace(
        graph,
        n_node=graph.n_node + isolated,
        n_local=graph.n_local + isolated,
        destination_row_ptr=torch.cat(
            [graph.destination_row_ptr] + [graph.destination_row_ptr[-1:]] * isolated
        ),
        source_row_ptr=torch.cat(
            [graph.source_row_ptr] + [graph.source_row_ptr[-1:]] * isolated
        ),
    )
    isolated_types = {
        1: [0, 0, 0, 0, 0, 0, 0, 0],
        3: [1, 2, 0, 0, 0, 0, 0, 0],
        6: [1, 2, 3, 4, 5, 0, 0, 0],
    }[slot_types]
    few_types = torch.cat([atype, atype.new_tensor(isolated_types)])
    all_types = torch.cat([atype, atype.new_tensor([1, 2, 3, 4, 5, 6, 7, 8])])
    generator = torch.Generator(device="cuda").manual_seed(13)
    moments = torch.randn(
        few_types.shape[0], 3, dtype=torch.float32, device="cuda", generator=generator
    )
    fitting = _energy_fitting(descriptor)
    nodes = atype.shape[0]
    slotted = dpa4c_canonical_compress_energy_force(
        descriptor, fitting, extended, few_types, fitting.bias_atom_e[:, 0], moments
    )
    pairwise = dpa4c_canonical_compress_energy_force(
        descriptor, fitting, extended, all_types, fitting.bias_atom_e[:, 0], moments
    )
    scale = max(pairwise[2].abs().max().item(), 1.0)
    torch.testing.assert_close(
        slotted[2][:nodes], pairwise[2][:nodes], atol=2e-5 * scale, rtol=2e-5
    )
    torch.testing.assert_close(
        slotted[5][:nodes], pairwise[5][:nodes], atol=2e-5, rtol=2e-5
    )
    assert torch.equal(slotted[2][:nodes], pairwise[2][:nodes]) is not slots
    assert torch.equal(slotted[5][:nodes], pairwise[5][:nodes]) is not slots


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
@pytest.mark.parametrize("bridged", [False, True])
@pytest.mark.parametrize("spin_conditioned", [False, True])
def test_backward_operator_satisfies_its_schema(
    spin_conditioned: bool,
    bridged: bool,
) -> None:
    """The backward operator declares four independent results.

    All four are unannotated, so any two of them sharing storage would be an
    alias the schema does not describe, which is undefined under
    functionalization. ``opcheck`` decides that mechanically, including on the
    spin-free path without a pair potential, where three of the four are
    absent and an allocation shared between them would otherwise go unnoticed.
    """
    from deepmd.dpmodel.atomic_model.inner_potential import (
        InnerPotential,
    )

    descriptor = _build_spin_descriptor(8) if spin_conditioned else _build_descriptor(8)
    graph, atype = _build_graph(descriptor, canonical=True)
    arguments = _arguments(descriptor, graph, atype)
    pair = (
        (
            torch.as_tensor(InnerPotential(["O", "H"]).pair_table, device="cuda"),
            torch.ones(atype.shape[0], dtype=torch.float64, device="cuda"),
        )
        if bridged
        else (graph.edge_vec.new_empty(0),) * 2
    )
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
        (torch.ones_like(output), state, graph.edge_vec, *arguments, *pair),
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


# === Fitting routes of the fused energy-force operator ===


def _route_fitting(descriptor: DescrptDPA4C, width: int) -> Any:
    """Return a three-layer fp32 SiLU energy fitting of hidden width ``width``."""
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )

    return (
        EnergyFittingNet(
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            neuron=[width] * 3,
            resnet_dt=False,
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=31,
        )
        .cuda()
        .eval()
    )


def _route_case(node_count: int) -> tuple[DescrptDPA4C, Any, torch.Tensor]:
    """Return a compressed descriptor and the canonical graph of ``node_count`` atoms."""
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_neighbor_graph,
    )

    descriptor = _build_descriptor(8, 2, 0)
    descriptor._set_compression(build_compression_artifacts(descriptor))
    neighbor_graph, atype = _build_graph(
        descriptor,
        canonical=True,
        node_count=node_count,
        side=5.0 * (node_count / 40) ** (1.0 / 3.0),
    )
    graph = canonical_graph_from_neighbor_graph(
        dataclasses.replace(neighbor_graph, n_local=neighbor_graph.n_node)
    )
    return descriptor, graph, atype


def _single_route(choice: int) -> list[int]:
    """Return the fitting-route table that names ``choice`` for every run size."""
    from deepmd.pt_expt.kernels.fitting_routes import (
        ROUTE_TABLE_MAGIC,
    )

    fingerprint = torch.ops.deepmd.graph_fitting_route_fingerprint(
        torch.cuda.current_device()
    )
    return [ROUTE_TABLE_MAGIC, *fingerprint, 1, choice]


def _fitting_kernels(call: Callable[[], Any]) -> list[str]:
    """Return the kinds of the fitting kernels that ``call`` launches, sorted.

    A trace that records no CUDA kernel at all fails the check rather than
    reading as a route without fitting kernels.
    """
    from torch.profiler import (
        ProfilerActivity,
        profile,
    )

    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        call()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in trace.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert names, "the profiler recorded no CUDA kernel"
    kinds = set()
    for name in names:
        if "fused_network_kernel" in name:
            kinds.add("fused")
        elif "layer_forward_kernel" in name:
            kinds.add("per-layer")
    return sorted(kinds)


#: Run sizes of the route cases, in multiprocessors of the device: a run of
#: fewer 32-node panels than multiprocessors and one of more.
_ROUTE_RUNS = {"latency": 0.5, "throughput": 3.5}


@_GPU
@pytest.mark.parametrize("width", [64, 96, 192, 256, 384])  # fused hidden width
@pytest.mark.parametrize("run", sorted(_ROUTE_RUNS))  # run size regime
def test_every_fitting_route_returns_the_bits_of_the_rules(
    width: int, run: str
) -> None:
    """Every route of the fitting network gives the outputs of the rules bit for bit.

    A route table that names the per-layer kernels or a fused shape the device
    runs for the run size routes the network there, and every route evaluates
    the same contractions in the same order, so all seven outputs of the
    operator equal those of the rules, which an empty table takes.
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.kernels.fitting_routes import (
        PER_LAYER,
    )

    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    node_count = int(32 * multiprocessors * _ROUTE_RUNS[run])
    descriptor, graph, atype = _route_case(node_count)
    fitting = _route_fitting(descriptor, width)
    bias = fitting.bias_atom_e[:, 0]
    rules = dpa4c_canonical_compress_energy_force(
        descriptor, fitting, graph, atype, bias, fitting_routes=[]
    )
    choices = [
        PER_LAYER,
        *torch.ops.deepmd.graph_fitting_route_choices(
            torch.cuda.current_device(), width, node_count
        ),
    ]
    for choice in choices:
        routed = dpa4c_canonical_compress_energy_force(
            descriptor,
            fitting,
            graph,
            atype,
            bias,
            fitting_routes=_single_route(choice),
        )
        for name, expected, actual in zip(
            (
                "energy",
                "atom_energy",
                "force",
                "virial",
                "atom_virial",
                "force_mag",
                "mask",
            ),
            rules,
            routed,
            strict=True,
        ):
            assert torch.equal(expected, actual), (choice, name)


@_GPU
@pytest.mark.parametrize("width", [96, 192])  # fused hidden width
def test_a_tiled_call_routes_its_runs_and_its_tail_apart(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full runs and the tail of a tiled call take their routes apart.

    The operator routes every run of a tiled call by the run's own size, so a
    table that splits the run sizes at 1,000 nodes sends full runs of 3.5
    waves of panels and a tail of 100 nodes to the per-layer kernels, the
    rules or a fused shape in each combination, and the per-layer buffers
    serve whichever runs take those kernels. Every combination gives the
    outputs of the untiled call under the rules bit for bit.
    """
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.kernels.fitting_routes import (
        PER_LAYER,
        ROUTE_TABLE_MAGIC,
        RULES,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
    )

    device = torch.cuda.current_device()
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    tile, tail = int(32 * multiprocessors * _ROUTE_RUNS["throughput"]), 100
    descriptor, graph, atype = _route_case(2 * tile + tail)
    fitting = _route_fitting(descriptor, width)
    bias = fitting.bias_atom_e[:, 0]
    weights = list(fitting_operator_arguments(fitting).weights)
    fingerprint = torch.ops.deepmd.graph_fitting_route_fingerprint(device)

    def run(table: list[int]) -> tuple[torch.Tensor, ...]:
        return dpa4c_canonical_compress_energy_force(
            descriptor, fitting, graph, atype, bias, fitting_routes=table
        )

    monkeypatch.setenv("DP_NODE_TILE", "0")
    reference = run([])
    monkeypatch.setenv("DP_NODE_TILE", str(tile))
    combinations = [(PER_LAYER, RULES), (RULES, PER_LAYER), (PER_LAYER, PER_LAYER)]
    tail_shapes = torch.ops.deepmd.graph_fitting_route_choices(device, width, tail)
    full_shapes = torch.ops.deepmd.graph_fitting_route_choices(device, width, tile)
    if tail_shapes and full_shapes:
        combinations.append((tail_shapes[-1], full_shapes[0]))
    for tail_route, full_route in combinations:
        table = [ROUTE_TABLE_MAGIC, *fingerprint, 2, 1000, tail_route, full_route]
        for size, route in ((tail, tail_route), (tile, full_route)):
            if route != RULES:
                resolved = torch.ops.deepmd.graph_fitting_route_resolve(
                    weights, size, table
                )
                assert resolved == route, (size, route)
        for actual, expected in zip(run(table), reference, strict=True):
            assert torch.equal(actual, expected), (tail_route, full_route)


def _route_routing_case() -> tuple[DescrptDPA4C, Any, Any, torch.Tensor]:
    """Return the network of width 192 and a run of four waves of its panels."""
    multiprocessors = torch.cuda.get_device_properties("cuda").multi_processor_count
    descriptor, graph, atype = _route_case(32 * multiprocessors * 4)
    return descriptor, _route_fitting(descriptor, 192), graph, atype


def _route_tables() -> dict[str, list[int]]:
    """Return the tables of the routing cases, by name."""
    from deepmd.pt_expt.kernels.fitting_routes import (
        PER_LAYER,
        RULES,
    )

    per_layer = _single_route(PER_LAYER)
    foreign = list(per_layer)
    foreign[2] += 1  # the multiprocessor count of the fingerprint
    return {
        "empty": [],
        "rules": _single_route(RULES),
        "per-layer": per_layer,
        "foreign": foreign,
        "malformed": per_layer[:-1],
        "unknown shape": _single_route(0x7FFF_FFFF_FFFF),
    }


def _traced_routes() -> dict[str, list[str]]:
    """Return the fitting kernels that each routing table launches, by name."""
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )

    descriptor, fitting, graph, atype = _route_routing_case()
    bias = fitting.bias_atom_e[:, 0]
    return {
        name: _fitting_kernels(
            lambda table=table: dpa4c_canonical_compress_energy_force(
                descriptor, fitting, graph, atype, bias, fitting_routes=table
            )
        )
        for name, table in _route_tables().items()
    }


@_GPU
def test_a_fitting_route_table_routes_the_network() -> None:
    """The table decides the route; an unusable table leaves it to the rules.

    At a width of 192 the rules route a run of a few waves of panels to the
    fused kernel. A table that names the per-layer kernels routes it there; a
    table that names the rules, one measured on a device of another
    fingerprint, a malformed one and one that names a shape not compiled for
    the width leave the run to the rules.
    The routes resolve through the query of the operator, and the kernels they
    launch, which a profiler trace in a process of its own shows, follow them.
    """
    from deepmd.pt_expt.kernels.fitting_routes import (
        PER_LAYER,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
    )

    _descriptor, fitting, graph, _atype = _route_routing_case()
    weights = list(fitting_operator_arguments(fitting).weights)
    nodes = graph.destination_row_ptr.numel() - 1
    resolved = {
        name: torch.ops.deepmd.graph_fitting_route_resolve(weights, nodes, table)
        for name, table in _route_tables().items()
    }
    assert resolved == {
        "empty": 0,
        "rules": 0,
        "per-layer": PER_LAYER,
        "foreign": 0,
        "malformed": 0,
        "unknown shape": 0,
    }
    traced = _run_in_process("_traced_routes()")
    assert traced == {
        "empty": ["fused"],
        "rules": ["fused"],
        "per-layer": ["per-layer"],
        "foreign": ["fused"],
        "malformed": ["fused"],
        "unknown shape": ["fused"],
    }


def _route_model(compressed: bool = True) -> Any:
    """Return a DPA4C energy model whose fitting has a hidden width of 192."""
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    model = (
        get_model(
            {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "dpa4c",
                    "rcut": 4.0,
                    "channels": 8,
                    "lmax": 2,
                    "n_radial": 8,
                    "precision": "float32",
                    "seed": 3,
                },
                "fitting_net": {
                    "type": "ener",
                    "neuron": [192, 192, 192],
                    "activation_function": "silu",
                    "precision": "float32",
                    "resnet_dt": False,
                    "seed": 3,
                },
            }
        )
        .to("cpu")
        .eval()
    )
    if compressed:
        model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    return model


def _frozen_route_kernels(directory: str) -> dict[str, Any]:
    """Freeze the routing model under two held tables; return what each archive does.

    The tuner skips a network whose table it holds, so a table set beforehand
    stands for a measured one. Each archive is frozen to ``directory`` and run
    through AOTInductor, as a deployment runs it. Returns, for the table that
    names the per-layer kernels and for no table, the table argument of the
    fused operator in the exported program and the fitting kernels that the
    archive launches.
    """
    from torch._inductor import (
        aoti_load_package,
    )

    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_index_bytes,
    )
    from deepmd.pt_expt.utils.serialization import (
        _trace_and_export,
        deserialize_to_file,
    )

    os.environ["DP_CUDA_INFER"] = "2"
    model = _route_model()
    learned = model.atomic_model.fused_decomposition()[0]
    key = fitting_routes.network_key(learned.fitting_net)
    device = torch.device("cuda", torch.cuda.current_device())
    fingerprint = fitting_routes._fingerprint(device)
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    node_count = 32 * multiprocessors * 4
    graph = fitting_routes._bulk_graph(4.0, node_count, device)
    atype = torch.arange(node_count, device=device) % 2
    inputs = (
        atype,
        graph.n_node,
        graph.n_local,
        canonical_index_bytes(graph.source),
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
    )
    results = {}
    for name, table in (
        ("per-layer", _single_route(fitting_routes.PER_LAYER)),
        ("rules", []),
    ):
        fitting_routes._ROUTE_TABLES[(fingerprint, key)] = tuple(table)
        exported, _metadata, _model_json, _output_keys = _trace_and_export(
            {"model": model.serialize()},
            lower_kind="dpa4c_canonical",
            do_atomic_virial=True,
        )
        # The table is the last argument of the schema; a call that leaves it
        # to its default of no table records one argument fewer.
        schema = torch.ops.deepmd.dpa4c_canonical_compress_energy_force.default._schema
        literals = [
            node.args[-1] if len(node.args) == len(schema.arguments) else None
            for node in exported.graph_module.graph.nodes
            if "dpa4c_canonical_compress_energy_force" in str(node.target)
        ]
        archive = os.path.join(directory, f"{name}.pt2")
        deserialize_to_file(
            archive,
            {"model": model.serialize()},
            do_atomic_virial=True,
            lower_kind="dpa4c_canonical",
            with_comm=False,
        )
        runner = aoti_load_package(archive)
        results[name] = {
            "literals": [None if v is None else [int(x) for x in v] for v in literals],
            "kernels": _fitting_kernels(lambda runner=runner: runner(*inputs)),
        }
    return results


@_GPU
def test_the_measured_routes_reach_the_frozen_program(tmp_path: pathlib.Path) -> None:
    """The table held for a network routes the frozen archive of the model.

    The export of the compact canonical lower takes the held table as an
    argument of the fused operator, or no table where none is held, and the
    frozen archive, run through AOTInductor, routes the network as the table
    names: through the per-layer kernels under a table that names them, and
    through the fused kernel, which the rules choose for the run, without a
    table. The archives run in a process of their own, whose profiler trace
    shows the kernels.
    """
    from deepmd.pt_expt.kernels.fitting_routes import (
        PER_LAYER,
    )

    results = _run_in_process(f"_frozen_route_kernels({str(tmp_path)!r})")
    assert results == {
        "per-layer": {
            "literals": [_single_route(PER_LAYER)],
            "kernels": ["per-layer"],
        },
        "rules": {"literals": [None], "kernels": ["fused"]},
    }


@_GPU
def test_the_route_tuner_leaves_other_models_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tuner measures nothing for a model off the fused DPA4C operator.

    Nor does a freeze of a model that takes the operator measure anything for
    a lower that does not call it.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )
    from deepmd.pt_expt.kernels.autotune import (
        run_autotune,
    )

    tables: dict = {}
    monkeypatch.setattr(fitting_routes, "_ROUTE_TABLES", tables)
    device = torch.device("cuda", torch.cuda.current_device())
    for model in (_route_model(compressed=False), torch.nn.Linear(2, 2)):
        fitting_routes.tune_fitting_routes(model, 0, device)
    run_autotune(_route_model(), device, "graph")
    assert tables == {}
    descriptor = _build_descriptor(8, 2, 0)
    assert fitting_routes.fitting_routes(_route_fitting(descriptor, 96)) == []


@_GPU
def test_a_measured_table_names_only_its_departures_from_the_rules() -> None:
    """A measured table leaves the rules every run size that they route as measured.

    The measurement runs on the bands of the device with a stand-in for the
    timing, under which the per-layer kernels win from a run size inside one
    band of throughput runs to the end of the next band, and the route of the
    rules wins at every other run size, the per-layer kernels of the smallest
    runs among them. The table holds the rules wherever that winner is the
    route that graph_fitting_route_rule gives, and the per-layer kernels over
    the departure, whose start inside its band the measurement bisects for.
    The rules and the stand-in change their route only at the probed sizes,
    the bounds of the bands and of the departure.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
    )

    fitting = _route_fitting(_build_descriptor(8, 2, 0), 192)
    ws = list(fitting_operator_arguments(fitting).weights)
    device = torch.device("cuda", torch.cuda.current_device())
    largest, block = fitting_routes.MAX_RUN_NODES, fitting_routes.BLOCK
    bands = [
        int(bound)
        for bound in torch.ops.deepmd.graph_fitting_route_bands(
            device.index, int(ws[0].shape[1]), int(ws[0].shape[0]), largest
        )
    ]
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    band = next(
        index
        for index in range(1, len(bands) - 1)
        if bands[index - 1] >= 32 * multiprocessors
        and bands[index] - bands[index - 1] >= 4 * block
    )
    start = block * (math.ceil((bands[band - 1] + 1) / block) + 1)
    end = bands[band + 1]

    def winner(node_count: int, rule: int, choices: list[int]) -> int:
        return fitting_routes.PER_LAYER if start < node_count <= end else rule

    bounds, choices = fitting_routes._measure(
        types.SimpleNamespace(device=device, winner=winner), ws, largest
    )
    probes = {1, *(size + d for size in (*bands, start, end) for d in (0, 1))}
    for node_count in sorted(size for size in probes if size <= largest):
        rule = int(torch.ops.deepmd.graph_fitting_route_rule(ws, node_count))
        fastest = winner(node_count, rule, [])
        recorded = next(
            choice
            for bound, choice in zip([*bounds, largest], choices, strict=True)
            if node_count <= bound
        )
        expected = fitting_routes.RULES if fastest == rule else fastest
        assert recorded == expected, node_count
    assert fitting_routes.PER_LAYER in choices


def _route_stand_in(
    alternating: dict[int, float | tuple[float, ...]],
    trains: dict[str, dict[int, tuple]],
    type_count: int = 3,
) -> tuple[Any, list[int], list[tuple[str, int]]]:
    """Return a route bench whose timings are those of the given stand-ins.

    ``alternating`` holds the replay time of every route in the alternating
    replays of a sample, one time for every capture or one per capture after
    the untimed round. ``trains`` holds the times of the trains of one route
    at a time, one per capture, for each composition: every node of one type
    (``"one type"``) and the node types of the sample (``"sample"``), drawn
    among ``type_count`` types. The two lists returned with the bench record
    the routes of the alternating replays, the untimed round included, and the
    compositions and routes of the trains, in the order in which the bench
    times them.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    replayed: list[int] = []
    trained: list[tuple[str, int]] = []
    calls = 0

    def replay(graphs: dict, replays: int, times: dict[int, list[float]]) -> None:
        nonlocal calls
        # The first call of a sample is its untimed round, which takes the
        # times of the first capture; call k > 0 times capture k - 1.
        capture = max(calls - 1, 0)
        calls += 1
        for choice in graphs:
            replayed.append(choice)
            value = alternating[choice]
            ms = value[capture] if isinstance(value, tuple) else value
            times[choice].extend([ms] * replays)

    def capture(inputs: tuple, choices: list[int], rotation: int) -> dict:
        nodes = "sample" if bool(inputs[1].any()) else "one type"
        return {choice: (nodes, choice) for choice in choices}

    def train_ms(graph: tuple[str, int]) -> float:
        trained.append(graph)
        nodes, choice = graph
        return remaining[nodes][choice].pop(0)

    remaining = {
        nodes: {choice: list(times) for choice, times in routes.items()}
        for nodes, routes in trains.items()
    }
    bench = object.__new__(fitting_routes._Bench)
    bench.samples = bench.checked = bench.overturned = bench.replays = 0
    bench.types = type_count
    bench.one_type = torch.zeros(64, dtype=torch.long)
    bench._inputs = lambda node_count: (None, torch.arange(64) % type_count, None)
    bench._capture = capture
    bench._replay_ms = lambda graph: 5.0
    bench._replay = replay
    bench._train_ms = train_ms
    return bench, replayed, trained


def _alternating_ms(*gains: float) -> tuple[float, ...]:
    """Return per-capture replay times whose gains over 1 ms are ``gains``, in percent."""
    return tuple(1.0 / (1.0 + gain / 100.0) for gain in gains)


def _train_order(nodes: str, rule: int, route: int) -> list[tuple[str, int]]:
    """Return the trains of the two captures of one composition, in their order."""
    return [(nodes, choice) for choice in (rule, route, route, rule)]


@pytest.mark.parametrize(
    ("alternating_ms", "one_type_ms", "sample_ms", "per_layer_wins", "compositions"),
    [
        (1.05, (0.95, 0.95), (0.95, 0.95), False, 0),
        (0.95, (1.10, 1.10), (0.95, 0.95), False, 1),
        (0.95, (1.00, 1.04), (0.95, 0.95), False, 1),
        (0.97, (0.95, 0.95), (1.02, 1.02), False, 2),
        (0.97, (1.005, 1.005), (1.008, 1.008), True, 2),
        (0.97, (0.98, 1.03), (1.00, 1.00), True, 2),
    ],
    ids=[
        "slower-alternating",
        "slower-one-type",
        "median-beyond-the-tie",
        "slower-sample",
        "within-the-tie",
        "median-within-the-tie",
    ],
)
def test_a_departure_holds_on_trains_of_both_node_compositions(
    alternating_ms: float,
    one_type_ms: tuple,
    sample_ms: tuple,
    per_layer_wins: bool,
    compositions: int,
) -> None:
    """A route that wins the alternating replays departs where it also holds on trains.

    The stand-in times the route of the rules at 1 ms per replay and the
    per-layer kernels at ``alternating_ms`` in the alternating replays of a
    sample. On the trains of one route at a time, two captures for each
    composition, it times the route of the rules at 1 ms and the per-layer
    kernels at ``one_type_ms`` with every node of one type and at
    ``sample_ms`` with the node types of the sample. A board at its power limit
    can make a route that wins the alternating replays lose on trains, by an
    amount that depends on the composition, because it sets the clock of a
    route from the power that the route draws, which the node types of the
    system change.

    A route that clearly loses the alternating replays is settled after one
    capture and never timed on trains, and a route that wins them is timed on
    three captures at least. It departs from the rules where its median gain
    on the trains of each composition is at least -1%, the tie, and the check
    stops at the first composition on which it loses by more. The two captures
    of a composition time the two routes in opposite orders.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    rule, per_layer = 7, fitting_routes.PER_LAYER
    bench, replayed, trained = _route_stand_in(
        {per_layer: alternating_ms, rule: 1.0},
        {
            "one type": {per_layer: one_type_ms, rule: (1.0, 1.0)},
            "sample": {per_layer: sample_ms, rule: (1.0, 1.0)},
        },
    )
    expected = per_layer if per_layer_wins else rule
    assert bench.winner(4096, rule, [per_layer, rule]) == expected
    assert bench.samples == 1
    # The untimed round and the captures of the sample, two routes each.
    captures = 1 if alternating_ms > 1.0 else 3
    assert len(replayed) == 2 * (1 + captures)
    checked = int(compositions > 0)
    assert (bench.checked, bench.overturned) == (checked, checked - per_layer_wins)
    order = _train_order("one type", rule, per_layer) + _train_order(
        "sample", rule, per_layer
    )
    assert trained == order[: 4 * compositions]


@pytest.mark.parametrize(
    ("per_layer_gains", "shape_gains", "expected"),
    [
        ((0.5,) * 7, None, "rules"),
        ((2.0, 2.0, 10.0, 2.0, 2.0, 10.0, 2.0), None, "rules"),
        ((6.0, 5.0, 0.0, 6.0, 5.0, 0.0, 5.0), None, "per-layer"),
        ((2.5, 2.5, -6.0, 2.5, 2.5, -6.0, 2.5), (2.0,) * 7, "shape"),
    ],
    ids=[
        "within-the-tie",
        "within-the-half-range",
        "beyond-the-half-range",
        "runner-up-of-a-wide-best",
    ],
)
def test_a_candidate_exceeds_its_own_bound(
    per_layer_gains: tuple, shape_gains: tuple | None, expected: str
) -> None:
    """A route is a candidate where its median gain exceeds its own bound.

    The stand-in times the route of the rules at 1 ms per replay in the
    alternating replays of a sample and the per-layer kernels, and in the
    last case a fused shape, at the given per-capture gains over it, in
    percent; the trains time every route at 1 ms. The bound of a route is the
    tie of 1%, or the half-range of its own per-capture gains where that is
    larger. A steady gain of 0.5% within the tie and a median gain of 2% within
    a half-range of 4% leave the run size to the rules, a median gain of 5%
    beyond a half-range of 3% departs, and a fused shape of a steady 2% departs
    where the per-layer kernels do not, although their median gain of 2.5% is
    larger, since the half-range of their gains is 4.25%.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    rule, shape, per_layer = 7, 5, fitting_routes.PER_LAYER
    alternating = {per_layer: _alternating_ms(*per_layer_gains), rule: 1.0}
    if shape_gains is not None:
        alternating[shape] = _alternating_ms(*shape_gains)
    level = dict.fromkeys((per_layer, shape, rule), (1.0,) * 4)
    bench, _, _ = _route_stand_in(alternating, {"one type": level, "sample": level})
    routes = {"rules": rule, "per-layer": per_layer, "shape": shape}
    assert bench.winner(4096, rule, [*alternating]) == routes[expected]
    assert bench.checked == int(expected != "rules")


@pytest.mark.parametrize(
    ("per_layer_gains", "shape_gains", "expected", "stages"),
    [
        (
            (2.2, 2.2, -6.0, 0.0, 0.0, 0.0, 0.0),
            (1.9, 2.0, 2.1, 0.0, 0.0, 0.0, 0.0),
            "rules",
            3,
        ),
        ((6.0,) * 7, (-6.0,) * 7, "per-layer", 2),
        ((-6.0,) * 7, (6.0,) * 7, "shape", 2),
    ],
    ids=[
        "runner-up-near-its-bound",
        "departure-of-the-first-route",
        "departure-of-the-second-route",
    ],
)
def test_a_sample_settles_every_rival_before_it_stops(
    per_layer_gains: tuple, shape_gains: tuple, expected: str, stages: int
) -> None:
    """A sample takes captures until every route besides the rules' is settled.

    The stand-in times the route of the rules at 1 ms per replay and the
    per-layer kernels and a fused shape, given to the bench in that order, at
    the given per-capture gains over it, in percent; the trains time every
    route at 1 ms. In the first case, after three captures the per-layer
    kernels, whose median gain is the largest, lie clearly below their bound,
    a half-range of 4.1%, while the fused shape lies 1% above its bound,
    within the margin of that stage: the sample takes the captures of the
    third stage, over which both routes tie with the rules, and leaves the run
    size to them. In the other two cases one route lies clearly below its
    bound after one capture and the other clearly above its own, the first
    route given to the bench in one case and the second in the other: the
    sample still takes the captures of the second stage before the route
    above its bound departs.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    rule, shape, per_layer = 7, 5, fitting_routes.PER_LAYER
    level = dict.fromkeys((per_layer, shape, rule), (1.0,) * 4)
    bench, replayed, _ = _route_stand_in(
        {
            per_layer: _alternating_ms(*per_layer_gains),
            shape: _alternating_ms(*shape_gains),
            rule: 1.0,
        },
        {"one type": level, "sample": level},
    )
    routes = {"rules": rule, "per-layer": per_layer, "shape": shape}
    assert bench.winner(4096, rule, [per_layer, shape, rule]) == routes[expected]
    # The untimed round and the captures of the stages, three routes each.
    captures = sum(fitting_routes.STAGE_CAPTURES[:stages])
    assert len(replayed) == 3 * (1 + captures)


@pytest.mark.parametrize(
    ("one_type_ms", "expected"),
    [((1.03, 0.97), 1), ((1.03, 1.02), None), ((1.03, 1.03, 0.99), 2)],
    ids=["second-holds", "none-holds", "third-holds"],
)
def test_the_candidates_are_tried_in_the_order_of_their_gains(
    one_type_ms: tuple[float, ...], expected: int | None
) -> None:
    """The candidates of a sample are checked on trains in the order of their gains.

    The stand-in times the per-layer kernels and one or two fused shapes at
    0.95, 0.96 and 0.97 ms in the alternating replays of a sample, against
    1 ms for the route of the rules, so that every one exceeds the tie and
    their gains fall in that order; the bench receives the routes in the
    reverse order. On the trains with every node of one type candidate ``i``
    takes ``one_type_ms[i]`` against 1 ms for the route of the rules, so that
    the per-layer kernels lose 3% there, and with the node types of the sample
    every candidate takes 0.97 ms. The first candidate that holds on both
    compositions departs from the rules, and the rules keep the run size where
    none holds; the sample counts as overturned either way.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    rule = 7
    candidates = [fitting_routes.PER_LAYER, 5, 6][: len(one_type_ms)]
    alternating = {choice: 0.95 + 0.01 * i for i, choice in enumerate(candidates)}
    one_type = dict(zip(candidates, ((ms, ms) for ms in one_type_ms), strict=True))
    bench, _, trained = _route_stand_in(
        {**alternating, rule: 1.0},
        {
            "one type": {**one_type, rule: (1.0,) * (2 * len(candidates))},
            "sample": {**dict.fromkeys(candidates, (0.97, 0.97)), rule: (1.0, 1.0)},
        },
    )
    winner = rule if expected is None else candidates[expected]
    assert bench.winner(4096, rule, [*reversed(candidates), rule]) == winner
    assert (bench.checked, bench.overturned) == (1, 1)
    tried = candidates if expected is None else candidates[: expected + 1]
    order = [train for c in tried for train in _train_order("one type", rule, c)]
    if expected is not None:
        order += _train_order("sample", rule, winner)
    assert trained == order


@pytest.mark.parametrize(
    ("rule_ms", "rule_train_ms", "per_layer_wins", "compositions"),
    [(1.125, 1.0, False, 0), (1.25, 0.875, True, 2)],
    ids=["gain-at-its-bound", "train-gain-at-the-tie"],
)
def test_the_bound_and_the_tie_are_strict_and_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    rule_ms: float,
    rule_train_ms: float,
    per_layer_wins: bool,
    compositions: int,
) -> None:
    """A gain equal to its bound is no candidate, and a train loss equal to the tie holds.

    With a tie of 1/8, which the times of the stand-in represent exactly, the
    per-layer kernels take 1 ms in the alternating replays and on the trains.
    A route of the rules at 1.125 ms gives them a steady gain of exactly the
    tie, their bound, which a candidate has to exceed, so the run size stays
    with the rules. A route of the rules at 1.25 ms in the alternating replays
    and at 0.875 ms on the trains gives them a candidacy and a median gain of
    exactly minus the tie on the trains of both compositions, at which a
    candidate holds.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    monkeypatch.setattr(fitting_routes, "TIE", 0.125)
    rule, per_layer = 7, fitting_routes.PER_LAYER
    level = {per_layer: (1.0, 1.0), rule: (rule_train_ms, rule_train_ms)}
    bench, _, trained = _route_stand_in(
        {per_layer: 1.0, rule: rule_ms}, {"one type": level, "sample": level}
    )
    expected = per_layer if per_layer_wins else rule
    assert bench.winner(4096, rule, [per_layer, rule]) == expected
    assert len(trained) == 4 * compositions


def test_a_margin_equal_to_that_of_its_stage_takes_the_next_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sample decides only where every margin lies strictly beyond that of its stage.

    With a tie of 1/8 and stage margins of 1/4 and 1/8, which the times of the
    stand-in represent exactly, the route of the rules takes 1.25 ms and the
    per-layer kernels 1 ms per replay, a steady gain of 1/4. Its margin, the
    gain less its bound, is 1/8 after every stage. The first stage does not
    decide, since the route exceeds its bound, and the second does not either,
    since the margin equals that of the stage, so the sample takes the
    captures of the last stage before the per-layer kernels depart.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    monkeypatch.setattr(fitting_routes, "TIE", 0.125)
    monkeypatch.setattr(fitting_routes, "STAGE_MARGINS", (0.25, 0.125))
    rule, per_layer = 7, fitting_routes.PER_LAYER
    level = {per_layer: (1.0, 1.0), rule: (1.0, 1.0)}
    bench, replayed, _ = _route_stand_in(
        {per_layer: 1.0, rule: 1.25}, {"one type": level, "sample": level}
    )
    assert bench.winner(4096, rule, [per_layer, rule]) == per_layer
    # The untimed round and the captures of the three stages, two routes each.
    assert len(replayed) == 2 * (1 + sum(fitting_routes.STAGE_CAPTURES))


def test_the_candidates_are_ordered_by_their_gain_rather_than_their_margin() -> None:
    """Of two candidates, the one of the larger median gain is checked first.

    The stand-in times the route of the rules at 1 ms per replay, the
    per-layer kernels at per-capture gains of 6%, 5% and 0%, a median gain of
    5% above a bound of 3%, and a fused shape at a steady gain of 4%, above a
    bound of 1%, so that the fused shape lies farther above its bound while
    the per-layer kernels gain more. The trains time every route at 1 ms, so
    both candidates hold there, and the per-layer kernels, checked first, take
    the run size.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    rule, shape, per_layer = 7, 5, fitting_routes.PER_LAYER
    level = dict.fromkeys((per_layer, shape, rule), (1.0,) * 4)
    bench, _, trained = _route_stand_in(
        {
            per_layer: _alternating_ms(6.0, 5.0, 0.0, 6.0, 5.0, 0.0, 5.0),
            shape: _alternating_ms(*(4.0,) * 7),
            rule: 1.0,
        },
        {"one type": level, "sample": level},
    )
    assert bench.winner(4096, rule, [shape, per_layer, rule]) == per_layer
    assert trained == _train_order("one type", rule, per_layer) + _train_order(
        "sample", rule, per_layer
    )


@_GPU
@pytest.mark.parametrize(
    ("train_span_ms", "floor"), [(10.0, False), (0.5, True)], ids=["span", "floor"]
)
def test_a_train_follows_its_lead_in_and_returns_the_time_of_one_replay(
    monkeypatch: pytest.MonkeyPatch, train_span_ms: float, floor: bool
) -> None:
    """A train replays its route for the lead-in, then times a span of replays.

    Captured CUDA graphs of sleep kernels stand in for the operator: one
    replays during the lead-in, whose replays the bench times through
    ``_replay_ms``, and one of four times its length during the train, so that
    the time that the train returns tells its own replays from those of the
    lead-in also where other processes take time slices of a shared device and
    so lengthen the lead-in. With a lead-in of 20 ms of device time in chunks of
    5 ms, the first replay sets the size of the chunks, the lead-in replays the
    route in chunks until their device time reaches 20 ms, and the train then
    takes the replays that span ``train_span_ms`` at the mean replay time of
    the lead-in, nine at least: a span of 10 ms takes more than nine replays, a
    span of 0.5 ms nine. The bench counts every replay, and the train returns
    the median time of its own replays, which the test compares with the
    median of single replays of the same graph; a median, unlike a mean, stays
    in place under the time slices of other processes.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    monkeypatch.setattr(fitting_routes, "LEAD_IN_MS", 20.0)
    monkeypatch.setattr(fitting_routes, "LEAD_IN_CHUNK_MS", 5.0)
    monkeypatch.setattr(fitting_routes, "TRAIN_MS", train_span_ms)
    device = torch.device("cuda", torch.cuda.current_device())

    class Counted:
        """Replays the lead-in graph or the train graph and counts the replays."""

        def __init__(
            self, lead_in: torch.cuda.CUDAGraph, train: torch.cuda.CUDAGraph
        ) -> None:
            self.lead_in, self.train = lead_in, train
            self.leading, self.count = False, 0

        def replay(self) -> None:
            self.count += 1
            (self.lead_in if self.leading else self.train).replay()

    bench = object.__new__(fitting_routes._Bench)
    bench.device, bench.replays = device, 0
    timed: list[tuple[int, float]] = []
    replay_ms = bench._replay_ms

    def lead_in_replay_ms(graph: Counted, count: int = 1) -> float:
        graph.leading = True
        timed.append((count, replay_ms(graph, count)))
        graph.leading = False
        return timed[-1][1]

    bench._replay_ms = lead_in_replay_ms
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000)
        torch.cuda.synchronize(device)
        graphs = []
        for cycles in (250_000, 1_000_000):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                torch.cuda._sleep(cycles)
            graphs.append(graph)
        counted = Counted(*graphs)
        train_ms = bench._train_ms(counted)
        single_ms = statistics.median(replay_ms(graphs[1]) for _ in range(15))
    counts, times = zip(*timed, strict=True)
    chunk = max(1, math.ceil(fitting_routes.LEAD_IN_CHUNK_MS / times[0]))
    assert counts == (1, *(chunk,) * (len(counts) - 1))
    assert sum(times[:-1]) < fitting_routes.LEAD_IN_MS <= sum(times)
    span = math.ceil(fitting_routes.TRAIN_MS * sum(counts) / sum(times))
    if floor:
        assert span < fitting_routes.TRAIN_REPLAYS
    else:
        assert span > fitting_routes.TRAIN_REPLAYS
    assert counted.count - sum(counts) == max(fitting_routes.TRAIN_REPLAYS, span)
    assert bench.replays == counted.count
    assert 0.8 * single_ms <= train_ms <= 1.25 * single_ms


@pytest.mark.parametrize("type_count", [1, 2], ids=["one-type", "two-types"])
def test_the_trains_time_each_distinct_composition_once(type_count: int) -> None:
    """The trains time each distinct composition of a sample once.

    The node types of the sample of a model of one node type are those of the
    composition with every node of one type, so the bench times that
    composition alone; the sample of a model of two node types holds both
    types, and the bench times both compositions.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    rule, per_layer = 7, fitting_routes.PER_LAYER
    level = {per_layer: (0.95, 0.95), rule: (1.0, 1.0)}
    bench, _, trained = _route_stand_in(
        {per_layer: 0.95, rule: 1.0},
        {"one type": level, "sample": level},
        type_count=type_count,
    )
    assert bench.winner(4096, rule, [per_layer, rule]) == per_layer
    assert (bench.checked, bench.overturned) == (1, 0)
    order = _train_order("one type", rule, per_layer)
    if type_count > 1:
        order += _train_order("sample", rule, per_layer)
    assert trained == order


@pytest.mark.parametrize(
    ("one_type_ms", "sample_ms", "second_wins"),
    [
        ((1.10, 1.10), (1.0, 1.0), True),
        ((1.012, 0.995), (1.0, 1.0), True),
        ((0.995, 0.995), (1.03, 1.03), True),
        ((0.97, 0.97), (1.0, 1.0), False),
        ((1.02, 1.02), (0.95, 0.95), False),
        ((1.0, 1.0), (1.0, 1.0), False),
    ],
    ids=["clear", "within-range", "mean", "first-faster", "mean-against", "tie"],
)
def test_two_routes_besides_the_rules_are_decided_on_trains(
    one_type_ms: tuple, sample_ms: tuple, second_wins: bool
) -> None:
    """Two routes that are not the rules' are decided on trains alone.

    The bisection of a band compares the routes that won at its two ends,
    neither of them that of the rules, and the bench runs no alternating
    replays for such a pair. The stand-in times the per-layer kernels, the
    first route, faster than a fused shape in the alternating replays, and on
    the trains at ``one_type_ms`` with every node of one type and at
    ``sample_ms`` with the node types of the sample, against 1 ms for the
    fused shape. The fused shape takes the run size where the mean over the two
    compositions of its median gains over the per-layer kernels is positive,
    and the per-layer kernels keep it otherwise, a tie included.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )

    rule, shape, per_layer = 7, 5, fitting_routes.PER_LAYER
    bench, replayed, trained = _route_stand_in(
        {per_layer: 0.95, shape: 1.0},
        {
            "one type": {per_layer: one_type_ms, shape: (1.0, 1.0)},
            "sample": {per_layer: sample_ms, shape: (1.0, 1.0)},
        },
    )
    expected = shape if second_wins else per_layer
    assert bench.winner(4096, rule, [per_layer, shape]) == expected
    assert (bench.samples, bench.checked, bench.overturned) == (1, 1, 0)
    assert replayed == []
    assert trained == _train_order("one type", per_layer, shape) + _train_order(
        "sample", per_layer, shape
    )


@_GPU
def test_the_route_tuner_measures_a_table_that_changes_no_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tuner holds a table of the device under which no output changes.

    The routing model is measured over the run sizes of a node tile of 16,384.
    The table carries the fingerprint of the device and increasing bounds
    below the tile, and names on every interval the rules or a route that the
    operator takes there. On a bulk system at the upper end of every interval,
    the operator gives the outputs of the rules bit for bit under it.
    """
    from deepmd.pt_expt.kernels import (
        fitting_routes,
    )
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        dpa4c_canonical_compress_energy_force,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
    )

    tables: dict = {}
    monkeypatch.setattr(fitting_routes, "_ROUTE_TABLES", tables)
    monkeypatch.setenv("DP_NODE_TILE", "16384")
    device = torch.device("cuda", torch.cuda.current_device())
    model = _route_model()
    fitting_routes.tune_fitting_routes(model, 0, device)
    (table,) = tables.values()
    table = list(table)
    fingerprint = list(fitting_routes._fingerprint(device))
    header = 1 + len(fingerprint)
    assert table[:header] == [fitting_routes.ROUTE_TABLE_MAGIC, *fingerprint]
    intervals = table[header]
    bounds = table[header + 1 : header + intervals]
    choices = table[header + intervals :]
    assert len(choices) == intervals
    assert bounds == sorted(set(bounds)) and all(0 < b < 16384 for b in bounds)
    learned = model.atomic_model.fused_decomposition()[0]
    descriptor = learned.descriptor.to(device).eval()
    fitting = learned.fitting_net.to(device).eval()
    weights = list(fitting_operator_arguments(fitting).weights)
    bias = fitting.bias_atom_e[:, 0]
    for node_count, choice in zip([*bounds, 16384], choices, strict=True):
        if choice != fitting_routes.RULES:
            resolved = torch.ops.deepmd.graph_fitting_route_resolve(
                weights, node_count, table
            )
            assert resolved == choice, node_count
        graph = fitting_routes._bulk_graph(4.0, node_count, device)
        atype = torch.arange(node_count, device=device) % 2
        rules = dpa4c_canonical_compress_energy_force(
            descriptor, fitting, graph, atype, bias, fitting_routes=[]
        )
        routed = dpa4c_canonical_compress_energy_force(
            descriptor, fitting, graph, atype, bias, fitting_routes=table
        )
        for expected, actual in zip(rules, routed, strict=True):
            assert torch.equal(expected, actual), node_count
