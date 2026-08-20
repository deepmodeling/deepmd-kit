# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numerical contract of the compressed DPA4C CPU operators.

The CPU kernels evaluate the same compressed equations as the CUDA ones and
reduce in a different order, so the contract is a tolerance against the
portable reference rather than a bitwise identity. Every structural parameter
the kernels specialize on -- the scalar width, the angular degree, the mode
rank, and the two topology forms -- is covered, because each selects a
different compiled body or a different addressing mode.
"""

import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
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
    _reference_descriptor,
    build_compression_artifacts,
    descriptor_profile,
    ensure_registered,
    op_available,
)
from deepmd.pt_expt.kernels.edge_force_virial import (
    edge_force_virial,
)
from deepmd.pt_expt.kernels.edge_force_virial import op_available as force_op_available
from deepmd.pt_expt.kernels.graph_fitting import op_available as fitting_op_available
from deepmd.pt_expt.kernels.utils import (
    backend_device_type,
)

_CPU = pytest.mark.skipif(
    backend_device_type() != "cpu" or not op_available(),
    reason="the CPU backend and the compiled DPA4C CPU operator are required",
)
_CPU_FORCE = pytest.mark.skipif(
    backend_device_type() != "cpu" or not force_op_available(),
    reason="the CPU backend and the compiled force operator are required",
)
_CPU_FITTING = pytest.mark.skipif(
    backend_device_type() != "cpu" or not fitting_op_available(),
    reason="the CPU backend and the compiled fitting operator are required",
)


def _build_descriptor(
    channels: int,
    lmax: int = 2,
    radial_modes: int = 0,
) -> DescrptDPA4C:
    return DescrptDPA4C(
        rcut=3.0,
        ntypes=2,
        channels=channels,
        lmax=lmax,
        n_radial=8,
        radial_modes=radial_modes,
        precision="float32",
        seed=17,
    ).eval()


def _build_graph(descriptor: DescrptDPA4C, canonical: bool, node_count: int = 24):
    generator = torch.Generator().manual_seed(23)
    coordinate = 5.0 * torch.rand(
        1,
        node_count,
        3,
        dtype=torch.float32,
        generator=generator,
    )
    atype = torch.arange(node_count).reshape(1, -1) % 2
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coordinate,
        atype,
        descriptor.rcut,
        [48],
        mixed_types=True,
        box=None,
    )
    graph, flat_type = graph_from_dense_quartet(coord_ext, atype_ext, nlist, mapping)
    return attach_edge_csr(graph, flat_type.shape[0], canonicalize=canonical), flat_type


def _arguments(descriptor: DescrptDPA4C, graph, atype: torch.Tensor) -> tuple:
    ensure_registered()
    artifacts = build_compression_artifacts(descriptor, stride=0.01)
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
    """Drop the native spin block, which the CPU kernel does not implement."""
    return (*arguments[:15], *arguments[18:])


def _check_parity(descriptor: DescrptDPA4C, canonical: bool) -> None:
    graph, atype = _build_graph(descriptor, canonical)
    arguments = _arguments(descriptor, graph, atype)
    output, state = torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)
    assert state.shape == (
        atype.shape[0],
        descriptor_profile(descriptor.channels, descriptor.lmax).state_width,
    )

    reference_edge = graph.edge_vec.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        reference = _reference_descriptor(reference_edge, *_spin_free(arguments))
    torch.testing.assert_close(output, reference, atol=3.0e-5, rtol=3.0e-5)

    cotangent = torch.linspace(-0.7, 1.3, output.numel()).reshape_as(output)
    (reference_gradient,) = torch.autograd.grad(
        (reference * cotangent).sum(), reference_edge
    )
    gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        cotangent, state, graph.edge_vec, *arguments
    )[0]
    torch.testing.assert_close(gradient, reference_gradient, atol=8.0e-6, rtol=1.0e-4)


@_CPU
@pytest.mark.parametrize("channels", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("canonical", [False, True])
def test_forward_backward_parity(channels: int, canonical: bool) -> None:
    """Both topology forms reproduce the portable compressed descriptor."""
    _check_parity(_build_descriptor(channels), canonical)


@_CPU
@pytest.mark.parametrize("lmax", [2, 3, 4])
@pytest.mark.parametrize("radial_modes", [0, 2, 4, 8])
def test_degree_and_mode_parity(lmax: int, radial_modes: int) -> None:
    """Every compiled angular degree and mode rank reproduces the reference."""
    _check_parity(_build_descriptor(32, lmax=lmax, radial_modes=radial_modes), True)


@_CPU
def test_masked_edges_are_ignored() -> None:
    """A masked edge contributes nothing and receives a zero cotangent.

    The mask is the only reason a CSR row may address an edge the descriptor
    must not read, so the two paths through it are asserted directly rather
    than left to the size of a tolerance.
    """
    descriptor = _build_descriptor(32)
    graph, atype = _build_graph(descriptor, canonical=False)
    arguments = _arguments(descriptor, graph, atype)
    reference, _ = torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)

    padded = torch.cat([graph.edge_vec, graph.edge_vec[:8]])
    mask = torch.cat([graph.edge_mask, torch.zeros(8, dtype=torch.bool)])
    index = torch.cat([graph.edge_index, graph.edge_index[:, :8]], dim=1)
    order = torch.cat(
        [graph.destination_order, torch.arange(8) + graph.edge_vec.shape[0]]
    )
    extended = (index, mask, order, *arguments[3:])
    output, state = torch.ops.deepmd.dpa4c_graph_compress(padded, *extended)
    torch.testing.assert_close(output, reference)

    cotangent = torch.ones_like(output)
    gradient = torch.ops.deepmd.dpa4c_graph_compress_backward(
        cotangent, state, padded, *extended
    )[0]
    assert torch.all(gradient[graph.edge_vec.shape[0] :] == 0.0)


@_CPU_FORCE
def test_force_virial_matches_the_scatter_reference() -> None:
    """The CSR assembly reproduces the array-API scatter it replaces."""
    from deepmd.dpmodel.utils.neighbor_graph import (
        edge_force_virial as reference_assembly,
    )

    descriptor = _build_descriptor(16)
    graph, atype = _build_graph(descriptor, canonical=True)
    generator = torch.Generator().manual_seed(11)
    edge_gradient = torch.randn(
        graph.edge_vec.shape, dtype=torch.float64, generator=generator
    )
    edge_vec = graph.edge_vec.to(torch.float64)

    force, atom_virial, virial, _ = edge_force_virial(
        edge_gradient,
        edge_vec,
        graph.edge_index,
        graph.edge_mask,
        graph.destination_order,
        graph.destination_row_ptr,
        graph.source_order,
        graph.source_row_ptr,
        graph.n_node,
        edge_gradient.new_empty(0),
        atype.shape[0],
        True,
    )
    expected_force, expected_atom_virial, expected_virial = reference_assembly(
        edge_gradient,
        edge_vec,
        graph.edge_index,
        graph.edge_mask,
        graph.n_node,
        node_capacity=atype.shape[0],
    )
    torch.testing.assert_close(force, expected_force)
    torch.testing.assert_close(atom_virial, expected_atom_virial)
    torch.testing.assert_close(virial, expected_virial)


@_CPU_FITTING
@pytest.mark.parametrize("activation", ["tanh", "silu"])
def test_fitting_matches_the_dense_network(activation: str) -> None:
    """The fused fitting reproduces the plain MLP, forward and backward.

    Both activations are covered because they differ in what the forward
    leaves for the backward: tanh's derivative is algebraic in its output, so
    the state is the activation, while silu's needs its argument.
    """
    from deepmd.dpmodel.fitting.ener_fitting import (
        EnergyFittingNet as EnergyFittingNetDP,
    )
    from deepmd.pt_expt.fitting.ener_fitting import (
        EnergyFittingNet,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_operator_arguments,
    )

    torch.manual_seed(5)
    fitting = EnergyFittingNet(
        ntypes=2,
        dim_descrpt=24,
        neuron=[32, 32, 32],
        resnet_dt=False,
        activation_function=activation,
        precision="float32",
        mixed_types=True,
        seed=3,
    ).eval()
    arguments = fitting_operator_arguments(fitting)
    descriptor = torch.randn(19, 24, dtype=torch.float32)
    atype = torch.arange(19) % 2
    bias = torch.zeros(2, dtype=torch.float64)

    leaf = descriptor.clone().requires_grad_(True)
    with torch.enable_grad():
        reference = EnergyFittingNetDP.call_graph(fitting, leaf, atype)["energy"]
    energy, saved = torch.ops.deepmd.graph_fitting(
        descriptor,
        atype,
        arguments.weights,
        arguments.biases,
        arguments.residuals,
        arguments.head_weight,
        arguments.head_bias,
        bias,
        arguments.activation,
    )
    torch.testing.assert_close(
        energy.reshape(-1),
        reference.detach().double().reshape(-1),
        atol=2e-5,
        rtol=2e-5,
    )

    cotangent = torch.linspace(-1.0, 1.0, 19, dtype=torch.float64).reshape(-1, 1)
    gradient = torch.ops.deepmd.graph_fitting_backward(
        cotangent,
        saved,
        arguments.weights,
        arguments.biases,
        arguments.residuals,
        arguments.head_weight,
        arguments.activation,
    )
    (expected,) = torch.autograd.grad((reference.double() * cotangent).sum(), leaf)
    torch.testing.assert_close(gradient, expected.float(), atol=2e-5, rtol=2e-5)


@_CPU
def test_prepared_table_is_reused_across_calls() -> None:
    """A second call on the same artifacts reuses the re-laid-out tables.

    The re-layout costs a pass over a few megabytes, which is amortized only
    if it happens once per model rather than once per step. The observable
    consequence is that two calls agree exactly, since a rebuilt table would
    still agree; the assertion here is on the cost, measured as the ratio of
    the second call to the first on a system small enough that the table
    dominates.
    """
    import time

    descriptor = _build_descriptor(128)
    graph, atype = _build_graph(descriptor, canonical=True, node_count=8)
    arguments = _arguments(descriptor, graph, atype)

    start = time.perf_counter()
    first, _ = torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)
    first_seconds = time.perf_counter() - start

    start = time.perf_counter()
    second, _ = torch.ops.deepmd.dpa4c_graph_compress(graph.edge_vec, *arguments)
    second_seconds = time.perf_counter() - start

    torch.testing.assert_close(first, second)
    assert second_seconds < 0.5 * first_seconds


@_CPU
def test_spin_descriptor_keeps_the_reference_path() -> None:
    """A spin-conditioned descriptor is declined rather than mis-evaluated.

    The CPU kernels carry no magnetic branch, so the decline has to come from
    the availability gate. Structural eligibility is unaffected: the tables
    still belong to the snapshot, which a CUDA host can consume.
    """
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        ef_op_available,
        mega_eligible,
        op_available,
    )

    with_spin = DescrptDPA4C(
        rcut=3.0,
        ntypes=2,
        channels=32,
        lmax=2,
        n_radial=8,
        precision="float32",
        seed=17,
        use_spin=[True, False],
    ).eval()
    assert mega_eligible(with_spin)
    assert op_available()
    assert not op_available(spin=True)
    assert not ef_op_available(spin=True)


@_CPU
def test_empty_graph_produces_the_isolated_atom_descriptor() -> None:
    """A node with no edge reduces to the moment floor rather than to a NaN.

    The normalizer is ``sqrt(mass + 1/4)``, so an isolated atom is finite by
    construction; the guard matters because a deployed system routinely
    carries padding nodes that own no edge at all.
    """
    descriptor = _build_descriptor(32)
    graph, atype = _build_graph(descriptor, canonical=True)
    arguments = _arguments(descriptor, graph, atype)
    empty_rows = torch.zeros_like(graph.destination_row_ptr)
    output, state = torch.ops.deepmd.dpa4c_graph_compress(
        graph.edge_vec, *arguments[:3], empty_rows, *arguments[4:]
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()
    torch.testing.assert_close(state[:, 0], torch.full_like(state[:, 0], 0.5))
    torch.testing.assert_close(state[:, 1], torch.full_like(state[:, 1], 0.5))
