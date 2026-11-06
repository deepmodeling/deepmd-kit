# SPDX-License-Identifier: LGPL-3.0-or-later
"""Legacy vacuum references remain differentiable on the graph route."""

import pytest
import torch

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    graph_from_dense_quartet,
)
from deepmd.pt_expt.common import (
    auto_wrapped_class,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...common.dpmodel.test_vacuum_descriptor_graph import (
    make_model,
)


@pytest.mark.parametrize("kind", ["dpa1", "dpa2"])  # legacy descriptor
def test_graph_vacuum_reference_preserves_parameter_gradients(kind) -> None:
    model = auto_wrapped_class(DPAtomicModel).deserialize(make_model(kind).serialize())
    descriptor = model.descriptor
    blocks = (
        [descriptor.se_atten]
        if kind == "dpa1"
        else [descriptor.repinit, descriptor.repformers]
    )
    with torch.no_grad():
        for block in blocks:
            block.mean.fill_(0.3)
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]]],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    atype = torch.tensor([[0, 1]], dtype=torch.int64, device=env.DEVICE)
    nlist = torch.tensor([[[1, -1], [0, -1]]], dtype=torch.int64, device=env.DEVICE)
    graph, atype = graph_from_dense_quartet(coord, atype, nlist, None)
    gg = descriptor.call_graph(
        graph, atype, type_embedding=descriptor.graph_type_embedding_table()
    )[0]
    reference = model.vacuum_descriptor()
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    reference_grad = torch.autograd.grad(
        reference.sum(), parameters, allow_unused=True, retain_graph=True
    )
    assert any(g is not None and torch.any(g.abs() > 1e-7) for g in reference_grad)
    expected = model.fitting_net.call_graph(gg, atype, vacuum_descriptor=reference)[
        "energy"
    ]
    actual = model.forward_atomic_graph(graph, atype)["energy"]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected_grad = torch.autograd.grad(expected.sum(), parameters, allow_unused=True)
    actual_grad = torch.autograd.grad(actual.sum(), parameters, allow_unused=True)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
