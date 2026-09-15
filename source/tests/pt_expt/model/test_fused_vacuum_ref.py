# SPDX-License-Identifier: LGPL-3.0-or-later
"""The fused inference operators serve the isolated-atom reference through the bias.

On a fitting without conditioning the reference output of every type is a
constant, so the model-level fused energy/force route hands the operators the
bias minus that constant and reproduces the autograd route exactly: an
isolated atom gives its bias and a cluster gives the referenced energies.
"""

import numpy as np
import pytest
import torch

import deepmd.pt_expt.model.make_model as make_model
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.pt_expt.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt_expt.descriptor.dpa4c import (
    DescrptDPA4C,
)
from deepmd.pt_expt.fitting.ener_fitting import (
    EnergyFittingNet,
)
from deepmd.pt_expt.kernels.cuda.dpa1.graph_energy_force import (
    op_available as dpa1_op_available,
)
from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
    ef_op_available as dpa4c_op_available,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)

RCUT = 4.0
BIAS = np.array([[-3.0], [0.5]])


def make_model_with(
    descriptor: DescrptDPA1 | DescrptDPA4C, numb_fparam: int = 0
) -> EnergyModel:
    fitting = EnergyFittingNet(
        2,
        descriptor.get_dim_out(),
        neuron=[32, 32],
        mixed_types=True,
        precision="float32",
        resnet_dt=False,
        activation_function="tanh",
        vacuum_ref=True,
        numb_fparam=numb_fparam,
        seed=1,
    )
    fitting["bias_atom_e"] = BIAS.copy()
    return EnergyModel(descriptor, fitting, type_map=["O", "H"]).to(env.DEVICE).eval()


def make_dpa1(numb_fparam: int = 0) -> EnergyModel:
    return make_model_with(
        DescrptDPA1(
            rcut=RCUT,
            rcut_smth=0.5,
            sel=[20],
            ntypes=2,
            attn_layer=0,
            axis_neuron=4,
            neuron=[8, 16, 32],
            tebd_input_mode="concat",
            precision="float32",
            seed=1,
        ),
        numb_fparam,
    )


def make_dpa4c() -> EnergyModel:
    model = make_model_with(
        DescrptDPA4C(
            rcut=RCUT,
            ntypes=2,
            channels=32,
            lmax=2,
            n_radial=8,
            precision="float32",
            seed=17,
        )
    )
    model.atomic_model.descriptor.enable_compression(0.5)
    return model


def fused_and_autograd(
    model: EnergyModel, coord: np.ndarray, atype: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coord_t = torch.as_tensor(coord, dtype=torch.float64, device=env.DEVICE)
    atype_t = torch.as_tensor(atype, dtype=torch.long, device=env.DEVICE)
    graph = build_neighbor_graph(
        coord_t, atype_t, None, RCUT, with_csr=True, canonicalize=True
    )
    fused = make_model._fused_energy_force_graph(
        model, graph, atype_t.reshape(-1), False
    )
    assert fused is not None, "the fused energy/force route was not taken"
    autograd = model.atomic_model.forward_common_atomic_graph(
        graph, atype_t.reshape(-1)
    )
    return (
        fused["energy"][:, 0].detach().cpu().numpy(),
        autograd["energy"][:, 0].detach().cpu().numpy(),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused operators need CUDA"
)
@pytest.mark.parametrize(
    "build, available",
    [(make_dpa1, dpa1_op_available), (make_dpa4c, dpa4c_op_available)],
    ids=["dpa1", "dpa4c_compressed"],
)
def test_fused_route_matches_autograd_with_vacuum_ref(
    monkeypatch, build, available
) -> None:
    if not available():
        pytest.skip("the fused operator library is unavailable")
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    model = build()
    rng = np.random.default_rng(0)
    coord = rng.normal(size=(2, 6, 3)) * 1.2
    atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
    fused, autograd = fused_and_autograd(model, coord, atype)
    np.testing.assert_allclose(fused, autograd, rtol=1e-6, atol=1e-6)
    fused_iso, _ = fused_and_autograd(model, np.zeros((2, 1, 3)), np.array([[0], [1]]))
    np.testing.assert_allclose(fused_iso, BIAS[:, 0], rtol=1e-6, atol=1e-6)


def test_fused_route_yields_to_autograd_with_fparam(monkeypatch) -> None:
    """A reference that varies between atoms is served by the autograd lower."""
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    model = make_dpa1(numb_fparam=1)
    coord_t = torch.zeros((1, 1, 3), dtype=torch.float64, device=env.DEVICE)
    atype_t = torch.zeros((1, 1), dtype=torch.long, device=env.DEVICE)
    graph = build_neighbor_graph(
        coord_t, atype_t, None, RCUT, with_csr=True, canonicalize=True
    )
    assert (
        make_model._fused_energy_force_graph(model, graph, atype_t.reshape(-1), False)
        is None
    )
