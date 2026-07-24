# SPDX-License-Identifier: LGPL-3.0-or-later
"""neighbor_graph_method dispatch: builders are perf-only equivalents (same
energy + force); dpmodel/jax fail-fast on vesin/nv.

Builders run eagerly outside traced/compiled regions. Export and training
compile use synthetic dense graph inputs, so flipping the default to
``"auto"`` does not change ``.pt2`` artifacts — only the eager / DeepEval
builder selection. Dense is typically not the bottleneck at small sizes; the
O(N) win appears for large systems (time builders manually at N ≳ a few
thousand to document the crossover).
"""

import numpy as np
import pytest
import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nv_nlist import (
    is_nv_available,
)
from deepmd.pt_expt.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt_expt.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.utils.neighbor_graph_method import (
    resolve_auto_graph_builder,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)

GLOBAL_SEED = 20240101


def _make_model():
    rcut, rcut_smth, sel, nt = 6.0, 2.0, 20, 2
    ds = DescrptDPA1(
        rcut,
        rcut_smth,
        sel,
        nt,
        neuron=[3, 6],
        axis_neuron=2,
        attn=4,
        attn_layer=0,  # graph lower only supports attn_layer == 0
        attn_dotr=True,
        attn_mask=False,
        activation_function="tanh",
        set_davg_zero=False,
        type_one_side=True,
        precision="float64",
        seed=GLOBAL_SEED,
    ).to(env.DEVICE)
    ft = InvarFitting(
        "energy",
        nt,
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
        precision="float64",
        seed=GLOBAL_SEED,
    ).to(env.DEVICE)
    return EnergyModel(ds, ft, type_map=["O", "H"]).to(env.DEVICE)


def _eval(model, method):
    rng = np.random.default_rng(0)
    coord = torch.tensor(
        rng.random((1, 6, 3)) * 4.0, dtype=torch.float64, device=env.DEVICE
    )
    atype = torch.tensor([[0, 1, 1, 0, 1, 1]], dtype=torch.int64, device=env.DEVICE)
    box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * 6.0).reshape(1, 3, 3)
    ret = model.forward_common(coord, atype, box, neighbor_graph_method=method)
    # graph path returns the output-agnostic dict (no translated force/virial);
    # energy_redu = total energy, energy_derv_r = d energy / d coord (force parity)
    return ret["energy_redu"], ret["energy_derv_r"]


def _available_builders() -> list[str]:
    """Concrete builders available in this environment (always includes dense)."""
    methods = ["dense"]
    try:
        import ase  # noqa: F401

        methods.append("ase")
    except ImportError:
        pass
    if is_vesin_torch_available():
        methods.append("vesin")
    if env.DEVICE.type == "cuda" and is_nv_available():
        methods.append("nv")
    return methods


@pytest.mark.skipif(not is_vesin_torch_available(), reason="vesin[torch] not installed")
def test_vesin_matches_dense_energy_force():
    torch.manual_seed(0)
    model = _make_model()
    e_d, f_d = _eval(model, "dense")
    e_v, f_v = _eval(model, "vesin")
    tol = 1e-12 if env.DEVICE.type == "cpu" else 1e-10
    torch.testing.assert_close(e_v, e_d, rtol=tol, atol=tol)
    torch.testing.assert_close(f_v, f_d, rtol=tol, atol=tol)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_nv_available()),
    reason="nvalchemiops requires CUDA + nvalchemi-toolkit-ops",
)
def test_nv_matches_dense_energy_force():
    torch.manual_seed(0)
    model = _make_model()
    e_d, f_d = _eval(model, "dense")
    e_n, f_n = _eval(model, "nv")
    tol = 1e-10  # CUDA fp64: absorbs scatter-atomic / index_add nondeterminism
    torch.testing.assert_close(e_n, e_d, rtol=tol, atol=tol)
    torch.testing.assert_close(f_n, f_d, rtol=tol, atol=tol)


def test_all_available_builders_energy_force_parity():
    """Every available builder is value-transparent vs dense on the same system.

    Builders differ only in edge enumeration order; segment reductions are
    order-independent up to fp addition order (device-conditional tol).
    """
    torch.manual_seed(0)
    model = _make_model()
    methods = _available_builders()
    e_ref, f_ref = _eval(model, "dense")
    tol = 1e-12 if env.DEVICE.type == "cpu" else 1e-10
    for method in methods:
        if method == "dense":
            continue
        e_m, f_m = _eval(model, method)
        torch.testing.assert_close(e_m, e_ref, rtol=tol, atol=tol, msg=method)
        torch.testing.assert_close(f_m, f_ref, rtol=tol, atol=tol, msg=method)


def test_none_and_auto_match_resolved_builder():
    """Model-level None / 'auto' resolve to the same concrete builder policy."""
    torch.manual_seed(0)
    model = _make_model()
    resolved = resolve_auto_graph_builder(env.DEVICE)
    e_auto, f_auto = _eval(model, "auto")
    e_none, f_none = _eval(model, None)
    e_res, f_res = _eval(model, resolved)
    tol = 1e-12 if env.DEVICE.type == "cpu" else 1e-10
    torch.testing.assert_close(e_auto, e_res, rtol=tol, atol=tol)
    torch.testing.assert_close(f_auto, f_res, rtol=tol, atol=tol)
    torch.testing.assert_close(e_none, e_res, rtol=tol, atol=tol)
    torch.testing.assert_close(f_none, f_res, rtol=tol, atol=tol)


def test_dpmodel_backend_rejects_vesin():
    """dpmodel/jax fail-fast names vesin/nv as pt_expt-only."""
    from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DPDescrptDPA1
    from deepmd.dpmodel.fitting.invar_fitting import InvarFitting as DPInvarFitting
    from deepmd.dpmodel.model.ener_model import EnergyModel as DPEnergyModel

    rcut, rcut_smth, sel, nt = 6.0, 2.0, 20, 2
    ds = DPDescrptDPA1(
        rcut,
        rcut_smth,
        sel,
        nt,
        neuron=[3, 6],
        axis_neuron=2,
        attn=4,
        attn_layer=0,
        attn_dotr=True,
        attn_mask=False,
        activation_function="tanh",
        set_davg_zero=False,
        type_one_side=True,
        precision="float64",
        seed=GLOBAL_SEED,
    )
    ft = DPInvarFitting(
        "energy",
        nt,
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
        precision="float64",
        seed=GLOBAL_SEED,
    )
    model = DPEnergyModel(ds, ft, type_map=["O", "H"])
    coord = np.random.default_rng(0).random((1, 6, 3)) * 4.0
    atype = np.array([[0, 1, 1, 0, 1, 1]], dtype=np.int64)
    box = (np.eye(3) * 6.0).reshape(1, 3, 3)
    with pytest.raises(ValueError, match="pt_expt backend"):
        model.call_common(coord, atype, box, neighbor_graph_method="vesin")
    with pytest.raises(ValueError, match="pt_expt backend"):
        model.call_common(coord, atype, box, neighbor_graph_method="nv")


def test_explicit_method_fails_fast_for_ineligible_descriptor():
    """An EXPLICIT neighbor_graph_method must fail fast when the descriptor
    has no graph lower (mirrors the dpmodel guard; the default-path check in
    _resolve_graph_method does not protect explicit methods).
    """
    from deepmd.pt_expt.descriptor.se_e2_a import (
        DescrptSeA,
    )

    # se_e2_a: mixed_types() is False and there is no graph lower
    ds = DescrptSeA(
        6.0,
        2.0,
        [10, 10],
        neuron=[3, 6],
        axis_neuron=2,
        precision="float64",
        seed=GLOBAL_SEED,
    ).to(env.DEVICE)
    ft = InvarFitting(
        "energy",
        2,
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
        precision="float64",
        seed=GLOBAL_SEED,
    ).to(env.DEVICE)
    model = EnergyModel(ds, ft, type_map=["O", "H"]).to(env.DEVICE)
    coord = torch.rand(1, 4, 3, dtype=torch.float64, device=env.DEVICE) * 3
    atype = torch.zeros(1, 4, dtype=torch.int64, device=env.DEVICE)
    box = (torch.eye(3, dtype=torch.float64, device=env.DEVICE) * 6).reshape(1, 9)
    for method in ("dense", "ase", "vesin", "nv"):
        with pytest.raises(NotImplementedError, match="graph lower"):
            model.call_common(coord, atype, box, neighbor_graph_method=method)
