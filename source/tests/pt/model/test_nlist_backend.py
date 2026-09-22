# SPDX-License-Identifier: LGPL-3.0-or-later
"""``nlist_backend`` dispatch + O(N) strategy equivalence for the pt backend.

The pt model is reconstructed eagerly in ``DeepEval`` and evaluated via
``forward_common_lower`` when an O(N) neighbor-list strategy is selected.  Each
strategy (``vesin``, ``nv``) must give results identical to the native dense
builder, and the ``nlist_backend`` choice must dispatch / validate correctly.
Strategy tests are skipped when the backend (or, for ``nv``, a CUDA device) is
unavailable.
"""

import copy
from types import (
    SimpleNamespace,
)

import numpy as np
import pytest
import torch

from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils.nv_nlist import (
    NvNeighborList,
    is_nv_available,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    VesinNeighborList,
    is_vesin_torch_available,
)

# Each O(N) strategy: (backend name, builder class, availability skip mark).
_BACKEND_MARKS = {
    "vesin": pytest.mark.skipif(
        not is_vesin_torch_available(), reason="vesin.torch is not installed"
    ),
    "nv": pytest.mark.skipif(
        not (is_nv_available() and torch.cuda.is_available()),
        reason="nvalchemiops CUDA neighbor list unavailable",
    ),
}
_BUILDER_CLS = {"vesin": VesinNeighborList, "nv": NvNeighborList}
STRATEGIES = [pytest.param(name, marks=mark) for name, mark in _BACKEND_MARKS.items()]

TYPE_MAP = ["O", "H", "B"]

model_se_e2_a = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20, 8],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12],
        "resnet_dt": False,
        "axis_neuron": 4,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

model_dpa1 = {
    "type_map": TYPE_MAP,
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [6, 12, 24],
        "axis_neuron": 4,
        "attn": 16,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "set_davg_zero": True,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
}

ALL_MODELS = {"se_e2_a": model_se_e2_a, "dpa1": model_dpa1}


def _save_pt(md_dict: dict, path: str) -> None:
    """Write a minimal loadable .pt (state_dict + model_params) for DeepPot."""
    model = get_model(copy.deepcopy(md_dict)).to(torch.float64)
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(md_dict))
    torch.save(wrapper.state_dict(), path)


def _system():
    """Single periodic frame (every strategy supports a periodic box)."""
    rng = np.random.default_rng(20240604)
    coords = (rng.random((1, 8, 3)) * 6.0).astype(np.float64)
    atype = np.array([0, 0, 1, 1, 2, 0, 1, 2], dtype=np.int64)
    box = (np.eye(3) * 6.0).reshape(1, 9).astype(np.float64)
    return coords, atype, box


def _multiframe_system(nframes: int = 3):
    """Frames with different box sizes -> different per-frame ghost counts,
    exercising the builder's pad-to-common-nall + stack path.
    """
    rng = np.random.default_rng(20240604)
    atype = np.array([0, 0, 1, 1, 2, 0, 1, 2], dtype=np.int64)
    coords, boxes = [], []
    for ff in range(nframes):
        box_len = 6.0 + 1.5 * ff
        coords.append((rng.random((len(atype), 3)) * box_len).astype(np.float64))
        boxes.append((np.eye(3) * box_len).reshape(9).astype(np.float64))
    return np.stack(coords, axis=0), atype, np.stack(boxes, axis=0)


@pytest.fixture(scope="module")
def pt_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("nlist_backend")
    files = {}
    for name, md in ALL_MODELS.items():
        p = str(d / f"{name}.pt")
        _save_pt(md, p)
        files[name] = p
    return files


def _assert_eval_close(dp_ref, dp_test, coords, cells, atype, msg: str) -> None:
    ref = dp_ref.eval(coords, cells, atype, atomic=True)
    out = dp_test.eval(coords, cells, atype, atomic=True)
    for a, b, label in zip(ref, out, ["e", "f", "v", "ae", "av"], strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, err_msg=f"{msg} {label}")


# --- dispatch / selection ---------------------------------------------------


def test_invalid_backend_raises(pt_files) -> None:
    with pytest.raises(ValueError):
        DeepPot(pt_files["se_e2_a"], nlist_backend="bogus")


def test_native_uses_no_strategy(pt_files) -> None:
    dp = DeepPot(pt_files["se_e2_a"], nlist_backend="native")
    assert dp.deep_eval._nlist_builder is None


@pytest.mark.parametrize("backend", STRATEGIES)
def test_explicit_backend_selects_builder(pt_files, backend: str) -> None:
    dp = DeepPot(pt_files["se_e2_a"], nlist_backend=backend)
    assert isinstance(dp.deep_eval._nlist_builder, _BUILDER_CLS[backend])


@_BACKEND_MARKS["vesin"]
def test_auto_prefers_vesin(pt_files) -> None:
    # auto picks the first available O(N) builder; vesin is preferred.
    builder = DeepPot(pt_files["se_e2_a"]).deep_eval._nlist_builder
    assert isinstance(builder, VesinNeighborList)


def test_self_built_model_forces_native(pt_files, monkeypatch) -> None:
    # A model reporting use_self_built_nlist()=True keeps the native path and
    # ignores the requested backend (without even validating its name).
    deep_eval = DeepPot(pt_files["se_e2_a"], nlist_backend="native").deep_eval
    inner = deep_eval.dp.model["Default"]
    monkeypatch.setattr(inner, "use_self_built_nlist", lambda: True, raising=False)
    for backend in ("auto", "vesin", "nv", "bogus"):
        deep_eval._setup_nlist_backend(backend)
        assert deep_eval._nlist_builder is None


def test_edge_strategy_freezes_parameters_only_during_inference_call() -> None:
    from deepmd.pt.infer.deep_eval import DeepEval as PTDeepEval

    class ProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((), device="cpu"))
            self.saw_frozen = False

        def get_sel(self) -> list[int]:
            return [4]

        def forward_common_lower(self, *args, **kwargs) -> dict[str, torch.Tensor]:
            self.saw_frozen = not self.weight.requires_grad
            return {}

    class ProbeBuilder:
        def build(self, coord, atype, *args, **kwargs) -> SimpleNamespace:
            return SimpleNamespace(
                coord=coord,
                atype=atype,
                edge_index=torch.empty((2, 0), dtype=torch.long, device="cpu"),
                edge_vec=torch.empty((0, 3), dtype=coord.dtype, device="cpu"),
                edge_scatter_index=torch.empty((2, 0), dtype=torch.long, device="cpu"),
                edge_mask=torch.empty((0,), dtype=torch.bool, device="cpu"),
            )

    inner = ProbeModel()
    deep_eval = object.__new__(PTDeepEval)
    deep_eval.dp = ModelWrapper(inner)
    deep_eval._uses_edge_schema = True
    deep_eval._nlist_builder = ProbeBuilder()
    deep_eval.rcut = 4.0

    PTDeepEval._eval_lower_strategy(
        deep_eval,
        torch.zeros((1, 1, 3), device="cpu"),
        torch.zeros((1, 1), dtype=torch.long, device="cpu"),
        None,
        None,
        None,
        None,
        False,
    )

    assert inner.saw_frozen
    assert inner.weight.requires_grad


# --- equivalence with the native dense builder ------------------------------


@pytest.mark.parametrize("name", list(ALL_MODELS))
@pytest.mark.parametrize("backend", STRATEGIES)
def test_strategy_matches_native(pt_files, backend: str, name: str) -> None:
    """Each strategy matches native on a periodic single-frame system."""
    coords, atype, box = _system()
    dp_native = DeepPot(pt_files[name], nlist_backend="native")
    dp_strat = DeepPot(pt_files[name], nlist_backend=backend)
    _assert_eval_close(dp_native, dp_strat, coords, box, atype, f"{name} {backend}")


@pytest.mark.parametrize("name", list(ALL_MODELS))
@pytest.mark.parametrize("backend", STRATEGIES)
def test_strategy_matches_native_multiframe(pt_files, backend: str, name: str) -> None:
    """Each strategy matches native across frames with differing ghost counts."""
    coords, atype, box = _multiframe_system()
    dp_native = DeepPot(pt_files[name], nlist_backend="native")
    dp_strat = DeepPot(pt_files[name], nlist_backend=backend)
    _assert_eval_close(dp_native, dp_strat, coords, box, atype, f"{name} {backend} mf")


@_BACKEND_MARKS["vesin"]
@pytest.mark.parametrize("name", list(ALL_MODELS))
def test_vesin_matches_native_nonperiodic(pt_files, name: str) -> None:
    """Vesin also supports non-periodic systems."""
    coords, atype, _ = _system()
    dp_native = DeepPot(pt_files[name], nlist_backend="native")
    dp_vesin = DeepPot(pt_files[name], nlist_backend="vesin")
    _assert_eval_close(dp_native, dp_vesin, coords, None, atype, f"{name} vesin nopbc")


@_BACKEND_MARKS["nv"]
@pytest.mark.parametrize("name", list(ALL_MODELS))
def test_nv_matches_native_nonperiodic(pt_files, name: str) -> None:
    """NV also supports non-periodic systems."""
    coords, atype, _ = _system()
    dp_native = DeepPot(pt_files[name], nlist_backend="native")
    dp_nv = DeepPot(pt_files[name], nlist_backend="nv")
    _assert_eval_close(dp_native, dp_nv, coords, None, atype, f"{name} nv nopbc")
