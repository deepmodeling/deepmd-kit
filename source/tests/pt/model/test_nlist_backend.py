# SPDX-License-Identifier: LGPL-3.0-or-later
"""``nlist_backend`` dispatch + vesin/native equivalence for the pt backend.

The pt model is reconstructed eagerly in ``DeepEval`` and evaluated via
``forward_common_lower`` when the O(N) vesin neighbor list is selected (the
exported TorchScript graph is untouched).  native and vesin must give identical
results, and the ``nlist_backend`` choice must dispatch / validate correctly.
"""

import copy

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
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)

pytestmark = pytest.mark.skipif(
    not is_vesin_torch_available(), reason="vesin.torch is not installed"
)

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
    rng = np.random.default_rng(20240604)
    coords = (rng.random((1, 8, 3)) * 6.0).astype(np.float64)
    atype = np.array([0, 0, 1, 1, 2, 0, 1, 2], dtype=np.int64)
    box = (np.eye(3) * 6.0).reshape(1, 9).astype(np.float64)
    return coords, atype, box


def _multiframe_system(nframes: int = 3):
    """Frames with different box sizes -> different per-frame ghost counts,
    exercising the vesin builder's pad-to-common-nall + stack path.
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


def test_default_is_auto(pt_files) -> None:
    # vesin is available (module skip guard), non-spin/non-hessian -> auto picks it
    assert DeepPot(pt_files["se_e2_a"]).deep_eval._use_vesin is True


def test_native_disables_vesin(pt_files) -> None:
    dp = DeepPot(pt_files["se_e2_a"], nlist_backend="native")
    assert dp.deep_eval._use_vesin is False


def test_invalid_raises(pt_files) -> None:
    with pytest.raises(ValueError):
        DeepPot(pt_files["se_e2_a"], nlist_backend="bogus")


@pytest.mark.parametrize("name", list(ALL_MODELS))  # descriptor family
@pytest.mark.parametrize("periodic", [False, True])  # non-PBC vs PBC
def test_vesin_matches_native(pt_files, name: str, periodic: bool) -> None:
    """Vesin and native give identical energy/force/virial/atomic-virial."""
    coords, atype, box = _system()
    cells = box if periodic else None
    dp_native = DeepPot(pt_files[name], nlist_backend="native")
    dp_vesin = DeepPot(pt_files[name], nlist_backend="vesin")
    ref = dp_native.eval(coords, cells, atype, atomic=True)
    out = dp_vesin.eval(coords, cells, atype, atomic=True)
    for a, b, label in zip(ref, out, ["e", "f", "v", "ae", "av"], strict=True):
        np.testing.assert_allclose(
            a, b, rtol=1e-9, atol=1e-9, err_msg=f"{name} {label}"
        )


@pytest.mark.parametrize("name", list(ALL_MODELS))  # descriptor family
def test_vesin_matches_native_multiframe(pt_files, name: str) -> None:
    """Multi-frame eval (frames with differing ghost counts) matches native."""
    coords, atype, box = _multiframe_system()
    dp_native = DeepPot(pt_files[name], nlist_backend="native")
    dp_vesin = DeepPot(pt_files[name], nlist_backend="vesin")
    ref = dp_native.eval(coords, box, atype, atomic=True)
    out = dp_vesin.eval(coords, box, atype, atomic=True)
    for a, b, label in zip(ref, out, ["e", "f", "v", "ae", "av"], strict=True):
        np.testing.assert_allclose(
            a, b, rtol=1e-9, atol=1e-9, err_msg=f"{name} {label}"
        )
