# SPDX-License-Identifier: LGPL-3.0-or-later
"""The eager CPU vacuum fold must not execute deployment-only operators."""

import copy
import os

import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.pt_expt.kernels import (
    autotune,
)
from deepmd.pt_expt.kernels.cuda.dpa4 import (
    edge_radial,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.utils import (
    env,
    serialization,
)

from ..model.test_dpa4_vacuum_ref import (
    CONFIG,
    PRESET,
)

LEVELS = (
    "DP_CUDA_INFER",
    "DP_TRITON_INFER",
    "DP_CUTILE_INFER",
    "DP_CUTE_INFER",
    "DP_CUDA_TRAIN",
    "DP_TRITON_TRAIN",
    "DP_AMP_INFER",
)


@torch.library.custom_op(
    "deepmd_test::vacuum_cuda_radial", mutates_args=(), device_types="cuda"
)
def cuda_radial(
    edge_len: torch.Tensor, keep: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A CUDA-only operator, including on test hosts without a CUDA build."""
    raise AssertionError("the eager fold must never call the accelerator")


@cuda_radial.register_fake
def cuda_radial_fake(edge_len, keep, freqs):
    return edge_len.new_empty(edge_len.shape), edge_len.new_empty(
        (edge_len.shape[0], freqs.numel())
    )


def reference_model(monkeypatch, numb_fparam=0, numb_aparam=0, vacuum_ref=True):
    monkeypatch.setattr(env, "DEVICE", torch.device("cpu"))
    for name in LEVELS:
        monkeypatch.setenv(name, "0")
    config = copy.deepcopy(CONFIG)
    config["descriptor"].update(precision="float32")
    config["fitting_net"].update(
        precision="float32",
        vacuum_ref=vacuum_ref,
        numb_fparam=numb_fparam,
        numb_aparam=numb_aparam,
    )
    config["preset_out_bias"] = PRESET
    return get_model(config).eval()


@pytest.mark.parametrize(
    ("numb_fparam", "numb_aparam"), [(0, 0), (1, 0), (0, 1)]
)  # frame/atomic parameter dimensions
def test_cuda_export_folds_on_cpu_before_restoring_deployment_dispatch(
    monkeypatch, numb_fparam, numb_aparam
) -> None:
    """Construction-time CUDA bindings survive an eager, no-grad reference fold."""
    reference = reference_model(monkeypatch, numb_fparam, numb_aparam)
    data = {"model": reference.serialize()}
    reference.fold_vacuum_reference()
    expected = reference.atomic_model.fitting_net

    # Exercise the production constructor gate without requiring CUDA hardware
    # or its extension. This op has a fake implementation but no CPU kernel,
    # exactly the distinction between tracing and the eager vacuum evaluation.
    monkeypatch.setattr(
        edge_radial,
        "make_cuda_edge_radial",
        lambda envelope, basis: (
            lambda distance, keep: cuda_radial(distance, keep, basis.adam_freqs)
        ),
    )
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    model = BaseModel.deserialize(data["model"]).eval()
    am = model.atomic_model
    descriptor = am.descriptor
    assert descriptor._cuda_radial_fn is not None
    with pytest.raises(NotImplementedError, match="CPU"):
        model.fold_vacuum_reference()

    # A CUDA target is restored before the exporter tunes and traces. Setting
    # it only after construction also makes this test work on CPU-only hosts.
    monkeypatch.setattr(env, "DEVICE", torch.device("cuda"))
    monkeypatch.setattr(BaseModel, "deserialize", lambda data: model)
    events = []
    move = model.to
    fold = model.fold_vacuum_reference

    def move_to_cpu(device):
        events.append("move")
        assert device == "cpu"
        return move(device)

    def fold_on_cpu():
        events.append("fold")
        assert env.DEVICE.type == "cpu"
        assert not torch.is_grad_enabled()
        assert am.descriptor is not descriptor
        assert not am.descriptor.training
        assert am.descriptor._cuda_radial_fn is None
        assert all(t.device.type == "cpu" for t in model.parameters())
        assert all(t.device.type == "cpu" for t in model.buffers())
        fold()

    class StopBeforeTrace(Exception):
        pass

    def check_before_trace(model, target, lower_kind):
        events.append("trace")
        assert target.type == env.DEVICE.type == "cuda"
        assert torch.is_grad_enabled(), "force tracing still needs autograd"
        assert am.descriptor is descriptor
        assert os.environ["DP_CUDA_INFER"] == "1"
        raise StopBeforeTrace

    monkeypatch.setattr(model, "to", move_to_cpu)
    monkeypatch.setattr(model, "fold_vacuum_reference", fold_on_cpu)
    monkeypatch.setattr(autotune, "run_autotune", check_before_trace)
    with pytest.raises(StopBeforeTrace):
        serialization._trace_and_export_impl(data, target_device=env.DEVICE)
    assert events == ["move", "fold", "trace"]

    fitting = am.fitting_net
    assert fitting.vacuum_ref == expected.vacuum_ref
    torch.testing.assert_close(fitting.bias_atom_e, expected.bias_atom_e)
    torch.testing.assert_close(fitting.vacuum_table, expected.vacuum_table)
    if numb_fparam or numb_aparam:
        assert fitting.vacuum_table.device.type == "cpu"
        assert "vacuum_table" not in fitting.state_dict()
    assert all(t.device.type == "cpu" for t in model.buffers())

    # Restoring the descriptor must preserve optimized fake tracing, rather
    # than silently exporting the reference implementation for the real atoms.
    with torch.no_grad():
        traced = make_fx(
            lambda: am.vacuum_descriptor(),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )()
    assert any(node.target == cuda_radial._opoverload for node in traced.graph.nodes)


@pytest.mark.parametrize("failure", ["construction", "fold"])  # failing stage
def test_reference_context_restores_state_after_failure(monkeypatch, failure) -> None:
    model = reference_model(monkeypatch)
    am = model.atomic_model
    descriptor = am.descriptor
    monkeypatch.setattr(env, "DEVICE", torch.device("meta"))
    for i, name in enumerate(LEVELS):
        if i % 2:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, "1")
    saved = {name: os.environ.get(name) for name in LEVELS}

    def fail(data):
        raise RuntimeError("reference construction failed")

    if failure == "construction":
        monkeypatch.setattr(type(descriptor), "deserialize", fail)
    with pytest.raises(RuntimeError, match="failed"):
        with serialization._cpu_reference_descriptors(model):
            assert am.descriptor is not descriptor
            assert env.DEVICE.type == "cpu"
            assert all(os.environ[name] == "0" for name in LEVELS)
            raise RuntimeError("reference fold failed")
    assert am.descriptor is descriptor
    assert env.DEVICE.type == "meta"
    assert {name: os.environ.get(name) for name in LEVELS} == saved


@pytest.mark.parametrize("folded_table", [False, True])  # disabled vs already folded
def test_unneeded_reference_does_not_rebuild_descriptor(
    monkeypatch, folded_table
) -> None:
    model = reference_model(monkeypatch, numb_fparam=1, vacuum_ref=folded_table)
    if folded_table:
        model.fold_vacuum_reference()
    descriptor = model.atomic_model.descriptor

    def fail(data):
        raise AssertionError("an unreferenced descriptor should not be rebuilt")

    monkeypatch.setattr(type(descriptor), "deserialize", fail)
    monkeypatch.setattr(env, "DEVICE", torch.device("meta"))
    with serialization._cpu_reference_descriptors(model):
        assert model.atomic_model.descriptor is descriptor
        assert env.DEVICE.type == "meta"


def test_reference_descriptor_preserves_disabled_graph_route(monkeypatch) -> None:
    """Virtual-spin backbones disable graph lowering outside serialization."""
    monkeypatch.setattr(env, "DEVICE", torch.device("cpu"))
    for name in LEVELS:
        monkeypatch.setenv(name, "0")
    model = get_model(
        {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa1",
                "rcut": 4.0,
                "rcut_smth": 0.5,
                "sel": [8],
                "attn_layer": 0,
                "axis_neuron": 2,
                "neuron": [4, 8],
                "seed": 1,
            },
            "fitting_net": {
                "type": "ener",
                "neuron": [8],
                "vacuum_ref": True,
                "seed": 1,
            },
            "preset_out_bias": PRESET,
        }
    ).eval()
    am = model.atomic_model
    descriptor = am.descriptor
    descriptor.disable_graph_lower()
    with torch.no_grad():
        descriptor.se_atten.mean.fill_(0.3)
        expected = am.vacuum_descriptor()
        with serialization._cpu_reference_descriptors(model):
            assert not am.descriptor.uses_graph_lower()
            torch.testing.assert_close(am.vacuum_descriptor(), expected)
    assert am.descriptor is descriptor
