# SPDX-License-Identifier: LGPL-3.0-or-later

from types import (
    SimpleNamespace,
)

import pytest
import torch

from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.serialization import (
    _resolve_lower_kind,
    _trace_and_export,
    build_synthetic_graph_inputs,
)


def _config() -> dict:
    return {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 3.0,
            "channels": 16,
            "lmax": 4,
            "n_radial": 4,
            "precision": "float64",
            "seed": 17,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
        },
    }


def _compressed_config(channels: int = 8) -> dict:
    config = _config()
    descriptor = config["descriptor"]
    descriptor["channels"] = channels
    descriptor["lmax"] = 2
    descriptor["n_radial"] = 8
    descriptor["precision"] = "float32"
    fitting = config["fitting_net"]
    fitting["neuron"] = [32, 32]
    fitting["activation_function"] = "silu"
    fitting["precision"] = "float32"
    return config


def _run_graph(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    sample = build_synthetic_graph_inputs(
        model,
        e_max=None,
        nframes=2,
        nloc=7,
        dtype=torch.float64,
        device=env.DEVICE,
    )
    (
        atype,
        n_node,
        n_local,
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
        fparam,
        aparam,
        charge_spin,
    ) = sample
    return model.forward_common_lower_graph(
        atype,
        n_node,
        n_local,
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
        destination_sorted=True,
        do_atomic_virial=True,
        fparam=fparam,
        aparam=aparam,
        charge_spin=charge_spin,
    )


def test_graph_lower_energy_force_are_finite() -> None:
    model = get_model(_config()).to(env.DEVICE).eval()
    assert model.get_descriptor().uses_graph_lower()
    result = _run_graph(model)
    for key in (
        "energy",
        "energy_redu",
        "energy_derv_r",
        "energy_derv_c_redu",
    ):
        assert key in result
        assert torch.isfinite(result[key]).all(), key


def test_graph_force_loss_trains_descriptor() -> None:
    model = get_model(_config()).to(env.DEVICE).train()
    result = _run_graph(model)
    loss = result["energy_redu"].square().mean()
    loss = loss + result["energy_derv_r"].square().mean()
    loss.backward()
    descriptor = model.get_descriptor()
    gradients = {
        name: parameter.grad for name, parameter in descriptor.named_parameters()
    }
    for name, gradient in gradients.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_graph_export() -> None:
    model = get_model(_config()).to("cpu").eval()
    exported, _metadata, _model_json, _output_keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="graph",
        do_atomic_virial=True,
    )
    assert isinstance(exported, torch.export.ExportedProgram)


def test_compressed_graph_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    model = get_model(_compressed_config()).to("cpu").eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    exported, metadata, _model_json, _output_keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="graph",
        do_atomic_virial=True,
    )
    assert isinstance(exported, torch.export.ExportedProgram)
    assert metadata["graph_edge_dtype"] == "float32"


@pytest.mark.parametrize("channels", [8, 64])
def test_compact_canonical_graph_export(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    model = get_model(_compressed_config(channels)).to("cpu").eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    exported, metadata, _model_json, _output_keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind="dpa4c_canonical",
        do_atomic_virial=True,
    )
    assert isinstance(exported, torch.export.ExportedProgram)
    assert metadata["lower_input_kind"] == "dpa4c_canonical"
    assert metadata["graph_edge_dtype"] == "float32"
    assert metadata["canonical_index_dtype"] == "uint32"


@pytest.mark.parametrize("channels", [8, 64, 128])
def test_auto_lower_kind_selects_compact_canonical(channels: int) -> None:
    model = get_model(_compressed_config(channels)).to("cpu").eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    data = {"model": model.serialize()}
    assert _resolve_lower_kind("model.pt2", data, "auto") == "dpa4c_canonical"


def test_compact_canonical_eligibility_rejects_other_descriptors() -> None:
    from deepmd.kernels.cuda.dpa4c.canonical import (
        canonical_model_eligible,
    )

    model = SimpleNamespace(
        atomic_model=SimpleNamespace(
            descriptor=SimpleNamespace(compress=True),
            fitting_net=object(),
        )
    )
    assert not canonical_model_eligible(model)


@pytest.mark.parametrize("channels", [8, 64])
def test_compressed_level_two_matches_autograd(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    model = get_model(_compressed_config(channels)).to(env.DEVICE).eval()
    model.get_descriptor().enable_compression(min_nbor_dist=0.5)
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    reference = _run_graph(model)
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    actual = _run_graph(model)
    for key in (
        "energy",
        "energy_redu",
        "energy_derv_r",
        "energy_derv_c",
        "energy_derv_c_redu",
    ):
        torch.testing.assert_close(actual[key], reference[key])
