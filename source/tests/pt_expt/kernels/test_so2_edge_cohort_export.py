# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU-traced official exports keep the native, deployable backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA target required"
)


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
def test_export_keeps_native_backward(monkeypatch, tmp_path, backend):
    """Stop at the real export boundary and inspect the traced backward ops."""
    from deepmd.pt.entrypoints import freeze_pt2
    from deepmd.pt.model.model import get_model
    from deepmd.pt.train.wrapper import ModelWrapper
    from deepmd.pt_expt.kernels import autotune
    from deepmd.pt_expt.kernels.triton.sezm import so2_edge_cohort as ecs
    from deepmd.pt_expt.utils.serialization import _trace_and_export_impl

    for name in (
        "DP_TRITON_TRAIN",
        "DP_CUDA_INFER",
        "DP_CUDA_TRAIN",
        "DP_CUTE_INFER",
        "DP_CUTILE_INFER",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("DP_TRITON_INFER", "2")
    monkeypatch.setattr(freeze_pt2, "_tune_triton_configs", lambda *args: None)
    monkeypatch.setattr(autotune, "run_autotune", lambda *args: None)
    params = {
        "type": "SeZM",
        "type_map": ["C"],
        "use_compile": False,
        "descriptor": {
            "type": "SeZM",
            "sel": [32],
            "rcut": 4.0,
            "channels": 32,
            "n_focus": 2,
            "focus_dim": 32,
            "n_radial": 4,
            "radial_mlp": [32],
            "use_env_seed": True,
            "l_schedule": [3, 3, 0],
            "mmax": 1,
            "so2_layers": 3,
            "n_atten_head": 1,
            "ffn_neurons": 32,
            "ffn_blocks": 1,
            "use_amp": False,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {"neuron": [16], "precision": "float32", "seed": 7},
    }
    model = get_model(params).cpu().eval()

    class Boundary(Exception):
        pass

    def forbidden(*args):
        raise AssertionError("official export selected Python-runtime cohort launchers")

    monkeypatch.setattr(ecs, "cohort_backward", forbidden)

    def inspect(traced, *args, **kwargs):
        targets = [str(node.target) for node in traced.graph.nodes]
        assert any("sezm_triton.so2_mixing_stack_bwd." in name for name in targets)
        assert not any("sezm_triton.so2_cohort_" in name for name in targets)
        raise Boundary

    monkeypatch.setattr(torch.export, "export", inspect)
    with pytest.raises(Boundary):
        if backend == "pt":
            checkpoint = tmp_path / "model.pt"
            wrapper = ModelWrapper(model, model_params=params)
            torch.save({"model": wrapper.state_dict()}, checkpoint)
            freeze_pt2.freeze_sezm_to_pt2(
                str(checkpoint),
                str(tmp_path / "model.pt2"),
                device=torch.device("cuda:0"),
                with_comm=False,
            )
        else:
            _trace_and_export_impl(
                {"model": model.serialize(), "model_def_script": params},
                lower_kind="graph",
                do_atomic_virial=True,
                target_device=torch.device("cuda:0"),
            )
