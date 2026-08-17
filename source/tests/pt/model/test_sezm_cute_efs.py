# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end strict-FP32 acceptance tests for the Neo CuTe path."""

from __future__ import (
    annotations,
)

import contextlib
import importlib
import os
import subprocess
import sys
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )


TOL = 5.0e-5
OUTPUT_KEYS = ("energy", "force", "atom_energy", "virial")
SM80_CAPABILITIES = frozenset({(8, 0), (8, 6)})
SUPPORTED_CAPABILITIES = SM80_CAPABILITIES | frozenset(
    {(8, 9), (9, 0), (10, 0), (12, 0)}
)
REPO_ROOT = Path(__file__).resolve().parents[4]
_CHILD_MODE = "--neo-cute-efs-child"


def _cute_runtime_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "Neo CuTe E/F/S parity requires CUDA"
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in SUPPORTED_CAPABILITIES:
        return f"Neo CuTe E/F/S parity does not support {capability}"
    try:
        importlib.import_module("cutlass.cute")
        importlib.import_module("cuda.bindings.driver")
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"Neo CuTe E/F/S acceptance requires the CuTe DSL runtime: {exc}"
    return None


_CUTE_SKIP_REASON = _cute_runtime_skip_reason()


@contextlib.contextmanager
def _strict_fp32() -> Iterator[None]:
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    prior_matmul_tf32 = matmul.allow_tf32
    prior_cudnn_tf32 = cudnn.allow_tf32
    prior_precision = torch.get_float32_matmul_precision()
    try:
        matmul.allow_tf32 = False
        cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        matmul.allow_tf32 = prior_matmul_tf32
        cudnn.allow_tf32 = prior_cudnn_tf32
        torch.set_float32_matmul_precision(prior_precision)


def _neo_model(*, use_compile: bool) -> torch.nn.Module:
    from deepmd.pt.model.model import (
        get_sezm_model,
    )

    model = get_sezm_model(
        {
            "type": "SeZM",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "SeZM",
                "sel": 32,
                "rcut": 3.0,
                "channels": 32,
                "n_radial": 16,
                "use_env_seed": True,
                "lmax": 3,
                "mmax": 1,
                "n_blocks": 2,
                "so2_layers": 3,
                "radial_so2_mode": "degree_channel",
                "radial_so2_rank": 1,
                "n_focus": 2,
                "focus_dim": 0,
                "n_atten_head": 1,
                "message_node_so3": True,
                "ffn_neurons": 0,
                "ffn_so3_grid": True,
                "grid_mlp": False,
                "grid_branch": [0, 0, 1],
                "ffn_blocks": 1,
                "so3_readout": "mlp",
                "use_amp": False,
                "precision": "float32",
                "seed": 42,
            },
            "fitting_net": {
                "neuron": [0],
                "precision": "float32",
                "seed": 42,
            },
            "use_compile": use_compile,
            "enable_tf32": False,
        }
    ).to(device="cuda", dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _water_frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coord = torch.tensor(
        [
            [
                [0.70, 0.80, 0.90],
                [1.58, 0.80, 0.90],
                [0.43, 1.63, 0.90],
                [3.20, 3.00, 2.80],
                [4.08, 3.00, 2.80],
                [2.93, 3.83, 2.80],
            ]
        ],
        device="cuda",
        dtype=torch.float32,
    )
    atype = torch.tensor(
        [[0, 1, 1, 0, 1, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    box = torch.tensor(
        [[5.4, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 5.0]],
        device="cuda",
        dtype=torch.float32,
    )
    return coord, atype, box


def _triclinic_frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_matrix = torch.tensor(
        [
            [5.2, 0.0, 0.0],
            [0.7, 4.8, 0.0],
            [0.3, 0.5, 5.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    fractional = torch.tensor(
        [
            [0.08, 0.11, 0.14],
            [0.25, 0.14, 0.18],
            [0.12, 0.31, 0.22],
            [0.54, 0.57, 0.49],
            [0.72, 0.59, 0.51],
            [0.57, 0.76, 0.55],
            [0.88, 0.16, 0.81],
            [0.06, 0.22, 0.84],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    coord = (fractional @ box_matrix).unsqueeze(0)
    atype = torch.tensor(
        [[0, 1, 1, 0, 1, 1, 0, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    return coord, atype, box_matrix.reshape(1, 9)


def _detached_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: outputs[name].detach().cpu().clone() for name in OUTPUT_KEYS}


_FRAME_FACTORIES = {
    "water": _water_frame,
    "triclinic": _triclinic_frame,
}


def _assert_runtime_profile() -> dict[str, bool]:
    from deepmd.kernels.cute.neo import (
        k1,
        runtime_policy,
    )

    capability = tuple(torch.cuda.get_device_capability())
    if capability not in SUPPORTED_CAPABILITIES:
        raise AssertionError(f"unsupported K1 capability {capability}")
    policy_checks = {
        "master": runtime_policy.is_cute_infer_enabled(),
        "triton_infer_2": os.environ.get("DP_TRITON_INFER") == "2",
        "supported": runtime_policy.is_supported_k1_capability(capability),
        "packed_wigner": runtime_policy.is_packed_wigner_enabled(capability),
    }
    config = k1._architecture_default_config(capability)
    if capability in SM80_CAPABILITIES:
        architecture_checks = {
            "sm80_profile": runtime_policy.is_sm80_profile_enabled(capability),
            "gie": runtime_policy.is_gie_enabled(capability),
            "thin_wrapper": runtime_policy.is_k1_thin_wrapper_enabled(capability),
            "output_grid_bwd_panel": (
                runtime_policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(capability)
            ),
            "output_grid_fwd": (
                runtime_policy.is_output_grid_fwd_sm80_c96_n48_enabled(capability)
            ),
            "readout_fold": runtime_policy.is_readout_input_fold_sm80_enabled(
                capability
            ),
            "per_focus_so2_forward": config.per_focus_so2_fwd_pair,
            "no_native_sm90_path": not config.native_sm90_path,
        }
    elif capability == (9, 0):
        architecture_checks = {
            "native_sm90_path": config.native_sm90_path,
            "shared_so2_forward": not config.per_focus_so2_fwd_pair,
            "sm90_output_grid": (
                runtime_policy.is_output_grid_sm90_c96_asymmetric_panels_enabled(
                    capability
                )
            ),
            "sm90_readout_fold": runtime_policy.is_readout_input_fold_sm90_enabled(
                capability
            ),
        }
    elif capability in runtime_policy.FUSED_SO2_GATE_CAPABILITIES:
        architecture_checks = {
            "combined_so2_gate": config.combined_so2_gate,
            "shared_so2_forward": not config.per_focus_so2_fwd_pair,
            "no_native_sm90_path": not config.native_sm90_path,
        }
    else:
        architecture_checks = {
            "no_native_sm90_path": not config.native_sm90_path,
            "shared_so2_forward": not config.per_focus_so2_fwd_pair,
        }
    selected = {**policy_checks, **architecture_checks}
    missing = sorted(name for name, enabled in selected.items() if not enabled)
    if missing:
        raise AssertionError(
            "DP_NEO_CUTE_INFER did not select the expected runtime profile: "
            + ", ".join(missing)
        )
    return selected


def _run_efs_child(
    *,
    frame_name: str,
    use_compile: bool,
    output_path: Path,
) -> None:
    from deepmd.kernels.cute.neo import (
        k1,
    )

    torch.manual_seed(20260726)
    torch.cuda.manual_seed_all(20260726)
    runner_calls = 0
    original_build_runner = k1._build_runner

    def counted_build_runner(*args: Any, **kwargs: Any) -> Any:
        nonlocal runner_calls
        runner_calls += 1
        return original_build_runner(*args, **kwargs)

    if use_compile:
        k1._build_runner = counted_build_runner
    try:
        model = _neo_model(use_compile=use_compile)
        state_keys_before = tuple(model.state_dict())
        coord, atype, box = _FRAME_FACTORIES[frame_name]()
        with _strict_fp32():
            profile = _assert_runtime_profile() if use_compile else {}
            run_count = 2 if use_compile else 1
            runs = [
                _detached_outputs(model(coord, atype, box=box))
                for _ in range(run_count)
            ]
            torch.cuda.synchronize()
        state_keys_after = tuple(model.state_dict())
    finally:
        if use_compile:
            k1._build_runner = original_build_runner

    if state_keys_after != state_keys_before:
        raise AssertionError("CuTe warmup changed the model state_dict contract")
    if use_compile and runner_calls == 0:
        raise AssertionError("compiled CuTe run did not instantiate the K1 runner")
    torch.save(
        {
            "outputs": runs,
            "profile": profile,
            "runner_calls": runner_calls,
            "use_compile": use_compile,
        },
        output_path,
    )


def _clean_child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(("DP_CUTE_", "DP_NEO_CUTE_")):
            environment.pop(name)
    for name in (
        "DP_COMPILE_INFER",
        "DP_INTERFACE_PREC",
        "DP_TF32_INFER",
        "DP_TRITON_INFER",
        "NVIDIA_TF32_OVERRIDE",
    ):
        environment.pop(name, None)
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    )
    environment.update(
        {
            "DP_INTERFACE_PREC": "low",
            "DP_TF32_INFER": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _child_environment(*, use_compile: bool) -> dict[str, str]:
    environment = _clean_child_environment()
    if use_compile:
        environment.update(
            {
                "DP_COMPILE_INFER": "1",
                "DP_CUTE_STRICT": "1",
                "DP_NEO_CUTE_INFER": "1",
                "DP_TRITON_INFER": "2",
            }
        )
    else:
        environment.update(
            {
                "DP_COMPILE_INFER": "0",
                "DP_CUTE_INFER": "0",
                "DP_NEO_CUTE_INFER": "0",
                "DP_TRITON_INFER": "0",
            }
        )
    return environment


def _run_child_process(
    *,
    frame_name: str,
    use_compile: bool,
    output_path: Path,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            _CHILD_MODE,
            frame_name,
            "compiled" if use_compile else "eager",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=_child_environment(use_compile=use_compile),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode:
        pytest.fail(
            "Neo CuTe E/F/S child failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return torch.load(output_path, map_location="cpu", weights_only=True)


@pytest.mark.skipif(
    _CUTE_SKIP_REASON is not None,
    reason=_CUTE_SKIP_REASON or "CuTe runtime unavailable",
)
@pytest.mark.parametrize("frame_name", tuple(_FRAME_FACTORIES))
def test_neo_cute_compiled_profile_matches_eager_at_shared_5e5(
    tmp_path: Path,
    frame_name: str,
) -> None:
    eager = _run_child_process(
        frame_name=frame_name,
        use_compile=False,
        output_path=tmp_path / f"{frame_name}-eager.pt",
    )
    compiled = _run_child_process(
        frame_name=frame_name,
        use_compile=True,
        output_path=tmp_path / f"{frame_name}-compiled.pt",
    )

    assert not eager["use_compile"]
    assert compiled["use_compile"]
    assert compiled["runner_calls"] > 0
    assert compiled["profile"]
    assert all(compiled["profile"].values())
    expected = eager["outputs"][0]
    for run_index, actual in enumerate(compiled["outputs"]):
        for name in OUTPUT_KEYS:
            torch.testing.assert_close(
                actual[name],
                expected[name],
                atol=TOL,
                rtol=TOL,
                msg=(
                    f"compiled Neo CuTe {frame_name} run {run_index} {name} "
                    "differs from eager PyTorch"
                ),
            )


def _main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] != _CHILD_MODE:
        raise SystemExit("this test module is only executable in E/F/S child mode")
    _run_efs_child(
        frame_name=sys.argv[2],
        use_compile=sys.argv[3] == "compiled",
        output_path=Path(sys.argv[4]),
    )


if __name__ == "__main__":
    _main()
