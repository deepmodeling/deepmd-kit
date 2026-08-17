#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify a DeePMD-kit Python install for one backend and accelerator."""

from __future__ import (
    annotations,
)

import argparse
import json
import re
import sys
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)


@dataclass(frozen=True)
class CheckResult:
    """Represent one verification result."""

    name: str
    passed: bool
    detail: str


def _result(name: str, passed: bool, detail: object) -> CheckResult:
    """Create a normalized check result."""
    return CheckResult(name=name, passed=passed, detail=str(detail))


def _interpreter_prefix() -> Path:
    """Return the resolved prefix of the running interpreter."""
    return Path(sys.prefix).resolve(strict=False)


def _check_deepmd(
    expected_version: str | None, expected_prefix: str | None
) -> CheckResult:
    """Import DeePMD-kit and verify its requested identity."""
    try:
        import deepmd
    except (ImportError, OSError, RuntimeError) as exc:
        return _result("deepmd", False, f"{type(exc).__name__}: {exc}")
    version = getattr(deepmd, "__version__", "unknown")
    location_value = getattr(deepmd, "__file__", None)
    if not isinstance(location_value, str) or not location_value:
        return _result("deepmd", False, f"version={version} file=missing")
    try:
        location = Path(location_value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return _result(
            "deepmd",
            False,
            f"version={version} file={location_value} {type(exc).__name__}: {exc}",
        )
    actual_prefix = _interpreter_prefix()
    passed = True
    identity = [f"version={version}", f"file={location}", f"prefix={actual_prefix}"]
    if expected_version is not None:
        passed = passed and version == expected_version
        identity.append(f"expected_version={expected_version}")
    if expected_prefix is not None:
        expected = Path(expected_prefix).resolve(strict=False)
        imported_from_prefix = location == expected or expected in location.parents
        passed = passed and actual_prefix == expected and imported_from_prefix
        identity.append(f"expected_prefix={expected}")
    return _result("deepmd", passed, " ".join(identity))


def _tensorflow_accelerator(tf: Any) -> str | None:
    """Return TensorFlow's compiled GPU runtime from build metadata."""
    try:
        build_info = tf.sysconfig.get_build_info()
    except (AttributeError, RuntimeError, TypeError):
        return None
    if not isinstance(build_info, dict):
        return None
    if build_info.get("is_rocm_build") is True:
        return "rocm"
    if build_info.get("is_cuda_build") is True:
        return "cuda"
    return None


def _jax_accelerator(devices: list[Any]) -> str | None:
    """Return JAX's unambiguous GPU runtime from backend metadata."""
    runtimes: set[str] = set()
    for device in devices:
        client = getattr(device, "client", None)
        canonical_platforms = (
            str(getattr(client, "platform", "")).lower(),
            str(getattr(device, "platform", "")).lower(),
        )
        runtime = next(
            (item for item in canonical_platforms if item in {"cuda", "rocm"}),
            None,
        )
        if runtime is None:
            platform_version = str(getattr(client, "platform_version", "")).lower()
            is_cuda = "cuda" in platform_version
            is_rocm = "rocm" in platform_version or "hip" in platform_version
            if is_cuda != is_rocm:
                runtime = "cuda" if is_cuda else "rocm"
        if runtime is not None:
            runtimes.add(runtime)
    return next(iter(runtimes)) if len(runtimes) == 1 else None


def _check_build_variant(expected: str) -> CheckResult:
    """Check DeePMD-kit's recorded compiled build variant."""
    try:
        from deepmd.env import (
            GLOBAL_CONFIG,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        return _result("build_variant", False, f"{type(exc).__name__}: {exc}")
    actual = str(GLOBAL_CONFIG.get("dp_variant", "unknown")).lower()
    return _result(
        "build_variant",
        actual == expected,
        f"expected={expected} actual={actual}",
    )


def _commits_match(actual: str, expected: str) -> bool:
    """Return whether full or abbreviated hexadecimal commits identify one SHA."""
    actual = actual.strip().lower()
    expected = expected.strip().lower()
    if re.fullmatch(r"[0-9a-f]{7,40}", actual) is None:
        return False
    if re.fullmatch(r"[0-9a-f]{7,40}", expected) is None:
        return False
    return actual.startswith(expected) or expected.startswith(actual)


def _check_source_commit(expected: str) -> CheckResult:
    """Check the source commit recorded in DeePMD-kit's build metadata."""
    try:
        from deepmd.env import (
            GLOBAL_CONFIG,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        return _result("source_commit", False, f"{type(exc).__name__}: {exc}")
    actual = str(GLOBAL_CONFIG.get("git_hash", ""))
    return _result(
        "source_commit",
        _commits_match(actual, expected),
        f"expected={expected} actual={actual or 'unknown'}",
    )


def _check_pytorch(accelerator: str) -> list[CheckResult]:
    """Verify the PyTorch backend and a minimal tensor operation."""
    try:
        import torch

        import deepmd.pt  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:
        return [_result("pytorch", False, f"{type(exc).__name__}: {exc}")]
    details = (
        f"version={torch.__version__} cuda={torch.version.cuda} "
        f"hip={torch.version.hip} available={torch.cuda.is_available()}"
    )
    results = [_result("pytorch", True, details)]
    if accelerator == "cpu":
        tensor = torch.ones(1, device="cpu") + 1
        results.append(_result("pytorch_tensor", bool(tensor.item() == 2), "cpu"))
        return results
    if accelerator == "cuda" and torch.version.cuda is None:
        results.append(_result("pytorch_accelerator", False, "CUDA wheel required"))
        return results
    if accelerator == "rocm" and torch.version.hip is None:
        results.append(_result("pytorch_accelerator", False, "ROCm wheel required"))
        return results
    if not torch.cuda.is_available():
        results.append(_result("pytorch_accelerator", False, "no visible GPU"))
        return results
    try:
        tensor = torch.ones(1, device="cuda:0") + 1
        torch.cuda.synchronize(0)
    except RuntimeError as exc:
        results.append(_result("pytorch_tensor", False, exc))
        return results
    detail = f"device={torch.cuda.get_device_name(0)} value={tensor.item()}"
    results.append(_result("pytorch_tensor", bool(tensor.item() == 2), detail))
    return results


def _check_tensorflow(accelerator: str) -> list[CheckResult]:
    """Verify the TensorFlow backend and a minimal tensor operation."""
    try:
        import tensorflow as tf

        import deepmd.tf  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:
        return [_result("tensorflow", False, f"{type(exc).__name__}: {exc}")]
    results = [_result("tensorflow", True, f"version={tf.__version__}")]
    devices = tf.config.list_physical_devices("GPU")
    if accelerator == "cpu":
        device = "/CPU:0"
    elif not devices:
        results.append(_result("tensorflow_accelerator", False, "no visible GPU"))
        return results
    else:
        runtime = _tensorflow_accelerator(tf)
        if runtime != accelerator:
            results.append(
                _result(
                    "tensorflow_accelerator",
                    False,
                    f"expected={accelerator} actual={runtime or 'unknown'}",
                )
            )
            return results
        device = "/GPU:0"
    try:
        with tf.device(device):
            value = float(tf.reduce_sum(tf.ones((2,), dtype=tf.float32)).numpy())
    except (RuntimeError, ValueError) as exc:
        results.append(_result("tensorflow_tensor", False, exc))
        return results
    results.append(_result("tensorflow_tensor", value == 2.0, f"device={device}"))
    return results


def _check_jax(accelerator: str) -> list[CheckResult]:
    """Verify the JAX backend and a minimal device operation."""
    try:
        import jax
        import jax.numpy as jnp

        import deepmd.jax  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:
        return [_result("jax", False, f"{type(exc).__name__}: {exc}")]
    devices = list(jax.devices())
    results = [
        _result(
            "jax",
            True,
            f"version={jax.__version__} devices={[str(item) for item in devices]}",
        )
    ]
    if accelerator == "cpu":
        candidates = [item for item in devices if item.platform == "cpu"]
    else:
        candidates = [
            item for item in devices if item.platform in {"gpu", "cuda", "rocm"}
        ]
    if not candidates:
        results.append(_result("jax_accelerator", False, f"no {accelerator} device"))
        return results
    if accelerator != "cpu":
        runtime = _jax_accelerator(candidates)
        if runtime != accelerator:
            results.append(
                _result(
                    "jax_accelerator",
                    False,
                    f"expected={accelerator} actual={runtime or 'unknown'}",
                )
            )
            return results
    try:
        value = jax.device_put(jnp.ones((2,)), candidates[0]).sum()
        value.block_until_ready()
        scalar = float(value)
    except (RuntimeError, ValueError) as exc:
        results.append(_result("jax_tensor", False, exc))
        return results
    results.append(_result("jax_tensor", scalar == 2.0, f"device={candidates[0]}"))
    return results


def _check_paddle(accelerator: str) -> list[CheckResult]:
    """Verify the Paddle backend and a minimal tensor operation."""
    try:
        import paddle

        import deepmd.pd  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:
        return [_result("paddle", False, f"{type(exc).__name__}: {exc}")]
    results = [_result("paddle", True, f"version={paddle.__version__}")]
    if accelerator == "cpu":
        device = "cpu"
    elif accelerator == "cuda":
        if (
            not paddle.device.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() < 1
        ):
            results.append(_result("paddle_accelerator", False, "no visible CUDA GPU"))
            return results
        device = "gpu:0"
    else:
        is_rocm = getattr(paddle.device, "is_compiled_with_rocm", lambda: False)
        if not is_rocm():
            results.append(_result("paddle_accelerator", False, "ROCm build required"))
            return results
        device = "gpu:0"
    try:
        paddle.set_device(device)
        value = float(paddle.sum(paddle.ones([2], dtype="float32")).item())
    except (RuntimeError, ValueError) as exc:
        results.append(_result("paddle_tensor", False, exc))
        return results
    results.append(_result("paddle_tensor", value == 2.0, f"device={device}"))
    return results


def _check_custom_op() -> CheckResult:
    """Require the PyTorch customized operation library."""
    try:
        from deepmd.pt.cxx_op import (
            ENABLE_CUSTOMIZED_OP,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        return _result("pytorch_custom_op", False, f"{type(exc).__name__}: {exc}")
    return _result(
        "pytorch_custom_op", bool(ENABLE_CUSTOMIZED_OP), ENABLE_CUSTOMIZED_OP
    )


def _check_nv() -> CheckResult:
    """Require the nvalchemi neighbor-list integration."""
    try:
        from deepmd.pt.utils.nv_nlist import (
            is_nv_available,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        return _result("nvalchemi", False, f"{type(exc).__name__}: {exc}")
    available = bool(is_nv_available())
    return _result("nvalchemi", available, available)


def _check_vesin() -> CheckResult:
    """Require the vesin.torch neighbor-list integration."""
    try:
        from deepmd.pt_expt.utils.vesin_neighbor_list import (
            is_vesin_torch_available,
        )
    except (ImportError, OSError, RuntimeError) as exc:
        return _result("vesin", False, f"{type(exc).__name__}: {exc}")
    available = bool(is_vesin_torch_available())
    return _result("vesin", available, available)


def run_checks(
    *,
    backend: str,
    accelerator: str,
    expected_version: str | None,
    expected_prefix: str | None,
    expected_source_commit: str | None,
    expected_build_variant: str | None,
    expect_custom_op: bool,
    expect_nv: bool,
    expect_vesin: bool,
) -> list[CheckResult]:
    """Run all checks requested by the installation plan.

    Parameters
    ----------
    backend : str
        Backend name.
    accelerator : str
        Runtime accelerator.
    expected_version : str, optional
        Required DeePMD-kit version.
    expected_prefix : str, optional
        Required Python environment prefix.
    expected_source_commit : str, optional
        Required source commit recorded at build time.
    expected_build_variant : str, optional
        Required DeePMD-kit compiled variant.
    expect_custom_op : bool
        Require the PyTorch custom operation library.
    expect_nv : bool
        Require nvalchemi.
    expect_vesin : bool
        Require vesin.torch.

    Returns
    -------
    list of CheckResult
        Ordered verification results.
    """
    checks = [_check_deepmd(expected_version, expected_prefix)]
    if expected_source_commit is not None:
        checks.append(_check_source_commit(expected_source_commit))
    if expected_build_variant is not None:
        checks.append(_check_build_variant(expected_build_variant))
    backend_checks = {
        "pytorch": _check_pytorch,
        "tensorflow": _check_tensorflow,
        "jax": _check_jax,
        "paddle": _check_paddle,
    }
    checks.extend(backend_checks[backend](accelerator))
    if expect_custom_op:
        checks.append(_check_custom_op())
    if expect_nv:
        checks.append(_check_nv())
    if expect_vesin:
        checks.append(_check_vesin())
    return checks


def _print_results(checks: list[CheckResult], *, as_json: bool) -> None:
    """Print verification results."""
    if as_json:
        payload: dict[str, Any] = {
            "passed": all(item.passed for item in checks),
            "checks": [asdict(item) for item in checks],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for item in checks:
        status = "PASS" if item.passed else "FAIL"
        print(f"{status:<5} {item.name}: {item.detail}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and verify the requested Python runtime.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to the process arguments.

    Returns
    -------
    int
        Zero when every requested check passes, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("pytorch", "tensorflow", "jax", "paddle"),
    )
    parser.add_argument("--accelerator", required=True, choices=("cpu", "cuda", "rocm"))
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-prefix")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-build-variant", choices=("cpu", "cuda", "rocm"))
    parser.add_argument("--expect-custom-op", action="store_true")
    parser.add_argument("--expect-nv", action="store_true")
    parser.add_argument("--expect-vesin", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.backend != "pytorch" and any(
        (args.expect_custom_op, args.expect_nv, args.expect_vesin)
    ):
        parser.error("PyTorch-only checks require --backend pytorch")
    if (
        args.expected_prefix is not None
        and not Path(args.expected_prefix).is_absolute()
    ):
        parser.error("--expected-prefix must be absolute")
    if (
        args.expected_source_commit is not None
        and re.fullmatch(r"[0-9a-fA-F]{7,40}", args.expected_source_commit) is None
    ):
        parser.error("--expected-source-commit must be a 7-40 character commit SHA")
    checks = run_checks(
        backend=args.backend,
        accelerator=args.accelerator,
        expected_version=args.expected_version,
        expected_prefix=args.expected_prefix,
        expected_source_commit=args.expected_source_commit,
        expected_build_variant=args.expected_build_variant,
        expect_custom_op=args.expect_custom_op,
        expect_nv=args.expect_nv,
        expect_vesin=args.expect_vesin,
    )
    _print_results(checks, as_json=args.json)
    return 0 if all(item.passed for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
