# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest

from backend.read_env import (
    get_argument_from_env,
)


def _disable_ml_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DP_ENABLE_TENSORFLOW", "0")
    monkeypatch.setenv("DP_ENABLE_PYTORCH", "0")


def test_lammps_kokkos_build_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_ml_backends(monkeypatch)
    monkeypatch.setenv("DP_LAMMPS_VERSION", "stable_22Jul2025_update4")
    monkeypatch.setenv("DP_ENABLE_LAMMPS_KOKKOS", "1")
    get_argument_from_env.cache_clear()

    _, cmake_args, _, _, _, _ = get_argument_from_env()

    assert "-DDEEPMD_LAMMPS_KOKKOS:BOOL=TRUE" in cmake_args


def test_lammps_kokkos_requires_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_ml_backends(monkeypatch)
    monkeypatch.delenv("DP_LAMMPS_VERSION", raising=False)
    monkeypatch.setenv("DP_ENABLE_LAMMPS_KOKKOS", "1")
    get_argument_from_env.cache_clear()

    with pytest.raises(RuntimeError, match="requires DP_LAMMPS_VERSION"):
        get_argument_from_env()
