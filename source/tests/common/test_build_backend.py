# SPDX-License-Identifier: LGPL-3.0-or-later

from pathlib import (
    Path,
)

import pytest


@pytest.fixture
def read_env(monkeypatch: pytest.MonkeyPatch):
    """Load the checkout-only build backend even when testing an installed wheel."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    from backend.read_env import (
        get_argument_from_env,
    )

    get_argument_from_env.cache_clear()
    yield get_argument_from_env
    get_argument_from_env.cache_clear()


def _disable_ml_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DP_ENABLE_TENSORFLOW", "0")
    monkeypatch.setenv("DP_ENABLE_PYTORCH", "0")


def test_lammps_kokkos_build_flag(monkeypatch: pytest.MonkeyPatch, read_env) -> None:
    _disable_ml_backends(monkeypatch)
    monkeypatch.setenv("DP_LAMMPS_VERSION", "stable_22Jul2025_update4")
    monkeypatch.setenv("DP_ENABLE_LAMMPS_KOKKOS", "1")

    _, cmake_args, _, _, _, _ = read_env()

    assert "-DDEEPMD_LAMMPS_KOKKOS:BOOL=TRUE" in cmake_args


def test_lammps_kokkos_requires_plugin(
    monkeypatch: pytest.MonkeyPatch, read_env
) -> None:
    _disable_ml_backends(monkeypatch)
    monkeypatch.delenv("DP_LAMMPS_VERSION", raising=False)
    monkeypatch.setenv("DP_ENABLE_LAMMPS_KOKKOS", "1")

    with pytest.raises(RuntimeError, match="requires DP_LAMMPS_VERSION"):
        read_env()


@pytest.mark.parametrize("disabled", [None, "0"])
def test_lammps_kokkos_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, read_env, disabled
) -> None:
    """An enabled build followed by a disabled build must reset CMake's cache."""
    _disable_ml_backends(monkeypatch)
    monkeypatch.setenv("DP_LAMMPS_VERSION", "stable_22Jul2025_update4")
    monkeypatch.setenv("DP_ENABLE_LAMMPS_KOKKOS", "1")
    assert "-DDEEPMD_LAMMPS_KOKKOS:BOOL=TRUE" in read_env()[1]
    read_env.cache_clear()
    if disabled is None:
        monkeypatch.delenv("DP_ENABLE_LAMMPS_KOKKOS")
    else:
        monkeypatch.setenv("DP_ENABLE_LAMMPS_KOKKOS", disabled)
    cmake_args = read_env()[1]
    assert "-DDEEPMD_LAMMPS_KOKKOS:BOOL=FALSE" in cmake_args
    assert "-DDEEPMD_LAMMPS_KOKKOS:BOOL=TRUE" not in cmake_args
