# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native-spin host registration does not require a model artifact."""

import os

import pytest
from lammps import (
    lammps,
)


def test_native_spin_host_style_is_registered() -> None:
    """The runtime plugin, rather than the host binary, must expose dpa4spin."""
    plugin = os.environ.get("DEEPMD_TEST_PLUGIN")
    if not plugin:
        pytest.skip("DEEPMD_TEST_PLUGIN is required to validate plugin registration")

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    try:
        assert not lmp.has_style("pair", "dpa4spin"), (
            "host LAMMPS already provides dpa4spin; plugin provenance is ambiguous"
        )
        lmp.command(f"plugin load {plugin}")
        assert lmp.has_style("pair", "dpa4spin")
    finally:
        lmp.close()
