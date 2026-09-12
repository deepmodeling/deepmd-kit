# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native-spin host registration does not require a model artifact."""

import os

from lammps import (
    lammps,
)


def test_native_spin_host_style_is_registered() -> None:
    """The plugin must expose the host style, not only Kokkos aliases."""
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    try:
        if plugin := os.environ.get("DEEPMD_TEST_PLUGIN"):
            lmp.command(f"plugin load {plugin}")
        assert lmp.has_style("pair", "dpa4spin")
    finally:
        lmp.close()
