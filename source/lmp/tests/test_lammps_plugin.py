# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check style registration without requiring a model or inference backend.

CI loads the runtime plugin through LAMMPS_PLUGIN_PATH. A standalone plugin
can instead be loaded explicitly with DEEPMD_TEST_PLUGIN.
"""

import os

from lammps import (
    lammps,
)


def test_dpa4spin_host_style_registered() -> None:
    """The unsuffixed host style must also be available without Kokkos."""
    with lammps(cmdargs=["-log", "none", "-screen", "none"]) as lmp:
        if plugin := os.environ.get("DEEPMD_TEST_PLUGIN"):
            lmp.command(f'plugin load "{plugin}"')
        assert lmp.has_style("pair", "dpa4spin"), (
            "LAMMPS plugin did not register pair style dpa4spin"
        )
