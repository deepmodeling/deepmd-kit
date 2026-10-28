#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate a minimal native-spin DPA4C graph ``.pt2`` for LAMMPS ``dpa4spin``.

Unlike ``gen_dpa4c_spin.py``, this archive carries no fparam, aparam, or
charge/spin condition so ``pair_style dpa4spin`` can load it without extra
keywords. Used by ``source/lmp/tests/test_dpa4spin_magnetic_force_convention.py``.
"""

from __future__ import (
    annotations,
)

import copy
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpa4_fixtures import (
    jitter_zero_arrays,
)
from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
)

CFG = {
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": 4.0,
        "channels": 8,
        "lmax": 2,
        "n_radial": 4,
        "precision": "float64",
        "seed": 7,
    },
    "fitting_net": {"neuron": [8, 8], "precision": "float64", "seed": 7},
    "spin": {"use_spin": [True, False], "scheme": "native"},
}

_JITTER_SEED = 20260817
OUT = os.path.join(os.path.dirname(__file__), "deeppot_dpa4c_spin_lmp.pt2")


def main() -> None:
    from torch._inductor import config as inductor_config

    from deepmd.dpmodel.model.model import (
        get_model,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()
    model = get_model(copy.deepcopy(CFG))
    md = jitter_zero_arrays(model.serialize(), np.random.default_rng(_JITTER_SEED))
    data = {
        "model": md,
        "model_def_script": CFG,
        "backend": "dpmodel",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    print(f"Exporting {OUT}")  # noqa: T201
    # PyTorch 2.11's single-threaded CPU SIMD gather codegen can use an
    # index temporary before declaring it. This tiny fixture checks adapter
    # magnetic-force conventions, so scalar codegen avoids that compiler bug
    # without changing production exports or CUDA compilation.
    with inductor_config.patch({"cpp.simdlen": 1}):
        deserialize_to_file(OUT, data, do_atomic_virial=True, lower_kind="graph")
    with zipfile.ZipFile(OUT) as zf:
        meta = json.loads(zf.read("model/extra/metadata.json").decode())
    print(  # noqa: T201
        json.dumps(
            {
                k: meta.get(k)
                for k in (
                    "spin_scheme",
                    "has_comm_artifact",
                    "dim_fparam",
                    "dim_aparam",
                    "dim_chg_spin",
                    "lower_input_kind",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
